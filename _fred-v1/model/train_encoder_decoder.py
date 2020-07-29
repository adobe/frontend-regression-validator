import argparse
import json
import os
import pickle
import numpy as np
import sys
from datetime import datetime
from dataloader.dataloader import FREDDataset, COCODatasetEncoderPretrain
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from models.nnet import NNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VALID_MODELS = ['frednet', 'unet', 'pspnet', 'frednetv2', 'frednetv3', 'frednetv4', 'unetv2']


def train(model_name, n_epochs, use_residuals, learning_rate, optimizer_name, validation_split, random_seed, momentum,
          batch_size, betas, eps, prefix=None, pretrained_encoder=None, save_train=False, save_test=False,
          out_layer=False,
          save_pretrained_encoder=False, dataset='coco'):
    assert prefix is not None, 'Please specify a prefix for the saved model files'
    ds = None
    if dataset == 'fred':
        ds = FREDDataset(images_dir='/mnt/frednet/images', masks_dir='/mnt/frednet/concatenated')
    elif dataset == 'coco':
        ds = COCODatasetEncoderPretrain(images_dir='./coco')
    assert ds is not None, 'No valid dataset was chosen'

    if pretrained_encoder is not None:
        assert os.path.exists(pretrained_encoder), 'No such file {}'.format(pretrained_encoder)

    dataset_size = len(ds)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(ds, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    test_loader = DataLoader(ds, batch_size=batch_size, num_workers=4, sampler=test_sampler)

    model_parameters = {
        'model_name': model_name,
        'use_residuals': use_residuals,
        'pretrained_encoder': pretrained_encoder,
        'learning_rate': learning_rate,
        'optimizer_name': optimizer_name,
        'momentum': momentum,
        'batch_size': batch_size,
        'betas': betas,
        'eps': eps,
    }

    if not os.path.exists(os.path.join(os.getcwd(), 'runs')):
        os.mkdir('runs')
    now = datetime.now()
    current_time = now.strftime('%d-%m-%Y_%H:%M:%S')
    with open(os.path.join('./runs', model_name + '_' + current_time + '.json'), 'w') as f:
        json.dump(model_parameters, indent=2, fp=f)

    model = NNet(out_channels=5, use_residuals=use_residuals, model_name=model_name, out_layer=out_layer)

    assert model is not None, 'Failed to load model'
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(torch.load(pretrained_encoder, map_location=device))
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('[INFO] Using pretrained encoder: {}'.format(pretrained_encoder))

    print(model)
    model = model.to(device)

    optimizer = None

    if 'adam' in optimizer_name:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=eps, betas=betas)
    if 'sgd' in optimizer_name:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    criterion = nn.BCELoss()
    n_steps = len(train_loader)
    print("[INFO] Starting training model {} \n\toptimizer {}\n\tlearning rate {}\n\tbatch size {}".format(
        model_name, optimizer_name, learning_rate, batch_size))

    best_val_network_loss = 99999.0
    best_val_outlayer_loss = 9999.0
    epochs_not_improved = 0

    training_network = True


    model.train()
    if out_layer:
        model.out_layer.eval()
    for epoch in range(n_epochs):
        epoch_network_loss = 0.0
        epoch_loss = 0.0
        epoch_out_layer_loss = 0.0
        train_outputs = None
        test_outputs = None

        # Start training epoch
        for i, batch_data in enumerate(train_loader):
            images = Variable(batch_data['img']).to(device)
            labels = Variable(batch_data['mask']).to(device)
            if out_layer:
                # If we use outlayer, we compute the losses individually on the network and layer
                outputs_layer, outputs_network = model(images)
                loss_network = criterion(outputs_network, labels)
                loss_layer = criterion(outputs_layer, labels)

                epoch_out_layer_loss += loss_layer.item()
                epoch_network_loss += loss_network.item()

                optimizer.zero_grad()
                loss_network.backward()
                loss_layer.backward()
                optimizer.step()

                print(
                    'Epoch: {}/{}, Step: {}/{}, Loss network: {}, Loss layer {}'.format(epoch + 1,
                                                                                        n_epochs, i + 1,
                                                                                        n_steps,
                                                                                        loss_network.item(),
                                                                                        loss_layer.item()))
                sys.stdout.flush()
            else:
                # Otherwise we just compute the forward pass normally
                outputs = model(images)
                train_outputs = outputs
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                print('Epoch: {}/{}, Step: {}/{}, Loss: {}'.format(epoch + 1, n_epochs, i + 1, n_steps, loss.item()))
                sys.stdout.flush()

            del images
            del labels

        val_loss = 0.0
        val_network_loss = 0.0
        val_layer_loss = 0.0
        model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(test_loader):
                images = Variable(batch_data['img']).to(device)
                labels = Variable(batch_data['mask']).to(device)
                if out_layer:
                    # We compute the loss individually in validation
                    outputs_layer, outputs_network = model(images)
                    loss_train = criterion(outputs_network, labels)
                    loss_layer = criterion(outputs_network, labels)
                    val_network_loss += loss_train.item()
                    val_layer_loss += loss_layer.item()
                else:
                    outputs = model(images)
                    test_outputs = outputs
                    loss = criterion(outputs.view(-1), labels.view(-1))
                    val_loss += loss.item()
            if out_layer:
                print(
                    'Epoch train network loss {}, Epoch train out layer loss {}, Epoch test network loss {}, Epoch test out layer loss {}'.format(
                        epoch_network_loss / len(train_loader),
                        epoch_out_layer_loss / len(train_loader),
                        val_network_loss / len(test_loader),
                        val_layer_loss / len(test_loader)))
            else:
                print('Epoch train loss: {}, Epoch test loss: {}'.format(
                    epoch_loss / len(train_loader),
                    val_loss / len(test_loader)))
        # Save best model with the lowest encoder decoder loss
        if val_network_loss / len(test_loader) < best_val_network_loss:
            best_val_network_loss = val_network_loss / len(test_loader)
            torch.save(model.state_dict(), '{}_model_best_{}_{}.pth'.format(prefix, model_name, epoch))
        else:
            epochs_not_improved += 1

        # Early stopping for the encoder decoder at 3 epochs not improved
        if epochs_not_improved > 3 and out_layer:
            training_network = False

        # Start training outlayer only after network is at a minimum
        if out_layer:
            if training_network:
                model.encoder.train()
                model.decoder.train()
            else:
                model.out_layer.train()
        else:
            model.train()

        # Save best model with the lowest outlayer loss
        if val_layer_loss / len(test_loader) < best_val_outlayer_loss:
            best_val_outlayer_loss = val_layer_loss / len(test_loader)
            torch.save(model.state_dict(), '{}_model_best_{}_{}.pth'.format(prefix, model_name, epoch))

        sys.stdout.flush()

        torch.save(model.state_dict(), '{}_model_{}.pth'.format(prefix, model_name, epoch))
        if save_pretrained_encoder:
            torch.save(model.encoder.state_dict(), '{}_encoder_{}_{}.pth'.format(prefix, model_name, epoch))

        if save_train:
            with open('{}_train_outputs_{}_{}.pkl'.format(prefix, model_name, epoch), 'wb') as f:
                pickle.dump(train_outputs.detach().cpu().numpy(), file=f)
        if save_test:
            with open('{}_test_outputs_{}_{}.pkl'.format(prefix, model_name, epoch), 'wb') as f:
                pickle.dump(test_outputs.detach().cpu().numpy(), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='Name of the model you want to train', default='frednet')
    parser.add_argument('--learning-rate', help='Model learning rate', default=0.001)
    parser.add_argument('--n_epochs', help='Number of epochs to train on', default=1000000)
    parser.add_argument('--optimizer', help='Optimizer for model', default='sgd')
    parser.add_argument('--momentum', help='Momentum for SGD', default=0.9)
    parser.add_argument('--betas', help='Betas for ADAM', default=(0.9, 0.999))
    parser.add_argument('--eps', help='eps for ADAM', default=1e-8)
    parser.add_argument('--batch-size', help='Batch size for training', default=32)
    parser.add_argument('--validation-split', help='Percentage of data to put in the validation set', default=0.2)
    parser.add_argument('--seed', help='Seed to use for shuffling the dataset', default=42)
    parser.add_argument('--save-train', help='Save last batch from the training set', default='n')
    parser.add_argument('--save-test', help='Save last batch from the test set', default='n')
    parser.add_argument('--save-pretrained-encoder', help='Save trained encoder', default='n')
    parser.add_argument('--use-residuals', help='Use residuals in the neural network', default='n')
    parser.add_argument('--dataset', help='Use COCO or FRED dataset', default='coco')
    parser.add_argument('--use-pretrained-encoder',
                        help='.pth file containing the pretrained encoder to be used to train the decoder')
    parser.add_argument('--prefix', help='Prefix appended to the model files')
    parser.add_argument('--out-layer', help='Use an extra layer in the decoder to further remove artifacts',
                        default='n')

    args = parser.parse_args()

    save_train = True if args.save_train == 'y' else False

    out_layer = True if args.out_layer == 'y' else False

    if out_layer:
        print('[INFO] Using out layer')
    if save_train:
        print('[INFO] Saving last training batch')
    save_test = True if args.save_test == 'y' else False
    if save_test:
        print('[INFO] Saving last testing batch')
    save_trained_encoder = True if args.save_pretrained_encoder == 'y' else False
    if save_trained_encoder:
        print('[INFO] Saving trained encoder after each epoch')
    use_residuals = True if args.use_residuals == 'y' else False
    if use_residuals:
        print('[INFO] Using residuals')

    assert args.model is not None, 'Please supply a model name: {}'.format(','.join(VALID_MODELS))
    train(model_name=args.model, learning_rate=float(args.learning_rate), optimizer_name=args.optimizer,
          pretrained_encoder=args.use_pretrained_encoder,
          momentum=float(args.momentum), batch_size=int(args.batch_size),
          betas=args.betas, eps=float(args.eps), validation_split=float(args.validation_split), random_seed=args.seed,
          save_pretrained_encoder=save_trained_encoder, n_epochs=int(args.n_epochs),
          use_residuals=use_residuals,
          dataset=args.dataset,
          prefix=args.prefix,
          out_layer=out_layer
          )
