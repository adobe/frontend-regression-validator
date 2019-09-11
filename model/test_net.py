from models.nnet import NNet
import torch
from PIL import Image
import numpy as np
import os
import argparse
from config.config import VALID_MODELS

CHANNELS = sorted(['images', 'section', 'buttons', 'forms', 'textblock'])
CHANNELS_DICT = dict(zip(CHANNELS, range(len(CHANNELS))))


def prepare_for_input(pilim, flip_lr=False, flip_ud=False):
    input_array = np.asarray(pilim) / 255
    if flip_lr:
        input_array = np.fliplr(input_array)
    if flip_ud:
        input_array = np.flipud(input_array)

    return input_array


def get_tensor(input_array):
    tensor = torch.tensor(input_array.copy()).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def get_output(output):
    output = output[0].permute(1, 2, 0)
    out_image_array = output.detach().numpy()
    return out_image_array


def test_net(model_name, model_file, trained_with_residuals, trained_with_out_layer, image_file, channel):
    assert model_name in VALID_MODELS, 'Please choose a valid model: {}'.format(', '.join(VALID_MODELS))
    assert os.path.exists(model_file), 'No such file {}'.format(model_file)
    assert os.path.exists(image_file), 'No such file {}'.format(image_file)
    channel = int(channel)
    assert channel in list(range(len(CHANNELS))), 'Please choose a valid channel: {}'.format(CHANNELS_DICT)
    model = NNet(out_channels=5, use_residuals=trained_with_residuals, model_name=model_name, out_layer=trained_with_out_layer)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    pilim = Image.open(image_file).convert('L').convert('RGB')
    pilim.thumbnail((512, pilim.size[1]), Image.ANTIALIAS)
    new_h = pilim.size[1] - pilim.size[1] % 32
    pilim = pilim.resize((512, new_h), Image.ANTIALIAS)

    pilim.show()

    correct_input_array = prepare_for_input(pilim)
    lr_flipped_input_array = prepare_for_input(pilim, flip_lr=True)


    if trained_with_out_layer:
        _ , output = model(get_tensor(correct_input_array))
        correct_out_image_array = get_output(output)

        _ , output = model(get_tensor(lr_flipped_input_array))

        lr_out_image_array = np.fliplr(get_output(output))

    else:
        correct_out_image_array = get_output(model(get_tensor(correct_input_array)))
        lr_out_image_array = np.fliplr(get_output(model(get_tensor(lr_flipped_input_array))))

    out_image_array = (correct_out_image_array + lr_out_image_array) / 2

    out_image_array[out_image_array > 0.5] = 1
    out_image_array[out_image_array <= 0.5] = 0
    out_image_array *= 255

    out_image_array = np.array(out_image_array, dtype='uint8')

    out_pilim = Image.fromarray(out_image_array[:, :, channel])
    out_pilim.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Name of the model from {}'.format(', '.join(VALID_MODELS)))
    parser.add_argument('--model-file', help='.pth file containing the state dict of the model')
    parser.add_argument('--image-file', help='Image file to test on')
    parser.add_argument('--trained-with-residuals', help='True if the model was trained with residuals')
    parser.add_argument('--channel', help='What channel to show: {}'.format(CHANNELS_DICT))
    parser.add_argument('--trained-with-out-layer', help='Trained with extra out layer')

    args = parser.parse_args()

    trained_with_residuals = True if args.trained_with_residuals == 'y' else False
    trained_with_out_layer = True if args.trained_with_out_layer == 'y' else False
    test_net(args.model_name, args.model_file, trained_with_residuals,trained_with_out_layer ,args.image_file, channel=args.channel,)
