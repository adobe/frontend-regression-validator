import os, torch, logging
from ml.model import NNet
from PIL import Image
import numpy as np

def prepare_for_input(pilim, flip_lr=False):
    input_array = np.asarray(pilim) / 255
    if flip_lr:
        input_array = np.fliplr(input_array)

    return input_array


def get_tensor(input_array, device):
    tensor = torch.tensor(input_array.copy()).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)


def get_output(output):
    output = output[0].permute(1, 2, 0)
    out_image_array = output.detach().cpu().numpy()
    return out_image_array


def preprocess_pilim(pilim):
    logging.debug("Initial image size: {} : {:.2f}MP".format(pilim.size, pilim.size[0]*pilim.size[1]/1000/1000))
    pilim.thumbnail((512, pilim.size[1]), Image.ANTIALIAS)
    #logging.debug("512 resize h image size: {}".format(pilim.size))
    new_h = pilim.size[1] - pilim.size[1] % 32
    #logging.debug("new h: {}".format(new_h))
    pilim = pilim.resize((512, new_h), Image.ANTIALIAS)
    logging.debug("Final image size: {} : {:.2f}MP".format(pilim.size, pilim.size[0]*pilim.size[1]/1000/1000))

    return pilim


def threshold_output(out_array, threshold):
    assert 0.0 < threshold < 1.0, 'Threshold not in interval (0, 1)'
    out_array[out_array > threshold] = 1
    out_array[out_array <= threshold] = 0
    return out_array


def load_model():
    # disable PIL logging
    logging.getLogger("PIL").setLevel(logging.ERROR)
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logging.info('ML using device: GPU ['+torch.cuda.get_device_name(0)+']')
    else:
        logging.info('ML using device: CPU')

    model_path = os.path.join('ml/model_files', 'frednetv2.pth')
    if not os.path.exists(model_path):
        logging.error("Model file not found: ".format(model_path))
        return None, None, "Model file not found: ".format(model_path)

    error = ""
    try:
        model = NNet()
        model.load_state_dict(torch.load(model_path, map_location='cpu')) # because model was saved on GPU initially
        model.eval()
        model.to(device)
        logging.debug("Model loaded successfully")
    except Exception as ex:
        error = str(ex)

    return model, device, error

def predict(image_file, model, device):
    with torch.no_grad():
        pilim = Image.open(image_file).convert('L').convert('RGB')
        pilim = preprocess_pilim(pilim)
        input_array = prepare_for_input(pilim, flip_lr=False)

        lr_input_array = prepare_for_input(pilim, flip_lr=True)
        #print(torch.cuda.memory_summary(device=device))
        try:
            logging.debug("Start to predict [{}]".format(image_file))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            out_array = get_output(model(get_tensor(input_array, device)))

        except Exception as ex:
            return None, str(ex)
        #print(torch.cuda.memory_summary(device=device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logging.debug("Half-way there ...".format(image_file))
        lr_out_array = np.fliplr(get_output(model(get_tensor(lr_input_array, device))))
        logging.debug("End of prediction.")
        #print(torch.cuda.memory_summary(device=device))
        #print("__________________________________________")

    out_array = (out_array + lr_out_array) / 2
    out_array = threshold_output(out_array, 0.5)
    out_array *= 255
    out_array = np.array(out_array, dtype='uint8')

    out_array = np.delete(out_array, 3, 2)
    out_array = np.delete(out_array, 0, 2)
    out_array = np.delete(out_array, 0, 2)

    return out_array, ""
