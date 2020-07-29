import os
import torch
from models.model import NNet
from PIL import Image
import numpy as np
from utils.utils import eprint


def prepare_for_input(pilim, flip_lr=False):
    input_array = np.asarray(pilim) / 255
    if flip_lr:
        input_array = np.fliplr(input_array)

    return input_array


def get_tensor(input_array):
    tensor = torch.tensor(input_array.copy()).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def get_output(output):
    output = output[0].permute(1, 2, 0)
    out_image_array = output.detach().numpy()
    return out_image_array


def preprocess_pilim(pilim):
    pilim.thumbnail((512, pilim.size[1]), Image.ANTIALIAS)
    new_h = pilim.size[1] - pilim.size[1] % 32
    pilim = pilim.resize((512, new_h), Image.ANTIALIAS)
    return pilim


def threshold_output(out_array, threshold):
    assert 0.0 < threshold < 1.0, 'Threshold not in interval (0, 1)'
    out_array[out_array > threshold] = 1
    out_array[out_array <= threshold] = 0
    return out_array


def predict(image_file):
    model_path = os.path.join('inference/model_files', 'frednetv2.pth')
    if not os.path.exists(model_path):
        eprint("[ERR] Model file does not exist")
        exit(4)
    model = NNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        pilim = Image.open(image_file).convert('L').convert('RGB')
        pilim = preprocess_pilim(pilim)
        input_array = prepare_for_input(pilim, flip_lr=False)

        lr_input_array = prepare_for_input(pilim, flip_lr=True)
        try:
            out_array = get_output(model(get_tensor(input_array)))
        except:
            exit(2)

        lr_out_array = np.fliplr(get_output(model(get_tensor(lr_input_array))))

    out_array = (out_array + lr_out_array) / 2
    out_array = threshold_output(out_array, 0.5)
    out_array *= 255
    out_array = np.array(out_array, dtype='uint8')

    return out_array
