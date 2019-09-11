import pickle
import numpy as np
from PIL import Image
import os
import argparse
import shutil

CHANNELS = sorted(['button', 'card', 'carousel', 'form', 'menu', 'section', 'table', 'textblock'])


def visualize(pkl_file, n_images):
    """

    :param pkl_file: Pickle file containing the saved outputs of the neural net
    :param n_images: How many images to show
    :return:
    """
    assert os.path.exists(pkl_file), 'No such file exists'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    n_images = int(n_images)
    n_data = data.shape[0]
    assert n_images <= n_data, f'Number of images must be less than or equal to {n_data}'

    for i in range(n_images):
        save_dir = f'image_{i + 1}'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        assert os.path.exists(save_dir), f'Could not create new directory {save_dir}'
        channels = data[i] * 255
        for ch_idx, image in enumerate(channels):
            ch_name = CHANNELS[ch_idx]
            image = np.array(image, dtype='uint8')
            pilim = Image.fromarray(image, 'L')
            image_save_path = os.path.join(save_dir, ch_name + '.png')
            pilim.save(image_save_path, 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', help='Pickle file containing the saved outputs of the neural net')
    parser.add_argument('--count', help='How many images to show')

    args = parser.parse_args()

    pkl_file = args.file
    n_images = args.count

    visualize(pkl_file, n_images)
