import os
from PIL import Image
import numpy as np
import argparse


def preprocess_coco(root_dir):
    assert os.path.exists(root_dir), 'No such file or directory, {}'.format(root_dir)
    coco_images = os.listdir(root_dir)

    n_images = len(coco_images)

    for i, image in enumerate(coco_images):
        print('Opening {}'.format(image))
        image_path = os.path.join(root_dir, image)
        pilim = Image.open(image_path)
        pilim = pilim.resize((512, 512), Image.ANTIALIAS)
        np_array = np.asarray(pilim)
        save_path = image_path[:-4] + '.npy'
        np.save(save_path, np_array)
        print('{}/{} - Saved to {}'.format(i, n_images, save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='Directory containing the coco dataset')
    args = parser.parse_args()
    preprocess_coco(root_dir=args.root_dir)
