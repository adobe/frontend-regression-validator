import numpy as np
import os
from PIL import Image
import argparse

OUT_DIR = 'concatenated'


class Concatenator(object):
    def __init__(self, root_dir, images_dir, masks_dir_list, image_dim):
        """

        :param root_dir: Directory containing the dataset
        :param images_dir: Directory containing base images
        :param masks_dir_list: List of directories containing the masks that we want to concatenate
        """
        assert os.path.exists(root_dir), f'{root_dir} does not exist'
        self.root_dir = root_dir
        assert len(masks_dir_list) > 1, 'Not enough channels to concatenate'
        images_dir_path = os.path.join(root_dir, images_dir)
        assert os.path.exists(images_dir_path), f'{images_dir_path} does not exist'
        for directory in masks_dir_list:
            dir_path = os.path.join(root_dir, directory)
            assert os.path.exists(dir_path), f'{dir_path} does not exist'
        self.images_dir = images_dir
        self.masks_dir_list = sorted(masks_dir_list)
        self.n_channels = len(self.masks_dir_list)
        self.image_dim = image_dim

    def concat_channels(self):
        new_dir_path = os.path.join(self.root_dir, OUT_DIR)
        if os.path.exists(new_dir_path):
            os.rmdir(new_dir_path)
        os.mkdir(new_dir_path)
        assert os.path.exists(new_dir_path), 'Failed to create new directory'
        print(f'Created new directory {new_dir_path}')

        images_dir_path = os.path.join(self.root_dir, self.images_dir)
        images = os.listdir(images_dir_path)
        images = sorted([im for im in images if im.endswith('.png') and '_' in im])
        assert len(images) > 0, 'No images in the images directory'

        directory_files = {}
        for directory in self.masks_dir_list:
            directory_path = os.path.join(self.root_dir, directory)
            directory_images = os.listdir(directory_path)
            directory_images = sorted([im for im in directory_images if im.endswith('.png')])
            directory_files[directory] = directory_images

        n_images = len(images)
        for idx, image_name in enumerate(images):
            print(f'Current image: {image_name} - {idx + 1}/{n_images}')
            concat_image = np.zeros((self.image_dim, self.image_dim, self.n_channels))
            for ch, directory in enumerate(self.masks_dir_list):
                print(f'\tCurrent channel: {ch} - {directory}')
                images_in_directory = directory_files[directory]
                # If the image name can't be found in the masks folder, then skip the current channel
                if image_name not in images_in_directory:
                    print('\tSkipped\n')
                    continue

                # Else, open the image and append
                image_path = os.path.join(self.root_dir, directory, image_name)
                pil_image = Image.open(image_path).convert('L')
                np_image = np.array(pil_image, dtype='float') / 255
                print('\tAppended\n')
                concat_image[:, :, ch] = np_image
            print(f'Created concatenated image {image_name}')
            out_path = os.path.join(self.root_dir, OUT_DIR, image_name[:-4] + '.npy')
            np.save(out_path, concat_image)
            assert os.path.exists(out_path), f'Failed to save to {out_path}'
            print(f'Saved image to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='Directory containing the dataset')
    parser.add_argument('--images-dir', help='Name of directory inside dataset directory containing the images')
    parser.add_argument('--channels', action='store', type=str, nargs='*', help='Name of channels to concatenate')
    parser.add_argument('--image-dim', help='Size of the image')

    args = parser.parse_args()

    c = Concatenator(root_dir=args.root_dir, images_dir=args.images_dir, masks_dir_list=args.channels,
                     image_dim=int(args.image_dim))
    c.concat_channels()
