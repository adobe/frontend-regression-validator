from PIL import Image
import numpy as np
import os
import torch
import argparse


class Serializer(object):
    def __init__(self, root_dir, dir_type):
        """

        :param root_dir: Directory containing the photos to be serialized .png
        :param dir_type: I or M if the directory contains images or masks
        """
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir), f'Path {self.root_dir} does not exist'
        self.dir_type = dir_type
        assert self.dir_type == 'I' or self.dir_type == 'M', 'Arguments are I or M'

    def serialize(self):
        files = os.listdir(self.root_dir)
        n_files = len(files)
        assert n_files > 0, f'Directory is empty'

        for file_idx, file in enumerate(files):
            file_path = os.path.join(self.root_dir, file)
            if file.endswith('.png'):
                print(f'Current file: {file} - {file_idx + 1}/{n_files}')
                pil_image = None
                if self.dir_type == 'I':
                    pil_image = Image.open(file_path).convert('RGB')
                elif self.dir_type == 'M':
                    pil_image = Image.open(file_path).convert('L')
                assert pil_image is not None, f'Error: {file_path}'

                pil_image = np.array(pil_image, dtype='float')
                pil_image /= 255

                im_tensor = None
                if self.dir_type == 'I':
                    im_tensor = torch.from_numpy(pil_image).permute(2, 0, 1).float()
                elif self.dir_type == 'M':
                    im_tensor = torch.from_numpy(pil_image).unsqueeze(0).float()

                assert im_tensor is not None, 'im_tensor is None'

                save_path = file_path[:-4] + '.npy'
                with open(save_path, 'wb') as f:
                    np.save(save_path, im_tensor)
                    print(f'Saved numpy array to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Serializer that loads a directory containing .png images and saves them as preprocessed .npy arrays')
    parser.add_argument('--root-dir', help='Directory containing the photos to be serialized .png')
    parser.add_argument('--dir-type', help='I or M if the directory contains images or masks')

    args = parser.parse_args()
    root_dir = args.root_dir
    dir_type = args.dir_type

    s = Serializer(root_dir=root_dir, dir_type=dir_type)
    s.serialize()
