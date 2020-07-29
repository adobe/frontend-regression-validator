import os
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image

# FREDNET Mean rgb
means = (0.7486, 0.7486, 0.7556)
# FREDNET std rgb
stddevs = (0.3451, 0.3382, 0.3372)


class FREDDataset(Dataset):
    def __init__(self, images_dir, masks_dir, flip_threshold=0.5):
        """
        :param images_dir: the path where the images are located
        :param masks_dir: the path where the masks are located for the current model
        :param v_crop_size: height of each slice of a photo
        :param stride: stride of the cropping window
        """
        assert os.path.exists(images_dir), f"No '{images_dir}' images directory found"
        assert os.path.exists(masks_dir), f"No '{masks_dir}' masks directory found"
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_arr = []
        self.flip_threshold = flip_threshold
        self.__load_images()

    def __getitem__(self, index):
        """
        :param index: index of data
        :return: tuple containing (image, mask) tensors
        """
        image_path = os.path.join(self.images_dir, self.images_arr[index])
        mask_path = os.path.join(self.masks_dir, self.images_arr[index])

        im_array = np.load(image_path)
        im_array[0] = (im_array[0] - means[0]) / stddevs[0]
        im_array[1] = (im_array[1] - means[1]) / stddevs[1]
        im_array[2] = (im_array[2] - means[2]) / stddevs[2]

        image_tensor = torch.from_numpy(np.load(image_path))
        mask_tensor = torch.from_numpy(np.load(mask_path)).permute(2, 0, 1).float()

        if self.flip_threshold > np.random.uniform(0, 1, 1):
            image_tensor = torch.flip(image_tensor, [2])
            mask_tensor = torch.flip(mask_tensor, [2])

        sample = {'img': image_tensor, 'mask': mask_tensor}
        return sample

    def __len__(self):
        """
        :return: length of images arrays
        """
        return len(self.images_arr)

    def __load_images(self):
        temp_masks = os.listdir(self.masks_dir)
        all_masks = sorted([mask for mask in temp_masks if mask.endswith('.npy')])
        n_masks = len(all_masks)

        # Iterate through masks because sometimes masks are missing
        for i, image_name in enumerate(all_masks):
            if '_' not in image_name:
                continue
            self.images_arr.append(image_name)


class COCODatasetEncoderPretrain(Dataset):
    def __init__(self, images_dir):
        """
        :param images_dir: the path where the images are located
        """
        assert os.path.exists(images_dir), f"No '{images_dir}' images directory found"
        self.images_dir = images_dir
        self.images_arr = []
        self.__load_images()

    def __getitem__(self, index):
        """
        :param index: index of data
        :return: tuple containing (image, mask) tensors
        """
        image_path = os.path.join(self.images_dir, self.images_arr[index])
        pilim = Image.open(image_path).convert('RGB').resize((512, 512), Image.ANTIALIAS)
        np_arr = np.asarray(pilim)

        image_tensor = torch.from_numpy(np_arr / 255).permute(2, 0, 1).float()

        sample = {'img': image_tensor, 'mask': image_tensor}
        return sample

    def __len__(self):
        """
        :return: length of images arrays
        """
        return len(self.images_arr)

    def __load_images(self):
        temp_images = os.listdir(self.images_dir)
        all_masks = sorted([image for image in temp_images if image.endswith('.jpg')])

        # Iterate through masks because sometimes masks are missing
        for i, image_name in enumerate(all_masks):
            self.images_arr.append(image_name)
