from PIL import Image
import os
import argparse


class DataPreprocessor(object):
    def __init__(self, root_dir, new_w, v_crop_size, stride):
        assert os.path.exists(root_dir), f'{root_dir} does not exist'
        self.root_dir = root_dir
        self.v_crop_size = v_crop_size
        self.stride = stride
        self.new_w = new_w

    def preprocess(self, overwrite=True):
        images = os.listdir(self.root_dir)
        images = [im for im in images if im.endswith('.png')]
        assert len(images) > 0, 'No images in the directory'
        n_images = len(images)
        for idx, image_name in enumerate(images):
            print(f'Resizing {image_name} - {idx + 1}/{n_images}')
            image_path = os.path.join(self.root_dir, image_name)
            image = Image.open(image_path)
            w, h = image.size
            image.thumbnail((self.new_w, h), Image.ANTIALIAS)

            if image.size[1] < self.v_crop_size:
                image = self.__pad_image(image, self.v_crop_size + 10)

            print('Resized image. Now slicing image')
            image_slices = self.__slice_image(image)
            n_slices = len(image_slices)
            print('Sliced image. Saving slices')
            for i, image_slice in enumerate(image_slices):
                image_slice_path = os.path.join(self.root_dir, f"{image_name.split('.')[0]}_{i}.png")
                image_slice.save(image_slice_path, 'PNG')
                print(f'Saved slice {i}/{n_slices} to {image_slice_path}')

            print('Finished saving slices. Saving resized image')
            if overwrite:
                image.save(image_path, "PNG")
                print(f'Saved resized image to {image_path}')
            else:
                out_file = f'resized_{image_name}'
                out_file_path = os.path.join(self.root_dir, out_file)
                image.save(out_file_path, "PNG")
                print(f'Saved resized image to {out_file_path}')

    def __slice_image(self, im):
        """
        :param im: PIL image to be sliced
        :return: array of PIL images containing the slices
        """
        w, h = im.size
        slices = []
        x1 = 0
        y1 = 0
        x2 = w
        y2 = self.v_crop_size
        # Keep slicing until I escape the image height
        while y2 <= h:
            curr_slice = im.crop((x1, y1, x2, y2))
            slices.append(curr_slice)

            # Update the old area:
            # x1 = 0
            # y1 = y1_prev + stride
            # x2 = w
            # y2 = y2_prev + stride
            y1 += self.stride
            y2 += self.stride
        return slices

    def __pad_image(self, image, new_h):
        new_image = Image.new('RGB', (image.size[0], new_h), (255, 255, 255))
        new_image.paste(image)
        return new_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='Directory containing png files to be sliced')
    parser.add_argument('--new_width', help='Width of the saved slices')
    parser.add_argument('--v-crop-size', help='Height of saved slices')
    parser.add_argument('--stride', help='What stride to crop with')

    args = parser.parse_args()

    dp1 = DataPreprocessor(root_dir=int(args.root_dir), new_w=int(args.new_width), v_crop_size=int(args.v_crop_size),
                           stride=float(args.stride))

    dp1.preprocess()
