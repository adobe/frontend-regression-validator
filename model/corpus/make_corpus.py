import os
from corpus.masker import MaskGenerator
from corpus.datapreprocess import DataPreprocessor
from corpus.serialize_photos import Serializer
from corpus.concat_channels import Concatenator
import shutil
from PIL import Image
import numpy as np
from argparse import ArgumentParser


def remove_bad_dirs(scraped_websites_dir):
    for i, folder in enumerate(os.listdir(scraped_websites_dir)):
        folder_path = os.path.join(scraped_websites_dir, folder)
        if not os.path.exists(os.path.join(folder_path, 'website_screenshot.png')):
            shutil.rmtree(folder_path)
            continue
        img = Image.open(os.path.join(folder_path, 'website_screenshot.png'))
        if np.sum(np.asarray(img)) == 0:
            shutil.rmtree(folder_path)
            print("Removed folder {}".format(folder_path))


def make_masks(root_dir):
    websites = os.listdir(root_dir)
    idx = 0
    n_websites = len(websites)
    for website_dir in websites:
        if website_dir.startswith('.'):
            continue
        idx += 1
        print('{}/{} -- Current website: {}'.format(idx, n_websites, website_dir))
        website_dir_path = os.path.join(root_dir, website_dir)
        if 'images' in os.listdir(website_dir_path):
            continue
        json_name = 'components.json'
        mask_generator = MaskGenerator(website_dir_path, website_dir_path)
        mask_generator.generate_masks(json_name, idx)


def gather_directories(root_dir, main_dirs, out_dir):
    assert os.path.exists(root_dir), 'No such file or directory {}'.format(root_dir)
    os.mkdir(out_dir)
    websites = os.listdir(root_dir)
    n_websites = len(websites)
    for d in main_dirs:
        os.mkdir(os.path.join(out_dir, d))
    os.mkdir(os.path.join(out_dir, 'images'))
    for i, website in enumerate(websites):
        print('{}/{} Current website: {}'.format(i, n_websites, website))
        website_path = os.path.join(root_dir, website)
        for main_dir in main_dirs:
            main_dir_path = os.path.join(website_path, main_dir)
            if not os.path.exists(main_dir_path):
                continue
            image_file = os.listdir(main_dir_path)[0]
            image_file_path = os.path.join(main_dir_path, image_file)
            out_image_file_path = os.path.join(out_dir, main_dir, image_file)
            shutil.copy(image_file_path, out_image_file_path)
        shutil.copy(os.path.join(website_path, 'website_screenshot.png'),
                    os.path.join(out_dir, 'images', '{}.png'.format(i + 1)))


def make_corpus(scraped_websites_dir, class_dirs):
    assert os.path.exists(scraped_websites_dir), '{} - No such file or directory'.format(scraped_websites_dir)

    # Remove bad website directories
    remove_bad_dirs(scraped_websites_dir)

    # First, create the masked images based on the json files
    make_masks(scraped_websites_dir)

    # Gather everything in '../dataset' directory
    gather_directories(scraped_websites_dir, class_dirs, './dataset')

    # Resize and crop
    for c in class_dirs:
        dp = DataPreprocessor(os.path.join('./dataset/', c), 512, 512, 32)
        dp.preprocess()

    # Save .png images as .npy
    serializer = Serializer('./dataset/images', 'I')
    serializer.serialize()

    # Concatenate the channels
    concat = Concatenator('./dataset', 'images', class_dirs, 512)
    concat.concat_channels()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--scraped-websites-dir', type=str, help='Directory containing crawled websites')
    parser.add_argument('-c', '--class-dirs', action='store', type=str, nargs='*',
                        help='Classes to gather from the directory')

    args = parser.parse_args()
    make_corpus(scraped_websites_dir=args.scraped_websites_dir, class_dirs=args.class_dirs)
