import os
import shutil
import argparse


def gather_directories(root_dir, main_dirs, out_dir):
    assert os.path.exists(root_dir), 'No such file or directory {}'.format(root_dir)
    websites = os.listdir(root_dir)
    n_websites = len(websites)
    for i, website in websites:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='Directory containing the directories with the masks and websites')
    parser.add_argument('--main-dirs', help='Directories to look for in the websites root directory', nargs='*',
                        type=str, action='store')
    parser.add_argument('--out-dir', help='Directory to save the gathered dataset to')
    args = parser.parse_args()
    gather_directories(root_dir=args.root_dir, main_dirs=args.main_dirs, out_dir=args.out_dir)
