import argparse
import json
import os
from PIL import Image, ImageDraw

INDENT_SIZE = 7


class MaskGenerator(object):
    def __init__(self, path, desired_path):
        self.abs_path = path
        self.desired_path = desired_path
        self.paths = os.listdir(path)

    def generate_masks(self, json_path, counter):
        dimensions = os.popen("file " + self.abs_path + "/website_screenshot.png").read()
        dimensions = [int(dim) for dim in dimensions.split(",")[1].split("x")]
        print("Image has dimensions of {} by {}".format(dimensions[0], dimensions[1]))

        if not os.path.isdir(os.path.join(self.desired_path, "images")):
            os.makedirs(os.path.join(self.desired_path, "images"), exist_ok=True)
            print("Created images folder")
        os.system(
            "cp " + self.abs_path + "/website_screenshot.png " + self.desired_path + "/" + "images/" + str(
                counter) + ".png")

        with open(os.path.join(self.abs_path, json_path)) as json_file:
            json_dictionary = json.load(json_file)
            for key in json_dictionary:
                if json_dictionary[key]:
                    print("Painting {}".format(key))

                    image = Image.new('RGB', size=dimensions, color='black')
                    draw = ImageDraw.Draw(image)

                    for coordinates in json_dictionary[key]:
                        x1 = coordinates['x1'] + INDENT_SIZE
                        x2 = coordinates['x2'] - INDENT_SIZE
                        y1 = coordinates['y1'] + INDENT_SIZE
                        y2 = coordinates['y2'] - INDENT_SIZE
                        draw.rectangle(((x1, y1), (x2, y2)), fill='white')

                    if not os.path.exists(os.path.join(self.desired_path, key)):
                        os.makedirs(os.path.join(self.desired_path, key), exist_ok=True)

                    image.save(os.path.join(self.desired_path, key, str(counter) + ".png"))
                    print(
                        "Saved segment mask to {}".format(os.path.join(self.desired_path, key, str(counter)) + ".png"))
        print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to format the database in the required structure")
    parser.add_argument('--website-directory', help='Directory of a website from the dataset')
    parser.add_argument('--destination-directory', help='Website destination directory containing the masked image')
    parser.add_argument('--json-name',
                        help='JSON file name containing website components found in each website directory')
    args = parser.parse_args()

    mg = MaskGenerator(args.website_directory, args.destination_directory)
    mg.generate_masks(args.json_name, 0)
