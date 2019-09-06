from datetime import datetime
from PIL import Image
import numpy as np
import os
import sys
from collections import defaultdict


def dsum(dicts, avg=False):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    if avg:
        for k in ret:
            ret[k] = round(ret[k] / len(dicts), ndigits=4)
    return dict(ret)


def dstd(dicts):
    ret = defaultdict()

    for d in dicts:
        for k, v in d.items():
            if k not in ret:
                ret[k] = [v]
            else:
                ret[k].append(v)
    for k in ret:
        ret[k] = round(np.std(ret[k]), ndigits=4)
    return dict(ret)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_time():
    now = datetime.now()
    current_time = now.strftime('%d-%m-%Y %H:%M:%S')
    return current_time


def add_auth(url, username, password):
    url_components = url.split(":")
    url = url_components[0] + "://" + username + ":" + password + "@" + url_components[1][2:]
    return url


def load_image_helper(image_file):
    image = Image.open(image_file).convert('L').convert('RGB')
    image.thumbnail((512, image.size[1]), Image.ANTIALIAS)
    new_h = image.size[1] - image.size[1] % 32
    image = image.resize((image.size[0], new_h), Image.ANTIALIAS)
    image = np.asarray(image) / 255

    return image[:, :, 0]


def crop_images(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    crop_shape = min(image1.size, image2.size)

    image1.crop([0, 0, crop_shape[0], crop_shape[1]]).save(image1_path)
    image2.crop([0, 0, crop_shape[0], crop_shape[1]]).save(image2_path)


def error_score(im1, im2, white_count):
    err = np.sum(im1 == im2)
    err /= white_count
    return 1 - err


def preprocess_save_image(image):
    return np.array(image * 255 * 255, dtype='uint8')


def save_masks(out_dir, image_filename, mask_np_arr, image_np_arr):
    buttons = mask_np_arr[:, :, 0]
    forms = mask_np_arr[:, :, 1]
    images = mask_np_arr[:, :, 2]
    section = mask_np_arr[:, :, 3]
    textblock = mask_np_arr[:, :, 4]
    Image.fromarray(preprocess_save_image(image_np_arr * buttons)).save(
        os.path.join(out_dir, "buttons_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * forms)).save(os.path.join(out_dir, "forms_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * images)).save(
        os.path.join(out_dir, "images_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * section)).save(
        os.path.join(out_dir, "section_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * textblock)).save(
        os.path.join(out_dir, "textblock_" + image_filename))


def check_unique_prefix(prefix, id_dict):
    for id in id_dict:
        if id_dict[id]['prefix'] == prefix:
            return False
    return True


def cost(arr1, arr2):
    arr1b = np.array(arr1, dtype=np.bool)
    arr2b = np.array(arr2, dtype=np.bool)
    overlap = arr1b * arr2b
    union = arr1b + arr2b

    overlap = np.count_nonzero(overlap)
    union = np.count_nonzero(union)

    IOU = 1.0
    if union > 0.0:
        IOU = overlap / float(union)

    return 1 - IOU


def match_images(arr1, arr2, step):
    n_steps1 = arr1.shape[1] // step
    n_steps2 = arr2.shape[1] // step
    d_mat = np.zeros((n_steps1, n_steps2))

    # base case
    for slice_idx in range(0, n_steps2):
        d_mat[0, slice_idx] = cost(arr1[:step, :], arr2[slice_idx * step: (slice_idx + 1) * step, :])

    for slice_idx in range(0, n_steps1):
        d_mat[slice_idx, 0] = cost(arr2[:step, :], arr1[slice_idx * step: (slice_idx + 1) * step, :])

    # filling in
    for l in range(1, n_steps1):
        for c in range(1, n_steps2):
            d_mat[l, c] = \
                cost(
                    arr1[l * step: (l + 1) * step, :],
                    arr2[c * step: (c + 1) * step, :]
                ) + min(d_mat[l - 1, c], d_mat[l, c - 1], d_mat[l - 1, c - 1])

    path = [(n_steps1 - 1, n_steps2 - 1)]
    curr_point = (n_steps1 - 1, n_steps2 - 1)
    while curr_point != (0, 0):
        possible = []
        if curr_point[0] > 0 and curr_point[1] > 0:
            possible.append((curr_point[0] - 1, curr_point[1] - 1))
            possible.append((curr_point[0], curr_point[1] - 1))
            possible.append((curr_point[0] - 1, curr_point[1]))
        elif curr_point[0] > 0 and curr_point[1] == 0:
            possible.append((curr_point[0] - 1, curr_point[1]))
        elif curr_point[0] == 0 and curr_point[1] > 0:
            possible.append((curr_point[0], curr_point[1] - 1))
        curr_min = 9999999
        next_point = -1
        for val in possible:
            if d_mat[val[0], val[1]] < curr_min:
                curr_min = d_mat[val[0], val[1]]
                next_point = val
        path.append(next_point)
        curr_point = next_point
    return path


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def union(lst1, lst2):
    return list(set(lst1) | set(lst2))
