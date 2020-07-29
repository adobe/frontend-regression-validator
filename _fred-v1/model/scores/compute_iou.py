import numpy as np


def compute_iou(array1, array2):
    """
    :param array1: numpy array with first black and white image
    :param array2: numpy array with second black and white image
    :return: IOU of the 2 images
    """
    assert array1 is not None, "First parameter is None"
    assert array2 is not None, "Second parameter is None"
    array1 = np.array(array1, dtype=bool)
    array2 = np.array(array2, dtype=bool)
    intersection = array1 * array2
    union = array1 + array2

    non_zero_intersection = np.count_nonzero(intersection)
    non_zero_union = np.count_nonzero(union)

    assert non_zero_union > 0, 'Both images are black'
    iou = non_zero_intersection / float(non_zero_union)
    
    return iou
