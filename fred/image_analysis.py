from skimage.metrics import structural_similarity, normalized_root_mse
import imutils
import cv2
import os
import numpy as np
import glob, time, gc
from globals import MAX_MEGAPIXELS, URL_SLASH_REPLACEMENT_STR, MEDIAN_KERNEL_WIDTH_RATIO, IMAGE_RESIZE_RATIO
from numpy import zeros
import logging
from scipy.ndimage import median_filter


DYNAMIC_CONTENT_SUFFIX = "_dynamic_content.png"

def raw_screenshots_analysis(jobs, job_id):
    """
    Returns a report (dict) with:
        report["raw_screenshot_mse"] = 0. <- [0, 100], 0 meaning images are equal
        report["raw_screenshot_ssim"] = 0. <- [0, 100], 100 meaning images are equal
        report["raw_screenshot_diff_pixels"] = 0. <- [0, 100], 0 meaning images are equal
        report["raw_screenshot_divergence] = 0. <- [0, 100], 0 meaning images are equal, it is the combination of the indicators above
            * all of the above are mins()/maxes() of all pages. It basically reports the most diverging screenshot pair

        For each page:
        report["pages"][p]["raw_screenshot_analysis"][resolution]["mse"] = mse
        report["pages"][p]["raw_screenshot_analysis"][resolution]["ssim"] = ssim
        report["pages"][p]["raw_screenshot_analysis"][resolution]["diff_pixels"] = percent_different_pixels
        report["pages"][p]["raw_screenshot_analysis"][resolution]["baseline_file_path"] = "jobs//job_id//...//filename.png"
        report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path"] = ...
        report["pages"][p]["raw_screenshot_analysis"][resolution]["baseline_file_path_highlight"] = bf.replace(".png", "_highlight.png")
        report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_highlight"] = uf.replace(".png", "_highlight.png")
        report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_difference"] = uf.replace(".png", "_difference.png")
        report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_threshold"] = uf.replace(".png", "_threshold.png")

    """
    logging.info("Starting raw image analysis for job id {}".format(job_id))

    report = {}
    error = ""

    base_path = os.path.join("jobs", job_id)
    baseline_pages = [x for x in os.listdir(os.path.join(base_path, "baseline")) if
                      os.path.isdir(os.path.join(base_path, "baseline", x))]
    updated_pages = [x for x in os.listdir(os.path.join(base_path, "updated")) if
                     os.path.isdir(os.path.join(base_path, "updated", x))]

    common_pages = set(baseline_pages) & set(updated_pages)

    report["raw_screenshot_mse"] = 100.
    report["raw_screenshot_ssim"] = 0.
    report["raw_screenshot_diff_pixels"] = 100.
    report["raw_screenshot_divergence"] = 100.

    mse_scores = []
    ssim_scores = []
    diff_pixels = []
    avgs = []

    report["pages"] = {}
    for p in common_pages:
        # print("-"*50)
        # print(p)

        report["pages"][p] = {}
        report["pages"][p]["raw_screenshot_analysis"] = {}

        # get resolutions
        base_files = []
        for f in glob.glob(os.path.join("jobs", job_id, "baseline", p, "raw_*.png")):
            dir, name = os.path.split(f)
            parts = name.replace(".png", "").split("_")
            if len(parts) == 3:
                base_files.append(f)
        upgr_files = []
        for f in glob.glob(os.path.join("jobs", job_id, "updated", p, "raw_*.png")):
            dir, name = os.path.split(f)
            parts = name.replace(".png", "").split("_")
            if len(parts) == 3:
                upgr_files.append(f)

        pairs = []  # this part should never throw and error but we need to check everytime
        for bf in base_files:
            dir, name = os.path.split(bf)
            counterpart = os.path.join(base_path, "updated", p, name)
            if not os.path.exists(counterpart):
                pass  # TODO throw error if resolution does not exist
            if [bf, counterpart] not in pairs:
                pairs.append([bf, counterpart])
        for uf in upgr_files:
            dir, name = os.path.split(uf)
            counterpart = os.path.join(base_path, "baseline", p, name)
            if not os.path.exists(counterpart):
                pass  # TODO throw error if resolution does not exist
            if [counterpart, uf] not in pairs:
                pairs.append([counterpart, uf])

        if len(pairs) == 0:
            error += " Page {} does not have any screenshots! ".format(p)
            continue

        for [bf, uf] in pairs:
            _, name = os.path.split(uf)
            resolution = name.split("_")[1]  # e.g. raw_512_00.png
            logging.debug("Running raw analysis on {}, page {} resolution {} ...".format(job_id, p, resolution))

            report["pages"][p]["raw_screenshot_analysis"][resolution] = {}
            mse, ssim, percent_different_pixels, perror = raw_screenshot_analysis(bf, uf)
            mse_scores.append(mse)
            ssim_scores.append(ssim)
            diff_pixels.append(percent_different_pixels)
            avgs.append((mse + (100 - ssim) + percent_different_pixels) / 3)
            report["pages"][p]["raw_screenshot_analysis"][resolution]["baseline_file_path"] = bf
            report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path"] = uf
            report["pages"][p]["raw_screenshot_analysis"][resolution]["baseline_file_path_highlight"] = bf.replace(
                ".png", "_highlight.png")
            report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_highlight"] = uf.replace(
                ".png", "_highlight.png")
            report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_difference"] = uf.replace(
                ".png", "_difference.png")
            report["pages"][p]["raw_screenshot_analysis"][resolution]["updated_file_path_threshold"] = uf.replace(
                ".png", "_threshold.png")
            report["pages"][p]["raw_screenshot_analysis"][resolution]["mse"] = mse
            report["pages"][p]["raw_screenshot_analysis"][resolution]["ssim"] = ssim
            report["pages"][p]["raw_screenshot_analysis"][resolution]["diff_pixels"] = percent_different_pixels
            if perror:
                error += perror + " "

    if len(mse_scores) > 0:
        report["raw_screenshot_mse"] = max(mse_scores)
        report["raw_screenshot_ssim"] = min(ssim_scores)
        report["raw_screenshot_diff_pixels"] = max(diff_pixels)
        # HERE IS THE MAIN CALCULATION
        report["raw_screenshot_divergence"] = max(avgs)

    logging.info("Finished crawl analysis for job id {}".format(job_id))
    gc.collect()  # force RAM release

    return report, error


# @profile
def raw_screenshot_analysis(baseline_image_file, updated_image_file, threshold=5):
    """
        threshold is between 0 and 255
            For an image size: 57496 x 3840 = 210.56MP, it takes about 200seconds
            For about 20MP it takes around 10s and 0.5GB RAM

        Returns three numbers [0,100]
    """
    start_timer = time.perf_counter()
    mse = 1.
    ssim_score = 0.
    percent_different_pixels = 100.
    error = None

    try:
        base_img = cv2.imread(baseline_image_file, 1)
        upgr_img = cv2.imread(updated_image_file, 1)

        base_img_height, base_img_width, _ = base_img.shape
        upgr_img_height, upgr_img_width, _ = upgr_img.shape

        assert base_img_width == upgr_img_width, "Images have different widths, this is not allowed!"

        base_mp = base_img_height * base_img_width
        upgr_mp = upgr_img_height * upgr_img_width
        logging.debug("Image sizes: {} x {} = {:.2f}MP / {} x {} = {:.2f}MP ..".format(base_img_height, base_img_width,
                                                                                       base_mp / 1000 / 1000,
                                                                                       upgr_img_height, upgr_img_width,
                                                                                       upgr_mp / 1000 / 1000))
        original_mp = max(base_mp, upgr_mp) / 1000 / 1000

        # resize images if necessary
        if base_img_height > upgr_img_height:  # pad the updated image
            upgr_img = cv2.copyMakeBorder(upgr_img.copy(), 0, base_img_height - upgr_img_height, 0, 0,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if base_img_height < upgr_img_height:  # pad the baseline image
            base_img = cv2.copyMakeBorder(base_img.copy(), 0, upgr_img_height - base_img_height, 0, 0,
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # now, if we have images that are too large, truncate to MAX_MEGAPIXELS
        if MAX_MEGAPIXELS > 0:
            if original_mp > MAX_MEGAPIXELS:
                keep_height = MAX_MEGAPIXELS * 1024 * 1024 / base_img_width
                keep_height = int(min(keep_height, max(base_img_height, upgr_img_height)))  # sanity check
                logging.info("Images exceed max megapixels allowed ({:.2f}MP), keeping just the top {} pixels.".format(
                    MAX_MEGAPIXELS, keep_height))
                base_img = base_img[0:keep_height, :, :].copy()
                upgr_img = upgr_img[0:keep_height, :, :].copy()
                # gc.collect()

        # convert to grayscale
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        upgr_gray = cv2.cvtColor(upgr_img, cv2.COLOR_BGR2GRAY)

        # compute MSE and Structural Similarity Index (SSIM)
        mse = normalized_root_mse(base_gray, upgr_gray)
        # print("mse = {}".format(mse))
        (ssim_score, diff) = structural_similarity(base_gray, upgr_gray, full=True)
        diff = (diff * 255).astype("uint8")
        # print("SSIM: {}".format(ssim_score))
        del base_gray  # we still need upgr_gray

        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        bounding_base_img = base_img.copy()
        bounding_upgr_img = upgr_img.copy()
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(bounding_base_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(bounding_upgr_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # show the output images
        cv2.imwrite(baseline_image_file.replace(".png", "_highlight.png"), bounding_base_img)
        cv2.imwrite(updated_image_file.replace(".png", "_highlight.png"), bounding_upgr_img)

        # cv2.imwrite(updated_image_file.replace(".png", "_difference.png"), diff)
        cv2.imwrite(updated_image_file.replace(".png", "_threshold.png"), thresh)
        del bounding_base_img, bounding_upgr_img, diff, thresh

        diff = cv2.absdiff(base_img, upgr_img)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        imask = mask > threshold
        res = np.where(mask.flatten() > threshold)
        cnt_dif = len(res[0])
        all_pixels = mask.shape[0] * mask.shape[1]
        percent_different_pixels = cnt_dif / all_pixels
        # print("{:.4f}% pixels are different with a treshold of {}".format(percent_different_pixels, threshold))

        upgr_img[:, :, 0] = upgr_gray
        upgr_img[:, :, 1] = upgr_gray
        upgr_img[:, :, 2] = upgr_gray
        upgr_img[imask] = [0, 0, 255]

        cv2.imwrite(updated_image_file.replace(".png", "_difference.png"), upgr_img)

    except Exception as ex:
        error = "{}".format(ex)

    end_timer = time.perf_counter()
    logging.debug("Raw screenshot analysis done in {:.1f}s on {}".format(end_timer - start_timer, baseline_image_file))

    return mse * 100, ssim_score * 100, percent_different_pixels * 100, error


def process_variance_threshold(value, high_count, low_count, min_occurences, inferior_theshold, superior_threshold):
    if value > superior_threshold:
        high_count += 1
        low_count = 0
        value = 0
    else:
        if value <= inferior_theshold:
            low_count += 1
            high_count = 0
            value = 255
        else:
            if high_count >= min_occurences:
                value = 0
            else:
                if low_count >= min_occurences:
                    value = 255

    return value, high_count, low_count


def save_and_mask_image(image_data, file_path, baseline_file_path):
    error = ''
    try:
        dynamic_content_mask_path = baseline_file_path.replace(".png", DYNAMIC_CONTENT_SUFFIX)
        image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), 1)
        img_height, img_width, _ = image.shape
        dynamic_content_mask = cv2.imread(dynamic_content_mask_path, 1)
        resized_mask = cv2.resize(dynamic_content_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(file_path, cv2.bitwise_and(image, resized_mask))
    except Exception as ex:
        error = "{}".format(ex)
        logging.exception(ex)

    return error


def process_screenshot_iterations_consistency(image_array, file_path, min_occurences=12, inferior_theshold=15,
                                              superior_threshold=25):
    """ Get most average image, with a variance mask applied, intended to exclude dynamic content

    Args:
        image_array: array of binary image data of same size
        file_path: list specifying the width for the screenshots that should be catpured
        min_occurences: minimum occurence of over-threshold value before flooding initiated
        inferior_theshold: inferior pixel treshold value
        superior_threshold: superior pixel treshold value
    Returns:
        error
    """
    start_timer = time.perf_counter()
    error = None
    assert superior_threshold >= inferior_theshold, "Superior variance threshold must be grater than inferior one"
    dynamic_content_mask_path = file_path.replace(".png", DYNAMIC_CONTENT_SUFFIX)

    try:
        num_images = len(image_array)
        image_array_read = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), 1) for img in image_array]
        img_height, img_width, channels = image_array_read[0].shape
        img_height_downscaled = int(img_height / IMAGE_RESIZE_RATIO)
        img_width_downscaled = int(img_width / IMAGE_RESIZE_RATIO)

        image_grayscales = [cv2.cvtColor(
            cv2.resize(img, (img_width_downscaled, img_height_downscaled), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2GRAY) for img in image_array_read]

        variance_matrix = zeros((img_height_downscaled, img_width_downscaled))
        dif_median_value = zeros(num_images)

        for i in range(img_height_downscaled):
            high_count = 0
            low_count = 0
            for j in range(img_width_downscaled):
                pixels = [img[i][j] for img in image_grayscales]
                variance_norm = int(np.var(pixels) / num_images)
                median = np.median(pixels)
                dif_median_value += abs(pixels - median)

                variance_matrix[i][j], high_count, low_count = process_variance_threshold(variance_norm, high_count,
                                                                                          low_count, min_occurences,
                                                                                          inferior_theshold,
                                                                                          superior_threshold)

        logging.debug("Calculated image array variance in {:.1f}s ".format(time.perf_counter() - start_timer))

        kern_size = int(img_width_downscaled / MEDIAN_KERNEL_WIDTH_RATIO)
        dynamic_content_mask = median_filter(variance_matrix, size=kern_size)
        median_image = image_array_read[np.argmin(dif_median_value)]
        median_image_height, median_image_width, _ = median_image.shape
        dynamic_content_mask = cv2.cvtColor(cv2.resize(dynamic_content_mask, (median_image_width, median_image_height),
                                          interpolation=cv2.INTER_NEAREST).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Save mask for upgrade masking
        cv2.imwrite(dynamic_content_mask_path, dynamic_content_mask)

        # Mask and save image
        cv2.imwrite(file_path, cv2.bitwise_and(median_image, dynamic_content_mask))
    except Exception as ex:
        error = "{}".format(ex)
        logging.exception(ex)

    logging.debug("Baseline variance done int {:.1f}s ".format(time.perf_counter() - start_timer))

    return error



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-30s] %(message)s")

