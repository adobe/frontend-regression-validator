import os, logging
from utils.utils import load_image_helper, save_masks, match_images
from utils.scores import diff_pixelwise_divergence, diff_mask_divergence
from ml.predict import predict, load_model

def filter_screenshots(files):
    def intTryParse(value):
        try:
            return int(value), True
        except ValueError:
            return value, False
    ss = []
    for img in files:
        if not img.endswith(".png") or "raw_" not in img:
            continue
        parts = img.split("_")
        if len(parts) != 3:
            continue
        _, is_int = intTryParse(parts[1])
        _, is_int2 = intTryParse(parts[2].replace(".png",""))
        if not is_int or not is_int2:
            continue
        ss.append(img)
    return ss

def ml_predict(job_id: str, model, device):
    error = ""
    baseline_dir = os.path.join("jobs", job_id, "baseline")
    updated_dir = os.path.join("jobs", job_id, "updated")
    baseline_pages = sorted([os.path.split(f.path)[1] for f in os.scandir(baseline_dir) if f.is_dir()])  # just the last dir
    updated_pages = sorted([os.path.split(f.path)[1] for f in os.scandir(updated_dir) if f.is_dir()])  # just the last dir
    images = []
    for page in baseline_pages:
        if page not in updated_pages:
            logging.warning("Pages do not match: {} not found in updated pages!".format(page))
            continue

        b_images = filter_screenshots(os.listdir(os.path.join(baseline_dir, page)))
        b_images = sorted([os.path.join(baseline_dir, page, image) for image in b_images])
        u_images = filter_screenshots(os.listdir(os.path.join(updated_dir, page)))
        u_images = sorted([os.path.join(updated_dir, page, image) for image in u_images])

        for b_image in b_images:
            b_image_file = os.path.split(b_image)[1]
            if os.path.join(updated_dir, page, b_image_file) not in u_images:
                raise Exception("Images do not match: {} not found in updated page {}".format(b_image_file, page))
            images.append((b_image, os.path.join(updated_dir, page, b_image_file))) # add a tuple of images
    logging.debug("ML Worker has to process {} image pairs.".format(len(images)))

    ml_analysis = {}

    for (baseline_image_path, updated_image_path) in images:
        baseline_dir, image = os.path.split(baseline_image_path)
        updated_dir, _ = os.path.split(updated_image_path)
        resolution = image.split("_")[1]
        _, page = os.path.split(baseline_dir)
        if not page in ml_analysis:
            ml_analysis[page] = {}
            ml_analysis[page]["ml_screenshot_analysis"] = {}
        if resolution not in ml_analysis[page]["ml_screenshot_analysis"]:
            ml_analysis[page]["ml_screenshot_analysis"][resolution] = {}

        logging.debug("Working on page {} for image {} ...".format(baseline_dir, image))
        mask_matches = []

        baseline_image = load_image_helper(baseline_image_path)
        updated_image = load_image_helper(updated_image_path)

        logging.debug('Predicting baseline image {} ...'.format(baseline_image_path))
        baseline_image_mask, error = predict(baseline_image_path, model, device)

        if error:
            return None, None, error
        save_masks(baseline_dir, image, baseline_image_mask, baseline_image)
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['textblock_baseline_file_path'] = os.path.join(baseline_dir, "textblock_" + image)
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['images_baseline_file_path'] = os.path.join(baseline_dir,"images_" + image)

        logging.debug('Predicting updated image {} ...'.format(updated_image_path))
        updated_image_mask, error = predict(updated_image_path, model, device)

        if error:
            return None, None, error
        save_masks(updated_dir, image, updated_image_mask, updated_image)
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['textblock_updated_file_path'] = os.path.join(updated_dir, "textblock_"+image)
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['images_updated_file_path'] = os.path.join(updated_dir, "images_" + image)

        logging.debug('Finished predictions.')

        if baseline_image.shape != updated_image.shape:
            logging.warning('Images have different shapes. Using DP algo.')
            for c in [0,1]:#range(0, 2):
                mask_matches.append(match_images(baseline_image_mask[:, :, c], updated_image_mask[:, :, c], 1))

        logging.debug('Calculating mask divergence score for image {}'.format(image))
        mask_divergence_scores = diff_mask_divergence(baseline_image_mask // 255, updated_image_mask // 255,
                                                             mask_matches)

        logging.debug('Calculating pixelwise divergence score for image {}'.format(image))

        pixelwise_divergence_scores = diff_pixelwise_divergence(baseline_image, updated_image,
                                                                       baseline_image_mask // 255,
                                                                       updated_image_mask // 255, mask_matches)

        ml_analysis[page]["ml_screenshot_analysis"][resolution]['mask'] = mask_divergence_scores
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['content'] = pixelwise_divergence_scores
        ml_analysis[page]["ml_screenshot_analysis"][resolution]['overall'] = max(mask_divergence_scores['overall'], pixelwise_divergence_scores['overall'])

    # agregate max score per page from multiple scores from each resolution
    max_divergences = []
    for page in ml_analysis:
        max_divergence = max([float(ml_analysis[page]["ml_screenshot_analysis"][res]['overall']) for res in ml_analysis[page]["ml_screenshot_analysis"]])
        max_divergences.append(max_divergence)

    if len(max_divergences)>0:
        max_divergence = max(max_divergences)
    else:
        max_divergence = 0.

    return max_divergence, ml_analysis, error


# """ debug
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-30s] %(message)s")  # "%(asctime)s:%(levelname)s:%(message)s")
    from pprint import pprint
    from ml.predict import load_model
    model, device, error = load_model()
    max_divergence, ml_analysis, error = ml_predict("2020-05-11__09-37-31__test_timing_no_ml-current", model, device)
    pprint(ml_analysis)
# """

