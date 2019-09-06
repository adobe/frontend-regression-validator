import argparse
from inference.predict import predict
from scores.scores import Scores
import os
import json
from utils.utils import eprint, load_image_helper, save_masks, match_images
from config.constants import STEP
from process.process import LogProcessor


def work(baseline_dir, updated_dir, prefix):
    baseline_dir = os.path.join("./tmp", baseline_dir)
    updated_dir = os.path.join("./tmp", updated_dir)
    images = sorted([image for image in os.listdir(baseline_dir) if image.endswith('.png')])
    scores_dict = {}

    for i, image in enumerate(images):
        mask_matches = []
        baseline_image_path = os.path.join(baseline_dir, image)
        updated_image_path = os.path.join(updated_dir, image)
        baseline_image = load_image_helper(baseline_image_path)
        updated_image = load_image_helper(updated_image_path)

        eprint('[LOG] Making prediction for baseline - {}, image - {}'.format(prefix, image))
        baseline_image_mask = predict(baseline_image_path)
        eprint('[LOG] Saving masks for baseline - {}'.format(prefix))
        save_masks(baseline_dir, image, baseline_image_mask, baseline_image)
        eprint('[LOG] Making prediction for updated - {}, image - {}'.format(prefix, image))
        updated_image_mask = predict(updated_image_path)
        eprint('[LOG] Saving masks for updated - {}'.format(prefix))
        save_masks(updated_dir, image, updated_image_mask, updated_image)
        eprint('[LOG] Finished predictions')

        if baseline_image.shape != updated_image.shape:
            eprint('[LOG] Images have different shapes. Using DP algo')
            for c in range(0, 5):
                mask_matches.append(match_images(baseline_image_mask[:, :, c], updated_image_mask[:, :, c], STEP))

        eprint('[LOG] Calculating mask divergence score for {}, image - {}'.format(prefix, image))
        mask_divergence_scores = Scores.diff_mask_divergence(baseline_image_mask // 255, updated_image_mask // 255,
                                                             mask_matches)

        eprint('[LOG] Calculating pixelwise divergence score for {}, image - {}'.format(prefix, image))

        pixelwise_divergence_scores = Scores.diff_pixelwise_divergence(baseline_image, updated_image,
                                                                       baseline_image_mask // 255,
                                                                       updated_image_mask // 255, mask_matches)
        baseline_js_log_file = os.path.join(baseline_dir, image.split('.')[0] + "_js_log.json")
        updated_js_log_file = os.path.join(updated_dir, image.split('.')[0] + "_js_log.json")
        baseline_network_log_file = os.path.join(baseline_dir, image.split('.')[0] + "_network_log.json")
        updated_network_log_file = os.path.join(updated_dir, image.split('.')[0] + "_network_log.json")

        log_processor = LogProcessor(baseline_js_log_file, updated_js_log_file, baseline_network_log_file,
                                     updated_network_log_file)
        result = log_processor.run()

        ui_risk_score = max(mask_divergence_scores['overall'], pixelwise_divergence_scores['overall'])

        scores_dict[i + 1] = {
            'ui_stats': {'mask_div': mask_divergence_scores, 'pixelwise_div': pixelwise_divergence_scores,
                         'risk_score': ui_risk_score},
            'js_stats': result['javascript'],
            'network_stats': result['network'],
            'risk_score': result['risk_score']}
    with open(os.path.join('./tmp', prefix + '_scores.json'), 'w') as f:
        json.dump(scores_dict, f, indent=2)
    eprint('[LOG] Saved scores dictionary for {}'.format(prefix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=str)
    parser.add_argument('--updated-dir', type=str)
    parser.add_argument('--prefix', type=str)

    args = parser.parse_args()

    work(args.baseline_dir, args.updated_dir, args.prefix)
