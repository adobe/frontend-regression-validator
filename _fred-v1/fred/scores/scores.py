import numpy as np
from utils.utils import error_score, cost, intersection, union
from config.constants import STEP

CHANNELS = sorted(['images', 'textblock', 'section', 'buttons', 'forms'])


class Scores(object):
    @staticmethod
    def diff_mask_divergence(mask1, mask2, mask_matches=[]):
        score_dict = {'images': -1, 'textblock': -1, 'section': -1, 'buttons': -1, 'forms': -1, 'overall': -1}
        for ch in range(len(CHANNELS)):
            if len(mask_matches) > 0:
                curr_mask_matches = mask_matches[ch]
                curr_mask_score = 0
                for match in curr_mask_matches:
                    mask1_ch = mask1[match[0] * STEP: (match[0] + 1) * STEP, :, ch]
                    mask2_ch = mask2[match[1] * STEP: (match[1] + 1) * STEP, :, ch]
                    curr_mask_score += cost(mask1_ch, mask2_ch)
                ch_name = CHANNELS[ch]
                score_dict[ch_name] = round(curr_mask_score / len(curr_mask_matches), ndigits=4)
            else:
                mask1_ch = np.array(mask1[:, :, ch], dtype=np.bool)
                mask2_ch = np.array(mask2[:, :, ch], dtype=np.bool)
                ch_name = CHANNELS[ch]

                overlap = mask1_ch * mask2_ch
                union = mask1_ch + mask2_ch

                overlap = np.count_nonzero(overlap)
                union = np.count_nonzero(union)
                IOU = 1.0
                if union > 0:
                    IOU = overlap / float(union)
                score_dict[ch_name] = round(1 - IOU, ndigits=4)

        score_dict['overall'] = round(float((score_dict['images'] + score_dict['textblock']) / 2.0), ndigits=4)
        return score_dict

    @staticmethod
    def diff_pixelwise_divergence(im1, im2, mask1, mask2, mask_matches=[]):
        score_dict = {'images': -1, 'textblock': -1, 'section': -1, 'buttons': -1, 'forms': -1, 'overall': -1}
        for ch in range(len(CHANNELS)):
            if len(mask_matches) > 0:
                curr_mask_matches = mask_matches[ch]
                curr_mask_score = 0
                for match in curr_mask_matches:

                    mask1_ch = np.array(mask1[match[0] * STEP: (match[0] + 1) * STEP, :, ch], dtype=np.bool)
                    mask2_ch = np.array(mask2[match[1] * STEP: (match[1] + 1) * STEP, :, ch], dtype=np.bool)

                    im1_ch = im1[match[0] * STEP: (match[0] + 1) * STEP, :]
                    im2_ch = im2[match[1] * STEP: (match[1] + 1) * STEP, :]

                    union = mask1_ch + mask2_ch

                    if np.count_nonzero(union) > 0:
                        white = np.where(union == True)

                        diff_im1 = im1_ch[white]
                        diff_im2 = im2_ch[white]

                        mse_diff = error_score(diff_im1, diff_im2, len(white[0]))
                    else:
                        mse_diff = 0
                    curr_mask_score += mse_diff

                ch_name = CHANNELS[ch]
                score_dict[ch_name] = np.round(curr_mask_score / len(curr_mask_matches), decimals=4)
            else:
                mask1_ch = np.array(mask1[:, :, ch], dtype=np.bool)
                mask2_ch = np.array(mask2[:, :, ch], dtype=np.bool)
                union = mask1_ch + mask2_ch
                if np.count_nonzero(union) > 0:
                    white = np.where(union == True)

                    diff_im1 = im1[white]
                    diff_im2 = im2[white]

                    mse_diff = error_score(diff_im1, diff_im2, len(white[0]))

                    ch_name = CHANNELS[ch]
                    score_dict[ch_name] = np.round(mse_diff, decimals=4)
                else:
                    ch_name = CHANNELS[ch]
                    score_dict[ch_name] = 0
        score_dict['overall'] = round(float((score_dict['images'] + score_dict['textblock']) / 2.0), ndigits=4)
        return score_dict

    @staticmethod
    def logs_divergence(baseline_logs, updated_logs):
        log_intersection = len(intersection(baseline_logs, updated_logs))
        log_union = len(union(baseline_logs, updated_logs))
        IOU = 1.0
        if log_union > 0:
            IOU = log_intersection / float(log_union)
        return 1 - IOU
