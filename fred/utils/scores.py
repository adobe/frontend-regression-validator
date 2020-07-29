import numpy as np
from utils.utils import error_score, cost


#CHANNELS = sorted(['images', 'textblock', 'section', 'buttons', 'forms'])
CHANNELS = sorted(['images', 'textblock'])
STEP = 1

def diff_mask_divergence(mask1, mask2, mask_matches=[]):
    score_dict = {'images': -1, 'textblock': -1} #, 'section': -1, 'buttons': -1, 'forms': -1, 'overall': -1}
    # masks are ndarray of (h, w, ch)
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

    score_dict['overall'] = round(float((100.*score_dict['images'] + 100.*score_dict['textblock']) / 2.0), ndigits=4)
    return score_dict

def diff_pixelwise_divergence(im1, im2, mask1, mask2, mask_matches=[]):
    score_dict = {'images': -1, 'textblock': -1} #, 'section': -1, 'buttons': -1, 'forms': -1, 'overall': -1}
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
    score_dict['overall'] = round(float((100.*score_dict['images'] + 100.*score_dict['textblock']) / 2.0), ndigits=4)
    return score_dict


def log_analysis(baseline_logs, updated_logs):
    b = ["{}".format(x) for x in baseline_logs]
    u = ["{}".format(x) for x in updated_logs]
    # logs are now a list of strings

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    def union(lst1, lst2):
        return list(set(lst1) | set(lst2))

    log_intersection = len(intersection(b, u))
    log_union = len(union(b, u))
    IOU = 1.0
    if log_union > 0:
        IOU = log_intersection / float(log_union)

    # list differences
    differences = {}
    differences["on_baseline_not_on_updated"] = []
    differences["on_updated_not_on_baseline"] = []
    for elem in b:
        if elem not in u:
            differences["on_baseline_not_on_updated"].append(elem)
    for elem in u:
        if elem not in b:
            differences["on_updated_not_on_baseline"].append(elem)
    return (1 - IOU)*100, differences
