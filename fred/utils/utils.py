import sys
sys.path.append("..")
from fred.globals import jobs, jobs_lock, STATUS_CODES

from datetime import datetime
from PIL import Image
import numpy as np
import os
import sys, json, logging
from collections import defaultdict
import collections
from threading import Thread

def get_list_of_files(dirName, ext=""):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath, ext)
        else:
            allFiles.append(fullPath)

    if ext != "":
        allFiles = [x for x in allFiles if ext in x]

    return allFiles


def update_job_object (source, overrides):
    """
    This extends dict.update(new_dict) because the new_dict could have sub-dicts that will overwrite existing sub-dicts
    in dict.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = update_job_object(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def update_job_log(jobs, job_id):
    global jobs_lock
    try:
        if jobs is None:
            logging.error("update_job_log was called with None jobs object. job_id = {}".format(job_id))
    except Exception as e:
        return

    file = os.path.join("jobs", job_id, "log.json")
    if os.path.exists(file):
        jobs_lock.acquire()
        js = json.load(open(file, "r", encoding="utf8"))
        jobs_lock.release()
        if js is not None:
            js = update_job_object(js, jobs[job_id])
            js_to_write = js
        else:
            js_to_write = jobs[job_id]
    else:
        js_to_write = jobs[job_id]
    if js_to_write is not None:
        # automatically perform overall score computation
        if "report" in jobs[job_id]:
            if "input_data" in jobs[job_id]:
                if "network_divergence_weight" in jobs[job_id]["input_data"] and \
                        "visual_divergence_weight" in jobs[job_id]["input_data"] and \
                        "visual_divergence_ai_weight" in jobs[job_id]["input_data"]:
                    wa = jobs[job_id]["input_data"]["network_divergence_weight"]
                    wb = jobs[job_id]["input_data"]["visual_divergence_weight"]
                    wc = jobs[job_id]["input_data"]["visual_divergence_ai_weight"]
                    if "crawl_divergence" in jobs[job_id]["report"] and \
                        "raw_screenshot_divergence" in jobs[job_id]["report"]:
                        a = jobs[job_id]["report"]["crawl_divergence"]
                        b = jobs[job_id]["report"]["raw_screenshot_divergence"]
                        if jobs[job_id]["input_data"]["ml_enable"]:
                            if "ml_screenshot_divergence" in jobs[job_id]["report"]:
                                c = jobs[job_id]["report"]["ml_screenshot_divergence"]
                                js_to_write["report"]["overall_divergence"] = a * wa + b * wb + c * wc
                        else:
                            js_to_write["report"]["overall_divergence"] = a * wa + b * wb
                    elif "crawl_divergence" in jobs[job_id]["report"]:
                        js_to_write["report"]["overall_divergence"] = jobs[job_id]["report"]["crawl_divergence"]

        # wrtie object
        jobs_lock.acquire()
        json.dump(js_to_write, open(file, "w", encoding="utf8"), sort_keys=True, indent=4)
        jobs_lock.release()
    sys.stdout.flush()
    sys.stderr.flush()

def read_jobs_log():
    global jobs_lock
    jobs = {}
    try:
        jobs = {}
        logs = get_list_of_files("jobs", ext="log.json")
        jobs_lock.acquire()
        for log_file in logs:
            job_id = log_file.split("/")[1]
            jobs[job_id] = json.load(open(log_file, "r", encoding="utf8"))
    except Exception as e:
        logging.error("Failed to read logs, or log non-existent:"+str(e))
    finally:
        if jobs_lock.locked():
            jobs_lock.release()
    return jobs

def read_job_log(job_id):
    """
    Read a single log from jobs
    :param job_id: the id requested
    :return:
    """
    global jobs_lock
    job = None
    try:
        log_path = os.path.join("jobs", job_id, "log.json")
        if not os.path.exists(log_path):
            return None
        jobs_lock.acquire()
        job = json.load(open(log_path, "r", encoding="utf8"))
    except Exception as e:
        logging.error("Failed to read logs :" + str(e))
    finally:
        if jobs_lock.locked():
            jobs_lock.release()
    return job

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


#def eprint(*args, **kwargs):
#    print(*args, file=sys.stderr, **kwargs)


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
    """buttons = mask_np_arr[:, :, 0]
    forms = mask_np_arr[:, :, 1]
    images = mask_np_arr[:, :, 2]
    section = mask_np_arr[:, :, 3]
    textblock = mask_np_arr[:, :, 4]"""
    images = mask_np_arr[:, :, 0]
    textblock = mask_np_arr[:, :, 1]
    #Image.fromarray(preprocess_save_image(image_np_arr * buttons)).save(os.path.join(out_dir, "buttons_" + image_filename))
    #Image.fromarray(preprocess_save_image(image_np_arr * forms)).save(os.path.join(out_dir, "forms_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * images)).save(os.path.join(out_dir, "images_" + image_filename))
    #Image.fromarray(preprocess_save_image(image_np_arr * section)).save(os.path.join(out_dir, "section_" + image_filename))
    Image.fromarray(preprocess_save_image(image_np_arr * textblock)).save(os.path.join(out_dir, "textblock_" + image_filename))


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

class TimeoutThread(Thread):
    def __init__(self, event, check_every=30):
        Thread.__init__(self)
        self.stopped = event

        from globals import TIMEOUT_OVERALL_SEC, TIMEOUT_AI_SEC, TIMEOUT_CRAWL_SEC
        self.crawl_timeout = TIMEOUT_CRAWL_SEC
        self.ml_timeout = TIMEOUT_AI_SEC
        self.overall_timeout = TIMEOUT_OVERALL_SEC
        self.check_every = check_every

    def run(self):
        from datetime import datetime
        while not self.stopped.wait(self.check_every):
            # call a function
            tjobs = read_jobs_log()
            for job_id in tjobs:
                job = tjobs[job_id]

                # eval ml timeout
                if job["status"] == STATUS_CODES[6]: # running ML
                    start_time_str = job["stats"]["ml_started_at"]
                    if start_time_str:
                        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
                        #print(start_time)
                        diff = (datetime.now() - start_time).total_seconds()
                        #print (diff)
                        if diff > self.ml_timeout: # set status to done, and set error to timeout
                            logging.error("TimeoutThread found a ML job with id [{}] that was started and is {:.0f}s old. Closing it with a timeout error.".format(job_id, diff))
                            tjobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                            tjobs[job_id]["error"] = "Timed out running ML, after {:.0f} seconds.".format(diff)
                            tjobs[job_id]["status"] = STATUS_CODES[8]
                            update_job_log(tjobs, job_id)

                # eval ml timeout
                if job["status"] == STATUS_CODES[2]: # crawling
                    start_time_str = job["stats"]["cr_started_at"]
                    if start_time_str:
                        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
                        #print(start_time)
                        diff = (datetime.now() - start_time).total_seconds()
                        #print (diff)
                        if diff > self.crawl_timeout: # set status to done, and set error to timeout
                            logging.error("TimeoutThread found a crawling job with id [{}] that was started and is {:.0f}s old. Closing it with a timeout error.".format(job_id, diff))
                            tjobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                            tjobs[job_id]["error"] = "Timed out crawling, after {:.0f} seconds.".format(diff)
                            tjobs[job_id]["status"] = STATUS_CODES[8]
                            update_job_log(tjobs, job_id)

                # eval overall timeout
                if job["stats"]["finished_at"].strip() == "":
                    start_time_str = job["stats"]["queued_at"]
                    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
                    diff = (datetime.now() - start_time).total_seconds()
                    if diff > self.overall_timeout:  # set status to done, and set error to timeout
                        logging.error(
                            "TimeoutThread found a job with id [{}] that was queued and is {:.0f}s old. Closing it with a timeout error.".format(
                                job_id, diff))
                        tjobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                        tjobs[job_id]["error"] = "Timed out running job, after {:.0f} seconds.".format(diff)
                        tjobs[job_id]["status"] = STATUS_CODES[8]
                        update_job_log(tjobs, job_id)
