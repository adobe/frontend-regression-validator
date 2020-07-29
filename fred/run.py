import sys
sys.path.append('../')

from globals import jobs, jobs_lock, timeout_timer_lock, NUM_CRAWLING_THREADS, STATUS_CODES, CRAWLER_LOCAL_ADDRESS, MAX_MEGAPIXELS, TIMEOUT_OVERALL_SEC, TIMEOUT_AI_SEC, TIMEOUT_CRAWL_SEC
from fred import app

from crawl import crawl_job
from predict import ml_predict
from crawl_analysis import crawl_analysis
from image_analysis import raw_screenshots_analysis

from flask_restful import Api, Resource
from flask_cors import CORS

from flask import send_from_directory
import sys, uuid, logging, time, os, requests, argparse, json, subprocess
from flask import request
from queue import Queue
from datetime import datetime
from threading import Thread, Lock, Event
from globals import NUM_CRAWLING_THREADS, URL_SLASH_REPLACEMENT_STR

from fred.utils.utils import read_jobs_log, read_job_log, update_job_log, get_list_of_files, update_job_object, TimeoutThread

DEFAULT_RES_WIDTHS = [1024]

def normalize_url_name(url):
    return url.replace(URL_SLASH_REPLACEMENT_STR, "/")


@app.route('/static/<path:path>', methods=['GET', 'POST'])
def send_js(path):
    return send_from_directory('frontend', path)

def new_ml_job_handler(threadID, q):
    """This is the worker thread function. It processes items in the queue one after another.  These daemon threads
    go into an infinite loop, and only exit when the main thread ends. """
    global jobs
    jobs = read_jobs_log()
    while True:
        logging.debug("ML thread #{} waiting for new jobs ...".format(threadID))

        data = q.get()  # this blocks until it gets a new job

        try:
            job_id = data["job_id"]
            crawler_address = data["crawler_address"]
            logging.info("ML thread #{} started a new job with id = {}".format(threadID, job_id))
            if jobs is None: # for some reason known only to others not to the programmer, jobs is None. FTW.
                jobs = {}
            if job_id not in jobs:  # if the ML component is on another machine, the job_id won't exist so we create it
                jobs[job_id] = {}
            if "stats" not in jobs[job_id]:
                jobs[job_id]["stats"] = {}
            if "pages" not in jobs[job_id]:
                jobs[job_id]["pages"] = {}

            # load ML model once, and only if anybody calls the ML handler
            from ml.predict import load_model
            model, device, error = load_model()
            if error:
                msg = "ML thread #{} with job id {} failed to load model with error: [{}]".format(threadID, job_id, error)
                logging.error(msg)
                jobs[job_id]["error"] = msg
                jobs[job_id]["status"] = STATUS_CODES[8]  # Done
                jobs[job_id]["stats"]["ml_finished_at"] = str(datetime.now())
                logging.info("ML thread #{} finished job id {}.".format(threadID, job_id))
                jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                update_job_log(jobs, job_id)
                q.task_done()
                continue

            # call ML predict
            logging.debug("ML thread #{} running ML for job {} ...".format(threadID, job_id))
            jobs[job_id]["status"] = STATUS_CODES[6]  # Running ML
            update_job_log(jobs, job_id)

            max_divergence, page_report, error = ml_predict(job_id, model, device)
            if error:
                msg = "ML thread #{} with job id {} failed prediction with error: [{}]".format(threadID, job_id, error)
                logging.error(msg)
                jobs[job_id]["error"] = msg
                jobs[job_id]["status"] = STATUS_CODES[8]  # Done
                jobs[job_id]["stats"]["ml_finished_at"] = str(datetime.now())
                logging.info("ML thread #{} finished job id {}.".format(threadID, job_id))
                jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                update_job_log(jobs, job_id)
                q.task_done()
                continue

            # update data
            for page in page_report:
                if not page in jobs[job_id]["pages"]:
                    jobs[job_id]["pages"][page] = {}
                jobs[job_id]["pages"][page]["ml_screenshot_analysis"] = page_report[page]["ml_screenshot_analysis"]
            if "report" not in jobs[job_id]:
                jobs[job_id]["report"] = {}
            jobs[job_id]["report"]["ml_screenshot_divergence"] = max_divergence
            update_job_log(jobs, job_id)

            # done, update status and finish task
            logging.info("ML thread #{} finished job {}.".format(threadID, job_id))
            jobs[job_id]["status"] = STATUS_CODES[8]  # Done
            jobs[job_id]["stats"]["ml_finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)

            # report status and screenshots back to original sender
            files = {}
            # this needs to change to point to the actual raw screenshots. jobs->job_id
            img_files = get_list_of_files(os.path.join("jobs", str(job_id)), "_raw_")
            img_files_handler = []
            #logging.warning(img_files)

            for img_file in img_files:
                logging.debug("Packing [{}] to send ...".format(img_file))

                img_file_handler = open(img_file, "rb")
                img_files_handler.append(img_file_handler)  # keep to close handler after POST
                key = img_file[img_file.index("jobs"):]  # send path starting from jobs
                kk = key.split("/")
                key = os.path.join(kk[0], job_id, kk[2], kk[3], kk[4])
                files[key] = (os.path.basename(img_file), img_file_handler, 'application/octet-stream')

                """
                request is : key=jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png
                """

            payload = json.dumps(jobs[job_id]).encode('utf-8')
            files['json'] = ('json', payload, 'application/octet-stream')

            logging.debug("ML thread #{} with job id {} stated sending images ...".format(threadID, job_id))
            try:
                r = requests.post("http://"+crawler_address+"/api/mlreport", files=files)
                logging.debug("ML thread #{} with job id {} finished sending images ...".format(threadID, job_id))

                if r.status_code != 200:
                    # ML sending screenshots, but everything else completed successfully
                    logging.error("ML thread #{} with job id {} failed sending images: [{}]".format(threadID, job_id,
                                                                                                              r.text.strip()))
                    jobs[job_id]["error"] = "Error sending files back to crawler component: [" + r.text.strip() + "]"
                    logging.info("ML thread #{} finished job id {}.".format(threadID, job_id))
            except Exception as ex:
                jobs[job_id]["error"] = "Error connecting back to crawler component: [" + str(ex) + "]"
                logging.info("ML thread #{} finished job id {}.".format(threadID, job_id))
            finally:
                for img_file_handler in img_files_handler:  # close handles
                    img_file_handler.close()

            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)
            q.task_done()
        except Exception as ex:
            logging.error("ML thread #{} with job id {} failed with exception: {}".format(threadID, job_id, str(ex)))
            q.task_done()

def new_crawling_job_handler(threadID, q):
    """This is the worker thread function. It processes items in the queue one after another.  These daemon threads
    go into an infinite loop, and only exit when the main thread ends. """
    while True:
        logging.debug("Crawler thread #{} waiting for new jobs ...".format(threadID))
        request_data = q.get()  # this blocks until it gets a new job

        # start work
        job_id = request_data["job_id"]

        # TODO all this is in job[jobid][input data], skip definition here
        baseline_url = request_data["baseline_url"]
        updated_url = request_data["updated_url"]
        max_depth = request_data["max_depth"]
        max_urls = request_data["max_urls"]
        prefix = request_data["prefix"]
        ml_enable = request_data["ml_enable"]
        ml_address = request_data["ml_address"]
        auth_baseline_username = request_data["auth_baseline_username"]
        auth_baseline_password = request_data["auth_baseline_password"]
        auth_updated_username = request_data["auth_updated_username"]
        auth_updated_password = request_data["auth_updated_password"]

        logging.info("Crawler thread #{} just got a new job with id {} ...".format(threadID, job_id))

        jobs[job_id]["status"] = STATUS_CODES[2] # Crawling
        jobs[job_id]["stats"]["cr_started_at"] = str(datetime.now())
        update_job_log(jobs, job_id)

        # this section will call crawler, the crawler itself will further save logs

        """
        the crawl_job function in crawl.py will:
            1. save network diags in jobs[job_id]["baseline"/"updated"]["link"]["crawl"]["any param here"] = param value
            2. save screenshots in several resolutions in jobs/id/baseline(or updated)/link/raw_(res)_(counter).png like 512, 1024 and 1920, counter = 00, 01, .. 10 (to be sortable)
        """
        crawler_num_threads = NUM_CRAWLING_THREADS
        try:
            error = crawl_job(job_id, baseline_url, updated_url, max_depth, max_urls, crawler_num_threads,
                                                jobs[job_id]["input_data"]["requested_resolutions"], auth_baseline_username,
                  auth_baseline_password,
                  auth_updated_username, auth_updated_password )
        except Exception as ex:
            error = str(ex)

        # check fail status
        if error:
            logging.error("Crawler thread #{} with job id {} failed crawling with error(s): [{}]".format(threadID, job_id,
                                                                                                  str(error)))
            jobs[job_id]["error"] = "Error crawling: [" + error.strip() + "]"
            jobs[job_id]["status"] = STATUS_CODES[8]  # Done
            jobs[job_id]["stats"]["cr_finished_at"] = str(datetime.now())
            logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)
            q.task_done()
            continue

        # start crawl analysis
        jobs[job_id]["status"] = STATUS_CODES[4]  # Analyzing crawl
        update_job_log(jobs, job_id)
        report, error = crawl_analysis(job_id)
        jobs[job_id]["report"] = {}
        jobs[job_id]["report"]["crawl_divergence"] = report["crawl_divergence"]
        jobs[job_id]["report"]["unique_baseline_pages"] = report["unique_baseline_pages"]
        jobs[job_id]["report"]["unique_updated_pages"] = report["unique_updated_pages"]
        jobs[job_id]["pages"] = {}
        for p in report["pages"]:
            jobs[job_id]["pages"][p] = report["pages"][p]
        if error or len(jobs[job_id]["pages"])==0:
            if len(jobs[job_id]["pages"])==0:
                error = "No pages have been crawled!"
            logging.error("Crawler thread #{} with job id {} failed crawling analysis with error(s): [{}]".format(threadID, job_id,
                                                                                                  str(error)))
            jobs[job_id]["error"] = "Error analyzing crawl: [" + error.strip() + "]"
            jobs[job_id]["status"] = STATUS_CODES[8]  # Done
            jobs[job_id]["stats"]["cr_finished_at"] = str(datetime.now())
            logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)
            q.task_done()
            continue

        # start image analysis
        report, error = raw_screenshots_analysis(jobs, job_id)
        for k in report:
            if not isinstance(report[k], dict): # save at top level just the strings
                jobs[job_id]["report"][k] = report[k]
        for p in report["pages"]:
            for k in report["pages"][p]:
                jobs[job_id]["pages"][p][k] = report["pages"][p][k]

        jobs[job_id]["stats"]["cr_finished_at"] = str(datetime.now())
        update_job_log(jobs, job_id)
        if error:
            logging.error("Crawler thread #{} with job id {} failed image analysis with error(s): [{}]".format(threadID, job_id,
                                                                                                  str(error)))
            jobs[job_id]["error"] = "Error analyzing images: [" + error.strip() + "]"
            jobs[job_id]["status"] = STATUS_CODES[8]  # Done
            logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)
            q.task_done()
            continue

        # determine if we need to do ML
        if not jobs[job_id]["input_data"]["ml_enable"]: # no ML job, should be = to ml_enable
            logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
            jobs[job_id]["status"] = STATUS_CODES[8] # Done, job finished successfully
            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            update_job_log(jobs, job_id)
            q.task_done()
            continue

        # we're running the ML job now, stat by sending the files to the ML component
        jobs[job_id]["status"] = STATUS_CODES[5] # Sending screenshots to ML component
        update_job_log(jobs, job_id)
        files = {}
        all_img_files = get_list_of_files(os.path.join("jobs", str(job_id)), "raw_") # this needs to change to point to the actual raw screenshots. jobs->job_id
        img_files = [] # filter out other images
        for img in all_img_files:
            parts = img[img.rfind("/")+1:].split("_")
            if len(parts) == 3 and img.endswith(".png"):
                img_files.append(img)
        img_files_handler = []

        for img_file in img_files:
            logging.debug("Packing [{}] to send ...".format(img_file))
            img_file_handler = open(img_file, "rb")
            img_files_handler.append(img_file_handler)  # keep to close handler after POST
            key = img_file[img_file.index("jobs"):]  # send path starting from jobs
            files[key] = (os.path.basename(img_file), img_file_handler, 'application/octet-stream')

        """
        request is : key=jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png
        """

        """
        imga = open("/Users/sdumitre/work/frontend-regression-validator/fred/jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png","rb")
        name_img = "raw_512_00.png"
        pack =  (name_img, imga, 'application/octet-stream')#'multipart/form-data', {'Expires': '0'})
        data["baseline/l1/raw_512_00.png"] = pack

        imgb = open("/Users/sdumitre/work/frontend-regression-validator/fred/jobs/2020-02-27__10-10-41__a8166d20/updated/l1/raw_512_00.png","rb")
        name_img = "raw_512_00.png"
        pack = (name_img, imgb, 'application/octet-stream')#'multipart/form-data', {'Expires': '0'})
        data["updated/l1/raw_512_00.png"] = pack
        headers = {'Content-Type': 'image/jpeg'}

        for i in range(1):
            img = open(
                "/Users/sdumitre/work/frontend-regression-validator/fred/jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png",
                "rb")
            name_img = str(i)+"raw_512_00.png"
            pack = (name_img, img, 'application/octet-stream')  # 'multipart/form-data', {'Expires': '0'})
            data[name_img] = pack
        """

        # add data payload to request
        global CRAWLER_LOCAL_ADDRESS
        payload = json.dumps({"crawler_address": CRAWLER_LOCAL_ADDRESS}).encode('utf-8')
        files['json'] = ('json', payload, 'application/octet-stream')

        logging.debug("Crawler thread #{} with job id {} stated sending screenshots ...".format(threadID, job_id))
        try:
            r = requests.post("http://"+ml_address+"/api/mlrequest", files=files)

            for img_file_handler in img_files_handler:  # close handles
                img_file_handler.close()

            logging.debug("Crawler thread #{} with job id {} finished sending screenshots ...".format(threadID, job_id))

            if r.status_code != 200:
                # ML sending screenshots, but everything else completed successfully
                logging.error("Crawler thread #{} with job id {} failed sending screenshots: [{}]".format(threadID, job_id, r.text.strip()))
                jobs[job_id]["error"] = "Error sending files to ML component : [" + r.text.strip() + "]"
                jobs[job_id]["status"] = STATUS_CODES[8] # Done
                jobs[job_id]["stats"]["cr_finished_at"] = str(datetime.now())
                jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
                logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
                update_job_log(jobs, job_id)
                q.task_done()
                continue
        except Exception as ex: # can fail if port is not accessible or address is wrong
            logging.error("Crawler thread #{} with job id {} failed sending screenshots: [{}]".format(threadID, job_id, str(ex)))
            jobs[job_id]["error"] = "Error connecting to ML component : [" + str(ex) + "]"
            jobs[job_id]["status"] = STATUS_CODES[8] # Done
            jobs[job_id]["stats"]["cr_finished_at"] = str(datetime.now())
            jobs[job_id]["stats"]["finished_at"] = str(datetime.now())
            logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
            update_job_log(jobs, job_id)
            q.task_done()
            continue

        # set status as ML is running and release this worker
        logging.info("Crawler thread #{} finished job id {}.".format(threadID, job_id))
        jobs[job_id]["status"] = STATUS_CODES[6]  # Running ML
        jobs[job_id]["stats"]["ml_started_at"] = str(datetime.now())
        update_job_log(jobs, job_id)
        q.task_done()


class ApiVerify(Resource):
    def post(self):
        # explicitly define data in dict, otherwise we might get missing values
        global CRAWLER_LOCAL_ADDRESS
        json_data = {}

        json_data["baseline_url"] = request.json['baseline_url'].strip()
        json_data["updated_url"] = request.json['updated_url'].strip()
        json_data["max_depth"] = request.json.get('max_depth', 10)
        json_data["max_urls"] = request.json.get('max_urls', 10)
        json_data["prefix"] = request.json.get('prefix', '').strip()
        json_data["auth_baseline_username"] = request.json.get('auth_baseline_username', '')
        json_data["auth_baseline_password"] = request.json.get('auth_baseline_password', '')
        json_data["auth_updated_username"] = request.json.get('auth_updated_username', '')
        json_data["auth_updated_password"] = request.json.get('auth_updated_password', '')
        json_data["ml_enable"] = request.json.get('ml_enable', False)
        json_data["ml_address"] = request.json.get('ml_address', '0.0.0.0:5000').replace("http:","").replace("/","").replace("https://","")
        json_data["requested_resolutions"] = request.json.get('resolutions', '512,1024')
        json_data["requested_score_weights"] = request.json.get('score_weights', '0.1,0.4,0.5')
        json_data["requested_score_epsilon"] = request.json.get('score_epsilon', '10,0,0')
        json_data["crawler_address"] = CRAWLER_LOCAL_ADDRESS
        # TODO add weights in visual UI and append them here

        # check for parameter correctness
        if json_data["baseline_url"] == "" or json_data["updated_url"] == "":
            return {'error': 'URL is empty!'}, 500
        if not json_data["baseline_url"].startswith("http") or not json_data["updated_url"].startswith("http"):
            return {'error': 'URL must start with http*'}, 500
        try:
            json_data["max_depth"] = int(json_data["max_depth"])
        except ValueError:
            return {'error': 'max_depth must be an integer!'}, 500
        try:
            json_data["max_urls"] = int(json_data["max_urls"])
        except ValueError:
            return {'error': 'max_urls must be an integer!'}, 500
        try:
            json_data["ml_enable"] = bool(json_data["ml_enable"])
        except ValueError:
            return {'error': 'ml_enable must be a boolean!'}, 500

        # parse resolutions
        try:
            res = []
            parts = str(json_data["requested_resolutions"]).split(",")
            for part in parts:
                res.append(int(part.strip()))
            json_data["requested_resolutions"] = res
        except:
            json_data["requested_resolutions"] = DEFAULT_RES_WIDTHS

        # parse epsilon
        try:
            parts = str(json_data["requested_score_epsilon"]).split(",")
            if len(parts) == 3:
                json_data["network_divergence_epsilon_threshold"] = float(parts[0])
                json_data["visual_divergence_epsilon_threshold"] = float(parts[1])
                json_data["visual_divergence_ai_epsilon_threshold"] = float(parts[2])
            else:
                json_data["network_divergence_epsilon_threshold"] = 10.
                json_data["visual_divergence_epsilon_threshold"] = 0.
                json_data["visual_divergence_ai_epsilon_threshold"] = 0.
        except:  # default
            json_data["network_divergence_epsilon_threshold"] = 10.
            json_data["visual_divergence_epsilon_threshold"] = 0.
            json_data["visual_divergence_ai_epsilon_threshold"] = 0.

        # parse score weights
        try:
            parts = str(json_data["requested_score_weights"]).split(",")
            if len(parts) == 3:
                json_data["network_divergence_weight"] = float(parts[0])
                json_data["visual_divergence_weight"] = float(parts[1])
                json_data["visual_divergence_ai_weight"] = float(parts[2])
            else:
                json_data["network_divergence_weight"] = 0.1
                json_data["visual_divergence_weight"] = 0.4
                json_data["visual_divergence_ai_weight"] = 0.5
        except: # add default weights
            json_data["network_divergence_weight"] = 0.1
            json_data["visual_divergence_weight"] = 0.4
            json_data["visual_divergence_ai_weight"] = 0.5
        finally:
            # normalize weights (so they sum to 1)
            a = float(json_data["network_divergence_weight"])
            b = float(json_data["visual_divergence_weight"])
            c = float(json_data["visual_divergence_ai_weight"])
            json_data["network_divergence_weight"] = a / (a+b+c)
            json_data["visual_divergence_weight"] = b / (a+b+c)
            json_data["visual_divergence_ai_weight"] = c / (a+b+c)

        # adjust weights based on ml_enable to sum up to one
        if json_data["ml_enable"] == False:
            json_data["network_divergence_weight"] = a / (a + b)
            json_data["visual_divergence_weight"] = b / (a + b)
            json_data["visual_divergence_ai_weight"] = 0.

        if json_data["prefix"].strip() == "":
            json_data["prefix"] = str(uuid.uuid4().hex)[:8]

        logging.info("New job in the queue: {}".format(json_data))

        # create unique ID and dump status
        job_id = datetime.now().strftime("%Y-%m-%d__%H-%M-%S__") + json_data["prefix"]
        json_data["job_id"] = job_id

        # put new job in queue
        jobs[job_id] = {}
        jobs[job_id]["status"] = STATUS_CODES[1] # Scheduled
        jobs[job_id]["error"] = ""
        jobs[job_id]["input_data"] = json_data
        jobs[job_id]["stats"] = {}
        jobs[job_id]["stats"]["queued_at"] = str(datetime.now())
        jobs[job_id]["stats"]["cr_started_at"] = ""
        jobs[job_id]["stats"]["cr_finished_at"] = ""
        jobs[job_id]["stats"]["ml_started_at"] = ""
        jobs[job_id]["stats"]["ml_finished_at"] = ""
        jobs[job_id]["stats"]["finished_at"] = ""

        os.makedirs(os.path.join("jobs", job_id), exist_ok=True)
        update_job_log(jobs, job_id)

        # put job in queue
        crawler_queue.put(json_data)

        # return id to user
        return {'id': job_id}, 200

class MLRequest(Resource):
    """
    This class is run when the crawler sends a number of screenshots to be analyzed by the ML component.
    We save the images and put a job in the queue. To respond back, the ml worker will post a MLReport
    to the server saying here are my results.
    """
    def post(self):
        logging.info("New MLRequest received ...")
        try:
            # save screenshots
            for k in request.files:
                if k == "json":
                    payload = json.loads(request.files["json"].read().decode('utf-8'))
                    crawler_address = payload["crawler_address"]
                    logging.debug("Requester has address [{}]".format(crawler_address))
                    continue
                else:
                    # key=jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png
                    # print(request.files[k]) is a <FileStorage: 'raw_512_00.png' ('application/octet-stream')>
                    job_id = k.split("/")[1]
                    filename = os.path.basename(k)
                    if os.path.exists(k):
                        logging.debug("MLRequest file [{}] already exists locally, skipping ...".format(k))
                    else:
                        logging.debug("MLRequest writing file [{}] ...".format(k))
                        os.makedirs(k.replace(filename, ""), exist_ok=True)  # make dir if it does not exist
                        picture = request.files.get(k)  # get file handler
                        picture.save(k)

            # put job in ML queue
            data = {}
            data["job_id"] = job_id
            data["crawler_address"] = crawler_address
            logging.debug("Put a new job in the ML queue : ({},{})".format(job_id, crawler_address))
            ml_queue.put(data)
            return {}, 200
        except Exception as e:
            return str(e), 500

class MLReport(Resource):
    """
    This class is run when the ML component finishes and reports back its status.
    It will send images and the status json with its analysis
    """
    def post(self):
        logging.info("New MLReport received ...")
        try:
            # save screenshots
            for k in request.files:
                if k == "json":
                    payload = json.loads(request.files["json"].read().decode('utf-8'))
                    continue
                else:
                    # key=jobs/2020-02-27__10-10-41__a8166d20/baseline/l1/raw_512_00.png
                    # print(request.files[k]) is a <FileStorage: 'raw_512_00.png' ('application/octet-stream')>
                    job_id = k.split("/")[1]
                    filename = os.path.basename(k)
                    if os.path.exists(k):
                        logging.debug("MLReport file [{}] already exists locally, skipping ...".format(k))
                    else:
                        logging.debug("MLReport writing file [{}] ...".format(k))
                        os.makedirs(k.replace(filename, ""), exist_ok=True)  # make dir if it does not exist
                        picture = request.files.get(k)  # get file handler
                        picture.save(k)

            # update jobs with payload (which is jobs[job_id] as computed by the ML component
            jobs[job_id] = update_job_object(jobs[job_id], payload)
            update_job_log(jobs, job_id)
            return {}, 200
        except Exception as e:
            return str(e), 500


class ApiViewJobs(Resource):
    def get(self):
        # explicitly define data in dict, otherwise we might get missing values
        logging.info("Request to view jobs received ...")

        #global jobs
        jobs = read_jobs_log()

        # return id to user
        return {'jobs': jobs}, 200
        response = app.response_class(
            response=json.dumps(jobs),
            status=200,
            mimetype='application/json'
        )
        #return response

class ApiResult(Resource):
    """
    Used when a user wants a report based on an job ID
    """
    def get(self):
        job_id = request.args.get('id')
        logging.info("ApiRequest received for id=[{}]".format(job_id))
        report = read_job_log(job_id)

        if not report:
            return {'Error': 'Report does not exist'}, 404
        else:
            pages = report.get("pages")
            if pages is not None:
                for p in list(pages):
                    pages[normalize_url_name(p)] = pages[p]
                    del pages[p]

            return report, 200

def check_chromedriver_path():
    # for safety, check symlink to jobs folder
    try:
        #os.system("cd frontend/ && ln -s ../jobs jobs >/dev/null 2>&1")
        os.makedirs("jobs", exist_ok=True)
        if not os.path.islink(os.path.join("frontend","jobs")):
            logging.info("'frontend/jobs' symlink does not exist, creating it automatically ...")
            os.symlink("../jobs", os.path.join("frontend","jobs"), target_is_directory=True)

        if os.environ.get('CHROMEDRIVER_PATH') is None:
            logging.warning("CHROMEDRIVER_PATH is not set. Attempting to set the path automatically... (if this instance is not for crawling, ignore this message")
            import platform
            plt = platform.system()
            if plt == "Linux":
                os.environ['CHROMEDRIVER_PATH'] = '/usr/lib/chromium-browser/chromedriver'
                logging.info("Detected system is Linux. Default path set at: {}".format(os.environ['CHROMEDRIVER_PATH']))
            elif plt == "Darwin":
                os.environ['CHROMEDRIVER_PATH'] = '/usr/local/bin/chromedriver'
                logging.info("Detected system is MacOS. Default path set at: {}".format(os.environ['CHROMEDRIVER_PATH']))
            else:
                logging.info("Unidentified system, not setting the path - you will not be able to perform crawling!")
    except Exception as ex:
        logging.error("{}".format(str(ex)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Address that FRED will listen on.')
    parser.add_argument('--port', type=int, default=5000, help='Port that FRED will listen on.')
    parser.add_argument('--external_address', type=str, help='External address of FRED that another component will send results back at. Useful for NAT situations when the crawler component is hidden from a ML component.')
    parser.add_argument('--external_port', type=int, help='External port of FRED that another component will send results back at.')
    parser.add_argument('--crawl_threads', type=int, default=2, help='Number of parallel crawling requests FRED will handle.')
    parser.add_argument('--ai_threads', type=int, default=1, help='Number of parallel A.I. threads FRED will handle. Recommended 1 or 2 maximum to not get GPU memory errors if running on a GPU.')
    parser.add_argument('--max_megapixels', type=float, default=20.0, help='Maximum megapixels to run raw screenshot analysis. Images will be clipped to this MP number.')
    parser.add_argument('--log_level', type=str, default="debug", help='Minimum log level. Available levels: debug, warning, info, error.')
    parser.add_argument('--timeout_overall', type=int, default=1800,
                        help='Maximum timeout in seconds to wait for a job to be finished, from scheduled to done. Set 0 to disable, default is 30 minutes (1800 seconds).')
    parser.add_argument('--timeout_ai', type=int, default=1200,
                        help='Maximum timeout in seconds to wait for the AI analysis to be finished. Set 0 to disable, default is 20 minutes (1200 seconds).')
    parser.add_argument('--timeout_crawl', type=int, default=1200,
                        help='Maximum timeout in seconds to wait for the AI analysis to be finished. Set 0 to disable, default is 20 minutes (1200 seconds).')

    args = parser.parse_args()
    logging.debug("Starting with params: {}".format(args))

    TIMEOUT_OVERALL_SEC = int(args.timeout_overall)
    TIMEOUT_AI_SEC = int(args.timeout_ai)
    TIMEOUT_CRAWL_SEC = int(args.timeout_crawl)
    MAX_MEGAPIXELS = float(args.max_megapixels)

    # set logging level
    if args.log_level == "warning":
        logging.getLogger().setLevel(logging.WARNING)
    elif args.log_level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif args.log_level == "error":
        logging.getLogger().setLevel(logging.ERROR)

    # check environment path and double check symlink
    check_chromedriver_path()

    if args.external_address is not None:
        CRAWLER_LOCAL_ADDRESS = args.external_address + ":" + str(args.external_port) if args.external_port is not None else str(args.port)
    else:
        CRAWLER_LOCAL_ADDRESS = "{}:{}".format(args.host, args.port)
    logging.info("External FRED address {}".format(CRAWLER_LOCAL_ADDRESS))

    crawler_queue = Queue()
    ml_queue = Queue()

    # create jobs folder
    os.makedirs("jobs", exist_ok=True)

    # start timeout thread
    stopFlag = Event()
    thread = TimeoutThread(stopFlag, check_every=30) # number of seconds
    thread.start() # this will stop the timer: stopFlag.set()

    # start threads that will consume new crawling requests
    for threadId in range(args.crawl_threads):
        worker = Thread(target=new_crawling_job_handler, args=(threadId, crawler_queue,))
        worker.setDaemon(True)
        worker.start()

    for threadId in range(args.ai_threads):
        worker = Thread(target=new_ml_job_handler, args=(threadId, ml_queue,))
        worker.setDaemon(True)
        worker.start()

    logging.info("Started {} crawling threads and {} ML threads.".format(args.crawl_threads, args.ai_threads))

    if app.config["DEBUG"]:
        @app.after_request
        def after_request(response):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
            response.headers["Expires"] = 0
            response.headers["Pragma"] = "no-cache"
            return response

    cors = CORS(app, resources={r"*": {"origins": "*"}})

    api = Api(app)
    api.add_resource(ApiVerify, "/api/verify")
    api.add_resource(ApiViewJobs, "/api/viewjobs")
    api.add_resource(ApiResult, "/api/result")
    api.add_resource(MLRequest, "/api/mlrequest")
    api.add_resource(MLReport, "/api/mlreport")

    app.run(host=args.host, port=int(args.port), debug=True, threaded=True, use_reloader=False)
