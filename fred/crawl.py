import base64
import concurrent.futures
import json
import logging
import os
import sys
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from furl import furl
from globals import CRAWL_URL_RETRIES, PAGE_LOAD_TIMEOUT_SEC, NUM_CRAWLING_THREADS, URL_SLASH_REPLACEMENT_STR, \
    VARIANCE_NUM_ITERATIONS, VARIANCE_INTERATIONS_INTERVAL_SEC
from retry import retry
from selenium.common.exceptions import TimeoutException
from seleniumwire import webdriver
from urllib3.exceptions import MaxRetryError
import time
from image_analysis import raw_screenshot_analysis, process_screenshot_iterations_consistency, save_and_mask_image

BASELINE_DIR = "baseline"
UPDATED_DIR = "updated"

def get_chrome_driver(driver_path):
    """
    Instantiates a Chrome webdriver object

    Args:
        driver_path: path to the Chrome driver

    Returns:
        webdriver: returns a webdriver object
    """
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'normal'
    options.add_argument("--disable-logging")
    options.add_argument("--disable-login-animations")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument("--headless")
    options.add_argument("--hide-scrollbars")
    options.add_argument('--log-level 3')
    options.add_argument("--incognito")
    options.add_argument("--no-zygote")
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-infobars')
    driver = webdriver.Chrome(driver_path, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SEC)
    return driver


def chrome_take_full_screenshot(driver, viewport_width):
    """
    Takes full page screenshots at various width

    Args:
        driver: webdriver object
        viewport_width: viewport width

    Returns:
        list: list of image raw data for the taken screenshots
    """

    def send(cmd, params):
        resource = "/session/{}/chromium/send_command_and_get_result".format(driver.session_id)
        url = driver.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = driver.command_executor._request('POST', url, body)
        return response.get('value')

    def evaluate(script):
        response = send('Runtime.evaluate', {'returnByValue': True, 'expression': script})
        return response['result']['value']

    send('Emulation.clearDeviceMetricsOverride', {})
    metrics = evaluate(
        "({" +
        "width: " + str(viewport_width) + "," +
        "height: 0," +
        "deviceScaleFactor: window.devicePixelRatio || 1," +
        "mobile: typeof window.orientation !== 'undefined'" +
        "})")
    send('Emulation.setDeviceMetricsOverride', metrics)
    metrics = evaluate(
        "({" +
        "width: " + str(viewport_width) + "," +
        "height: Math.max(innerHeight, document.body.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight)|0," +
        "deviceScaleFactor: window.devicePixelRatio || 1," +
        "mobile: typeof window.orientation !== 'undefined'" +
        "})")
    send('Emulation.setDeviceMetricsOverride', metrics)
    screenshot = send('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
    return base64.b64decode(screenshot['data'])


@retry(TimeoutException, tries=CRAWL_URL_RETRIES, delay=1)
def get_with_retry(driver, url):
    driver.get(url)


class CrawlingContext:
    def __init__(self, driver_path, crawl_max=1, max_depth=1, allowed_domains=None, auth_username=None,
                 auth_password=None, crawled=None, depth=0, screenshot_tasks=[], executor=None, screenshot_res=None,
                 base_path=None, iterations=1, is_baseline=False):
        """Crawling context object

        Args:
            driver_path: Selenium webdriver object
            crawl_max: maximum number of links that should be returned
            max_depth: maximum depth for crawling (crawling is breadth first)
            allowed_domains: list of allowed domains. If empty all found URLs will be returned
            auth_username: username to be used for basic authentication
            auth_password: password to be used for basic authentication
            crawled: list of already crawled URLs
            depth: depth of the URL that will be scraped
            iterations: num iterations on page
            is_baseline: whether this is the baseline for the diff comparison
        """
        self.driver_path = driver_path
        self.crawl_max = crawl_max
        self.max_depth = max_depth
        self.allowed_domains = allowed_domains
        self.auth_username = auth_username
        self.auth_password = auth_password
        self.crawled = crawled
        self.depth = depth
        self.screenshot_tasks = screenshot_tasks
        self.executor = executor
        self.screenshot_res = screenshot_res
        self.base_path = base_path
        self.iterations = iterations
        self.is_baseline = is_baseline


def get_urls(parent_url, depth, context):
    """Recursively crawls the website passed via parent_url and returns the list of found URLs.

    Args:
        parent_url: URL that will be scraped
        depth: depth of the URL that will be scraped
        context: Crawling context object
    Returns:
        list: list of scraped URLs
    """

    if depth > context.max_depth:
        return

    logging.info("Crawling URL " + parent_url)
    url_list = [item["url"] for item in context.crawled]
    driver = get_chrome_driver(context.driver_path)
    try:

        if context.auth_username and context.auth_password:
            parsed_url = furl(parent_url)
            parsed_url.username = context.auth_username
            parsed_url.password = context.auth_password
            get_with_retry(driver, str(parsed_url))
        else:
            get_with_retry(driver, parent_url)

        save_screenshots_and_logs(parent_url, context.screenshot_res, context.base_path, driver,
                                  context.allowed_domains, context)
        html = driver.page_source.encode('utf-8')
        soup = BeautifulSoup(html, features='html.parser')
        urls = soup.findAll('a')

        for a in set(urls):
            url = a.get('href')

            # determine absolute URL for relative links
            if url and not urlparse(url).netloc:
                url = urljoin(parent_url, url)

            # skip urls for which href tag is missing or execute js
            if not is_valid_url(url, context.allowed_domains):
                logging.debug("Skipping invalid URL " + str(url))
                continue

            # cleanup URL
            url.rstrip('/')

            if len(url_list) < context.crawl_max and url not in url_list:
                logging.info("Identified link '{}' on URL '{}'".format(url, parent_url))
                url_list.append(url)
                context.crawled.append({"url": url, "visited": False, "depth": depth + 1})

        for u in context.crawled:
            if u['visited'] is False:
                u['visited'] = True
                future_to_url = context.executor.submit(get_urls, u["url"], u["depth"], context)
                context.screenshot_tasks.append(future_to_url)
    except MaxRetryError:
        logger.error("Max retries {} reached while crawling URL {}".format(CRAWL_URL_RETRIES, parent_url))
    except:
        e = sys.exc_info()[0]
        logger.exception("Error while crawling URL " + parent_url + " with error " + str(e))
    finally:
        logging.info("Finished crawling URL " + parent_url)
        driver.quit()


def is_valid_url(url, allowed_domains):
    # skip invalid, JS using and anchor links
    if not url or any(substr in url for substr in ("javascript", "#")):
        logging.debug("{} skipped. URL not valid".format(url))
        return False
    elif url.endswith((".js", ".mp4", ".mov", ".avi", ".opus", ".mp4v", ".mp4v", ".3gpp", ".3gp2")):
        logging.debug("{} skipped. Media type unsupported.".format(url))
        return False

    parsed_url = urlparse(url)
    # skip URLs that are not in the list of allowed domains
    if allowed_domains and parsed_url.netloc not in allowed_domains:
        logging.debug("{} skipped. Not in list of allowed domains".format(url))
        return False

    return True


def get_job_url_path(base_path, url, domains):
    url_dir = url.replace("\\", URL_SLASH_REPLACEMENT_STR) \
        .replace("/", URL_SLASH_REPLACEMENT_STR) \
        .replace("www.", "") \
        .replace("http:", "") \
        .replace("https:", "") \
        .replace(URL_SLASH_REPLACEMENT_STR + URL_SLASH_REPLACEMENT_STR, URL_SLASH_REPLACEMENT_STR)
    for domain in domains:
        url_dir = url_dir.replace(domain, "")

    if url_dir.startswith(URL_SLASH_REPLACEMENT_STR):
        url_dir = url_dir[1:]
    if url_dir.endswith(URL_SLASH_REPLACEMENT_STR):
        url_dir = url_dir[:-1]

    if not url_dir:
        url_dir = URL_SLASH_REPLACEMENT_STR
    return os.path.join(base_path, url_dir)


def save_screenshots_and_logs(url, screenshot_res, base_path, driver, domains, context):
    """Gets full page screenshots, console logs and network logs for a specific URL

    Args:
        url: URL to process
        screenshot_res: list specifying the width for the screenshots that should be catpured
        base_path: path on disk where the screenshots should be saved
        driver: Selenium driver
        domains: crawled domains
        context: crawl context
=======
    Returns:
        None
    """

    parsed_url = urlparse(url)
    base_url = "" if parsed_url.netloc is None else parsed_url.netloc
    logging.debug('Getting screenshots and logs for {}'.format(url))
    path = get_job_url_path(base_path, url, domains)

    if not os.path.exists(path):
        os.makedirs(path)

    console_logs = driver.get_log('browser')
    for console_log in console_logs:
        console_log.pop('timestamp', None)
    with open(os.path.join(path, "console_logs"), 'w') as f:
        f.write(json.dumps(console_logs))

    network_logs = []
    for request in driver.requests:
        if request.response and request.path is not None:
            trimmed_url_path = re.sub("=(\d{13}$|\d{10}$|\d{13}&|\d{10}&)", '', request.path.replace(base_url, ''))
            network_logs.append({'path': trimmed_url_path, 'status_code': request.response.status_code})
    with open(os.path.join(path, "network_logs"), 'w') as f:
        f.write(json.dumps(network_logs))


    for size in screenshot_res:
        # initial warmup, not sure why this works
        get_with_retry(driver, url)
        time.sleep(VARIANCE_INTERATIONS_INTERVAL_SEC)
        chrome_take_full_screenshot(driver, size)

        screenshots = []
        for i in range(context.iterations):
            get_with_retry(driver, url)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(VARIANCE_INTERATIONS_INTERVAL_SEC)

            screenshots_data = chrome_take_full_screenshot(driver, size)
            screenshots.append(screenshots_data)

        file_path = os.path.join(path, "raw_{}_00.png".format(size))
        if context.is_baseline:
            error = process_screenshot_iterations_consistency(screenshots, file_path)
        else:
            error = save_and_mask_image(screenshots[0], file_path, file_path.replace(UPDATED_DIR, BASELINE_DIR))
        if error is not None:
            logging.error(error)


def crawl(url, max_depth, max_urls, screenshot_res, base_path, chromedriver_path, auth_username=None,
          auth_password=None, workers=NUM_CRAWLING_THREADS, is_baseline=False):
    """Crawls an URL, saves screenshots, network and console logs for the scraped URLs

    Args:
        url: URL to process
        max_depth: maximum depth for crawling (crawling is breadth first)
        max_urls: maximum number of links that should be returned
        screenshot_res: list specifying the width for the screenshots that should be catpured
        base_path: path on disk where the screenshots should be saved
        chromedriver_path: path to the Chrome driver
        auth_username: username to be used for basic authentication
        auth_password: password to be used for basic authentication
        workers: number of worker threads
        is_baseline: whether this is the baseline for the diff comparison
    Returns:
        list: list of encountered errors
    """

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    allowed_domains = [urlparse(url).netloc]
    crawl_list = [{"url": url, "visited": True, "depth": 0}]
    screenshot_tasks = []
    task_errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:

        iterations = 1
        if is_baseline:
            iterations = VARIANCE_NUM_ITERATIONS

        context = CrawlingContext(chromedriver_path, max_urls, max_depth, allowed_domains,
                                  auth_username, auth_password, crawled=crawl_list,
                                  screenshot_tasks=screenshot_tasks, executor=executor,
                                  screenshot_res=screenshot_res, base_path=base_path, iterations=iterations,
                                  is_baseline=is_baseline)
        future_to_url = executor.submit(get_urls, url, 0, context)
        screenshot_tasks.append(future_to_url)

        while screenshot_tasks:
            fs = screenshot_tasks[:]
            for f in fs:
                screenshot_tasks.remove(f)
                try:
                    concurrent.futures.wait(fs)
                except Exception as exc:
                    error_message = 'generated an exception: %s' % exc
                    logging.error(error_message)
                    task_errors.append(error_message)

    url_list = [entry["url"] for entry in crawl_list if entry['visited'] is True]
    logging.info("Visited urls" + str(url_list))
    with open(os.path.join(base_path, "scraped_urls"), 'w', encoding='UTF-8') as f:
        f.write("\n".join(url_list))

    return task_errors


def verify_chrome_driver(driver_path):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--hide-scrollbars")
        options.page_load_strategy = 'normal'
        driver = webdriver.Chrome(driver_path, options=options)
        if driver is not None:
            return None
        else:
            return "Invalid chrome driver loaded"
    except Exception as e:
        return "Error loading chrome driver" + repr(e)


def crawl_job(job_id, baseline_url, updated_url, max_depth, max_urls, workers, screenshot_res,
              auth_baseline_username=None, auth_baseline_password=None, auth_updated_username=None,
              auth_updated_password=None, chromedriver_path=None):
    """
    Runs crawl jobs for the baseline and the updated URLs

    Args:
        base_path: path on disk where the screenshots should be saved
        baseline_url: baseline URL
        updated_url: updated URL
        max_depth: maximum depth for crawling (crawling is breadth first)
        max_urls: maximum number of links that should be returned
        workers: number of worker threads
        screenshot_res: list specifying the width for the screenshots that should be catpured
        chromedriver_path: path to the Chrome driver
        auth_baseline_username: username to be used for basic authentication for the baseline URL
        auth_baseline_password: password to be used for basic authentication for the baseline URL
        auth_updated_username: username to be used for basic authentication for the updated URL
        auth_updated_password: password to be used for basic authentication for the updated URL

    Returns:
        list: list of encountered errors
    """

    # disable most logging for selenium
    selenium_logger = logging.getLogger('selenium.webdriver.remote.remote_connection')
    selenium_logger.setLevel(logging.ERROR)

    max_depth = int(max_depth)
    max_urls = int(max_urls)

    if not chromedriver_path:
        chromedriver_path = os.environ['CHROMEDRIVER_PATH']

    chrome_driver_error = verify_chrome_driver(chromedriver_path)
    if chrome_driver_error is not None:
        return chrome_driver_error

    base_path = os.path.join("jobs", job_id)
    baseline_path = os.path.join(base_path, BASELINE_DIR)
    updated_path = os.path.join(base_path, UPDATED_DIR)
    baseline_errors = crawl(baseline_url, max_depth, max_urls, screenshot_res, baseline_path, chromedriver_path,
                            auth_baseline_username, auth_baseline_password, workers, is_baseline=True)
    updated_errors = crawl(updated_url, max_depth, max_urls, screenshot_res, updated_path, chromedriver_path,
                           auth_updated_username, auth_updated_password, workers, is_baseline=False)

    return baseline_errors + updated_errors


if __name__ == "__main__":
    FORMAT = '%(asctime)-15s [%(levelname)s] %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('crawler')
    logger.setLevel(logging.DEBUG)
