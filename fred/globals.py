from threading import Lock

timeout_timer_lock = False
jobs_lock = Lock()
jobs = {}

FRED_VERSION = "2.0-a"


STATUS_CODES = {
    1: 'Scheduled',
    2: 'Crawling',
    3: 'Taking screenshots',
    4: 'Analyzing crawl',
    5: 'Sending screenshots to ML component',
    6: 'Running ML',
    7: 'Analyzing ML',
    8: 'Done'
}


# Image variance settings
MEDIAN_KERNEL_WIDTH_RATIO = 8.0
IMAGE_RESIZE_RATIO = 8.0
VARIANCE_NUM_ITERATIONS = 3
VARIANCE_INTERATIONS_INTERVAL_SEC = 10


# Crawling options
URL_SLASH_REPLACEMENT_STR = "@"
MAX_MEGAPIXELS = 20.0
CRAWLER_LOCAL_ADDRESS = ""

CRAWL_URL_RETRIES = 3
PAGE_LOAD_TIMEOUT_SEC = 60
NUM_CRAWLING_THREADS = 10


# Timeouts
TIMEOUT_OVERALL_SEC = 3600
TIMEOUT_AI_SEC = 1800
TIMEOUT_CRAWL_SEC = 1800
