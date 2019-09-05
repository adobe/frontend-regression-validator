from urllib.parse import urlparse
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.common.exceptions import InvalidArgumentException
import os
from utils.utils import eprint


def _recursive_get_urls(crawled_urls, driver, max_urls, parent_url, domain, depth=0):
    if depth == 0 or len(crawled_urls) == max_urls:
        return crawled_urls
    driver.get(parent_url)
    html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(html, features='html.parser')

    urls = soup.findAll('a')
    for a in set(urls):
        url = a.get('href')
        if url is None:
            continue
        if url.startswith('/'):
            url = parent_url.rstrip('/') + url
        if urlparse(url).netloc == domain and url not in crawled_urls:
            if len(crawled_urls) <= max_urls:
                crawled_urls.append(url)
                print('[LOG] Added: {}'.format(url))
                _recursive_get_urls(crawled_urls, driver, max_urls, url, domain, depth - 1)


def get_recursive_urls(parent_url, max_depth, max_urls):
    scraped_urls = [parent_url]
    domain = urlparse(parent_url).netloc
    if not 'CHROMEDRIVER_PATH' in os.environ:
        eprint('[ERR] CHROMEDRIVER_PATH not provided in env variables')
        exit(5)
    driver_path = os.environ['CHROMEDRIVER_PATH']
    assert os.path.exists(driver_path), 'No such file {}'.format(driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-infobars")
    options.add_argument("--headless")
    driver = webdriver.Chrome(driver_path, chrome_options=options)
    try:
        driver.get(parent_url)
    except InvalidArgumentException:
        eprint('[ERR] Invalid website')
        driver.close()
        exit(1)
    _recursive_get_urls(scraped_urls, driver, max_urls, parent_url, domain, depth=max_depth)
    eprint('[LOG] Finished crawling URLs for {}'.format(parent_url))
    driver.close()
    return scraped_urls
