from urllib.parse import urlparse
from bs4 import BeautifulSoup
from utils.utils import eprint
from pyppeteer import launch
import asyncio


class MyPage(object):
    def __init__(self):
        self.source = None

    def set(self, source):
        self.source = source


async def get_page(test_page, url):
    browser = await launch({'args': ['--disable-dev-shm-usage', '--no-sandbox']})
    page = await browser.newPage()
    await page.goto(url)
    out = await page.content()
    test_page.set(out)
    await browser.close()


def _recursive_get_urls(crawled_urls, test_page, max_urls, parent_url, domain, depth=0):
    if depth == 0 or len(crawled_urls) == max_urls:
        return crawled_urls
    asyncio.get_event_loop().run_until_complete(get_page(test_page, parent_url))

    html = test_page.source
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
                eprint('[LOG] Added: {}'.format(url))
                _recursive_get_urls(crawled_urls, max_urls, url, domain, depth - 1)


def get_recursive_urls(parent_url, max_depth, max_urls):
    scraped_urls = [parent_url]
    domain = urlparse(parent_url).netloc
    page = MyPage()
    asyncio.get_event_loop().run_until_complete(get_page(page, parent_url))
    _recursive_get_urls(scraped_urls, page, max_urls, parent_url, domain, depth=max_depth)
    eprint('[LOG] Finished crawling URLs for {}'.format(parent_url))
    return scraped_urls
