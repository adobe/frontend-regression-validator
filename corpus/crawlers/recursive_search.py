from urllib.parse import urlparse

from bs4 import BeautifulSoup

forbidden = ['facebook', 'instagram', 'google', '.jpg', '.png' , 'mailto', 'tel' 'sigin']


def get_domain(url):
    parsed = urlparse(url)
    return parsed.scheme + '://' + parsed.netloc


def recursive_get_urls(crawled_urls, driver, max_urls, parent_url, domain, depth=0):
    if depth == 0 or len(crawled_urls) == max_urls:
        return crawled_urls
    try:
        driver.get(parent_url)
        html = driver.page_source.encode('utf-8')
        soup = BeautifulSoup(html, features='html.parser')

        urls = soup.findAll('a')
        for a in set(urls):
            url = a.get('href')
            if url is None or url.startswith("#"):
                continue
            if url.startswith("/"):
                url = "https://" + domain + url
            if not url.startswith('http') and not url.startswith("www"):
                url = domain + url
            if urlparse(url).netloc == domain and url not in crawled_urls and not any(s in url for s in forbidden):
                if len(crawled_urls) < max_urls:
                    crawled_urls.append(url)
                    recursive_get_urls(crawled_urls, driver, max_urls, url, domain, depth - 1)
    except:
        print("Failed")
        print(parent_url)
