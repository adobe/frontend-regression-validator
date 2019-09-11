from urllib.parse import urlparse

from crawlers.recursive_search import recursive_get_urls
from selenium import webdriver

DRIVER = "./app/driver/chromedriver"
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--headless")
options.add_argument("--force-device-scale-factor=2")
options.add_argument("--disable-infobars")

driver = webdriver.Chrome(DRIVER, chrome_options=options)


website_homepages = open('websites').readlines()

website_full_list = []
for i, website in enumerate(website_homepages):
    website = website.strip("\n")
    print(f"{100*i/len(website_homepages)}%")
    domain = urlparse(website).netloc
    recursive_get_urls(crawled_urls=website_full_list, driver=driver, max_urls=500000, depth=2, parent_url=website, domain=domain)
    print(len(website_full_list))
website_full_list = set(website_full_list)
with open('../full_website_list', 'w') as f:
    for website in website_full_list:
        f.write("%s\n" % website)