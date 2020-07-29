import sys
import time
import os
from selenium import webdriver
import argparse
import os, json, base64


def chrome_full_screenshot(driver):
    def send(cmd, params):
        resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
        url = driver.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = driver.command_executor._request('POST', url, body)
        return response.get('value')

    def evaluate(script):
        response = send('Runtime.evaluate', {'returnByValue': True, 'expression': script})
        return response['result']['value']

    metrics = evaluate( \
        "({" + \
        "width: Math.max(window.innerWidth, document.body.scrollWidth, document.documentElement.scrollWidth)|0," + \
        "height: Math.max(innerHeight, document.body.scrollHeight, document.documentElement.scrollHeight)|0," + \
        "deviceScaleFactor: window.devicePixelRatio || 1," + \
        "mobile: typeof window.orientation !== 'undefined'" + \
        "})")
    send('Emulation.setDeviceMetricsOverride', metrics)
    screenshot = send('Page.captureScreenshot', {'format': 'png', 'fromSurface': True})
    send('Emulation.clearDeviceMetricsOverride', {})

    return base64.b64decode(screenshot['data'])


class SiteExplorer:
    def __init__(self, base_url, output_folder="/mnt/parsed_websites/"):
        self.base_url = base_url
        self.page_description = {}
        self.path = str(hash(base_url) % ((sys.maxsize + 1) * 2))
        self.output_folder = os.path.join(output_folder, self.path)
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        self.out_json = f'{self.output_folder}/components.json'

        DRIVER =os.environ['CHROMEDRIVER_PATH']
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--force-device-scale-factor=2")
        options.add_argument("--disable-infobars")
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(DRIVER, chrome_options=options)
        self.driver.maximize_window()
        self.driver.fullscreen_window()
        try:
            self.driver.get(base_url)
            self.height = self.driver.execute_script("""var body = document.body,
                                                        html = document.documentElement;
                                                        return Math.max( body.scrollHeight, body.offsetHeight, 
                                                        html.clientHeight, html.scrollHeight, html.offsetHeight );""")
            for i in range(4):
                self.driver.execute_script("window.scrollBy(0, " + str(self.height / 5) + ")")
                time.sleep(3)

            if not os.path.exists(f'{self.output_folder}/website_screenshot.png'):
                png = chrome_full_screenshot(self.driver)
                with open(f'{self.output_folder}/website_screenshot.png', 'wb') as f:
                    f.write(png)

            time.sleep(3)
            self.driver.execute_script("window.scrollTo(0, 0)")
        except:
            pass

    def detect_clickable_elements(self, class_key, dest_class):
        myscript = open('./app/scripts/js/get_element_coordinates.js', 'r').read()
        myscript = myscript.replace('argumentName', dest_class, 1)
        coordin = self.driver.execute_script(myscript)

        images_coord_script = open('./app/scripts/js/get_images_coordinates.js', 'r').read()
        images_coordin = self.driver.execute_script(images_coord_script)

        component_coords = []
        for i, coordinates in enumerate(coordin):
            x1 = 2 * coordinates['x']
            y1 = 2 * coordinates['y']
            x2 = x1 + 2 * coordinates['width']
            y2 = y1 + 2 * coordinates['height']
            json_data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            if x1 != x2:
                component_coords.append(json_data)

        self.page_description[class_key] = component_coords

        filtered_coords = []
        for i, coordinates in enumerate(images_coordin):
            x1 = 2 * coordinates['x']
            y1 = 2 * coordinates['y']
            x2 = x1 + 2 * coordinates['width']
            y2 = y1 + 2 * coordinates['height']
            json_data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            if x1 != x2:
                filtered_coords.append(json_data)

        self.page_description['pics'] = filtered_coords

    def export_json(self):
        with open(self.out_json, 'w') as f:
            json.dump(self.page_description, f, indent=4, separators=(',', ': '))
        self.driver.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="HTML scraper for classes that takes a picture of the extracted element")
    parser.add_argument('--site-url', help='URL of the website to scrape')
    parser.add_argument('--dest-dir', help='Where to save the scraped elements')
    parser.add_argument('--dest-class', help='Classes to be scraped')
    parser.add_argument('--dest-json', help='JSON file name')

    args = parser.parse_args()

    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    site = SiteExplorer(args.site_url, args.dest_class)
