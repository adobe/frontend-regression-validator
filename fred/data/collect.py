import os
from selenium import webdriver
from selenium.webdriver import DesiredCapabilities
import time
import json
import base64
from browsermobproxy import Server
from pprint import pprint

SCRIPT = """var body = document.body,
            html = document.documentElement;
            return Math.max( body.scrollHeight, body.offsetHeight, 
            html.clientHeight, html.scrollHeight, html.offsetHeight );"""


def _chrome_full_screenshot(driver):
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


def collect_data(url, output_folder, output_filename):
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
    output_folder = os.path.join("./tmp", output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    server = Server("utils/browsermob_proxy/bin/browsermob-proxy", options={'port': 8090})
    server.start()
    proxy = server.create_proxy()
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--force-device-scale-factor=2")
    options.add_argument("--disable-infobars")
    options.add_argument("--headless")
    options.add_argument('--proxy-server=%s' % proxy.proxy)
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    capabilities = DesiredCapabilities.CHROME
    capabilities['goog:loggingPrefs'] = {'browser': 'ALL'}

    driver = webdriver.Chrome(os.environ['CHROMEDRIVER_PATH'], chrome_options=options,
                              desired_capabilities=capabilities, service_args=["--verbose"])
    driver.maximize_window()
    driver.fullscreen_window()
    proxy.new_har("Logs")
    driver.get(url)

    logs = driver.get_log('browser')
    with open(os.path.join(output_folder, output_filename.split('.')[0] + '_js_log.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    with open(os.path.join(output_folder, output_filename.split('.')[0] + '_network_log.json'), 'w') as f:
        json.dump(proxy.har, f, indent=2)

    height = driver.execute_script(SCRIPT)
    for i in range(4):
        driver.execute_script("window.scrollBy(0, " + str(height / 5) + ")")
        time.sleep(3)
    png = _chrome_full_screenshot(driver)
    with open(os.path.join(output_folder, output_filename), 'wb') as f:
        f.write(png)
    driver.execute_script("window.scrollTo(0, 0)")
    driver.close()
    server.stop()
