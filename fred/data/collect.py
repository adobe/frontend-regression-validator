import os
import asyncio
import json
from pyppeteer import launch

network = []
javascript = []


async def intercept_network_response(response):
    network.append(str(response.status) + response.url)


async def intercept_console(response):
    javascript.append(response.text)


async def collect_msgs_and_screenshot(url, ss_path):
    browser = await launch()
    page = await browser.newPage()

    page.on('response', intercept_network_response)
    page.on('console', intercept_console)
    page.on('requestfailed', intercept_console)
    page.on('pageerror', intercept_console)
    page.on('error', intercept_console)

    await page.goto(url)
    await page.screenshot({'path': ss_path, 'fullPage': True})

    await browser.close()


def collect_data(url, output_folder, output_filename):
    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")
    output_folder = os.path.join("./tmp", output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    asyncio.get_event_loop().run_until_complete(
        collect_msgs_and_screenshot(url, os.path.join(output_folder, output_filename)))

    with open(os.path.join(output_folder, output_filename.split('.')[0] + '_js_log.json'), 'w') as f:
        json.dump(javascript, f, indent=2)

    with open(os.path.join(output_folder, output_filename.split('.')[0] + '_network_log.json'), 'w') as f:
        json.dump(network, f, indent=2)
