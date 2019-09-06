from crawler.crawler import get_recursive_urls
from data.collect import collect_data
import argparse
from urllib.parse import urlparse
import json
from subprocess import Popen
from utils.utils import eprint, add_auth, dsum, dstd
import os


def get_domain(url):
    parsed = urlparse(url)
    return parsed.scheme + '://' + parsed.netloc


def get_path(url):
    parsed = urlparse(url)
    path = parsed.path
    if len(path) == 0:
        path = '/'
    if len(parsed.query) > 0:
        path += '/?' + parsed.query
    if len(parsed.fragment) > 0:
        path += '#' + parsed.fragment
    return path


def work(baseline_url, updated_url, max_depth, max_urls, prefix, auth_baseline_username, auth_baseline_password,
         auth_updated_username, auth_updated_password):
    baseline_url = add_auth(url=baseline_url, username=auth_baseline_username, password=auth_baseline_password)
    updated_url = add_auth(url=updated_url, username=auth_updated_username, password=auth_updated_password)
    crawled_baseline = get_recursive_urls(baseline_url, max_depth, max_urls)[:max_urls]
    crawled_upgraded = get_recursive_urls(updated_url, max_depth, max_urls)[:max_urls]

    baseline_domain = get_domain(baseline_url)
    updated_domain = get_domain(updated_url)

    crawled_baseline_paths = [get_path(path) for path in crawled_baseline]
    crawled_updated_paths = [get_path(path) for path in crawled_upgraded]

    all_paths = list(set(crawled_baseline_paths) | set(crawled_updated_paths))
    ss_report = {}

    for i, path in enumerate(all_paths):
        eprint('[LOG] Taking screenshots for {} - {}'.format(prefix, path))
        collect_data(baseline_domain + path, prefix + '_baseline', '{}.png'.format(i + 1))
        collect_data(updated_domain + path, prefix + '_updated', '{}.png'.format(i + 1))
        ss_report[i + 1] = {'baseline': baseline_domain + path, 'updated': updated_domain + path, 'endpoint': path,
                            'baseline_assets': 'tmp/' + prefix + "_baseline/",
                            'updated_assets': 'tmp/' + prefix + "_updated/"}
    eprint('[LOG] Finished taking screenshots for {}'.format(prefix))
    with open(os.path.join('./tmp', prefix + '_ss_report.json'), 'w') as f:
        json.dump(ss_report, f, indent=2)

    p = Popen(['python3', 'worker_predict.py', '--baseline-dir', prefix + '_baseline', '--updated-dir',
               prefix + '_updated', '--prefix', prefix])
    if p.poll() is not None and p.poll() > 0:
        eprint('[ERR] Failed to launch inference process')
        exit(3)
    eprint('[LOG] Waiting for {}'.format(prefix))
    p.wait()
    if p.poll() != 0:
        eprint('[ERR] Prediction script failed for {}'.format(prefix))
        exit(p.poll())
    eprint('[LOG] Finished prediction for {}'.format(prefix))

    ui_risk_scores = []
    network_risk_scores = []
    js_stats_total = []
    net_stats_total = []
    pixelwise_div_total = []
    mask_div_total = []
    with open(os.path.join('./tmp', prefix + '_report.json'), 'w') as f:
        scores_report = json.load(open(os.path.join('./tmp', prefix + '_scores.json')))
        screenshots_report = json.load(open(os.path.join('./tmp', prefix + '_ss_report.json')))
        page_report = {}
        for i in range(1, len(all_paths) + 1):
            page_report[i] = scores_report[str(i)]
            js_stats_total.append(scores_report[str(i)]["js_stats"])
            net_stats_total.append(scores_report[str(i)]["network_stats"])
            page_report[i]['links'] = screenshots_report[str(i)]
            ui_risk_scores.append(page_report[i]["ui_stats"]["risk_score"])
            network_risk_scores.append(page_report[i]["risk_score"])
            pixelwise_div_total.append(page_report[i]['ui_stats']['pixelwise_div'])
            mask_div_total.append(page_report[i]['ui_stats']['mask_div'])
        page_report['risk_score'] = max(max(ui_risk_scores), max(network_risk_scores))

        page_report['js_stats'] = dsum(js_stats_total)
        page_report['ui_stats'] = {'pixelwise_div_mean': dsum(pixelwise_div_total, True),
                                   'mask_div_mean': dsum(mask_div_total, True),
                                   'pixelwise_div_std': dstd(pixelwise_div_total),
                                   'mask_div_std': dstd(mask_div_total)}
        page_report['network_stats'] = dsum(net_stats_total)
        json.dump(page_report, f, indent=4)
        eprint('[LOG] Saved {} report to {}'.format(prefix, prefix + '_report.json'))
        os.remove(os.path.join('./tmp', prefix + '_scores.json'))
        os.remove(os.path.join('./tmp', prefix + '_ss_report.json'))

    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-url', type=str)
    parser.add_argument('--updated-url', type=str)
    parser.add_argument('--max-depth', type=int)
    parser.add_argument('--max-urls', type=int)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--auth-baseline-username', type=str, default='')
    parser.add_argument('--auth-baseline-password', type=str, default='')
    parser.add_argument('--auth-updated-username', type=str, default='')
    parser.add_argument('--auth-updated-password', type=str, default='')

    args = parser.parse_args()

    work(args.baseline_url, args.updated_url, args.max_depth, args.max_urls, args.prefix, args.auth_baseline_username,
         args.auth_baseline_password, args.auth_updated_username, args.auth_updated_password)
