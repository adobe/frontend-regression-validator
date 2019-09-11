import time
from subprocess import Popen
from argparse import ArgumentParser
import os


def split_list(num_procceses):
    website_list = open('full_website_list', 'r').readlines()
    len_of_list = len(website_list) // num_procceses
    if not os.path.exists('tmp/'):
        os.mkdir('tmp')
    for i in range(num_procceses):
        with open('tmp/website_list_part_' + str(i), 'w') as f:
            for j in range(i * len_of_list, (i + 1) * len_of_list):
                f.write(website_list[j])


def start_scraping_parallel(num_processes, output_folder):
    split_list(num_processes)
    processes = []
    out_file = open("/dev/null", 'w')
    for i in range(num_processes):
        log_file = open("log_process_{}".format(i), 'w')
        processes.append(Popen(
            ['python3', 'pipeline.py', '--website-list', 'tmp/website_list_part_{}'.format(i), '--output-folder', output_folder],
            stdout=log_file,
            stderr=out_file))

    while True:
        time.sleep(1)
        for i in range(num_processes):
            if processes[i].poll() is not None and processes[i].poll() != 0:
                with open("log_process_" + str(i), 'r') as f:
                    logs = f.readlines()
                    logs = list(filter(lambda x: 'Current' in x, logs))
                    last_site = int(logs[-1].split('/')[0])
                    total_sites = int(logs[-1].split('/')[1].split(' ')[0])
                    if last_site < total_sites:
                        log_file = open("log_process_{}".format(i), 'w')
                        processes[i] = Popen(
                            ['python3', 'pipeline.py', '--website-list', 'tmp/website_list_part_{}'.format(i),
                             '--start-index', str(i),'--output-folder', output_folder],
                            stdout=log_file,
                            stderr=log_file)
                        print("Restarted process {} from index {}".format(str(i), str(last_site)))
            else:
                continue
        if all(processes[i].poll == 0 for i in range(num_processes)):
            exit(0)


argparser = ArgumentParser()
argparser.add_argument("--num-processes", help="How many parallel scraping processes to run")
argparser.add_argument("--output-folder", help="Where to save the crawl folders")

args = argparser.parse_args()

start_scraping_parallel(int(args.num_processes), args.output_folder)
