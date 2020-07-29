import json, os, logging
from utils.scores import log_analysis


def crawl_analysis(job_id):
    """
    The function returns a dict with keys:
        "crawl_divergence":
            0 if everything is identical
            100 if there are baseline pages are not in updated pages, or the other way around
            [0, 100] as the max divergence of all individual pages. WARNING, the console logs are not taken into account!
        "unique_baseline_pages": list of pages that are only found in the baseline crawl
        "unique_updated_pages": list of pages that are only found in the updated crawl
        "pages":
            <page k>:
                "console_logs":
                    "baseline_filepath": path to json
                    "updated_filepath": path to json
                    "divergence": a score [0, 100] measuring the differences between console logs
                    "report": dict with the differences, like: {'on_baseline_not_on_updated': [], 'on_updated_not_on_baseline': []}
                "network_logs":
                    "baseline_filepath": path to json
                    "updated_filepath": path to json
                    "divergence": a score [0, 100] measuring the differences between network logs
                    "report": dict with differences, like: {'on_baseline_not_on_updated': [], 'on_updated_not_on_baseline': []}
    """


    logging.info("Starting crawl analysis for job id {}".format(job_id))

    report = {}
    error = ""

    try:
        base_path = os.path.join("jobs", job_id)
        baseline_pages = [x for x in os.listdir(os.path.join(base_path, "baseline")) if os.path.isdir(os.path.join(base_path, "baseline", x))]
        updated_pages = [x for x in os.listdir(os.path.join(base_path, "updated")) if os.path.isdir(os.path.join(base_path, "updated", x))]

        unique_baseline_pages = [p for p in baseline_pages if p not in updated_pages]
        unique_updated_pages = [p for p in updated_pages if p not in baseline_pages]
        common_pages = set(baseline_pages) & set(updated_pages)

        logging.debug("Baseline pages: {}".format(baseline_pages))
        logging.debug("Updated pages : {}".format(updated_pages))
        logging.debug("Common pages: {}".format(common_pages))

        # network and console log analysis
        report["crawl_divergence"] = 0.
        report["unique_baseline_pages"] = []
        report["unique_updated_pages"] = []

        if len(unique_baseline_pages) > 0 or len(unique_updated_pages) > 0 or len(list(common_pages)) == 0:
            report["crawl_divergence"] = 100.
            report["unique_baseline_pages"] = unique_baseline_pages
            report["unique_updated_pages"] = unique_updated_pages

        crawl_scores = []
        report["pages"] = {}
        for p in common_pages:
            report["pages"][p] = {}

            b_console_filepath = os.path.join(base_path, "baseline", p, "console_logs")
            b_network_filepath = os.path.join(base_path, "baseline", p, "network_logs")
            u_console_filepath = os.path.join(base_path, "updated", p, "console_logs")
            u_network_filepath = os.path.join(base_path, "updated", p, "network_logs")


            b_console_log = json.load(open(b_console_filepath, "r", encoding="utf8"))
            b_network_log = json.load(open(b_network_filepath, "r", encoding="utf8"))
            u_console_log = json.load(open(u_console_filepath, "r", encoding="utf8"))
            u_network_log = json.load(open(u_network_filepath, "r", encoding="utf8"))

            score, differences = log_analysis(b_console_log, u_console_log)
            report["pages"][p]["console_logs"] = {}
            report["pages"][p]["console_logs"]["baseline_filepath"] = b_console_filepath
            report["pages"][p]["console_logs"]["updated_filepath"] = u_console_filepath
            report["pages"][p]["console_logs"]["divergence"] = score
            report["pages"][p]["console_logs"]["divergence"] = score
            report["pages"][p]["console_logs"]["report"] = differences

            score, differences = log_analysis(b_network_log, u_network_log)
            report["pages"][p]["network_logs"] = {}
            report["pages"][p]["network_logs"]["baseline_filepath"] = b_network_filepath
            report["pages"][p]["network_logs"]["updated_filepath"] = u_network_filepath
            report["pages"][p]["network_logs"]["divergence"] = score
            report["pages"][p]["network_logs"]["report"] = differences

            logging.debug("Crawl analysis of {} has score: {}".format(p,score))
            crawl_scores.append(score)
    except Exception as ex:
        error = "{}".format(ex)
        logging.error("Error crawling: {}".format(str(ex)))

    if len(crawl_scores)>0:
        report["crawl_divergence"] = max(crawl_scores)# NOTE: IGNORING CONSOLE LOGS

    logging.info("Finished crawl analysis for job id {}".format(job_id))
    return report, error

if __name__ == "__main__":
    print("Analysis testing")
    jobs = {}
    job_id = "2020-03-23__22-14-49__dota"
    report, error = crawl_analysis(job_id)
    print(report)
    print(error)