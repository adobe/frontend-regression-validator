from app.HTMLScraper import SiteExplorer
import argparse

CLASSES = {
    'textblock': ['.jumbotron', '.footer-single', '.timeline-panel', '.post-preview', '.masthead-content',
                  '.intro-text', '.p-5', '.lead', '.blockquote', '.blog-post', '.featurette-heading', '.article', 'p',
                  'h1', 'h2', 'h3', 'h4', 'h5', 'h6', '.review-text-sub-contents', ".a-span1",
                  ".nav_a", "reviewText",
                  ".a-text-bold", ".a-text-normal", ".s-item__title", "s-item__price", ".s-item__shipping",
                  ".s-item__hotness", ".b-textlink", '.a-link-normal'
                                                     ".b-pageheader__text", ".SECONDARY_INFO"],
    'section': ['.section', '.container', '.page-section', '.container-fluid', '.history-wrapper', 'section', 'footer',
                'header', '.header', '.rhf-border', '.a-section', '.ws-widget-container', '.feature', '.a-container'],

    'button': ['.btn', '.about__social', '.btn-group', '.social-media-link', '.social-info', '.social-buttons',
               '.social', '.social-icons', '.social-profile', '.social', '.facebook', '.twitter', '.instagram',
               '.google',
               ".w5_btn_label", ".a-button-input", ".a-button-inner", ".a-button-text", ".a-box-inner", ".ghspr"
               ],

    'form': ['.form-group', '.input-group', '.subscribe', '.form-control'],

}


def run_pipeline(sites, index, output_folder):
    sites = open(sites, 'r').readlines()
    sites = [site.strip() for site in sites]
    for counter, site in enumerate(sites[index:]):
        site_url = site.strip()
        print(f'{counter + index}/{len(sites)} Current website: {site_url}')
        se = SiteExplorer(site_url, output_folder)

        for k in CLASSES:
            v = ", ".join(CLASSES[k])
            se.detect_clickable_elements(k, v)
        se.export_json()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--website-list", help="Webiste list from which to start scraping")
    argparser.add_argument("--start-index", help="Start index in the list. This is used for resuming processses",
                           default=0)
    argparser.add_argument("--output-folder", help="Output folder for the scraping", default='parsed_websites')

    args = argparser.parse_args()

    run_pipeline(args.website_list, int(args.start_index), args.output_folder)
