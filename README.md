[![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![Version 2.0](https://img.shields.io/badge/version-2.0-red.svg)]()

# Frontend Regression Validator (FRED)

FRED is an opensource visual regression tool used to compare two instances of a website. 
FRED is responsible for automatic visual regression testing, with the purpose of ensuring that functionality is not broken by comparing a current(baseline) and an updated version of a website. 
FRED compares the following:
* Console and network logs
* Visual: screenshot analysis
* Visual AI: screenshot analysis using Machine Learning techniques
   
The visual analysis computes the Normalized Mean Squared error and the Structural Similarity Index on the screenshots of the baseline and updated sites, while the visual AI looks at layout and content changes independently by applying image segmentation Machine Learning techniques to recognize high-level text and image visual structures. This reduces the impact of dynamic content yielding false positives.

Use FRED if you need:
* Screenshot comparision
* Automatic layout/content verification

FRED is designed to be scalable. It has an internal queue and can process websites in parallel depending on the amount of RAM and CPUs (or GPUs) available.

##### NOTE: The entire repo for version 1.0 is available in ``_fred-v1``. Please note that v2.x (current version) does not contain the code to train/retrain the ML model. If you need to do that, please check the original code in the v1 folder. The models are identical, so if you create your custom-trained model, plug it in v2 and it will work.  

## Start FRED

You can start FRED either as a docker or as a local process. 

#### Quickstart Docker
If you just want to clone and run the software, we have provided a Dockerfile. To run it:

```shell script
git clone https://github.com/adobe/frontend-regression-validator.git
cd frontend-regression-validator/docker
docker build --no-cache -t fred .
docker run -p 5000:5000 -m 8g --memory-reservation=8g --oom-kill-disable=True --memory-swap 8G fred
```

If you still encounter issues with out of memory errors, allocate more memory from the UI Docker app. Simply click the Docker icon in your toolbar, go to `Preferences`-`Advanced` and then pull the slider to `8GB` or more, especially if you plan to use ML (optional).
We recommended running it locally instead of using the Dockerfile or to increase the memory allocated to docker to `at least 8GB, prefferably 16GB`.

#### Start Locally

Ensure you have installed `chromedriver`. If you don't have it, install it on MAC with:
```
brew tap homebrew/cask && brew cask install chromedriver
```
or on Linux with:
```
sudo apt-get install chromium-chromedriver
```
then run the following:
```
git clone https://github.com/adobe/frontend-regression-validator.git
cd frontend-regression-validator
pip install -r requirements.txt
cd fred/ml
cat model_files.bz2.parta* > model_files.bz2
tar xjf model_files.bz2
cd ..
python3 run.py
```

This will launch a Flask instance that answers to requests as well as offering a web user interface. Quicknote: use ``--port`` to specify the listening port, by default it listens on ``5000``. Please view more details about FRED's [startup parameters here](DETAILS.md#fred-startup-settings).

## Use FRED
Interaction with FRED is done either by web UI or by API calls. The UI simply allows a user to send calls to the API endpoint and view results.

To open the web interface navigate to ``http://0.0.0.0:5000/static/submit.html`` (adjust port accordingly). Fill in all the required fields, run the job and wait until it completes. View results by clicking the ``Jobs`` link in the header.

To use the API please view the dedicated [API readme here](DETAILS.md#fred-api).

## Overlook on how FRED operates

FRED waits until it receives a request to perform a website comparison (POST call to ``/api/verify``). It starts the crawl process. We can request to see all jobs with a GET call to ``/api/viewjobs`` and get the status of a particular job with a GET to ``/api/results`` providing the job id as a parameter.

* As such, the input to FRED is a pair of URLs to compare. 

* The process begins with FRED crawling the URLs to extract a number of pages to compare, and then renders each page and takes screenshots. 

* The console and network logs are compared.

* Each screenshot is analyzed (as pairs of baseline/updated screenshots, for each specified resolution).

* If enabled, each screenshot pair also undergoes ML analysis

* Results are saved locally (a user must periodically check via the API until the ``status`` is set to ``Done`` and/or some ``error`` is set.)

* A result is a json object that in the ``report`` key contains a number of scores. The ``overall_divergence`` score is the weighted sum of the network, visual and ai-visual(if enabled) divergences. A score of ``0`` means a perfect match (no difference between baseline and updated), while higher scores, up to ``100`` highlight differences.

* If needed use the visual interface to quickly investigate results. Otherwise, the ``report`` also contains links to the raw images as well as analysis images that highlight differences if you want to use FRED in an automated fashion.

Because FRED is designed to be scalable, it is logically split in two components: ``crawler`` and ``ML``. The ``crawler`` component is the main entry point that the user interacts with. The ``ML`` component, while being the same code as the ``crawler`` component, is simply another endpoint listening for API calls. The logic behind this split is that GPUs are expensive while CPUs are not. So we can have many crawlers that in turn make requests to a few GPU enabled FRED instances (called ``ML`` components) to perform the ML analysis. 

For example, imagine a scenario where we have 1000 websites to analyze daily. We create 10 virtual machines, each with 32GB RAM and 8 vCPUs. Each instance will receive 100 ``/api/verify`` calls. Assume we set ``--crawler_threads`` to 5, meaning we can concurrently crawl 5 websites. Furthermore, as we only have a single GPU machine with 4 GPUs, we launch on it a FRED instance, which we'll call the ``ML`` component. On this instance we set ``--ai_threads`` to 4, meaning we concurrently run 4 ML validations. Now, on each of the POST API requests to the ``crawler`` components we set the ``ml_address`` to the ``ml`` component's address. What will now happen is that whenever a ``crawler`` component finishes to crawl and analyize (non-AI) a website pair, it will send the ``ML`` component its screenshots and request to analyze them. The ``ml`` component will add this request in its queue and when a GPU is available it will run the comparison on it. When finished, it will automatically report back to the originating ``crawler`` component its analysis. Basically, this approach scales performance linearly with the number of available testing machines. 

#### Performance analysis

FRED runtimes vary greatly on the complexity of the site. Most of the time is spent in the crawler component, as (unfortunately) loading a website is not a deterministic process. Sometimes, a website simply hangs, or a popup randomly appears, or some external resource refuses to load. Internally, we have the only remedy to this: a sort of ``try-catch`` that reloads a website if something horrible happens. But this, coupled with the fact that we wait a few seconds after the page says it loaded, plus the repeated screenshots to discover dynamic content, dramatically increase crawl time.

The crawl part usually takes 2-10 minutes, depending on the number of pages crawled.

Visual analysis (with each screenshot limited to at most 20 Megapixels) takes ~5-10 seconds per image pair. Each extra resolution means another set of image pairs.

AI (ML) visual analysis takes 0-30 seconds per image pair **on a GPU**. Any GPU will do, even an old K80 will run very fast, as the ML par is a U-NET (stacked convolutional layers). You can always run on a CPU, but instead of 30 seconds per image pair, you might wait 5 minutes per image pair. 

Overall, the rule of thumb for a ML-enabled crawl, start to finish, is 1 minute or less per page.  
