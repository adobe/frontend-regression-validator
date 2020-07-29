# FRED Details


## FRED startup settings

FRED is configured from 2 sources:
* FRED is started by running ``run.py``. Use ``--help`` to see all available parameters.
* Edit the ``globals.py`` file.  

#### Startup parameters

* ``--host`` is the local address that FRED will listen on. Default is ``0.0.0.0``
* ``--port`` is the local port FRED will listen on. Default is ``5000``
* ``--external_address`` is the external address of FRED that an outside ML component will send results back at. Useful for NAT situations when the crawler component is hidden from a ML component
* ``--external_port`` is the complementary port of the external address
* ``--crawl_threads`` is the number of concurrent threads that handle crawling. Please note that chromium requires a few GB of RAM per instance, so scale accordingly to avoid overflowing RAM
* ``--ai_threads`` is the number of concurrent threads that handle the ML image analysis. Default is 1, because if you use a GPU (which you should anyway if you want ML analysis) sequential running won't overflow the GPU's RAM
* ``--max_megapixels`` is the number of megapixels that the (raw) image analysis will crop each screenshot to. This option is set at 20MP default
* ``--log_level`` standard pyhon logging level. Default is ``debug``, available levels are: ``debug``, ``warning``, ``info``, ``error``
* ``--timeout_overall`` is the maximum timeout in seconds to wait for a job to be finished, from scheduled to done. Set 0 to disable, default is 60 minutes (3600 seconds).')
* ``--timeout_crawl`` is the maximum timeout in seconds to wait for the AI analysis to be finished. Set 0 to disable, default is 30 minutes (1800 seconds).')
* ``--timeout_ai``is the maximum timeout in seconds to wait for the AI analysis to be finished. Set 0 to disable, default is 30 minutes (1800 seconds)

#### globals.py parameters

Edit these only if you know what you're doing. Some of the parameters here are set by the ``run.py`` arguments. A few of the more interesting ones are:

##### Image variance settings (for raw image analysis)

* ``MEDIAN_KERNEL_WIDTH_RATIO = 8.0``
* ``IMAGE_RESIZE_RATIO = 8.0``
* ``VARIANCE_NUM_ITERATIONS = 3``
* ``VARIANCE_INTERATIONS_INTERVAL_SEC = 10``

##### Crawling options

* ``CRAWL_URL_RETRIES = 3``
* ``PAGE_LOAD_TIMEOUT_SEC = 60``
* ``NUM_CRAWLING_THREADS = 10``

## FRED Process

What is actually happening when a FRED process starts?
Remember, a process starts by sending a request with a baseline URL and an updated URL (and a couple other params, detailed further below). FRED is set as a async state machine, meaning that each job (request) is treated in a queued manner and always has a status attached. On your local file system you will find a ``fred/jobs`` folder that will contain each individual job.
A job is uniquely identified by a timestamp and a suffix; this id is the folder name in the ``fred/jobs`` folder. A ``log.json`` file will contain all the information about the job, and will be updated async, as soon as anything changes. 

Here's what happens:

###### Step 1. Status = Scheduled

A request is received, a unique id is generated, a folder appears in ''fred/jobs'' with this id, and a ''log.json'' file is created.

The job waits for a free crawler thread.

###### Step 2. Status = Crawling

As soon as a thread is available, the job is updated with the status ``Crawling``. The baseline is crawled up to a predefined depth and links are collected. This initial link exploration is done in a depth-first manner, up to a maximum depth, and a number of links will be extracted for being compared.

###### Step 3. Status = Taking screenshots

Each link obtained in the previous step is crawled (meaning fully loaded in chromium) and screenshots are taken, first on the baseline site, then on the updated site. Screenshots are taken at all specified resolution.

The ``log.json`` file is updated. A new ``pages`` value will appear in its dictionary with stats about each page. Locally, in the job's folder, a ``baseline`` and ``updated`` folders will appear, and each page will have its own subfolder with screenshots saved as ``raw_xyz_00.png`` where ``xyz`` is the resolution.

###### Step 4. Status = Analyzing crawl

Each screenshot is analyzed: dynamic content is masked (best-effort), and different similarity measures are computed on each resolution - URL pair. 

The ``log.json`` file is updated. A new ``pages`` value will appear in its dictionary with stats about each page, and a ``results`` will contain raw image analysis. Each page subfolder will now contain image analysis highlighting differences between images (only in the ``updated`` folder)

###### Step 5. Status = Sending screenshots to ML component

If the request had the ``ml_enabled`` property set, then the process continues with this status. Else, status is set to ``Done`` and the crawler thread is free to pick up another job.

What happens here is that all raw images are packed in a HTTP request and sent to the address specified in the ``ml_component`` field of the initial request.
    
###### Step 6. Status = Running ML

The FRED instance awaits a request sent by a component in step 5. It unpacks all screenshots, writes them on disk (if we are using the same local instance, nothing happens as the files are already there) and starts the ML analysis. 

Here, the ML model is loaded and run on all resolution pairs. If a GPU is available, it will automatically be used to speed up computation.

###### Step 7. Status = Analyzing ML

As the ML process creates a ``textblock`` and a ``image`` output file, in this step the alignment between identified masks is performed, and content is compared pixel-by-pixel. Results are written in the ``results`` key in the ``log.json`` file locally, and all this information (screenshots + analysis) is sent back to the requester address/port.

The requester also updates its ``log.json`` file (if not already done if using the same local address for ML). 

###### Step 8. Status = Done

This status is reached if all previous steps have completed successfully. This means that a job is finished. 

*** Please note: to judge a job status you have to look at both the ``status`` field, which should be set to ``Done`` *and* the ``error`` field, which should be empty. If the status is ``Done`` and the error field is not empty, it means that the job has failed somehow, with the error message being written in the ``error`` field.
   

## FRED API

FRED exposes the following methods:

###### /api/verify 

Call this when you want FRED to validate a website. Method: **POST**

* Parameters:

``baseline_url`` = URL of the original site. Type: ``str``, *Required*
``updated_url`` = URL of the updated site. Type: ``str``, *Required*
``max_depth`` = Max depth to search for links. Type: ``int``, Default: ``10``, *Optional*
``max_urls`` = Number of URLs (pages) to compare. Type: ``int``, Default: ``10``, *Optional*
``prefix`` = Use this field if you want to add an identifier after the job id (which is a time value). If left empty a short random hexa code will be generated and appended. Even though this is a suffix, the name ``prefix`` is kept for backward compatibility. Do NOT use spaces in this field. Type: ``str``, Default: (empty), *Optional*
``ml_enable`` = Set to ``True`` to enable ML processing. Type: ``bool``, Default: ``False``, *Optional*
``ml_address`` = Set the ML address of the FRED instance which has a GPU (usually you want to do ML on a GPU). Type: ``str``, Default: ``0.0.0.0:5000``, *Optional*
``requested_resolutions`` = List of resolutions screenshots will be taken at. Either set a single value or join multiple values with a comma. Type: ``list``, Default: ``512,1024``, *Optional*
``requested_score_weights`` = The overall divergence is a weighted sum of the network score, visual analysis score and AI visual analysis score. This value is always a 3-valued list with the weights for each of these components. Note that if ML enabled is set to False, it will automatically rescale the first two to sum to one (and ignore the third which belongs to the AI analysis). Type: ``list``, Default: ``0.1,0.4,0.5``, *Optional*
``requested_score_epsilon`` = This is supposed to be an epsilon to allow fuzzy scores. *Not yet implemented*. Type: ``epsilon``, Default: ``10,0,0``, *Optional*

* Success Response:
  
  A status 200 code is returned with the following content:
  ```
  { id : job_id }
  ```

* Error Response:

  This method cannot really fail as all it does is locally schedule a job for crawling and it just allocates an unique id. If it fails then it has a != 200 status code and will most likely be some HTTP code indicating that the server is inaccessible, etc.
  
* Sample Call:
  `
    arr = {
      "baseline_url": "https://www.test.com",
      "updated_url": "https://www.test.com",
      "max_depth": "1",
      "max_urls": "1",
      "prefix": "test"
    };
    $.ajax({
        type: "POST",
        url: "/api/verify",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(arr),
    });
  `

###### /api/result 

Call this to obtain the status of a job based on a job id. Method: **GET**

     if not report:
            return {'Error': 'Report does not exist'}, 404
        else:
            pages = report.get("pages")
            if pages is not None:
                for p in list(pages):
                    pages[normalize_url_name(p)] = pages[p]
                    del pages[p]

            return report, 200

* Parameters:

``id`` = id of the job. Type: ``str``, *Required*

* Success Response:
  
  A status 200 code is returned with a report (a dictionary) with the status and details of the respective job.

* Error Response:

  A 404 is returned if the job does not exist (invalid job id specified).
  
* Sample Call:
  `
    arr = {
      "baseline_url": "https://www.test.com",
      "updated_url": "https://www.test.com",
      "max_depth": "1",
      "max_urls": "1",
      "prefix": "test"
    };
    $.ajax({
        type: "POST",
        url: "/api/verify",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(arr),
    });
  `

###### /api/viewjobs 

Call this to obtain a status of all jobs on a FRED instance. Method: **GET**

* Parameters:

No parameters are needed here. 

* Success Response:
  
  A status 200 code is returned with the full dictionary containing all jobs (as they are in the moment of calling). Each job is a key in the dict. Please see the log details section for more info about how this dict is structured.

* Error Response:

  If there are no jobs, an empty response will be sent with status 200. Otherwise, the default HTTP codes apply if there are any errors. 
  
* Sample Call:
  


## FRED log object details

The log object contains all the information regarding a job. It is represented as a JSON, with the following top-level structure:
```json
    error: "",
    input_data: {},
    pages: {},
    report: {},
    stats: {},
    status: "Done"
```
Let's analyze each field:

###### status & error

These are the two main fields indicating the status of the job (as a string) - one of the steps [detailed here](DETAILS.md#fred-process). The error field (a string) indicates if there was an error processing the job. Thus:
* a **successful** job has status ``Done`` and an empty error field.
* an **unsuccessful** job has status ``Done`` and a **non-empty** error field.
* a job **in progress** has a status different from ``Done``.

###### input_data

This is a dictionary containing information regarding the parameters this job was instantiated with:

###### stats

This dictionary contains durations for crawling, ML processing (if enabled) and overall.

```json
    cr_finished_at: "2020-06-24 16:47:15.704698",
    cr_started_at: "2020-06-24 16:45:09.074632",
    finished_at: "2020-06-24 16:47:26.722477",
    ml_finished_at: "2020-06-24 16:47:26.686243",
    ml_started_at: "2020-06-24 16:47:15.833722",
    queued_at: "2020-06-24 16:45:09.066220"
```

Note that ``cr_started_at`` (crawling started) is not necessarily the same as `queued_at`, as the actual starting time depends on available crawler threads.

###### pages

The ``pages`` object contains a list of crawled pages. Each such page is identified by the page address, and is a dictionary itself containing:
* ``console_logs`` and ``network_logs``: they have a similar structure, and contain a ``divergence`` score (0-100) and a listing on what differs from baseline and updated.
* ``raw_screenshot_analysis`` contains a list of resolutions as key, with the following values:

```json 
"1920": {
         "diff_pixels": 0.0,
         "mse": 0.0,
         "ssim": 100.0,
        }
```

This is an example of an 1920px (width) baseline-updated screenshot analysis, with the ``diff_pixels`` being the percent of different pixels between the screenshots, ``mse`` being the normalized mean square error and ``ssim`` being the structural similarity index. For a pair of identical screenshots, ``diff_pixels`` and ``mse`` are 0, while ``ssim`` is 100.`

* ``ml_screenshot_analysis``

This dict contain a list of resolutions. Each screenshot resolution has a ``content`` and a ``mask`` object, each of them having three entries: ``images`` and ``textblock`` are values (0-100) that indicate how similar the images and the detected texts are, with the ``overall`` value being their average. The resolution also contains an ``overall`` entry which is the average of the ``content`` and ``mask``'s ``overall``s.

```json
  1920: {
      content: {
        images: 0,
        overall: 0,
        textblock: 0
      },
      mask: {
        images: 0,
        overall: 0,
        textblock: 0
      },
      overall: 0,
  }
```

###### report

The report is the essential part of the log object. It summarizes all divergences.

It has the following fields:

```json
    crawl_divergence: 16.279069767441857,
    ml_screenshot_divergence: 0,
    overall_divergence: 1.6279069767441858,
    raw_screenshot_diff_pixels: 0,
    raw_screenshot_divergence: 0,
    raw_screenshot_mse: 0,
    raw_screenshot_ssim: 100,
    unique_baseline_pages: [],
    unique_updated_pages: []
```

* ``crawl_divergence`` is computed as the maximum of all the pages' average console and network log divergences
* ``raw_screenshot_diff_pixels`` similarly, the maximum of all the pages' different pixels divergence
* ``raw_screenshot_mse`` similarly for the mse divergence
* ``raw_screenshot_ssim`` similarly for the structural similarity index value
* ``raw_screenshot_divergence`` it is the maximum of the average of the three values above, considering all crawled pages 
* ``ml_screenshot_divergence`` the maximum divergence value from all pages, where the value is the ``overall`` divergence 
* ``overall_divergence`` this is the overall divergence that FRED outputs, computed as a weighted sum of ``crawl_divergence``, ``raw_screenshot_divergence``, and ``ml_screenshot_divergence``(if enabled)
* ``unique_baseline_pages`` and ``unique_updated_pages`` list pages (addresses) that were unique, either on the baseline site or on the updated site. If any of these entries contain values, then a maximum divergence will be reported, because that means that the baseline-updated sites are not identical as they don't serve the same pages. 

Note, FRED runs under the assumption that if a page is very different baseline-to-updated, then this is the value reported back to the user, even if all other pages might be identical. So, we always compute the **maximum divergence for all pages for all resolutions**.
