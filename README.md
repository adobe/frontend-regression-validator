# Frontend-Regression-Validator(FRED)

FRED is an opensource visual regression tool used to compare two instances of a website. FRED is responsible for automatic visual regression testing, with the purpose of ensuring that functionality is not broken on instances that are being upgraded. The main idea is that detecting breaking changes can be achieved through visual comparison of screenshots between a baseline and an upgraded instance of a website.
Using visual regression is a good idea for the “Zero Cost Upgrade” process. Mainly we are only updating/upgrading the underlying framework and we are not pushing any changes to the custom code or content. Thus, FRED expects both instances (the baseline and the upgraded one) to be visually identical.
However, this is not always the case and we draw the attention to dynamic content, which stands for content that can change based on random variables, timers etc. We include here random newsfeeds, commercials and custom designed image-based effects e.g. a carrousel that randomly choses the first image) and many others.

Use FRED if you need:
* Screenshot comparision
* Automatic layout/content verification

## Setup
In order to use this software, simply run:
```
	git clone https://github.com/adobe/frontend-regression-validator.git
	pip install -r requirements.txt
	cd fred/
	python3 run.py
```
This will launch a Flask API server that answers to requests. In order to view the API readme, follow this link[LINK]. If you want to view the UI and access the API this way, then navigate to http://localhost:5000/static/submit.html. To use the UI, you have to fill in the forms:

| Baseline URL                    | URL of the baseline instance              |
| Updated URL                     | URL of the updated instance               |
| Max depth                       | Max depth to crawl links in the two pages |
| Max URLs                        | Max URLs to save for regression testing   |
| Prefix                          | Prefix to append to output directories    |
| Baseline URL username(optional) | Username for the baseline instance        |
| Baseline URL password(optional) | Password for the baseline instance        |
| Updated URL username(optional)  | Username for the updated instance         |
| Updated URL password(optional)  | Password for the updated instance         |