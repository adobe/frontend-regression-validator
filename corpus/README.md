# Create a new corpus
If you want to create a new dataset and retrain our models, we have provided the scripts in this directory. In order to use it, follow the next steps:\
* Create a file named `full_website_list` containing all the websites that you want to scrape, take screenshots and mask
* Run `python3 parallel_pipelines.py` with the parameter `--num-processes`, which specifies how many processes are running in parallel to scrape the websites in the list and `--output-folder`, which specifies where to save the results

Afterwards, you need to run the scripts in the `model` directory to finish building the dataset.