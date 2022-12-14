# DSC 180A - Quarter One Reproduction Project
### Brian Huang
#### Reproduction of "It is just a flu": Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations (Papadamou et al. 2020)
## Overview
The following project is a reproduction of "It is just a flu": Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations (Papadamou et al. 2020). It was done as the Quarter 1 Project for DSC 180A under Professor Stuart Geiger. 

Given the scope of the quarter, the goal of this project was to reproduce the classifier in the original model and run it on a mini-audit of YouTube search results. In the audit, they used COVID-19 search terms and flat-earth search terms, so the mini dataset was constructed by taking the first ten results of the following search terms: `['flat earth', 'covid19', 'vaccination']`. These terms were consistent with the original terms. Another set of conspiracies was added by including `'Alex Jones'` (a well known conspiracy theorist) as another search term. The top six videos were taken for this search.

The classifier was run on this mini-dataset and the following conclusions were made:

Note: Dark mode makes the labels hard to read, please try using light mode instead.

![plot](https://github.com/brianjhuang/dsc180a-quarter-one/blob/master/plots/results.png)

In the original audit, it was shown that search results had MORE pseudoscience videos populated. In my results, it was shown that pseudoscience videos populated search results much less.

One thing to note about this result is that many of the conspiracy videos that were present in the audit when it was first done have now been taken down from YouTube. In fact only 86% of the original dataset comments were still availabe on YouTube. This may have skewed the classifier more towards predicting scientific vs pseudoscience. Another thing to note, in regards to model accuracy, is that the dataset we audit on is labelled by myself, and many of the videos are news videos covering rising pseudoscience, not actual pseudoscience. More details on the findings, results, and methods are detailed in the final report submitted on GradeScope.

## Disclaimer to the Graders

The project relies heavily on three things: The dataset, MongoDB to store the dataset, and the model weights. Because of this, it is quite difficult to launch on DSMLP.

- While you could get the dataset, the comments are not provided due to YouTube TOS. Only comment IDs may be shared, meaning all comments need be scraped on your own. This takes multiple days.
- Model weights can be shared, but even compressed take around 25gb.
- Test data could be made, but the library provided by the original researchers heavily relies on fetching data from a MongoDB instance. This is difficult to replicate in DSMLP. 

These changes have been brought up with Professor Rampure. This repo can be reproduced locally, and if model weights are needed please reach out to bjh009@ucsd.edu. I would be happy to hop on Zoom as well and demonstrate the repository in it's working state or provide any other assistance in the grading process. For now I've been told to detail how the project would be reproduced, without providing model weights. The original dataset is provided, but requires access request approval from the original researchers. Please keep that in mind as you work with this repository as reproducing this project takes a bit of waiting.

## Installation
Follow the steps below to install and configure all prerequisites for both the training and usage of the Pseudoscientific Content Detection Classifier (Part 1) and conduct the 'mini-audit' (Part 2).

**Create and activate Python >= 3.6 Virtual Environment**
```bash
python3 -m venv virtualenv

source virtualenv/bin/activate
```
**Install required packages**
```bash
pip install -r requirements.txt
```

### Install MongoDB
To store the metadata of YouTube videos, as well as for other information we use MongoDB. Install MongoDB on your own system or server using:

- **Ubuntu:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/">here</a>.
- **Mac OS X:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/">here</a>.


#### MongoDB Graphical User Interface:
The researchers suggested the use of <a href="https://robomongo.org/">Robo3T</a> as a user interface for interacting with your MongoDB instance.

### Additional Requirements

#### Install the youtube-dl package
```bash
pip install youtube-dl
```
**Make use of ```youtube-dl``` wisely, carefully sending requests so that you do not spam YouTube with requests and get blocked.

#### Install Google APIs Client library for Python
This is the library utilized to call the YouTube Data API from Python
```bash
pip install --upgrade google-api-python-client
```

### HTTPS Proxies
The codebase uses HTTPS Proxies for multiple purposes: 
- For downloading the transcripts of YouTube videos; and 
- The YouTube Recommendation Algorithm Audit Framework uses an HTTPS Proxy for each one of the user profiles and browser instances that it maintains. 
  This is mainly to ensure that all User Profiles used in our framework have the same geolocation and avoid changes to our results due to geolocation personalization.

You can either use your own HTTPS Proxies or buy some online and set them in the following files:
- ```youtubeauditframework/userprofiles/info/user_profiles_info.json```: Includes the HTTPS Proxies used to simulate distinct logged-in user profiles accessing YouTube from specific geolocations. 
  Preferrably, according to our Audit framework, all HTTPS Proxies set in this file MUST be from similar locations (e.g., "US-San Fransisco-California"). 
- ```youtubehelpers/config/YouTubeAPIConfig.py```: Includes the HTTPS Proxies used to download the transcript of YouTube videos using ```youtube-dl```.

**Alternative to HTTPS Proxies**

As this audit is a smaller individual scale audit, HTTP proxies are not requried. In the `YouTubeAPIConfig.py` file, you can change the following from:
```python
# HTTPS Proxies for Video Transcript download
HTTPS_PROXIES_LIST = [
    # 'HOST:PORT',
    # 'HOST:PORT',
    # 'HOST:PORT',
    # 'HOST:PORT',
    # 'HOST:PORT',
]
```
to this:
```python
# HTTPS Proxies for Video Transcript download
HTTPS_PROXIES_LIST = [
    'localhost:80'
]
```

### YouTube Data API
Our codebase uses the YouTube Data API to download video metadata and for many other purposes like searching YouTube. 
Hence, it is important that you create an API key for the YouTube Data API and set it in the configuration files of our codebase.
You can enable the YouTube Data API for your Google account and obtain an API key following the steps <a href="https://developers.google.com/youtube/v3/getting-started">here</a>.

Once you have a **YouTube Data API Key**, please set the ```YOUTUBE_DATA_API_KEY``` variable in your environment:

You can do so but going to your home directory and doing something like so:

```
nano .bash_profile
```

Inside your bash profile, you can go ahead and set this at the topL

```
# YOUTUBE API KEY
export YOUTUBE_DATA_API_KEY="YOUR_API_KEY"
```

The following tutorials cover how to do this as well:

https://www.youtube.com/watch?v=5iWhQWVXosU&t=1s (Mac/Linux)

https://www.youtube.com/watch?v=IolxqkL7cD8 (Windows)

Now within Python you can access your API key by doing the following:
```
import os

os.environ.get("YOUTUBE_DATA_API_KEY")
```

If you opt to not store your values in your environment variable, please modify the ```YOUTUBE_DATA_API_KEY``` variable in the following files:
- ```youtubehelpers/config/YouTubeAPIConfig.py```
- ```youtubeauditframework/utils/YouTubeAuditFrameworkConfig.py```
- ```config/package_config.py```

# Part 1: Detection of Pseudoscientific Videos
A deep learning model is used to detect pseudoscientific YouTube videos. 
As described in the paper, to train and test the model use the dataset available <a href="https://zenodo.org/record/4558469#.YDlltl37Q6F">here</a>.

## 1.1. Classifier Architecture
![Model Architecture Diagram](https://github.com/kostantinos-papadamou/pseudoscience-paper/blob/main/classifier/architecture/model_architecture.png)
(Papadamou et al. 2020)

**Note:** For more information about the classifier, please visit the <a href = https://github.com/kostantinos-papadamou/pseudoscience-paper> original repository. </a>

## 1.2. Prerequisites
### 1.2.1.  Download pre-trained fastText word vectors that we fine-tune during feature engineering on our dataset: 
```bash
cd pseudoscientificvideosdetection/models/feature_extraction

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

unzip wiki-news-300d-1M.vec.zip
```

### 1.2.2. Create MongoDB Database and Collections

Before running anything, please start your MongoDB instance:
```bash
brew services start mongodb-community@6.0
```

To stop the service please do:
```bash
brew services stop mongodb-community@6.0
```

When the service is started, you can connect to your local DB using Robo3T's GUI. Find information on your local service by using:
```bash
mongosh
```

1. Create a MongoDB database called: ```youtube_pseudoscience_dataset``` either using Robo3T GUI or from the terminal. 

2. Create the following MongoDB collections under the ```youtube_pseudoscience_dataset``` database that you just created:
- ```groundtruth_videos```
- ```groundtruth_videos_comments```
- ```groundtruth_videos_transcripts```

I would recommend you do this in Robo3T GUI. You can right click the database you created earlier and create a new collection. Import the JSON data for each of the data groups into their correpsonding collections.

**Note:** Although the original dataset comes with the data for ```groundtruth_videos``` and ```groundtruth_videos_transcripts```, it does not have the original comments. To train and run the model in it's entirety, you will need to get a YouTube API Key and download those values yourself.

In the notebooks, a `ScrapingComments.ipynb` is included. This is also included in `src/dataset/FetchComments.py` which can be run as a script as opposed to within the notebook. I'd recommend doing it through the notebook, however, as it makes it easier to check in on progress or halt the requests. 

If it is your first time running either the notebook or script, please create `fetchedComments/raw_comment_responses.json`. It should look like so:
```json
{
  "comments" : []
}
```

When scraping is completed, you can clean and output the final `groundtruth_videos_comments.json` using `CleanCommentsData.ipynb` or the methods in `src/dataset/FetchComments.py`. The process takes around five days.

If you would like to avoid training, please reach out to `bjh009@ucsd.edu` for model weights and parameters. If you are just loading in the model weights rather than downloading the data and training yourself, please still set up MongoDB and replace the ```groundtruth_videos_comments``` collection with the empty json.

You can also skip model training entirely if you have the weights by changing `TRAIN_MODEL` to False in `config/package_config.py`.

Note: This is default **FALSE**.

If you would like to change where files save or any other settings in the model, please refer to `config/package_config.py`.

## 1.3. Training the Classifier
Before running anything, please start your MongoDB instance:
```bash
brew services start mongodb-community@6.0
```

To stop the service please do:
```bash
brew services stop mongodb-community@6.0
```

To see if your service is running you can do:
```bash
brew services list
```

To see URI/ConfigureL:
```bash
mongosh
```

Additional Info:
```bash
mongotop
```

Install Page/Help:

https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/

To train the model, `run.py all` will run the neccesary steps to load data, train fasttext embeddings, and make predictions. Please ensure that you have all the following files in the right place.

**If you chose to collect the data:**
- Ensure that both `data/raw_data/video_ids.txt` and `data/raw_data/video_labels.txt` are in their right folder. These contain the video links and data used for the mini-audit.
- Check that the filepaths in `config/package_config.py` are the correct filepaths that you want values to save in.

**If you chose to load weights**:
- Ensure that the same steps in the snippet above are followed.
- In `src/dataset/data` ensure that `feature_engineering_models_data` contains four files ending in `train_data.txt` and `input_features` contains four files ending in `embeddings.p`. `data` should also contain four files ending with `features.p`.
- In `src/pseudoscientificvideosdetection/models/feature_extraction` ensure that there are four files ending in `.bin` (these are our model weights) and one ending in `.vec` (the fasttext pre-trained vectors loaded in earlier). 
- In `src/pseudoscientificvideosdetection/models` you should find `pseudoscience_model_final.hdf5`

## 1.4. Using the Classifier and Performing the Audit

`run.py all` handles the predictions and saves them to the `runs/all_runs` folder. You can see the results of the predictions save in a JSON file there.

`run.py test` handles predictions on a pred-downloaded set of YouTube data in `test/test_data`. It saves values to `runs/test_runs`.

If you would like to run the audit on YOUR OWN DATA:

Change the values within `data/raw_data`. Include your own list of YouTube videoIds and labels in their corresponding files. If you would like to add your own files but run inference with that data, please change the following parameters in `package_config.py`:
```python
DATA_FILE_NAME = "video_ids"
AUDIT_DATA = f"data/raw_data/{DATA_FILE_NAME}.txt"
DOWNLOADED_AUDIT_DATA = "data/test_data/"
```

### Citations
```latex
@article{papadamou2020just,
    title={'It is just a flu': Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations},
    author={Papadamou, Kostantinos and Zannettou, Savvas and Blackburn, Jeremy and De Cristofaro, Emiliano and Stringhini, Gianluca and Sirivianos, Michael},
    journal={arXiv preprint arXiv:2010.11638},
    year={2020}
}
```
