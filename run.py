#!/usr/bin/env python

import sys
import os
import json
import time
import logging
import logging.handlers
from pymongo.errors import ServerSelectionTimeoutError

# Config files
from config.package_config import PackageConfig

# Allow us to import scripts from src without specifying src
sys.path.insert(0, 'src')

# Imports from source folder
from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining
from pseudoscientificvideosdetection.PseudoscienceClassifier import PseudoscienceClassifier
from youtubehelpers.YouTubeVideoDownloader import YouTubeVideoDownloader

if PackageConfig.ENABLE_LOGGING:
    # Logging variables
    totalLogs = len(os.listdir('logs'))
    logFileName = 'logs/log_{0}.txt'.format(totalLogs)


    # Set up the settings to log information as we run our build pipeline
    logging.basicConfig(filename=logFileName, 
                filemode='a', 
                level=logging.INFO,
                datefmt='%H:%M:%S',
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')

def loadData():
    '''
    This function will load data in from 
    mongoDB so we can train our classifier.
    '''

    print("Attempting to load in data...")
    logging.info("Attempted to load in data...")
    try:
        dataset = DatasetUtils()
        print("Data load succesful!")
        logging.info("Data load succesful!")
    except ServerSelectionTimeoutError as e:
        print(f"Data load failed with exception: \n {e}")
        logging.info(f"Data load failed with exception: \n {e}")
        print("Did you launch your MongoDB server?")
        exit(1)

    return dataset

def fineTuneBranches(dataset):
    '''
    This function will allow us 
    to fine tune each of the branches of 
    our 'fusion network'.

    NOTE: This is only called once, afterwards it reloads pre-trained saved values.
    '''

    print("Attempting to fine tune fasttext...")
    logging.info("Attempted to fine tune fasttext...")
    start = time.time()

    try:
        featureEngineeringModels = FeatureEngineeringModels(dataset_object=dataset)

        # Generate Video Snippet fastText input features
        featureEngineeringModels.prepare_fasttext_data(model_type='video_snippet')

        # Fine-tune a fastText model for Video Snippet
        featureEngineeringModels.finetune_model(model_type='video_snippet')

        # Generate Video Tags fastText input features
        featureEngineeringModels.prepare_fasttext_data(model_type='video_tags')

        # Fine-tune a fastText model for Video Tags
        featureEngineeringModels.finetune_model(model_type='video_tags')

        # Generate Video Transcript fastText input features
        featureEngineeringModels.prepare_fasttext_data(model_type='video_transcript')

        # Fine-tune a fastText model for Video Transcript
        featureEngineeringModels.finetune_model(model_type='video_transcript')

        # Generate Video Comments fastText input features
        featureEngineeringModels.prepare_fasttext_data(model_type='video_comments')

        # Fine-tune a fastText model for Video Comments
        featureEngineeringModels.finetune_model(model_type='video_comments')

        end = time.time()
        print(f"Completed fine tuning in {end - start} seconds.")
        logging.info(f"Completed fine tuning in {end - start} seconds.")

    except Exception as e:
        end = time.time()
        print(f"Failed fine tuning in {end - start} seconds.")
        logging.info(f"Failed fine tuning in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return

def trainClassifier(dataset, override = PackageConfig.OVERRIDE_WEIGHTS):
    '''
    Train our classifier
    '''        
    
    print("Attempting to train classifer...")
    logging.info("Attempted to train classifier...")
    start = time.time()
    try:
        # Create our classifier (which also creates our feature embeddings)
        classifierTrainingObject = ClassifierTraining(dataset_object=dataset)
        if override or not os.path.exists(PackageConfig.WEIGHTS):
            classifierTrainingObject.train_model()
        else:
            print(f"Model has already been trained weights located at: {PackageConfig.WEIGHTS}")
            logging.info(f"Model has already been trained weights located at: {PackageConfig.WEIGHTS}")

        end = time.time()
        print(f"Completed training in {end - start} seconds.")
        logging.info(f"Completed training in {end - start} seconds.")

    except Exception as e:
        end = time.time()
        print(f"Failed training in {end - start} seconds.")
        logging.info(f"Failed training in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return classifierTrainingObject

def loadModel():
    '''
    Load in our model weights.
    '''
    print("Loading model weights...")
    logging.info("Loading model weights...")
    start = time.time()

    try:
        model = PseudoscienceClassifier()
        end = time.time()
        print(f"Completed loading in {end - start} seconds.")
        logging.info(f"Completed loading in {end - start} seconds.")

    except Exception as e:
        end = time.time()
        print(f"Failed loading in {end - start} seconds.")
        logging.info(f"Failed loading in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return model

def loadAuditData():
    '''
    Load the list of IDs for our Audit Data.
    
    For Test, this function does not execute.
    '''

    start = time.time()
    print("Loading our audit video IDs...")
    logging.info("Loading our audit video IDs...")
    try:
        with open(PackageConfig.AUDIT_DATA) as fh:
            video_ids = [line.strip("\n") for line in fh.readlines()]

        end = time.time()
        print(f"Loaded our video IDs in {end - start} seconds")
        logging.info(f"Loaded our video IDs in {end - start} seconds")

    except Exception as e:
        end = time.time()
        print(f"Failed loading in {end - start} seconds.")
        logging.info(f"Failed loading in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return video_ids

def downloadAuditData(video_ids):
    '''
    Download our data with our IDs

    For Test, this function does not execute.
    '''
    
    start = time.time()
    logging.info("Downloading our audit videos...")
    print("Downloading our audit videos")

    try: 
        ytDownloader = YouTubeVideoDownloader()

        if (os.path.exists(PackageConfig.DOWNLOADED_AUDIT_DATA + f'/{PackageConfig.DATA_FILE_NAME}.json')):
            print("Data already downloaded!")
            logging.info("Data already downloaded!")
            return json.load(open(PackageConfig.DOWNLOADED_AUDIT_DATA + f'/{PackageConfig.DATA_FILE_NAME}.json'))['items']

        videos = []
        for id in video_ids:
            logging.info(f"Downloading {id}...")
            print(f"Downloading {id}...")

            video_details = ytDownloader.download_video(video_id=id)
            videos.append(video_details)

            if len(video_details) > 0:
                print(f"Downloaded {id} successfully")
                logging.info(f"Downloaded {id} successfully")
            else:
                print(f"Download for {id} failed")
                logging.info(f"Download for {id} failed")

        end = time.time()
        print(f"Downloaded our videos in {end-start} seconds")
        logging.info(f"Downloaded our videos in {end-start} seconds")

        # Write our downloaded data
        with open(PackageConfig.DOWNLOADED_AUDIT_DATA + f'/{PackageConfig.DATA_FILE_NAME}.json', 'w+') as f:
            f.write('{ \"items\":[')
            for i in range(len(videos)):
                video = videos[i]
                if i == len(videos) - 1:
                    f.write(json.dumps(video) + '\n')
                else:
                    f.write(json.dumps(video) + ',\n')
            f.write(']}')

    except Exception as e:
        end = time.time()
        print(f"Failed downloading in {end - start} seconds.")
        logging.info(f"Failed downloading in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return videos

def predict(video_details, model, isTest = False):
    '''
    Make our prediction
    '''
    start = time.time()
    try:
        prediction, confidence_score = model.classify(video_details=video_details)
    except Exception as e:
        end = time.time()
        print(f"Failed prediction(s) in {end - start} seconds.")
        logging.info(f"Failed prediction(s) in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)
    
    if isTest:
        return {
        "video_id" : video_details['contentDetails']['videoId'],
        "prediction" : prediction,
        "confidence_score": confidence_score,
        }

    return {
        "video_id" : video_details['id'],
        "prediction" : prediction,
        "confidence_score": confidence_score,
    }

def writeResults(results, isTest = False):
    '''
    Write our results to our runs folder.

    test_runs - stores run.py test runs
    all_runs - stores run.py all runs
    '''

    start = time.time()

    try:
        if isTest:
            # Run variables
            totalRuns = len(os.listdir('runs/test_runs'))
            with open('runs/test_runs/test_run_{0}.json'.format(totalRuns), 'w+') as f:
                f.write('{ \"items\":[')
                for i in range(len(results)):
                    result = results[i]
                    if i == len(results) - 1:
                        f.write(json.dumps(result) + '\n')
                    else:
                        f.write(json.dumps(result) + ',\n')
                f.write(']}')
            return

        # Run variables
        totalRuns = len(os.listdir('runs/all_runs'))
        with open('runs/all_runs/run_{0}.json'.format(totalRuns), 'w+') as f:
            f.write('{ \"items\":[')
            for i in range(len(results)):
                result = results[i]
                if i == len(results) - 1:
                    f.write(json.dumps(result) + '\n')
                else:
                    f.write(json.dumps(result) + ',\n')
            f.write(']}')

    except Exception as e:
        end = time.time()
        print(f"Failed writing file in {end - start} seconds.")
        logging.info(f"Failed writing file in {end - start} seconds.")
        print(f"Caught exception: \n {e}")
        logging.info(f"Caught exception: \n {e}")
        exit(1)

    return

def clean(paths = PackageConfig.CLEAN_PATHS):
    '''
    Clean our folder
    '''
    gitIgnore = '.gitignore'
    for path in paths:
        print(f"Cleaning {path}...")
        logging.info(f"Cleaning {path}...")
        for file in os.listdir(path):
            try:
                if file != gitIgnore:
                    os.remove(path + file)
                    print(f"Removed: {path + file}...")
                    logging.info(f"Removed: {path + file}...")
            except PermissionError as e:
                if file != gitIgnore:
                    # Clean out values
                    for innerFile in os.listdir(path + file):
                        if file != gitIgnore:
                            os.remove(path + file + "/" + innerFile)

                    os.rmdir(path + file)
                    print(f"Removed directory: {path + file}...")
                    logging.info(f"Removed directory: {path + file}...")
            except Exception as e:
                print(f"Encountered error: {e}")
                logging.info(f"Encountered error: {e}")
                exit(1)
            
        print(f"Finished cleaning {path}...")
        logging.info(f"Finished cleaning {path}...")

    

def main(targets):
    
    if 'all' in targets:
        # LOAD OUR DATA
        data = loadData()

        # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
        fineTuneBranches(data)

        # TRAIN OUR CLASSIFIER
        trainClassifier(data)

        # LOAD OUR MODEL
        model = loadModel()

        # LOAD OUR AUDIT DATA
        video_ids = loadAuditData()

        # DOWNLOAD OUR AUDIT DATA
        videos = downloadAuditData(video_ids)

        # MAKE PREDICTIONS
        print("Running inference...")
        logging.info("Running inference...")

        start = time.time()
        predictions = [predict(video_details = video, model=model) for video in videos]
        end = time.time()

        print(f"Completed inference in {end-start} seconds")
        logging.info(f"Completed inference in {end-start} seconds")

        print("Writing to file...")
        logging.info("Writing to file...")

        # WRITE TO RUNS FOLDER
        writeResults(predictions)

        print("Completed pipeline...")
        logging.info("Completed pipeline...")

        return predictions

    if 'test' in targets:
        # LOAD OUR DATA
        data = loadData()

        # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
        fineTuneBranches(data)

        # TRAIN OUR CLASSIFIER
        trainClassifier(data)

        # LOAD OUR MODEL
        model = loadModel()

        # LOAD OUR AUDIT DATA
        start = time.time()
        logging.info("Loading our audit video IDs...")
        video_ids = []
        end = time.time()
        logging.info(f"Loaded our video IDs in {end - start} seconds")

        # DOWNLOAD OUR AUDIT DATA
        start = time.time()
        logging.info("Downloading our audit video...")
        
        with open(PackageConfig.TEST_AUDIT_DATA) as fh:
            response = json.load(fh)
            videos = response['items']
        end = time.time()
        logging.info(f"Downloaded our videos in {end-start} seconds")

        # MAKE PREDICTIONS
        print("Running inference...")
        logging.info("Running inference...")

        start = time.time()
        predictions = [predict(video_details = video, model=model, isTest = True) for video in videos]
        end = time.time()

        print(f"Completed inference in {end-start} seconds")
        logging.info(f"Completed inference in {end-start} seconds")

        print("Writing to file...")
        logging.info("Writing to file...")

        # WRITE TO RUNS FOLDER
        writeResults(predictions, isTest = True)

        print("Completed pipeline...")
        logging.info("Completed pipeline...")

        return predictions

    if 'data' in targets:
         # LOAD OUR DATA
        data = loadData()

    if 'finetune' in targets:
        # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
        fineTuneBranches(data)

    if 'train' in targets:
        # TRAIN OUR CLASSIFIER
        trainClassifier(data)

    if 'audit' in targets:
        # LOAD OUR AUDIT DATA
        video_ids = loadAuditData()

        # DOWNLOAD OUR AUDIT DATA
        videos = downloadAuditData(video_ids)

        # LOAD OUR MODEL
        model = loadModel()

        # MAKE PREDICTIONS
        print("Running inference...")
        logging.info("Running inference...")

        start = time.time()
        predictions = [predict(video_details = video, model=model) for video in videos]
        end = time.time()

        print(f"Completed inference in {end-start} seconds")
        logging.info(f"Completed inference in {end-start} seconds")

        print("Writing to file...")
        logging.info("Writing to file...")

        # WRITE TO RUNS FOLDER
        writeResults(predictions)

        print("Completed pipeline...")
        logging.info("Completed pipeline...")
    
    if 'clean' in targets:
        clean()



    return

if __name__ == '__main__':
    targets = [target.lower() for target in sys.argv[1:]]

    main(targets)
    logging.info('END OF BUILD.\n')