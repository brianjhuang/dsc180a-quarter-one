#!/usr/bin/env python


import sys
import os
import json
import time
import logging
import logging.handlers

# Config files
from config.package_config import PackageConfig

# Allow us to import scripts from src without specifying src
sys.path.insert(0, 'src')

# Imports from source folder
from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining

if PackageConfig.ENABLE_LOGGING:
    # Logging variables
    totalLogs = len(os.listdir('logs'))
    logFileName = f'logs/log_{totalLogs}.txt'


    # Set up the settings to log information as we run our build pipeline
    logging.basicConfig(filename=logFileName, 
                filemode='a', 
                level=logging.INFO,
                datefmt='%H:%M:%S',
                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')

def getCommentDataFromIds(videoIds):
    return

def loadData():
    '''
    This function will load data in from 
    mongoDB so we can train our classifier.
    '''
    dataset = DatasetUtils()

    return dataset

def fineTuneBranches(dataset):
    '''
    This function will allow us 
    to fine tune each of the branches of 
    our 'fusion network'.

    NOTE: This is only called once, afterwards it reloads pre-trained saved values.
    '''
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
    return

def trainClassifier(dataset, override = PackageConfig.OVERRIDE_WEIGHTS):
    '''
    Train our classifier
    '''
    # Create our classifier (which also creates our feature embeddings)
    classifierTrainingObject = ClassifierTraining(dataset_object=dataset)
    if override or not os.path.exists('src/pseudoscientificvideosdetection/models/pseudoscience_model_final.hdf5'):
        classifierTrainingObject.train_model()
    else:
        logging.info(f"Model has already been trained: {os.path.exists('src/pseudoscientificvideosdetection/models/pseudoscience_model_final.hdf5')}")

    return

def loadModel():
    '''
    Load in our model weights.
    '''
    return

def predict():
    '''
    Make our prediction
    '''
    return

def visualization():
    '''
    Generate our visualizations
    '''

def main(targets):
    
    if 'all' in targets:
        # LOAD OUR DATA
        logging.info("Attempting to load in data...")
        data = loadData()
        logging.info("Data load succesful!")

        # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
        logging.info("Attempted to fine tune fasttext...")
        start = time.time()
        fineTuneBranches(data)
        end = time.time()
        logging.info(f"Completed fine tuning in {end - start} seconds.")

        # TRAIN OUR CLASSIFIER
        logging.info("Attempted to train classifier...")
        start = time.time()
        trainClassifier(data, True)
        end = time.time()
        logging.info(f"Completed training in {end - start} seconds.")

    if 'test' in targets:
        # LOAD OUR DATA
        logging.info("Attempting to load in data...")
        data = loadData()
        logging.info("Data load succesful!")

        # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
        logging.info("Attempted to fine tune fasttext...")
        start = time.time()
        fineTuneBranches(data)
        end = time.time()
        logging.info(f"Completed fine tuning in {end - start} seconds.")

        # TRAIN OUR CLASSIFIER
        logging.info("Attempted to train classifier...")
        start = time.time()
        trainClassifier(data)
        end = time.time()
        logging.info(f"Completed training in {end - start} seconds.")
        return
    
    # LOAD OUR DATA
    if 'data' in targets:
        logging.info("Attempting to load in data...")
        data = loadData()
        logging.info("Data load succesful!")

    # FINE TUNE BRANCHES AND GET FASTTEXT EMBEDDINGS
    if 'fineTune' in targets:
        logging.info("Attempted to fine tune fasttext...")
        start = time.time()
        fineTuneBranches(data)
        end = time.time()
        logging.info(f"Completed fine tuning in {end - start} seconds.")

    # TRAIN OUR CLASSIFIER
    if 'train' in targets:
        logging.info("Attempted to train classifier...")
        start = time.time()
        trainClassifier(data)
        end = time.time()
        logging.info(f"Completed training in {end - start} seconds.")

    return

if __name__ == '__main__':
    targets = [target.lower() for target in sys.argv[1:]]
    main(targets)
    logging.info('END OF BUILD.\n')