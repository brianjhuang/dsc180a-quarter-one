import os
import json
import time
import logging
import random

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from googleapiclient.discovery import build
import googleapiclient.errors

from config.package_config import PackageConfig

class FetchComments:
    """
    This helper class allows us to fetch 
    the comments from comment ids when 
    we are provided a dataset from somewhere else.
    """

    def __init__(self):
        # Get our API key from our enviroment variables
        self.api_key = os.environ.get('YOUTUBE_DATA_API_KEY')

        # Create our API request object
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

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

    def write_json(self, comment_response, filename='fetchedComments/raw_comment_responses.json'):
        '''
        Given a response for a single comment, write that comment to our json file.
        '''
        with open(filename,'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["comments"].append(comment_response)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)

    def get_comments(self, videoIds, delay = 5):
        '''
        Given a set of video IDs, make requests to start getting IDs.
        NOTE: THIS WILL TAKE A WHILE OT RUN
        '''

        # Load in all the requests we have completed so far
        comments = json.load(open("fetchedComments/raw_comment_responses.json"))
        completed_requests = [list(comment.keys())[0] for comment in comments['comments']]

        logging.info("\n")
        p_bar = tqdm(range((len(videoIds))))
        already_completed = 0
        total_time = 0
        previous_time = 0
        previous_id = "START"

        for number in p_bar:
            start = time.time()
            id = videoIds[number]
            p_bar.set_description(f'Working on {id}... Last Write: {previous_time} seconds for {previous_id}')
            if id in completed_requests:
                logging.info(f"{id} already scraped")
                already_completed += 1
                continue

            try:
                video_response=self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=id,
                    maxResults = 100
                ).execute()
                time.sleep(delay)
            except googleapiclient.errors.HttpError as e:
                logging.info(f"Encountered {e}")
                video_response = {"noComments":[]}

            self.write_json({id : video_response})
            completed_requests.append(id)
            logging.info(f"Wrote {id} to JSON")
            end = time.time()
            previous_time = end-start
            previous_id = id
            total_time += (previous_time)

    def clean_comments(self):
        data = json.load(open("fetchedComments/raw_comment_responses.json"))
        
        comment_data = data['comments']
        videoIds = [list(comment.keys())[0] for comment in comment_data]

        def getCommentsFromVideo(video):
            '''
            Given a video comment(s), return it's values formatted
            for mongo DB.

            No comments or a length of zero returns nothing.
            '''
            comments = list(video.values())[0]
            if 'noComments' in comments.keys():
                return
                # return pd.DataFrame({
                #     'videoId': video_id,
                #     'textDisplay': "",
                #     'textOriginal': "",
                #     'authorDisplayName': "",
                #     'authorProfileImageUrl': "",
                #     'authorChannelUrl':"",
                #     'authorChannelId':{'value':""},
                #     'canRate':False,
                #     'viewerRating':"",
                #     'likeCount':0,
                #     'publishedAt':"",
                #     'updatedAt':"",
                #     'commentId':""
                # })
            else:
                comments = comments['items']

            if len(comments) == 0:
                return
                # return pd.DataFrame({
                #     'videoId': video_id,
                #     'textDisplay': "",
                #     'textOriginal': "",
                #     'authorDisplayName': "",
                #     'authorProfileImageUrl': "",
                #     'authorChannelUrl':"",
                #     'authorChannelId':{'value':""},
                #     'canRate':False,
                #     'viewerRating':"",
                #     'likeCount':0,
                #     'publishedAt':"",
                #     'updatedAt':"",
                #     'commentId':""
                # })
            
            comment_data = []
            for comment in comments:
                datum = comment['snippet']['topLevelComment']
                comment_id = datum['id']
                comment_info = datum['snippet']
                comment_info['commentId'] = comment_id
                comment_data.append(comment_info)
            
            return pd.DataFrame(comment_data)
    
        def loadVideos(data):
            '''
            Given data, concatenate all the dataframes together
            '''
            dataframes = []
            for video in data:
                dataframes.append(getCommentsFromVideo(video))

            return pd.concat(dataframes, ignore_index=True)

        dataFull = loadVideos(comment_data)
        # Join the DF together and group it
        commentsGrouped = pd.DataFrame(dataFull[['videoId', 'textOriginal']].groupby('videoId')['textOriginal'].apply(list)).reset_index()
        commentsGrouped = commentsGrouped.rename(columns={"textOriginal":"comments", "videoId":"id"}).reindex(columns = ['comments', 'id'])

        commentsJson = commentsGrouped.to_json(orient = 'records')
        with open("fetchedComments/groundtruth_videos_comments.json", "w") as f:
            f.write(commentsJson)
    

