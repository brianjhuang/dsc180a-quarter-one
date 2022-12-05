#!/usr/bin/env python
import os

class PackageConfig(object):
    """
    Static class that contains the configuration of our package.
    """

    # Config for run.py
    ENABLE_LOGGING = True
    OVERRIDE_WEIGHTS = False
    TRAIN_MODEL = False

    # RELATIVE PATH FOR AUDIT DATA Note: The path your run.py is executing look something like this:
    # User/repo/run.py, so final path would be User/repo/relative_path
    DATA_FILE_NAME = "video_ids"
    AUDIT_DATA = f"data/raw_data/{DATA_FILE_NAME}.txt"
    DOWNLOADED_AUDIT_DATA = "data/test_data/"
    TEST_AUDIT_DATA = "test/testdata/test_user_watch_history_videos.json"
    WEIGHTS = "src/pseudoscientificvideosdetection/models/pseudoscience_model_final.hdf5"

    #CLEAN UP PATHS
    DATA = 'data/test_data/'
    LOGS = 'logs/'
    ALL_RUNS = 'runs/all_runs/'
    TEST_RUNS = 'runs/test_runs/'
    COMMENTS = 'videosdata/comments/'
    TRANSCRIPTS = 'videosdata/transcript/'

    CLEAN_PATHS = [DATA, LOGS, ALL_RUNS, TEST_RUNS, COMMENTS, TRANSCRIPTS]

    # API KEYS
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_DATA_API_KEY")