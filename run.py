import sys
import os
import json
import logging
import time

# Allow us to import scripts from src without specifying src
sys.path.insert(0, 'src')

# Set up the settings to log information as we run our build pipeline
logging.basicConfig(filename='log.txt', 
		    filemode='a', 
		    level=logging.INFO,
		    datefmt='%H:%M:%S',
		    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s')

def data_target(filepath = ""):
    # TODO: Add try and except for filepath not found
    if filepath == "":
        logging.info("Invalid filepath, no data returned.")
        return []

    with open(filepath) as fh:
        response = json.load(fh)
        data = ['https://youtube.com/watch?v=' + entry['contentDetails']['videoId'] for entry in response['items']]
    logging.info("Data loaded from source.")
    logging.info(f"{len(data)} video IDs loaded.")
    return data

def history_target(data = []):
    if len(data) == 0:
        logging.info(f"Failed to find data.")
        return data

    logging.info(f"Data loaded with {len(data)} entries.")
    data = data
    logging.info("Features and labels extracted from data.")
    return data

def audit_target(data = [], kind = ""):
    if len(data) == 0:
        logging.info(f"Failed to find data.")
        return data

    logging.info(f"Data loaded with {len(data)} entries.")
    data = data
    if kind.lower() == "":
        logging.info(f"What kind of audit are you trying to do? None specified, please try again.")
        return data

    if kind.lower() == "home":
        logging.info("Running audit on home page...")
        start = time.time()
        end = time.time()
        logging.info(f"Home page audit ran in {end - start} seconds.")
    
    if kind.lower() == "search":
        logging.info("Running audit on search results...")
        start = time.time()
        end = time.time()
        logging.info(f"Search results audit ran in {end - start} seconds.")

    if kind.lower() == "recommend":
        logging.info("Running audit on user recommendations...")
        start = time.time()
        end = time.time()
        logging.info(f"Recommendations audit ran in {end - start} seconds.")
    return data
    
def main(targets):
    '''
    Runs the main project build pipeline logic, given targets.

    NOTE: This only ones the audit section. We are using a pre-trained 
    model on our data. For model training, please reference the train.py script.

    TODO: Implement train.py, complete implementation of the audit
    '''

    # Extract video data links to build user watch history
    if 'data' in targets:
        data_target()

    # Build user watch history
    if 'history' in targets:
        history_target()
    
    # Audit the YouTube HomePage for a user
    if 'home' in targets:
        audit_target("home")

    # Audit the YouTube Search Results for a user
    if 'search' in targets:
        audit_target("search")

    # Audit the YouTube Recommendations for a user
    if 'recommend' in targets:
        audit_target("recommend")

    # Run the entire audit
    if 'all' in targets:
        data = data_target()
        history_target(data)
        audit_target(data, "home")        
        audit_target(data, "search")        
        audit_target(data, "recommend")    

    # Run the entire pipeline on our test data
    if 'test' in targets:
        data = data_target('./test/testdata/test_user_watch_history_videos.json')
        history_target(data)
        audit_target(data, "home")        
        audit_target(data, "search")        
        audit_target(data, "recommend")

    return

if __name__ == '__main__':
    targets = [target.lower() for target in sys.argv[1:]]
    main(targets)
    logging.info('END OF BUILD.\n')