#!/usr/bin/env python
import os

class PackageConfig(object):
    """
    Static class that contains the configuration of our package.
    """

    # Config for run.py
    ENABLE_LOGGING = True
    OVERRIDE_WEIGHTS = False

    # API KEYS
    YOUTUBE_API_KEY = os.environ.get("YOUTUBE_DATA_API_KEY")