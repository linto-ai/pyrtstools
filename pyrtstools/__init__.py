import sys
import os
import json

from pyrtstools.base import *
import pyrtstools.vad
import pyrtstools.listenner
import pyrtstools.kws
import pyrtstools.features
import pyrtstools.utils
import pyrtstools.transform

if getattr(sys, 'frozen', False):
    DIR_PATH = os.path.dirname(sys.executable)
else:
    DIR_PATH = os.path.dirname(os.path.dirname(__file__))

__name__ = "pyrtstools"

try:
    __version__ = json.load(open(os.path.join(DIR_PATH, "manifest.json"), "r"))["version"]
except:
    __version__ = None