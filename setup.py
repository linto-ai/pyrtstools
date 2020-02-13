#!/usr/bin/env python3
from setuptools import setup, find_packages
import json

with open("manifest.json", 'r') as f:
    _version = json.load(f)["version"]
setup(
    name="pyrtstools",
    version=_version,
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16',
        'tensorflow>=2.0.0',
        'pyaudio>=0.2',
        'webrtcvad>=2.0',
        'sonopy>=0.1.2',
        'requests>=2.22.0'],
    
    author="Rudy Baraglia",
    author_email="baraglia.rudy@gmail.com",
    description="Tools for real time speech processing, keyword spotting",
    long_description="See detail on github",
    long_description_content_type="text/markdown",
    license="AGPLv3+",
    keywords="kws hotword keyword vad utterance voice-command speech",
    url="https://github.com/linto-ai/pyrtstools.git",
    py_modules=['pyrtstools'],
    project_urls={
        "github" : "https://github.com/linto-ai/pyrtstools.git",
        "pypi" : "https://pypi.org/project/pyrtstools/"

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)