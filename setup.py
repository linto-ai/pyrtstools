#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()
setup(
    name="pyrtstools",
    version="0.2.6",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
        'pyaudio',
        'webrtcvad',
        'sonopy',
        'requests',
        'speechpy'],
    
    author="Rudy Baraglia",
    author_email="baraglia.rudy@gmail.com",
    description="Tools for real time speech processing, keyword spotting (compatibility for python 3.5 and earlier)",
    long_description=long_description,
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