#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()
setup(
    name="pyrtstools",
    version="0.2.8a",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16',
        'tensorflow',
        'pyaudio>=0.2',
        'webrtcvad>=2.0',
        'sonopy>=0.1.2',
        'requests>=2.22.0',
        'speechpy>=2.4'],
    
    author="Rudy Baraglia",
    author_email="baraglia.rudy@gmail.com",
    description="Tools for real time speech processing, keyword spotting",
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