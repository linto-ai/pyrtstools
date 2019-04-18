#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()
setup(
    name="linspeech",
    version="0.1.1",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
        'pyaudio',
        'webrtcvad',
        'sonopy',
        'requests'],
    
    author="Rudy Baraglia",
    author_email="baraglia.rudy@gmail.com",
    description="Tools for speech processing, keyword spotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="AGPLv3+",
    keywords="kws hotword keyword vad utterance voice-command speech",
    url="https://github.com/linto-ai/linspeech.git",
    py_modules=['linspeech'],
    project_urls={
        "github" : "https://github.com/linto-ai/linspeech.git",
        "pypi" : "https://pypi.org/project/linspeech/"

    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)