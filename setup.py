#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="linspeech",
    version="0.1.0",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
        'pyaudio',
        'webrtcvad',
        'sonopy'
    ],
    
    author="Rudy Baraglia",
    author_email="baraglia.rudy@gmail.com",
    description="Tools for speech processing, keyword spotting",
    license="AGPL V3",
    keywords="kws hotword keyword vad utterance voice-command speech",
    url="https://github.com/linto-ai/linspeech.git",
    py_modules=['linspeech'],
    project_urls={
        "github" : "https://github.com/linto-ai/linspeech.git"
    },
    long_description="Refer to README"
)