# LinSpeech 

## Introduction

Linspeech is a collection of classes designed to develop a real-time speech processing pipeline for voice user interface.

Disclaimer:
This is an early version designed to provide a voice command detection pipeline for LinTO.
However the elements are designed to be generic and can be used for other purposes.

## Features

Linspeech features different blocks:

* Audio acquisition
* Voice activity detection
* Feature extraction
* Keyword spotting

All the element are designed to be easy to use and easy to interconnect.

## Installation

In order to install the package you need python3 and pip/setuptools installed.

Recquired libraries are:
* portaudio19-dev (For pyaudio microphone input)

The python dependecies are automaticly installed. 
(Note that it may takes some time as some of them -numpy, tensorflow- are faily large)

### pypi

```bash
sudo pip3 install linspeech
```

### From source

```bash
git clone https://github.com/linto-ai/linspeech.git
cd linspeech
sudo ./setup.py install
```

### Usage

Here are a simple pipeline designed to detect hotword from microphone

```python
import linspeech as lsp

audioParam = lsp.listenner.AudioParams() # Hold signal parameters
listenner = lsp.listenner.Listenner(audioParam) # Microphone input
vad = lsp.vad.VADer() # Voice activity detection
btn = lsp.transform.ByteToNum(normalize=True) #Convert raw signal to numerical
featParams = lsp.features.MFCCParams() # Hold MFCC features parameters
mfcc = lsp.features.SonopyMFCC(featParams) # Extract MFCC
kws = lsp.kws.KWS("/path/to/your-model.pb", (30,13)) # Hotword spotting 
pipeline = lsp.Pipeline([listenner, vad, btn, mfcc, kws]) # Holds elements and links them
pipeline.start() # Start all the elements
try:
    listenner.join() # Wait for the microphone to finish (To block the execution)
except KeyboardInterrupt:
    pipeline.close()
```

Every block is located in a subpackage:

* Audio acquisition: ```linspeech.listenner```
* Voice activity detection: ```linspeech.vad```
* Features extraction: ```linspeech.features```
* Keyword spotting: ```linspeech.kws```
* Signal transformation: ```linspeech.transform```

Every element and class is documented.

## Licence
This project is under aGPLv3 licence, feel free to use and modify the code under those terms.
See LICENCE

## Used libraries

* [Numpy](http://www.numpy.org/)
* [Tensorflow / keras](https://github.com/tensorflow/tensorflow)
* [Sonopy](https://github.com/MycroftAI/sonopy)
* [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/docs/index.html)
* [py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
