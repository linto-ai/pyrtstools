# LinSpeech 

## Introduction

Linspeech is a collection of classes designed to develop a speech processing pipeline for voice user interface.

## Features

Linspeech features different blocks:

* Audio acquisition
* Voice activity detection
* Feature extraction
* Keyword spotting

All the element are designed to be easy to use and easy to interconnect.

## Installation

In order to install the package you need python3 and setuptools installed.
The other dependencies are automaticly fetched.

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

```python
import linspeech as lsp
```

Every block is located in a subpackage:

* Audio acquisition: ```linspeech.listenner```
* Voice activity detection: ```linspeech.vad```
* Feature extraction: ```linspeech.features```
* Keyword spotting: ```linspeech.kws```

Exemple:

```python
#!/usr/bin/env python3
import linspeech as lsp 

def main():
    """ simple remote keyword spotting from microphone input """
    params = lsp.features.FeaturesParams() # Default audio parameters
    listenner = listenner = lsp.listenner.Listenner() #Microphone input
    featurer = lsp.features.SonopyMFCC(params) # Feature extraction
    kws = lsp.kws.kwsclient.KWSClient("https://dev.linto.ai/kws/v1/models/hotword:predict",(30,13), inference_step=6) # Remote (TF serving) keyword spotting
    
    #Connect the different elements
    listenner.on_new_data = featurer.push_data
    featurer.on_new_features = kws.push_features
    kws.on_detection = lambda *args: print("Keyword detected !")

    #Start the input
    listenner.start()
    try:
        listenner.join()
    except KeyboardInterrupt:
        listenner.stop()

if __name__ == '__main__':
    main()
```

You can learn more by reading the docstrings.

## Licence
This project is under aGPLv3 licence, feel free to use and modify the code under those terms.
See LICENCE

## Used libraries

* [Numpy](http://www.numpy.org/)
* [Tensorflow / keras](https://github.com/tensorflow/tensorflow)
* [Sonopy](https://github.com/MycroftAI/sonopy)
* [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/docs/index.html)