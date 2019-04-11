""" 
Copyright (c) 2019 Linagora.

This file is part of linspeech

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
import pkg_resources

from numpy import array, newaxis, any, argmax
from numpy import concatenate

from linspeech.kws._inferer import Inferer

__author__ = "Rudy BARAGLIA rbaraglia@linagora.com"
__license__ = "AGPLv3"
__version__ = pkg_resources.get_distribution('linspeech').version

class KWS(object):
    def __init__(self, model_path: str,
                       input_shape: tuple,
                       on_detection : callable = lambda x, y: print("threshold reached for {} ({})".format(x, y)),
                       threshold: float = 0.5,
                       inference_step: int = 1):
        """KWS is an interface allowing hotword spotting from audio features.

        Keyword arguments:
        ==================
        model_path (str) -- absolute path to a tensorflow (.pb) or a keras model (.net)

        input_shape (tuple) -- the model input shape, (number_of_features, feature_length)
        
        on_detection (callable(int, float)) -- called when a prediction is superior to the specified threshold. Arguments are callable(index, value)

        threshold (float) -- output activation threshold, must be between [0.0, 1.1] (default 0.5)

        inference_step (int) -- steps between two inferences, 1 is an inference at each new features (default 1)

        Raises:
        =======
        AssertionError -- some parameter are wrongly formated or out of bounds

        FileNotFoundError -- model file not found 
        """

        assert len(input_shape) == 2, "input_shape format must be (n_features, len_features)"
        assert threshold >= 0 and threshold <= 1, "threshold must be between [0.0,1.0]"
        assert inference_step > 0, "inference_step must be positive"

        self._inferer = Inferer(model_path)
        
        self._n_features = input_shape[0]
        self._feature_length = input_shape[1]
        self.on_detection = on_detection
        self._threshold = threshold
        self._inf_step = inference_step
        self._step = 0
        self._feat_buffer = array([[0.0] * input_shape[1]] * input_shape[0])

    def clear_buffer(self):
        """Fill the features buffer with zeros. To be used if two series of buffers are not following each other in time."""
        self._feat_buffer = array([[0.0] * self._feature_length[1]] * self._n_features[0])
        self._step = 0

    def push_features(self, features: array) -> array:
        """ Detects hotwords, if an hotword is detected, triggers on_detection.

        Keyword arguments:
        ==================
        features (numpy.array) -- an array of features, shape must be (?, feature_length) with feature_length defined during __init__

        Raises:
        =======
        AssertionError -- submitted features are wrongly formated
        """
        assert features.shape[1] == self._feature_length, "Wrong feature shape {}".format(features.shape)
        self._feat_buffer = concatenate((self._feat_buffer[len(features):], features))
        self._step += len(features)
        if self._step >= self._inf_step:
            self._step = 0
            pred = self._inferer.predict(self._feat_buffer[newaxis])[0]
            if pred[pred > self._threshold].any():
                self.on_detection(argmax(pred), max(pred))

    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        if value > 0.0 and value <= 1.0:
            raise ValueError("Threshold must be between ]0.0,1.0]")
        self._threshold = value