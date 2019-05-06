""" 
Copyright (c) 2019 Linagora.

This file is part of pyrtstools

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
import numpy as np

from pyrtstools.base import _Consumer, InputError
from pyrtstools.kws._inferer import Inferer

class KWS(_Consumer):
    """ KWS element use tensorflow or keras model to spot hotword from input features.

    Capacities
    ===========
    Input
    -----
    numpy.array -- input features. Input data shape must be (?, input_shape[1]) with input shape set during __init__ 
    """
    __name__ = "kws"
    _input_cap = [np.array]

    def __init__(self, model_path: str,
                       input_shape: tuple,
                       on_detection : callable = lambda x, y: print("threshold reached for {} ({})".format(x, y), flush=True),
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
        _Consumer.__init__(self)
        self._step = 0
        self._feat_buffer = np.array([[0.0] * input_shape[1]] * input_shape[0])
        
        assert len(input_shape) == 2, "input_shape format must be (n_features, len_features)"
        assert threshold >= 0 and threshold <= 1, "threshold must be between [0.0,1.0]"
        assert inference_step > 0, "inference_step must be positive"
        
        self._n_features = input_shape[0]
        self._feature_length = input_shape[1]
        self.on_detection = on_detection
        self._threshold = threshold
        self._inf_step = inference_step

        self._inferer = Inferer(model_path)
        
    def clear_buffer(self):
        """Fill the features buffer with zeros."""
        self._feat_buffer = np.array([[0.0] * self._feature_length] * self._n_features)
        self._step = 0

    def input(self, data: np.array):
        if not data.shape[1] == self._feature_length:
            raise InputError("Wrong feature shape {}".format(data.shape))
        if len(data) > self._n_features:
            self._feat_buffer = data[-self._n_features:]
        else:
            self._feat_buffer = np.concatenate((self._feat_buffer[len(data):], data))
        self._step += len(data)

        with self._condition:
            self._condition.notify()

    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
            if self._step >= self._inf_step:
                self.process()
            else:
                with self._condition:
                    self._condition.wait()

    def process(self):
        self._processing = True
        self._step = 0
        pred = self._inferer.predict(self._feat_buffer[np.newaxis])[0]
        if any(pred > self._threshold):
            self.on_detection(np.argmax(pred), max(pred))
            self.clear_buffer() #Prevent successive multiple activations
        self._processing = False
        with self._condition:
            self._condition.notify()

    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        if value >= 0.0 and value <= 1.0:
            raise ValueError("Threshold must be between ]0.0,1.0]")
        self._threshold = value