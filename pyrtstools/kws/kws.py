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
                       n_act_recquire: int = 1):
        """KWS is an interface allowing hotword spotting from audio features.

        Keyword arguments:
        ==================
        model_path (str) -- absolute path to a tensorflow (.pb) or a keras model (.net)

        input_shape (tuple) -- the model input shape, (number_of_features, feature_length)
        
        on_detection (callable(int, float)) -- called when a prediction is superior to the specified threshold. Arguments are callable(index, value)

        threshold (float) -- output activation threshold, must be between [0.0, 1.1] (default 0.5)

        n_act_recquire: (int) -- Number of successive activation recquired to detect (default 1)

        Raises:
        =======
        AssertionError -- some parameter are wrongly formated or out of bounds

        FileNotFoundError -- model file not found 
        """
        _Consumer.__init__(self)
        self._feat_buffer = np.array([[0.0] * input_shape[1]] * input_shape[0])
        
        assert len(input_shape) == 2, "input_shape format must be (n_features, len_features)"
        assert threshold >= 0 and threshold <= 1, "threshold must be between [0.0,1.0]"
        
        self._n_features = input_shape[0]
        self._feature_length = input_shape[1]
        self.on_detection = on_detection
        self._threshold = threshold

        self.n_act_req = n_act_recquire
        self.n_act = 0
        self.last_kw_i = 0

        self._inferer = Inferer(model_path)
        
    def clear_buffer(self):
        """Fill the features buffer with zeros."""
        self._feat_buffer = np.array([[0.0] * self._feature_length] * self._n_features)

    def input(self, data: np.array):
        if not data.shape[1] == self._feature_length:
            raise InputError("Wrong feature shape {}".format(data.shape))

        self._feat_buffer = np.concatenate((self._feat_buffer, data))
        with self._condition:
            self._condition.notify()

    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
                continue
            if len(self._feat_buffer) >= self._n_features:
                self.process()
            else:
                with self._condition:
                    self._condition.wait()

    def process(self):
        self._processing = True
        inputs = np.array([np.array(self._feat_buffer[i:i+self._n_features]) for i in range(len(self._feat_buffer) - self._n_features + 1)])
        self._feat_buffer = self._feat_buffer[len(inputs):]
        preds = self._inferer.predict(inputs)
        for pred in preds:
            if any(pred > self._threshold):
                kws_i = np.argmax(pred)
                if kws_i == self.last_kw_i:
                    self.n_act += 1
                    if self.n_act >= self.n_act_req:
                        self.on_detection(kws_i, max(pred))
                        self.clear_buffer()
                        break
                else:
                    self.n_act = 1
                    self.last_kw_i = kws_i
            else:
                self.n_act = 0       
            
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