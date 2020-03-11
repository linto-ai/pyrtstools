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
                       input_shape: tuple = None,
                       on_detection : callable = lambda x, y: print("threshold reached for {} ({})".format(x, y), flush=True),
                       threshold: float = 0.5,
                       n_act_recquire: int = 1,
                       debug: bool = False):
        """KWS is an interface allowing hotword spotting from audio features.

        Keyword arguments:
        ==================
        model_path (str) -- absolute path to a tensorflow (.pb), keras model (.net/ .hdf5 /.h5) or tensorflowLite (.tflite)

        input_shape (tuple) -- DEPRECIATED, input shape is now extracted from model
        
        on_detection (callable(int, float)) -- called when a prediction is superior to the specified threshold. Arguments are callable(index, value)

        threshold (float) -- output activation threshold, must be between [0.0, 1.1] (default 0.5)

        n_act_recquire: (int) -- Number of successive activation recquired to detect (default 1)

        debug (bool) -- Prompt every prediction (default false)

        Raises:
        =======
        AssertionError -- some parameter are wrongly formated or out of bounds

        FileNotFoundError -- model file not found 
        """
        _Consumer.__init__(self)

        assert threshold >= 0 and threshold <= 1, "threshold must be between [0.0,1.0]"
        self._debug = debug
        if input_shape is not None:
            print("[KWS] WARNING: Input shape is depreciated, parameter ignored.")

        self._inferer = Inferer(model_path)
        model_input_shape = self._inferer.input_shape # Discard first value which is batch size 
        self._n_features = model_input_shape[1]
        self._max_batch = model_input_shape[0]
        self._feature_length = model_input_shape[2]
        
        self._feat_buffer = np.zeros((self._n_features, self._feature_length))
        
        self.on_detection = on_detection
        self._threshold = threshold

        self.n_act_req = n_act_recquire
        self.n_act = 0
        self.last_kw_i = 0
   
    def clear_buffer(self):
        """Fill the features buffer with zeros."""
        self._feat_buffer = np.zeros((self._n_features, self._feature_length))

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
        if self._max_batch is not None:
            preds = np.concatenate([self._inferer.predict(inp[np.newaxis]) for inp in inputs])
        else:
            preds = self._inferer.predict(inputs)
        if self._debug:
            print(preds, flush=True)
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