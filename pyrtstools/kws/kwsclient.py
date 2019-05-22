#!/usr/bin/env python3
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
import json
import requests

import numpy as np

from pyrtstools.base import _Consumer, InputError

class KWSClient(_Consumer):
    """KeyWord Spotting client meant to connect to a tensorflow serving API """
    __name__ = "kwsclient"
    _header = {"content-type": "application/json"}
    _input_cap = [np.array]

    def __init__(self, request_uri: str,
                 input_shape: tuple,
                 on_detection: callable = lambda x, y: print("threshold reached for {} ({})".format(x, y)),
                 threshold: float = 0.5,
                 inference_step: int = 1, 
                 on_error: callable = lambda x : print(x)):
        """ Create a KWS client

        Keyword arguments:
        ==================
        request_uri (str) -- tensorflow serving API uri

        input_shape (tuple(int, int)) -- input expected from the serving API

        on_detection (callable(int, float)) -- callback function called when a prediction is superior to the specified threshold. Arguments are callable(index, value).

        threshold (float) -- value to trigger detection, must be between ]0,1]

        inference_step (int) -- steps between two inferences, 1 is an inference at each new features (default 1)

        on_error (callable(str)) -- called when an error occurs.

        Raises:
        =======
        AssertionError(str) -- Wrong input shape

        ValueError(str) -- Wrong parameter value
        """
        _Consumer.__init__(self)
        self._step = 0
        self._inference_step = 1
        self._threshold = 0.5
        self._feat_buffer = np.array([[0.0] * input_shape[1]] * input_shape[0])

        assert len(input_shape) == 2, "input shape must be (n_features, feature_length)"
        assert threshold >= 0 and threshold <= 1, "threshold must be between [0.0,1.0]"
        assert inference_step > 0, "inference_step must be positive"

        self.uri = request_uri
        self._n_features = input_shape[0]
        self._feature_length = input_shape[1]
        self.on_detection = on_detection
        self.threshold = threshold
        self._inf_step = inference_step
        self.on_error = on_error

    
    def _submit(self) -> np.array:
        data = json.dumps({"signature_name": "serving_default", "instances": [self._feat_buffer.tolist()]})
        try:
            json_response = requests.post(self.uri, data=data, headers=self._header)
        except Exception as err:
            self.on_error(err)
        else:
            if json_response.status_code == 200:
                try:
                    pred = np.array(json.loads(json_response.text)['predictions'])
                    return pred
                except json.JSONDecodeError:
                    self.on_error("Could not parse response json")
            else:
                self.on_error("Could not process request {}: {}".format(json_response.status_code, json_response.text))

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
                    continue
            if self._step >= self._inf_step:
                self.process()
            else:
                with self._condition:
                    self._condition.wait()

    def process(self):
        self._processing = True
        self._step = 0
        pred = self._submit()
        if any(pred > self._threshold):
            self.on_detection(np.argmax(pred), max(pred))
            self.clear_buffer() #Prevent successive multiple activations
        
        self._processing = False
        with self._condition:
            self._condition.notify()

    def clear_buffer(self):
        """Fill the features buffer with zeros."""
        self._feat_buffer = np.array([[0.0] * self._feature_length] * self._n_features)
        self._step = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if value > 0 and value <=1:
            self._threshold = value
        else:
            raise ValueError("threshold value be in between ]0.0,1.0]")

if __name__ == '__main__':
    client = KWSClient("https://dev.linto.ai/kws/v1/models/hotword:predict",(30,13), inference_step=6)
    for i in range(90):
        client.input(array([0.0]*13)[newaxis])
    