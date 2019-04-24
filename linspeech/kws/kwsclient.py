#!/usr/bin/env python3
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
import json
import requests

from numpy import array, squeeze, argmax, newaxis

class KWSClient:
    """KeyWord Spotting client meant to connect to a tensorflow serving API """
    _header = {"content-type": "application/json"}
    _threshold = 0.5
    _inference_step = 1
    _c = 0
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
        self.uri = request_uri
        assert len(input_shape) == 2, "input shape must be (n_features, feature_length)"
        self._n_feats = input_shape[0]
        self._feat_l = input_shape[1]
        self.on_detection = on_detection
        self.on_error = on_error
        self.threshold = threshold
        self.inference_step = inference_step
        self._features = array([[0.0] * input_shape[1]]* input_shape[0])
    
    def _submit(self):
        data = json.dumps({"signature_name": "serving_default", "instances": [self._features.tolist()]})
        try:
            json_response = requests.post(self.uri, data=data, headers=self._header)
        except Exception as err:
            self.on_error(err)
        else:
            if json_response.status_code == 200:
                try:
                    prediction = array(json.loads(json_response.text)['predictions'])
                    if prediction[prediction > self._threshold].any():
                        self._on_detection(argmax(prediction), max(prediction))
                except json.JSONDecodeError:
                    self.on_error("Could not parse response json")
            else:
                self.on_error("Could not process request {}: {}".format(json_response.status_code, json_response.text))

    def push_features(self, features: array):
        assert features.shape[1] == self._feat_l, "expected features of shape (?, {}) got {}".format(self._feat_l, features.shape)
        n_feats = len(features)
        if n_feats > self._n_feats:
            self._features = features[:self._n_feats]
        else:
            self._features[:-n_feats] = self._features[n_feats:]
            self._features[-n_feats:] = features
        self._c += n_feats
        if self._c >= self.inference_step:
            self._submit()
            self._c = 0

    def clear_buffer(self):
        """Fill the features buffer with zeros. To be used if two series of buffers are not following each other in time."""
        self._features = array([[0.0] * self._feat_l] * self._n_feats)

    def _on_detection(self, index, value):
        self.on_detection(index, value)
        self.clear_buffer()

    @property
    def threshold(self) -> float:
        return self._threshold
    @threshold.setter
    def threshold(self, value: float):
        if value > 0 and value <=1:
            self._threshold = value
        else:
            raise ValueError("threshold value be in between ]0.0,1.0]")

    @property
    def inference_step(self) -> int:
        return self._inference_step
    @inference_step.setter
    def inference_step(self, value: int):
        self._inference_step = value if value > 0 else 1


if __name__ == '__main__':
    client = KWSClient("https://dev.linto.ai/kws/v1/models/hotword:predict",(30,13), inference_step=6)
    for i in range(90):
        client.push_features(array([0.0]*13)[newaxis])
    