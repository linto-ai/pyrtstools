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
from numpy import array, frombuffer, concatenate, int16, float, iinfo, squeeze
from sonopy import mfcc_spec

class FeaturesParams:
    """ Class designed to hold parameter related to audio processing and feature extraction.
    
    Parameters for base linspeech classes are set. Other parameter can be set freely during initialization. 
    """
    sample_rate = 16000
    sample_depth = 2
    channels = 1
    window_d = 64
    stride_d = 32
    n_filters = 20
    n_fft = 512
    n_ceps = 13
    low_freq = 0
    high_freq = None

    def __init__(self, **kwargs):
        """ Construct a parameter class.
        
        Keyword arguments:
        ==================
        sample_rate (int) -- frame per second (default 16000)
        
        sample_depth (int) -- encoding in Byte (default 2)
         
        channels (int) -- number of channel (default 1)
        
        window_d (int) -- window duration in ms (default 64)
        
        stride_d (int) -- window stride in ms (default 32)
        
        n_filters (int) -- number of filter for mfcc based features (default 20)
        
        n_fft (int) -- fft vector size (default 512)
        
        n_ceps (int) -- mfcc output size (must be <= n_filters) (default 13)
        
        low_freq (int) -- low frequencies cut (default 0)
        
        high_freq (int) -- high frequencies cut (default None)

        ??? (???) -- Any relevant parameter 
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @property
    def window_l(self) -> int:
        """ Window duration converted to number of frames"""
        return int(self.sample_rate * self.window_d / 1000)
    
    @property
    def stride_l(self) -> int:
        """ stride duration converted to number of frames"""
        return int(self.sample_rate * self.stride_d / 1000)

class Featurer:
    """ Featurer is an interface to extract features from raw audio. This is an abstract class do not instanciate."""
    _raw_buffer = b''
    _num_buffer = array([])
    _dtype = int16
    def __init__(self, params : FeaturesParams,
                       on_new_features: callable = lambda x : print("{} features extracted".format(len(x))),
                       normalize : bool = True, 
                       return_raw: bool = False):
        self.params = params
        self.on_new_features = on_new_features
        self.normalize = normalize
        self.return_raw = return_raw
    
    def push_data(self, data: bytes):
        self._raw_buffer += data
        new_data = frombuffer(data, self._dtype).astype(float) / (1 if not self.normalize else iinfo(self._dtype).max)
        self._num_buffer = concatenate((self._num_buffer, new_data),0)

        while len(self._num_buffer) > self.params.window_l:
            feats = self._extract_features(self._num_buffer[:self.params.window_l])
            if self.return_raw:
                self.on_new_features(feats, self._raw_buffer[:self.params.window_l * self.params.sample_depth])
            else:
                self.on_new_features(feats)
            self._raw_buffer = self._raw_buffer[self.params.stride_l * self.params.sample_depth:]
            self._num_buffer = self._num_buffer[self.params.stride_l:]

    def _extract_features(self, data: array) -> array:
        pass

    def flush_buffer(self):
        self._raw_buffer = b''
        self._num_buffer = array([])


class SonopyMFCC(Featurer):
    """ Extract mfcc audio feature from raw audio using sonopy library"""
    def __init__(self, params : FeaturesParams,
                       on_new_features: callable = lambda x : print("{} features extracted".format(len(x))),
                       normalize : bool = True, 
                       return_raw: bool = False):
        """
        Keyword arguments:
        ==================
        params (FeaturesParams) -- Audio and extraction parameters

        on_new_featuress ([callable(array) | callable(array, bytes)]) -- called when new feature are generated

        normalize (bool) -- Normalize audio before feature extraction

        return_raw (bool) -- if set to true on_new_features return the feature matching raw_audio a second parameter
        """
        super().__init__(params, on_new_features, normalize, return_raw)
        from sonopy import mfcc_spec

    def _extract_features(self, data: array) -> array:
        return mfcc_spec(data, self.params.sample_rate, window_stride=(self.params.window_l, self.params.stride_l), num_coeffs=self.params.n_ceps)
        