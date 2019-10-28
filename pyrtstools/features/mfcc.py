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
import numpy as np
from sonopy import mfcc_spec

from pyrtstools.base import _Processor

class MFCCParams:
    """ Class designed to hold parameters for MFCC features extraction."""

    def __init__(self, **kwargs):
        """ Construct a parameter class.
        
        Keyword arguments:
        ==================       
        window_d (float) -- window duration in s (default 0.064)
        
        stride_d (float) -- window stride in s (default 0.032)
        
        n_filters (int) -- number of filter for mfcc based features (default 20)
        
        n_fft (int) -- fft vector size (default 512)
        
        n_ceps (int) -- mfcc output size (must be <= n_filters) (default 13)
        
        low_freq (int) -- low frequencies cut (default 0)
        
        high_freq (int) -- high frequencies cut (default None)
        """
        self.sample_rate = 16000
        self.window_d = 0.064
        self.stride_d = 0.032
        self.n_filt = 20
        self.n_fft = 512
        self.n_coef = 13
        self.low_freq = 0
        self.high_freq = None
        self.energy = False
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    @property
    def window_l(self) -> int:
        """ Window duration converted to number of frames"""
        return int(self.sample_rate * self.window_d)
    
    @property
    def stride_l(self) -> int:
        """ stride duration converted to number of frames"""
        return int(self.sample_rate * self.stride_d)

from sonopy import mfcc_spec
class SonopyMFCC(_Processor):
    """ SonopyMFCC extract MFCC features using the sonopy library

    Capacities
    ===========
    Input
    -----
    numpy.array -- signal normalized as a numpy.array of values.

    Ouput
    -----
    numpy.array -- MFCC features
    """
    __name__ = "sonopymfcc"
    _input_cap = [np.array]
    _output_cap = [np.array]

    def __init__(self, mfccParams: MFCCParams):
        _Processor.__init__(self)
        self._buffer = np.array([])
        
        self.mfccParams = mfccParams
    
    def input(self, data: np.array):
        self._buffer = np.concatenate([self._buffer, data])
        with self._condition:
            self._condition.notify()
    
    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
                    continue
            if len(self._buffer) >= self.mfccParams.window_l:
                self._process()
            else:
                with self._condition:
                    self._condition.wait()

    def stop(self):
        self._buffer = np.array([])
        super(SonopyMFCC, self).stop()

    def _process(self):
        self._processing = True
        features = mfcc_spec(self._buffer,
                                sample_rate=self.mfccParams.sample_rate,
                                window_stride=(self.mfccParams.window_l, self.mfccParams.stride_l),
                                num_coeffs=self.mfccParams.n_coef + (not self.mfccParams.energy),
                                num_filt=self.mfccParams.n_filt,
                                fft_size=self.mfccParams.n_fft)
        if not self.mfccParams.energy:
            features = features[:, 1:]
        self._buffer = self._buffer[len(features) * self.mfccParams.stride_l:]

        if self._consumer is not None:
            self._consumer.input(features)
        self._processing = False
        with self._condition:
            self._condition.notify()

from speechpy.feature import mfcc as speechpy_mfcc
class SpeechpyMFCC(_Processor):
    """ SonopyMFCC extract MFCC features using the speechpy library

    Capacities
    ===========
    Input
    -----
    numpy.array -- signal normalized as a numpy.array of values.

    Ouput
    -----
    numpy.array -- MFCC features
    """
    __name__ = "speechpymfcc"
    _input_cap = [np.array]
    _output_cap = [np.array]

    def __init__(self, mfccParams: MFCCParams):
        _Processor.__init__(self)

        self._buffer = np.array([])

        self.mfccParams = mfccParams

    def input(self, data: np.array):
        self._buffer = np.concatenate([self._buffer, data])
        with self._condition:
            self._condition.notify()
    
    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
            if len(self._buffer) >= self.mfccParams.window_l + self.mfccParams.stride_l:
                self._process()
            else:
                with self._condition:
                    self._condition.wait()

    def _process(self):
        self._processing = True
        features = speechpy_mfcc(self._buffer, 
                                     self.mfccParams.sample_rate,
                                     frame_length=self.mfccParams.window_d,
                                     frame_stride=self.mfccParams.stride_d,
                                     num_cepstral=self.mfccParams.n_ceps,
                                     num_filters=self.mfccParams.n_filters,
                                     fft_length=self.mfccParams.n_fft,
                                     low_frequency=self.mfccParams.low_freq,
                                     high_frequency=self.mfccParams.high_freq)
        self._buffer = self._buffer[len(features) * self.mfccParams.stride_l:]
        if self._consumer is not None:
            self._consumer.input(features)
        self._processing = False
        with self._condition:
            self._condition.notify()