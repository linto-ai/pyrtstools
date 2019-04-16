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
from enum import Enum
import pkg_resources

import webrtcvad

__author__ = "Rudy BARAGLIA rbaraglia@linagora.com"
__license__ = "AGPLv3"
__version__ = pkg_resources.get_distribution('linspeech').version

class Utt_Status(Enum):
    CANCELED = 0
    THREACHED = 1
    TIMEOUT = -1

class VADer:
    """ VADer is a wrapper around webrtcvad designed to perform voice activity detection and utterance boundaries detection."""
    _sample_rate = 16000 #frames/s
    _window_length = 480 #frames
    _sample_depth = 2 #bytes
    _utt_callback = lambda x, y : print(x, len(y))
    _buffer = b''
    def __init__(self, sample_rate: int = 16000,
                       window_length: int = 30,
                       tail : int = 2,
                       mode : int = 3):
        """ Initialize voice activity detection and utterance detection. Only support 16bits integer inputs
        
        Keyword arguments:
        ==================
        sample_rate (int) -- input audio sample rate, supported rates are [8000,1600,32000,48000] (default 16000)

        windows_length (int) -- inputs audio length in ms, supported length are [10,20,30] (default 30)

        tail (int) -- number of frame to keep as speech after speech labeled frames

        mode (int) -- webrtcvad mode: 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive (default 3)
        
        Raises:
        =======
        ValueError(str) -- Wrong argument type or argument out of bound
        """
        self.sample_rate = sample_rate
        self.window_length = window_length
        self._tail_c, self._tail = 0, tail
        self._sil_c, self._sil_th = 0, 0
        self._speech_c, self._speech_th = 0, 0
        self._timeout = 0
        self._vad = webrtcvad.Vad(mode)
        self._utt_det = False

    def detect(self, data : bytes) -> bool:
        """ Detect if audio window contains speech.

        Keyword arguments:
        ==================

        data (bytes) -- audio input as bytes. len(data) must be equal to (sample_rate(sample/s) * window_lenght(ms) / 1000 * 2. Only accept 16bits integers.

        Returns:
        ========
        is_speech (bool) -- either the window contains speech or not

        Raises:
        =======
        AssertionError(str) -- Wrong input data length
        """
        if len(data) == 0:
            return False
        assert len(data) == self._window_length * self._sample_depth, "Expected data of length {}B got {}B".format(self._window_length * self._sample_depth, len(data))
        if self._utt_det:
            self._buffer += data
            if self._speech_c >= self._speech_th and self._sil_c > self._sil_th:
                self._on_utterance(Utt_Status.THREACHED)
            elif self._sil_c > self._timeout:
                self._on_utterance(Utt_Status.TIMEOUT)
        if self._vad.is_speech(data, self._sample_rate):
            self._tail_c = 0
            self._sil_c = 0
            self._speech_c += 1
            return True
        elif self._tail_c < self._tail:
            self._tail_c +=1
            return True
        else:
            self._sil_c += 1
            return False

    
    def detect_utterance(self, callback: callable, sil_th: int = 600, speech_th: int = 300, time_out: int = 10000):
        """ Start utterance detection. This call marks the beginning of an utterance.
        
        Successive calls to this function before utterance's end restart the process.

        Keyword arguments:
        ==================
        callback (callable(Utt_Status, [bytes | None])) -- Function to call when the utterance's end is detected, if threshold has been reached 
        the audio buffer is joined else returns None as second parameter.

        sil_th (int) -- the amount of consecutive silence -in ms- required to end the utterance after enough speech has been collected (default 600ms)

        speech_th (int) -- the amount of overall speech -in ms- required to consider an utterance (default 300ms)

        time_out (int) -- the amount of consecutive silence that trigger a timeout (default 10000ms)

        """
        self._utt_callback = callback 
        self._sil_th = sil_th // self.window_length
        self._speech_th = speech_th // self.window_length
        self._timeout = time_out // self.window_length
        self._buffer = b''
        self._speech_c = 0
        self._utt_det = True

    def cancel_utterance(self):
        """ Cancel current utterance detection. Send Utt_Status.CANCELED to callback function if utterance detection was running."""
        if self._utt_det:
            self._utt_det = False
        self._on_utterance(Utt_Status.CANCELED)

        
    def _on_utterance(self, status: int):
        self._utt_callback(status, self._buffer if status == Utt_Status.THREACHED else None)
        self._utt_det = False
        
    @property
    def sample_rate(self):
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: int):
        if value not in [8000,16000,32000,48000]:
            raise ValueError("supported sample_rate are [8000,1600,32000,48000], given {}".format(value))
        self._sample_rate = value
    
    @property
    def window_length(self):
        return self._window_length * 1000 / self._sample_rate
    
    @window_length.setter
    def window_length(self, value: int):
        if value not in [10,20,30]:
            raise ValueError("supported window_length are [10,20,30], given {}".format(value))
        self._window_length = int(self._sample_rate * value / 1000)