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
from enum import Enum
from collections import deque

import webrtcvad

from pyrtstools.base import _Processor

class Utt_Status(Enum):
    """ Return status of VADer Element"""
    CANCELED = 0
    THREACHED = 1
    TIMEOUT = -1

class VADer(_Processor):
    """ VADer is a processing element that detect speech in input signal and forward it to the next element.
    It also permits to detect utterance using the detect_utterance function.
    
    Capacities
    ===========
    Input
    -----
    bytes -- audio signal as bytes, only support 2B little endian integers

    Ouput
    -----
    bytes -- audio signal as bytes
    """
    __name__ = "vader"
    _input_cap = [bytes]
    _output_cap = [bytes]

    def __init__(self, sample_rate: int = 16000,
                       window_length: int = 30,
                       head : int = 2,
                       tail : int = 2,
                       mode : int = 3):
        """ Initialize voice activity detection and utterance detection. Only support 16bits integer inputs
        
        Keyword arguments:
        ==================
        sample_rate (int) -- input audio sample rate, supported rates are [8000,1600,32000,48000] (default 16000)

        windows_length (int) -- inputs audio length in ms on witch speech analysis is done, supported length are [10,20,30] (default 30)

        head (int) -- number of frame to keep as speech before speech labeled frames (default 2)
        
        tail (int) -- number of frame to keep as speech after speech labeled frames (default 2)

        mode (int) -- webrtcvad mode: 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive (default 3)
        
        Raises:
        =======
        ValueError(str) -- Wrong argument type or argument out of bound
        """
        _Processor.__init__(self)

        self._buffer = b'' #input buffer

        self._vad = webrtcvad.Vad(3)
        self._sample_rate = 16000 #frames/s
        self._window_length = 480 #frames
        self._sample_depth  = 2 #bytes
        self._utt_callback = lambda x, y : print(x, len(y))
        self._utt_buffer = b'' #contains current utterance
        self._head_buffer = deque([], maxlen=head)

        self._utt_det = False
        self._tail_c = 0
        self._tail = 2
        self._speech_c = 0
        self._speech_th = 0
        self._sil_c = 0
        self._sil_th = 0
        self._time_out = 0

        self.sample_rate = sample_rate
        self.window_length = window_length
        self._tail = tail
        self._vad.set_mode(mode)

    def input(self, data : bytes):
        self._buffer += data
        with self._condition:
            self._condition.notify()

    def _process(self):
        self._processing = True
        data = self._buffer[:self._window_length * self._sample_depth]
        self._buffer = self._buffer[self._window_length * self._sample_depth:]
        if self._utt_det:
            self._utt_buffer += data
            if self._speech_c >= self._speech_th and self._sil_c > self._sil_th:
                self._on_utterance(Utt_Status.THREACHED)
            elif self._sil_c > self._timeout:
                self._on_utterance(Utt_Status.TIMEOUT)
        if self._vad.is_speech(data, self._sample_rate):
            if self._consumer is not None:
                if self._sil_c > 0 and len(self._head_buffer) > 0:
                    self._consumer.input(b''.join(list(self._head_buffer)[:self._sil_c]))
                self._consumer.input(data)
            self._tail_c = 0
            self._sil_c = 0
            self._speech_c += 1
        elif self._tail_c < self._tail:
            if self._consumer is not None:
                self._consumer.input(data)
            self._tail_c +=1
        else:
            self._sil_c += 1
            self._head_buffer.append(data)
        self._processing = False
        with self._condition:
            self._condition.notify()

    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
                    continue
            if len(self._buffer) >= self._window_length * self._sample_depth and not self._processing:
                self._process()
            else:
                with self._condition:
                    self._condition.wait()

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
        self._utt_buffer = b''
        self._speech_c = 0
        self._utt_det = True

    def cancel_utterance(self):
        """ Cancel current utterance detection. Send Utt_Status.CANCELED to callback function if utterance detection was running."""
        if self._utt_det:
            self._utt_det = False
        self._on_utterance(Utt_Status.CANCELED)

        
    def _on_utterance(self, status: int):
        self._utt_callback(status, self._utt_buffer if status == Utt_Status.THREACHED else None)
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