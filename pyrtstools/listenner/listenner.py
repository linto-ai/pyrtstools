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
import pyaudio
import numpy as np

from pyrtstools.base import _Producer, _Consumer

class AudioParams:
    """AudioParams hold parameters describing audio signal """
    sample_rate = 16000
    dtype = np.int16
    channels = 1
    frame_per_buffer = 1024

    def __init__(self, **kwargs):
        """Keyword arguments:
        =====================
        sample_rate (int) -- sampling rate (default 16000)

        dtype (numpy type) -- encoding. Must be a numpy type or have a nbytes function returning the number of bytes (default numpy.int16)

        channels (int) -- number of channels (default 1)

        frame_per_buffer (int) -- window size (default 1024) 

        """
        for key, value in kwargs.items():
            if key in self.__dir__():
                self.__setattr__(key, value)
    
    @property
    def nbytes(self) -> int:
        return self.dtype(0).nbytes

class Listenner(_Producer):
    """Listenner is a producer element that read audio from the default system microphone using portaudio.
    
    Capacities
    ===========
    Ouput
    -----
    bytes - audio signal as bytes using constructor audio parameters
    """
    __name__ = "listenner"
    _output_cap = [bytes]

    _chunk_size = 1024

    def __init__(self, params: AudioParams,
                 on_error: callable = lambda x : print(x)):
        """Instanciate a Listenner element. Use connect_to to link audio output to the next element

        Keyword arguments:
        ==================
        params (AudioParams) -- instance of AudioParams with input audio parameters         
    
        on_error (callable(str)) -- called when the stream ends unexpectedly

        Raises:
        =======
        ValueError(str) -- a wrong value has been given
        """
        _Producer.__init__(self)
        self.params = params
        self.on_error = on_error
        self._audio = pyaudio.PyAudio()      

    def run(self):
        self._running = True
        self._stream = self._audio.open(format=pyaudio.get_format_from_width(self.params.nbytes),
                                        channels=self.params.channels,
                                        rate=self.params.sample_rate,
                                        input=True, 
                                        frames_per_buffer=self.params.frame_per_buffer)
        while self._stream.is_active() and self._running:
            if self._paused:
                self._stream.stop_stream()
                with self._condition:
                    self._condition.wait()
                    self._stream.start_stream()
            data = self._stream.read(self._chunk_size, exception_on_overflow=False)
            if self._consumer is not None:
                self._consumer.input(data)

        if self._stream.is_active():
            self._stream.stop_stream()
        if self._running:
            self.on_error("Stream has unexpectedly stopped")
        self._stream.close()
        self._audio.terminate()
    