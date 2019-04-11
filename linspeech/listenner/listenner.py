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
import time
from threading import Thread
import pkg_resources

import pyaudio

__author__ = "Rudy BARAGLIA rbaraglia@linagora.com"
__license__ = "AGPLv3"
__version__ = pkg_resources.get_distribution('linspeech').version

class Listenner(Thread):
    """ A class that use port audio to read microphone input and calls a given function at every new data."""
    _sample_rate = 16000
    _channels = 1
    _chunk_size = 1024
    _sample_depth = 16
    def __init__(self, on_new_data: callable = lambda x : print("got {} bytes of data".format(len(x))),
                 sample_rate: int = 16000,
                 channels: int = 1,
                 sample_depth: int = 2,
                 on_error: callable = lambda x : print(x),
                 frame_per_buffer: int = 1024):
        """ Listenner uses portaudio to read input audio. Is a subclass of Thread. Call Listenner.start() to start the input and Listenner.stop() to stop it.

        Keyword arguments:
        ==================
        on_new_data (callable(bytes)) -- function called when new data is read. Data size is frame_per_buffer * channels * sample_depth. Send b'' when the stream is stopped.
        
        sample_rate (int) -- input sampling rate (default 16000)

        channels (int) -- number of input channels (default 1)

        sample_depth (int) -- bytes per sample must in [1,2,3,4] (default 2)
    
        on_error (callable(str)) -- called when the stream ends unexpectedly

        frame_per_buffer (int) -- number of sample read at a time

        Raises:
        =======
        ValueError(str) -- A wrong value has been given

        """
        Thread.__init__(self)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_depth = sample_depth
        self.on_new_data = on_new_data
        self.on_error = on_error
        self.frame_per_buffer = frame_per_buffer
        self._audio = pyaudio.PyAudio()

        self._running = False
        

    def run(self):
        self._running = True
        self._stream = self._audio.open(format=self._sample_depth,
                                        channels=self._channels,
                                        rate=self._sample_rate,
                                        input=True, 
                                        frames_per_buffer=self._chunk_size)
        while self._stream.is_active() and self._running:
            data = self._stream.read(self._chunk_size, exception_on_overflow=False)
            self.on_new_data(data)

        self.on_new_data(b'')
        if self._stream.is_active():
            self._stream.stop_stream()
        if self._running:
            self.on_error("Stream has unexpectedly stopped")
        self._stream.close()
        self._audio.terminate()
    
    def stop(self):
        """ stop the listenner, as a thread it cannot be launch again """
        self._running = False


    @property
    def sample_rate(self):
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value: int):
        if value <= 0:
            raise ValueError("sample_rate must be positive, given {}".format(value))
        self._sample_rate = value
    
    @property
    def channels(self):
        return self._channels
    
    @channels.setter
    def channels(self, value: int):
        if value <= 0:
            raise ValueError("channels must be positive, given {}".format(value))
        self._channels = value

    @property
    def frame_per_buffer(self):
        return self._chunk_size
    
    @frame_per_buffer.setter
    def frame_per_buffer(self, value: int):
        if value <= 0:
            raise ValueError("frame_per_buffer must be positive, given {}".format(value))
        self._chunk_size = value

    @property
    def sample_depth(self):
        return self._chunk_size
    
    @sample_depth.setter
    def sample_depth(self, value: int):
        if value not in [1,2,3,4]:
            raise ValueError("supported sample_depth are 1,2,3,4, given {}".format(value))
        self._sample_depth = [pyaudio.paInt8, pyaudio.paInt16, pyaudio.paInt24, pyaudio.paInt32][value - 1]
