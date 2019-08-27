import numpy as np

from pyrtstools.base import _Processor

class PreEmphasis(_Processor):
    """ PreEmphasis is a processor element that amplify the high frequencies
    
    Capacities
    ===========
    Input
    -----
    bytes -- audio signal as bytes

    Ouput
    -----
    numpy.array -- signal converted {and normalized} as a numpy.array of values.

    """
    __name__ = "preEmphasis"
    _input_cap = [np.array]
    _output_cap = [np.array]

    def __init__(self, emphasis_factor: float, keep_last_value: bool = True):
        """ Instanciate a ByteToNum element. Use connect_to to link audio output to the next element

        Keyword arguments:
        ==================
        emphasis_factor (numpy type) -- the emphasis factor [0.0, 1.0]

        keep_last_value (bool) -- keep the last value for the next input (default True)
        """
        _Processor.__init__(self)
        self._buffer = np.array([])
        self.keep_last_value = keep_last_value
        assert  0.0 < emphasis_factor < 1.0, "emphasis factor must be [0.0,1.0]: given {}".format(emphasis_factor)
        self.emphasis_factor = emphasis_factor
        self.last_value = 0.0
        
    
    def input(self, data):
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
            if len(self._buffer) > 0:
                self.process()
            else:
                with self._condition:
                    self._condition.wait()
    
    def process(self):
        self._processing = True
        if self.keep_last_value:
            data = self._buffer - np.concatenate([[self.last_value], self._buffer[:-1]]) * self.emphasis_factor
            self.last_value = self._buffer[-1]
        else:
            data = np.concatenate([[self._buffer[0]], self._buffer[1:] - self._buffer[:-1] * self.emphasis_factor])
        self._buffer = np.array([])
        if self._consumer is not None:
            self._consumer.input(data)
        
        self._processing = False
        with self._condition:
            self._condition.notify()

            