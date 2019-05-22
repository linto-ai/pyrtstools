import numpy as np

from pyrtstools.base import _Processor

class ByteToNum(_Processor):
    """ ByteToNum is a processor element that convert input audio to numpy array
    
    Capacities
    ===========
    Input
    -----
    bytes -- audio signal as bytes

    Ouput
    -----
    numpy.array -- signal converted {and normalized} as a numpy.array of values.

    """
    __name__ = "bytetonum"
    _input_cap = [bytes]
    _output_cap = [np.array]

    def __init__(self, dtype=np.int16, normalize: bool = False):
        """ Instanciate a ByteToNum element. Use connect_to to link audio output to the next element

        Keyword arguments:
        ==================
        dtype (numpy type) -- the input data type (default numpy.int16)

        normalize (bool) -- either to normalize the data or not (default False)
        """
        _Processor.__init__(self)
        self._buffer = b''
        assert "nbytes" in dir(dtype), "Input data type must have nbytes method" 
        self._dtype = dtype
        self.normalize = normalize
    
    def input(self, data):
        self._buffer += data
        with self._condition:
            self._condition.notify()

    def run(self):
        self._running = True
        while self._running:
            if self._paused or self._processing:
                with self._condition:
                    self._condition.wait()
                    continue
            if len(self._buffer) >= self._dtype(0).nbytes:
                self.process()
            else:
                with self._condition:
                    self._condition.wait()
    
    def process(self):
        self._processing = True
        data = np.frombuffer(self._buffer, dtype=self._dtype) / (1 if not self.normalize else np.iinfo(self._dtype).max)
        self._buffer = b''
        if self._consumer is not None:
            self._consumer.input(data)
        
        self._processing = False
        with self._condition:
            self._condition.notify()