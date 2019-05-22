from collections.abc import Iterable
from typing import Union, Type
from threading import Thread, Condition

class CapIncompatibilityError(Exception):
    pass

class InputError(Exception):
    pass

class _Element(Thread):
    """ ABSTRACT _Element is the base class for all pipeline elements """ 
    __name__ = "element"

    def __init__(self):
        Thread.__init__(self)
        self.on_error = lambda err: print(err)
        self._running = False
        self._paused = False
        self._condition = Condition()
    
    def run(self):
        pass
    
    def stop(self):
        if not self._paused:
            self._paused = True
    
    def resume(self):
        if self._paused:
            self._paused = False
            with self._condition:
                self._condition.notify()

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notifyAll()

class _Consumer(_Element):
    """ ABSTRACT _Consumer is the base class for all data consuming elements."""
    __name__ = "consumer"
    _input_cap = [] # Input type capabilities
        
    def __init__(self):
        _Element.__init__(self)
        self._input_type = None # Input type
        self._producer = None
        self._processing = False
    
    def get_input_cap(self):
        return self._input_cap
    
    def input(self, data):
        pass
    
    def _process(self):
        pass

class _Producer(_Element):
    """ ABSTRACT _Producer is the base class for all data producing elements. """
    __name__ = "producer"
    _output_cap = [] # Output type capabilities
    
    def __init__(self):
        _Element.__init__(self)
        self._output_type = None # Output type
        self._consumer = None
    
    def get_output_cap(self):
        return self._output_cap

    def connected_to(self) -> _Consumer:
        return self._consumer

    def connect_to(self, consumer: _Consumer, dtype = None):
        if dtype is None:
            auto_type = [t for t in self._output_cap if t in consumer._input_cap]
            if len(auto_type) == 0:
                raise CapIncompatibilityError("Could not find data stream compatible between {} and {}".format(self.__name__, consumer.__name__))
            self._output_type = auto_type[0]
            consumer._input_type = auto_type[0]
        else:
            if dtype not in self._output_cap:
                raise CapIncompatibilityError("{} cannot produce {}".format(self.__name__, dtype))
            if dtype not in consumer._input_cap:
                raise CapIncompatibilityError("{} cannot consume {}".format(consumer.__name__, dtype))
            self._output_type = dtype
            consumer._input_type = dtype
        
        self._consumer = consumer
        self._consumer._producer = self
        
class _Processor(_Producer, _Consumer):
    """ABSTRACT _Processor is the base class for all data processing elements. """
    __name__ = "processing"

    def __init__(self):
        _Producer.__init__(self)
        _Consumer.__init__(self)


class Pipeline:
    """ The Pipeline class allow to group of elements used in a process, and control their behavior (start/stop/resume/close) collectively."""
    def __init__(self, elements: list = []):
        self._running = False
        self._paused = False
        self._closed = False

        self.elements = []
        self.add(elements)
    
    def add(self, element):
        """ Add an element or a iterable of elements """
        if self._running:
            raise RuntimeError("Cannot add element while pipeline is running")
        if isinstance(element, Iterable):
            assert all([issubclass(type(e), _Element) for e in element]), "pipeline elements must derivate from _Element"
            self.elements.extend(element)
        else:
            assert(issubclass(type(element), _Element)), "pipeline elements must derivate from _Element"
            self.elements.append(element)

    def start(self):
        """ Start all elements """
        if not self._running and not self._closed:
            self._running = True
            for i, element in enumerate(self.elements[:-1]):
                element.connect_to(self.elements[i+1])
            for element in self.elements:
                element.start()

    def stop(self):
        """ Stop all elements """
        if self._running and not self._paused and not self._closed:
            self._paused = True
            for element in self.elements:
                element.stop()

    def resume(self):
        """ Resume all stopped element """
        if self._running and self._paused and not self._closed:
            self._paused = False
            for element in self.elements:
                element.resume()

    def close(self):
        """ Stop and close all elements""" 
        if not self._closed:
            self.stop()
            for element in self.elements:
                element.close()
