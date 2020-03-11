import os

from tensorflow import lite
from tensorflow.keras import models
from tensorflow import saved_model, constant
class Inferer(object):
    """ Given a model path, generates a predict function based on model format"""
    def __init__(self, model_path: str):
        assert model_path.split('.')[-1] in ['pb', 'net','hdf5', 'tflite'], "Supported mode files are .pb, .net and .tflite"
        if model_path.endswith('.net') or model_path.endswith('.hdf5'):
            self._predict_fun = self._load_keras_model(model_path)
        elif model_path.endswith('.tflite'):
            self._predict_fun = self._load_tensorflowLite_model(model_path)
        else:
            self._predict_fun = self._load_tensorflow_model(model_path)
    
    def _load_keras_model(self, model_path : str):
        """ Load a Keras HDF5 model file and return predict function
        """
        self.model = models.load_model(model_path)
        self.model._make_predict_function()
        self.input_shape = self.model.get_input_shape_at(0)
        return self.model.predict
    
    def _load_tensorflow_model(self, model_path : str):
        """ Load a Tensorflow flatbuffer model file and return predict function
        """
        serving_dir = os.path.dirname(model_path)
        self.model = saved_model.load(serving_dir)
        self.infer = self.model.signatures["serving_default"]
        self.input_shape = tuple(self.infer.structured_input_signature[1][list(self.infer.structured_input_signature[1].keys())[0]].shape) #oof
        return lambda x: self._tfPredict(x)

    def _load_tensorflowLite_model(self, model_path:str):
        """ Load a TensorflowLite compressed flatbuffer model file and return predict function
        """
        self.model = lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        return lambda x : self._tflitePredict(x)

    def _tfPredict(self, inputs):
        res = self.infer(constant(inputs.astype('float32')))
        return res[list(res)[0]].numpy()

    def _tflitePredict(self, inputs):
        self.model.set_tensor(self.input_details[0]['index'], inputs.astype('float32'))
        self.model.invoke()
        return self.model.get_tensor(self.output_details[0]['index'])

    def predict(self, inputs):
        return self._predict_fun(inputs)
