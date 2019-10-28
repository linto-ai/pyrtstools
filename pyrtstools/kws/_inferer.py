import os

import tensorflow as tf
class Inferer(object):
    def __init__(self, model_path: str):
        assert model_path.endswith('.net') or model_path.endswith('.pb'), "Supported mode files are .pb and .net"
        if model_path.endswith('.net'):
            self._predict_fun = self._load_keras_model(model_path)
        else:
            self._predict_fun = self._load_tensorflow_model(model_path)
    
    def _load_keras_model(self, model_path : str):
        from keras.models import load_model
        self.model = load_model(model_path)
        self.model._make_predict_function()
        return self.model.predict
    
    def _load_tensorflow_model(self, model_path : str):
        serving_dir = os.path.dirname(model_path)
        self.model = tf.saved_model.load(serving_dir)
        self.infer = self.model.signatures["serving_default"]

        return lambda x: self.infer(tf.constant(x.astype('float32')))['output'].numpy()

    
    def predict(self, inputs):
        return self._predict_fun(inputs)
