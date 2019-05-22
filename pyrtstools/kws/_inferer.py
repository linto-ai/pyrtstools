
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
        from tensorflow import Graph, GraphDef, import_graph_def, Session
        graph = Graph()
        graph_def = GraphDef()
        with open(model_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            import_graph_def(graph_def)
        self.inp_var = graph.get_operation_by_name("import/{}".format(graph_def.node[0].name)).outputs[0]
        self.out_var = graph.get_operation_by_name('import/net_output').outputs[0]
        self.session = Session(graph=graph)
        return lambda x : self.session.run(self.out_var, {self.inp_var:x})        
    
    def predict(self, inputs):
        return self._predict_fun(inputs)
