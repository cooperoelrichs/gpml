class ModelSetup(object):
    """Handle model specific operations."""

    def __init__(self, model, parameter_grid):
        self.model = model
        self.parameter_grid = parameter_grid

    def dump_model_to_json_obj(self, model, file_name):
        err_text = (
            'Instantiate this class from one of its model specific variants'
        )
        raise NotImplementedError(err_text)


class LRModelSetup(ModelSetup):
    """Logistic Regression Model Setup."""

    def dump_model_to_json_obj(self, model):
        model_dump = {}
        model_dump['model_type_name'] = type(model).__name__
        model_dump['parameters'] = model.get_params()
        coefs = model.coef_
        model_dump['coefficients'] = {
            'list': coefs.tolist(),
            'dtype': str(coefs.dtype),
            'shape': coefs.shape
        }

        return model_dump


class SVCModelSetup(ModelSetup):
    """Support Vector Classifier Model Setup."""

    def dump_model_to_json_obj(self, model):
        model_dump = {}
        model_dump['model_type_name'] = type(model).__name__
        model_dump['parameters'] = model.get_params()
        coefs = model.support_vectors_
        model_dump['coefficients'] = {
            'list': coefs.tolist(),
            'dtype': str(coefs.dtype),
            'shape': coefs.shape
        }

        return model_dump
