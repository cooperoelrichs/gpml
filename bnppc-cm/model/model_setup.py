import json
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from gpml.model import model_maker
from xgboost.sklearn import XGBClassifier
import xgboost
import matplotlib.pyplot as plt


class ModelSetup(object):
    """Handle model specific operations."""

    def __init__(self, parameter_grid):
        self.model = self.basic_model()
        self.parameter_grid = parameter_grid
        self.name = type(self.model).__name__

    def json_load(self, file_name):
        with open(file_name) as f:
            model_load = json.load(f)
        return model_load

    @staticmethod
    def json_dump(obj, file_name):
        with open(file_name, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=4)
            f.write('\n')

    def make_json_dump(self, model, coefs):
        model_dump = {}
        model_dump['model_type_name'] = type(model).__name__
        model_dump['parameters'] = model.get_params()
        model_dump['coefficients'] = {
            'list': coefs.tolist(),
            'dtype': str(coefs.dtype),
            'shape': coefs.shape
        }

        return model_dump

    def process_model_load(self, file_name):
        print('Loading model from JSON.')
        model_load = self.open_model_json(file_name)

        coef_dtype = model_load['coefficients']['dtype']
        coef_shape = tuple(model_load['coefficients']['shape'])
        coefs = np.array(
            model_load['coefficients']['list'],
            dtype=coef_dtype
        ).reshape(coef_shape)
        self.check_coef_load(coefs, coef_dtype, coef_shape)

        empty_model = self.basic_model()
        empty_model.set_params(**model_load['parameters'])

        results = model_load['results']
        print('Loaded model results:')
        model_maker.print_dict_as_indented_list(results)
        return empty_model, coefs, results

    def check_coef_load(self, coefs, coef_dtype, coef_shape):
        if coefs.dtype != coef_dtype:
            raise RuntimeError(
                'Coef dtype is not correct, ' +
                'expected: %s, ' % coef_dtype +
                'was: %s.' % coefs.shape)
        if coefs.shape != coef_shape:
            raise RuntimeError(
                'Coef shape is not correct, ' +
                'expected: %s, ' % str(coef_shape) +
                'was: %s.' % str(coefs.shape))

    @staticmethod
    def not_implemented_error():
        raise NotImplementedError(
            'Model specific method, inheritors must overload this'
        )

    def dump_model_to_json_obj(self, model):
        ModelSetup.not_implemented_error()

    @staticmethod
    def basic_model():
        ModelSetup.not_implemented_error()

    def load_model(self):
        ModelSetup.not_implemented_error()


class LRModelSetup(ModelSetup):
    """Logistic Regression Model Setup."""

    def dump_model_to_json_obj(self, model):
        coefs = model.coef_
        model_dump = self.make_json_dump(model, coefs)
        return model_dump

    @staticmethod
    def basic_model():
        lr = LogisticRegression(
            penalty='l1',
            C=0.1,
            # class_weight='balanced',  # This ruins both the acc and ll scores
            # max_iter=100,
            random_state=1,
            # solver='lbfgs',
            # tol=0.000001,
            # n_jobs=-1
        )
        return lr


class SVCModelSetup(ModelSetup):
    """Support Vector Classifier Model Setup."""

    def dump_model_to_json_obj(self, model):
        support_vectors = model.support_vectors_
        model_dump = self.make_json_dump(model, support_vectors)
        return model_dump

    @staticmethod
    def basic_model():
        svc = SVC(
            C=0.0001,
            kernel='rbf',
            probability=True,
            # class_weight='balanced',
            max_iter=10,
            random_state=1,
            tol=0.000001
        )
        return svc


class SGDCModelSetup(ModelSetup):
    """SGD Classifier Model Setup."""

    def dump_model_to_json_obj(self, model):
        coefs = model.coef_
        model_dump = self.make_json_dump(model, coefs)
        return model_dump

    @staticmethod
    def basic_model():
        sgdc = SGDClassifier(
            loss='log',
            penalty='elasticnet',
            alpha=0.001,  # Constant that multiplies the regularization term.
            l1_ratio=0.1,  # The Elastic Net mixing parameter.
            n_iter=5000,
            random_state=1,
            average=False,
            # verbose=1,
            learning_rate='optimal'
            # class_weight='balanced',  # This ruins the acc and ll scores.
            # n_jobs=-1
        )
        return sgdc

    def load_model_from_json(self, file_name):
        empty_model, coefs, results = self.process_model_load(file_name)
        empty_model.coef_ = coefs
        return empty_model, results


class XGBModelSetup(ModelSetup):
    """XGBoost Model Setup."""

    @staticmethod
    def basic_model():
        xgb = XGBClassifier(
            max_depth=11,
            learning_rate=0.01,  # Boosting learning rate (xgb's "eta")
            n_estimators=2000,  # num_boost_round
            # silent=False,
            objective='binary:logistic',
            nthread=-1,
            # gamma=1.0,
            min_child_weight=1,
            # max_delta_step=1,
            subsample=0.96,
            colsample_bytree=0.45,
            # colsample_bylevel=1.0,
            # reg_alpha=1.0,  # (xgb's alpha), L2 regularization term
            # reg_lambda=1.0,  # (xgb's lambda), L1 regularization term
            # scale_pos_weight=1.0,
            # base_score=0.5,
            # seed=1,
            # missing=None
        )

        # eval_metric: ['error', 'logloss', 'auc', 'mae']

        # max_depth=3,
        # learning_rate=0.1,
        # n_estimators=100,
        # silent=True,
        # objective="binary:logistic",
        # nthread=-1,
        # gamma=0,
        # min_child_weight=1,
        # max_delta_step=0,
        # subsample=1,
        # colsample_bytree=1,
        # colsample_bylevel=1,
        # reg_alpha=0,
        # reg_lambda=1,
        # scale_pos_weight=1,
        # base_score=0.5,
        # seed=0,
        # missing=None

        return xgb

    def dump_model(self, model, results, file_name):
        model._Booster.save_model(file_name)
        text_file_name = file_name.replace('.model', '.text')
        model._Booster.dump_model(text_file_name, with_stats=True)
        results_file_name = file_name.replace('.model', '_results.json')
        self.json_dump(results, results_file_name)

    @staticmethod
    def plot_stuf(model):
        f, axes = plt.subplots(2, 1)
        axes[0] = xgboost.plot_importance(model._Booster)
        axes[1] = xgboost.plot_tree(model._Booster, num_trees=2)
        plt.show()
