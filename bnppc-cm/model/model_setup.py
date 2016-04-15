import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gpml.model import model_maker
from gpml.xgboost_mod.improved_early_stopping import XGBCEarlyStoppingCV
import xgboost as xgb
import matplotlib.pyplot as plt
from gpml.model.ensemble import AveragingEnsemble


class ModelSetup(object):
    """Handle model specific operations."""

    def __init__(self, config):
        self.model = self.basic_model()
        self.parameter_grid = config.parameter_grids[self.name]

    def json_load(self, file_name):
        with open(file_name) as f:
            obj = json.load(f)
        return obj

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

    def __init__(self, parameter_grid):
        # XGBoost multi-thread support
        # import os
        # from multiprocessing import set_start_method

        # os.environ["OMP_NUM_THREADS"] = "7"
        super().__init__(parameter_grid)

    @staticmethod
    def basic_model():
        xgb = XGBCEarlyStoppingCV(
            max_depth=9,
            learning_rate=0.01,  # Boosting learning rate (xgb's "eta")
            n_estimators=10000,  # num_boost_round
            # silent=False,
            objective='binary:logistic',
            nthread=-1,
            gamma=2,
            min_child_weight=10,
            max_delta_step=1,
            subsample=0.96,
            colsample_bytree=0.45,
            colsample_bylevel=1,
            reg_alpha=1,  # (xgb's alpha), L2 regularization term
            reg_lambda=1,  # (xgb's lambda), L1 regularization term
            # scale_pos_weight=1.0,
            # base_score=0.5,
            # seed=1,
            # missing=None
        )
        return xgb

    @staticmethod
    def empty_booster():
        return xgb.Booster()

    def dump_model(self, model, results, file_name):
        model._Booster.save_model(file_name)
        text_file_name = file_name.replace('.model', '.text')
        model._Booster.dump_model(text_file_name, with_stats=True)
        results_file_name = file_name.replace('.model', '_results.json')
        self.json_dump(results, results_file_name)

    @staticmethod
    def plot_stuff(model, dir):
        # xgboost.plot_importance(model._Booster)
        importance = model._Booster.get_fscore()
        # df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df = pd.DataFrame.from_dict(importance, orient='index')
        df.columns = ['fscore']
        df['feature'] = df.index
        df = df.sort_values(by='fscore')
        df = df[-50:]

        plt.figure()
        df.plot(kind='barh', x='feature', y='fscore',
                legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Importance')
        plt.gcf().savefig(dir + 'xgb_feature_importance.png')

        # xgboost.plot_tree(model._Booster, num_trees=2)
        # plt.savefig(dir + 'xgb_tree.png')

    def load_model(self, file_name):
        model = self.basic_model()
        model._Booster = self.empty_booster()
        model._Booster.load_model(file_name)
        print('Parameters of the loaded model:')
        model_maker.print_dict_as_indented_list(model.get_params())
        results_file_name = file_name.replace('.model', '_results.json')
        results = self.json_load(results_file_name)
        model_maker.print_result_from_dict(results)
        return model, results


class ETCModelSetup(ModelSetup):
    """Extra Trees Classifier Model Setup."""

    def __init__(self, config):
        self.name = 'ExtraTreesClassifier'
        super().__init__(config)

    @staticmethod
    def basic_model():
        etc = ExtraTreesClassifier(
            n_estimators=1000,
            max_features=50,
            criterion='entropy',
            min_samples_split=4,
            max_depth=35,
            min_samples_leaf=2,

            n_jobs=-1,
            verbose=0,
            random_state=2
        )

        # ETC with current params and small data set
        #  - Accuracy: 0.782, LL: nan, ROC AUC: 0.749, Avg P: 0.754,
        #    Acc 1s: 0.951, Acc 0s: 0.241
        # ETC with previous params and expanded data set
        #  - Accuracy: 0.787, LL: nan, ROC AUC: 0.759, Avg P: 0.751,
        #    Acc 1s: 0.958, Acc 0s: 0.237
        # ETC with current params and expanded data set
        #  - Accuracy: 0.786, LL: 0.462, ROC AUC: 0.766, Avg P: 0.758,
        #    Acc 1s: 0.971, Acc 0s: 0.193
        # ETC with above + categorical nans set to -1
        #  - Accuracy: 0.787, LL: 0.460, ROC AUC: 0.768, Avg P: 0.758,
        #    Acc 1s: 0.974, Acc 0s: 0.186
        # ETC with normalised numerics
        #  - Accuracy: 0.787, LL: 0.460, ROC AUC: 0.768, Avg P: 0.758,
        #    Acc 1s: 0.975, Acc 0s: 0.185
        # ETC with normalised numerics and numeric nans set to -1
        #  - Accuracy: 0.786, LL: 0.461, ROC AUC: 0.767, Avg P: 0.757,
        #    Acc 1s: 0.974, Acc 0s: 0.182
        # ETC with normalised numerics and one-hot-encoded columns
        #  -
        # ETC with params from et-classifier
        #  - Accuracy: 0.786, LL: 0.460, ROC AUC: 0.766, Avg P: 0.756,
        #    Acc 1s: 0.965, Acc 0s: 0.214
        # ETC with factorised and one-hot-encoded categoricals
        #  -

        return etc

    @staticmethod
    def empty_booster():
        raise NotImplementedError()

    def dump_model(self, model, results, file_name):
        model_json = {
            'model_type_name': type(model).__name__,
            'parameters': model.get_params(),
            'results': results,
        }

        self.json_dump(model_json, file_name)

        results_file_name = file_name.replace('.json', '_results.json')
        self.json_dump(results, results_file_name)

    def load_model(self, file_name):
        model_dump = self.json_load(file_name)
        results = model_dump['results']
        print('Loaded model results:')
        model_maker.print_dict_as_indented_list(results)
        model = self.basic_model()
        model.set_params(**model_dump['parameters'])
        return model, results


class RFCModelSetup(ModelSetup):
    """Random Forest Classifier Model Setup."""

    def __init__(self, config):
        self.name = 'ExtraTreesClassifier'
        super().__init__(config)

    @staticmethod
    def basic_model():
        rfc = RandomForestClassifier(
            n_estimators=1000,
            criterion='entropy',
            max_depth=35,  # None,
            min_samples_split=4,  # 2,
            min_samples_leaf=2,  # 1,
            min_weight_fraction_leaf=0.0,
            max_features=50,  # 'auto',
            max_leaf_nodes=None,
            bootstrap=False,  # True,

            # n_estimators=1000,
            # max_features=50,
            # criterion='entropy',
            # min_samples_split=4,
            # max_depth=35,
            # min_samples_leaf=2,

            n_jobs=-1,
            verbose=0,
            random_state=1
        )

        return rfc

    @staticmethod
    def empty_booster():
        raise NotImplementedError()

    def dump_model(self, model, results, file_name):
        model_json = {
            'model_type_name': type(model).__name__,
            'parameters': model.get_params(),
            'results': results,
        }

        self.json_dump(model_json, file_name)

        results_file_name = file_name.replace('.json', '_results.json')
        self.json_dump(results, results_file_name)

    def load_model(self, file_name):
        model_dump = self.json_load(file_name)
        results = model_dump['results']
        print('Loaded model results:')
        model_maker.print_dict_as_indented_list(results)
        model = self.basic_model()
        model.set_params(**model_dump['parameters'])
        return model, results


class AvgEnsModelSetup(ModelSetup):
    """Averaging Ensemble."""

    def __init__(self, config):
        self.name = 'AveragingEnsemble'
        self.model_setups = [
            LRModelSetup(config),
            SVCModelSetup(config),
            SGDCModelSetup(config),
            XGBModelSetup(config),
            ETCModelSetup(config),
            RFCModelSetup(config)
        ]
        self.weights = config.weights
        super().__init__(config)

    @staticmethod
    def basic_model(self):
        AveragingEnsemble(
            [ms.model for ms in self.model_setups],
            weights
        )

    @staticmethod
    def empty_booster():
        raise NotImplementedError()

    def dump_model(self, model, results, file_name):
        raise NotImplementedError()
        for model_setup, (model, results) in zip(model_setups, trained):
            model_setup.dump_model(
                model,
                results,
                config.model_dump_file_names[model_setup.name]
            )

    def load_model(self, file_name):
        raise NotImplementedError()
