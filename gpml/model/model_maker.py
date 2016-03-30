import numpy as np
import pandas as pd
import json
import numbers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from gpml.data_set import data_set_maker


def print_coefs(feature_names, model):
    for feature, coef in zip(feature_names, model.coef_[0]):
        print('%s - %.3f' % (feature, coef))


def basic_lr():
    lr = LogisticRegression(
        penalty='l2',
        C=0.1,
        class_weight='balanced',
        max_iter=100,
        random_state=1,
        solver='lbfgs',
        tol=0.000001,
        # n_jobs=-1
    )
    return lr


def basic_svc():
    svc = SVC(
        C=0.1,
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        max_iter=1000,
        random_state=1,
        tol=0.000001
    )
    return svc


class ValidationResult(object):
    def __init__(self, accuracy, log_loss):
        self.acc = accuracy
        self.ll = log_loss


class ValidationResults(object):
    def __init__(self):
        self.results = []

    def append(self, acc, ll):
        self.results.append(ValidationResult(acc, ll))

    def print_results(self):
        for r in self.results:
            print('Accuracy: %.3f, ll: %.3f' % (r.acc, r.ll))

    def get_accuracies(self):
        return np.array([r.acc for r in self.results])

    def get_log_losses(self):
        return np.array([r.ll for r in self.results])

    def print_mean_results(self):
        accuracies = self.get_accuracies()
        log_losses = self.get_log_losses()

        print("Mean accuracy: %0.3f (+/- %0.3f)"
              % (accuracies.mean(), accuracies.std() * 2))
        print("Mean LL: %0.3f (+/- %0.3f)"
              % (log_losses.mean(), log_losses.std() * 2))

    def add_validation_result(self, model,
                              X_train, y_train,
                              X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)

        acc = model.score(X_test, y_test)
        ll = log_loss(y_test,
                      predictions,
                      eps=10 ^ -15)
        self.append(acc, ll)

    def get_mean_results(self):
        return {
            'accuracy': self.get_accuracies().mean(),
            'log_loss': self.get_log_losses().mean()
        }


def kfolds_evaluation(X, y, model):
    kfr = ValidationResults()
    kf = KFold(y.shape[0], n_folds=5)
    for train_index, test_index in kf:
        kfr.add_validation_result(
            model,
            X.values[train_index, :], y.values[train_index],
            X.values[test_index], y.values[test_index]
        )

    return kfr


def evaluate_model(X_train, y_train,
                   X_test, y_test,
                   model):
    kfr = ValidationResults()
    kfr.add_validation_result(model, X_train, y_train, X_test, y_test)
    kfr.print_results()
    return kfr.get_mean_results()


def make_and_save_submission(X_train, y_train,
                             X_test, id_column,
                             model, file_name):
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    predictions = pd.Series(predictions[:, 1], name='PredictedProb',
                            index=id_column.index)
    submission = pd.concat((id_column, predictions), axis=1)
    submission.to_csv(file_name, sep=',', index=False)


def dump_model(model, file_name, results):
    print('Dumping model to JSON.')
    model_dump = {}
    model_dump['model_type_name'] = type(model).__name__
    model_dump['parameters'] = model.get_params()
    coefs = model.coef_
    model_dump['coefficients'] = {
        'list': coefs.tolist(),
        'dtype': str(coefs.dtype),
        'shape': coefs.shape
    }
    model_dump['results'] = results

    with open(file_name, 'w') as f:
        json.dump(model_dump, f, sort_keys=True, indent=4)
        f.write('\n')


def check_coef_load(coefs, coef_dtype, coef_shape):
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


def load_model(file_name):
    # TODO Probably need one of these for each model type...
    print('Loading model from JSON.')
    with open(file_name) as f:
        model_load = json.load(f)

    coef_dtype = model_load['coefficients']['dtype']
    coef_shape = tuple(model_load['coefficients']['shape'])
    coefs = np.array(
        model_load['coefficients']['list'],
        dtype=coef_dtype
    ).reshape(coef_shape)
    check_coef_load(coefs, coef_dtype, coef_shape)

    empty_model = make_model_of_type(model_load['model_type_name'])
    empty_model.set_params(**model_load['parameters'])
    empty_model.coef_ = coefs

    results = model_load['results']
    print('Loaded model results:')
    print_dict_as_indented_list(results)
    return empty_model, results


def make_model_of_type(model_type_name):
    if model_type_name == 'LogisticRegression':
        return basic_lr()
    elif model_type_name == 'SVC':
        return basic_svc()


def print_part_of_dict_as_indented_list(keys, dict_thing):
    part_of_dict_thing = dict((key, dict_thing[key]) for key in keys)
    print_dict_as_indented_list(part_of_dict_thing)


def print_dict_as_indented_list(dict_thing):
    for key, value in dict_thing.items():
        if isinstance(value, str):
            value_str = value
        elif isinstance(value, numbers.Integral):  # Int and Bool
            value_str = str(value)
        elif isinstance(value, numbers.Real):  # (float)
            value_str = '%.3f' % value
        else:
            raise RuntimeError('Type not supported: %s' % type(value))

        print(' - %s: %s' % (key, value_str))


def do_grid_search(model, param_grid, X, y):
    print('Running Grid Search.')
    gs = GridSearchCV(model, param_grid, scoring='log_loss', n_jobs=1, cv=5)
    gs.fit(X, y)
    best_est = gs.best_estimator_
    best_params = best_est.get_params()

    for grid in param_grid:
        print('Chosen parameters:')
        print_part_of_dict_as_indented_list(grid.keys(), best_params)

    check_for_edge_cases(param_grid, best_params)
    return best_est


def check_for_edge_cases(param_grid, best_params):
    for grid in param_grid:
        for key, options in grid.items():
            value = best_params[key]
            if (isinstance(value, numbers.Number) and
                    (value == max(options) or value == min(options))):
                raise ValueError(
                    'Chosen value for %s, of %s, has hit an edge of: %s' %
                    (key, str(value), str(options))
                )
