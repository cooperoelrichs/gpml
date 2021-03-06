import numpy as np
import pandas as pd
import numbers
import json
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.grid_search import GridSearchCV


def get_x(df, not_x_labels):
    feature_names = df.columns.difference(not_x_labels)
    return df[feature_names]


def get_y(df, y_label):
    return df[y_label]


def get_x_and_y(df, not_x_labels, y_label):
    return get_x(df, not_x_labels), get_y(df, y_label)


def get_xs_and_ys(train, test, not_x_labels, y_label):
    X_train, y_train = get_x_and_y(train, not_x_labels, y_label)
    X_test, y_test = get_x_and_y(test, not_x_labels, y_label)
    return X_train, y_train, X_test, y_test


def evaluate_model_using_kfolds(model, model_setup, config):
    X_train_eval, y_train_eval, _, _ = get_xs_and_ys(
        config.evaluation_data_set_frames['evaluation_training_data_set'],
        config.evaluation_data_set_frames['evaluation_testing_data_set'],
        config.meta_columns + [config.y_label], config.y_label
    )

    results = kfolds_evaluation(
        X_train_eval, y_train_eval, model,
        config.fitting_parameters[model_setup.name]
    )
    results.print_mean_results()
    return results


def evaluate_model_against_evaluation_data(model, model_setup, config):
    X_train_eval, y_train_eval, X_test_eval, y_test_eval = get_xs_and_ys(
        config.evaluation_data_set_frames['evaluation_training_data_set'],
        config.evaluation_data_set_frames['evaluation_testing_data_set'],
        config.meta_columns + [config.y_label], config.y_label
    )

    results = evaluate_model(
        X_train_eval, y_train_eval,
        X_test_eval, y_test_eval,
        model,
        config.fitting_parameters[model_setup.name]
    )
    return results


def print_coefs(feature_names, model):
    for feature, coef in zip(feature_names, model.coef_[0]):
        print('%s - %f' % (feature, coef))


class ValidationResult(object):
    def __init__(self, accuracy, log_loss, roc_auc, average_prediction,
                 accuracy_for_ones, accuracy_for_zeros):
        self.acc = accuracy
        self.ll = log_loss
        self.roc_auc = roc_auc
        self.avg_p = average_prediction
        self.acc_1s = accuracy_for_ones
        self.acc_0s = accuracy_for_zeros


class ValidationResults(object):
    def __init__(self):
        self.results = []

    def append(self, acc, ll, roc_auc, avg_p, acc_1s, acc_0s):
        self.results.append(ValidationResult(
            acc, ll, roc_auc, avg_p, acc_1s, acc_0s
        ))

    def print_results(self):
        for r in self.results:
            self.print_result(r)

    @staticmethod
    def print_result(r):
        print(', '.join([
            'Accuracy: %.3f' % r.acc,
            'LL: %.5f' % r.ll,
            'ROC AUC: %.3f' % r.roc_auc,
            'Avg P: %.3f' % r.avg_p,
            'Acc 1s: %.3f' % r.acc_1s,
            'Acc 0s: %.3f' % r.acc_0s,
        ]))

    def get_accuracies(self):
        return np.array([r.acc for r in self.results])

    def get_log_losses(self):
        return np.array([r.ll for r in self.results])

    def get_roc_aucs(self):
        return np.array([r.roc_auc for r in self.results])

    def get_average_predictions(self):
        return np.array([r.avg_p for r in self.results])

    def get_1s_accuracries(self):
        return np.array([r.acc_1s for r in self.results])

    def get_0s_accuracries(self):
        return np.array([r.acc_0s for r in self.results])

    def get_mean_results(self):
        return {
            'acc': self.get_accuracies().mean(),
            'll': self.get_log_losses().mean(),
            'roc_auc': self.get_roc_aucs().mean(),
            'acc_1s': self.get_1s_accuracries().mean(),
            'acc_0s': self.get_0s_accuracries().mean(),

            # We can JSON seralize float64 but not float32.
            'avg_p': np.float64(self.get_average_predictions().mean())
        }

    def print_mean_results(self):
        accuracies = self.get_accuracies()
        log_losses = self.get_log_losses()
        roc_aucs = self.get_roc_aucs()

        print("Mean accuracy: %.3f (+/- %.3f)"
              % (accuracies.mean(), accuracies.std() * 2))
        print("Mean LL: %.5f (+/- %.3f)"
              % (log_losses.mean(), log_losses.std() * 2))
        print("Mean AUC: %.5f (+/- %.3f)"
              % (roc_aucs.mean(), roc_aucs.std() * 2))

    def validate_model_and_add_result(
            self, model, X_train, y_train, X_test, y_test,
            fitting_parameters, verbose=False):
        model.fit(
            X_train, y_train,
            **fitting_parameters
        )

        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        acc_1s = accuracy_score(y_test[y_test > 0.5],
                                preds[y_test.values > 0.5])
        acc_0s = accuracy_score(y_test[y_test < 0.5],
                                preds[y_test.values < 0.5])
        ll = log_loss(y_test, probs[:, 1])
        roc_auc = roc_auc_score(y_test, probs[:, 1])
        avg_p = probs[:, 1].mean()
        self.append(acc, ll, roc_auc, avg_p, acc_1s, acc_0s)

        if verbose:
            self.print_result(self.results[-1])


def kfolds_evaluation(X, y, model, fitting_parameters):
    kfr = ValidationResults()
    kf = StratifiedKFold(y, n_folds=3, shuffle=True, random_state=1)

    for train_index, test_index in kf:
        kfr.validate_model_and_add_result(
            model,
            X.iloc[train_index], y.iloc[train_index],
            X.iloc[test_index], y.iloc[test_index],
            fitting_parameters,
            verbose=True
        )
    return kfr


def evaluate_model(X_train, y_train,
                   X_test, y_test,
                   model, fitting_parameters):
    kfr = ValidationResults()
    kfr.validate_model_and_add_result(
        model, X_train, y_train, X_test, y_test, fitting_parameters)
    kfr.print_results()
    return kfr.get_mean_results()


def print_result_from_dict(result_dict):
    kfr = ValidationResults()
    kfr.append(**result_dict)
    kfr.print_results()


def train_make_and_save_submission(model_setup, config):
    print('\nTraining on the entire data set, and making predictions.')
    config.open_data_sets()
    X_train, y_train = get_x_and_y(
        config.data_set_frames['training_data_set'],
        config.meta_columns + [config.y_label], config.y_label)
    X_submission = get_x(
        config.data_set_frames['testing_data_set'],
        config.meta_columns + [config.y_label])

    id_column = config.data_set_frames['testing_data_set']['ID']
    make_and_save_submission(
        X_train, y_train,
        X_submission, id_column,
        model_setup.model,
        config.fitting_parameters[model_setup.name],
        config.submission_file_names[model_setup.name]
    )


def make_and_save_submission(X_train, y_train,
                             X_test, id_column,
                             model, fitting_parameters, file_name):
    model.fit(
        X_train, y_train,
        **fitting_parameters
    )

    predictions = model.predict_proba(X_test)
    predictions = pd.Series(predictions[:, 1], name='PredictedProb',
                            index=id_column.index)
    submission = pd.concat((id_column, predictions), axis=1)
    print('\nSaving submission file: %s' % file_name)
    submission.to_csv(file_name, sep=',', index=False)


def dump_model(model_json, file_name, results):
    print('Dumping model to JSON.')
    model_json['results'] = results

    with open(file_name, 'w') as f:
        json.dump(model_json, f, sort_keys=True, indent=4)
        f.write('\n')


def print_part_of_dict_as_indented_list(keys, dict_thing):
    part_of_dict_thing = dict((key, dict_thing[key]) for key in keys)
    print_dict_as_indented_list(part_of_dict_thing)


def print_dict_as_indented_list(dict_thing):
    for key, value in dict_thing.items():
        # if isinstance(value, str):
        #     value_str = value
        # elif isinstance(value, numbers.Integral):  # Int and Bool
        #     value_str = str(value)
        # elif isinstance(value, numbers.Real):  # (float)
        #     value_str = '%.3f' % value
        # else:
        #     raise RuntimeError('Type not supported: %s' % type(value))

        value_str = str(value)
        print(' - %s: %s' % (key, value_str))


def do_grid_search(model, param_grid, X, y):
    print('Running Grid Search.')
    gs = GridSearchCV(
        model, param_grid, scoring='log_loss', n_jobs=1, cv=1,
        # We can't use cv=1, this complicates things...
        verbose=10,
        fit_params={
            'eval_metric': 'logloss',
            'early_stopping_rounds': 50,
            'nfold': 3,
            'verbose': False
        }
    )
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
                err_str = (
                    'Chosen value for %s, of %s, has hit an edge of: %s' %
                    (key, str(value), str(options))
                )
                print(err_str)
                # raise ValueError(err_str)
