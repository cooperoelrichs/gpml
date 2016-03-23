import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from gpml.data_set import data_set_maker


def print_coefs(feature_names, lr):
    for feature, coef in zip(feature_names, lr.coef_[0]):
        print('%s - %.3f' % (feature, coef))


def basic_lr():
    lr = LogisticRegression(
        penalty='l1',
        C=0.01,
        class_weight='balanced',
        max_iter=100,
        random_state=1,
        solver='lbfgs',
        tol=0.000001,
        # n_jobs=-1
    )
    return lr


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
            print('LR accuracy: %.3f, ll: %.3f' % (r.acc, r.ll))

    def get_accuracies(self):
        return np.array([r.acc for r in self.results])

    def get_log_losses(self):
        return np.array([r.ll for r in self.results])

    def print_mean_results(self):
        accuracies = self.get_accuracies()
        log_losses = self.get_log_losses()

        print("Mean LR accuracy: %0.3f (+/- %0.2f)"
              % (accuracies.mean(), accuracies.std() * 2))
        print("Mean LR LL: %0.3f (+/- %0.2f)"
              % (log_losses.mean(), log_losses.std() * 2))

    def add_validation_result(self, model,
                              X_train, y_train,
                              X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)

        # print(np.column_stack((y_test, predictions[:, 0], predictions[:, 1])))

        acc = model.score(X_test, y_test)
        ll = log_loss(y_test,
                      predictions,
                      eps=10 ^ -15)
        self.append(acc, ll)


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


def make_and_save_submission(X_train, y_train,
                             X_test, id_column,
                             model, file_name):
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    predictions = pd.Series(predictions[:, 1], name='PredictedProb',
                            index=id_column.index)
    submission = pd.concat((id_column, predictions), axis=1)
    submission.to_csv(file_name, sep=',', index=False)
