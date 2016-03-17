import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from . import configer


def x_column_names(data_set):
    return data_set.columns.difference(
        ['result', 'team1', 'team2', 'Id', 'year'])


def get_data(config):
    basic_data_set = pd.read_hdf(config.basic_data_set_file_name,
                                 key='table')

    column_names = x_column_names(basic_data_set)
    X = basic_data_set[column_names].values
    y = basic_data_set['result'].values
    return X, y, column_names


def make_predictions(config, lr, submission_data_set):
    column_names = x_column_names(submission_data_set)
    X_submission = submission_data_set[column_names].values

    predictions = lr.predict_proba(X_submission)
    return predictions


def print_coefs(feature_names, lr):
    for feature, coef in zip(feature_names, lr.coef_[0]):
        print('%s - %.3f' % (feature, coef))


def save_submission(lr, config):
    submission_file = config.data_dir + 'submission.csv'
    print('Saving submission - %s' % submission_file)

    submission_data_set = pd.read_hdf(config.submission_data_set_file_name,
                                      key='table')
    predictions = make_predictions(config, lr, submission_data_set)
    submission_data_set['Pred'] = predictions[:, 1]
    submission_data_set[['Id', 'Pred']].to_csv(
        submission_file, sep=',', index=False)


def make_mmlm2016_model():
    # TODO:
    # 0. Difference from tournament year.
    # 1. Seeds or seed difference?
    # 2. Ensemble with other Kagglers shared results.

    config = configer.from_json('model/config_mmlm2016.json')
    X, y, feature_names = get_data(config)

    # TODO Seperate out a 'final test' set from cv data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1)

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

    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print_coefs(feature_names, lr)
    print('LR Score: %.2f' % score)

    save_submission(lr, config)

    # print('Submission predictions example:')
    # print(type(submission_data_set))
    # print(submission_data_set[0:2])
    print('Finished.')
