import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


def make_mmlm2016_model():
    project_dir = '/Users/cooperoelrichs/Projects/Kaggle/'
    data_dir = project_dir + 'march-machine-learning-mania-2016-v1/'
    basic_data_set_file_name = 'basic_data_set.h5'

    basic_data_set = pd.read_hdf(data_dir + basic_data_set_file_name,
                                 key='table')

    X = basic_data_set[
        ['win_ratio_team1', 'win_ratio_team2', 'win_ratio_difference']
    ].values
    y = basic_data_set['result'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1)

    lr = LogisticRegression(
        penalty='l1',
        C=0.1,
        class_weight='balanced',
        max_iter=100,
        random_state=1,
        solver='lbfgs',
        tol=0.000001,
        # n_jobs=-1
    )

    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)

    print('LR Score: %0.2f' % score)

    # TODO Seperate out a 'final test' set from cv data
    # submission = lr.predict_proba(X_submission)
    # save_submission(submission)
