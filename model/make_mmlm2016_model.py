import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from . import configer


def make_mmlm2016_model():
    # TODO:
    # 1. Win ratio by year.
    # 2. Score difference
    # 3. Seeds or seed difference?
    # 4. Ensemble with other Kagglers shared results.

    config = configer.from_json('model/config_mmlm2016.json')
    basic_data_set = pd.read_hdf(config.basic_data_set_file_name,
                                 key='table')

    X = basic_data_set[
        ['win_ratio_team1', 'win_ratio_team2', 'win_ratio_difference']
    ].values
    y = basic_data_set['result'].values

    # TODO Seperate out a 'final test' set from cv data

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

    submission_data_set = pd.read_hdf(config.submission_data_set_file_name,
                                      key='table')

    X_submission = submission_data_set[
        ['win_ratio_team1', 'win_ratio_team2', 'win_ratio_difference']
    ].values

    predictions = lr.predict_proba(X_submission)
    submission_data_set['Pred'] = predictions[:, 1]
    submission_data_set[['Id', 'Pred']].to_csv(
        config.data_dir + 'submission.csv', sep=',', index=False)

    print('Submission predictions example:')
    print(type(submission_data_set))
    print(submission_data_set[0:2])
    print('Finished.')
