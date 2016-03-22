from gpml.model import model_maker
from . import configer
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression


def train_and_validate_model():
    print('Splitting data set in a local training and testing sets')
    config = configer.from_json('model/config.json')
    config.open_local_data_sets()

    train = config.local_data_set_frames['local_training_data_set']

    feature_names = train.columns.difference(config.not_x_labels)
    X = train[feature_names]
    y = train[config.y_label]

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

    kf_scores = []
    kf = KFold(y.shape[0], n_folds=5)

    for train_index, test_index in kf:
        print(test_index.shape[0] /
              (test_index.shape[0] + train_index.shape[0]))
              
        lr.fit(X.values[train_index, :], y['target'].values[train_index])
        score = lr.score(X.values[test_index], y.values[test_index])
        # model_maker.print_coefs(feature_names, lr)
        kf_scores.append(score)

    for score in kf_scores:
        print('LR score: %.2f' % score)

    print('Mean LR score: %.2f' % np.mean(kf_scores))
    print('Finished.')
