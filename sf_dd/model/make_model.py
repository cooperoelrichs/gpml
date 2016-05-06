import pandas as pd
import numpy as np
import datetime
from keras.utils import np_utils
# from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold

from gpml.data_set import data_set_maker
# from gpml.model import model_maker
from . import distracted_driver_configer
from gpml.model.best_only_early_stopping import BestOnlyEarlyStopping

from gpml.model.keras_models import SimpleConvNet


def run(project_dir):
    config = get_config(project_dir)
    # train_and_validate_simple_nn(config)
    train_and_validate_simple_conv_net(config)
    # make_lr_submission(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def train_and_validate_simple_conv_net(config):
    # TODO:
    # Why is LB score so bad?
    #
    # Rotate images by random(-10, 10).
    # Centre the pixel average (-= X.mean) - test this.
    # Centre the average of each image (-= img.mean) - test this.
    # Submission based on CV average.

    model_setup = SimpleConvNet(config)
    train_and_validate(config, model_setup)


def train_and_validate(config, model_setup):
    X, y, X_subm, image_list = load_data(config)

    print('Cross Validating model.')
    cross_validate_model(X, y, model_setup, config, image_list)

    print('\nTraining full model with %i epochs.' % model_setup.nb_epoch)
    # callbacks = [EarlyStopping(monitor='train_loss', patience=1, verbose=0)]
    model_setup.regenerate_model()
    model_setup.model.fit(
        X, y,
        batch_size=model_setup.batch_size, nb_epoch=model_setup.nb_epoch,
        shuffle=True, verbose=2,
        # callbacks=callbacks
    )

    print('\nMaking and saving a submission.')
    predictions = model_setup.model.predict_proba(
        X_subm, batch_size=model_setup.batch_size, verbose=2
    )
    make_submission_from_preductions(
        predictions, config.sample_submission,
        config.submission_dir, config.submission_file_names[model_setup.name]
    )


def cross_validate_model(X, y, model_setup, config, image_list):
    unique_subjects = np.unique(image_list['subject'].values)
    kf = KFold(
        len(unique_subjects), n_folds=model_setup.cv_folds,
        shuffle=True, random_state=1
    )

    cv_results = np.empty((model_setup.cv_folds, 2))
    print('Starting CV with %i folds:' % model_setup.cv_folds)
    for i, (train_subjects_indicies, test_subjects_indicies) in enumerate(kf):
        train_subjects = unique_subjects[train_subjects_indicies]
        test_subjects = unique_subjects[test_subjects_indicies]
        X_train, y_train = split_on_subject(X, y, image_list, train_subjects)
        X_test, y_test = split_on_subject(X, y, image_list, test_subjects)

        callbacks = [
            BestOnlyEarlyStopping(
                monitor='val_loss',
                patience=model_setup.es_patience,
                verbose=0
            )
            # ModelCheckpoint(path, monitor='val_loss', save_best_only=True)
        ]

        model_setup.regenerate_model()
        model_setup.model.fit(
            X_train, y_train,
            batch_size=model_setup.batch_size,
            nb_epoch=model_setup.es_max_epoch,
            shuffle=True, verbose=2, validation_data=(X_test, y_test),
            callbacks=callbacks
        )

        results = model_setup.model.evaluate(
            X_test, y_test, batch_size=model_setup.batch_size,
            verbose=2, sample_weight=None
        )

        cv_results[i, :] = results
        print('%i. loss: %0.4f, acc: %0.4f' % (i, results[0], results[1]))

    print('Mean CV loss: %0.4f, acc: %0.4f' % tuple(cv_results.mean(axis=0)))


def split_on_subject(X, y, image_list, subjects):
    subset = image_list['subject'].isin(subjects).values
    X_subset = X[subset]
    y_subset = y[subset]
    return X_subset, y_subset


def make_submission_from_preductions(
        predictions, sample_submission_path,
        submission_dir, submission_name):
    sample_submission = pd.read_csv(sample_submission_path)
    submission = pd.DataFrame(
        data=predictions,
        columns=sample_submission.columns[1:], index=sample_submission.index
    )
    submission['img'] = sample_submission['img']

    date_time_str = get_datetime_str()
    file_name = '%s_%s.csv' % (submission_name, date_time_str)
    submission.to_csv(file_name, sep=',', index=False)


def load_data(config):
    X = data_set_maker.open_array_from_hdf(
        config.data_sets['training_images'])
    y, img_list = load_y_and_img_list(config.driver_imgs_list)

    X_subm = data_set_maker.open_array_from_hdf(
        config.data_sets['testing_images'])

    print('X: %s' % str(X.shape))
    print('y: %s' % str(y.shape))
    return X, y, X_subm, img_list


def load_y_and_img_list(list_path):
    img_list = load_img_list(list_path)
    y = convert_list(img_list['classname'])
    y = np_utils.to_categorical(y, 10)
    return y, img_list


def split_training_subset(X, list_path):
    y, training_img_list = load_y_and_img_list(list_path)
    X = X[training_img_list.index]
    return X, y


def load_img_list(list_path):
    return pd.read_csv(list_path)


def convert_list(class_names):
    class_conversion = {
        'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
        'c5': 5, 'c6': 6, 'c7': 7, 'c8': 9, 'c9': 9
    }

    y = class_names.map(
        lambda x: class_conversion[x]
    ).values
    return y


def get_datetime_str():
    now = datetime.datetime.now()
    datetime_str = str(now.strftime("%Y-%m-%d-%H-%M"))
    return datetime_str
