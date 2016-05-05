import pandas as pd
import numpy as np
import datetime

# from gpml.model import model_maker
from gpml.data_set import data_set_maker
from . import distracted_driver_configer

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam


def run(project_dir):
    config = get_config(project_dir)
    # train_and_validate_simple_nn(config)
    train_and_validate_keras(config)
    # make_lr_submission(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def train_and_validate_keras(config):
    X, y, X_train, y_train, X_test, y_test, X_subm = load_data(config)

    model = Sequential([
        Convolution2D(
            16, 3, 3,
            border_mode='same', subsample=(2, 2),
            input_shape=config.image_size
        ),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        # MaxPooling2D(pool_size=(4, 4)),
        Dropout(0.2),
        Flatten(),
        Dense(30),
        Activation('relu'),
        Dropout(0.2),
        Dense(10),
        Activation('softmax')
    ])

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('\nEvaluating model.')
    model.fit(X_train, y_train, nb_epoch=2, batch_size=2**8)
    e = model.evaluate(
        X_test, y_test, batch_size=32, verbose=1, sample_weight=None)
    print('Evaluation loss: %0.4f, acc: %0.4f' % (e[0], e[1]))

    print('\nTraining full model.')
    model.fit(X, y, nb_epoch=2, batch_size=2**8)

    print('\nMaking and saving a submission.')
    predictions = model.predict_proba(X_subm, batch_size=32, verbose=0)
    make_submission_from_preductions(
        predictions, config.sample_submission,
        config.submission_dir, config.submission_file_names['SimpleNet']
    )


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
    X_train = data_set_maker.open_array_from_hdf(
        config.data_sets['training_images'])
    # X_eval = np.expand_dims(X_eval, axis=1)

    X_eval_train, y_eval_train = split_training_subset(
        X_train,
        config.evaluation_imgs_list['train']
    )

    X_eval_test, y_eval_test = split_training_subset(
        X_train,
        config.evaluation_imgs_list['test']
    )

    X_subm = data_set_maker.open_array_from_hdf(
        config.data_sets['testing_images'])
    # X_subm = np.expand_dims(X_subm, axis=1)

    y_train = np.vstack((y_eval_train, y_eval_test))
    return (
        X_train, y_train,
        X_eval_train, y_eval_train,
        X_eval_test, y_eval_test,
        X_subm
    )


def split_training_subset(X, list_path):
    training_list = pd.read_csv(list_path)
    X = X[training_list.index]
    y = convert_list(training_list['classname'])
    y = np_utils.to_categorical(y, 10)
    return X, y


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
