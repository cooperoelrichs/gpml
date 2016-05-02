import pandas as pd
import numpy as np
import time

import os
import glob
import math

# from gpml.model import model_maker
from . import distracted_driver_configer

import cv2

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import lasagne
from lasagne.updates import adam
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer
from lasagne.layers import Conv2DLayer as ConvLayer

import theano
from theano import tensor as T


def run(project_dir):
    config = get_config(project_dir)
    train_and_validate_simple_nn(config)
    # make_lr_submission(config)


def get_config(project_dir):
    return distracted_driver_configer.from_json(
        'sf_dd/model/config.json', project_dir)


def train_and_validate_simple_nn(config):
    """
    Simple Lasagne NN.

    Based on:
    https://www.kaggle.com/florianm/state-farm-distracted-driver-detection/theano-lasange-starter
    """
    pixels = 24 * 2
    image_size = pixels * pixels
    num_features = image_size

    batch_size = 32
    learning_rate = 0.001 / 2
    iterations = 60

    X = T.tensor4('X')
    Y = T.ivector('y')

    # Set up theano functions to generate output by feeding data
    # through network, any test outputs should be deterministic.
    output_layer = ZFTurboNet(X, pixels)
    output_train = lasagne.layers.get_output(output_layer)
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)

    # Set up the loss that we aim to minimize, when using cat cross
    # entropy our Y should be ints not one-hot.
    loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
    loss = loss.mean()

    # Set up loss functions for validation dataset.
    valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
    valid_loss = valid_loss.mean()

    valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y),
                       dtype=theano.config.floatX)

    # Get parameters from network and set up sgd with nesterov momentum
    # to update parameters.
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = adam(loss, params, learning_rate=learning_rate)

    # set up training and prediction functions
    train_fn = theano.function(inputs=[X, Y], outputs=loss, updates=updates)
    valid_fn = theano.function(inputs=[X, Y], outputs=[valid_loss, valid_acc])

    # set up prediction function
    predict_proba = theano.function(inputs=[X], outputs=output_test)

    # Load training data and start training.
    encoder = LabelEncoder()

    # Load the training and validation data sets.
    train_X, train_y, valid_X, valid_y, encoder = load_train_cv(
        config, encoder, pixels, num_features)
    print('Train shape:', train_X.shape, 'Test shape:', valid_X.shape)

    # load data
    X_test, X_test_id = load_test(config, pixels, num_features)

    # Loop over training functions for however many iterations,
    # print information while training.
    try:
        for epoch in range(iterations):
            # do the training
            start = time.time()
            # training batches
            train_loss = []
            for batch in iterate_minibatches(train_X, train_y, batch_size):
                inputs, targets = batch
                train_loss.append(train_fn(inputs, targets))
            train_loss = np.mean(train_loss)
            # validation batches
            valid_loss = []
            valid_acc = []
            for batch in iterate_minibatches(valid_X, valid_y, batch_size):
                inputs, targets = batch
                valid_eval = valid_fn(inputs, targets)
                valid_loss.append(valid_eval[0])
                valid_acc.append(valid_eval[1])
            valid_loss = np.mean(valid_loss)
            valid_acc = np.mean(valid_acc)
            # get ratio of TL to VL
            ratio = train_loss / valid_loss
            end = time.time() - start
            # print training details
            print(
                'iter:', epoch,
                '| TL:', np.round(train_loss, decimals=3),
                '| VL:', np.round(valid_loss, decimals=3),
                '| Vacc:', np.round(valid_acc, decimals=3),
                '| Ratio:', np.round(ratio, decimals=2),
                '| Time:', np.round(end, decimals=1)
            )

    except KeyboardInterrupt:
        pass

    # make predictions
    print('Making predictions')
    prediction_batch = 2

    predictions = []
    for pred_batch in iterate_pred_minibatches(X_test, prediction_batch):
        predictions.extend(predict_proba(pred_batch))
    predictions = np.array(predictions)

    print('pred shape')
    print(predictions.shape)

    print('Creating Submission')
    create_submission(config, predictions, X_test_id)


def create_submission(config, predictions, test_id):
    columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    result1 = pd.DataFrame(predictions, columns=columns)
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv(config.submission_file_names["ZFTurboNet"], index=False)


def iterate_pred_minibatches(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


def load_train_cv(config, encoder, pixels, num_features):
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        # path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')
        path = os.path.join(config.image_dirs['train'], 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        print('Load folder c%i: %i images' % (j, len(files)))
        for fl in files:
            img = cv2.imread(fl, 0)

            if img is None:
                raise RuntimeError('Image reading failed: %s' % fl)

            img = cv2.resize(img, (pixels, pixels))
            # img = img.transpose(2, 0, 1)
            img = np.reshape(img, (1, num_features))
            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print('Memory size (array.nbytes): %i MB' % (X_train.nbytes / 1024**2))

    y_train = encoder.fit_transform(y_train).astype('int32')
    X_train, y_train = shuffle(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(
        X_train.shape[0], 1, pixels, pixels
    ).astype('float32') / 255.0

    X_test = X_test.reshape(
        X_test.shape[0], 1, pixels, pixels
    ).astype('float32') / 255.0

    return X_train, y_train, X_test, y_test, encoder


def load_test(config, pixels, num_features):
    print('Read test images')
    # path = os.path.join('..', 'input', 'test', '*.jpg')
    path = os.path.join(config.image_dirs['test'], '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl, 0)
        img = cv2.resize(img, (pixels, pixels))
        # img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)
    print('Memory size (array.nbytes): %i MB' % (X_test.nbytes / 1024**2))

    X_test = X_test.reshape(
        X_test.shape[0], 1, pixels, pixels).astype('float32') / 255.

    return X_test, X_test_id


def ZFTurboNet(input_var, pixels):
    """Lasagne Model ZFTurboNet and Batch Iterator."""
    l_in = InputLayer(shape=(None, 1, pixels, pixels), input_var=input_var)

    l_conv = ConvLayer(
        l_in, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify
    )
    l_convb = ConvLayer(
        l_conv, num_filters=8, filter_size=3, pad=1, nonlinearity=rectify
    )
    l_pool = MaxPool2DLayer(l_convb, pool_size=2)  # feature maps 12x12

    # l_dropout1 = DropoutLayer(l_pool, p=0.25)
    l_hidden = DenseLayer(l_pool, num_units=128, nonlinearity=rectify)
    # l_dropout2 = DropoutLayer(l_hidden, p=0.5)

    l_out = DenseLayer(l_hidden, num_units=10, nonlinearity=softmax)

    return l_out


def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]
