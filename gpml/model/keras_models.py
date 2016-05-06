import numpy as np
from math import floor
from .model_setup import ModelSetup

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


class SimpleConvNet(ModelSetup):
    """Simple Keras Convolutional Neural Network."""

    def __init__(self, config):
        name = 'SimpleConvNet'
        self.nb_epoch = floor(np.mean([4, 12, 12, 12, 20]))
        self.es_max_epoch = 50
        self.batch_size = 64
        self.image_size = config.image_size

        self.cv_folds = 5
        self.es_patience = 3

        super().__init__(name, config)

    def make_model(self, _):
        model = Sequential([
            Convolution2D(
                32, 3, 3, border_mode='same', init='he_normal',
                input_shape=self.image_size
            ),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),

            Convolution2D(64, 3, 3, border_mode='same', init='he_normal'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),

            Convolution2D(128, 3, 3, border_mode='same', init='he_normal'),
            MaxPooling2D(pool_size=(8, 8)),
            Dropout(0.5),

            Flatten(),
            Dense(10),
            Activation('softmax')
        ])

        model.compile(
            Adam(lr=1e-3), loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def regenerate_model(self):
        print('Regenerating model.')
        self.model = self.make_model(None)

    def dump_model(self, model, results, file_name):
        raise NotImplementedError()

    def load_model(self, file_name):
        raise NotImplementedError()
