import numpy as np
import keras as ks
import tensorflow as tf
from keras.datasets import mnist
import copy

encoding_size = 15
conv_encoder = 32
conv_decoder = 128

model_path = 'data/model_mnist.h5'
num_classes = 10

# input image dimensions
img_rows, img_cols, img_channels = 28, 28, 1


def _get_shared_layers():
    # Input layer
    shared_layers = [ks.layers.InputLayer(input_shape=(img_rows, img_cols, img_channels))]

    # Convolutional layers (5 x (Conv + Max Pool))
    for n_filters in list(conv_encoder * 2 ** np.array(range(2))): #+ [encoding_size]:
        shared_layers.append(ks.layers.Conv2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same',
        ))
        shared_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    shared_layers += [
        # ks.layers.Flatten(),
        ks.layers.GlobalAveragePooling2D(),
        ks.layers.Dense(encoding_size, activation='relu'),
    ]

    return shared_layers


def _get_encoder_layers():
    encoder_layers = _get_shared_layers()

    encoder_layers += [
        ks.layers.Dense(7 * 7 * 8, activation='relu'),
        ks.layers.Reshape((7, 7, 8))
    ]

    # Deconvolutional layers (4 x Deconv + 1 x Deconv with 3 filters for the last layer)
    for n_filters in list(conv_decoder // 2 ** np.array(range(1))) + [1]:
        encoder_layers.append(ks.layers.Deconvolution2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same', strides=(2, 2)
        ))

    return encoder_layers


def get_encoder():
    # Create and build the model from layers, print model info
    encoder_model = ks.models.Sequential(_get_encoder_layers())

    # encoder_model.load_weights(model_path)

    return encoder_model


def get_classifier():
    # Create and build the model from layers, print model info
    classifier_model = ks.models.Sequential()

    weights_model = ks.models.Sequential(_get_encoder_layers())
    weights_model.load_weights(model_path)

    for layer in weights_model.layers[:len(_get_shared_layers())]:
        layer.trainable = False
        classifier_model.add(layer)

    # classifier_model.add(ks.layers.Flatten())
    # classifier_model.add(ks.layers.Dense(196, activation='relu'))
    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Dense(40, activation='relu'))
    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Dense(num_classes, activation='softmax'))

    return classifier_model
