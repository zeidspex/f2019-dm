import numpy as np
import keras as ks
import tensorflow as tf
from keras.datasets import mnist
import copy

encoding_size = 36
dense_size = 64
conv_encoder = 16
conv_decoder = 16

model_path = 'data/model_mnist.h5'
num_classes = 10

# input image dimensions
img_rows, img_cols, img_channels = 28, 28, 1


def _get_shared_layers():
    shared_layers = []

    shared_layers.append(ks.layers.Conv2D(
        filters=conv_encoder, kernel_size=(3, 3),
        padding='same', use_bias=False,
        input_shape=(img_rows, img_cols, img_channels),
    ))
    shared_layers.append(ks.layers.BatchNormalization())
    shared_layers.append(ks.layers.Activation('relu'))
    shared_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layers
    for n_filters in list(conv_encoder*2 * 2 ** np.array(range(1))):
        shared_layers.append(ks.layers.Conv2D(
            filters=n_filters, kernel_size=(3, 3),
            padding='same', use_bias=False,
        ))
        shared_layers.append(ks.layers.BatchNormalization())
        shared_layers.append(ks.layers.Activation('relu'))
        shared_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

    shared_layers.append(ks.layers.Conv2D(
        filters=dense_size, kernel_size=(3, 3),
        padding='same', use_bias=False,
    ))
    shared_layers.append(ks.layers.BatchNormalization())
    shared_layers.append(ks.layers.Activation('relu'))

    shared_layers.append(ks.layers.Conv2D(
        filters=dense_size, kernel_size=(3, 3),
        padding='same', use_bias=False,
    ))
    shared_layers.append(ks.layers.BatchNormalization())
    shared_layers.append(ks.layers.Activation('relu'))

    # Dense layers
    shared_layers += [
        ks.layers.Flatten(),
        ks.layers.Dense(encoding_size, use_bias=False),
        ks.layers.BatchNormalization(),
        ks.layers.Activation('relu'),
    ]

    return shared_layers


def _get_encoder_layers():
    encoder_layers = _get_shared_layers()

    encoder_layers += [
        ks.layers.Dense(7 * 7 * dense_size, use_bias=False),
        ks.layers.BatchNormalization(),
        ks.layers.Activation('relu'),
        ks.layers.Reshape((7, 7, dense_size))
    ]

    encoder_layers.append(ks.layers.Deconvolution2D(
        filters=dense_size, kernel_size=(3, 3),
        padding='same', use_bias=False,
    ))
    encoder_layers.append(ks.layers.BatchNormalization())
    encoder_layers.append(ks.layers.Activation('relu'))

    # Deconvolutional layers (4 x Deconv + 1 x Deconv with 3 filters for the last layer)
    for n_filters in list(conv_decoder // 2 ** np.array(range(2))):
        encoder_layers.append(ks.layers.Deconvolution2D(
            filters=n_filters, kernel_size=(3, 3),
            padding='same', strides=(2, 2), use_bias=False,
        ))
        encoder_layers.append(ks.layers.BatchNormalization())
        encoder_layers.append(ks.layers.Activation('relu'))

    encoder_layers.append(ks.layers.Deconvolution2D(
        filters=1, kernel_size=(3, 3),
        padding='same', use_bias=False
    ))
    encoder_layers.append(ks.layers.BatchNormalization())
    encoder_layers.append(ks.layers.Activation('relu'))

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

    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Deconvolution2D(
        filters=8, kernel_size=(3, 3),
        padding='same', use_bias=False,
    ))
    classifier_model.add(ks.layers.BatchNormalization())
    classifier_model.add(ks.layers.Activation('relu'))

    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Deconvolution2D(
        filters=8, kernel_size=(3, 3),
        padding='same', use_bias=False,
    ))
    classifier_model.add(ks.layers.BatchNormalization())
    classifier_model.add(ks.layers.Activation('relu'))

    classifier_model.add(ks.layers.Flatten())
    # classifier_model.add(ks.layers.Dropout(0.2))
    # classifier_model.add(ks.layers.Dense(196, use_bias=False))
    # classifier_model.add(ks.layers.BatchNormalization())
    # classifier_model.add(ks.layers.Activation('relu'))
    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Dense(40, use_bias=False))
    classifier_model.add(ks.layers.BatchNormalization())
    classifier_model.add(ks.layers.Activation('relu'))
    classifier_model.add(ks.layers.Dropout(0.2))
    classifier_model.add(ks.layers.Dense(num_classes, use_bias=False))
    classifier_model.add(ks.layers.BatchNormalization())
    classifier_model.add(ks.layers.Activation('softmax'))

    return classifier_model
