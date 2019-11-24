import re
import h5py
import json
import random
import tarfile
import itertools
import visualization
import classification
import numpy as np
import keras as ks
import tensorflow as tf
import sklearn.model_selection as model_selection



def get_fashion_encoder_model():
    
    model_layers = [ks.layers.InputLayer(input_shape=(28, 28,1))]

    # Convolutional layers (5 x (Conv + Max Pool))
    for n_filters in 64 * 2 ** np.array(range(2)):
        model_layers.append(ks.layers.Conv2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same',
        ))
        model_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    model_layers += [
        ks.layers.Flatten(),
        ks.layers.Dense(7 * 7 * 128),
        ks.layers.Reshape((7, 7, 128))
    ]

    # Deconvolutional layers
    for n_filters in list(128 // 2 ** np.array(range(1))) + [1]:
        model_layers.append(ks.layers.Deconvolution2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same', strides=(2, 2)
        ))

    return model_layers
