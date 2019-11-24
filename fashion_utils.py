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

encoding_size = 15

def get_fashion_encoder_model():

    model_layers = [ks.layers.InputLayer(input_shape=(28, 28,1))]

    # Convolutional layers (5 x (Conv + Max Pool))
    for n_filters in 32 * 2 ** np.array(range(2)):
        model_layers.append(ks.layers.Conv2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same',
        ))
        model_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    model_layers += [
        ks.layers.Flatten(),
        ks.layers.Dense(392, activation='relu'),
        ks.layers.Dense(196, activation='relu'),
        ks.layers.Reshape((7, 7, 4))
    ]

    # Deconvolutional layers
    for n_filters in list(128 // 2 ** np.array(range(1))) + [1]:
        model_layers.append(ks.layers.Deconvolution2D(
            filters=n_filters, kernel_size=(3, 3),
            activation='relu', padding='same', strides=(2, 2)
        ))

    return model_layers

def get_fashion_classifier_model(model_path, num_classes):

    # Create and build the model from layers, print model info
    fashion_class_model = ks.models.Sequential()
    # fashion_class_model = [ks.layers.InputLayer(input_shape=(28, 28,1))]

    weights_model = ks.models.Sequential(get_fashion_encoder_model())
    weights_model.load_weights(model_path)

    for layer in weights_model.layers[:len(get_fashion_encoder_model())]:
        layer.trainable = False
        fashion_class_model.add(layer)


    fashion_class_model.add(ks.layers.Dropout(0.2))
    fashion_class_model.add(ks.layers.Dense(40, activation='relu'))
    fashion_class_model.add(ks.layers.Dropout(0.2))
    fashion_class_model.add(ks.layers.Dense(num_classes, activation='softmax'))
    


    return fashion_class_model
