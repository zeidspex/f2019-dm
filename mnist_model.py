#%%
import numpy as np
import keras as ks
import tensorflow as tf
from keras.datasets import mnist
import copy

num_classes = 10

# input image dimensions
img_rows, img_cols, img_channels = 28, 28, 1

#%%
# Input layer
shared_layers = [ks.layers.InputLayer(input_shape=(img_rows, img_cols, img_channels))]

# Convolutional layers (5 x (Conv + Max Pool))
for n_filters in 64 * 2 ** np.array(range(2)):
    shared_layers.append(ks.layers.Conv2D(
        filters=n_filters, kernel_size=(3, 3),
        activation='relu', padding='same',
    ))
    shared_layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

# Dense layers
shared_layers += [
    ks.layers.Flatten(),
    ks.layers.Dense(1024),
]

encoder_layers = copy.copy(shared_layers)

encoder_layers += [
    ks.layers.Dense(7 * 7 * 128),
    ks.layers.Reshape((7, 7, 128))
]

# Deconvolutional layers (4 x Deconv + 1 x Deconv with 3 filters for the last layer)
for n_filters in list(128 // 2 ** np.array(range(1))) + [1]:
    encoder_layers.append(ks.layers.Deconvolution2D(
        filters=n_filters, kernel_size=(3, 3),
        activation='relu', padding='same', strides=(2, 2)
    ))

# Create and build the model from layers, print model info
encoder_model = ks.models.Sequential(encoder_layers)

classifier_layers = copy.copy(shared_layers)

classifier_layers += [
    ks.layers.Dense(256),
    ks.layers.Dense(num_classes, activation='softmax'),
]


# Create and build the model from layers, print model info
classifier_model = ks.models.Sequential(classifier_layers)
