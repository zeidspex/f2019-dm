######################################
########## Import Functions ##########

import re
import h5py
import json
import random
import tarfile
import itertools
import testing
import clustering
import visualization
import classification
import numpy as np
import keras as ks
import tensorflow as tf
import sklearn.model_selection as model_selection


#######################################
#### Variable Initalizations ##########
model_path = 'data/model_FashionMnist.h5'
num_classes=10
seed=66
val_size=0.2
batch_size = 128
epochs = 10
patience = 40
learn_rate = 0.0001

#######################################
#######Data Preparing##################

#loading data, Fashion Mnist beging with 60,000 greyscale images size 28x28 for training
#10,000 testing images

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#reshape training and testing sets
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255
train_labels = ks.utils.to_categorical(train_labels, num_classes)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255
test_labels = ks.utils.to_categorical(test_labels, num_classes)



# Split data into testing and validations sets
unlabeled_train_images, unlabeled_val_images = model_selection.train_test_split(
    train_images, test_size=val_size, random_state=seed
)

print("Train images are of size: {0}, \n Train labels are of size {1}\n".format(train_images.shape,train_labels.shape))
print("Test images are of size: {0}, \n Test labels are of size {1}\n".format(test_images.shape,test_labels.shape))

#######################################
####### Encoder Model Creation#########
# Input layer
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

# Create and build the model from layers, print model info
encoder_model = ks.models.Sequential(model_layers)
encoder_model.compile(
    optimizer=ks.optimizers.Adam(lr=learn_rate),
    loss=ks.losses.mean_squared_error
)
encoder_model.summary()

#######################################
######### Train the Model ############

# Train model
encoder_model.fit(unlabeled_train_images, unlabeled_train_images,
    validation_data=(unlabeled_val_images, unlabeled_val_images),
    batch_size=batch_size,  shuffle='batch',
    epochs=epochs,
    verbose=1,
    callbacks=[
        ks.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss',
            verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5
        ),
        ks.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=patience,
            verbose=0, mode='auto'
        ),
        ks.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_delta=1E-7
        )

    ]
)


encoder_model.load_weights(model_path)
score = encoder_model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:')
print(score)
