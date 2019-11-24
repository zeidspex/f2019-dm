######################################
########## Import Functions ##########

import re
import h5py
import json
import random
import tarfile
import itertools
import visualization
import classification
import fashion_utils
import numpy as np
import keras as ks
import tensorflow as tf
import sklearn.model_selection as model_selection


#######################################
#### Variable Initalizations ##########

batch_size = 128
num_classes=10
seed=66
val_size=0.2

##Autoencoder variables
e_model_path = 'data/model_FashionMnist.h5'
e_epochs = 1
e_patience = 40
e_learn_rate = 0.0001

##Classification variables
c_model_path = 'data/model_FashionMnist_classifier.h5'
c_epochs = 1
c_patience = 40
c_learn_rate = 0.001


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

# #######################################
# ####### Encoder Model Creation#########

e_model_layers = fashion_utils.get_fashion_encoder_model()
# Create and build the model from layers, print model info
encoder_model = ks.models.Sequential(e_model_layers)
encoder_model.compile(
    optimizer=ks.optimizers.Adam(lr=e_learn_rate),
    loss=ks.losses.mean_squared_error
)
encoder_model.summary()

##################################################
######### Train the AutoEncoder Model ############

# Train model
encoder_model.fit(unlabeled_train_images, unlabeled_train_images,
    validation_data=(unlabeled_val_images, unlabeled_val_images),
    batch_size=batch_size,  shuffle='batch',
    epochs=e_epochs,
    verbose=1,
    callbacks=[
        ks.callbacks.ModelCheckpoint(
            filepath=e_model_path, monitor='val_loss',
            verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5
        ),
        ks.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=e_patience,
            verbose=0, mode='auto'
        ),
        ks.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_delta=1E-7
        )

    ]
)

#Test Model
encoder_model.load_weights(e_model_path)
score = encoder_model.evaluate(test_images, test_images, verbose=0)
print('Test loss for the trained encoder model:')
print(score)



#######################################################
#########Classifier from Autoencoder model############

# c_model_layers = fashion_utils.get_fashion_classifier_model()
# classifier_model = ks.models.Sequential(c_model_layers)
# classifier_model.compile(
#     optimizer=ks.optimizers.Adam(lr=c_learn_rate),
#     loss=ks.losses.categorical_crossentropy,
#     metrics=['accuracy']
# )
#
#encoder_model.summary()
#Train Model

#Test Model



#######################################
######### Visualize Results ############
