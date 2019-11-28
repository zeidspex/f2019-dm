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
import warnings
import mnist_model

# Set allow_growth to true to avoid memory hogging
ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

#######################################
#### Variable Initalizations ##########

batch_size = 32
num_classes=10
seed=66
val_size=0.2
labeled_samples_per_class = 10

np.seterr(divide='ignore', invalid='ignore')
x = np.ones((1000, 1000)) * np.nan

classifier_path = r'models/mnist/clf.tar'
figure_out_path = r'figures/mnist-confusion.png'
results_out_path = 'results/mnist.csv'

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    foo = np.nanmean(x, axis=1)
#######################################
#######Data Preparing##################

#loading data, Fashion Mnist beging with 60,000 greyscale images size 28x28 for training
#10,000 testing images

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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
print(" unlabeledimages are of size: {0}, \n Train labels are of size {1}\n".format(unlabeled_train_images.shape, unlabeled_val_images.shape))

#%%
train_labels = np.array([np.argmax(x) for x in train_labels])
test_labels = np.array([np.argmax(x) for x in test_labels])

####################################################################################################
# Create classifier
####################################################################################################

#%%
model = mnist_model.get_encoder(weights=True)
clf = classification.create_model(
    model, mnist_model.get_embeddings_index(), train_images.reshape((-1, 28, 28, 1)), train_labels,
    labeled_samples_per_class
)
classification.save_model(clf, classifier_path)

#%%
####################################################################################################
# Test classifier
####################################################################################################

clf = classification.load_model(classifier_path)
class_names = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
]
classification.test_model(
    clf, test_images.reshape((-1, 28, 28, 1)), test_labels, class_names, results_out_path
)

####################################################################################################
# Visualize results
####################################################################################################

yp_test = clf.predict(test_images.reshape((-1, 28, 28, 1)))
visualization.visualize_confusion_matrix(
    test_images.reshape((-1, 28, 28, 1)), test_labels, yp_test, figure_out_path
)
