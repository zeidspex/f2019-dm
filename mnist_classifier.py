from __future__ import print_function
import keras as ks
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import h5py
import mnist_model

model_path = 'data/model_mnist_classifier.h5'
batch_size = 128
num_classes = 10
epochs = 500
patience = 40
learn_rate = 0.0001

# input image dimensions
img_rows, img_cols, img_channels = 28, 28, 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = ks.utils.to_categorical(y_train, num_classes)
y_test = ks.utils.to_categorical(y_test, num_classes)

# Create and build the model from layers, print model info
model = mnist_model.classifier_model
model.compile(
    optimizer=ks.optimizers.Adam(lr=learn_rate),
    loss=ks.losses.categorical_crossentropy,
    metrics=['accuracy']
)
model.summary()


#%%
# Restore prepared data in HDF5 file
with h5py.File('data/mnist.hdf5', 'r') as data_file:
    seed = data_file['seed']
    num_classes = data_file['num_classes']
    ux_train = data_file['ux_train']
    ux_val = data_file['ux_val']
    lx_train = data_file['x_train']
    ly_train = data_file['y_train']
    lx_val = data_file['x_val']
    ly_val = data_file['y_val']
    x_test = data_file['x_test']
    y_test = data_file['y_test']

    # Train model
    model.fit(lx_train, ly_train,
        validation_data=(lx_val, ly_val),
        batch_size=batch_size,  shuffle='batch',
        epochs=epochs,
        verbose=1,
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath=model_path, monitor='val_loss',
                verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
            ),
            ks.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=patience,
                verbose=0, mode='auto'
            ),
            ks.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_delta=1E-7
            ),
        ]          
    )

    model.load_weights(model_path)	# Load the model from the saved checkpoint

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
