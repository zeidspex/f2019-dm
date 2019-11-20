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
epochs = 200
patience = 40
learn_rate = 0.01

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

    print('x_train shape:', lx_train.shape)
    print(lx_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

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
