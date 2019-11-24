import re
import h5py
import json
import tarfile
import numpy as np
import keras as ks
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K
from keras.models import load_model
import mnist_model

# Set allow_growth to true to avoid memory hogging
ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

print("TensorFlow version: ")
print(tf.__version__)

print("Keras version: ")
print(ks.__version__)

model_path = 'data/model_mnist.h5'
batch_size = 128
num_classes = 10
epochs = 100
patience = 20
learn_rate = 0.001

# input image dimensions
img_rows, img_cols, img_channels = 28, 28, 1

model = mnist_model.get_encoder()
model.compile(
    optimizer=ks.optimizers.Adam(lr=learn_rate),
    loss=ks.losses.mean_squared_error
)
model.summary()


# Restore prepared data in HDF5 file
with h5py.File('data/mnist.hdf5', 'r') as data_file:
    seed = data_file['seed']
    num_classes = data_file['num_classes']
    ux_train = np.array(data_file['ux_train'])
    ux_val = np.array(data_file['ux_val'])
    lx_train = np.array(data_file['x_train'])
    ly_train = np.array(data_file['y_train'])
    lx_val = np.array(data_file['x_val'])
    ly_val = np.array(data_file['y_val'])
    x_test = np.array(data_file['x_test'])
    y_test = np.array(data_file['y_test'])

    # Train model
    model.fit(ux_train, ux_train,
        validation_data=(ux_val, ux_val),
        batch_size=batch_size,  shuffle='batch',
        epochs=epochs,
        verbose=1,
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath=model_path, monitor='val_loss',
                verbose=0, save_best_only=True, save_weights_only=False, mode='auto',
            ),
            ks.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=1E-6, patience=patience,
                verbose=0, mode='auto'
            ),
            ks.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_delta=1E-6
            ),
        ]          
    )
    model.load_weights(model_path)
    score = model.evaluate(x_test, x_test, verbose=0)
    print('Test loss:')
    print(score)
