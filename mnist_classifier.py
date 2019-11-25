from __future__ import print_function
import keras as ks
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
import h5py
import mnist_model

model_path = 'data/model_mnist_classifier.h5'
batch_size = 128
num_classes = 10
epochs = 200
patience = 40
learn_rate = 0.001

train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=5)
valid_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=5)

# Create and build the model from layers, print model info
model = mnist_model.get_classifier()
model.compile(
    optimizer=ks.optimizers.Adam(lr=learn_rate),
    loss=ks.losses.categorical_crossentropy,
    metrics=['accuracy']
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

    print(f"Number of training samples: {len(lx_train)}")
    print(f"Number of validation samples: {len(lx_val)}")

    train_datagen.fit(lx_train)
    valid_datagen.fit(lx_val)

    train_generator = train_datagen.flow(lx_train, ly_train, batch_size=batch_size)
    valid_generator = valid_datagen.flow(lx_val, ly_val, batch_size=batch_size)

    # Train model
    model.fit_generator(train_generator,
        validation_data=valid_generator,
        steps_per_epoch = 20,
        validation_steps = 5,
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
                monitor='val_loss', factor=0.5, patience=10, min_delta=1E-6
            ),
        ]          
    )

    model.load_weights(model_path)	# Load the model from the saved checkpoint

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
