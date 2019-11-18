#%%
import os
import re
import h5py
import pickle
import random
import tarfile
import itertools
import testing
import visualization
import classification
import numpy as np
import keras as ks
import tensorflow as tf


# Set allow_growth to true to avoid memory hogging
ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

data_path = ''
preprocessed_data_path = ''
checkpoint_path = 'models/stl-10'
classifier_path = ''
figure_out_path = ''
batch_size = 32
epochs = 250
patience = 20
seed = 66
learn_rate = 0.0001
sample_size = 30

#%%
####################################################################################################
# Prepare data
####################################################################################################

# Read data from tar file and store it in a dictionary
with tarfile.open(data_path, 'r:gz') as in_file:
    data = {
        re.match('.*/(.*)\\..*', name).groups()[0]: np.frombuffer(
            in_file.extractfile(name).read(), dtype=np.uint8
        )
        for name in in_file.getnames()[1:]
    }

# Reshape features to 96x96x3 and normalize them
for name in ['unlabeled_X', 'train_X', 'test_X']:
    data[name] = data[name]\
       .reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1)).astype('float32') / 255

# Store prepared data in HDF5 file
# Rename train_X to x_train, and etc.
with h5py.File(preprocessed_data_path, 'w') as out_file:
    out_file['seed'] = seed
    out_file['class_names'] = data['class_names']
    out_file['ux_train'] = data['unlabeled_X']

    for name in ['train_X', 'train_y', 'test_X', 'test_y']:
        new_name = (lambda x: '_'.join([y.lower() for y in x.split('_')[::-1]]))(name)

        if new_name in ['y_train', 'y_test']:
            out_file[new_name] = data[name] - 1
        else:
            out_file[new_name] = data[name]

# Clear the memory
del data

#%%
####################################################################################################
# Create model
####################################################################################################

# Input layer
layers = [ks.layers.InputLayer(input_shape=(96, 96, 3))]

# Convolutional layers (5 x (Conv + Max Pool))
for n_filters in 32 * 2 ** np.array(range(5)):
    layers.append(ks.layers.Conv2D(
        filters=n_filters, kernel_size=(3, 3),
        activation='relu', padding='same',
    ))
    layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

# Dense layers
layers += [
    ks.layers.Flatten(),
    ks.layers.Dense(1024),
    ks.layers.Dense(4608),
    ks.layers.Reshape((3, 3, 512))
]

# Deconvolutional layers (4 x Deconv + 1 x Deconv with 3 filters for the last layer)
for n_filters in list(512 // 2 ** np.array(range(4))) + [3]:
    layers.append(ks.layers.Deconv2D(
        filters=n_filters, kernel_size=(3, 3),
        activation='relu', padding='same', strides=(2, 2)
    ))

# Create and build the model from layers, print model info
model = ks.models.Sequential(layers)
model.compile(
    optimizer=ks.optimizers.Adam(lr=learn_rate),
    loss=ks.losses.mean_squared_error
)
model.summary()

#%%
####################################################################################################
# Train model
####################################################################################################

# Train model
with h5py.File(preprocessed_data_path, 'r') as data_file:
    def data_generator(data, n_batches, batch_size):
        while True:
            for i in random.sample(range(n_batches), n_batches):
                yield tuple(itertools.repeat(data[i * batch_size:(i + 1) * batch_size], 2))

    train_batches = int(np.ceil(len(data_file['ux_train']) / batch_size))
    val_batches = int(len(data_file['x_train']) / batch_size)
    history = model.fit_generator(
        generator=data_generator(data_file['ux_train'], train_batches, batch_size),
        validation_data=data_generator(data_file['x_train'], val_batches, batch_size),
        steps_per_epoch=train_batches, validation_steps=val_batches,
        epochs=epochs, verbose=1,
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath='%s/model_{epoch:04d}-{val_loss:.5f}.hdf5' % checkpoint_path,
                monitor='val_loss', verbose=1, save_best_only=False,
                save_weights_only=False, mode='auto', period=5
            ),
            ks.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=patience,
                verbose=1, mode='auto'
            ),
            ks.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=patience, cooldown=1, verbose=1, factor=0.5
            )
        ]
    )

    # Save history
    with open('%s/history.pkl' % checkpoint_path, 'wb') as hist_file:
        hist_file.write(pickle.dumps(history.history))

####################################################################################################
# Create classifier
####################################################################################################

#%%
with h5py.File(preprocessed_data_path, 'r') as in_file:
    model = ks.models.load_model('%s/%s' % (checkpoint_path, os.listdir(checkpoint_path)[-1]))
    clf = classification.create_model(
        model, 12, np.array(in_file['x_train']), np.array(in_file['y_train']), sample_size
    )
    classification.save_model(clf, classifier_path)

####################################################################################################
# Test classifier
####################################################################################################

with h5py.File(preprocessed_data_path, 'r') as in_file:
    clf = classification.load_model(classifier_path)
    class_names = bytes(in_file['class_names']).decode('UTF-8').splitlines()
    testing.test_classifier(
        clf, np.array(in_file['x_test']), np.array(in_file['y_test']), class_names
    )

####################################################################################################
# Visualize results
####################################################################################################

with h5py.File(preprocessed_data_path, 'r') as in_file:
    yp_test = clf.predict(np.array(in_file['x_test']))
    visualization.visualize_confusion_matrix(
        np.array(in_file['x_test']), np.array(in_file['y_test']), yp_test, figure_out_path
    )
