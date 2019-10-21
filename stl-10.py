#%%
import re
import h5py
import json
import tarfile
import numpy as np
import keras as ks
import tensorflow as tf
import sklearn.model_selection as model_selection

# Set allow_growth to true to avoid memory hogging
ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

data_path = r''
batch_size = 32
epochs = 100
patience = 20
seed = 66
learn_rate = 0.0001

#%%
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
       .reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1)).astype('float16') / 255

# Split unlabeled data into training and validation sets
ux_train, ux_val = model_selection.train_test_split(
    data['unlabeled_X'], test_size=0.2, random_state=seed
)

# Store prepared data in HDF5 file
# Rename train_X to x_train, and etc.
with h5py.File('data/stl-10.hdf5', 'w') as out_file:
    out_file['seed'] = seed
    out_file['class_names'] = data['class_names']
    out_file['ux_train'] = ux_train
    out_file['ux_val'] = ux_val

    for name in ['train_X', 'train_y', 'test_X', 'test_y']:
        new_name = (lambda x: '_'.join([y.lower() for y in x.split('_')[::-1]]))(name)
        out_file[new_name] = data[name]

# Clear the memory
del ux_train, ux_val, data

#%%
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
# Train model
with h5py.File('data/stl-10.hdf5', 'r') as data_file:
    history = model.fit(
        data_file['ux_train'], data_file['ux_train'],
        validation_data=(data_file['ux_val'], data_file['ux_val']),
        epochs=epochs, verbose=1,
        batch_size=batch_size, shuffle='batch',
        callbacks=[
            ks.callbacks.ModelCheckpoint(
                filepath='models/model_{epoch:04d}-{val_loss:.5f}.hdf5', monitor='val_loss',
                verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5
            ),
            ks.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=patience,
                verbose=0, mode='auto'
            ),
            ks.callbacks.LearningRateScheduler(
                (lambda epoch: learn_rate / (2**(epoch // patience)))
            )
        ]
    )

    # Save history
    with open('models/history.json', 'w') as hist_file:
        hist_file.write(json.dumps(history.history, indent=4))
