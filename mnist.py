#%%
import os
import gzip
import pickle
import visualization
import classification
import numpy as np
import keras as ks
import tensorflow as tf

# Set allow_growth to true to avoid memory hogging
ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

data_path = r''
checkpoint_path = r''
classifier_path = r''
figure_out_path = r''
batch_size = 32
epochs = 250
patience = 20
seed = 66
learn_rate = 0.0001
sample_size = 10

#%%
####################################################################################################
# Prepare data
####################################################################################################

#%%
# Information for retrieving data
file_mappings = {
    'train': {'x': 'train-images-idx3-ubyte.gz', 'y': 'train-labels-idx1-ubyte.gz', 'count': 60000},
    'test': {'x': 't10k-images-idx3-ubyte.gz', 'y': 't10k-labels-idx1-ubyte.gz', 'count': 10000}
}
data = {}

# Read, parse and normalize the data
for mapping in file_mappings:
    with gzip.open('%s/%s' % (data_path, file_mappings[mapping]['x']), 'r') as in_file:
        data['x_%s' % mapping] = np.array(
            [int(x) for x in in_file.read()[16:]]
        ).reshape((file_mappings[mapping]['count'], 28, 28, 1)).astype(np.float32) / 255
    with gzip.open('%s/%s' % (data_path, file_mappings[mapping]['y']), 'r') as in_file:
        data['y_%s' % mapping] = np.array(
            [int(x) for x in in_file.read()[8:]]
        ).reshape((file_mappings[mapping]['count'], 1)).reshape(-1)

    data['class_names'] = [str(x) for x in range(10)]

#%%
####################################################################################################
# Create model
####################################################################################################

# Input layer
layers = [ks.layers.InputLayer(input_shape=(28, 28, 1))]

# Convolutional layers (2 x (Conv + Max Pool))
for n_filters in 32 * 2 ** np.array(range(2)):
    layers.append(ks.layers.Conv2D(
        filters=n_filters, kernel_size=(3, 3),
        activation='relu', padding='same',
    ))
    layers.append(ks.layers.MaxPooling2D(pool_size=(2, 2)))

# Dense layers
layers += [
    ks.layers.Flatten(),
    ks.layers.Dense(25, activation='relu'),
    ks.layers.Dense(3136, activation='relu'),
    ks.layers.Reshape((7, 7, 64))
]

# Deconvolutional layers (1 x Deconv + 1 x Deconv with 1 filter for the last layer)
for n_filters in list(128 // 2 ** np.array(range(1))) + [1]:
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

history = model.fit(
    data['x_train'], data['x_train'],
    validation_split=1/6,
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
# Create embedding model
model = ks.models.load_model('%s/%s' % (checkpoint_path, os.listdir(checkpoint_path)[-1]))
clf = classification.create_model(model, 6, data['x_train'], data['y_train'], sample_size)
classification.save_model(clf, classifier_path)

####################################################################################################
# Test classifier
####################################################################################################

clf = classification.load_model(classifier_path)
classification.test_model(clf, data['x_test'], data['y_test'], data['class_names'])

####################################################################################################
# Visualize results
####################################################################################################

yp_test = clf.predict(data['x_test'])
visualization.visualize_confusion_matrix(data['x_test'], data['y_test'], yp_test, figure_out_path)
