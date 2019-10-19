#%%
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


#%%
data_path = r''
batch_size = 32
epochs = 100
patience = 20
seed = 66
learn_rate = 0.0001


# Generator used to retrieve data
def data_generator(split, n_batches):
    while True:
        for i in range(n_batches):
            yield (lambda x: (x, x))(np.load('data/x_%s_%s.npy' % (split, i)) / 255)


def lr_scheduler(epoch):
    return learn_rate / (2**(epoch // patience))


#%%
# Load data from tar file
# Parse it using Numpy and convert it to 96x96x3
# Split data into training and validation sets
data_file = tarfile.open(data_path, 'r:gz')
x_train = data_file.extractfile('stl10_binary/unlabeled_X.bin').read()
x_train = np.frombuffer(x_train, dtype=np.uint8).reshape((-1, 3, 96, 96))
x_train = np.transpose(x_train, (0, 3, 2, 1))
x_train, x_val = model_selection.train_test_split(x_train, test_size=0.2, random_state=seed)

#%%
# Prepare batches and save them to the hard drive
for data, split in zip([x_train, x_val], ['train', 'val']):
    n_batches = int(np.ceil(data.shape[0] / batch_size))

    with open('data/%s_meta.txt' % split, 'w') as file:
        file.write(str(n_batches))

    for i in range(int(np.ceil(data.shape[0] / batch_size))):
        np.save('data/x_%s_%s' % (split, i), data[i * batch_size:(i + 1) * batch_size])

# Free the memory
del x_train, x_val

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
# Read number of batches for training and validation sets
n_batches = {}

for split in ['train', 'val']:
    with open('data/%s_meta.txt' % split) as file:
        n_batches[split] = int(file.read())

# Train model
history = model.fit_generator(
    generator=data_generator('train', n_batches['train']),
    validation_data=data_generator('val', n_batches['val']),
    steps_per_epoch=n_batches['train'], validation_steps=n_batches['val'],
    epochs=epochs, verbose=1,
    callbacks=[
        ks.callbacks.ModelCheckpoint(
            filepath='models/model_{epoch:04d}-{val_loss:.5f}.hdf5', monitor='val_loss',
            verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5
        ),
        ks.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=patience,
            verbose=0, mode='auto'
        ),
        ks.callbacks.LearningRateScheduler(lr_scheduler)
    ]
)

# Save history
with open('models/history.json', 'w') as file:
    file.write(json.dumps(history.history, indent=4))
