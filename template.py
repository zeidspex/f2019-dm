import json
import keras as ks

# TODO - Parse data
# MNIST / Fashion MNIST:
#   X Shape: (-1, 28, 28, 1)
#   Y Shape: (-1, 10)
# STL-10:
#   X Shape: (-1, 96, 96, 3)
#   Y Shape: (-1, 10)
x_train, x_val, x_test = None, None, None
y_train, y_val, y_test = None, None, None

# TODO - customize autoencoder for your data
# Play with number of filters, number of layers, dense layer size, etc.
model = ks.models.Sequential([
    ks.layers.InputLayer(input_shape=(96, 96, 3)),

    ks.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
    ks.layers.MaxPooling2D(pool_size=(2, 2)),

    ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    ks.layers.MaxPooling2D(pool_size=(2, 2)),

    ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    ks.layers.MaxPooling2D(pool_size=(2, 2)),

    ks.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    ks.layers.MaxPooling2D(pool_size=(2, 2)),

    ks.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    ks.layers.MaxPooling2D(pool_size=(2, 2)),

    ks.layers.Flatten(),
    ks.layers.Dense(128),
    ks.layers.Dense(1152),
    ks.layers.Reshape((3, 3, 128)),

    ks.layers.Deconv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)),
    ks.layers.Deconv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)),
    ks.layers.Deconv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)),
    ks.layers.Deconv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)),
    ks.layers.Deconv2D(filters=3, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2))
])


# Create model
model.compile(
    optimizer=ks.optimizers.Adam(lr=0.0001),
    loss=ks.losses.mean_squared_error
)

# Print model info
model.summary()

# Train model
history = model.fit(
    x_train, x_train,
    batch_size=32, epochs=100, verbose=1,
    validation_data=(x_val, x_val),
    callbacks=[
        ks.callbacks.ModelCheckpoint(
            filepath='model_{epoch:04d}-{val_loss:.5f}.hdf5', monitor='val_loss',
            verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=5
        )
    ]
)

# Save history
with open('history.json', 'w') as file:
    file.write(json.dumps(history.history, indent=4))
