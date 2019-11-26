import re
import h5py
import keras as ks
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras import backend as K

seed = 66
num_classes = 10

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

data = {}

# Split labeled data into unlabeled data and labedled data sets
ux_train, lx_train, _, ly_train = train_test_split(x_train, y_train, 
    test_size=50, stratify=y_train)

# Split unlabeled data into training and validation sets
ux_train, ux_val = train_test_split(
    ux_train, test_size=0.2, random_state=seed
)

# Split labeled data into training and validation sets
lx_train, lx_val, ly_train, ly_val = train_test_split(
    lx_train, ly_train, test_size=0.2, random_state=seed, stratify=ly_train
)

# Store prepared data in HDF5 file
with h5py.File('data/mnist.hdf5', 'w') as out_file:
    out_file['seed'] = seed
    out_file['num_classes'] = num_classes
    out_file['ux_train'] = ux_train
    out_file['ux_val'] = ux_val
    out_file['x_train'] = lx_train
    out_file['y_train'] = ly_train
    out_file['x_val'] = lx_val
    out_file['y_val'] = ly_val
    out_file['x_test'] = x_test
    out_file['y_test'] = y_test
