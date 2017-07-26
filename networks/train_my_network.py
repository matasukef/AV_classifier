import os
import sys
import numpy as np
import glob

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append('../functions/')
from functions import *
from settings import *


classes = os.listdir(TRAIN_DIR)

batch_size = 64
nb_classes = len(classes)
data_augumentation = True

# input_image specifications
img_rows, img_cols, img_channels = (256, 256, 3)
input_shape = (img_rows, img_cols, img_channels)

train_data_dir = TRAIN_DIR
validation_data_dir = VALID_DIR

nb_train_samples = nb_classes * 24
nb_val_samples = nb_classes * 6

nb_epoch = 1000


if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(REPORT_DIR):
        os.mkdir(REPORT_DIR)
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)


def conv_bn_relu(x, out_ch, name):
    x = Convolution2D(out_ch, 3, 3, border_mode='same', name=name)(x)
    x = BatchNormalization(name='{0}_bn'.format(name))(x)
    x = Activation('relu', name='{0}_relu'.format(name))(x)
    return x

"""
def model(input_shape=(3, 96, 96), nb_classes, weights_path=None):
    
    inputs = Input(shape=input_shape, name='inoput')

    x = conv_bn_relu(inputs, 64, name='block1_conv1')
    x = conv_bn_relu(x, 64, name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_bn_relu(x, 128, name='block2_conv1')
    x = conv_bn_relu(x, 128, name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_bn_relu(x, 256, name='block3_conv1')
    x = conv_bn_relu(x, 256, name='block3_conv2')
    x = conv_bn_relu(x, 256, name='block3_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = conv_bn_relu(x, 512, name='block4_conv1')
    x = conv_bn_relu(x, 512, name='block4_conv2')
    x = conv_bn_relu(x, 512, name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(input=inputs, output=x)
"""

if __name__ == '__main__':

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    
    model_json_str = model.to_json()
    open(os.path.join(MODEL_DIR, 'my_model.json'), 'w').write(model_json_str)

    #model.loadweights(os.path.join(WEIGHTS_DIR, 'my_model_weights.hdf5'))

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale = 1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode='categorical',
        classes=classes,
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        class_mode='categorical',
        classes=classes,
        batch_size=batch_size,
        shuffle=True
    )

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples)

    model.save_weights(os.path.join(WEIGHTS_DIR, 'my_weights.h5'))
    save_history(history, os.path.join(RESULT_DIR, 'report', 'history_my_network.txt'))
