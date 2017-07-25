import os
import sys
sys.path.append('../setup')
sys.path.append('../functions')
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import glob
from settings import *
from functions import *

# get the name of each classes
classes = os.listdir(TRAIN_DIR)

ratio = 0.7

#batch_size = 256
batch_size = 64
nb_class = len(classes)
data_augmentation = True

# input image dimensions
#img_rows, img_cols, channels = (96, 96, 3)
img_rows, img_cols, channels = (256, 256, 3)
input_shape = (img_rows, img_cols, channels)

train_data_dir = TRAIN_DIR
validation_data_dir = VALID_DIR

nb_train_samples = nb_class * 35
nb_val_samples = nb_class * 15
nb_epoch = 500

result_dir = os.path.join(HOME_DIR, 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def preprocess(x):
    x = x[:, :, ::-1]
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, 'w') as f:
        f.write('epoch\tloss\tacc\tval_loss\tval_acc\n\n')
        for i in range(nb_epoch):
            f.write('{0}\t{1}\t{2}\t{3}\t{4}'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))


if __name__ == "__main__":

    input_tensor = Input(shape=(input_shape))
    vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
    #vgg19.summary()

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg19.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_class, activation='softmax'))


    model = Model(input=vgg19.input, output=top_model(vgg19.output))

    for layer in model.layers[:18]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy']
     )

    # save model
    model_json_str = model.to_json()
    open(os.path.join(MODEL_DIR, 'model_67.json'), 'w').write(model_json_str)
    
    #model.load_weights(os.path.join(RESULT_DIR, 'mid', 'mid_weights_256_122.hdf5'))

    train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            rescale=1.0/255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            rescale=1.0/255,
    )

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size = (img_rows, img_cols),
            color_mode = 'rgb',
            classes = classes,
            class_mode = 'categorical',
            batch_size = batch_size,
            shuffle = True
    )

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size = (img_rows, img_cols),
            color_mode = 'rgb',
            classes = classes,
            class_mode = 'categorical',
            batch_size = batch_size,
            shuffle = True
    )

    # save weight on the way
    checkpointer = ModelCheckpoint(os.path.join(result_dir, 'mid', 'mid_weights_256_67.h5'), verbose=5, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit_generator(
        train_generator,
        samples_per_epoch = nb_train_samples,
        nb_epoch = nb_epoch,
        validation_data = validation_generator,
        nb_val_samples = nb_val_samples,
        callbacks = [checkpointer, early_stopping]
        #callbacks = [checkpointer]
    )

    model.save_weights(os.path.join(result_dir, 'weights', 'weights_256_67.hdf5'))
    save_history(history, os.path.join(result_dir, 'report', 'history_finetuning_67.txt'))

