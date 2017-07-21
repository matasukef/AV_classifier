import os
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from settings import *
import numpy as np
import glob

# get the name of each classes
class_list = os.listdir(os.path.join(OUTPUT_DIR))

batch_size = 128
nb_claas = len(class_list)
data_augmentation = True

# input image dimensions
img_rows, img_cols, channels = (96, 96, 3)
input_shape = (img_rows, img_cols, channels)

train_data_dir = 'train_images'
validation_data_dir = 'test_images'

result_dir = os.path.join(HOME_DIR, 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def calc_mean(img_dir):


def preprocess(x):
    x = x[:, :, ::-1]
    X[:, :, 0] -= 103.939
    X[:, :, 1] -= 116.779
    X[:. :, 2] -= 123.68
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
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_class, activation='softmax'))

    #top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

    model = Model(input=vgg19.input, output=top_model(vgg19.output))

    for layer in model.layers[:18]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            rescale=1.0/255,
            sheer_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess,
            rescale=1.0/255)

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

