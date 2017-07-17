import os
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
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

result_dir = os.path.join(HOME_DIR, 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


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
