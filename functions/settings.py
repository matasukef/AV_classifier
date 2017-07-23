import os

# images
IMAGE_DIR = os.path.join('..', 'Images')

INPUT_DIR = os.path.join('..', 'Images', 'add')
OUTPUT_DIR = os.path.join('..', 'Images', 'faces')
HOME_DIR = ".."

# creating datasets
BASE_DIR = os.path.join('..', 'Images', 'data')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')

# results
RESULT_DIR = os.path.join('..', 'results')
MODEL_DIR = os.path.join( RESULT_DIR, 'models')
WEIGHTS_DIR = os.path.join(RESULT_DIR, 'weights')
