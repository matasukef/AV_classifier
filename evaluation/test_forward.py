import sys
import os
import argparse
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

parser = argparse.ArgumentParser(description='test classifier network')
parser.add_argument(
        'img_path', type=str,
        help="choose specified file path"
        )

args = parser.parse_args()

if not os.path.isfile(args.img_path):
    print("File doesn't exist")
    stys.exit(1)

print('input file:', args.img_path)


classes = os.listdir(TRAIN_DIR)
nb_class = len(claases)

fn_model = 'model.json'
fn_weights = 'finetuned_weights.h5'

json_string = open(os.path.join(MODEL_DIR, fn_model)).read()
model = model_from_json(json_string)

model.load_weights(os.path.join(WEIGHTS_DIR, fn_weights))

img_rows, img_cols, img_channels = (96, 96, 3)

target_size = (img_rows, img_cols, img_channels)

model.compile(
        loss='categorical_crossentropy',
        optimizers=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
)

#model.summary()



img = image.load_img(args.img_path, target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#x /= 255.0

pre = model.predict(x)[0]

print(pre)
