import os
import keras.models import model_from_json
import keras.preprocessing.image import ImageDataGenerator
from settings import *
import functions.functions

fn_model = os.path.join(RESULT_DIR, 'model.json')
fn_weights = os.path.join(RESUTL_DIR, 'finetuned_weights.hdf5')

# read model
json_string = open(fn_model).read()
model = model_from_json(json_string)

# read weights
model.load_from_json(fn_weights)

#model.summary()

model.compile(
        loss='categorical_crossentropy',
        optimizers=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
)
test_datagen = ImageDataGenerator(
        preprocessing_function = preprocess
)

test_datagetn = test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(96, 96),
        batch_size = 32,
        class_mode='categorical_crossentropy'
        )

print(test_generator.class_indices)

scores = model.evaluate_generator(test_generator, 500)

print('TEST LOSS:', scores[0])
print('TEST ACCURACY', scores[1])
