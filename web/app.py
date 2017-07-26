import os
import sys
import flask
from flask import Flask, render_template, request, jsonify, Response
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import base64
import cv2
import csv
sys.path.append('../')
from functions.settings import *
from face_detecter.detect_faces import *
from json import dumps


# get the class list
names = []
classes = os.path.join(RESULT_DIR, 'lists', 'name_67.csv')
with open(classes, 'r') as f:
    reader = csv.reader(f)
    for name in reader:
        names.append(name[0])

fn_model = "model_67.json"
fn_weighs = "weights_256_67.hdf5"

model_json = open(os.path.join(MODEL_DIR, fn_model)).read()
img_row, img_col, channels = (256, 256, 3)
target_size = (img_row, img_col, channels)

#for cutting images
tmp = "tmp.jpg"
cut = "cut_img.jpg"
rec = "rec.jpg"


app = Flask(__name__)


@app.route('/api', methods=['POST'])
def classifier():
    if 'file' not in request.files:
        reslut = "File doesn't exist."
        return result


    model = model_from_json(model_json)
    model.load_weights(os.path.join(WEIGHTS_DIR, fn_weighs))
   
    model.compile(loss="categorical_crossentropy",
                optimizer='adam',
                metrics=["accuracy"]
    )

    file = request.files['file']
    file_name = file.filename
    file.save(tmp)
    cut_img, img = detectFaces(tmp)
    
    cv2.imwrite('test.jpg', img)

    if cut_img is 0 and img is 0:
        return jsonify(results=[0])

    cv2.imwrite(rec, img)
    f2 = open(rec, 'rb')
    img2 = f2.read()

    base64_bytes = base64.b64encode(img2)
    img2 = base64_bytes.decode('utf-8')

    img1 = []
    name = []
    prod = []
    
    for i in range(0,len(cut_img)):
        
        #add image to array for sending them to browser
        cv2.imwrite(cut, cut_img[i])
        f1 = open(cut, 'rb')
        img = f1.read()
        base64_bytes = base64.b64encode(img)
        img = base64_bytes.decode('utf-8')
        img1.append(img)
        
        #calc probability
        target = image.load_img(cut, target_size=target_size)
        x = image.img_to_array(target)
        x = np.expand_dims(x, axis=0)
        x /= 255.0
        pred = model.predict(x)[0]
    
        top = 10
        top_indices = pred.argsort()[-top:][::-1]
        result = [ (names[i], pred[i]) for i in top_indices ]
        
        for n, p in result:
            name.append(str(n))
            prod.append(str(p))

        for i in result:
            print(i)
    
    os.remove(tmp)
    os.remove(cut)
    os.remove(rec)

    return jsonify(results=[img2, img1, name, prod])


@app.route('/')
def index():
    title="AV Classifier"
    return render_template('index.html', title=title)

if __name__ == "__main__" :
    app.debug = True
    app.run()
