from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np


# Keras

from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


Model= load_model('model/model.h5')     

class_labels=['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']
 


def model_predict(img_path, Model):
    class_labels=['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']
    #img = image.load_img(img_path, target_size=(32,32,3))        #for model203pfm
    img = image.load_img(img_path, target_size=(224,224,3))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = Model.predict(x)[0]
    pred=preds
    pred[pred.argmax()]=np.min(preds)
    label=class_labels[preds.argmax()]
    return label


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path , Model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
