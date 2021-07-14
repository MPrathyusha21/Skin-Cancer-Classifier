from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import pickle
import numpy as np
from PIL import Image as pil_image

# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


Model= load_model('C:/Users/acer/Desktop/skin cancer classification/model/modelmobile.h5')     

class_labels=['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']
'''lesion_classes_dict = {
    0 : 'Melanocytic nevi',
    1 : 'Melanoma',
    2 : 'Benign keratosis-like lesions ',
    3 : 'Basal cell carcinoma',
    4 : 'Actinic keratoses',
    5 : 'Vascular lesions',
    6 : 'Dermatofibroma'
}
'''


def model_predict(img_path, Model):
    class_labels=['Melanocytic nevi','Melanoma','Benign keratosis-like lesions ','Basal cell carcinoma','Actinic keratoses','Vascular lesions','Dermatofibroma']
    #img = image.load_img(img_path, target_size=(32,32,3))        #for model203pfm
    img = image.load_img(img_path, target_size=(224,224,3))
  
    #img = np.asarray(pil_image.open('img').resize((120,90)))
    #x = np.asarray(img.tolist())

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    preds = Model.predict(x)[0]
    pred=preds
    pred[pred.argmax()]=np.min(preds)
    print(np.argmax(pred))
    print(preds)
    print(preds.argmax())
    label=class_labels[preds.argmax()]
    print(label)
    #img = image.load_img(img_path, target_size=(48, 48))
 
    # Preprocessing the image
    
 
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
        print(file_path)
        # Make prediction
        preds = model_predict(file_path , Model)

        # Process your result for human
        '''pred_class=[]

        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        pr = lesion_classes_dict[pred_class[0]]
        result =str(pr)
        print(pred_class)
        return result'''
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
