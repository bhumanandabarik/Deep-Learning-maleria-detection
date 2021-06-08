from flask import Flask, render_template, url_for, request, redirect
import tensorflow as tf
#import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import h5py
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import sys
import os
import glob
import re

app = Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

MODEL_PATH = 'model_vgg19.h5'  # model path
model = load_model(MODEL_PATH)  #load your trained model
reviews = [
    {
        'title': 'What is meaning',
        'name': 'Mrs Marvin',
        'comment': 'It depends on person, how he sees it!',
        'date_posted': '14th Dec 2020'
    },
    {
        'title': '42',
        'name': 'Joe Blogs',
        'comment': 'What does it mean!',
        'date_posted': '5th Aug 2020'
    }

]


def model_predict(image_path, model):
    img = image.load_img(image_path,target_size=(224,224))

    #preprocessin the image

    x = image.img_to_array(img)
    #scaling
    x=x/255
    #print(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    #print(model.predict(img_data))
    preds = np.argmax(preds, axis=1)
    #print(preds)
    if preds==0:
        preds='The cell is maleria.'
    elif preds==1:
        preds='The cell is not maleria.'
    #elif preds==2:
        #preds='The insect is thrip.'
    #else:
        #preds='The insect is weevil.'''''

    return preds




@app.route('/')
@app.route('/home')
def helloworld():
    return render_template('home.html', reviews=reviews)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

@app.route('/addcomments')
def addcomments():
    return render_template('addcomments.html')



if __name__=='__main__':
    app.run(debug=True)



