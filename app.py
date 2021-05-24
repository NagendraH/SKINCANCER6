import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json 
from flask import Flask
from flask_ngrok import run_with_ngrok
from flask import render_template
from flask import request
import h5py

app = Flask(__name__)
UPLOAD_FOLDER='static/'


def predict(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32)/255.
    image=np.expand_dims(image, axis=0)
    model = tf.keras.models.load_model('/content/saved_model/model')
    predictions=model.predict_classes(image)
    #print(predictions)
    return predictions   
#run_with_ngrok(app) 
@app.route("/",methods=['GET','POST'])
def upload_predict():
    if request.method=='POST':
      image_file=request.files["image"]
      if image_file:
        image_location=UPLOAD_FOLDER+image_file.filename
        image_file.save(image_location)
        pred=predict(image_location)
        return render_template('index.html',prediction=pred[0],image_loc=image_file.filename)
    return render_template('index.html',prediction=0,image_loc=None)
if __name__=="__main__":
  app.run()
