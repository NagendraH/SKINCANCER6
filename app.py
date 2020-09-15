import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import model_from_yaml
from flask import Flask
from flask import render_template
from flask import request


app = Flask(__name__)
UPLOAD_FOLDER='static/'
loaded_model=None

def predict(image_path,model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img=cv2.imread(str(image_path))
    img = cv2.resize(img, (75,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img=np.reshape(img,(1,75,100,3))
    predictions=model.predict(img)
    #print(predictions)
    return np.vstack((tf.sigmoid(predictions))).ravel()    
@app.route("/",methods=['GET','POST'])
def upload_predict():
    if request.method=='POST':
      image_file=request.files["image"]
      if image_file:
        image_location=UPLOAD_FOLDER+image_file.filename
        image_file.save(image_location)
        pred=predict(image_location,loaded_model)
        return render_template('index.html',prediction=pred[0],image_loc=image_file.filename)
    return render_template('index.html',prediction=0,image_loc=None)
if __name__=="__main__":
  loaded_model=keras.models.load_model("new_model.h5")
  app.run()