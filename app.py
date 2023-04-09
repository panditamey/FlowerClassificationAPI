from flask import Flask, request, jsonify ,render_template , redirect
from pydantic import BaseModel
import pickle
import json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
import gdown
import lightgbm as lgb
from PIL import Image


app = Flask(__name__)

id = "1ry4L9L1-kyc79F1MnYMemJ5P81Gr_mHP"
output = "model_flowers_classification.h5"
gdown.download(id=id, output=output, quiet=False)
   

crop_disease_ml=load_model('model_flowers_classification.h5')

@app.route("/upload-image", methods=["POST"])
def upload_image():
    # if request.method == "POST":
        if request.files:
            imag = request.files["image"]
            try:
                contents = imag.file.read()
                with open(imag.filename, 'wb') as f:
                    f.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                imag.file.close()
            print(imag)
            classes = ['Lilly','Lotus','Orchid','Sunflower', 'Tulip']
            img=image.load_img(str(imag.filename),target_size=(224,224))
            x=image.img_to_array(img)
            x=x/255
            img_data=np.expand_dims(x,axis=0)
            prediction = crop_disease_ml.predict(img_data)
            predictions = list(prediction[0])
            max_num = max(predictions)
            index = predictions.index(max_num)
            print(classes[index])
            os.remove(str(imag.filename))
            return {"output":classes[index]}


if __name__ =="__main__":
    app.run()