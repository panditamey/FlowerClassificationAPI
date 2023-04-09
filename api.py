from fastapi import FastAPI,UploadFile,File
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

CHUNK_SIZE = 1024

app = FastAPI(
    title='Flower Classification API',
    description='API for Flower Classification',
)

id = "1ry4L9L1-kyc79F1MnYMemJ5P81Gr_mHP"
output = "model_flowers_classification.h5"
gdown.download(id=id, output=output, quiet=False)
# from zipfile import ZipFile
# with ZipFile("modelcrops.zip", 'r') as zObject:
#     zObject.extractall(
#         path="")
    

crop_disease_ml=load_model('model_flowers_classification.h5')


@app.post('/cropdisease')
async def cropdisease(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    classes = ['Lilly','Lotus','Orchid','Sunflower', 'Tulip']
    img=image.load_img(str(file.filename),target_size=(224,224))
    x=image.img_to_array(img)
    x=x/255
    img_data=np.expand_dims(x,axis=0)
    prediction = crop_disease_ml.predict(img_data)
    predictions = list(prediction[0])
    max_num = max(predictions)
    index = predictions.index(max_num)
    print(classes[index])
    os.remove(str(file.filename))
    return {"output":classes[index]}