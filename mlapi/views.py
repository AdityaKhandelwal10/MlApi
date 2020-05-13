
from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings 
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback
import cv2
import json 




def predict(model,image_path):
    class_names = [('Dog', 0), ('Cat', 1)]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(img, dsize=(75, 75))
    res = res/255.0
    res = np.asarray(res)
    res = res.reshape((-1, 75, 75,1))
    y_prob = model.predict(res)
    probs = dict()
    for i in range(0, len(y_prob[0])):
        probs[i] = y_prob[0][i]
    probs = sorted(probs.items(), key=lambda x: x[1], reverse = True) 
    pred = "Prediction: " + '\n'
    for i in range(0,1):
        index = probs[i][0]
        labels = {value: key for key, value in class_names}
        name = labels[index].split('-')
        name = name[1:]
        name = '-'.join(name)
        name = name.capitalize()
        pred += str(i+1) + '.' + ' ' + str(name) + '\t' + str(int(probs[i][1]*100)) + '%' + "\n"
    return pred

def index(request):
    if  request.method == "POST":
        f = request.FILES['sentFile'] # here you get the files needed
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)

        response = predict(settings.model, file_url)
        
        return render(request,'homepage.html',response)
    else:
        return render(request,'homepage.html')
       
