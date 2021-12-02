# Imports
import os

import pickle
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd

from PIL import Image
import cv2

import tensorflow as tf


def convert_to_feature(image):
    """
    Extract average value of HSV and RGB layers
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = np.average(h)
    s = np.average(s)
    v = np.average(v)

    rgb = image
    r, g, b = cv2.split(rgb)
    r = np.average(r)
    g = np.average(g)
    b = np.average(b)

    return [h, s, v, r, g, b]


# Models paths
PATH_TO_NIGHT_MODEL = 'models/NightForest.sav'
PATH_TO_PRECIPITATION_MODEL = 'models/PrecipitationCNN.h5'
PATH_TO_FOG_MODEL = 'models/FogCNN.h5'

# Load the models
night_model = pickle.load(open(PATH_TO_NIGHT_MODEL, 'rb'))
precipitation_model = tf.keras.models.load_model(PATH_TO_PRECIPITATION_MODEL)
fog_model = tf.keras.models.load_model(PATH_TO_FOG_MODEL)

# Folder of images to run inference on
images_path = "inference_images"

# Save the images and their names in a list
names = []
images = []

# Preprocessing
for image_name in os.listdir(images_path):
    names.append(image_name)
    image_path = os.path.join(images_path, image_name)
    image = Image.open(image_path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image, dtype=np.uint8)
    images.append(image)

images = np.array(images, dtype=np.uint8)

# Generate features
features = np.zeros((images.shape[0], 6))
for i in range(images.shape[0]):
    features[i] = convert_to_feature(images[i])

# Prediction
night_predictions = night_model.predict(features)
precipitation_predictions = precipitation_model.predict(images)
fog_predictions = fog_model.predict(images)

# Convert prediction to integer
night_labels = night_predictions
precipitation_labels = argmax(precipitation_predictions, axis=1)
fog_labels = fog_predictions > 0.5

# Result
weather_results = []
fog_results = []

# Convert predictions to single label
for i in range(images.shape[0]):
    if night_labels[i] == 1:
        weather_results.append("night")
    else:
        if precipitation_labels[i] == 0:
            weather_results.append("clear")
        elif precipitation_labels[i] == 1:
            weather_results.append("rain")
        else:
            weather_results.append("snow")
    if fog_labels[i] == 1:
        fog_results.append("fog")
    else:
        fog_results.append("no_fog")

# Save names and predictions in a dataframe
df = pd.DataFrame(columns=['name', 'weather', 'fog'])
df['name'] = names
df['weather'] = weather_results
df['fog'] = fog_results

# Save dataframe as csv
df.to_csv('inference_results.csv', index=False)
