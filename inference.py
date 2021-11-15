# Imports
import os

import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
from PIL import Image

import tensorflow as tf

# Load the models
night_model = tf.keras.models.load_model('models/NightCNN.h5')
precipitation_model = tf.keras.models.load_model('models/PrecipitationCNN.h5')
fog_model = tf.keras.models.load_model('models/FogCNN.h5')

# Folder of images to run inference on
images_path = "inference_images"

# Save the images and their names in a list
names = []
images = []

for image_name in os.listdir(images_path):
    names.append(image_name)
    image_path = os.path.join(images_path, image_name)
    image = Image.open(image_path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.array(image, dtype=np.uint8)
    images.append(image)

# Run inference on the images
images = np.array(images, dtype=np.uint8)

night_predictions = night_model.predict(images)
precipitation_predictions = precipitation_model.predict(images)
fog_predictions = fog_model.predict(images)

# Convert prediction to integer
night_labels = night_predictions > 0.5
precipitation_labels = argmax(precipitation_predictions, axis=1)
fog_labels = fog_predictions > 0.5

# If night then precipitation is clear and fog is clear
precipitation_labels = [0 if x == 1 else y for x,
                        y in zip(night_labels, precipitation_labels)]
fog_labels = [0 if x == 1 else y for x, y in zip(night_labels, fog_labels)]

# Convert prediction to string
night_labels = ["night" if x == 1 else "day" for x in night_labels]
precipitation_labels = ["clear" if x == 0 else (
    "rain" if x == 1 else "snow") for x in precipitation_labels]
fog_labels = ["fog" if x == 1 else "no fog" for x in fog_labels]

# Save names and predictions in a dataframe
df = pd.DataFrame(columns=['name', 'night', 'precipitation', 'fog'])
df['name'] = names
df['night'] = night_labels
df['precipitation'] = precipitation_labels
df['fog'] = fog_labels

# Save dataframe as csv
df.to_csv('inference_results.csv', index=False)
