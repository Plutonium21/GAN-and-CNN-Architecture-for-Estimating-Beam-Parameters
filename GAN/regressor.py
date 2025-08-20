
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

df = pd.read_csv('bpm_gan_dataset/labels.csv')
X, y = [], []
for _, row in df.iterrows():
    img = cv2.imread(os.path.join('bpm_gan_dataset/images', row['filename']), cv2.IMREAD_GRAYSCALE)
    X.append(img / 255.0)
    y.append([row['x'], row['y'], row['sigma_x'], row['sigma_y'], row['tilt']])

X = np.expand_dims(np.array(X), -1)
y = np.array(y)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
model.save('regressor_model.h5')
