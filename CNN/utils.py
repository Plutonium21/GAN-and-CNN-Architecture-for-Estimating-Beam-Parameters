import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from scipy.ndimage import rotate
import cv2
import os
import pickle

np.random.seed(42)
tf.random.set_seed(42)

# ------- Image Generator and Manipulation Functions -------
def generate_gaussian_image_with_tilt(size=64):
    x0 = np.random.uniform(20, 44)
    y0 = np.random.uniform(20, 44)
    sigma_x = np.random.uniform(2, 8)
    sigma_y = np.random.uniform(2, 8)
    theta = np.random.uniform(-45, 45)  # in degrees

    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    X, Y = np.meshgrid(x, y)

    a = np.cos(np.deg2rad(theta))**2 / (2*sigma_x**2) + np.sin(np.deg2rad(theta))**2 / (2*sigma_y**2)
    b = -np.sin(2*np.deg2rad(theta)) / (4*sigma_x**2) + np.sin(2*np.deg2rad(theta)) / (4*sigma_y**2)
    c = np.sin(np.deg2rad(theta))**2 / (2*sigma_x**2) + np.cos(np.deg2rad(theta))**2 / (2*sigma_y**2)

    image = np.exp(-(a*(X - x0)**2 + 2*b*(X - x0)*(Y - y0) + c*(Y - y0)**2))
    label = np.array([x0, y0, sigma_x, sigma_y])
    return image, label

def add_random_offset(image):
    # 50% chance for low, 30% for medium, 20% for high offset (adding random offsets in brackets for robustness and better realistic scenarios)
    p = np.random.rand()
    if p < 0.5:
        offset = np.random.uniform(0, 40)     # Low offset added
        offset_class = 'low'
    elif p < 0.8:
        offset = np.random.uniform(40, 80)    # Medium offset added
        offset_class = 'medium'
    else:
        offset = np.random.uniform(80, 120)   # High offset added
        offset_class = 'hign'

    image_with_offset = image + offset

    # Normalize after offset
    image_with_offset -= np.min(image_with_offset)
    image_with_offset /= np.max(image_with_offset)

    return image_with_offset, offset

def add_salt_and_pepper_noise(image, amount=0.01):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 1

    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)