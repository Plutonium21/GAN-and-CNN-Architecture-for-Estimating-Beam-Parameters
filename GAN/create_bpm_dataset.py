
import os
import csv
import numpy as np
import cv2
from utils import generate_gaussian_image, add_salt_pepper_noise, add_gaussian_noise

np.random.seed(42)
os.makedirs('bpm_gan_dataset/images', exist_ok=True)

num_samples = 10000
labels = []

for i in range(num_samples):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)
    sigma_x = np.random.uniform(2, 5)
    sigma_y = np.random.uniform(2, 5)
    tilt = np.random.uniform(-45, 45)
    
    image = generate_gaussian_image(x, y, sigma_x, sigma_y, tilt)
    
    noise_type = np.random.choice(['salt_pepper', 'gaussian'])
    if noise_type == 'salt_pepper':
        image = add_salt_pepper_noise(image)
    else:
        image = add_gaussian_noise(image)
    
    filename = f'image_{i:05d}.png'
    path = os.path.join('bpm_gan_dataset/images', filename)
    cv2.imwrite(path, (image * 255).astype(np.uint8))
    labels.append([filename, x, y, sigma_x, sigma_y, tilt, noise_type])

with open('bpm_gan_dataset/labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'x', 'y', 'sigma_x', 'sigma_y', 'tilt', 'noise'])
    writer.writerows(labels)
