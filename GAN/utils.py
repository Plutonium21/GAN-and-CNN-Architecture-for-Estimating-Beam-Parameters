
import numpy as np
import cv2

def generate_gaussian_image(x, y, sigma_x, sigma_y, tilt, size=64):
    X, Y = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))
    Xc, Yc = X - size // 2, Y - size // 2
    tilt_rad = np.deg2rad(tilt)
    Xr = Xc * np.cos(tilt_rad) - Yc * np.sin(tilt_rad)
    Yr = Xc * np.sin(tilt_rad) + Yc * np.cos(tilt_rad)
    g = np.exp(-0.5 * ((Xr - x)**2 / sigma_x**2 + (Yr - y)**2 / sigma_y**2))
    return g / np.max(g)

def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[coords] = 0
    return noisy

def add_gaussian_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)