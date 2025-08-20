import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# IMPORT FROM UTILS
from utils import (generate_gaussian_image_with_tilt, add_random_offset, 
                   add_salt_and_pepper_noise, add_blur)


def generate_dataset(n_samples=25000, size=64):
    images = []
    labels = []
    offsets = []
    for _ in range(n_samples):
        #Original Imanges with tilt
        img, label = generate_gaussian_image_with_tilt(size)
        images.append(img)
        labels.append(label)
        offsets.append(0)

        #Noisy/Blurry Images
        if np.random.rand() < 0.5:
            img_nb = add_salt_and_pepper_noise(img, amount=0.01)
        else:
            img_nb = add_blur(img, ksize=3)
        images.append(img_nb)
        labels.append(label)
        offsets.append(0)


        #Offset Imanges
        img_off, offset = add_random_offset(img)
        images.append(img_off)
        labels.append(label)
        offsets.append(offset)
        
        #Images with Noise/Blur AND Offset
        if np.random.rand() < 0.5:
            img_nb_off = add_salt_and_pepper_noise(img_off, amount=0.01)
        else:
            img_nb_off = add_blur(img, ksize=3)
        images.append(img_nb_off)
        labels.append(label)
        offsets.append(offset)
        
    images = np.expand_dims(np.array(images), axis=-1)
    labels = np.array(labels)
    offsets = np.array(offsets)
    return images, labels, offsets


# ------- Dataset ------- 
#Paths to save
dataset_X_path = "X_dataset_new_notilt.npy"
dataset_y_path = "y_dataset_new_notilt.npy"
offsets_path = "offsets_new_notilt.npy"

#Checkpoint flag
#Loading Dataset if exists
if os.path.exists(dataset_X_path):
    print("Loading dataset from checkpoint...")
    X = np.load(dataset_X_path)
    y = np.load(dataset_y_path)
    offsets = np.load(offsets_path)

#Generating New Dataset if doesn't exist
else:
    print("Generating dataset from scratch...")

    X, y, offsets = generate_dataset()
    
    # Save checkpoint
    np.save(dataset_X_path, X)
    np.save(dataset_y_path, y)
    np.save(offsets_path, offsets)

#Shuffling the dataset        
X, y, offsets = shuffle(X, y, offsets, random_state=42)
#Training and Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

''
if __name__ == "__main__":
    print("Running generator.py as a standalone script...")
    # ------- Sample Images from Dataset ------- 
    i = np.random.randint(0, len(X))
    for j in range(5):
        plt.imshow(X[i+j].squeeze(), cmap='viridis')
        plt.title(f"Sample Image\nOffset: {offsets[i+j]:.2f}\nTrue Label: x={y[i+j][0]:.2f}, y={y[i+j][1]:.2f}, sx={y[i+j][2]:.2f}, sy={y[i+j][3]:.2f}")
        plt.colorbar()
        plt.show()