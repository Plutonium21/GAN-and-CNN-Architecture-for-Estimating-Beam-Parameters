
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Set image shape to 64x64
image_shape = (64, 64, 1)
latent_dim = 100

# Updated Generator for 64x64
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 128, input_dim=latent_dim),
        layers.Reshape((8, 8, 128)),
        layers.UpSampling2D(),  # 16x16
        layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),

        layers.UpSampling2D(),  # 32x32
        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),

        layers.UpSampling2D(),  # 64x64
        layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")
    ])
    return model

# Discriminator stays the same, input shape updated to 64x64
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=image_shape),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Real image loader now resizes to 64x64
def load_real_images(path, batch_size):
    images = []
    for fname in sorted(os.listdir(path))[:10000]:
        img = Image.open(os.path.join(path, fname)).resize((64, 64))
        img = np.array(img).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # grayscale
        images.append(img)
    images = np.expand_dims(np.array(images), -1) if np.array(images).ndim == 3 else np.array(images)
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(10000).batch(batch_size)
    return dataset

# Instantiate models and loss
generator = build_generator()
discriminator = build_discriminator()
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def train(dataset, epochs=100, batch_size=32):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    for epoch in range(epochs):
        for real_images in dataset:
            batch_size = tf.shape(real_images)[0]
            noise = tf.random.normal([batch_size, latent_dim])
            generated_images = generator(noise)

            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as disc_tape:
                real_output = discriminator(real_images)
                fake_output = discriminator(generated_images)
                disc_loss = cross_entropy(real_labels, real_output) + cross_entropy(fake_labels, fake_output)
            grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

            noise = tf.random.normal([batch_size, latent_dim])
            with tf.GradientTape() as gen_tape:
                fake_images = generator(noise)
                predictions = discriminator(fake_images)
                gen_loss = cross_entropy(real_labels, predictions)
            grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs} | D Loss: {disc_loss.numpy():.4f} | G Loss: {gen_loss.numpy():.4f}")
        noise = tf.random.normal([1, latent_dim])
        sample = generator(noise)[0].numpy().squeeze() * 255
        Image.fromarray(sample.astype(np.uint8)).save(f"generated_samples/sample_epoch{epoch+1:03d}.png")

    generator.save_weights("generator_weights.h5")

dataset = load_real_images("bpm_gan_dataset/images", batch_size=32)
os.makedirs("generated_samples", exist_ok=True)
train(dataset, epochs=10)
