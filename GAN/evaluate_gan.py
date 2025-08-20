
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from gan_training import build_generator
from sklearn.metrics import mean_squared_error, mean_absolute_error

latent_dim = 100
regressor = load_model('regressor_model.h5')
generator = build_generator()
generator.load_weights('generator_weights.h5')

noise = tf.random.normal([100, latent_dim])
generated_images = generator(noise).numpy()

predictions = regressor.predict(generated_images)
true_values = np.random.uniform(-10, 10, (100, 5))

mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')
