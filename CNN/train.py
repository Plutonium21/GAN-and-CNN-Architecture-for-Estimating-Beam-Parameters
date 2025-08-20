import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# IMPORT DATA AND MODEL
from generator import X_train, y_train 
from model import model

# Defining callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=1e-4)
model_ckpt = ModelCheckpoint("bpm_cnn_notilt.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)

if __name__ == "__main__":
    print("Starting Training Session...")
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, 
                        batch_size=64, callbacks=[early_stop, model_ckpt, reduce_lr])
else:
    history = None


'''
# ------- Loading the trained model (uncomment when wanna load) -------
from tensorflow.keras.models import load_model # type: ignore
model = load_model("bpm_cnn_notilt.keras")
'''