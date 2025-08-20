import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
# IMPORT FROM OTHER FILES
from generator import X_test, y_test, generate_dataset
from model import model 
# Uncomment if we want to use the history from train.py for the loss plot:
from train import history

if __name__ == "__main__":
    # ------- Testing and Evaluation ------- 
    y_pred = model.predict(X_test)

    #Compute RMSE and R2 for each output
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    for i, name in enumerate(["x", "y", "σx", "σy"]):
        print(f"{name}: RMSE = {rmse[i]:.2f}, R² = {r2[i]:.2f}")
    print(f"Overall RMSE: {overall_rmse:.2f}")

    img, y1, offsets = generate_dataset(1)
    for j in range(4):
        plt.imshow(img[j].squeeze(), cmap='viridis')
        plt.title(f"Generated Image\nOffset: {offsets[j]:.2f}\nTrue Label: x={y1[j][0]:.2f}, y={y1[j][1]:.2f}, sx={y1[j][2]:.2f}, sy={y1[j][3]:.2f}")
        plt.colorbar()
        plt.show()
        img[j] = (img[j] - np.min(img[j])) / (np.max(img[j]) - np.min(img[j]))
        pred = model.predict(np.expand_dims(img[j], axis=0))
        print(pred)


    # ------- Plot Training Loss ------- 
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()




    # ------- Prediction vs Ground Truth ------- 
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    param_names = ["x", "y", "σx", "σy"]

    for i in range(4):
        axs[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=10)
        axs[i].plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'r')
        axs[i].set_title(f"{param_names[i]}: GT vs Pred")
        axs[i].set_xlabel("Ground Truth")
        axs[i].set_ylabel("Prediction")
        axs[i].grid()

    plt.tight_layout()
    plt.show()
