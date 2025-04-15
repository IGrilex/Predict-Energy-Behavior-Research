import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(y_true, y_pred, sample_size: int = 100):
    """
    Plot the ground truth and predictions.
    """
    # Get sample values
    y_true_sample = y_true[:sample_size]
    y_pred_sample = y_pred[:sample_size]
    
    plt.figure(figsize=(12,6))
    plt.plot(y_true_sample, label="Ground truth", alpha=0.8)
    plt.plot(y_pred_sample, label="Prediction", alpha=0.6)
    plt.xlabel("Timestep")
    plt.ylabel("Energy amount")
    plt.title("Predictions vs Ground Truth")
    plt.legend()
    plt.show()

def plot_loss(train_losses, val_losses):
    """
    Plot the training and validation losses over epochs.
    """
    plt.figure(figsize=(10,5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.title("Training and Validation Loss Per Epoch")
    plt.legend()
    plt.show()
