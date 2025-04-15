import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import custom modules and configuration
from config import (PROSUMER_DATA_PATH, WEATHER_FORECAST_PATH, WEATHER_STATIONS_PATH,
                    CLIENT_DATA_PATH, ELECTRICITY_PRICES_PATH, GAS_PRICES_PATH, 
                    DATA_SAMPLE_SIZE, WEATHER_FEATURE_NAMES)
import data_processing as dp
import modeling
import evaluation
import visualization

def create_sequences(X, y, seq_len):
    """
    Create sequences using a sliding window.
    
    For each sequence of length `seq_len` from X, the corresponding target is the value at index (i + seq_len)
    in y.
    """
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i: i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

def main():
    print("Starting energy prediction pipeline with GRU model...")
    
    # ---------------------------
    # Hyperparameters
    # ---------------------------
    TRAIN_SPLIT_RATIO = 0.8      # 80% of sequences for training, remaining for validation
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    SEQ_LEN = 50               # Sequence length for RNN input
    
    # ---------------------------
    # Data Loading
    # ---------------------------
    print("Loading data...")
    prosumer = pd.read_csv(PROSUMER_DATA_PATH)
    weather_forecast = pd.read_csv(WEATHER_FORECAST_PATH)
    weather_stations = pd.read_csv(WEATHER_STATIONS_PATH)
    client = pd.read_csv(CLIENT_DATA_PATH)
    electricity_prices = pd.read_csv(ELECTRICITY_PRICES_PATH)
    gas_prices = pd.read_csv(GAS_PRICES_PATH)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    print("Making the dataset...")
    dataset = dp.make_dataset(prosumer.head(DATA_SAMPLE_SIZE), 
                              weather_forecast.iloc[:DATA_SAMPLE_SIZE, :], 
                              weather_stations, 
                              client, 
                              electricity_prices, 
                              gas_prices,
                              WEATHER_FEATURE_NAMES,
                              scaler)
    print("Dataset shape:", dataset.shape)
    
    # ---------------------------
    # Prepare data for GRU (convert to sequences)
    # ---------------------------
    X = dataset.drop("target", axis=1).values
    y = dataset.target.values
    
    # Create sequences with the specified sequence length
    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
    
    # Split into training and validation sets based on sequences
    split_index = int(len(X_seq) * TRAIN_SPLIT_RATIO)
    X_train, X_val = X_seq[:split_index], X_seq[split_index:]
    y_train, y_val = y_seq[:split_index], y_seq[split_index:]
    print(f"Training sequences: {X_train.shape}, Validation sequences: {X_val.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create DataLoaders (each sample shape: (seq_len, feature_dim))
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---------------------------
    # Model Creation and CUDA check
    # ---------------------------
    input_dim = X_train.shape[2]  # number of features per timestep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA device is available. Training on GPU.")
    else:
        print("CUDA device is not available. Training on CPU.")
    
    # Create GRU model
    model = modeling.create_gru_model(input_dim)
    model.to(device)
    
    # ---------------------------
    # Train the model
    # ---------------------------
    print("Starting training...")
    train_losses, val_losses = modeling.train_pytorch_model(model, 
                                                            train_loader, 
                                                            val_loader, 
                                                            epochs=EPOCHS, 
                                                            lr=LEARNING_RATE, 
                                                            device=device)
    
    # ---------------------------
    # Evaluate on validation set
    # ---------------------------
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor.to(device)).squeeze().cpu().numpy()
    eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
    print("Evaluation scores on validation data:", eval_scores)
    
    # ---------------------------
    # Visualization
    # ---------------------------
    print("Plotting predictions and loss curves...")
    visualization.plot_predictions(y_val, y_val_pred, sample_size=100)
    visualization.plot_loss(train_losses, val_losses)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
