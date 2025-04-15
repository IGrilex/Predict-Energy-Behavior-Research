import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch  # No longer used for model training here—but still used for other parts of the pipeline if needed
from tqdm import tqdm

# Import custom modules and configuration
from config import (PROSUMER_DATA_PATH, WEATHER_FORECAST_PATH, WEATHER_STATIONS_PATH,
                    CLIENT_DATA_PATH, ELECTRICITY_PRICES_PATH, GAS_PRICES_PATH, 
                    DATA_SAMPLE_SIZE, WEATHER_FEATURE_NAMES)
import data_processing as dp
import modeling
import evaluation
import visualization

def main():
    print("Starting energy prediction pipeline with XGBoost gradient boosting...")
    
    # ---------------------------
    # Hyperparameters for training
    # ---------------------------
    TRAIN_SPLIT_RATIO = 0.8      # 80% for training, 20% for validation
    XGB_NUM_ROUNDS = 100         # Number of boosting rounds for XGBoost
    
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
    # Prepare data for gradient boosting (tabular format)
    # ---------------------------
    X = dataset.drop("target", axis=1).values
    y = dataset.target.values
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42)
    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
    
    # ---------------------------
    # Model Creation and Training with XGBoost
    # ---------------------------
    print("Creating XGBoost model parameters...")
    params = modeling.create_xgb_model()  # Using default parameters tuned for time series tasks
    print("Training the XGBoost model...")
    xgb_model = modeling.train_xgb_model(params, X_train, y_train, X_val, y_val, num_rounds=XGB_NUM_ROUNDS)
    
    # ---------------------------
    # Evaluate on validation set
    # ---------------------------
    print("Evaluating the model...")
    y_val_pred = modeling.predict_xgb_model(xgb_model, X_val)
    eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
    print("Evaluation scores on validation data:", eval_scores)
    
    # ---------------------------
    # Visualization
    # ---------------------------
    print("Plotting predictions...")
    # Convert predictions and ground-truth values to Series for plotting
    visualization.plot_predictions(pd.Series(y_val), pd.Series(y_val_pred), sample_size=100)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
