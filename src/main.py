import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.model_selection import train_test_split
# --- Add joblib for caching ---
from joblib import Memory
# --- End Add ---
# import torch # No longer used for model training here
from tqdm import tqdm

# Import custom modules and configuration
from config import (PROSUMER_DATA_PATH, WEATHER_FORECAST_PATH, WEATHER_STATIONS_PATH,
                    CLIENT_DATA_PATH, ELECTRICITY_PRICES_PATH, GAS_PRICES_PATH,
                    DATA_SAMPLE_SIZE, WEATHER_FEATURE_NAMES)
# Ensure data_processing module (dp) has the version with the process_weather_func argument in make_dataset
import data_processing as dp
import modeling
import evaluation
import visualization

# --- Caching Setup ---
CACHE_DIR = "./.cache" # Define a directory for caching
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
memory = Memory(CACHE_DIR, verbose=0) # Initialize joblib Memory cache

# --- Cache the weather processing function ---
# This uses the dp.process_weather_data function from the currently loaded data_processing.py
cached_process_weather_data = memory.cache(dp.process_weather_data)
# --- End Caching Setup ---


def main():
    print("Starting energy prediction pipeline with XGBoost gradient boosting...")

    # ---------------------------
    # Hyperparameters for training
    # ---------------------------
    TRAIN_SPLIT_RATIO = 0.8      # 80% for training, 20% for validation
    XGB_NUM_ROUNDS = 100         # Number of boosting rounds for XGBoost
    RANDOM_STATE = 42            # For reproducible train/test split

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

    # --- Scaler initialization moved after data split ---
    # scaler = StandardScaler()

    print("Making the dataset...")
    # Call make_dataset without the scaler argument
    # --- Use the CACHED weather processing function ---
    print("Processing data (weather processing may be loaded from cache)...")
    dataset = dp.make_dataset(prosumer.head(DATA_SAMPLE_SIZE),
                              weather_forecast.iloc[:DATA_SAMPLE_SIZE, :], # Use iloc for weather sample
                              weather_stations,
                              client,
                              electricity_prices,
                              gas_prices,
                              WEATHER_FEATURE_NAMES,
                              # Pass the cached function to make_dataset
                              process_weather_func=cached_process_weather_data
                              )
    # --- End Use Cached ---
    print("Dataset shape:", dataset.shape)

    # Drop rows with NaN values that might have been introduced during merging
    # Check if dataset is not None before dropna
    if dataset is not None and not dataset.empty:
        initial_rows = len(dataset)
        dataset.dropna(inplace=True) # Original script had this dropna after make_dataset
        rows_after_drop = len(dataset)
        print(f"Dataset shape after dropping NaN: {dataset.shape}")
        if initial_rows != rows_after_drop:
            print(f"Note: Dropped {initial_rows - rows_after_drop} rows containing NaNs before train/test split.")
    elif dataset is None:
         print("Error: Dataset is None after make_dataset. Cannot proceed.")
         return
    else: # Dataset is empty
         print("Warning: Dataset is empty after make_dataset. Cannot proceed.")
         return


    # ---------------------------
    # Prepare data for gradient boosting (tabular format)
    # ---------------------------
    # Handle empty dataset case (redundant check, but safe)
    if dataset.empty:
        print("Error: Dataset is empty after processing and NaN drop. Cannot proceed.")
        return # Exit script

    # Separate features (X) and target (y)
    X = dataset.drop("target", axis=1)
    y = dataset.target

    # Get feature names before scaling (needed for DMatrix if using feature names)
    feature_names = X.columns.tolist()

    # Convert to numpy arrays for scikit-learn compatibility
    X = X.values # Original code converted to numpy here
    y = y.values # Original code converted to numpy here

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=RANDOM_STATE)
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # ---------------------------
    # Feature Scaling (Corrected Workflow)
    # ---------------------------
    print("Applying feature scaling...")
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data
    # Since X_train is now numpy, no need to select numeric cols explicitly if all are numeric
    scaler.fit(X_train)

    # Transform both training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("Scaling complete.")


    # ---------------------------
    # Model Creation and Training with XGBoost
    # ---------------------------
    print("Creating XGBoost model parameters...")
    # Pass feature names if your modeling script uses them for DMatrix
    params = modeling.create_xgb_model() # Using default parameters

    print("Training the XGBoost model...")
    # Train the model using the SCALED data
    xgb_model = modeling.train_xgb_model(params,
                                          X_train_scaled, y_train, # Use scaled training data
                                          X_val_scaled, y_val,     # Use scaled validation data
                                          num_rounds=XGB_NUM_ROUNDS,
                                          # Optional: Pass feature_names if train_xgb_model uses them
                                          # feature_names=feature_names
                                          )

    # ---------------------------
    # Evaluate on validation set
    # ---------------------------
    print("Evaluating the model...")
    # Predict using the SCALED validation data
    y_val_pred = modeling.predict_xgb_model(xgb_model, X_val_scaled) # Use scaled validation data
    eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
    print("Evaluation scores on validation data:", eval_scores)

    # ---------------------------
    # Visualization
    # ---------------------------
    print("Plotting predictions...")
    # Convert predictions and ground-truth values to Series for plotting
    visualization.plot_predictions(pd.Series(y_val), pd.Series(y_val_pred), sample_size=100) # Pass numpy arrays directly if visualization handles it

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
