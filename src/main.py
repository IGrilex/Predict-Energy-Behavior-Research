import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Memory
from tqdm import tqdm
import xgboost as xgb # Import xgb for DMatrix

# Import custom modules and configuration
from config import (PROSUMER_DATA_PATH, WEATHER_FORECAST_PATH, WEATHER_STATIONS_PATH,
                    CLIENT_DATA_PATH, ELECTRICITY_PRICES_PATH, GAS_PRICES_PATH,
                    DATA_SAMPLE_SIZE, WEATHER_FEATURE_NAMES)
# Ensure data_processing module (dp) has the version with time features and process_weather_func argument
import data_processing as dp
import modeling
import evaluation # Keep for evaluate_model function
import visualization

# --- Caching Setup ---
CACHE_DIR = "./.cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
memory = Memory(CACHE_DIR, verbose=0)
cached_process_weather_data = memory.cache(dp.process_weather_data)
# --- End Caching Setup ---

# --- Multiple Runs Configuration ---
N_RUNS = 10 # Number of times to run with different seeds
BASE_SEED = 63641 # Starting seed (or choose another base)
SEEDS = [BASE_SEED + i for i in range(N_RUNS)] # Create a list of seeds
# --- End Configuration ---


def main():
    print("Starting energy prediction pipeline with XGBoost gradient boosting...")
    print(f"Running {N_RUNS} evaluations with different random seeds.")

    # ---------------------------
    # Hyperparameters (used in each run)
    # ---------------------------
    TRAIN_SPLIT_RATIO = 0.8      # 80% for training, 20% for validation
    XGB_NUM_ROUNDS = 1000        # Max rounds for training (early stopping applies)
    # RANDOM_STATE is now handled by the loop seeds

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

    # ---------------------------
    # Data Processing
    # ---------------------------
    print("Making the dataset...")
    print("Processing data (weather processing may be loaded from cache)...")
    dataset = dp.make_dataset(prosumer.head(DATA_SAMPLE_SIZE),
                              weather_forecast.iloc[:DATA_SAMPLE_SIZE, :], # Use iloc for weather sample
                              weather_stations,
                              client,
                              electricity_prices,
                              gas_prices,
                              WEATHER_FEATURE_NAMES,
                              process_weather_func=cached_process_weather_data
                              )
    print("Dataset shape after make_dataset:", dataset.shape)

    # --- FIX: Drop NaNs BEFORE adding lag/rolling features ---
    # Drop rows with NaN values potentially introduced during merging
    if dataset is not None and not dataset.empty:
        initial_rows = len(dataset)
        # Optional: Specify subset=['target'] if you only want to drop based on target NaNs
        dataset.dropna(inplace=True)
        rows_after_drop = len(dataset)
        print(f"Dataset shape after initial NaN drop: {dataset.shape}")
        if initial_rows != rows_after_drop:
            print(f"Note: Dropped {initial_rows - rows_after_drop} rows containing NaNs from merging.")
    elif dataset is None or dataset.empty:
         print("Error: Dataset is None or empty after make_dataset. Cannot proceed.")
         return
    # --- End FIX ---

    # --- Add Lag and Rolling Features ---
    if not dataset.empty:
         group_cols = ['county', 'product_type', 'is_business', 'is_consumption']
         if all(col in dataset.columns for col in group_cols):
              dataset = dp.add_lag_rolling_features(dataset, group_cols=group_cols)
              print("Lag/Rolling features added.")
              print("Dataset shape after adding lag/roll features:", dataset.shape)
         else:
              missing_group_cols = [col for col in group_cols if col not in dataset.columns]
              print(f"Warning: Cannot add grouped lag/rolling features. Missing group columns: {missing_group_cols}. Skipping this step.")
    # --- End Lag/Rolling Feature Addition ---

    # --- FIX: REMOVE the dropna call that was here ---
    # dataset.dropna(inplace=True) # REMOVED - XGBoost handles NaNs in features

    if dataset.empty: # Check if empty after potential issues in lag/roll add
        print("Error: Dataset became empty after adding lag/roll features (or was empty before). Cannot proceed.")
        return

    # ---------------------------
    # Prepare data (Features X, Target y)
    # ---------------------------
    # Drop columns used only for grouping/sorting if they won't be features
    # Also drop the original datetime column after features are created
    cols_to_drop_before_train = ['datetime', 'date'] + group_cols # Drop group cols if not used as features
    X = dataset.drop(columns=["target"] + cols_to_drop_before_train, errors='ignore') # Drop target and others
    y = dataset.target
    feature_names = X.columns.tolist() # Get final feature names

    # Ensure X contains only numeric types for scaling/XGBoost DMatrix
    bool_cols = X.select_dtypes(include='bool').columns
    for col in bool_cols:
        X[col] = X[col].astype(int) # Convert bools included as features

    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"Error: Non-numeric columns found in features: {non_numeric_cols.tolist()}")
        print("Please handle or remove non-numeric features before training.")
        return

    # Convert to numpy arrays (as per original working script)
    # XGBoost DMatrix can handle NaNs in numpy arrays
    X_np = X.values
    y_np = y.values

    print(f"Data prepared: X shape {X_np.shape}, y shape {y_np.shape}")
    print(f"Features: {feature_names}")

    # ---------------------------
    # Multiple Runs with Different Seeds
    # ---------------------------
    all_scores = [] # List to store evaluation scores from each run
    print(f"\n--- Starting {N_RUNS} Training Runs ---")

    for i, current_seed in enumerate(SEEDS):
        print(f"\nRun {i+1}/{N_RUNS} (Seed: {current_seed})")

        # --- Split data using current seed ---
        X_train, X_val, y_train, y_val = train_test_split(X_np, y_np,
                                                          test_size=(1 - TRAIN_SPLIT_RATIO),
                                                          random_state=current_seed) # Use current seed
        print(f"  Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        # --- Scaling ---
        print("  Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        # Note: Scaler might produce NaNs if a column has zero variance in training fold.
        # XGBoost should handle these if they occur.
        print("  Scaling complete.")

        # --- Model Training ---
        print("  Creating XGBoost model parameters...")
        # Get base parameters and set the current seed
        params = modeling.create_xgb_model() # Assumes modeling.py has the desired base params
        params['seed'] = current_seed # Set XGBoost's internal seed

        print("  Training the XGBoost model...")
        # Pass feature names to train_xgb_model if it uses them for DMatrix
        xgb_model = modeling.train_xgb_model(params,
                                              X_train_scaled, y_train,
                                              X_val_scaled, y_val,
                                              num_rounds=XGB_NUM_ROUNDS
                                              # feature_names=feature_names # Uncomment if train_xgb_model accepts/uses it
                                              )

        # --- Evaluation ---
        print("  Evaluating the model...")
        # Pass feature names to predict_xgb_model if needed
        y_val_pred = modeling.predict_xgb_model(xgb_model, X_val_scaled) # feature_names=feature_names
        # Use evaluation.evaluate_model which returns a dict {'MAE': ..., 'MSE': ...}
        eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
        print(f"  Validation Scores: {eval_scores}")
        all_scores.append(eval_scores)
        # --- End Run ---

    # ---------------------------
    # Calculate and Print Statistics
    # ---------------------------
    print("\n--- Overall Training Summary ---")
    if not all_scores:
        print("No scores collected.")
        return

    mae_values = [score['MAE'] for score in all_scores]
    mse_values = [score['MSE'] for score in all_scores] # Collect MSE as well

    avg_mae = np.mean(mae_values)
    min_mae = np.min(mae_values)
    max_mae = np.max(mae_values)
    std_mae = np.std(mae_values)

    avg_mse = np.mean(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    std_mse = np.std(mse_values)

    print(f"Validation MAE over {N_RUNS} runs:")
    print(f"  Average: {avg_mae:.4f}")
    print(f"  Min    : {min_mae:.4f}")
    print(f"  Max    : {max_mae:.4f}")
    print(f"  Std Dev: {std_mae:.4f}")

    print(f"\nValidation MSE over {N_RUNS} runs:")
    print(f"  Average: {avg_mse:.4f}")
    print(f"  Min    : {min_mse:.4f}")
    print(f"  Max    : {max_mse:.4f}")
    print(f"  Std Dev: {std_mse:.4f}")


    # ---------------------------
    # Visualization (Optional: Plot results from median run?)
    # ---------------------------
    # Plotting the last run might not be representative.
    # print("\nPlotting predictions from the last run...")
    # visualization.plot_predictions(pd.Series(y_val), pd.Series(y_val_pred), sample_size=100)

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
