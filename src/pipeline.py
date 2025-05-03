import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Memory # Import joblib

# Import from other modules in the package using absolute imports from 'src'
# Assuming you run from the parent directory of 'src'
from src import data_loader
from src import data_processing
# --- FIX: Correct import name ---
from src import data_engineering # Use correct module name
# --- End FIX ---
from src import modeling
from src import evaluation
from src import visualization
from src import config
import os # Import os for cache directory creation

# --- Caching Setup ---
# Define cache directory relative to this file's location might be more robust
# Or use an absolute path if preferred
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".pipeline_cache") # Place cache outside src
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR)
else:
    print(f"Using cache directory: {CACHE_DIR}")
memory = Memory(CACHE_DIR, verbose=0) # Initialize joblib Memory cache

# --- Cache the weather processing function ---
# Wrap the function imported from data_processing
# Ensure data_processing.py is the correct version (parallel weather, original apply ops)
cached_process_weather_data = memory.cache(data_processing.process_weather_data)
# --- End Caching Setup ---


def run_pipeline(
    data_sample_size=config.DATA_SAMPLE_SIZE,
    train_split_ratio=0.8,
    xgb_num_rounds=1000,
    random_state=63641,
    n_runs=10 # Number of runs for multi-seed evaluation
    ):
    """
    Runs the complete energy prediction pipeline.
    """
    print("Starting energy prediction pipeline...")

    # 1. Load Data
    # -----------
    all_data = data_loader.load_all_data()
    prosumer = all_data["prosumer"]
    weather_forecast = all_data["weather_forecast"]
    # Apply sampling early
    prosumer = prosumer.head(data_sample_size)
    weather_forecast = weather_forecast.iloc[:data_sample_size, :] # Use iloc for weather

    # 2. Initial Processing & Merging
    # -------------------------------
    # Process individual components (weather uses parallel geopy)
    # Use the CACHED version of the weather processing function
    print("Processing weather data (will load from cache if available)...")
    weather_data = cached_process_weather_data(
        weather_forecast,
        all_data["weather_stations"],
        config.WEATHER_FEATURE_NAMES
    )
    # Process other data normally
    prosumer_data = data_processing.process_prosumer_data(
        prosumer,
        all_data["client"]
    )
    electricity_data = data_processing.process_elec_price_data(
        all_data["electricity_prices"]
    )
    gas_data = data_processing.process_gas_price_data(
        all_data["gas_prices"]
    )

    # Merge processed data
    # Note: make_merged_dataset does NOT need the process_weather_func argument anymore
    merged_data = data_processing.make_merged_dataset(
        prosumer_data,
        weather_data,
        electricity_data,
        gas_data
    )
    print("Dataset shape after merging:", merged_data.shape)

    # 3. Feature Engineering
    # ----------------------
    # Add Time Features (using the datetime column from prosumer_data)
    # --- FIX: Use correct module name ---
    data_with_time_feats = data_engineering.add_time_features(merged_data)
    # --- End FIX ---

    # Add Lag/Rolling Features
    group_cols = ['county', 'product_type', 'is_business', 'is_consumption']
    if all(col in data_with_time_feats.columns for col in group_cols):
         # --- FIX: Use correct module name ---
         data_with_all_feats = data_engineering.add_lag_rolling_features(
             data_with_time_feats,
             group_cols=group_cols
         )
         # --- End FIX ---
         print("Lag/Rolling features added.")
    else:
         missing_group_cols = [col for col in group_cols if col not in data_with_time_feats.columns]
         print(f"Warning: Cannot add lag/rolling features. Missing group columns: {missing_group_cols}.")
         data_with_all_feats = data_with_time_feats # Continue without lag/roll

    print("Dataset shape after feature engineering:", data_with_all_feats.shape)

    # 4. Final NaN Drop & Preparation
    # -------------------------------
    if data_with_all_feats.empty:
        print("Error: Dataset is empty before final NaN drop.")
        return

    # --- Keep the logic to NOT drop NaNs aggressively here ---
    dataset = data_with_all_feats # Use the dataframe directly (contains NaNs in lag/roll features)
    # --- End Keep ---

    # Check if empty (unlikely now, but good practice)
    if dataset.empty:
        print("Error: Dataset is empty (this shouldn't happen here). Cannot proceed.")
        return
    print(f"Dataset shape before training (includes NaNs from lags/rolls): {dataset.shape}")


    # Prepare features (X) and target (y)
    # --- FIX: Add 'forecast_date' to columns to drop ---
    cols_to_drop_before_train = ['datetime', 'date', 'forecast_date'] + group_cols
    # --- End FIX ---
    # Also drop target from X
    X = dataset.drop(columns=["target"] + cols_to_drop_before_train, errors='ignore')
    y = dataset.target # Target Series might still have NaNs if original target had them
    feature_names = X.columns.tolist()

    # --- Handle potential NaNs in Target ---
    # XGBoost cannot handle NaNs in the target variable 'y'
    # Find rows where target is NOT NaN
    valid_target_indices = y.dropna().index
    if len(valid_target_indices) < len(y):
        print(f"Warning: Dropping {len(y) - len(valid_target_indices)} rows due to NaN in target variable.")
        X = X.loc[valid_target_indices]
        y = y.loc[valid_target_indices]
        if X.empty:
            print("Error: Dataset empty after dropping target NaNs.")
            return
    # --- End Target NaN Handling ---


    # Ensure numeric types in X
    bool_cols = X.select_dtypes(include='bool').columns
    for col in bool_cols: X[col] = X[col].astype(int)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"Error: Non-numeric columns found: {non_numeric_cols.tolist()}")
        return

    X_np = X.values
    y_np = y.values
    print(f"Data prepared for training (after target NaN drop): X shape {X_np.shape}, y shape {y_np.shape}")
    print(f"Features: {feature_names}")


    # 5. Model Training & Evaluation (Multiple Seeds)
    # -----------------------------------------------
    all_scores = []
    seeds = [random_state + i for i in range(n_runs)]
    print(f"\n--- Starting {n_runs} Training Runs ---")

    for i, current_seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (Seed: {current_seed})")
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=(1 - train_split_ratio), random_state=current_seed
        )
        print(f"  Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        print("  Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        print("  Scaling complete.")

        print("  Creating XGBoost model parameters...")
        params = modeling.create_xgb_model() # Assumes modeling.py has desired base params
        params['seed'] = current_seed

        print("  Training the XGBoost model...")
        xgb_model = modeling.train_xgb_model(
            params, X_train_scaled, y_train, X_val_scaled, y_val, num_rounds=xgb_num_rounds
        )

        print("  Evaluating the model...")
        y_val_pred = modeling.predict_xgb_model(xgb_model, X_val_scaled)
        eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
        print(f"  Validation Scores: {eval_scores}")
        all_scores.append(eval_scores)

    # 6. Report Results
    # -----------------
    print("\n--- Overall Training Summary ---")
    if not all_scores:
        print("No scores collected.")
        return

    mae_values = [score['MAE'] for score in all_scores]
    mse_values = [score['MSE'] for score in all_scores]

    avg_mae = np.mean(mae_values)
    min_mae = np.min(mae_values)
    max_mae = np.max(mae_values)
    std_mae = np.std(mae_values)

    avg_mse = np.mean(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    std_mse = np.std(mse_values)

    print(f"Validation MAE over {n_runs} runs:")
    print(f"  Average: {avg_mae:.4f}")
    print(f"  Min    : {min_mae:.4f}")
    print(f"  Max    : {max_mae:.4f}")
    print(f"  Std Dev: {std_mae:.4f}")

    print(f"\nValidation MSE over {n_runs} runs:")
    print(f"  Average: {avg_mse:.4f}")
    print(f"  Min    : {min_mse:.4f}")
    print(f"  Max    : {max_mse:.4f}")
    print(f"  Std Dev: {std_mse:.4f}")

    # 7. Visualization (Optional - using last run's data)
    # ---------------------------------------------------
    # print("\nPlotting predictions from the last run...")
    # visualization.plot_predictions(pd.Series(y_val), pd.Series(y_val_pred), sample_size=100)

    print("\nPipeline finished successfully.")

