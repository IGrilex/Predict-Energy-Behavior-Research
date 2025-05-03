import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Memory # Import joblib
# --- Add TensorBoard Import ---
from torch.utils.tensorboard import SummaryWriter
# --- End Add ---

# Import from other modules in the package using absolute imports from 'src'
from src import data_loader
from src import data_processing
from src import data_engineering
# --- Import the updated modeling module ---
from src import modeling
# --- End Import ---
from src import evaluation
from src import visualization
from src import config
import os # Import os for cache directory creation

# --- Caching Setup ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".pipeline_cache")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR)
else:
    print(f"Using cache directory: {CACHE_DIR}")
memory = Memory(CACHE_DIR, verbose=0)
cached_process_weather_data = memory.cache(data_processing.process_weather_data)
# --- End Caching Setup ---

# --- TensorBoard Log Directory ---
LOG_DIR = "logs" # Base directory for TensorBoard logs
# --- End Log Dir ---


def run_pipeline(
    data_sample_size=config.DATA_SAMPLE_SIZE,
    train_split_ratio=0.8,
    xgb_num_rounds=1000,
    random_state=63641,
    n_runs=10 # Number of runs for multi-seed evaluation
    ):
    """
    Runs the complete energy prediction pipeline with TensorBoard logging.
    """
    print("Starting energy prediction pipeline...")

    # 1. Load Data
    # -----------
    all_data = data_loader.load_all_data()
    prosumer = all_data["prosumer"]
    weather_forecast = all_data["weather_forecast"]
    prosumer = prosumer.head(data_sample_size)
    weather_forecast = weather_forecast.iloc[:data_sample_size, :]

    # 2. Initial Processing & Merging
    # -------------------------------
    print("Processing weather data (will load from cache if available)...")
    weather_data = cached_process_weather_data(
        weather_forecast, all_data["weather_stations"], config.WEATHER_FEATURE_NAMES
    )
    prosumer_data = data_processing.process_prosumer_data(prosumer, all_data["client"])
    electricity_data = data_processing.process_elec_price_data(all_data["electricity_prices"])
    gas_data = data_processing.process_gas_price_data(all_data["gas_prices"])
    merged_data = data_processing.make_merged_dataset(
        prosumer_data, weather_data, electricity_data, gas_data
    )
    print("Dataset shape after merging:", merged_data.shape)

    # 3. Feature Engineering
    # ----------------------
    data_with_time_feats = data_engineering.add_time_features(merged_data)
    group_cols = ['county', 'product_type', 'is_business', 'is_consumption']
    if all(col in data_with_time_feats.columns for col in group_cols):
         data_with_all_feats = data_engineering.add_lag_rolling_features(
             data_with_time_feats, group_cols=group_cols
         )
         print("Lag/Rolling features added.")
    else:
         missing_group_cols = [col for col in group_cols if col not in data_with_time_feats.columns]
         print(f"Warning: Cannot add lag/rolling features. Missing group columns: {missing_group_cols}.")
         data_with_all_feats = data_with_time_feats

    print("Dataset shape after feature engineering:", data_with_all_feats.shape)

    # 4. Final NaN Drop & Preparation
    # -------------------------------
    if data_with_all_feats.empty: return print("Error: Dataset empty before NaN drop.")
    dataset = data_with_all_feats # Keep NaNs from lag/roll
    print(f"Dataset shape before training (includes NaNs from lags/rolls): {dataset.shape}")

    cols_to_drop_before_train = ['datetime', 'date', 'forecast_date'] + group_cols
    X = dataset.drop(columns=["target"] + cols_to_drop_before_train, errors='ignore')
    y = dataset.target
    feature_names = X.columns.tolist()

    valid_target_indices = y.dropna().index
    if len(valid_target_indices) < len(y):
        print(f"Warning: Dropping {len(y) - len(valid_target_indices)} rows due to NaN in target variable.")
        X = X.loc[valid_target_indices]
        y = y.loc[valid_target_indices]
        if X.empty: return print("Error: Dataset empty after dropping target NaNs.")

    bool_cols = X.select_dtypes(include='bool').columns
    for col in bool_cols: X[col] = X[col].astype(int)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty: return print(f"Error: Non-numeric columns found: {non_numeric_cols.tolist()}")

    X_np = X.values
    y_np = y.values
    print(f"Data prepared for training (after target NaN drop): X shape {X_np.shape}, y shape {y_np.shape}")
    print(f"Features: {feature_names}")


    # 5. Model Training & Evaluation (Multiple Seeds with TensorBoard)
    # ----------------------------------------------------------------
    all_scores = []
    seeds = [random_state + i for i in range(n_runs)]
    print(f"\n--- Starting {n_runs} Training Runs ---")

    # Create base log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    for i, current_seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (Seed: {current_seed})")
        run_name = f"run_seed_{current_seed}" # Unique name for this run

        # --- TensorBoard Setup for this run ---
        log_path = os.path.join(LOG_DIR, run_name)
        writer = SummaryWriter(log_dir=log_path)
        print(f"  TensorBoard logs will be saved to: {log_path}")
        # --- End TensorBoard Setup ---

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
        params = modeling.create_xgb_model()
        params['seed'] = current_seed

        # --- Create and add TensorBoard callback ---
        tb_callback = modeling.TensorBoardCallback(writer, run_name=run_name)
        callbacks_list = [tb_callback]
        # --- End Add Callback ---

        print("  Training the XGBoost model...")
        xgb_model = modeling.train_xgb_model(
            params, X_train_scaled, y_train, X_val_scaled, y_val,
            num_rounds=xgb_num_rounds,
            callbacks=callbacks_list # Pass the callback list
        )

        print("  Evaluating the model...")
        y_val_pred = modeling.predict_xgb_model(xgb_model, X_val_scaled)
        eval_scores = evaluation.evaluate_model(y_val, y_val_pred)
        print(f"  Validation Scores: {eval_scores}")
        all_scores.append(eval_scores)

        # --- Log HParams and Final Metrics to TensorBoard ---
        # Ensure parameters are suitable for logging (convert complex objects if needed)
        hparams_log = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
        final_metrics_log = {
            f'hparam/final_MAE': eval_scores['MAE'],
            f'hparam/final_MSE': eval_scores['MSE'],
            f'hparam/best_iteration': xgb_model.best_iteration + 1
        }
        # Note: add_hparams expects metrics in the metric_dict argument
        writer.add_hparams(hparams_log, final_metrics_log)
        # --- End HParam Logging ---

        # --- Close TensorBoard Writer for this run ---
        writer.close()
        # --- End Close ---

    # 6. Report Results
    # -----------------
    print("\n--- Overall Training Summary ---")
    if not all_scores: return print("No scores collected.")

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
    print(f"\nTo view TensorBoard logs, run:\ntensorboard --logdir {LOG_DIR}")

