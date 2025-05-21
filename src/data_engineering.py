import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def add_time_features(df: pd.DataFrame, datetime_col='datetime') -> pd.DataFrame:
    """Adds time-based features (hour, dayofweek, month, etc.) to the DataFrame."""
    tqdm.write("Creating time-based features...")
    # Ensure datetime column exists and is datetime type
    if datetime_col not in df.columns:
        tqdm.write(f"Error: '{datetime_col}' column missing. Cannot add time features.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        tqdm.write(f"Converting '{datetime_col}' to datetime...")
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col]) # Drop failures

    if df.empty:
        tqdm.write("Warning: DataFrame empty after datetime handling in add_time_features.")
        return df

    # Extract time features
    dt_col_accessor = df[datetime_col].dt
    df['hour'] = dt_col_accessor.hour
    df['dayofweek'] = dt_col_accessor.dayofweek # Monday=0, Sunday=6
    df['dayofmonth'] = dt_col_accessor.day
    df['dayofyear'] = dt_col_accessor.dayofyear
    df['month'] = dt_col_accessor.month
    df['year'] = dt_col_accessor.year
    df['weekofyear'] = dt_col_accessor.isocalendar().week.astype(int)
    df['quarter'] = dt_col_accessor.quarter
    tqdm.write("Time-based features added.")
    return df


def add_lag_rolling_features(df: pd.DataFrame,
                             group_cols=['county', 'product_type', 'is_business', 'is_consumption'], # Columns defining unique time series
                             datetime_col='datetime', # Specify the datetime column for sorting
                             target_col='target',
                             lags=[1, 2, 3, 24, 48, 168], # e.g., 1h, 2h, 3h, 1day, 2day, 1week
                             windows=[3, 6, 12, 24], # e.g., 3h, 6h, 12h, 24h rolling stats
                             features_to_roll=['target', 'electricity_price', 'avg_gas_price'] # Add relevant features
                             ):
    """
    Adds lag and rolling window features to the dataframe.
    Assumes df is sorted by time within groups if group_cols are provided.
    """
    tqdm.write(f"Adding lag/rolling features for target '{target_col}' and features {features_to_roll}...")

    # Ensure datetime column exists for sorting
    if datetime_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        tqdm.write(f"Error: '{datetime_col}' column missing or not datetime type. Cannot reliably sort for lags/rolling.")
        return df # Return original df

    # Ensure group columns exist
    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
         tqdm.write(f"Warning: Cannot add grouped lag/rolling features. Missing group columns: {missing_group_cols}. Skipping this step.")
         return df

    # Sort data to ensure correct shifts/rolling
    # Sorting by group columns first, then time is crucial for grouped operations
    tqdm.write(f"Sorting data by {group_cols + [datetime_col]}...")
    df = df.sort_values(by=group_cols + [datetime_col])

    # Create Lag Features for the target
    if target_col in df.columns:
        tqdm.write(f"  Creating lag features for '{target_col}' with lags: {lags}")
        # Group by the specified columns and then apply shift
        grouped_data = df.groupby(group_cols, observed=False, group_keys=False)[target_col]
        for lag in lags:
            lag_col_name = f'{target_col}_lag_{lag}'
            df[lag_col_name] = grouped_data.shift(lag)
    else:
        tqdm.write(f"Warning: Target column '{target_col}' not found for lag features.")

    # Create Rolling Window Features
    for feature in features_to_roll:
        if feature in df.columns:
            tqdm.write(f"  Creating rolling window features for '{feature}' with windows: {windows}")
            # Group by the specified columns first
            grouped_feature = df.groupby(group_cols, observed=False, group_keys=False)[feature]
            # Shift by 1 within each group to use past data only
            shifted_feature = grouped_feature.shift(1)

            for window in windows:
                roll_mean_col = f'{feature}_roll_mean_{window}'
                roll_std_col = f'{feature}_roll_std_{window}'

                # Calculate rolling stats on the shifted data within each group
                # Assign back using the original index
                df[roll_mean_col] = shifted_feature.rolling(window=window, min_periods=1).mean()
                df[roll_std_col] = shifted_feature.rolling(window=window, min_periods=1).std()

            # Fill initial NaNs in std dev (first value has no std dev)
            std_cols = [f'{feature}_roll_std_{w}' for w in windows if f'{feature}_roll_std_{w}' in df.columns]
            df[std_cols] = df[std_cols].fillna(0) # Replace NaN std dev with 0
        else:
            tqdm.write(f"Warning: Column '{feature}' not found for rolling window features.")

    tqdm.write("Finished adding lag/rolling features.")
    return df

