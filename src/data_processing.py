import numpy as np
import pandas as pd
# --- Use geopy for accuracy ---
from geopy import distance
# --- Add joblib for parallelization ---
from joblib import Parallel, delayed
# --- End Add ---
# Import tqdm for progress bars
from tqdm.auto import tqdm

# Initialize tqdm for pandas apply functions
tqdm.pandas(desc="Pandas Apply")


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Generic function to load a CSV file.
    """
    return pd.read_csv(file_path, **kwargs)

def load_json(file_path: str, typ="series") -> pd.DataFrame:
    """
    Load JSON file; here we're assuming the JSON represents a series.
    """
    return pd.read_json(file_path, typ=typ)

# --- Weather Station Lookup Function (Optimized for Parallelization) ---
def find_closest_weather_station(point_tuple: tuple, wss_coords_list: list, wss_original_indices: pd.Index, wss_lookup: pd.DataFrame):
    """
    Find closest weather station with county information to a given point tuple.
    Accepts pre-calculated station coordinates and indices for efficiency.
    (Original Slow Version using geopy - Optimized for Parallel Call)
    """
    point_latitude = getattr(point_tuple, 'latitude', np.nan)
    point_longitude = getattr(point_tuple, 'longitude', np.nan)
    if pd.isna(point_latitude) or pd.isna(point_longitude): return None, None, np.inf
    point_coordinates = [point_latitude, point_longitude]
    if not wss_coords_list: return None, None, np.inf
    dists = [distance.distance(point_coordinates, station_coords).km for station_coords in wss_coords_list]
    if not dists: return None, None, np.inf
    closest_dist_idx_in_list = np.argmin(dists)
    closest_dist = dists[closest_dist_idx_in_list]
    closest_station_original_idx = wss_original_indices[closest_dist_idx_in_list]
    closest_station = wss_lookup.loc[closest_station_original_idx]
    if closest_station is not None and closest_dist < 30:
        return closest_station.county_name, closest_station.county, closest_dist
    else:
        return None, None, closest_dist

# --- MODIFIED: Parallelized Weather Processing (Optimized Worker Call) ---
def process_weather_data(wd: pd.DataFrame, wss: pd.DataFrame, feature_names: list):
    """
    Process weather data by adding weather station info using geopy (parallelized),
    and aggregating by county and timestamp. (Parallelized Version - Optimized Worker Call)
    """
    tqdm.write("Finding closest weather stations (Original Slow Method - Parallelized)...")
    wss_lookup_cols = ['latitude', 'longitude', 'county_name', 'county']
    wss_filtered = wss[wss_lookup_cols].dropna().copy()
    if wss_filtered.empty:
        tqdm.write("Warning: Weather station data has no valid coordinates or required columns.")
        return pd.DataFrame(columns=['county', 'forecast_datetime'] + feature_names).set_index(['county', 'forecast_datetime'])
    wss_coords_list = wss_filtered[['latitude', 'longitude']].values.tolist()
    wss_original_indices = wss_filtered.index
    wss_lookup_df = wss_filtered[['county_name', 'county']]
    results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(find_closest_weather_station)(row_tuple, wss_coords_list, wss_original_indices, wss_lookup_df)
        for row_tuple in tqdm(wd.itertuples(index=False), total=len(wd), desc="Processing Weather Rows Parallel")
    )
    county_names = [res[0] for res in results]
    county_ids = [res[1] for res in results]
    wd_processed = wd.copy()
    wd_processed["county_name"] = county_names
    wd_processed["county"] = county_ids
    wd_processed = wd_processed.dropna(subset=["county"], axis=0)
    if wd_processed.empty:
        tqdm.write("Warning: Weather data empty after finding stations and dropping NaNs.")
        return pd.DataFrame(columns=['county', 'forecast_datetime'] + feature_names).set_index(['county', 'forecast_datetime'])

    # Format timestamps using original apply (to keep string format for grouping/merging)
    tqdm.write("Formatting weather timestamps (Original Apply)...")
    # Ensure 'forecast_datetime' is datetime before formatting
    if 'forecast_datetime' in wd_processed.columns and not pd.api.types.is_datetime64_any_dtype(wd_processed['forecast_datetime']):
        wd_processed['forecast_datetime'] = pd.to_datetime(wd_processed['forecast_datetime'], errors='coerce')
        wd_processed = wd_processed.dropna(subset=['forecast_datetime']) # Drop if conversion fails

    if wd_processed.empty:
         tqdm.write("Warning: Weather data empty after datetime conversion/dropna.")
         return pd.DataFrame(columns=['county', 'forecast_datetime'] + feature_names).set_index(['county', 'forecast_datetime'])

    # Use progress_apply for the string formatting step
    # Create new string column for index/merge key
    wd_processed["forecast_datetime_str"] = wd_processed["forecast_datetime"].progress_apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else None)
    wd_processed = wd_processed.dropna(subset=["forecast_datetime_str"]) # Drop if formatting failed

    if wd_processed.empty:
        tqdm.write("Warning: Weather data empty after datetime formatting/dropna.")
        return pd.DataFrame(columns=['county', 'forecast_datetime'] + feature_names).set_index(['county', 'forecast_datetime'])


    # Aggregate data per county and timestamp string
    tqdm.write("Aggregating weather data...")
    # Ensure county is appropriate type before grouping
    wd_processed['county'] = wd_processed['county'].astype(int)
    # Group by the STRING datetime column
    # Add observed=False to silence future warnings and maintain current behavior
    wd_aggregated = wd_processed.groupby(["county", "forecast_datetime_str"], observed=False)[feature_names].mean()
    tqdm.write(f"Weather data aggregated shape: {wd_aggregated.shape}")
    return wd_aggregated # Index is now (county, forecast_datetime_str)
# --- End Parallelized Weather Processing ---


# --- MODIFIED: Added Time Feature Engineering ---
def process_prosumer_data(prosumers: pd.DataFrame, clients: pd.DataFrame):
    """
    Process prosumer data: adds time features, merges with client data.
    (Original apply for date string, Original Inner Joins for client)
    """
    prosumers_proc = prosumers.copy()

    # --- Feature Engineering: Time-Based Features ---
    tqdm.write("Creating time-based features...")
    # Ensure datetime column exists and is datetime type
    if 'datetime' not in prosumers_proc.columns:
        tqdm.write("Error: 'datetime' column missing in prosumer data.")
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(prosumers_proc['datetime']):
        prosumers_proc['datetime'] = pd.to_datetime(prosumers_proc['datetime'], errors='coerce')
        prosumers_proc = prosumers_proc.dropna(subset=['datetime']) # Drop failures

    if prosumers_proc.empty:
        tqdm.write("Warning: Prosumer data empty after initial datetime handling.")
        return pd.DataFrame()

    # Extract time features
    dt_col = prosumers_proc['datetime']
    prosumers_proc['hour'] = dt_col.dt.hour
    prosumers_proc['dayofweek'] = dt_col.dt.dayofweek # Monday=0, Sunday=6
    prosumers_proc['dayofmonth'] = dt_col.dt.day
    prosumers_proc['dayofyear'] = dt_col.dt.dayofyear
    prosumers_proc['month'] = dt_col.dt.month
    prosumers_proc['year'] = dt_col.dt.year
    prosumers_proc['weekofyear'] = dt_col.dt.isocalendar().week.astype(int)
    prosumers_proc['quarter'] = dt_col.dt.quarter
    # --- End Feature Engineering ---


    tqdm.write("Formatting prosumer date string (Original Apply)...")
    # Create the string date column needed for client merge
    prosumers_proc["date"] = prosumers_proc.datetime.progress_apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None) # Handle potential NaT
    prosumers_proc = prosumers_proc.dropna(subset=['date']) # Drop if formatting failed

    if prosumers_proc.empty:
        tqdm.write("Warning: Prosumer data empty after date string formatting.")
        return pd.DataFrame()

    # Aggregate consumption and capacity by group
    # Ensure client 'date' column is also string 'YYYY-MM-DD'
    if 'date' in clients.columns:
        if not pd.api.types.is_string_dtype(clients['date']):
             clients['date'] = pd.to_datetime(clients['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        clients = clients.dropna(subset=['date'])
    else:
        tqdm.write("Warning: 'date' column missing in client data. Cannot aggregate/merge.")
        return pd.DataFrame()

    if clients.empty:
        tqdm.write("Warning: Client data is empty after date processing. Cannot aggregate/merge.")
        return pd.DataFrame()


    tqdm.write("Aggregating client data...")
    # Grouping using original string date
    # Add observed=False to silence future warnings and maintain current behavior
    cons = clients.groupby(["product_type", "county", "is_business", "date"], observed=False)["eic_count"].sum().reset_index()
    cap = clients.groupby(["product_type", "county", "is_business", "date"], observed=False)["installed_capacity"].sum().reset_index()

    if cons.empty or cap.empty:
        tqdm.write("Warning: Client aggregation resulted in empty dataframes. Inner join will fail.")
        return pd.DataFrame()


    tqdm.write("Merging prosumer and client data (inner join)...")
    # Ensure merge keys are of compatible types
    try:
        # Check if 'county' exists in both and if types differ
        if 'county' in prosumers_proc.columns and 'county' in cons.columns:
            target_county_dtype = cons['county'].dtype
            if prosumers_proc['county'].dtype != target_county_dtype:
                prosumers_proc['county'] = prosumers_proc['county'].astype(target_county_dtype)
        elif 'county' not in prosumers_proc.columns:
             tqdm.write("Warning: 'county' column missing in prosumer data before client merge.")
             return pd.DataFrame() # Cannot merge without county
    except Exception as e:
        tqdm.write(f"Warning: Could not match county dtypes before client merge. Error: {e}")

    # Define merge keys explicitly
    merge_keys = ["product_type", "county", "is_business", "date"]
    # Check if all keys exist in both dataframes
    if not all(key in prosumers_proc.columns for key in merge_keys):
        tqdm.write(f"Error: Missing one or more merge keys in prosumer data: {merge_keys}")
        return pd.DataFrame()
    if not all(key in cons.columns for key in merge_keys):
        tqdm.write(f"Error: Missing one or more merge keys in client 'cons' data: {merge_keys}")
        return pd.DataFrame()
    if not all(key in cap.columns for key in merge_keys):
         tqdm.write(f"Error: Missing one or more merge keys in client 'cap' data: {merge_keys}")
         return pd.DataFrame()


    # Merge on the string 'date' column created by apply
    prosumers_proc = pd.merge(prosumers_proc, cons, on=merge_keys, how="inner")
    if prosumers_proc.empty:
        tqdm.write("Warning: Prosumer data empty after merging with client eic_count.")
        return pd.DataFrame()

    prosumers_proc = pd.merge(prosumers_proc, cap, on=merge_keys, how="inner")
    tqdm.write(f"Processed prosumer data shape: {prosumers_proc.shape}")
    # Return dataframe with time features, 'datetime' as datetime object, and 'date' as string
    return prosumers_proc
# --- End MODIFIED Prosumer Processing ---

# --- Original Elec Price Processing ---
def process_elec_price_data(data: pd.DataFrame):
    """
    Process electricity price data: rename column and ensure 'forecast_date'
    is string 'YYYY-MM-DD HH:MM:SS' format for merging.
    """
    data_processed = data.rename(columns={"euros_per_mwh": "electricity_price"})
    # --- Ensure forecast_date is correct string format ---
    if 'forecast_date' in data_processed.columns:
        if not pd.api.types.is_string_dtype(data_processed['forecast_date']):
            # Convert to datetime first (handles various inputs), then format
            data_processed['forecast_date'] = pd.to_datetime(data_processed['forecast_date'], errors='coerce')
            data_processed = data_processed.dropna(subset=['forecast_date']) # Drop failures
            # Format to match the prosumer datetime string key
            data_processed['forecast_date'] = data_processed['forecast_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        tqdm.write("Warning: 'forecast_date' missing in electricity price data.")
    return data_processed
# --- End Original Elec Price Processing ---

# --- REVERTED: Original Gas Price Processing (using apply) ---
def process_gas_price_data(data: pd.DataFrame):
    """
    Process gas price data: calculate the average gas price.
    (Original Version using apply)
    """
    data_processed = data.copy()
    tqdm.write("Calculating average gas price (Original Apply)...")
    # --- REVERTED to .apply ---
    data_processed["avg_gas_price"] = data.progress_apply(
        lambda row: np.mean([row.lowest_price_per_mwh, row.highest_price_per_mwh])
                    if pd.notna(row.lowest_price_per_mwh) and pd.notna(row.highest_price_per_mwh) else np.nan,
        axis=1)
    # --- End REVERT ---

    # --- Ensure forecast_date is correct string format ---
    if 'forecast_date' in data_processed.columns:
         if not pd.api.types.is_string_dtype(data_processed['forecast_date']):
             data_processed['forecast_date'] = pd.to_datetime(data_processed['forecast_date'], errors='coerce').dt.strftime('%Y-%m-%d')
         data_processed = data_processed.dropna(subset=['forecast_date'])
    else:
         tqdm.write("Warning: 'forecast_date' missing in gas price data.")
    return data_processed
# --- End REVERTED Gas Price Processing ---


# --- NEW FUNCTION: Add Lag and Rolling Window Features ---
def add_lag_rolling_features(df: pd.DataFrame,
                             group_cols=['county', 'product_type', 'is_business', 'is_consumption'], # Columns defining unique time series
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

    # Ensure datetime column exists for sorting (should exist from process_prosumer_data)
    if 'datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        tqdm.write("Error: 'datetime' column missing or not datetime type. Cannot reliably sort for lags/rolling.")
        return df # Return original df

    # Sort data globally first to ensure correct shifts/rolling if not grouping
    # If grouping, sorting within group is handled by pandas groupby().shift/rolling()
    # It's safer to sort explicitly here if the order isn't guaranteed earlier
    df = df.sort_values(by=['datetime']) # Sort globally first
    # If grouping is essential for correct sorting (e.g., multiple series interleaved)
    # df = df.sort_values(by=group_cols + ['datetime']) # Sort by groups then time

    # Create Lag Features for the target
    if target_col in df.columns:
        tqdm.write(f"  Creating lag features for '{target_col}' with lags: {lags}")
        for lag in lags:
            lag_col_name = f'{target_col}_lag_{lag}'
            if group_cols:
                # Calculate lags within each group to prevent leakage across series
                df[lag_col_name] = df.groupby(group_cols, observed=False, group_keys=False)[target_col].shift(lag)
            else:
                # Calculate lags globally if no grouping
                df[lag_col_name] = df[target_col].shift(lag)
    else:
        tqdm.write(f"Warning: Target column '{target_col}' not found for lag features.")

    # Create Rolling Window Features
    for feature in features_to_roll:
        if feature in df.columns:
            tqdm.write(f"  Creating rolling window features for '{feature}' with windows: {windows}")
            for window in windows:
                # Shift by 1 to ensure rolling window uses past data only (avoids data leakage)
                shifted_feature = df.groupby(group_cols, observed=False, group_keys=False)[feature].shift(1) if group_cols else df[feature].shift(1)

                # Calculate rolling mean and std dev
                roll_mean_col = f'{feature}_roll_mean_{window}'
                roll_std_col = f'{feature}_roll_std_{window}'

                if group_cols:
                    # Apply rolling within each group
                    # Need to group by the same columns again for rolling
                    # Using transform is often cleaner for assigning back
                    df[roll_mean_col] = shifted_feature.groupby(df[group_cols].apply(tuple, axis=1), observed=False).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                    df[roll_std_col] = shifted_feature.groupby(df[group_cols].apply(tuple, axis=1), observed=False).rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
                    # Alternative using transform (might be more robust)
                    # df[roll_mean_col] = df.groupby(group_cols, observed=False, group_keys=False)[feature].shift(1).transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                    # df[roll_std_col] = df.groupby(group_cols, observed=False, group_keys=False)[feature].shift(1).transform(lambda x: x.rolling(window=window, min_periods=1).std())
                else:
                    df[roll_mean_col] = shifted_feature.rolling(window=window, min_periods=1).mean()
                    df[roll_std_col] = shifted_feature.rolling(window=window, min_periods=1).std()

            # Fill initial NaNs in std dev (first value has no std dev)
            std_cols = [f'{feature}_roll_std_{w}' for w in windows if f'{feature}_roll_std_{w}' in df.columns]
            df[std_cols] = df[std_cols].fillna(0) # Replace NaN std dev with 0
        else:
            tqdm.write(f"Warning: Column '{feature}' not found for rolling window features.")

    tqdm.write("Finished adding lag/rolling features.")
    return df
# --- End NEW FUNCTION ---


# --- MODIFIED: Ensure unique columns before returning from make_dataset ---
def make_dataset(prosumer: pd.DataFrame,
                 weather_forecast_data: pd.DataFrame,
                 weather_stations: pd.DataFrame,
                 client: pd.DataFrame,
                 electricity_prices: pd.DataFrame,
                 gas_prices: pd.DataFrame,
                 weather_feature_names: list,
                 process_weather_func=process_weather_data # Accept cached func
                 ):
    """
    Create a combined dataset for prediction by merging processed data.
    Uses specified weather processing function (potentially cached) and
    ORIGINAL apply operations, maintaining original merge logic and STRING keys.
    Includes time features. Lag/Rolling features added AFTER this function in main.py.
    """
    tqdm.write("Starting dataset creation (Parallel Weather, Original Apply Ops, Time Features)...")
    # Process individual data sources
    weather_data = process_weather_func(weather_forecast_data, weather_stations, weather_feature_names) # Parallel Weather, index is (county, forecast_datetime_str)
    prosumer_data = process_prosumer_data(prosumer, client) # Original apply, returns 'date' (string) + time features
    electricity_data = process_elec_price_data(electricity_prices) # Returns 'forecast_date' as string YYYY-MM-DD HH:MM:SS
    gas_data = process_gas_price_data(gas_prices) # Original apply, returns 'forecast_date' as string YYYY-MM-DD

    # --- Check for empty dataframes after processing ---
    if prosumer_data.empty:
         tqdm.write("Warning: Prosumer data is empty after processing. Cannot proceed.")
         return pd.DataFrame()
    if weather_data.empty:
        tqdm.write("Warning: Weather data processing resulted in an empty DataFrame. Inner join will fail.")
        return pd.DataFrame()

    tqdm.write(f"Shape before merging prosumer/weather: Prosumer={prosumer_data.shape}, Weather={weather_data.shape}")

    # --- Merge prosumer and weather data (Original Inner Join on Strings) ---
    tqdm.write("Merging prosumer and weather data...")
    # Create the string datetime key from prosumer's datetime column
    if 'datetime' in prosumer_data.columns:
        if pd.api.types.is_datetime64_any_dtype(prosumer_data['datetime']):
             prosumer_data['datetime_str_key'] = prosumer_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.api.types.is_string_dtype(prosumer_data['datetime']):
             prosumer_data['datetime_str_key'] = prosumer_data['datetime']
        else:
             tqdm.write(f"Converting prosumer datetime from {prosumer_data['datetime'].dtype} to string key")
             prosumer_data['datetime_str_key'] = pd.to_datetime(prosumer_data['datetime'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

        prosumer_data = prosumer_data.dropna(subset=['datetime_str_key'])
        left_dt_key = 'datetime_str_key'
        if prosumer_data.empty:
            tqdm.write("Warning: Prosumer data empty after creating/validating string datetime key.")
            return pd.DataFrame()
    else:
        tqdm.write("Error: 'datetime' column missing in prosumer_data before weather merge.")
        return pd.DataFrame()

    # Ensure county types match
    if 'county' in weather_data.index.names:
         if 'county' in prosumer_data.columns:
             target_weather_county_dtype = weather_data.index.get_level_values('county').dtype
             if prosumer_data['county'].dtype != target_weather_county_dtype:
                  try:
                      prosumer_data['county'] = prosumer_data['county'].astype(target_weather_county_dtype)
                  except Exception as e:
                      tqdm.write(f"Warning: Could not match county dtypes before weather merge. Error: {e}")
         else:
             tqdm.write("Warning: 'county' column missing in prosumer_data for weather merge.")
             return pd.DataFrame()
    else:
        tqdm.write("Warning: 'county' not found in weather_data index. Cannot ensure type match.")

    # Merge using the string key ('datetime_str_key') and string index ('forecast_datetime_str')
    data = pd.merge(prosumer_data,
                    weather_data, # Index is (county, forecast_datetime_str)
                    left_on=["county", left_dt_key], # Use string key from prosumer
                    right_index=True,
                    how="inner")
    tqdm.write(f"Shape after merging prosumer/weather: {data.shape}")
    if data.empty:
        tqdm.write("Warning: Data is empty after inner join between prosumer and weather data.")
        return pd.DataFrame()


    # --- Add price data (Original Left Joins on Strings) ---
    tqdm.write("Merging electricity price data...")
    if electricity_data.empty or 'forecast_date' not in electricity_data.columns or electricity_data['forecast_date'].isnull().all():
        tqdm.write("Warning: Electricity price data missing, lacks 'forecast_date', or column is all NaN. Skipping merge.")
        data['electricity_price'] = np.nan
    else:
        data = pd.merge(data,
                        electricity_data[["forecast_date", "electricity_price"]],
                        left_on=left_dt_key, # Use prosumer datetime string key
                        right_on="forecast_date", # Use electricity forecast_date (string)
                        how="left")
        if 'forecast_date_y' in data.columns: data = data.drop(columns=['forecast_date_y'])
        if 'forecast_date_x' in data.columns: data = data.rename(columns={'forecast_date_x': 'forecast_date'})
    tqdm.write(f"Shape after merging electricity prices: {data.shape}")


    tqdm.write("Merging gas price data...")
    if gas_data.empty or 'forecast_date' not in gas_data.columns or gas_data['forecast_date'].isnull().all():
        tqdm.write("Warning: Gas price data missing, lacks 'forecast_date', or column is all NaN. Skipping merge.")
        data['avg_gas_price'] = np.nan
    else:
        if 'date' not in data.columns: # 'date' is the YYYY-MM-DD string from process_prosumer_data
             tqdm.write("Error: 'date' column missing before gas price merge.")
             data['avg_gas_price'] = np.nan
        else:
             data = pd.merge(data,
                             gas_data[["forecast_date", "avg_gas_price"]],
                             left_on="date", # Use prosumer date string key
                             right_on="forecast_date", # Use gas forecast_date (string YYYY-MM-DD)
                             how="left")
             if 'forecast_date_y' in data.columns: data = data.drop(columns=['forecast_date_y'])
             if 'forecast_date_x' in data.columns: data = data.rename(columns={'forecast_date_x': 'forecast_date'})
    tqdm.write(f"Shape after merging gas prices: {data.shape}")


    # --- Final Feature Selection (Ensure Unique Columns) ---
    # Define the base feature columns
    base_feats = ["is_consumption", "eic_count", "installed_capacity", "electricity_price", "avg_gas_price"] \
            + weather_feature_names

    # Define the time features added in process_prosumer_data
    time_feats = ['hour', 'dayofweek', 'dayofmonth', 'dayofyear', 'month', 'year', 'weekofyear', 'quarter']

    # Define grouping columns needed for lag/roll function (and potentially as features)
    group_cols_for_lag = ['county', 'product_type', 'is_business', 'is_consumption']

    # Combine all columns to keep initially
    keep_cols_initial = base_feats + time_feats + ["target", "datetime"] + group_cols_for_lag

    # --- FIX: Ensure unique columns ---
    # Get columns that actually exist in the dataframe 'data' from the initial list
    present_cols = [col for col in keep_cols_initial if col in data.columns]
    # Get unique columns while preserving order (important for consistency)
    unique_cols = list(pd.Index(present_cols).unique())
    # --- End FIX ---

    # Select the unique present columns
    data_final = data[unique_cols].copy()

    # Report any initially requested columns that were missing
    missing = set(keep_cols_initial) - set(data.columns)
    if missing:
        tqdm.write(f"Warning: Missing expected columns after merges: {missing}.")

    # Check target existence again after selection
    if "target" not in data_final.columns:
         tqdm.write("Error: 'target' column missing after final selection.")
         return pd.DataFrame()

    if data_final.empty:
        tqdm.write("Warning: Dataframe is empty after final feature selection.")
        return pd.DataFrame(columns=unique_cols)

    # --- NO FINAL DROPNA in original code ---
    tqdm.write("Skipping final dropna step (original logic).")

    data_final = data_final.reset_index(drop=True)
    tqdm.write(f"Final dataset shape before returning (Original Logic): {data_final.shape}")

    # Clean up temporary string key if it exists
    if 'datetime_str_key' in data_final.columns:
        data_final = data_final.drop(columns=['datetime_str_key'])

    # Return dataframe including 'datetime' and group columns needed for next step
    return data_final
# --- End MODIFIED make_dataset ---

