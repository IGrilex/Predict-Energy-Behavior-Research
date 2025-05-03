import numpy as np
import pandas as pd
from geopy import distance
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Initialize tqdm for pandas apply functions
tqdm.pandas(desc="Pandas Apply")

# --- Weather Station Lookup Function (Optimized for Parallelization) ---
# (Keep this function as is from the previous working version)
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

# --- Parallelized Weather Processing ---
def process_weather_data(wd: pd.DataFrame, wss: pd.DataFrame, feature_names: list):
    """
    Process weather data by adding weather station info using geopy (parallelized),
    and aggregating by county and timestamp string.
    """
    tqdm.write("Finding closest weather stations (Original Slow Method - Parallelized)...")
    wss_lookup_cols = ['latitude', 'longitude', 'county_name', 'county']
    wss_filtered = wss[wss_lookup_cols].dropna().copy()
    if wss_filtered.empty:
        tqdm.write("Warning: Weather station data has no valid coordinates or required columns.")
        return pd.DataFrame(columns=['county', 'forecast_datetime_str'] + feature_names).set_index(['county', 'forecast_datetime_str']) # Adjusted index name
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
        return pd.DataFrame(columns=['county', 'forecast_datetime_str'] + feature_names).set_index(['county', 'forecast_datetime_str'])

    tqdm.write("Formatting weather timestamps (Original Apply)...")
    if 'forecast_datetime' in wd_processed.columns and not pd.api.types.is_datetime64_any_dtype(wd_processed['forecast_datetime']):
        wd_processed['forecast_datetime'] = pd.to_datetime(wd_processed['forecast_datetime'], errors='coerce')
        wd_processed = wd_processed.dropna(subset=['forecast_datetime'])

    if wd_processed.empty:
         tqdm.write("Warning: Weather data empty after datetime conversion/dropna.")
         return pd.DataFrame(columns=['county', 'forecast_datetime_str'] + feature_names).set_index(['county', 'forecast_datetime_str'])

    wd_processed["forecast_datetime_str"] = wd_processed["forecast_datetime"].progress_apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else None)
    wd_processed = wd_processed.dropna(subset=["forecast_datetime_str"])

    if wd_processed.empty:
        tqdm.write("Warning: Weather data empty after datetime formatting/dropna.")
        return pd.DataFrame(columns=['county', 'forecast_datetime_str'] + feature_names).set_index(['county', 'forecast_datetime_str'])

    tqdm.write("Aggregating weather data...")
    wd_processed['county'] = wd_processed['county'].astype(int)
    wd_aggregated = wd_processed.groupby(["county", "forecast_datetime_str"], observed=False)[feature_names].mean()
    tqdm.write(f"Weather data aggregated shape: {wd_aggregated.shape}")
    return wd_aggregated

# --- Prosumer Processing (Original Apply) ---
def process_prosumer_data(prosumers: pd.DataFrame, clients: pd.DataFrame):
    """
    Process prosumer data: ensures datetime, creates date string, merges with client data.
    Time features are added separately.
    """
    prosumers_proc = prosumers.copy()

    # Ensure datetime column exists and is datetime type
    if 'datetime' not in prosumers_proc.columns:
        tqdm.write("Error: 'datetime' column missing in prosumer data.")
        return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(prosumers_proc['datetime']):
        prosumers_proc['datetime'] = pd.to_datetime(prosumers_proc['datetime'], errors='coerce')
        prosumers_proc = prosumers_proc.dropna(subset=['datetime'])

    if prosumers_proc.empty:
        tqdm.write("Warning: Prosumer data empty after initial datetime handling.")
        return pd.DataFrame()

    # Create the string date column needed for client merge
    tqdm.write("Formatting prosumer date string (Original Apply)...")
    prosumers_proc["date"] = prosumers_proc.datetime.progress_apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None)
    prosumers_proc = prosumers_proc.dropna(subset=['date'])

    if prosumers_proc.empty:
        tqdm.write("Warning: Prosumer data empty after date string formatting.")
        return pd.DataFrame()

    # Aggregate consumption and capacity by group
    if 'date' in clients.columns:
        if not pd.api.types.is_string_dtype(clients['date']):
             clients['date'] = pd.to_datetime(clients['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        clients = clients.dropna(subset=['date'])
    else:
        tqdm.write("Warning: 'date' column missing in client data.")
        return pd.DataFrame()
    if clients.empty:
        tqdm.write("Warning: Client data is empty after date processing.")
        return pd.DataFrame()

    tqdm.write("Aggregating client data...")
    cons = clients.groupby(["product_type", "county", "is_business", "date"], observed=False)["eic_count"].sum().reset_index()
    cap = clients.groupby(["product_type", "county", "is_business", "date"], observed=False)["installed_capacity"].sum().reset_index()
    if cons.empty or cap.empty:
        tqdm.write("Warning: Client aggregation resulted in empty dataframes.")
        return pd.DataFrame()

    tqdm.write("Merging prosumer and client data (inner join)...")
    merge_keys = ["product_type", "county", "is_business", "date"]
    if not all(key in prosumers_proc.columns for key in merge_keys): return pd.DataFrame() # Basic check
    if not all(key in cons.columns for key in merge_keys): return pd.DataFrame()
    if not all(key in cap.columns for key in merge_keys): return pd.DataFrame()

    try:
        target_county_dtype = cons['county'].dtype
        if prosumers_proc['county'].dtype != target_county_dtype:
            prosumers_proc['county'] = prosumers_proc['county'].astype(target_county_dtype)
    except Exception as e:
        tqdm.write(f"Warning: Could not match county dtypes before client merge. Error: {e}")

    prosumers_proc = pd.merge(prosumers_proc, cons, on=merge_keys, how="inner")
    if prosumers_proc.empty: return pd.DataFrame()
    prosumers_proc = pd.merge(prosumers_proc, cap, on=merge_keys, how="inner")

    tqdm.write(f"Processed prosumer data shape: {prosumers_proc.shape}")
    # Return dataframe with 'datetime' as datetime object and 'date' as string
    return prosumers_proc

# --- Price Processing Functions (Keep as is, ensuring string date formats) ---
def process_elec_price_data(data: pd.DataFrame):
    """Process electricity price data."""
    data_processed = data.rename(columns={"euros_per_mwh": "electricity_price"})
    if 'forecast_date' in data_processed.columns:
        if not pd.api.types.is_string_dtype(data_processed['forecast_date']):
            data_processed['forecast_date'] = pd.to_datetime(data_processed['forecast_date'], errors='coerce')
            data_processed = data_processed.dropna(subset=['forecast_date'])
            data_processed['forecast_date'] = data_processed['forecast_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        tqdm.write("Warning: 'forecast_date' missing in electricity price data.")
    return data_processed

def process_gas_price_data(data: pd.DataFrame):
    """Process gas price data."""
    data_processed = data.copy()
    tqdm.write("Calculating average gas price (Original Apply)...")
    data_processed["avg_gas_price"] = data.progress_apply(
        lambda row: np.mean([row.lowest_price_per_mwh, row.highest_price_per_mwh])
                    if pd.notna(row.lowest_price_per_mwh) and pd.notna(row.highest_price_per_mwh) else np.nan,
        axis=1)
    if 'forecast_date' in data_processed.columns:
         if not pd.api.types.is_string_dtype(data_processed['forecast_date']):
             data_processed['forecast_date'] = pd.to_datetime(data_processed['forecast_date'], errors='coerce').dt.strftime('%Y-%m-%d')
         data_processed = data_processed.dropna(subset=['forecast_date'])
    else:
         tqdm.write("Warning: 'forecast_date' missing in gas price data.")
    return data_processed

# --- Refactored make_merged_dataset (No cache arg needed) ---
def make_merged_dataset(prosumer_data: pd.DataFrame,
                        weather_data: pd.DataFrame,
                        electricity_data: pd.DataFrame,
                        gas_data: pd.DataFrame):
    """
    Merges the processed dataframes based on the original logic (string keys).
    Assumes input dataframes are already processed.
    """
    tqdm.write("Merging processed datasets...")

    # --- Check for empty dataframes ---
    if prosumer_data.empty or weather_data.empty:
        tqdm.write("Warning: Prosumer or Weather data is empty before merging. Returning empty DataFrame.")
        return pd.DataFrame()

    tqdm.write(f"Shape before merging prosumer/weather: Prosumer={prosumer_data.shape}, Weather={weather_data.shape}")

    # --- Merge prosumer and weather data ---
    # Create the string datetime key from prosumer's datetime column
    if 'datetime' in prosumer_data.columns and pd.api.types.is_datetime64_any_dtype(prosumer_data['datetime']):
        prosumer_data['datetime_str_key'] = prosumer_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        left_dt_key = 'datetime_str_key'
    else:
        tqdm.write("Error: 'datetime' column missing or not datetime type in prosumer_data.")
        return pd.DataFrame()

    # Ensure county types match
    if 'county' in weather_data.index.names and 'county' in prosumer_data.columns:
        target_weather_county_dtype = weather_data.index.get_level_values('county').dtype
        if prosumer_data['county'].dtype != target_weather_county_dtype:
            try: prosumer_data['county'] = prosumer_data['county'].astype(target_weather_county_dtype)
            except Exception as e: tqdm.write(f"Warning: Could not match county dtypes. Error: {e}")
    else:
        tqdm.write("Warning: County columns missing or mismatched for weather merge.")
        # Don't necessarily return empty, merge might still work if county isn't the only key issue
        # Let the merge attempt happen

    # Merge (weather_data index is assumed to be [county, forecast_datetime_str])
    data = pd.merge(prosumer_data, weather_data,
                    left_on=["county", left_dt_key], right_index=True, how="inner")
    tqdm.write(f"Shape after merging prosumer/weather: {data.shape}")
    if data.empty:
        tqdm.write("Warning: Data empty after prosumer/weather merge.")
        return pd.DataFrame() # Return empty if inner join failed

    # --- Merge electricity price data ---
    if not electricity_data.empty and 'forecast_date' in electricity_data.columns:
        tqdm.write("Merging electricity price data...")
        data = pd.merge(data, electricity_data[["forecast_date", "electricity_price"]],
                        left_on=left_dt_key, right_on="forecast_date", how="left")
        if 'forecast_date_y' in data.columns: data = data.drop(columns=['forecast_date_y'])
        if 'forecast_date_x' in data.columns: data = data.rename(columns={'forecast_date_x': 'forecast_date'})
    else:
        tqdm.write("Skipping electricity price merge (data missing or no date).")
        data['electricity_price'] = np.nan
    tqdm.write(f"Shape after merging electricity prices: {data.shape}")

    # --- Merge gas price data ---
    if not gas_data.empty and 'forecast_date' in gas_data.columns:
        tqdm.write("Merging gas price data...")
        if 'date' in data.columns: # 'date' is YYYY-MM-DD string
             data = pd.merge(data, gas_data[["forecast_date", "avg_gas_price"]],
                             left_on="date", right_on="forecast_date", how="left")
             if 'forecast_date_y' in data.columns: data = data.drop(columns=['forecast_date_y'])
             if 'forecast_date_x' in data.columns: data = data.rename(columns={'forecast_date_x': 'forecast_date'})
        else:
            tqdm.write("Warning: 'date' column missing before gas price merge.")
            data['avg_gas_price'] = np.nan
    else:
        tqdm.write("Skipping gas price merge (data missing or no date).")
        data['avg_gas_price'] = np.nan
    tqdm.write(f"Shape after merging gas prices: {data.shape}")

    # Clean up temporary key
    if left_dt_key in data.columns:
        data = data.drop(columns=[left_dt_key])

    # Keep original datetime column for potential sorting/feature engineering later
    # Keep grouping columns as they might be needed for lag/roll features

    data = data.reset_index(drop=True) # Reset index after merges
    tqdm.write("Finished merging datasets.")
    return data
