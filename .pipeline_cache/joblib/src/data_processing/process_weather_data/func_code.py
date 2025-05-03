# first line: 35
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
