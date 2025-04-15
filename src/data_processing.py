import numpy as np 
import pandas as pd 
from geopy import distance

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

def find_closest_weather_station(point: pd.Series, wss: pd.DataFrame):
    """
    Find closest weather station with county information to a given point.
    """
    point_coordinates = [point.latitude, point.longitude]
    weather_station_coordinates = [[ws.latitude, ws.longitude] for ws in wss.itertuples()]
    
    # calculate distances to every weather station
    dists = [distance.distance(point_coordinates, station).km for station in weather_station_coordinates]
    closest_dist = np.min(dists)
    closest_station = wss.iloc[np.argmin(dists), :]
    return closest_station, closest_dist

def process_weather_data(wd: pd.DataFrame, wss: pd.DataFrame, feature_names: list):
    """
    Process weather data by adding weather station info to data
    and aggregating by county and timestamp.
    """
    county_names = []
    county_ids = []
    
    for row in wd.itertuples():
        closest_station, dist = find_closest_weather_station(row, wss)
        if dist < 30:  # within 30 km of a station
            county_names.append(closest_station.county_name)
            county_ids.append(closest_station.county)
        else:
            county_names.append(None)
            county_ids.append(None)
    
    wd_processed = wd.copy()
    wd_processed["county_name"] = county_names
    wd_processed["county"] = county_ids

    # Keep only data points with available county info
    wd_processed = wd_processed.dropna(subset=["county"], axis=0)
    
    # Format timestamps
    wd_processed["forecast_datetime"] = wd_processed.forecast_datetime.apply(
        lambda x: pd.to_datetime(x).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"))
    
    # Aggregate data per county and timestamp
    wd_processed = wd_processed.groupby(["county", "forecast_datetime"])[feature_names].mean()
    return wd_processed

def process_prosumer_data(prosumers: pd.DataFrame, clients: pd.DataFrame):
    """
    Process prosumer data by merging it with client data (capacity and consumption info).
    """
    prosumers_proc = prosumers.copy()
    # Add feature for date
    prosumers_proc["date"] = prosumers_proc.datetime.apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
    
    # Aggregate consumption and capacity by group
    cons = clients.groupby(["product_type", "county", "is_business", "date"])["eic_count"].sum().reset_index()
    cap = clients.groupby(["product_type", "county", "is_business", "date"])["installed_capacity"].sum().reset_index()
    
    # Merge the aggregated values with the prosumer data
    prosumers_proc = pd.merge(prosumers_proc, cons, on=["product_type", "county", "is_business", "date"])
    prosumers_proc = pd.merge(prosumers_proc, cap, on=["product_type", "county", "is_business", "date"])
    return prosumers_proc

def process_elec_price_data(data: pd.DataFrame):
    """
    Process electricity price data: just a simple column rename here.
    """
    data_processed = data.rename(columns={"euros_per_mwh": "electricity_price"})
    return data_processed

def process_gas_price_data(data: pd.DataFrame):
    """
    Process gas price data: calculate the average gas price.
    """
    data_processed = data.copy()
    data_processed["avg_gas_price"] = data.apply(
        lambda row: np.mean([row.lowest_price_per_mwh, row.highest_price_per_mwh]), axis=1)
    return data_processed

def make_dataset(prosumer: pd.DataFrame, 
                 weather_forecast_data: pd.DataFrame, 
                 weather_stations: pd.DataFrame,
                 client: pd.DataFrame,
                 electricity_prices: pd.DataFrame,
                 gas_prices: pd.DataFrame,
                 weather_feature_names: list,
                 scaler
                 ):
    """
    Create a combined dataset for prediction by merging processed data.
    """
    from sklearn.preprocessing import StandardScaler  # if not already imported globally

    weather_data = process_weather_data(weather_forecast_data, weather_stations, weather_feature_names)
    prosumer_data = process_prosumer_data(prosumer, client)
    electricity_data = process_elec_price_data(electricity_prices)
    gas_data = process_gas_price_data(gas_prices)
    
    # Merge prosumer and weather data
    data = pd.merge(prosumer_data, 
                    weather_data, 
                    left_on=["county", "datetime"], 
                    right_index=True,
                    how="inner")
    
    # Add price data
    data = pd.merge(
                pd.merge(data, 
                         electricity_data[["forecast_date", "electricity_price"]], 
                         left_on="datetime", 
                         right_on="forecast_date", 
                         how="left"), 
                gas_data[["forecast_date", "avg_gas_price"]], 
                left_on="date", 
                right_on="forecast_date", 
                how="left")
    
    feats = ["is_consumption", "eic_count", "installed_capacity", "electricity_price", "avg_gas_price"] \
            + weather_feature_names + ["target"]
    data = data[feats].reset_index(drop=True)
    
    if scaler is not None:
        target = data.target
        data_no_target = data.drop("target", axis=1)
        data_scaled = scaler.fit_transform(data_no_target)
        data = pd.DataFrame(data_scaled, columns=data_no_target.columns)
        data["target"] = target
    
    return data
