import os

# Get the current working directory and construct data directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Example: path to prosumer data
PROSUMER_DATA_PATH = os.path.join(DATA_DIR, "prosumer.csv")
WEATHER_HISTORY_PATH = os.path.join(DATA_DIR, "weather_history.csv")
WEATHER_FORECAST_PATH = os.path.join(DATA_DIR, "weather_forecast.csv")
GAS_PRICES_PATH = os.path.join(DATA_DIR, "gas_prices.csv")
ELECTRICITY_PRICES_PATH = os.path.join(DATA_DIR, "electricity_prices.csv")
CLIENT_DATA_PATH = os.path.join(DATA_DIR, "client.csv")
COUNTY_MAP_PATH = os.path.join(DATA_DIR, "county_id_to_name_map.json")
WEATHER_STATIONS_PATH = os.path.join(DATA_DIR, "weather_station_to_county_mapping.csv")

# Other constants
DATA_SAMPLE_SIZE = 10000
WEATHER_FEATURE_NAMES = ['direct_solar_radiation','surface_solar_radiation_downwards']
