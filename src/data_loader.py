import pandas as pd
from . import config # Use relative import within the package

def load_prosumer_data() -> pd.DataFrame:
    """Loads the prosumer data."""
    print("Loading prosumer data...")
    # Consider adding optimized dtypes here if needed later
    return pd.read_csv(config.PROSUMER_DATA_PATH)

def load_weather_forecast_data() -> pd.DataFrame:
    """Loads the weather forecast data."""
    print("Loading weather forecast data...")
    return pd.read_csv(config.WEATHER_FORECAST_PATH)

def load_weather_stations_data() -> pd.DataFrame:
    """Loads the weather stations data."""
    print("Loading weather stations data...")
    return pd.read_csv(config.WEATHER_STATIONS_PATH)

def load_client_data() -> pd.DataFrame:
    """Loads the client data."""
    print("Loading client data...")
    return pd.read_csv(config.CLIENT_DATA_PATH)

def load_electricity_prices_data() -> pd.DataFrame:
    """Loads the electricity prices data."""
    print("Loading electricity prices data...")
    return pd.read_csv(config.ELECTRICITY_PRICES_PATH)

def load_gas_prices_data() -> pd.DataFrame:
    """Loads the gas prices data."""
    print("Loading gas prices data...")
    return pd.read_csv(config.GAS_PRICES_PATH)

def load_all_data() -> dict[str, pd.DataFrame]:
    """Loads all necessary data files into a dictionary."""
    print("Loading all data sources...")
    data = {
        "prosumer": load_prosumer_data(),
        "weather_forecast": load_weather_forecast_data(),
        "weather_stations": load_weather_stations_data(),
        "client": load_client_data(),
        "electricity_prices": load_electricity_prices_data(),
        "gas_prices": load_gas_prices_data(),
    }
    print("All data loaded.")
    return data

