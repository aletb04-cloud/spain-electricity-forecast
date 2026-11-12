import time
import pandas as pd
import requests

from src.config import (
    DEMAND_RAW_PATH,
    END_DATE,
    OPEN_METEO_ARCHIVE_URL,
    REDATA_URL,
    START_DATE,
    WEATHER_NODES,
    WEATHER_RAW_PATH,
    WEATHER_VARS,
)


def fetch_weather_data() -> pd.DataFrame:
    """Fetches weather data from the Open-Meteo API for the specified weather nodes."""
    city_dfs = []

    for city, info in WEATHER_NODES.items():
        print(f"Downloading weather data for {city.capitalize()}...")
        params = {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
        }

        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses

        data = response.json()
        df_city = pd.DataFrame(data["hourly"])

        # Add city and weight columns to the DataFrame
        df_city["city"] = city
        df_city["weight"] = info["weight"]

        city_dfs.append(df_city)
        time.sleep(0.5)  # Pause to avoid hitting API rate limits

    # Concatenate all city DataFrames into a single DataFrame
    df_weather = pd.concat(city_dfs, ignore_index=True)

    # Save the weather data to a Parquet file
    WEATHER_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_weather.to_parquet(WEATHER_RAW_PATH, index=False)
    print(f"Weather data saved to {WEATHER_RAW_PATH} ({len(df_weather)} rows).")

    return df_weather

def fetch_demand_data() -> pd.DataFrame:
    """Download hourly peninsular electricity demand from REData month by month."""
    monthly_dfs = []
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}

    # Generate start dates for every month in the target range
    month_starts = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")

    for m_start in month_starts:
        m_end = m_start + pd.offsets.MonthEnd(1)
        start_str = f"{m_start.strftime('%Y-%m-%d')}T00:00"
        end_str = f"{m_end.strftime('%Y-%m-%d')}T23:59"

        print(f"Downloading demand data: {start_str[:7]}...")

        params = {
            "start_date": start_str,
            "end_date": end_str,
            "time_trunc": "hour",
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741",
        }

        # Retry loop (up to 3 attempts) to handle transient API errors
        for attempt in range(1, 4):
            try:
                response = requests.get(
                    REDATA_URL, params=params, headers=headers, timeout=30
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"  Warning: attempt {attempt}/3 failed ({e}). Retrying in 2s...")
                if attempt == 3:
                    raise
                time.sleep(2)

        data = response.json()
        values = data["included"][0]["attributes"]["values"]

        # Convert the monthly list of dicts into a DataFrame
        df_month = pd.DataFrame(values)
        monthly_dfs.append(df_month)

        time.sleep(0.4)

    # Concatenate all monthly DataFrames into a single table
    df_demand = pd.concat(monthly_dfs, ignore_index=True)

    # Save raw demand data as Parquet
    DEMAND_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_demand.to_parquet(DEMAND_RAW_PATH, index=False)
    print(f"Demand data saved to {DEMAND_RAW_PATH} ({len(df_demand)} rows).\n")

    return df_demand

if __name__ == "__main__":
    fetch_weather_data()
    fetch_demand_data()