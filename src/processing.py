import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    DEMAND_RAW_PATH,
    WEATHER_RAW_PATH,
    WEATHER_VARS,
)

def process_demand(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw demand data, standardize timestamps to UTC, and impute grid outages."""
    df = df_raw.copy()

    # Parse local ISO timestamps (+01:00 / +02:00) and convert to UTC
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.rename(columns={"value": "demand_mwh"})

    # Keep only relevant columns, sort chronologically, and drop duplicates
    df = df[["timestamp_utc", "demand_mwh"]]
    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    df = df.reset_index(drop=True)

    # Impute extreme grid-outage anomalies (< 15,000 MWh, e.g., April 28, 2025 blackout)
    # using the same hour from the previous week (lag 168h) to preserve hourly continuity
    outage_mask = df["demand_mwh"] < 15000
    df.loc[outage_mask, "demand_mwh"] = df["demand_mwh"].shift(168)[outage_mask]
    
    return df


def process_weather(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted peninsular average for meteorological variables in UTC."""
    df = df_raw.copy()

    df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)

    # Multiply each weather variable by its city population/consumption weight
    for var in WEATHER_VARS:
        df[var] = df[var] * df["weight"]

    # Sum weighted values across the 5 cities for each UTC hour
    df_weighted = (
        df.groupby("timestamp_utc", as_index=False)[WEATHER_VARS]
        .sum()
        .sort_values("timestamp_utc")
    )

    return df_weighted.reset_index(drop=True)


def build_clean_dataset() -> pd.DataFrame:
    """Load raw datasets, process them, merge on UTC timestamp, and save to Parquet."""
    print("Loading raw parquet files...")
    df_demand_raw = pd.read_parquet(DEMAND_RAW_PATH)
    df_weather_raw = pd.read_parquet(WEATHER_RAW_PATH)

    print("Processing demand and weather datasets...")
    df_demand = process_demand(df_demand_raw)
    df_weather = process_weather(df_weather_raw)

    print("Merging datasets on UTC timestamp...")
    df_clean = pd.merge(
        df_demand,
        df_weather,
        on="timestamp_utc",
        how="inner",
    )

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(CLEAN_DATA_PATH, index=False)

    print(f"Clean dataset saved to {CLEAN_DATA_PATH}")
    print(f"Final shape: {df_clean.shape}")
    print(f"Date range: {df_clean['timestamp_utc'].min()} -> {df_clean['timestamp_utc'].max()}")

    return df_clean


if __name__ == "__main__":
    build_clean_dataset()