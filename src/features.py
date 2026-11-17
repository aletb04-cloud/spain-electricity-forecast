# Feature engineering pipeline for hourly electricity demand forecasting.

import logging
import holidays
import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    COMFORT_TEMP_C,
    DEMAND_LAGS,
    DEMAND_ROLLING_WINDOWS,
    FEATURES_DATA_PATH,
    FORECAST_HORIZON,
    LOCAL_TZ,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract local-time calendar, holiday, and cyclical features.

    # Converts UTC timestamps to Europe/Madrid local time so daily and weekly
    # human activity patterns align properly across daylight saving transitions.
    
    df = df.copy()

    # Convert UTC to Peninsular Spain local time
    local_dt = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)

    # Raw calendar integers
    df["hour"] = local_dt.dt.hour
    df["day_of_week"] = local_dt.dt.day_of_week
    df["month"] = local_dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Spanish national holidays based on local date
    years = local_dt.dt.year.unique().tolist()
    es_holidays = holidays.country_holidays("ES", years=years)
    local_dates = local_dt.dt.date
    df["is_holiday"] = local_dates.isin(es_holidays).astype(int)

    # Cyclical encoding (sine/cosine) to preserve continuity at boundaries
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create non-linear thermal degree hours and thermal inertia features.

    # Addresses the U-shaped temperature-demand relationship around the comfort
    # threshold (16.5 C) and building heat retention over the past 24 hours.

    df = df.copy()

    # Heating and Cooling Degree Hours (piecewise linear representation of the U-curve)
    df["hdh"] = (COMFORT_TEMP_C - df["temperature_2m"]).clip(lower=0.0)
    df["cdh"] = (df["temperature_2m"] - COMFORT_TEMP_C).clip(lower=0.0)

    # Thermal inertia (lagged and rolling temperature)
    df["temp_lag_24h"] = df["temperature_2m"].shift(24)
    df["temp_diff_24h"] = df["temperature_2m"] - df["temp_lag_24h"]
    df["temp_roll_mean_24h"] = df["temperature_2m"].rolling(window=24).mean()

    return df


def add_demand_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    # Generate autoregressive demand lags and rolling stats without data leakage.

    # All features strictly use demand observations at or before (t - FORECAST_HORIZON).
    
    df = df.copy()

    # 1. Direct autoregressive lags (>= 24h)
    for lag in DEMAND_LAGS:
        if lag < FORECAST_HORIZON:
            raise ValueError(
                f"Lag {lag}h is shorter than forecast horizon ({FORECAST_HORIZON}h) "
                "and would cause data leakage."
            )
        df[f"demand_lag_{lag}h"] = df["demand_mwh"].shift(lag)

    # 2. Weekly momentum (yesterday vs. same hour one week ago)
    df["demand_diff_24h_168h"] = df["demand_lag_24h"] - df["demand_lag_168h"]

    # 3. Shifted rolling statistics (computed on series already shifted by 24h)
    demand_shifted_24h = df["demand_mwh"].shift(FORECAST_HORIZON)
    for window in DEMAND_ROLLING_WINDOWS:
        df[f"demand_roll_mean_{window}h"] = demand_shifted_24h.rolling(
            window=window
        ).mean()
        df[f"demand_roll_std_{window}h"] = demand_shifted_24h.rolling(
            window=window
        ).std()

    return df


def build_features() -> pd.DataFrame:
    # Execute the full feature engineering pipeline and save to Parquet.
    logger.info("Loading clean dataset from %s", CLEAN_DATA_PATH)
    df = pd.read_parquet(CLEAN_DATA_PATH)

    # Ensure strict chronological ordering before any shift/rolling operation
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    initial_rows = len(df)

    logger.info("Generating calendar, holiday, and cyclical features...")
    df = add_calendar_features(df)

    logger.info("Generating weather degree-hours and thermal inertia features...")
    df = add_weather_features(df)

    logger.info("Generating leak-free demand lags and rolling statistics...")
    df = add_demand_lags_and_rolling(df)

    # Drop initial warmup rows containing NaNs from lookback windows (24 + 168 - 1 = 191h)
    df_clean = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df_clean)

    logger.info(
        "Dropped %d warmup rows due to lags/rolling windows. Final shape: %s",
        dropped_rows,
        df_clean.shape,
    )

    PROCESSED_DATA_DIR_PATH = FEATURES_DATA_PATH.parent
    PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(FEATURES_DATA_PATH, index=False)
    logger.info("Features dataset successfully saved to %s", FEATURES_DATA_PATH)

    return df_clean


if __name__ == "__main__":
    build_features()
