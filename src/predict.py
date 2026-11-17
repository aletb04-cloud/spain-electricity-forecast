"""Real-time 24h-ahead inference pipeline using live REData & Open-Meteo APIs."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import (
    FORECAST_HORIZON,
    INFERENCE_HISTORY_DAYS,
    LOCAL_TZ,
    MODEL_OUTPUT_PATH,
    OPEN_METEO_FORECAST_URL,
    REDATA_URL,
    WEATHER_NODES,
    WEATHER_VARS,
)
from src.features import (
    add_calendar_features,
    add_demand_lags_and_rolling,
    add_weather_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_recent_demand(days: int = INFERENCE_HISTORY_DAYS) -> pd.DataFrame:
    """Fetch recent hourly electricity demand from REData API for lag computation."""
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = now_utc - timedelta(days=days)

    params = {
        "start_date": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "end_date": now_utc.strftime("%Y-%m-%dT%H:%M"),
        "time_trunc": "hour",
        "geo_trunc": "electric_system",
        "geo_limit": "peninsular",
        "geo_ids": "8741",
    }
    headers = {"Accept": "application/json"}

    logger.info(
        "Fetching recent demand from REData (%s to %s)...",
        params["start_date"],
        params["end_date"],
    )
    response = requests.get(REDATA_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()

    # Extract Peninsular Real Demand series ("Demanda real")
    included = payload.get("included", [])
    real_demand_series = next(
        (
            item
            for item in included
            if "real" in item.get("type", "").lower()
            or "real" in item.get("attributes", {}).get("title", "").lower()
        ),
        included[0],
    )
    values = real_demand_series["attributes"]["values"]

    df = pd.DataFrame(values)[["datetime", "value"]].rename(
        columns={"datetime": "timestamp_utc", "value": "demand_mwh"}
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")

    # Guarantee an unbroken hourly grid (fill minor API reporting delays if any)
    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    full_idx = pd.date_range(
        df["timestamp_utc"].min(), df["timestamp_utc"].max(), freq="h", tz="UTC"
    )
    df = (
        df.set_index("timestamp_utc")
        .reindex(full_idx)
        .rename_axis("timestamp_utc")
        .reset_index()
    )
    df["demand_mwh"] = df["demand_mwh"].ffill().bfill()

    return df


def fetch_weather_forecast() -> pd.DataFrame:
    """Fetch weighted hourly weather (past 48h + next 72h) from Open-Meteo Forecast API."""
    logger.info(
        "Fetching weighted weather forecast across %d Spanish nodes...",
        len(WEATHER_NODES),
    )
    weighted_dfs: list[pd.DataFrame] = []

    for city, meta in WEATHER_NODES.items():
        params = {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
            "past_days": 2,
            "forecast_days": 3,
        }
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()
        hourly = response.json()["hourly"]

        city_df = pd.DataFrame(
            {"timestamp_utc": pd.to_datetime(hourly["time"], utc=True)}
        )
        for var in WEATHER_VARS:
            city_df[var] = pd.to_numeric(hourly[var], errors="coerce") * meta["weight"]

        weighted_dfs.append(city_df.set_index("timestamp_utc"))

    combined_weather = sum(weighted_dfs).reset_index()
    return combined_weather.sort_values("timestamp_utc").reset_index(drop=True)


def generate_24h_forecast() -> list[dict[str, Any]]:
    """Build inference features and predict the next 24 hours of electricity demand."""
    if not MODEL_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_OUTPUT_PATH}. Run 'python -m src.train' first."
        )

    # 1. Load trained model and expected feature order
    artifact = joblib.load(MODEL_OUTPUT_PATH)
    model = artifact["model"]
    feature_cols: list[str] = artifact["feature_cols"]

    # 2. Fetch live recent demand and weather forecast
    demand_df = fetch_recent_demand()
    weather_df = fetch_weather_forecast()

    # 3. Define the next 24 hours right after the latest known demand timestamp
    last_known_ts = demand_df["timestamp_utc"].max()
    forecast_horizon_ts = pd.date_range(
        start=last_known_ts + pd.Timedelta(hours=1),
        periods=FORECAST_HORIZON,
        freq="h",
        tz="UTC",
    )

    # 4. Stitch historical demand + empty future rows, then merge weather
    future_df = pd.DataFrame(
        {"timestamp_utc": forecast_horizon_ts, "demand_mwh": np.nan}
    )
    full_df = pd.concat([demand_df, future_df], ignore_index=True)

    full_df = full_df.merge(weather_df, on="timestamp_utc", how="left")
    for var in WEATHER_VARS:
        full_df[var] = full_df[var].ffill().bfill()

    # 5. Reuse the exact feature engineering pipeline from training
    full_df = add_calendar_features(full_df)
    full_df = add_weather_features(full_df)
    full_df = add_demand_lags_and_rolling(full_df)

    # 6. Isolate the 24 target future hours and run model inference
    target_df = full_df[full_df["timestamp_utc"].isin(forecast_horizon_ts)].copy()
    if len(target_df) != FORECAST_HORIZON:
        raise RuntimeError(
            f"Expected {FORECAST_HORIZON} forecast rows, got {len(target_df)}."
        )

    if target_df[feature_cols].isna().any().any():
        missing_cols = (
            target_df[feature_cols].columns[target_df[feature_cols].isna().any()].tolist()
        )
        raise ValueError(f"NaN values detected in inference features: {missing_cols}")

    preds = model.predict(target_df[feature_cols])
    target_df["predicted_demand_mwh"] = np.round(preds, 2)
    target_df["timestamp_local"] = target_df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)

    results = [
        {
            "timestamp_utc": row["timestamp_utc"].isoformat(),
            "timestamp_local": row["timestamp_local"].isoformat(),
            "predicted_demand_mwh": float(row["predicted_demand_mwh"]),
            "temperature_2m_c": round(float(row["temperature_2m"]), 2),
            "is_holiday": int(row["is_holiday"]),
        }
        for _, row in target_df.iterrows()
    ]

    logger.info(
        "Generated %d hourly predictions starting at %s (First hour: %.2f MWh)",
        len(results),
        results[0]["timestamp_local"],
        results[0]["predicted_demand_mwh"],
    )
    return results


if __name__ == "__main__":
    forecast = generate_24h_forecast()
    print(pd.DataFrame(forecast).to_string(index=False))
