"""Unit tests for feature engineering transformations and anti-leakage guarantees."""

import numpy as np
import pandas as pd

from src.features import (
    add_calendar_features,
    add_demand_lags_and_rolling,
    add_weather_features,
)


def _make_sample_df(rows: int = 250) -> pd.DataFrame:
    """Create a synthetic hourly dataframe for deterministic feature testing."""
    timestamps = pd.date_range("2026-05-01 00:00:00", periods=rows, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "demand_mwh": rng.uniform(22000.0, 34000.0, size=rows),
            "temperature_2m": rng.uniform(8.0, 30.0, size=rows),
            "apparent_temperature": rng.uniform(7.0, 31.0, size=rows),
            "shortwave_radiation": rng.uniform(0.0, 800.0, size=rows),
        }
    )


def test_degree_hours_calculation() -> None:
    """Verify Heating and Cooling Degree Hours around the 16.5 C comfort threshold."""
    df = _make_sample_df(rows=30)
    df.loc[0, "temperature_2m"] = 10.0  # Cold: 6.5 HDH, 0.0 CDH
    df.loc[1, "temperature_2m"] = 25.0  # Hot:  0.0 HDH, 8.5 CDH
    df.loc[2, "temperature_2m"] = 16.5  # Neutral: 0.0 HDH, 0.0 CDH

    result = add_weather_features(df)

    assert np.isclose(result.loc[0, "hdh"], 6.5)
    assert np.isclose(result.loc[0, "cdh"], 0.0)
    assert np.isclose(result.loc[1, "hdh"], 0.0)
    assert np.isclose(result.loc[1, "cdh"], 8.5)
    assert np.isclose(result.loc[2, "hdh"], 0.0)
    assert np.isclose(result.loc[2, "cdh"], 0.0)


def test_no_data_leakage_in_demand_features() -> None:
    """Ensure modifying demand inside the last 24h window does not alter final features."""
    df_original = _make_sample_df(rows=250)
    df_sabotaged = df_original.copy()

    # Corrupt the last 24 hours of demand (t-23 to t) with extreme values
    df_sabotaged.loc[df_sabotaged.index[-24:], "demand_mwh"] = 999999.0

    feats_original = add_demand_lags_and_rolling(add_calendar_features(df_original))
    feats_sabotaged = add_demand_lags_and_rolling(add_calendar_features(df_sabotaged))

    feature_cols = [
        c for c in feats_original.columns if c not in ["timestamp_utc", "demand_mwh"]
    ]

    # For the final target timestamp (index -1), all engineered features must match 100%
    pd.testing.assert_series_equal(
        feats_original.iloc[-1][feature_cols],
        feats_sabotaged.iloc[-1][feature_cols],
    )
