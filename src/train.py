"""Training and evaluation pipeline for 24h-ahead electricity demand forecasting."""

import json
import logging
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from src.config import (
    FEATURES_DATA_PATH,
    METRICS_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    MODELS_DIR,
    NON_FEATURE_COLS,
    TARGET_COL,
    TEST_START_DATE,
    VAL_START_DATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    # Calculate standard regression metrics for electricity load forecasting.
    return {
        "mae_mwh": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse_mwh": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 3),
    }


def split_temporal_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Split dataset chronologically into Train, Validation, and Out-of-Time Test sets.

    val_cutoff = pd.Timestamp(VAL_START_DATE, tz="UTC")
    test_cutoff = pd.Timestamp(TEST_START_DATE, tz="UTC")

    train_df = df[df["timestamp_utc"] < val_cutoff].copy()
    val_df = df[
        (df["timestamp_utc"] >= val_cutoff) & (df["timestamp_utc"] < test_cutoff)
    ].copy()
    test_df = df[df["timestamp_utc"] >= test_cutoff].copy()

    logger.info(
        "Temporal Split -> Train: %d rows | Val: %d rows | Test: %d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMRegressor:
    
    # Train a LightGBM regressor with early stopping on the validation set.

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    logger.info("Best iteration reached at tree #%d", model.best_iteration_)
    return model


def run_training_pipeline() -> dict[str, Any]:
    # Execute full model training, baseline comparison, and artifact persistence.
    
    logger.info("Loading features dataset from %s", FEATURES_DATA_PATH)
    df = pd.read_parquet(FEATURES_DATA_PATH)

    train_df, val_df, test_df = split_temporal_data(df)

    feature_cols = [col for col in df.columns if col not in NON_FEATURE_COLS]
    logger.info("Training with %d features: %s", len(feature_cols), feature_cols)

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    # 1. Evaluate Naive Baselines on Test Set
    baseline_24h_metrics = compute_metrics(y_test, test_df["demand_lag_24h"])
    baseline_168h_metrics = compute_metrics(y_test, test_df["demand_lag_168h"])

    logger.info("Test Baseline (Lag 24h)  -> %s", baseline_24h_metrics)
    logger.info("Test Baseline (Lag 168h) -> %s", baseline_168h_metrics)

    # 2. Train LightGBM Model
    logger.info("Fitting LightGBM model...")
    model = train_lgbm_model(X_train, y_train, X_val, y_val)

    # 3. Evaluate LightGBM on Validation and Test Sets
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_metrics = compute_metrics(y_val, val_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    logger.info("Validation LightGBM -> %s", val_metrics)
    logger.info("Test LightGBM       -> %s", test_metrics)

    # 4. Feature Importance ranking (by Gain)
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    logger.info("Top 10 Most Important Features (by Gain):\n%s", importance_df.head(10).to_string(index=False))

    # 5. Save Model Artifact & Metrics Report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
    }
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    logger.info("Model artifact saved to %s", MODEL_OUTPUT_PATH)

    report = {
        "best_iteration": int(model.best_iteration_),
        "num_features": len(feature_cols),
        "test_baseline_lag_24h": baseline_24h_metrics,
        "test_baseline_lag_168h": baseline_168h_metrics,
        "val_lgbm": val_metrics,
        "test_lgbm": test_metrics,
    }
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Metrics report saved to %s", METRICS_OUTPUT_PATH)

    return report


if __name__ == "__main__":
    run_training_pipeline()
