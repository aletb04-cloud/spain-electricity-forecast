"""FastAPI application serving 24h-ahead Peninsular Spain electricity demand forecasts."""

import json
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import ForecastResponse, HealthResponse, ModelInfoResponse
from src.config import FORECAST_HORIZON, METRICS_OUTPUT_PATH, MODEL_OUTPUT_PATH
from src.predict import generate_24h_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spain Electricity Demand Forecast API",
    description=(
        "Production REST API predicting 24-hour-ahead hourly electricity demand (MWh) "
        "for Peninsular Spain using LightGBM, live REData telemetry, and Open-Meteo forecasts."
    ),
    version="1.0.0",
)

@app.get("/predict/next-24h", response_model=ForecastResponse, tags=["Forecasting"])
def predict_next_24h() -> ForecastResponse:
    """Generate a live 24-hour hourly electricity demand forecast for Peninsular Spain."""
    try:
        predictions = generate_24h_forecast()
        return ForecastResponse(
            horizon_hours=len(predictions),
            predictions=predictions,
        )
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during live forecast generation")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate 24h electricity demand forecast: {exc}",
        ) from exc

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Verify that the API service is running and the model artifact exists on disk."""
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_OUTPUT_PATH.exists(),
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info() -> ModelInfoResponse:
    """Return metadata and out-of-time test evaluation metrics for the active model."""
    if not METRICS_OUTPUT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics report not found. Run 'python -m src.train' first.",
        )

    with open(METRICS_OUTPUT_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    test_metrics = metrics["test_lgbm"]
    return ModelInfoResponse(
        model_type="LightGBM Regressor",
        forecast_horizon_hours=FORECAST_HORIZON,
        num_features=metrics["num_features"],
        best_iteration=metrics["best_iteration"],
        test_mae_mwh=test_metrics["mae_mwh"],
        test_rmse_mwh=test_metrics["rmse_mwh"],
        test_mape_pct=test_metrics["mape_pct"],
    )

