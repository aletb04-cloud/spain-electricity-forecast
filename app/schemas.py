"""Pydantic schemas for API request/response validation and OpenAPI documentation."""

from pydantic import BaseModel, Field

# Check if the service is running
class HealthResponse(BaseModel):
    """Schema for the service health-check endpoint."""
    status: str = Field(..., examples=['ok'])
    model_loaded: bool = Field(..., examples=[True])

# Model metadata
class ModelInfoResponse(BaseModel):
    """Schema exposing trained model metadata and evaluation metrics."""
    model_type: str = Field(..., examples=['LightGBM'])   
    forecast_horizon_hours: int = Field(..., examples=[24])
    num_features: int = Field(..., examples=[27])
    best_iteration: int = Field(..., examples=[1407])
    test_mae_mwh: float = Field(..., examples=[537.66])
    test_rmse_mwh: float = Field(..., examples=[714.69])
    test_mape_pct: float = Field(..., examples=[1.837]) 

# Forecast request and response
class HourlyPrediction(BaseModel):
    """Schema for a single hourly electricity demand prediction."""
    timestamp_utc: str = Field(..., examples=["2026-11-17T17:00:00+00:00"])
    timestamp_local: str = Field(..., examples=["2026-11-17T18:00:00+01:00"])
    predicted_demand_mwh: float = Field(..., gt=0, examples=[32051.24])
    temperature_2m_c: float = Field(..., examples=[18.69])
    is_holiday: int = Field(..., ge=0, le=1, examples=[0])

class ForecastResponse(BaseModel):
    """Schema for the complete 24-hour forecast response."""
    region: str = Field(default="Peninsular Spain (REE geo_id=8741)")
    horizon_hours: int = Field(..., examples=[24])
    predictions: list[HourlyPrediction]