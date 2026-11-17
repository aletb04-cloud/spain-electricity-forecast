"""Integration tests for FastAPI endpoints and Pydantic response contracts."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify /health returns HTTP 200 and confirms the model artifact is loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_model_info_endpoint() -> None:
    """Verify /model/info exposes valid evaluation metrics and 24h horizon."""
    response = client.get("/model/info")
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_type"] == "LightGBM Regressor"
    assert payload["forecast_horizon_hours"] == 24
    assert payload["test_mape_pct"] > 0


@patch("app.main.generate_24h_forecast")
def test_predict_next_24h_endpoint(mock_forecast) -> None:
    """Verify /predict/next-24h validates and returns 24 hourly predictions."""
    mock_forecast.return_value = [
        {
            "timestamp_utc": f"2026-11-18T{hour:02d}:00:00+00:00",
            "timestamp_local": f"2026-11-18T{hour:02d}:00:00+01:00",
            "predicted_demand_mwh": 28500.0 + hour * 100,
            "temperature_2m_c": 18.5,
            "is_holiday": 0,
        }
        for hour in range(24)
    ]

    response = client.get("/predict/next-24h")
    assert response.status_code == 200

    payload = response.json()
    assert payload["horizon_hours"] == 24
    assert len(payload["predictions"]) == 24
    assert payload["predictions"][0]["predicted_demand_mwh"] == 28500.0
