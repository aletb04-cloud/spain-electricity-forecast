from pathlib import Path

# 1. PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw ingestion output files
DEMAND_RAW_PATH = RAW_DATA_DIR / "demand_raw.parquet"
WEATHER_RAW_PATH = RAW_DATA_DIR / "weather_raw.parquet"

# Processed output files
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "dataset_clean.parquet"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "features.parquet"

# 2. HISTORICAL TIME WINDOW
START_DATE = "2022-01-01"
END_DATE = "2026-10-31"

# 3. API ENDPOINTS & PARAMETERS
REDATA_URL = "https://apidatos.ree.es/en/datos/demanda/evolucion"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
]

# 4. WEATHER NODES (Coordinates & population/consumption weights)
# The weights are based on the population distribution of the cities
WEATHER_NODES = {
    "madrid": {"lat": 40.4168, "lon": -3.7038, "weight": 0.35},
    "barcelona": {"lat": 41.3851, "lon": 2.1734, "weight": 0.28},
    "valencia": {"lat": 39.4699, "lon": -0.3763, "weight": 0.15},
    "sevilla": {"lat": 37.3891, "lon": -5.9845, "weight": 0.14},
    "bilbao": {"lat": 43.2630, "lon": -2.9350, "weight": 0.08},
}

# 5. FEATURE ENGINEERING PARAMETERS
LOCAL_TZ = "Europe/Madrid"
FORECAST_HORIZON = 24
COMFORT_TEMP_C = 16.5
DEMAND_LAGS = [24, 48, 168]
DEMAND_ROLLING_WINDOWS = [24, 168]

# 6. MODELING & SPLITS CONFIGURATION
MODEL_OUTPUT_PATH = MODELS_DIR / "lgbm_demand_24h.joblib"
METRICS_OUTPUT_PATH = MODELS_DIR / "metrics.json"

VAL_START_DATE = "2026-01-01"
TEST_START_DATE = "2026-07-01"

TARGET_COL = "demand_mwh"
NON_FEATURE_COLS = ["timestamp_utc", "demand_mwh"]

# 7. INFERENCE & API CONFIGURATION
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
INFERENCE_HISTORY_DAYS = 10

