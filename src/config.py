from pathlib import Path

# 1. PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Raw ingestion output files
DEMAND_RAW_PATH = RAW_DATA_DIR / "demand_raw.parquet"
WEATHER_RAW_PATH = RAW_DATA_DIR / "weather_raw.parquet"

# Raw ingestion output files
DEMAND_RAW_PATH = RAW_DATA_DIR / "demand_raw.parquet"
WEATHER_RAW_PATH = RAW_DATA_DIR / "weather_raw.parquet"

# Processed output file
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "dataset_clean.parquet"

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
