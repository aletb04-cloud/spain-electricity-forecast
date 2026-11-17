# Spain Electricity Demand Forecasting (24h-Ahead End-to-End ML System)

Production-grade Machine Learning and Data Engineering pipeline predicting hourly electricity demand (**MWh**) for Peninsular Spain (`REE geo_id=8741`) over a **24-hour forecast horizon**, served via a **FastAPI** REST service and containerized with **Docker**.

---

## 1. System Architecture & Pipeline

```text
┌───────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐
│  REData API (REE) │────▶│   src/ingestion.py       │     │                         │
└───────────────────┘     │   src/processing.py      │────▶│  data/processed/        │
┌───────────────────┐     │  (UTC alignment &        │     │  dataset_clean.parquet  │
│ Open-Meteo ERA5   │────▶│   blackout imputation)   │     │                         │
└───────────────────┘     └──────────────────────────┘     └────────────┬────────────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────▼────────────┐
│  src/features.py  ──▶  Leak-free features (Europe/Madrid calendar, Spanish         │
│                        holidays, HDH/CDH degree-hours, lags >= 24h, rolling stats) │
└───────────────────────────────────────────────────────────────────────┬────────────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────▼────────────┐
│  src/train.py     ──▶  Chronological Train/Val/Test split + LightGBM Regressor     │
│                        Artifact: models/lgbm_demand_24h.joblib                     │
└───────────────────────────────────────────────────────────────────────┬────────────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────▼────────────┐
│  app/main.py (FastAPI) + src/predict.py  ──▶  Live 24h REST API (/predict/next-24h)│
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Engineering Decisions
* **Strict Anti-Leakage Design:** To mirror real-time 24h-ahead operations, all autoregressive demand features (`demand_lag_24h`, `demand_lag_48h`, `demand_lag_168h`, weekly momentum `demand_diff_24h_168h`, and shifted rolling statistics) strictly rely on observations at or before $t - 24\text{h}$.
* **Domain-Driven Feature Engineering (`27` predictors):**
  * **Local-Time Human Activity:** Converts UTC timestamps to `Europe/Madrid` (handling CET/CEST transitions) and integrates Spanish national holidays via `holidays`, alongside cyclical sine/cosine encodings.
  * **Non-Linear Thermal Load:** Captures the U-shaped temperature-demand relationship around the $16.5^\circ\text{C}$ comfort threshold using **Heating/Cooling Degree Hours (`HDH`/`CDH`)** and 24h rolling thermal inertia across 5 consumption-weighted Spanish cities: Madrid (`35%`), Barcelona (`28%`), Valencia (`15%`), Sevilla (`14%`), and Bilbao (`8%`).
* **Data Quality & Anomaly Handling:** Automatically detects and imputes the **April 28, 2025 Iberian Peninsula blackout** (13 hours dropping down to `518 MWh`) using weekly seasonal persistence (`shift(168)`), preventing distortion of rolling statistics.
* **Zero Training-Serving Skew:** `src/predict.py` reuses the exact pure transformation functions from `src/features.py` when stitching live REData telemetry with Open-Meteo `/v1/forecast` weather predictions.

---

## 2. Out-of-Time Evaluation (`July 1 – October 31, 2026`)

Trained chronologically on `2022-01-08` to `2025-12-31` (`34,873` rows), tuned with early stopping on `Jan 1 – Jun 30, 2026` (`4,344` rows; best iteration `#1407`, validation `MAE = 590.34 MWh`), and evaluated on an untouched 4-month out-of-time test window (`2,951` hourly observations):

| Model | MAE (MWh) | RMSE (MWh) | MAPE (%) | Error Reduction vs. Weekly Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence (`Lag 24h`)** | 1,810.63 | 2,663.93 | 6.415% | — |
| **Weekly Persistence (`Lag 168h`)** | 1,405.32 | 1,955.82 | 4.944% | Baseline |
| **XGBoost Regressor** | 554.56 | 733.30 | 1.897% | -60.5% |
| **LightGBM Regressor (Production)** | **537.66** | **714.69** | **1.837%** | **-61.7%** |

---

## 3. Repository Structure

```text
spain-electricity-forecast/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI endpoints (/health, /model/info, /predict/next-24h)
│   └── schemas.py               # Pydantic v2 response validation contracts
├── models/
│   ├── lgbm_demand_24h.joblib   # Serialized LightGBM model & feature schema
│   └── metrics.json             # Official validation & test evaluation report
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis & blackout audit
│   └── 02_modeling.ipynb        # Residual diagnostics, feature gain & XGBoost benchmark
├── src/
│   ├── config.py                # Centralized paths, API endpoints & hyperparameters
│   ├── ingestion.py             # Historical REData & Open-Meteo ERA5 ingestion
│   ├── processing.py            # UTC alignment, weighted weather & anomaly cleaning
│   ├── features.py              # Leak-free feature engineering pipeline
│   ├── train.py                 # Chronological splitting, training & model persistence
│   └── predict.py               # Real-time inference connecting live APIs
├── tests/
│   ├── test_api.py              # FastAPI endpoint & contract integration tests
│   └── test_features.py         # Anti-leakage & thermal calculation unit tests
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 4. Quickstart

### Local Environment Setup
```bash
# 1. Create environment and install dependencies
conda create -n electricity python=3.11 -y
conda activate electricity
pip install -r requirements.txt
pip install -e . --no-deps

# 2. Run automated unit & integration tests
pytest -v

# 3. Start the REST API server
uvicorn app.main:app --reload
```
Interactive Swagger UI documentation is available at **`http://127.0.0.1:8000/docs`**.

### Docker Deployment
```bash
# 1. Build production container image
docker build -t spain-electricity-api:1.0 .

# 2. Run container on port 8000
docker run -d --name electricity-server -p 8000:8000 spain-electricity-api:1.0

# 3. Stop and remove container when finished
docker stop electricity-server && docker rm electricity-server
```

---

## 5. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check and model artifact availability |
| `GET` | `/model/info` | Active model architecture, feature count, and out-of-time test metrics |
| `GET` | `/predict/next-24h` | Live 24-hour hourly electricity demand forecast (`UTC` & `Europe/Madrid`) |

---

## 6. Future Improvements (v2 Roadmap)
* **Post-Holiday Transition Feature:** Residual diagnostics revealed elevated errors on Tuesdays immediately following Monday national holidays (e.g., October 13, 2026 after October 12), where `demand_lag_24h` reflects holiday consumption. Adding `was_prev_day_holiday = is_holiday.shift(24)` will explicitly signal return-to-work rebounds.
* **Regional Holiday Weighting:** Weighting regional holidays across autonomous communities (Madrid, Catalonia, Andalusia, Valencia, Basque Country) proportionally to node consumption weights.
