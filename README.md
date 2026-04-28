# Predictive Maintenance MLOps Platform
## Name - Shlok Shetty
## Roll - MM22B003

An end-to-end MLOps platform for industrial equipment failure prediction. The system predicts equipment failure within a 24-hour window using multivariate sensor telemetry, exposing predictions via a REST API and an operator-facing dashboard.

---

## Architecture

```
Raw CSV Data (DVC)
       ↓
Apache Airflow (DAG orchestration)
       ↓
Apache Spark (feature engineering, 876k rows)
       ↓
MLflow (experiment tracking + model registry)
       ↓
FastAPI (inference engine, /predict, /health, /ready, /drift/status)
       ↓
React Frontend (operator dashboard, CSV upload, fleet overview)
       +
Prometheus + Grafana + AlertManager (monitoring, alerting, drift detection)
```

All services run in Docker. One command brings everything up.

---

## Stack

| Layer | Tool |
|---|---|
| Orchestration | Apache Airflow (LocalExecutor) |
| Data Processing | Apache Spark (local mode) |
| Experiment Tracking | MLflow |
| Versioning | Git + DVC (Google Drive remote) |
| Model Serving | FastAPI + MLflow models serve |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana + AlertManager |
| Frontend | React + Vite |

---

## Dataset

**Microsoft Azure Predictive Maintenance** ([Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance))

5 CSV files, joinable on `machineID`:

| File | Rows | Description |
|---|---|---|
| PdM_telemetry.csv | 876,100 | Hourly sensor readings (volt, rotate, pressure, vibration) |
| PdM_failures.csv | 761 | Component failure events |
| PdM_errors.csv | 3,919 | Non-breaking error events |
| PdM_maint.csv | 3,286 | Maintenance records |
| PdM_machines.csv | 100 | Machine metadata (model, age) |

---

## Quickstart

### Prerequisites

- Docker Desktop with WSL2
- Git
- DVC (`pip install dvc[gdrive]`)
- Node.js 20+ (for frontend dev only)

### 1. Clone and pull data

```bash
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
dvc pull
```

### 2. Start all services

```bash
docker compose up --build
```

Wait ~60 seconds for all services to initialize.

### 3. Run the pipeline

```bash
# Feature engineering (Spark)
python ml/features/feature_engineering.py

# Train model
python ml/training/train.py

# Or run full pipeline via DVC
dvc repro
```

### 4. Register model in MLflow

Go to `http://localhost:5000` → Models → `pdm_failure_classifier` → add alias `production` to the best version.

### 5. Load model in API

```bash
curl -X POST http://localhost:8000/model/reload
```

### 6. Open dashboard

Navigate to `http://localhost:3000`

---

## Services

| Service | URL | Credentials |
|---|---|---|
| React Dashboard | http://localhost:3000 | — |
| FastAPI | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |
| AlertManager | http://localhost:9093 | — |
| MLflow Serve | http://localhost:8001 | — |

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `GET /ready`
```json
{
  "model_loaded": true,
  "model_name": "pdm_failure_classifier",
  "model_version": "6",
  "tracking_uri": "http://mlflow:5000"
}
```

### `POST /predict`

**Input:**
```json
{
  "machineID": 1,
  "volt_mean_3h": 170.0,
  "volt_std_3h": 13.0,
  "volt_mean_24h": 170.0,
  "volt_std_24h": 14.0,
  "volt_min_24h": 141.0,
  "volt_max_24h": 200.0,
  "rotate_mean_3h": 446.0,
  "rotate_std_3h": 46.0,
  "rotate_mean_24h": 446.0,
  "rotate_std_24h": 49.0,
  "rotate_min_24h": 347.0,
  "rotate_max_24h": 545.0,
  "pressure_mean_3h": 100.0,
  "pressure_std_3h": 9.0,
  "pressure_mean_24h": 100.0,
  "pressure_std_24h": 10.0,
  "pressure_min_24h": 80.0,
  "pressure_max_24h": 120.0,
  "vibration_mean_3h": 40.0,
  "vibration_std_3h": 4.0,
  "vibration_mean_24h": 40.0,
  "vibration_std_24h": 5.0,
  "vibration_min_24h": 30.0,
  "vibration_max_24h": 50.0,
  "error1_count_24h": 0.0,
  "error2_count_24h": 0.0,
  "error3_count_24h": 0.0,
  "error4_count_24h": 0.0,
  "error5_count_24h": 0.0,
  "comp1_serviced": 0.0,
  "comp2_serviced": 0.0,
  "comp3_serviced": 0.0,
  "comp4_serviced": 0.0,
  "age": 5.0
}
```

**Output:**
```json
{
  "machineID": 1,
  "failure_probability": 0.0036,
  "failure_predicted": false,
  "risk_level": "low",
  "threshold_used": 0.7
}
```

### `POST /predict/batch`
```json
{"records": [<SensorReading>, ...]}
```

### `GET /drift/status`
```json
{
  "drifted": false,
  "features_checked": 24,
  "drifted_features": [],
  "psi_scores": {"volt_mean_3h": 0.04, ...}
}
```

### `GET /model/info`
```json
{
  "model_name": "pdm_failure_classifier",
  "run_id": "95eb0b34...",
  "tracking_uri": "http://mlflow:5000",
  "threshold": 0.7,
  "feature_count": 34
}
```

### `POST /model/reload`
Hot-swap the model without restart. Loads whichever version has the `production` alias in MLflow registry.
```json
{"status": "reloaded", "version": "6", "run_id": "95eb0b34..."}
```

### `GET /metrics`
Prometheus metrics in text format.

---

## MLflow Serve

The model is also served via native `mlflow models serve` on port 8001:

```bash
curl -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"volt_mean_3h": 170.0, ...}]}'
```

Returns: `{"predictions": [0]}` (0 = no failure, 1 = failure)

---

## Model Training

### Via CLI
```bash
python ml/training/train.py \
  --n_estimators 500 \
  --max_depth 8 \
  --learning_rate 0.03 \
  --scale_pos_weight 10 \
  --threshold 0.65 \
  --run_name "my_experiment"
```

### Via MLflow Projects
```bash
python -m mlflow run ml/ -e train --env-manager=local \
  -P n_estimators=500 \
  -P max_depth=8 \
  -P run_name="my_experiment"
```

### Via DVC
```bash
dvc repro
```

---

## Model Switching (Hot Swap)

1. Go to `http://localhost:5000` → Models → `pdm_failure_classifier`
2. Click the version you want to promote → Aliases → add `production`
3. Call `curl -X POST http://localhost:8000/model/reload`

The API instantly loads the new model. No rebuild, no restart.

---

## Feature Engineering

34 features engineered from raw telemetry:

| Feature Group | Count | Description |
|---|---|---|
| Voltage rolling stats | 6 | mean/std (3h), mean/std/min/max (24h) |
| Rotation rolling stats | 6 | same |
| Pressure rolling stats | 6 | same |
| Vibration rolling stats | 6 | same |
| Error counts | 5 | 24h count per error type (error1-5) |
| Component serviced | 4 | comp1-4 maintenance flag |
| Machine age | 1 | static metadata |

Failure label: `failure_within_24h` — 1 if any component fails within 24 hours of this reading.

---

## Model Performance

Best model (Version 6, `xgboost_deep_spw10_thres0.65`):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9985 |
| PR-AUC | 0.9434 |
| F1 | 0.8687 |
| Recall | 0.9407 |
| Precision | 0.8069 |
| Inference Latency | ~29ms |

Training config: `n_estimators=500, max_depth=8, learning_rate=0.03, scale_pos_weight=10, threshold=0.65`

All experiments tracked in MLflow at `http://localhost:5000`.

---

## Monitoring

Prometheus scrapes `http://api:8000/metrics` every 15 seconds.

Custom metrics:

| Metric | Type | Description |
|---|---|---|
| `pdm_prediction_requests_total` | Counter | Requests by endpoint and status |
| `pdm_prediction_latency_seconds` | Histogram | Latency distribution |
| `pdm_failure_predictions_total` | Counter | Predictions by outcome |
| `pdm_model_loaded` | Gauge | 1 if model loaded |
| `pdm_drift_detected` | Gauge | PSI drift flag per feature |
| `pdm_batch_size` | Histogram | Batch request sizes |

Grafana dashboard auto-provisioned at `http://localhost:3001` (admin/admin).

AlertManager rules:
- Error rate > 5% for 1 minute → `HighErrorRate` alert
- Drift detected for 1 minute → `DriftDetected` alert
- Model not loaded for 1 minute → `ModelNotLoaded` alert

---

## Drift Detection

During training, statistical baselines (mean, std, min, max) are computed for all 24 rolling sensor features and saved to `data/features/drift_baseline.csv`.

At inference time, incoming predictions are buffered (last 1000 requests). The `/drift/status` endpoint computes PSI scores:

```
PSI = (|current_mean - baseline_mean| / baseline_std + |current_std - baseline_std| / baseline_std) / 2
```

Features with PSI > 0.2 are flagged as drifted.

---

## DVC Pipeline

```bash
dvc dag
```

```
+---------------------+
| feature_engineering |
+---------------------+
          *
          *
          *
       +-------+
       | train |
       +-------+
```

Reproduce full pipeline:
```bash
dvc repro
```

Push data to remote:
```bash
dvc push
```

Pull data on a new machine:
```bash
dvc pull
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

44 tests across 4 suites — all passing:

| Suite | Tests |
|---|---|
| Unit: Feature Engineering | 13 |
| Unit: API Schemas | 6 |
| Unit: Drift Detection | 6 |
| Integration: Pipeline | 15 |

---

## Project Structure

```
Predictive_maintenance_project/
├── airflow/
│   ├── dags/
│   │   └── pdm_pipeline.py          # Airflow DAG
│   └── Dockerfile                   # Custom image with Java + PySpark
├── api/
│   ├── main.py                      # FastAPI app
│   ├── schemas.py                   # Pydantic schemas
│   ├── metrics.py                   # Prometheus metrics
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── raw/azure/                   # DVC tracked raw CSVs
│   └── features/                    # DVC tracked engineered features
├── frontend/
│   ├── src/App.jsx                  # React dashboard
│   └── Dockerfile
├── ml/
│   ├── features/
│   │   └── feature_engineering.py  # Spark feature engineering
│   ├── training/
│   │   └── train.py                # XGBoost training + MLflow
│   ├── MLproject                   # MLflow Projects entry points
│   └── python_env.yaml             # Environment spec
├── monitoring/
│   ├── prometheus.yml
│   ├── alerts.yml
│   ├── alertmanager.yml
│   └── grafana/provisioning/       # Auto-provisioned dashboards
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   └── report.tex                  # LaTeX documentation
├── docker-compose.yml
├── dvc.yaml                        # DVC pipeline stages
└── requirements.txt
```

---

## Documentation

Documentation is present in Assignment_report in project root

---

## Known Limitations

1. MLflow artifact UI browser does not render on Windows due to path resolution differences between host and container. Artifacts exist on disk and are loaded correctly by the API.
2. `mlflow run` CLI is blocked by Device Guard policy on some Windows machines. Use `python -m mlflow run` or call training directly via `python ml/training/train.py`.
3. Dashboard SCAN ALL UNITS uses randomized sensor values — a production deployment would pipe live telemetry.
4. Automated retraining on drift is designed but not implemented.

