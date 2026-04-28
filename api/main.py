import os
import time
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from collections import deque
import threading
import mlflow
import mlflow.sklearn

from schemas import SensorReading, PredictionResponse, BatchRequest, BatchResponse
from schemas import HealthResponse, ReadyResponse, DriftResponse, ModelInfoResponse
from metrics import (
    REQUEST_COUNT, PREDICTION_LATENCY, FAILURE_PREDICTIONS,
    MODEL_LOADED, DRIFT_DETECTED, BATCH_SIZE
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "pdm_failure_classifier")
THRESHOLD = float(os.getenv("THRESHOLD", "0.7"))
DRIFT_BASELINE_PATH = os.getenv("DRIFT_BASELINE_PATH", "/app/data/drift_baseline.csv")

FEATURE_COLS = [
    "volt_mean_3h", "volt_std_3h", "volt_mean_24h", "volt_std_24h", "volt_min_24h", "volt_max_24h",
    "rotate_mean_3h", "rotate_std_3h", "rotate_mean_24h", "rotate_std_24h", "rotate_min_24h", "rotate_max_24h",
    "pressure_mean_3h", "pressure_std_3h", "pressure_mean_24h", "pressure_std_24h", "pressure_min_24h", "pressure_max_24h",
    "vibration_mean_3h", "vibration_std_3h", "vibration_mean_24h", "vibration_std_24h", "vibration_min_24h", "vibration_max_24h",
    "error1_count_24h", "error2_count_24h", "error3_count_24h", "error4_count_24h", "error5_count_24h",
    "comp1_serviced", "comp2_serviced", "comp3_serviced", "comp4_serviced",
    "age"
]

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_version = None
drift_baseline = None
prediction_buffer = deque(maxlen=1000)
buffer_lock = threading.Lock()


def load_model():
    global model, model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME)
    if not versions:
        raise RuntimeError(f"No versions found for model {MODEL_NAME}")
    latest = versions[-1]
    model_version = latest.version
    run_id = latest.run_id
    model_path = f"/mlflow-data/artifacts/{run_id}/artifacts/model"
    model = mlflow.sklearn.load_model(model_path)
    MODEL_LOADED.set(1)
    print(f"Model {MODEL_NAME} v{model_version} loaded from {model_path}")


def load_drift_baseline():
    global drift_baseline
    if os.path.exists(DRIFT_BASELINE_PATH):
        drift_baseline = pd.read_csv(DRIFT_BASELINE_PATH)
        print("Drift baseline loaded.")
    else:
        print(f"Drift baseline not found at {DRIFT_BASELINE_PATH}")




@app.on_event("startup")
async def startup():
    load_drift_baseline()
    try:
        load_model()
    except Exception as e:
        print(f"Model load failed on startup: {e}")
        MODEL_LOADED.set(0)


def get_risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "critical"
    elif prob >= 0.6:
        return "high"
    elif prob >= 0.4:
        return "medium"
    return "low"


def make_prediction(record: SensorReading) -> PredictionResponse:
    df = pd.DataFrame([record.model_dump()])[FEATURE_COLS].astype(float)
    prob = float(model.predict_proba(df)[0][1])
    predicted = prob >= THRESHOLD
    FAILURE_PREDICTIONS.labels(predicted=str(predicted)).inc()
    with buffer_lock:
        prediction_buffer.append(df.iloc[0].to_dict())
    return PredictionResponse(
        machineID=record.machineID,
        failure_probability=round(prob, 4),
        failure_predicted=predicted,
        risk_level=get_risk_level(prob),
        threshold_used=THRESHOLD,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return ReadyResponse(
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_version=str(model_version) if model_version else None,
        tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: SensorReading):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        result = make_prediction(record)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(time.time() - start)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    BATCH_SIZE.observe(len(batch.records))
    try:
        predictions = [make_prediction(r) for r in batch.records]
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        return BatchResponse(predictions=predictions)
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(time.time() - start)



@app.get("/drift/status", response_model=DriftResponse)
def drift_status():
    if drift_baseline is None:
        raise HTTPException(status_code=503, detail="Drift baseline not loaded")

    with buffer_lock:
        buffer_snapshot = list(prediction_buffer)

    sensor_cols = [c for c in FEATURE_COLS if any(s in c for s in ["volt", "rotate", "pressure", "vibration"])]
    baseline_means = drift_baseline[drift_baseline["summary"] == "mean"]
    baseline_stds = drift_baseline[drift_baseline["summary"] == "stddev"]

    psi_scores = {}
    drifted_features = []
    PSI_THRESHOLD = 0.2

    if len(buffer_snapshot) < 10:
        for col in sensor_cols:
            psi_scores[col] = 0.0
    else:
        recent_df = pd.DataFrame(buffer_snapshot)
        for col in sensor_cols:
            if col not in baseline_means.columns:
                continue
            baseline_mean = float(baseline_means[col].values[0])
            baseline_std = float(baseline_stds[col].values[0]) if col in baseline_stds.columns else 1.0
            if baseline_std == 0:
                baseline_std = 1e-9
            current_mean = float(recent_df[col].mean())
            current_std = float(recent_df[col].std()) if len(recent_df) > 1 else 0.0
            mean_shift = abs(current_mean - baseline_mean) / baseline_std
            std_shift = abs(current_std - baseline_std) / (baseline_std + 1e-9)
            psi = round((mean_shift + std_shift) / 2, 4)
            psi_scores[col] = psi
            if psi > PSI_THRESHOLD:
                drifted_features.append(col)
                DRIFT_DETECTED.labels(feature=col).set(1)
            else:
                DRIFT_DETECTED.labels(feature=col).set(0)

    DRIFT_DETECTED.labels(feature="any").set(len(drifted_features) > 0)

    return DriftResponse(
        drifted=len(drifted_features) > 0,
        features_checked=len(sensor_cols),
        drifted_features=drifted_features,
        psi_scores=psi_scores,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        run_id=None,
        tracking_uri=MLFLOW_TRACKING_URI,
        threshold=THRESHOLD,
        feature_count=len(FEATURE_COLS),
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)