from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "pdm_prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)

PREDICTION_LATENCY = Histogram(
    "pdm_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"]
)

FAILURE_PREDICTIONS = Counter(
    "pdm_failure_predictions_total",
    "Total failure predictions made",
    ["predicted"]
)

MODEL_LOADED = Gauge(
    "pdm_model_loaded",
    "Whether model is loaded"
)

DRIFT_DETECTED = Gauge(
    "pdm_drift_detected",
    "Whether drift is detected",
    ["feature"]
)

BATCH_SIZE = Histogram(
    "pdm_batch_size",
    "Batch prediction size",
    buckets=[1, 5, 10, 25, 50, 100]
)