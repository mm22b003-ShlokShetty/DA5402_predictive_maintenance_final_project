import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import pyarrow.parquet as pq
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FEATURES_DIR = "data/features"
EXPERIMENT_NAME = "pdm_failure_prediction"
THRESHOLD = 0.7

FEATURE_COLS = [
    "volt_mean_3h", "volt_std_3h", "volt_mean_24h", "volt_std_24h", "volt_min_24h", "volt_max_24h",
    "rotate_mean_3h", "rotate_std_3h", "rotate_mean_24h", "rotate_std_24h", "rotate_min_24h", "rotate_max_24h",
    "pressure_mean_3h", "pressure_std_3h", "pressure_mean_24h", "pressure_std_24h", "pressure_min_24h", "pressure_max_24h",
    "vibration_mean_3h", "vibration_std_3h", "vibration_mean_24h", "vibration_std_24h", "vibration_min_24h", "vibration_max_24h",
    "error1_count_24h", "error2_count_24h", "error3_count_24h", "error4_count_24h", "error5_count_24h",
    "comp1_serviced", "comp2_serviced", "comp3_serviced", "comp4_serviced",
    "age"
]
TARGET_COL = "failure_within_24h"

PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": 10,
}

def load_features():
    table = pq.read_table(f"{FEATURES_DIR}/azure_features.parquet")
    df = table.to_pandas()
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def load_drift_baseline():
    return pd.read_csv(f"{FEATURES_DIR}/drift_baseline.csv")

def get_git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location = "file:///" + os.path.abspath("mlflow-data/artifacts").replace("\\", "/")
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading features...")
    df = load_features()

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Failure rate train: {y_train.mean():.4f} | test: {y_test.mean():.4f}")

    with mlflow.start_run(run_name="xgboost_final_spw10_thres0.7") as run:
        mlflow.log_params(PARAMS)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("failure_window_hours", 24)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_param("dataset", "azure_pdm")
        mlflow.log_param("feature_version", "v1_rolling_windows")
        mlflow.log_param("class_0_train", int((y_train == 0).sum()))
        mlflow.log_param("class_1_train", int((y_train == 1).sum()))

        print("Training XGBoost...")
        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.log_metric("pr_auc",    pr_auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall",    report["1"]["recall"])
        mlflow.log_metric("f1",        report["1"]["f1-score"])
        mlflow.log_metric("accuracy",  report["accuracy"])

        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metric("true_positives",  int(tp))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("true_negatives",  int(tn))
        mlflow.log_metric("false_negatives", int(fn))

        importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        mlflow.log_dict(importance, "feature_importance.json")

        baseline_df = load_drift_baseline()
        mlflow.log_dict(baseline_df.to_dict(), "drift_baseline.json")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test.iloc[:5],
        )

        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "pdm_failure_classifier"
        )

        print(f"\nRun ID: {run.info.run_id}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"PR-AUC:   {pr_auc:.4f}")
        print(f"F1:       {report['1']['f1-score']:.4f}")
        print(f"Recall:   {report['1']['recall']:.4f}")
        print(f"Precision:{report['1']['precision']:.4f}")

if __name__ == "__main__":
    main()