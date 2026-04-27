from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os

PROJECT_DIR = "/opt/airflow/data"
RAW_DIR = f"{PROJECT_DIR}/raw/azure"
FEATURES_DIR = f"{PROJECT_DIR}/features"

REQUIRED_FILES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
    "PdM_machines.csv",
    "PdM_failures.csv",
]

def validate_raw_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f"{RAW_DIR}/{f}")]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")
    print(f"All {len(REQUIRED_FILES)} raw files present.")

def validate_features():
    parquet_dir = f"{FEATURES_DIR}/azure_features.parquet"
    baseline = f"{FEATURES_DIR}/drift_baseline.csv"
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError("Feature parquet missing.")
    if not os.path.exists(baseline):
        raise FileNotFoundError("Drift baseline missing.")
    parts = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not parts:
        raise ValueError("Parquet directory is empty.")
    print(f"Features validated: {len(parts)} parquet parts, drift baseline present.")

with DAG(
    dag_id="pdm_feature_pipeline",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["predictive-maintenance", "feature-engineering"],
) as dag:

    validate_inputs = PythonOperator(
        task_id="validate_raw_files",
        python_callable=validate_raw_files,
    )

    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=(
            "cd /opt/airflow && "
            "JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 "
            "python ml/features/feature_engineering.py"
        ),
        env={
            "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",
            "PATH": "/usr/lib/jvm/java-17-openjdk-amd64/bin:/usr/local/bin:/usr/bin:/bin",
            "HADOOP_HOME": "/opt/hadoop",
            "PYSPARK_PYTHON": "python3",
        },
    )

    validate_outputs = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
    )

    validate_inputs >> run_feature_engineering >> validate_outputs