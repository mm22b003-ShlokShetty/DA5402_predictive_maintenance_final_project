from pydantic import BaseModel
from typing import List, Optional

class SensorReading(BaseModel):
    machineID: int
    volt_mean_3h: float
    volt_std_3h: float
    volt_mean_24h: float
    volt_std_24h: float
    volt_min_24h: float
    volt_max_24h: float
    rotate_mean_3h: float
    rotate_std_3h: float
    rotate_mean_24h: float
    rotate_std_24h: float
    rotate_min_24h: float
    rotate_max_24h: float
    pressure_mean_3h: float
    pressure_std_3h: float
    pressure_mean_24h: float
    pressure_std_24h: float
    pressure_min_24h: float
    pressure_max_24h: float
    vibration_mean_3h: float
    vibration_std_3h: float
    vibration_mean_24h: float
    vibration_std_24h: float
    vibration_min_24h: float
    vibration_max_24h: float
    error1_count_24h: float = 0.0
    error2_count_24h: float = 0.0
    error3_count_24h: float = 0.0
    error4_count_24h: float = 0.0
    error5_count_24h: float = 0.0
    comp1_serviced: float = 0.0
    comp2_serviced: float = 0.0
    comp3_serviced: float = 0.0
    comp4_serviced: float = 0.0
    age: float = 0.0

class PredictionResponse(BaseModel):
    machineID: int
    failure_probability: float
    failure_predicted: bool
    risk_level: str
    threshold_used: float

class BatchRequest(BaseModel):
    records: List[SensorReading]

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str

class ReadyResponse(BaseModel):
    model_loaded: bool
    model_name: str
    model_version: Optional[str]
    tracking_uri: str

class DriftResponse(BaseModel):
    drifted: bool
    features_checked: int
    drifted_features: List[str]
    psi_scores: dict

class ModelInfoResponse(BaseModel):
    model_name: str
    run_id: Optional[str]
    tracking_uri: str
    threshold: float
    feature_count: int