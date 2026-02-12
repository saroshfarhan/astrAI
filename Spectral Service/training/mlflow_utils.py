# spectral_service/training/mlflow_utils.py
from __future__ import annotations
import os
import mlflow

def setup_mlflow(experiment_name: str = "spectral-service", tracking_uri: str | None = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def safe_log_param(k, v):
    try: mlflow.log_param(k, v)
    except Exception: pass

def safe_log_metric(k, v, step=None):
    try: mlflow.log_metric(k, float(v), step=step)
    except Exception: pass
