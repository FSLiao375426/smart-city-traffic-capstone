"""Shared building blocks for every Part 3 script.

* one feature set used by all models (time, cyclical, weather, holiday)
* one chronological train/test split (D7: train to 2017, test on 2018)
* one MLflow configuration (local SQLite store inside this folder)

Keeping these in one place means every model is trained and judged on
exactly the same data, so their scores can be compared fairly.
"""

import logging
import os
from pathlib import Path

# Keep MLflow fully local: no usage telemetry, no assistant hints in the log
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

from common import config
from common.data import load_features

logger = logging.getLogger(__name__)

PART3_DIR = config.ROOT_DIR / "part3_machine_learning"
MODELS_DIR = PART3_DIR / "models"
RESULTS_DIR = PART3_DIR / "results"
FIGURES_DIR = PART3_DIR / "figures"
LOG_FILE = PART3_DIR / "part3.log"
MLFLOW_DB = PART3_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = PART3_DIR / "mlruns"

# Common feature set (brief: time features, weather encodings, holiday flag,
# at least one cyclical encoding of hour and day of week).
# traffic_volume and anything derived from it are never features.
TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "month",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
HOLIDAY_FEATURES = ["is_holiday"]
WEATHER_NUMERIC = ["temp_c", "rain_1h_log", "snow_1h", "clouds_all", "weather_severity", "is_precipitation"]


def weather_dummies(df: pd.DataFrame) -> list:
    return sorted(c for c in df.columns if c.startswith("wx_"))


def feature_columns(df: pd.DataFrame) -> list:
    return TIME_FEATURES + HOLIDAY_FEATURES + WEATHER_NUMERIC + weather_dummies(df)


def load_model_data() -> pd.DataFrame:
    """Features table without flagged road-closure hours (D9)."""
    df = load_features()
    n_anomaly = int(df["traffic_anomaly"].sum())
    df = df[df["traffic_anomaly"] == 0].reset_index(drop=True)
    logger.warning("Excluded %d flagged road-closure hours (traffic_anomaly) from modelling", n_anomaly)
    logger.info("Model data: %d rows", len(df))
    return df


def time_split(df: pd.DataFrame):
    """Chronological split (D7). Hourly traffic is autocorrelated, so a random
    split would let the model see the neighbours of every test hour."""
    test_mask = df["date_time"] >= pd.Timestamp(config.TEST_START)
    train, test = df[~test_mask], df[test_mask]
    logger.info("Train: %d rows (%s to %s), test: %d rows (%s to %s)",
                len(train), train["date_time"].min().date(), train["date_time"].max().date(),
                len(test), test["date_time"].min().date(), test["date_time"].max().date())
    return train, test


def setup_mlflow(experiment: str) -> None:
    """Use a local SQLite store so runs, metrics and model versions live in the repo."""
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)
    logger.info("MLflow experiment '%s' (store %s)", experiment, MLFLOW_DB.name)


def ensure_dirs() -> None:
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
