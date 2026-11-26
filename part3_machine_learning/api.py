"""Part 3 Task 6.3: FastAPI deployment mock-up.

Serves the regression model and the recommendation system over HTTP.

Start the server from the repository root:

    uvicorn part3_machine_learning.api:app --port 8000

Then open http://127.0.0.1:8000/docs for an interactive page where you can
send requests, or run the scripted demo (no server needed):

    python -m part3_machine_learning.api_demo

Endpoints
---------
GET  /health      model and version information
POST /predict     traffic volume for one hour
POST /recommend   lowest-traffic travel window for a day
"""

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from common import config
from common.logging_setup import setup_logging
from part3_machine_learning import ml_common as mc
from part3_machine_learning.recommender import (RecommendationError, WEATHER_TYPES, build_day_features,
                                                load_assets, recommend)

setup_logging(mc.PART3_DIR / "api.log", append=True)
logger = logging.getLogger(__name__)

MODEL_VERSION = "traffic_volume_hist_gradient_boosting (see MLflow registry)"

app = FastAPI(title="I-94 Traffic Intelligence API",
              description="Traffic volume prediction and travel-time recommendations for westbound I-94.",
              version="1.0.0")


class PredictRequest(BaseModel):
    date_time: str = Field(..., examples=["2018-10-02 08:00"], description="Hour to predict, YYYY-MM-DD HH:MM")
    weather_main: str = Field("Clear", examples=["Clear"])
    temp_c: Optional[float] = Field(None, examples=[12.0], description="Default: typical for the month")

    @field_validator("weather_main")
    @classmethod
    def known_weather(cls, v: str) -> str:
        match = [w for w in WEATHER_TYPES if w.lower() == v.lower()]
        if not match:
            raise ValueError(f"weather_main must be one of {WEATHER_TYPES}")
        return match[0]


class RecommendRequest(BaseModel):
    date: str = Field(..., examples=["2018-10-02"])
    earliest: int = Field(6, ge=0, le=23)
    latest: int = Field(20, ge=0, le=23)
    window: int = Field(1, ge=1, le=12)
    weather_main: str = Field("Clear", examples=["Snow"])
    temp_c: Optional[float] = None


def category(volume: float) -> str:
    low, high = config.CATEGORY_BANDS
    return "Low" if volume < low else ("Medium" if volume <= high else "High")


def deployed_test_mae():
    """Test MAE of the deployed model, read from the supervised results (not hard-coded)."""
    try:
        rows = json.loads((mc.RESULTS_DIR / "supervised_results.json").read_text(encoding="utf-8"))["regression"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        logger.warning("supervised_results.json not readable; /health reports no test MAE")
        return None
    chosen = [r for r in rows if r.get("selected") == "yes"]
    return round(chosen[0]["test_MAE"], 1) if chosen else None


@app.get("/health")
def health() -> dict:
    model, features, *_ = load_assets()
    return {"status": "ok", "model": MODEL_VERSION, "n_features": len(features),
            "trained_on": "2012-10 to 2017-12", "model_chosen_on": "2017 validation year",
            "test_MAE_2018": deployed_test_mae()}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    logger.info("POST /predict %s", req.model_dump())
    try:
        ts = pd.Timestamp(datetime.strptime(req.date_time, "%Y-%m-%d %H:%M"))
    except ValueError:
        logger.error("Invalid date_time '%s'", req.date_time)
        raise HTTPException(status_code=422, detail="date_time must look like 2018-10-02 08:00") from None
    model, features, weather_medians, month_temp, *_ = load_assets()
    X = build_day_features(ts.normalize(), req.weather_main, req.temp_c, features, weather_medians, month_temp)
    volume = float(max(0.0, model.predict(X.iloc[[ts.hour]])[0]))
    result = {"date_time": ts.strftime("%Y-%m-%d %H:%M"), "weather_main": req.weather_main,
              "temp_c": round(float(X["temp_c"].iloc[0]), 1), "predicted_volume": round(volume),
              "traffic_category": category(volume), "congested": volume > config.CONGESTION_THRESHOLD}
    logger.info("Prediction %s -> %d vehicles (%s)", result["date_time"], result["predicted_volume"],
                result["traffic_category"])
    return result


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest) -> dict:
    logger.info("POST /recommend %s", req.model_dump())
    try:
        return recommend(req.date, req.earliest, req.latest, req.window, req.weather_main, req.temp_c)
    except RecommendationError as exc:
        logger.error("Invalid recommendation request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from None
