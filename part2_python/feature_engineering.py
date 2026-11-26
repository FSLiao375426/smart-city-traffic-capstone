"""Part 2 Task 2: feature engineering with NumPy and pandas.

Reads data/processed/traffic_clean.csv (the pipeline output) and writes
data/processed/traffic_features.csv, the ML-ready table used by Part 3.

Run from the repository root (after the pipeline):

    python -m part2_python.feature_engineering            # appends to pipeline.log
    python -m part2_python.feature_engineering --debug    # also logs intermediate values

Feature groups
--------------
Time      hour, day_of_week, is_weekend, month, year, time_of_day,
          cyclical sin/cos encodings of hour, day of week and month
Weather   temp_c, weather_severity, is_severe_weather, is_low_visibility,
          is_precipitation, multi-hot weather indicators (wx_*, D21)
Scaled    temp_c_z (z-score), clouds_all_mm (min-max), rain_1h_log
Targets   congested (> 5,500, Part 1), traffic_category (Low/Medium/High bands),
          congestion_category (quartiles, Part 3), high_risk (proxy label, Part 3)

Fitted values (D20): the z-score parameters, the min-max range and the
quartile thresholds are learned from the training period only (before
config.VALID_START) and saved to data/processed/feature_params.json.
The 2017 validation year and the 2018 test year are then transformed with
those saved values, so they never influence how a feature or label is defined.
"""

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from common import config
from common.data import load_clean
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
def cyclical(values: pd.Series, period: int) -> tuple:
    """Map a repeating value onto a circle so the end of the cycle sits next to the start.

    Hour 23 and hour 0 are 23 apart as numbers but 1 hour apart on the clock.
    As (sin, cos) points on a unit circle they are neighbours.
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def time_of_day(hour: pd.Series) -> pd.Series:
    """Label hours with the bands seen in the Part 1 hourly profile."""
    bins = [-1, 5, 9, 14, 18, 23]
    labels = ["Night", "Morning peak", "Midday", "Evening peak", "Evening"]
    return pd.cut(hour, bins=bins, labels=labels).astype("string")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["date_time"]
    df = df.assign(
        hour=dt.dt.hour,
        day_of_week=dt.dt.day_of_week,          # 0 = Monday ... 6 = Sunday
        day_name=dt.dt.day_name(),
        month=dt.dt.month,
        year=dt.dt.year,
        date=dt.dt.date.astype("string"),
    )
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_type"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")
    df["time_of_day"] = time_of_day(df["hour"])

    df["hour_sin"], df["hour_cos"] = cyclical(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = cyclical(df["day_of_week"], 7)
    df["month_sin"], df["month_cos"] = cyclical(df["month"] - 1, 12)

    logger.info("Added time features: hour, day_of_week, is_weekend, month, year, time_of_day "
                "and cyclical encodings of hour, day of week and month")
    return df


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------
def has_any_label(labels: pd.Series, wanted: list) -> pd.Series:
    """1 when at least one of the hour's weather labels is in `wanted`."""
    return labels.apply(lambda found: int(any(label in wanted for label in found)))


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather features from ALL labels of the hour (D21), not only the most severe.

    An hour labelled "Rain|Mist" is both precipitation and low visibility.
    weather_main (the most severe label) still drives weather_severity, so
    charts and group summaries keep one non-overlapping category per hour.
    """
    df = df.copy()
    df["temp_c"] = df["temp"] - 273.15
    # Invert the severity rank so a larger number means harsher weather (Clear = 0)
    worst = max(config.WEATHER_SEVERITY.values())
    df["weather_severity"] = worst - df["weather_main"].map(config.WEATHER_SEVERITY)

    labels = df["weather_labels"].str.split("|")
    df["is_severe_weather"] = has_any_label(labels, config.SEVERE_WEATHER)
    df["is_low_visibility"] = has_any_label(labels, config.LOW_VISIBILITY)
    precipitation = has_any_label(labels, ["Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"])
    df["is_precipitation"] = ((precipitation == 1) | (df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)

    # Multi-hot: one column per known condition, so the columns are the same in every run
    for condition in config.WEATHER_SEVERITY:
        df[f"wx_{condition}"] = has_any_label(labels, [condition])

    primary_only_lowvis = int(df["weather_main"].isin(config.LOW_VISIBILITY).sum())
    logger.info("Added weather features: temp_c, weather_severity, is_severe_weather, "
                "is_low_visibility, is_precipitation and %d multi-hot weather columns",
                len(config.WEATHER_SEVERITY))
    logger.info("Low-visibility hours: %d using all labels vs %d using only the most severe label",
                int(df["is_low_visibility"].sum()), primary_only_lowvis)
    logger.debug("Weather severity scores: %s",
                 {k: worst - v for k, v in sorted(config.WEATHER_SEVERITY.items(), key=lambda kv: kv[1])})
    return df


# ---------------------------------------------------------------------------
# Scaled numerical features
# ---------------------------------------------------------------------------
def reference_period(df: pd.DataFrame) -> pd.Series:
    """True for training-period rows, the only rows used to fit values (D20)."""
    return df["date_time"] < pd.Timestamp(config.VALID_START)


def add_scaled_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    ref = df.loc[reference_period(df)]

    mean, std = ref["temp_c"].mean(), ref["temp_c"].std()
    df["temp_c_z"] = (df["temp_c"] - mean) / std
    logger.debug("temp_c z-score parameters (training period): mean = %.3f, std = %.3f", mean, std)

    lo, hi = ref["clouds_all"].min(), ref["clouds_all"].max()
    df["clouds_all_mm"] = (df["clouds_all"] - lo) / (hi - lo)
    logger.debug("clouds_all min-max parameters (training period): min = %s, max = %s", lo, hi)

    # Rainfall is mostly 0 with a long right tail, so log1p compresses it (nothing to fit)
    df["rain_1h_log"] = np.log1p(df["rain_1h"])
    logger.debug("rain_1h before log1p: share of zeros = %.3f, max = %.2f",
                 (df["rain_1h"] == 0).mean(), df["rain_1h"].max())

    params["temp_c_z"] = {"mean": float(mean), "std": float(std)}
    params["clouds_all_mm"] = {"min": float(lo), "max": float(hi)}
    logger.info("Added scaled features: temp_c_z (z-score), clouds_all_mm (min-max), rain_1h_log (log1p); "
                "parameters fitted on %d training-period hours", len(ref))
    return df


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Create the three congestion definitions (D5) and the proxy risk label.

    congestion_category uses data-driven quartiles of traffic_volume. The
    thresholds come from the training period only (D20), so each class holds
    about a quarter of the training hours:
        Low <= Q1 < Medium <= Q2 < High <= Q3 < Severe
    """
    df = df.copy()
    vol = df["traffic_volume"]

    df["congested"] = (vol > config.CONGESTION_THRESHOLD).astype(int)

    low, high = config.CATEGORY_BANDS
    df["traffic_category"] = np.select([vol < low, vol <= high], ["Low", "Medium"], default="High")

    ref_vol = vol[reference_period(df)]
    q1, q2, q3 = ref_vol.quantile([0.25, 0.5, 0.75]).to_numpy()
    logger.debug("Congestion quartile thresholds (training period, %d hours): Q1 = %.1f, Q2 = %.1f, Q3 = %.1f",
                 len(ref_vol), q1, q2, q3)
    params["congestion_quartiles"] = {"q1": float(q1), "q2": float(q2), "q3": float(q3)}

    labels = []
    for v in vol.to_numpy():               # explicit loop mirrors the brief's bucket() function
        if v <= q1:
            labels.append("Low")
        elif v <= q2:
            labels.append("Medium")
        elif v <= q3:
            labels.append("High")
        else:
            labels.append("Severe")
    df["congestion_category"] = labels

    high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = (df["is_severe_weather"] == 1) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)

    counts = df["congestion_category"].value_counts().reindex(["Low", "Medium", "High", "Severe"])
    logger.info("Created congestion_category from quartiles: %s", counts.to_dict())
    logger.info("Created proxy label high_risk: %d positive hours (%.1f%%)",
                df["high_risk"].sum(), 100 * df["high_risk"].mean())
    if df["high_risk"].mean() < 0.10:
        logger.warning("high_risk is imbalanced (%.1f%% positive); use class weights and "
                       "precision/recall/F1 rather than accuracy in Part 3", 100 * df["high_risk"].mean())
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Add every feature group. Fitted values are written into `params`."""
    params = {} if params is None else params
    params["fitted_on"] = f"date_time < {config.VALID_START} (training period)"
    logger.info("Dataset shape before feature engineering: %d rows, %d columns", *df.shape)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_features(df, params)
    df = add_targets(df, params)
    logger.info("Dataset shape after feature engineering: %d rows, %d columns", *df.shape)
    return df


def save_params(params: dict) -> None:
    """Save the fitted values so later steps (and graders) can see and reuse them."""
    config.FEATURE_PARAMS_JSON.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved fitted feature parameters to %s", config.FEATURE_PARAMS_JSON)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build ML-ready features from traffic_clean.csv")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    args = parser.parse_args(argv)
    setup_logging(config.LOG_FILE, debug=args.debug, append=True)  # continues pipeline.log

    try:
        df = load_clean()
    except FileNotFoundError:
        logger.error("traffic_clean.csv not found. Run 'python -m part2_python.pipeline' first.", exc_info=True)
        return 1
    logger.info("Loaded cleaned data for feature engineering from %s", config.CLEAN_CSV.name)

    try:
        params = {}
        features = build_features(df, params)
        features.to_csv(config.FEATURES_CSV, index=False)
        save_params(params)
    except (KeyError, ValueError, OSError):
        logger.error("Feature engineering did not complete", exc_info=True)
        return 1
    logger.info("Saved features to %s", config.FEATURES_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
