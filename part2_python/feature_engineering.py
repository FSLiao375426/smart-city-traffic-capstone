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
          is_precipitation, one-hot weather_main (wx_*)
Scaled    temp_c_z (z-score), clouds_all_mm (min-max), rain_1h_log
Targets   congested (> 5,500, Part 1), traffic_category (Low/Medium/High bands),
          congestion_category (quartiles, Part 3), high_risk (proxy label, Part 3)

Scaling note: the scaled columns here use the whole dataset and serve
exploration. Part 3 refits its scalers on the training years only, so no
information from the 2018 test set leaks into training.
"""

import argparse
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
        day_of_week=dt.dt.dayofweek,          # 0 = Monday ... 6 = Sunday
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
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_c"] = df["temp"] - 273.15
    # Invert the severity rank so a larger number means harsher weather (Clear = 0)
    worst = max(config.WEATHER_SEVERITY.values())
    df["weather_severity"] = worst - df["weather_main"].map(config.WEATHER_SEVERITY)
    df["is_severe_weather"] = df["weather_main"].isin(config.SEVERE_WEATHER).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(config.LOW_VISIBILITY).astype(int)
    df["is_precipitation"] = (
        df["weather_main"].isin(["Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"])
        | (df["rain_1h"] > 0) | (df["snow_1h"] > 0)
    ).astype(int)

    dummies = pd.get_dummies(df["weather_main"], prefix="wx", dtype=int)
    df = pd.concat([df, dummies], axis=1)

    logger.info("Added weather features: temp_c, weather_severity, is_severe_weather, "
                "is_low_visibility, is_precipitation and %d one-hot weather columns", dummies.shape[1])
    logger.debug("Weather severity scores: %s",
                 {k: worst - v for k, v in sorted(config.WEATHER_SEVERITY.items(), key=lambda kv: kv[1])})
    return df


# ---------------------------------------------------------------------------
# Scaled numerical features
# ---------------------------------------------------------------------------
def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mean, std = df["temp_c"].mean(), df["temp_c"].std()
    df["temp_c_z"] = (df["temp_c"] - mean) / std
    logger.debug("temp_c z-score parameters: mean = %.3f, std = %.3f", mean, std)

    lo, hi = df["clouds_all"].min(), df["clouds_all"].max()
    df["clouds_all_mm"] = (df["clouds_all"] - lo) / (hi - lo)
    logger.debug("clouds_all min-max parameters: min = %s, max = %s", lo, hi)

    # Rainfall is mostly 0 with a long right tail, so log1p compresses it
    df["rain_1h_log"] = np.log1p(df["rain_1h"])
    logger.debug("rain_1h before log1p: share of zeros = %.3f, max = %.2f",
                 (df["rain_1h"] == 0).mean(), df["rain_1h"].max())

    logger.info("Added scaled features: temp_c_z (z-score), clouds_all_mm (min-max), rain_1h_log (log1p)")
    return df


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three congestion definitions (D5) and the proxy risk label.

    congestion_category uses data-driven quartiles of traffic_volume, so each
    class holds about a quarter of the hours:
        Low <= Q1 < Medium <= Q2 < High <= Q3 < Severe
    """
    df = df.copy()
    vol = df["traffic_volume"]

    df["congested"] = (vol > config.CONGESTION_THRESHOLD).astype(int)

    low, high = config.CATEGORY_BANDS
    df["traffic_category"] = np.select([vol < low, vol <= high], ["Low", "Medium"], default="High")

    q1, q2, q3 = vol.quantile([0.25, 0.5, 0.75]).to_numpy()
    logger.debug("Congestion quartile thresholds: Q1 = %.1f, Q2 = %.1f, Q3 = %.1f", q1, q2, q3)

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
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dataset shape before feature engineering: %d rows, %d columns", *df.shape)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_features(df)
    df = add_targets(df)
    logger.info("Dataset shape after feature engineering: %d rows, %d columns", *df.shape)
    return df


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
        features = build_features(df)
        features.to_csv(config.FEATURES_CSV, index=False)
    except (KeyError, ValueError, OSError):
        logger.error("Feature engineering did not complete", exc_info=True)
        return 1
    logger.info("Saved features to %s", config.FEATURES_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
