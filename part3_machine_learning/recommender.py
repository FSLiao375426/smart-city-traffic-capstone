"""Part 3 Task 5: travel-time recommendation system.

The dataset covers one corridor, so the system recommends WHEN to travel,
not which route. For a given date it:

1. builds the model features for every hour of that day (calendar, holiday,
   expected weather and temperature),
2. predicts hourly traffic with the gradient boosting model from supervised.py,
3. finds the lowest-traffic window inside the user's acceptable time range,
4. compares it with the busiest window and the historical norm,
5. checks how many past hours match the request (same weather, day type and
   hours). With fewer than MIN_SUPPORT hours the forecast is still given but
   flagged as low confidence, because the model has seen too few examples, and
6. writes a plain-language recommendation.

Unlike the Part 2 CLI (historical averages only), the model adjusts for the
specific date: month, holidays, and the weather the user expects.

Run from the repository root (after supervised.py):

    python -m part3_machine_learning.recommender --date 2018-10-02 --earliest 6 --latest 10 --window 1
    python -m part3_machine_learning.recommender --date 2018-12-25 --earliest 8 --latest 18 --window 2
    python -m part3_machine_learning.recommender --date 2018-11-14 --weather Snow --temp -5
"""

import argparse
import logging
import sys
from datetime import date as Date
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from common import config
from common.data import load_features
from common.logging_setup import setup_logging
from part3_machine_learning import ml_common as mc

logger = logging.getLogger(__name__)
WEATHER_TYPES = sorted(config.WEATHER_SEVERITY)
MIN_SUPPORT = 30     # past hours with the same weather, day type and hour range


class RecommendationError(ValueError):
    """Invalid request (bad date, unknown weather, empty time range)."""


@lru_cache(maxsize=1)
def load_assets():
    """Model, feature list and historical medians (cached: loaded once per process)."""
    bundle = joblib.load(mc.MODELS_DIR / "regression_model.joblib")
    hist = load_features()
    hist = hist[hist["traffic_anomaly"] == 0]           # same exclusion as training (D9)
    weather_medians = hist.groupby("weather_main")[["rain_1h_log", "snow_1h", "clouds_all"]].median()
    month_temp = hist.groupby("month")["temp_c"].median()
    profile = hist.groupby(["is_weekend", "hour"])["traffic_volume"].mean()
    # Support counts: past hours (the model's development years) per weather label, day type and hour
    dev = hist[hist["date_time"] < pd.Timestamp(config.TEST_START)]
    support = {w: dev[dev[f"wx_{w}"] == 1].groupby(["is_weekend", "hour"]).size() for w in WEATHER_TYPES}
    return bundle["model"], bundle["features"], weather_medians, month_temp, profile, support


def history_support(support: dict, weather: str, weekend: int, earliest: int, latest: int) -> int:
    """Past hours with this weather on the same day type within the requested hours."""
    counts = support[weather]
    if weekend not in counts.index.get_level_values(0):
        return 0
    return int(counts.loc[weekend].reindex(range(earliest, latest + 1), fill_value=0).sum())


def is_federal_holiday(day: pd.Timestamp) -> bool:
    cal = USFederalHolidayCalendar()
    return day.normalize() in cal.holidays(start=day - pd.Timedelta(days=1), end=day + pd.Timedelta(days=1))


def build_day_features(day: pd.Timestamp, weather: str, temp_c, features: list,
                       weather_medians: pd.DataFrame, month_temp: pd.Series) -> pd.DataFrame:
    """One row per hour of `day`, with every feature the model expects."""
    hours = np.arange(24)
    dow, month = day.day_of_week, day.month
    if temp_c is None:
        temp_c = float(month_temp.loc[month])      # typical temperature for that month
    w = weather_medians.loc[weather]
    row = {
        "hour": hours, "day_of_week": dow, "is_weekend": int(dow >= 5), "month": month,
        "hour_sin": np.sin(2 * np.pi * hours / 24), "hour_cos": np.cos(2 * np.pi * hours / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12), "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_holiday": int(is_federal_holiday(day)), "temp_c": temp_c,
        "rain_1h_log": w["rain_1h_log"], "snow_1h": w["snow_1h"], "clouds_all": w["clouds_all"],
        "weather_severity": max(config.WEATHER_SEVERITY.values()) - config.WEATHER_SEVERITY[weather],
        "is_precipitation": int(weather in ("Rain", "Drizzle", "Snow", "Thunderstorm", "Squall")),
    }
    df = pd.DataFrame(row)
    for f in features:
        if f.startswith("wx_"):
            df[f] = int(f == f"wx_{weather}")
    return df[features]


def recommend(day: str, earliest: int = 6, latest: int = 20, window: int = 1,
              weather: str = "Clear", temp_c=None) -> dict:
    """Return the recommendation as a dictionary (used by the CLI and the API)."""
    try:
        ts = pd.Timestamp(Date.fromisoformat(day))
    except ValueError:
        raise RecommendationError(f"Invalid date '{day}'. Use YYYY-MM-DD.") from None
    if not (0 <= earliest <= 23 and 0 <= latest <= 23):
        raise RecommendationError("earliest and latest must be hours between 0 and 23")
    if latest - earliest + 1 < window or window < 1:
        raise RecommendationError(f"A {window}-hour window does not fit between {earliest}:00 and {latest}:59")
    match = [w for w in WEATHER_TYPES if w.lower() == str(weather).lower()]
    if not match:
        raise RecommendationError(f"Unknown weather '{weather}'. Choose from: {', '.join(WEATHER_TYPES)}")
    weather = match[0]

    model, features, weather_medians, month_temp, profile, support = load_assets()
    X = build_day_features(ts, weather, temp_c, features, weather_medians, month_temp)
    hourly = pd.Series(model.predict(X), index=range(24)).clip(lower=0)
    windows = hourly.rolling(window).mean().shift(-(window - 1)).loc[earliest:latest - window + 1].dropna()

    best, worst = int(windows.idxmin()), int(windows.idxmax())
    saving = 1 - windows.min() / windows.max()
    typical = profile.loc[int(ts.day_of_week >= 5)]
    vs_typical = windows.min() / typical.loc[best:best + window - 1].mean() - 1
    day_type = "weekend" if ts.day_of_week >= 5 else "weekday"
    holiday = bool(X["is_holiday"].iloc[0])
    temp_used = float(X["temp_c"].iloc[0])

    text = (f"For your {ts:%A %d %B %Y} journey between {earliest:02d}:00 and {latest + 1:02d}:00"
            f"{' (a public holiday)' if holiday else ''}, in {weather} conditions around {temp_used:.0f} °C, "
            f"consider travelling {best:02d}:00-{best + window:02d}:00. Expected traffic is about "
            f"{windows.min():,.0f} vehicles per hour, {saving:.0%} lower than the busiest time in your range "
            f"({worst:02d}:00-{worst + window:02d}:00, about {windows.max():,.0f}).")
    if windows.min() > config.CONGESTION_THRESHOLD:
        text += " Every window in your range is expected to be congested, so consider widening it."
    n_support = history_support(support, weather, int(ts.day_of_week >= 5), earliest, latest)
    confidence = "normal" if n_support >= MIN_SUPPORT else "low"
    if confidence == "low":
        seen = f"only {n_support}" if n_support else "no"
        text += (f" Caution: {seen} past {day_type} hours in this time range had {weather} "
                 f"conditions, so this forecast is uncertain.")
        logger.warning("Low support for %s on a %s between %02d:00 and %02d:00: %d past hours (minimum %d)",
                       weather, day_type, earliest, latest, n_support, MIN_SUPPORT)

    result = {
        "date": ts.strftime("%Y-%m-%d"), "day_type": day_type, "holiday": holiday,
        "weather": weather, "temperature_c": round(temp_used, 1),
        "best_window": f"{best:02d}:00-{best + window:02d}:00", "best_expected_volume": round(float(windows.min())),
        "busiest_window": f"{worst:02d}:00-{worst + window:02d}:00",
        "busiest_expected_volume": round(float(windows.max())),
        "reduction_vs_busiest": round(float(saving), 3),
        "best_vs_typical_same_hours": round(float(vs_typical), 3),
        "history_support_hours": n_support, "confidence": confidence,
        "hourly_forecast": {f"{h:02d}:00": round(float(v)) for h, v in hourly.items()},
        "recommendation": text,
    }
    logger.info("Recommendation for %s (%s, %s): best %s, %.0f%% below busiest",
                result["date"], weather, day_type, result["best_window"], 100 * saving)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Recommend a lower-traffic travel window on I-94 westbound")
    p.add_argument("--date", required=True, help="travel date, YYYY-MM-DD")
    p.add_argument("--earliest", type=int, default=6, help="earliest departure hour (0-23)")
    p.add_argument("--latest", type=int, default=20, help="latest hour the window may cover (0-23)")
    p.add_argument("--window", type=int, default=1, help="trip length in hours")
    p.add_argument("--weather", default="Clear", help=f"expected weather: {', '.join(WEATHER_TYPES)}")
    p.add_argument("--temp", type=float, default=None, help="expected temperature in °C (default: monthly norm)")
    args = p.parse_args(argv)
    setup_logging(mc.LOG_FILE, append=True, console_level=logging.WARNING)
    logger.info("recommender invoked with %s", vars(args))
    try:
        r = recommend(args.date, args.earliest, args.latest, args.window, args.weather, args.temp)
    except RecommendationError as exc:
        logger.error("Invalid request: %s", exc)
        return 2
    except FileNotFoundError:
        logger.error("Model not found. Run 'python -m part3_machine_learning.supervised' first.")
        return 1
    print(f"\n{r['recommendation']}\n")
    print("  Hourly forecast in your range:")
    for h in range(args.earliest, args.latest + 1):
        v = r["hourly_forecast"][f"{h:02d}:00"]
        bar = "█" * int(v / 250)
        print(f"    {h:02d}:00  {v:>6,}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
