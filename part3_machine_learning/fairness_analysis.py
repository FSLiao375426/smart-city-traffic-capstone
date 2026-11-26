"""Part 3 Task 7: how evenly are model errors distributed?

Breaks down the 2018 test errors of the deployed models by condition:
weather, time of day, day type, holiday, and month. Uneven errors mean some
travellers or situations are served worse by the system, which the bias and
fairness report discusses.

Weather is shown two ways (D21): by the most severe label of the hour
(weather_main, groups do not overlap) and by every label present (an hour
with rain and mist counts in both groups, so those groups overlap).
A rate whose denominator is zero is reported as missing, never as 0% or 100%.

    python -m part3_machine_learning.fairness_analysis
"""

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from common import config
from common.logging_setup import setup_logging
from part3_machine_learning import ml_common as mc

logger = logging.getLogger(__name__)
MIN_HOURS = 30          # groups smaller than this are reported but flagged as unreliable


def regression_errors(test: pd.DataFrame, by: str) -> pd.DataFrame:
    g = test.groupby(by, observed=True)
    out = pd.DataFrame({
        "hours": g.size(),
        "avg_volume": g["traffic_volume"].mean(),
        "MAE": g["abs_error"].mean(),
        "bias_mean_error": g["error"].mean(),          # positive = model over-predicts
    })
    out["MAE_pct_of_volume"] = out["MAE"] / out["avg_volume"]
    out["MAE_vs_overall"] = out["MAE"] / test["abs_error"].mean()
    out["reliable"] = out["hours"] >= MIN_HOURS
    return out.reset_index()


def regression_errors_any_label(test: pd.DataFrame) -> pd.DataFrame:
    """Errors for every hour that had a weather label, whatever else it had (groups overlap)."""
    rows = []
    for w in sorted(config.WEATHER_SEVERITY):
        g = test[test[f"wx_{w}"] == 1]
        if g.empty:
            continue
        rows.append({"weather_label_present": w, "hours": len(g), "avg_volume": g["traffic_volume"].mean(),
                     "MAE": g["abs_error"].mean(), "bias_mean_error": g["error"].mean(),
                     "MAE_vs_overall": g["abs_error"].mean() / test["abs_error"].mean(),
                     "reliable": len(g) >= MIN_HOURS})
    return pd.DataFrame(rows)


def safe_rate(numerator: int, denominator: int) -> float:
    """numerator / denominator, or NaN when there is nothing to divide by."""
    return numerator / denominator if denominator else np.nan


def classification_errors(test: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in test.groupby(by, observed=True):
        actual, pred = g["high_risk"] == 1, g["risk_pred"] == 1
        tp, fp = int((actual & pred).sum()), int((~actual & pred).sum())
        rows.append({by: key, "hours": len(g), "positives": int(actual.sum()),
                     "predicted_positives": int(pred.sum()),
                     "recall": safe_rate(tp, int(actual.sum())),
                     "precision": safe_rate(tp, int(pred.sum())),
                     "false_positive_rate": safe_rate(fp, int((~actual).sum())),
                     "reliable": len(g) >= MIN_HOURS})
    return pd.DataFrame(rows)


def main() -> int:
    setup_logging(mc.LOG_FILE, append=True)
    try:
        reg = joblib.load(mc.MODELS_DIR / "regression_model.joblib")
        clf = joblib.load(mc.MODELS_DIR / "classification_model.joblib")
        _, test = mc.time_split(mc.load_model_data())
    except FileNotFoundError:
        logger.error("Models missing. Run supervised.py first.", exc_info=True)
        return 1

    test = test.copy()
    test["pred"] = reg["model"].predict(test[reg["features"]])
    test["error"] = test["pred"] - test["traffic_volume"]
    test["abs_error"] = test["error"].abs()
    test["risk_pred"] = clf["model"].predict(test[clf["features"]])
    test["holiday_flag"] = np.where(test["is_holiday"] == 1, "Holiday", "Not holiday")
    test["month_name"] = test["date_time"].dt.strftime("%m %b")
    test["visibility"] = np.where(test["is_low_visibility"] == 1, "Low visibility", "Normal visibility")
    # Snow and low visibility often occur together (D21); split them to see which drives the error
    snow = np.where(test["wx_Snow"] == 1, "Snow", "No snow")
    test["snow_and_visibility"] = pd.Series(snow, index=test.index) + " + " + test["visibility"].str.lower()
    logger.info("Overall 2018 regression MAE %.1f over %d hours", test["abs_error"].mean(), len(test))

    groupings = ["weather_main", "visibility", "snow_and_visibility", "time_of_day", "day_type",
                 "holiday_flag", "month_name"]
    reg_tables = {by: regression_errors(test, by) for by in groupings}
    any_label = regression_errors_any_label(test)
    clf_tables = {by: classification_errors(test, by)
                  for by in ["weather_main", "visibility", "time_of_day", "day_type"]}

    for by, t in reg_tables.items():
        worst = t[t["reliable"]].sort_values("MAE_vs_overall", ascending=False).iloc[0]
        logger.info("Largest error by %s: %s (MAE %.0f, %.2fx overall, %d hours)",
                    by, worst[by], worst["MAE"], worst["MAE_vs_overall"], worst["hours"])
        if worst["MAE_vs_overall"] > 1.5:
            logger.warning("Errors are uneven across %s: %s has %.1fx the overall MAE",
                           by, worst[by], worst["MAE_vs_overall"])

    md = ["# Error distribution across conditions (2018 test set)", "",
          f"Overall regression MAE: {test['abs_error'].mean():.1f} vehicles per hour. "
          f"Groups with fewer than {MIN_HOURS} hours are marked unreliable. "
          "bias_mean_error > 0 means the model over-predicts traffic for that group.", ""]
    for by, t in reg_tables.items():
        md += [f"## Regression error by {by}", "", t.round(3).to_markdown(index=False), ""]
        if by == "weather_main":
            md += ["## Regression error by every weather label present (groups overlap)", "",
                   any_label.round(3).to_markdown(index=False), ""]
    for by, t in clf_tables.items():
        md += [f"## Proxy risk classifier by {by}", "",
               t.round(3).astype(object).where(t.notna(), None).to_markdown(index=False, missingval="n/a"), ""]
    (mc.RESULTS_DIR / "fairness_error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    (mc.RESULTS_DIR / "fairness_error_analysis.json").write_text(json.dumps(
        {**{by: t.round(3).to_dict("records") for by, t in reg_tables.items()},
         "weather_label_present": any_label.round(3).to_dict("records"),
         "classifier": {by: t.round(3).astype(object).where(t.notna(), None).to_dict("records")
                        for by, t in clf_tables.items()}},
        indent=2, default=str), encoding="utf-8")
    logger.info("Saved error distribution analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
