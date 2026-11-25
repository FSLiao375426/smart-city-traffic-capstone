"""Part 3 Task 7: how evenly are model errors distributed?

Breaks down the 2018 test errors of the deployed models by condition:
weather, time of day, day type, holiday, and month. Uneven errors mean some
travellers or situations are served worse by the system, which the bias and
fairness report discusses.

    python -m part3_machine_learning.fairness_analysis
"""

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

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


def classification_errors(test: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in test.groupby(by, observed=True):
        pos = int(g["high_risk"].sum())
        rows.append({by: key, "hours": len(g), "positives": pos,
                     "recall": recall_score(g["high_risk"], g["risk_pred"], zero_division=np.nan) if pos else np.nan,
                     "precision": precision_score(g["high_risk"], g["risk_pred"], zero_division=np.nan),
                     "false_positive_rate": ((g["risk_pred"] == 1) & (g["high_risk"] == 0)).sum()
                     / max(1, (g["high_risk"] == 0).sum())})
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
    logger.info("Overall 2018 regression MAE %.1f over %d hours", test["abs_error"].mean(), len(test))

    groupings = ["weather_main", "time_of_day", "day_type", "holiday_flag", "month_name"]
    reg_tables = {by: regression_errors(test, by) for by in groupings}
    clf_tables = {by: classification_errors(test, by) for by in ["weather_main", "time_of_day", "day_type"]}

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
    for by, t in clf_tables.items():
        md += [f"## Proxy risk classifier by {by}", "", t.round(3).to_markdown(index=False), ""]
    (mc.RESULTS_DIR / "fairness_error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    (mc.RESULTS_DIR / "fairness_error_analysis.json").write_text(json.dumps(
        {by: t.round(3).to_dict("records") for by, t in reg_tables.items()}, indent=2, default=str), encoding="utf-8")
    logger.info("Saved error distribution analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
