"""Part 3 Tasks 6.4-6.5: model monitoring and alerting simulation.

Treats 2018 as "production" for a model trained on data up to 2017 and checks:

* feature distribution drift  Population Stability Index (PSI) and KS test,
                               2018 vs the same months (Jan-Sep) of 2017, so
                               that seasonality is not mistaken for drift
* prediction error drift       monthly MAE in 2018 vs the overall 2018 test MAE

Each check returns PASS / Normal or ALERT / Requires investigation. Two
scenarios are evaluated:

1. actual 2018 data
2. a simulated incident: a 30% traffic surge after 1 June 2018 (for example a
   detour onto I-94 after a nearby road closure), to show the alert firing

Outputs: results/monitoring_report.md, results/monitoring_status.json and a
dashboard figure figures/monitoring_dashboard.png.

    python -m part3_machine_learning.monitoring
"""

import json
import logging
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

from common import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from part2_python.visualizations import apply_style, subtitle, BLUE, ORANGE, TEXT_2, MUTED  # noqa: E402
from part3_machine_learning import ml_common as mc  # noqa: E402

logger = logging.getLogger(__name__)

DRIFT_FEATURES = ["temp_c", "clouds_all", "rain_1h_log", "weather_severity", "traffic_volume"]
PSI_WARN, PSI_ALERT = 0.10, 0.25       # common rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major
ERROR_RATIO_ALERT = 1.25               # a month whose MAE is 25% above the baseline triggers an alert
SURGE = 1.30


def validation_baseline(df: pd.DataFrame, model, features: list) -> float:
    """Validation MAE of the deployed model type, from supervised.py's frozen selection.

    Read from results/model_selection.json so monitoring and model selection use
    the same number. If the file is missing, it is recomputed the same way.
    """
    path = mc.RESULTS_DIR / "model_selection.json"
    try:
        selections = json.loads(path.read_text(encoding="utf-8"))["selections"]
        reg = next(s for s in selections if s["task"] == "regression")
        mae = float(reg["validation_scores"][reg["chosen"]]["MAE"])
        logger.info("Baseline MAE %.1f: validation MAE (2017) of the chosen model, read from %s",
                    mae, path.name)
        return mae
    except (FileNotFoundError, KeyError, StopIteration, json.JSONDecodeError):
        logger.warning("%s not readable; recomputing the baseline (train to 2016, validate on 2017)", path.name)
    pre_train = df[df["date_time"] < pd.Timestamp(config.VALID_START)]
    validation = df[(df["date_time"] >= pd.Timestamp(config.VALID_START))
                    & (df["date_time"] < pd.Timestamp(config.TEST_START))]
    validator = clone(model).fit(pre_train[features], pre_train["traffic_volume"])
    mae = mean_absolute_error(validation["traffic_volume"], validator.predict(validation[features]))
    logger.info("Baseline MAE %.1f (model trained to 2016, validated on 2017)", mae)
    return mae


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with bins from reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # almost constant feature: use its distinct values
        edges = np.unique(np.concatenate([reference.unique(), current.unique()]))
        edges = np.append(edges, edges[-1] + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in DRIFT_FEATURES:
        value = psi(reference[f], current[f])
        ks = stats.ks_2samp(reference[f], current[f])
        level = "ALERT" if value > PSI_ALERT else ("WATCH" if value > PSI_WARN else "PASS")
        rows.append({"feature": f, "PSI": value, "KS_statistic": ks.statistic, "KS_p_value": ks.pvalue,
                     "reference_mean": reference[f].mean(), "current_mean": current[f].mean(), "status": level})
        logger.debug("PSI %s = %.3f (KS %.3f)", f, value, ks.statistic)
    return pd.DataFrame(rows)


def error_drift(current: pd.DataFrame, pred: np.ndarray, baseline_mae: float) -> pd.DataFrame:
    df = current.assign(abs_error=np.abs(current["traffic_volume"].to_numpy() - pred))
    monthly = df.groupby(df["date_time"].dt.to_period("M"))["abs_error"].mean().rename("MAE").to_frame()
    monthly["ratio_to_baseline"] = monthly["MAE"] / baseline_mae
    monthly["status"] = np.where(monthly["ratio_to_baseline"] > ERROR_RATIO_ALERT, "ALERT", "PASS")
    monthly.index = monthly.index.astype(str)
    return monthly.reset_index().rename(columns={"date_time": "month"})


def overall_status(drift: pd.DataFrame, errors: pd.DataFrame) -> tuple:
    reasons = [f"{r.feature} PSI {r.PSI:.2f}" for r in drift.itertuples() if r.status == "ALERT"]
    reasons += [f"{r.month} MAE x{r.ratio_to_baseline:.2f}" for r in errors.itertuples() if r.status == "ALERT"]
    if reasons:
        return "ALERT / Requires investigation", reasons
    return "PASS / Normal", []


def run_scenario(name: str, reference: pd.DataFrame, current: pd.DataFrame, model, features,
                 baseline_mae: float) -> dict:
    pred = model.predict(current[features])
    drift = feature_drift(reference, current)
    errors = error_drift(current, pred, baseline_mae)
    status, reasons = overall_status(drift, errors)
    if reasons:
        logger.warning("[%s] %s: %s", name, status, "; ".join(reasons))
    else:
        logger.info("[%s] %s: no feature drift above PSI %.2f and no month above %.2fx baseline MAE",
                    name, status, PSI_ALERT, ERROR_RATIO_ALERT)
    return {"scenario": name, "status": status, "reasons": reasons, "drift": drift, "errors": errors}


def plot_dashboard(results: list, baseline_mae: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw={"hspace": 0.6, "wspace": 0.3})
    for col, r in enumerate(results):
        ax = axes[0, col]
        d = r["drift"]
        colors = [ORANGE if s == "ALERT" else BLUE for s in d["status"]]
        ax.barh(d["feature"], d["PSI"], color=colors, height=0.6)
        ax.axvline(PSI_ALERT, color=TEXT_2, linestyle="--", linewidth=1)
        ax.axvline(PSI_WARN, color=MUTED, linestyle=":", linewidth=1)
        ax.set(xlabel="PSI (dashed line = alert at 0.25)", xlim=(0, max(0.35, d["PSI"].max() * 1.15)))
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{r['scenario']}: {r['status']}", pad=22, fontsize=11.5,
                     color=ORANGE if r["status"].startswith("ALERT") else TEXT_2)
        subtitle(ax, "Feature drift vs Jan-Sep 2017")

        ax = axes[1, col]
        e = r["errors"]
        ax.plot(e["month"], e["MAE"], color=BLUE, marker="o")
        ax.axhline(baseline_mae * ERROR_RATIO_ALERT, color=TEXT_2, linestyle="--", linewidth=1)
        ax.annotate("alert threshold", (len(e) - 1, baseline_mae * ERROR_RATIO_ALERT), xytext=(0, 4),
                    textcoords="offset points", ha="right", color=TEXT_2, fontsize=9)
        for x, (m, s) in enumerate(zip(e["MAE"], e["status"])):
            if s == "ALERT":
                ax.plot(x, m, "o", color=ORANGE, markersize=8)
        ax.set(ylabel="MAE (vehicles per hour)", ylim=(0, max(e["MAE"].max(), baseline_mae * 1.4) * 1.15))
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Monthly prediction error", pad=10, fontsize=11.5)
    fig.suptitle("Model monitoring dashboard: traffic volume model in 2018", x=0.06, ha="left",
                 fontsize=14, fontweight="bold")
    path = mc.FIGURES_DIR / "monitoring_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path.relative_to(config.ROOT_DIR))


def main() -> int:
    setup_logging(mc.LOG_FILE, append=True)
    mc.ensure_dirs()
    apply_style()
    try:
        bundle = joblib.load(mc.MODELS_DIR / "regression_model.joblib")
        model, features = bundle["model"], bundle["features"]
        df = mc.load_model_data()
    except FileNotFoundError:
        logger.error("Model or data missing. Run supervised.py first.", exc_info=True)
        return 1

    reference = df[(df["date_time"].dt.year == 2017) & (df["date_time"].dt.month <= 9)]
    current = df[df["date_time"] >= pd.Timestamp(config.TEST_START)]

    # Baseline error = what we could have known BEFORE deployment: the chosen model
    # type trained up to 2016 and validated on 2017 (D17, D19). Using 2018's own
    # error as the baseline would be circular.
    baseline_mae = validation_baseline(df, model, features)
    logger.info("Reference: %d hours (Jan-Sep 2017). Current: %d hours (Jan-Sep 2018)", len(reference), len(current))

    surge = current.copy()
    after = surge["date_time"] >= pd.Timestamp("2018-06-01")
    surge.loc[after, "traffic_volume"] = (surge.loc[after, "traffic_volume"] * SURGE).round()
    logger.info("Simulated incident: traffic x%.2f for %d hours from 2018-06-01", SURGE, int(after.sum()))

    results = [run_scenario("Actual 2018", reference, current, model, features, baseline_mae),
               run_scenario("Simulated surge", reference, surge, model, features, baseline_mae)]
    plot_dashboard(results, baseline_mae)

    md = ["# Model monitoring report", "",
          f"Model: gradient boosting regressor trained on 2012-2017. Baseline MAE = {baseline_mae:.1f} "
          "(same model type trained to 2016 and validated on 2017, i.e. the error expected at deployment).",
          f"Feature drift: PSI vs Jan-Sep 2017, alert above {PSI_ALERT}. "
          f"Error drift: alert when a month's MAE exceeds {ERROR_RATIO_ALERT}x the baseline.", ""]
    for r in results:
        md += [f"## {r['scenario']}: **{r['status']}**", ""]
        if r["reasons"]:
            md += ["Reasons: " + ", ".join(r["reasons"]), ""]
        md += ["### Feature drift", "", r["drift"].round(3).to_markdown(index=False), "",
               "### Monthly prediction error", "", r["errors"].round(3).to_markdown(index=False), ""]
    (mc.RESULTS_DIR / "monitoring_report.md").write_text("\n".join(md), encoding="utf-8")
    (mc.RESULTS_DIR / "monitoring_status.json").write_text(json.dumps(
        [{"scenario": r["scenario"], "status": r["status"], "reasons": r["reasons"]} for r in results],
        indent=2), encoding="utf-8")
    logger.info("Saved monitoring report and status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
