"""Part 1 Tasks 2 and 3: descriptive statistics, correlation and probability.

Reads traffic_clean from the SQLite database built by load_sqlite.py, so the
numbers come from exactly the same table as the SQL analysis.

Definitions (common/config.py):
    Congestion (C)    traffic_volume > 5,500
    Clear weather     weather_main == "Clear"
    Cloudy weather    weather_main == "Clouds"
    High temperature  temp > 292 K (about 18.9 C)

Run from the repository root:

    python -m part1_data_analytics.stats_probability
"""

import json
import logging
import math
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import config
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

PART1_DIR = config.ROOT_DIR / "part1_data_analytics"
RESULTS_DIR = PART1_DIR / "results"
LOG_FILE = PART1_DIR / "part1.log"


def load_from_sqlite() -> pd.DataFrame:
    with sqlite3.connect(config.SQLITE_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM traffic_clean", conn, parse_dates=["date_time"])
    logger.info("Loaded traffic_clean from SQLite: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Task 2.1  Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_stats(volume: pd.Series) -> dict:
    result = {
        "n_hours": int(volume.count()),
        "mean": float(volume.mean()),
        "median": float(volume.median()),
        "std": float(volume.std()),            # sample standard deviation (n - 1)
        "variance": float(volume.var()),       # sample variance (n - 1)
        "min": int(volume.min()),
        "max": int(volume.max()),
        "range": int(volume.max() - volume.min()),
        "q1": float(volume.quantile(0.25)),
        "q3": float(volume.quantile(0.75)),
        "coefficient_of_variation": float(volume.std() / volume.mean()),
        "skewness": float(volume.skew()),
    }
    logger.info("Traffic volume: mean %.1f, median %.1f, std %.1f, range %d",
                result["mean"], result["median"], result["std"], result["range"])
    return result


# ---------------------------------------------------------------------------
# Task 2.2  Correlation (and why it is not causation)
# ---------------------------------------------------------------------------
def correlation_analysis(df: pd.DataFrame) -> dict:
    r, p = stats.pearsonr(df["temp"], df["traffic_volume"])
    rho, _ = stats.spearmanr(df["temp"], df["traffic_volume"])

    # Confounder check: hour of day drives both temperature and traffic.
    # Remove each hour's average from both variables, then correlate what is left.
    hour = df["date_time"].dt.hour
    temp_resid = df["temp"] - df.groupby(hour)["temp"].transform("mean")
    vol_resid = df["traffic_volume"] - df.groupby(hour)["traffic_volume"].transform("mean")
    r_within_hour, _ = stats.pearsonr(temp_resid, vol_resid)

    r_temp_hour = df["temp"].corr(df.groupby(hour)["temp"].transform("mean"))
    result = {
        "pearson_r": float(r),
        "pearson_p_value": float(p),
        "r_squared": float(r ** 2),
        "spearman_rho": float(rho),
        "r_after_removing_hour_of_day": float(r_within_hour),
        "r_temp_vs_hourly_mean_temp": float(r_temp_hour),
    }
    logger.info("Pearson r(temp, volume) = %.3f (r^2 = %.3f); after removing hour-of-day effect r = %.3f",
                r, r ** 2, r_within_hour)
    return result


# ---------------------------------------------------------------------------
# Task 3  Probability and congestion
# ---------------------------------------------------------------------------
def probability_analysis(df: pd.DataFrame) -> dict:
    n = len(df)
    C = df["traffic_volume"] > config.CONGESTION_THRESHOLD
    clear = df["weather_main"] == "Clear"
    clouds = df["weather_main"] == "Clouds"
    hot = df["temp"] > config.HIGH_TEMP_K

    p_c = C.mean()
    p_clear = clear.mean()
    p_c_and_clear = (C & clear).mean()
    p_clear_given_c = (C & clear).sum() / C.sum()
    p_hot_given_c = (C & hot).sum() / C.sum()
    p_hot = hot.mean()
    p_c_given_clear = (C & clear).sum() / clear.sum()
    p_c_given_clouds = (C & clouds).sum() / clouds.sum()

    # Independence: compare P(C and Clear) with P(C) * P(Clear)
    expected_if_independent = p_c * p_clear
    ratio = p_c_and_clear / expected_if_independent
    table = pd.crosstab(C, clear)
    chi2, chi_p, _, _ = stats.chi2_contingency(table)

    # Odds ratio of congestion, clear vs cloudy, with a 95% confidence interval (Woolf method)
    a = int((C & clear).sum());  b = int((~C & clear).sum())
    c = int((C & clouds).sum()); d = int((~C & clouds).sum())
    odds_clear, odds_clouds = a / b, c / d
    odds_ratio = odds_clear / odds_clouds
    se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci = (math.exp(math.log(odds_ratio) - 1.96 * se_log_or), math.exp(math.log(odds_ratio) + 1.96 * se_log_or))

    # Confounder check: clear skies are more common at night, when congestion never
    # happens. A Mantel-Haenszel odds ratio compares clear vs cloudy WITHIN each hour
    # of the day, then pools the hours, so time of day can no longer explain the gap.
    hour = df["date_time"].dt.hour
    num = den = 0.0
    for h in range(24):
        m = hour == h
        ai = int((C & clear & m).sum()); bi = int((~C & clear & m).sum())
        ci_ = int((C & clouds & m).sum()); di = int((~C & clouds & m).sum())
        ni = ai + bi + ci_ + di
        if ni:
            num += ai * di / ni
            den += bi * ci_ / ni
    odds_ratio_mh = num / den if den else float("nan")
    peak = hour.isin(config.PEAK_HOURS)
    p_c_given_clear_peak = (C & clear & peak).sum() / (clear & peak).sum()
    p_c_given_clouds_peak = (C & clouds & peak).sum() / (clouds & peak).sum()
    logger.info("Odds ratio adjusted for hour of day (Mantel-Haenszel) = %.3f", odds_ratio_mh)
    logger.debug("Share of clear hours: night %.3f, day %.3f",
                 clear[~hour.between(6, 18)].mean(), clear[hour.between(6, 18)].mean())

    result = {
        "n_hours": n,
        "n_congested": int(C.sum()),
        "P(Congestion)": p_c,
        "P(Clear)": p_clear,
        "P(Congestion AND Clear)": p_c_and_clear,
        "P(Clear | Congestion)": p_clear_given_c,
        "P(High temp)": p_hot,
        "P(High temp | Congestion)": p_hot_given_c,
        "P(Congestion | Clear)": p_c_given_clear,
        "P(Congestion | Clouds)": p_c_given_clouds,
        "P(C) x P(Clear) if independent": expected_if_independent,
        "observed / expected ratio": ratio,
        "chi_square": float(chi2),
        "chi_square_p_value": float(chi_p),
        "contingency_clear_vs_clouds": {"C&Clear": a, "notC&Clear": b, "C&Clouds": c, "notC&Clouds": d},
        "odds_congestion_clear": odds_clear,
        "odds_congestion_clouds": odds_clouds,
        "odds_ratio_clear_vs_clouds": odds_ratio,
        "odds_ratio_95ci": list(ci),
        "odds_ratio_adjusted_for_hour (Mantel-Haenszel)": odds_ratio_mh,
        "P(Congestion | Clear, peak hours)": p_c_given_clear_peak,
        "P(Congestion | Clouds, peak hours)": p_c_given_clouds_peak,
    }
    logger.info("P(C) = %.4f, P(Clear) = %.4f, P(C and Clear) = %.4f vs %.4f if independent",
                p_c, p_clear, p_c_and_clear, expected_if_independent)
    logger.info("Odds ratio of congestion, clear vs cloudy = %.3f (95%% CI %.3f-%.3f)", odds_ratio, *ci)
    return result


def write_report(desc: dict, corr: dict, prob: dict) -> None:
    f = lambda x: f"{x:.4f}"  # noqa: E731
    lines = [
        "# Part 1 statistics and probability results",
        "",
        "Generated by `stats_probability.py` from the `traffic_clean` table (one row per hour).",
        "",
        "## Task 2.1 Traffic volume statistics",
        "",
        "| Statistic | Value |",
        "| --- | --- |",
    ]
    for k in ["n_hours", "mean", "median", "std", "variance", "min", "max", "range", "q1", "q3",
              "coefficient_of_variation", "skewness"]:
        v = desc[k]
        lines.append(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v:,} |")

    lines += ["", "## Task 2.2 Correlation: temperature vs traffic volume", "", "| Measure | Value |", "| --- | --- |"]
    for k, v in corr.items():
        lines.append(f"| {k} | {v:.4g} |")

    lines += ["", "## Task 3 Probability", "", "| Quantity | Value |", "| --- | --- |"]
    for k, v in prob.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {f(v)} |")
        elif isinstance(v, list):
            lines.append(f"| {k} | {v[0]:.3f} to {v[1]:.3f} |")
        else:
            lines.append(f"| {k} | {v} |")

    out = RESULTS_DIR / "stats_probability.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "stats_probability.json").write_text(
        json.dumps({"descriptive": desc, "correlation": corr, "probability": prob}, indent=2, default=float),
        encoding="utf-8",
    )
    logger.info("Saved statistics and probability results to %s", out.relative_to(config.ROOT_DIR))


def main() -> int:
    setup_logging(LOG_FILE, append=True)  # continues the log started by load_sqlite
    if not config.SQLITE_DB.exists():
        logger.error("Database not found. Run 'python -m part1_data_analytics.load_sqlite' first.")
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = load_from_sqlite()
        desc = descriptive_stats(df["traffic_volume"])
        corr = correlation_analysis(df)
        prob = probability_analysis(df)
        write_report(desc, corr, prob)
    except (sqlite3.Error, KeyError, ValueError, ZeroDivisionError):
        logger.error("Statistics and probability analysis did not complete", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
