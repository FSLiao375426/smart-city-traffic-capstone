"""Part 3 Task 2: unsupervised learning.

K-means clustering   groups hours by time of day, day type, weather severity
                     and traffic volume, then describes each group in plain words
Association rules    Apriori on discretised time, day type, weather and
                     temperature, keeping rules that predict a congestion level

Run from the repository root:

    python -m part3_machine_learning.unsupervised
"""

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlxtend.frequent_patterns import apriori, association_rules  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import config  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from part2_python.visualizations import apply_style, subtitle, TEXT_2  # noqa: E402
from part3_machine_learning import ml_common as mc  # noqa: E402

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "is_weekend", "weather_severity", "traffic_volume"]
K_RANGE = range(2, 9)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
def choose_k(X: np.ndarray) -> tuple:
    """Elbow (inertia) and silhouette for each k. Silhouette is computed on a
    fixed sample because it is quadratic in the number of points."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = rng.choice(len(X), size=min(6000, len(X)), replace=False)
    scores = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED).fit(X)
        sil = silhouette_score(X[sample], km.labels_[sample])
        scores.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.debug("k = %d: inertia %.0f, silhouette %.3f", k, km.inertia_, sil)
    scores = pd.DataFrame(scores)
    # Silhouette keeps creeping up as k grows, because extra clusters just split
    # existing groups. Choose the best k among sizes a person can still interpret.
    eligible = scores[scores["k"] <= config.KMEANS_MAX_K]
    best_k = int(eligible.loc[eligible["silhouette"].idxmax(), "k"])
    logger.info("Chose k = %d: highest silhouette (%.3f) for k <= %d. Larger k only splits existing groups "
                "(silhouette %.3f at k = %d)", best_k, eligible["silhouette"].max(), config.KMEANS_MAX_K,
                scores["silhouette"].iloc[-1], scores["k"].iloc[-1])
    return best_k, scores


def circular_mean_hour(group: pd.DataFrame) -> float:
    angle = np.arctan2(group["hour_sin"].mean(), group["hour_cos"].mean())
    return (angle % (2 * np.pi)) * 24 / (2 * np.pi)


def name_cluster(p: pd.Series) -> str:
    """Plain-language name built from the cluster profile.

    The time label uses the share of the cluster's hours that fall between
    06:00 and 18:59, because a circular mean hour can land on midday for a
    group that actually spans the whole day.
    """
    d = p["daytime_share"]
    period = "daytime" if d > 0.75 else ("off-peak hours" if d < 0.35 else "mixed hours")
    day = "Weekend" if p["weekend_share"] > 0.6 else ("Weekday" if p["weekend_share"] < 0.2 else "Any day")
    v = p["avg_volume"]
    level = ("congested" if p["congested_share"] > 0.3 else "light" if v < 1500 else "moderate" if v < 3000
             else "busy")
    weather = "rain or snow" if p["avg_weather_severity"] >= 4 else "fair weather"
    return f"{day} {period}, {level} traffic, {weather}"


def run_kmeans(df: pd.DataFrame) -> tuple:
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    best_k, scores = choose_k(X)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=config.RANDOM_SEED).fit(X)
    df = df.assign(cluster=km.labels_)

    rows = []
    for c, g in df.groupby("cluster"):
        rows.append({
            "cluster": c,
            "hours": len(g),
            "share_of_hours": len(g) / len(df),
            "typical_hour": round(circular_mean_hour(g), 1),
            "hour_range_p10_p90": f"{int(g['hour'].quantile(.1))}-{int(g['hour'].quantile(.9))}",
            "daytime_share": g["hour"].between(6, 18).mean(),
            "weekend_share": g["is_weekend"].mean(),
            "avg_volume": g["traffic_volume"].mean(),
            "congested_share": g["congested"].mean(),
            "avg_weather_severity": g["weather_severity"].mean(),
            "severe_or_lowvis_share": ((g["is_severe_weather"] + g["is_low_visibility"]) > 0).mean(),
            "top_weather": ", ".join(g["weather_main"].value_counts().head(3).index),
        })
    profile = pd.DataFrame(rows)
    profile["name"] = profile.apply(name_cluster, axis=1)
    profile = profile.sort_values("avg_volume").reset_index(drop=True)
    for _, p in profile.iterrows():
        logger.info("Cluster %d (%d hours): %s, avg volume %.0f, congested %.0f%%",
                    p["cluster"], p["hours"], p["name"], p["avg_volume"], 100 * p["congested_share"])
    return df, profile, scores


def plot_k_selection(scores: pd.DataFrame, best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(scores["k"], scores["inertia"], color=CATEGORICAL[0], marker="o")
    axes[0].set(xlabel="Number of clusters k", ylabel="Inertia (within-cluster spread)")
    axes[0].set_title("Elbow", pad=10)
    axes[1].plot(scores["k"], scores["silhouette"], color=CATEGORICAL[0], marker="o")
    axes[1].axvline(best_k, color=TEXT_2, linestyle="--", linewidth=1)
    axes[1].annotate(f"chosen k = {best_k}", (best_k, scores["silhouette"].max()), xytext=(6, -4),
                     textcoords="offset points", color=TEXT_2, fontsize=9)
    axes[1].set(xlabel="Number of clusters k", ylabel="Silhouette score")
    axes[1].set_title("Silhouette (higher is better)", pad=10)
    fig.tight_layout()
    save(fig, "kmeans_k_selection.png")


def plot_clusters(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    sample = df.sample(n=min(8000, len(df)), random_state=config.RANDOM_SEED)
    jitter = np.random.default_rng(config.RANDOM_SEED).uniform(-0.35, 0.35, len(sample))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in profile.iterrows():
        pts = sample[sample["cluster"] == p["cluster"]]
        ax.scatter(pts["hour"] + jitter[sample["cluster"] == p["cluster"]], pts["traffic_volume"], s=6,
                   alpha=0.45, color=CATEGORICAL[i % len(CATEGORICAL)], label=p["name"], linewidths=0)
    ax.set(xlabel="Hour of day", ylabel="Vehicles per hour", xticks=range(0, 24, 2))
    ax.set_title("K-means separates hours by time of day, traffic level and weather", pad=22)
    subtitle(ax, f"{len(sample):,} sampled hours coloured by cluster")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, markerscale=3, frameon=False)
    save(fig, "kmeans_clusters.png")


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------
def weather_group(w: str) -> str:
    if w in ("Rain", "Drizzle"):
        return "Rain"
    if w in config.LOW_VISIBILITY:
        return "Low visibility"
    if w in ("Thunderstorm", "Squall"):
        return "Storm"
    return w   # Clear, Clouds, Snow


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame({
        "time": "time=" + df["time_of_day"],
        "day": "day=" + df["day_type"],
        "weather": "weather=" + df["weather_main"].map(weather_group),
        "temp": "temp=" + pd.cut(df["temp_c"], [-100, 0, 20, 100], labels=["Below 0C", "0-20C", "Above 20C"]).astype(str),
        "holiday": np.where(df["is_holiday"] == 1, "holiday=Yes", "holiday=No"),  # "No" dropped below
        "congestion": "congestion=" + df["congestion_category"],
    })
    onehot = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    # "holiday=No" is true for 97% of hours, so it only creates near-duplicate rules
    onehot = onehot.drop(columns="holiday=No")
    logger.info("Association rules: %d transactions, %d items", len(onehot), onehot.shape[1])
    return onehot


def plain_language(row: pd.Series) -> str:
    conditions = " and ".join(sorted(i.split("=", 1)[0] + " is " + i.split("=", 1)[1] for i in row["antecedents"]))
    level = next(iter(row["consequents"])).split("=", 1)[1]
    return (f"When {conditions}, traffic is {level} {row['confidence']:.0%} of the time, "
            f"{row['lift']:.1f} times more often than usual.")


def run_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    onehot = build_transactions(df)
    frequent = apriori(onehot, min_support=config.APRIORI_MIN_SUPPORT, use_colnames=True)
    logger.debug("Frequent itemsets with support >= %.3f: %d", config.APRIORI_MIN_SUPPORT, len(frequent))
    rules = association_rules(frequent, metric="confidence", min_threshold=config.APRIORI_MIN_CONFIDENCE)

    is_congestion = lambda s: all(i.startswith("congestion=") for i in s)  # noqa: E731
    no_congestion = lambda s: not any(i.startswith("congestion=") for i in s)  # noqa: E731
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1 and is_congestion(s))
                  & rules["antecedents"].apply(no_congestion)]
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules["rule"] = rules.apply(
        lambda r: " AND ".join(sorted(r["antecedents"])) + " => " + next(iter(r["consequents"])), axis=1)
    rules["plain_language"] = rules.apply(plain_language, axis=1)
    logger.info("Kept %d rules that predict a congestion level (support >= %.3f, confidence >= %.2f)",
                len(rules), config.APRIORI_MIN_SUPPORT, config.APRIORI_MIN_CONFIDENCE)
    for _, r in rules.head(5).iterrows():
        logger.info("Rule: %s (support %.3f, confidence %.2f, lift %.2f)",
                    r["rule"], r["support"], r["confidence"], r["lift"])
    return rules[["rule", "support", "confidence", "lift", "plain_language"]]


def save(fig, name: str) -> None:
    path = mc.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path.relative_to(config.ROOT_DIR))


def main() -> int:
    setup_logging(mc.LOG_FILE, append=True)
    mc.ensure_dirs()
    apply_style()
    try:
        df = mc.load_model_data()
        df, profile, scores = run_kmeans(df)
        plot_k_selection(scores, int(profile.shape[0]))
        plot_clusters(df, profile)
        rules = run_association_rules(df)
    except FileNotFoundError:
        logger.error("Features not found. Run the Part 2 steps first.", exc_info=True)
        return 1
    except (ValueError, KeyError):
        logger.error("Unsupervised analysis did not complete", exc_info=True)
        return 1

    profile.to_csv(mc.RESULTS_DIR / "kmeans_cluster_profiles.csv", index=False)
    rules.to_csv(mc.RESULTS_DIR / "association_rules.csv", index=False)
    by_level = {lvl: rules[rules["rule"].str.endswith(f"congestion={lvl}")].head(3)
                for lvl in ["Severe", "High", "Medium", "Low"]}
    md = ["# Unsupervised learning results", "",
          "## K-means: choosing k", "", scores.round(3).to_markdown(index=False), "",
          "## K-means: cluster profiles", "",
          profile.drop(columns=["cluster"]).round(2).to_markdown(index=False), "",
          "## Association rules: top 10 by lift", "",
          rules.head(10).drop(columns="plain_language").round(3).to_markdown(index=False), "",
          "### In plain language", ""]
    md += [f"- {t}" for t in rules.head(10)["plain_language"]]
    md += ["", "## Strongest rule for each congestion level", ""]
    for lvl, r in by_level.items():
        md += [f"**{lvl}**", ""] + [f"- {t}" for t in r["plain_language"]] + [""]
    (mc.RESULTS_DIR / "unsupervised_results.md").write_text("\n".join(md), encoding="utf-8")
    (mc.RESULTS_DIR / "unsupervised_summary.json").write_text(json.dumps({
        "k": int(profile.shape[0]),
        "clusters": profile[["name", "hours", "avg_volume", "congested_share"]].round(3).to_dict("records"),
        "top_rules": rules.head(10)[["rule", "lift", "confidence"]].round(3).to_dict("records"),
    }, indent=2), encoding="utf-8")
    logger.info("Saved unsupervised results to %s", mc.RESULTS_DIR.relative_to(config.ROOT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
