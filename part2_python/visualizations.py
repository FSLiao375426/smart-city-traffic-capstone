"""Part 2 Task 3: Matplotlib visualisations of traffic patterns.

Reads data/processed/traffic_features.csv and saves PNG figures to
part2_python/figures/. Each saved figure is logged at INFO with its path.

Run from the repository root (after feature_engineering):

    python -m part2_python.visualizations
"""

import argparse
import logging
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import config  # noqa: E402
from common.data import load_features  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Palette: validated categorical slots and a one-hue sequential ramp (blue)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT_2, "ytick.labelcolor": TEXT_2,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except OSError:
        logger.error("Could not save figure %s", path, exc_info=True)
        raise
    finally:
        plt.close(fig)
    logger.info("Saved figure %s", path.relative_to(config.ROOT_DIR))


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9.5, va="bottom")


# ---------------------------------------------------------------------------
# Figure 1: hourly demand, weekday vs weekend
# ---------------------------------------------------------------------------
def fig_hourly_profile(df: pd.DataFrame) -> None:
    profile = df.groupby(["day_type", "hour"])["traffic_volume"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for day_type, color, dy in [("Weekday", BLUE, -7), ("Weekend", ORANGE, 7)]:
        ax.plot(profile.index, profile[day_type], color=color, marker="o", markersize=4, label=day_type)
        ax.annotate(day_type, (23, profile[day_type].iloc[-1]), xytext=(6, dy), textcoords="offset points",
                    color=TEXT, va="center", fontsize=10)
    ax.axhline(config.CONGESTION_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("Congestion threshold (5,500)", (23.5, config.CONGESTION_THRESHOLD), xytext=(0, 4),
                textcoords="offset points", color=TEXT_2, fontsize=9, ha="right")
    ax.set(xlabel="Hour of day", ylabel="Average vehicles per hour", xticks=range(0, 24, 2), xlim=(-0.5, 23.5),
           ylim=(0, 6500))
    ax.set_title("Weekdays have two commuter peaks, weekends have none", pad=22)
    subtitle(ax, "Average westbound I-94 traffic by hour, 2012-2018")
    ax.legend(loc="upper left")
    save(fig, "fig1_hourly_profile_weekday_weekend.png")


# ---------------------------------------------------------------------------
# Figure 2: distribution of traffic volume with category thresholds
# ---------------------------------------------------------------------------
def fig_distribution(df: pd.DataFrame) -> None:
    vol = df["traffic_volume"]
    q1, q2, q3 = vol.quantile([0.25, 0.5, 0.75])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(vol, bins=np.arange(0, 7500, 150), color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    for x, label in [(q1, "Q1"), (q2, "Median"), (q3, "Q3")]:
        ax.axvline(x, color=TEXT_2, linewidth=1)
        ax.annotate(f"{label}\n{x:,.0f}", (x, ax.get_ylim()[1]), xytext=(4, -4), textcoords="offset points",
                    va="top", color=TEXT, fontsize=9)
    ax.axvline(config.CONGESTION_THRESHOLD, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.annotate("Congested\n> 5,500", (config.CONGESTION_THRESHOLD, ax.get_ylim()[1] * 0.6), xytext=(4, 0),
                textcoords="offset points", color=TEXT, fontsize=9)
    ax.set(xlabel="Vehicles per hour", ylabel="Number of hours")
    ax.set_title("Hourly traffic falls into three clusters: night, shoulder hours and daytime", pad=22)
    subtitle(ax, f"Distribution of {len(vol):,} hourly counts; mean {vol.mean():,.0f}, median {q2:,.0f}")
    save(fig, "fig2_traffic_distribution.png")


# ---------------------------------------------------------------------------
# Figure 3: congestion by weather, all hours vs peak hours (confounding, D12)
# ---------------------------------------------------------------------------
def fig_weather_confounding(df: pd.DataFrame) -> None:
    common_weather = df["weather_main"].value_counts()
    common_weather = common_weather[common_weather >= 500].index
    sub = df[df["weather_main"].isin(common_weather)]
    peak = sub["hour"].isin(config.PEAK_HOURS)
    rates = pd.DataFrame({
        "All hours": sub.groupby("weather_main")["congested"].mean(),
        "Peak hours only": sub[peak].groupby("weather_main")["congested"].mean(),
    }).sort_values("Peak hours only") * 100

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(rates))
    h = 0.38
    ax.barh(y - h / 2, rates["All hours"], height=h - 0.04, color=BLUE, label="All hours")
    ax.barh(y + h / 2, rates["Peak hours only"], height=h - 0.04, color=ORANGE,
            label="Peak hours only (6-8, 15-17)")
    for yi, (a, p) in enumerate(zip(rates["All hours"], rates["Peak hours only"])):
        ax.text(a + 0.8, yi - h / 2, f"{a:.0f}%", va="center", color=TEXT_2, fontsize=8.5)
        ax.text(p + 0.8, yi + h / 2, f"{p:.0f}%", va="center", color=TEXT_2, fontsize=8.5)
    ax.set_yticks(y, rates.index)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Share of hours congested (%)", xlim=(0, 70))
    ax.set_title("Weather barely matters once time of day is held fixed, except snow", pad=22)
    subtitle(ax, "Congested = more than 5,500 vehicles per hour; weather types with at least 500 hours")
    ax.legend(loc="lower right")
    save(fig, "fig3_congestion_by_weather_peak_vs_all.png")


# ---------------------------------------------------------------------------
# Figure 4: heatmap of hour x day of week
# ---------------------------------------------------------------------------
def fig_heatmap(df: pd.DataFrame) -> None:
    grid = df.pivot_table(index="day_name", columns="hour", values="traffic_volume", aggfunc="mean")
    grid = grid.reindex(DAY_ORDER)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=BLUE_RAMP, vmin=0, vmax=6000)
    ax.set_xticks(range(24), range(24))
    ax.set_yticks(range(7), [d[:3] for d in DAY_ORDER])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Hour of day")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Average vehicles per hour", color=TEXT_2)
    cbar.outline.set_visible(False)
    ax.set_title("Weekdays peak at 7:00 and 16:00; weekends rise slowly to a midday plateau", pad=22)
    subtitle(ax, "Average traffic volume by day of week and hour, 2012-2018")
    save(fig, "fig4_heatmap_hour_by_day.png")


# ---------------------------------------------------------------------------
# Figure 5: temperature vs traffic (density, because 40k points overplot)
# ---------------------------------------------------------------------------
def fig_temperature(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hb = ax.hexbin(df["temp_c"], df["traffic_volume"], gridsize=45, cmap=BLUE_RAMP, mincnt=1, linewidths=0)
    r = df["temp_c"].corr(df["traffic_volume"])
    ax.grid(False)
    cbar = fig.colorbar(hb, ax=ax, pad=0.01)
    cbar.set_label("Hours in cell", color=TEXT_2)
    cbar.outline.set_visible(False)
    ax.set(xlabel="Temperature (°C)", ylabel="Vehicles per hour")
    ax.set_title(f"Temperature explains almost nothing on its own (r = {r:.2f})", pad=22)
    subtitle(ax, "Each hexagon counts hours; every temperature spans the full range of traffic")
    save(fig, "fig5_temperature_vs_traffic.png")


FIGURES = [fig_hourly_profile, fig_distribution, fig_weather_confounding, fig_heatmap, fig_temperature]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create traffic pattern figures")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    args = parser.parse_args(argv)
    setup_logging(config.LOG_FILE, debug=args.debug, append=True)  # continues pipeline.log

    try:
        df = load_features()
    except FileNotFoundError:
        logger.error("traffic_features.csv not found. Run 'python -m part2_python.feature_engineering' first.",
                     exc_info=True)
        return 1
    logger.info("Loaded features for visualisation: %d rows", len(df))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    failed = 0
    for make_figure in FIGURES:
        try:
            make_figure(df)
        except (KeyError, ValueError, OSError):
            failed += 1
            logger.error("Figure %s failed", make_figure.__name__, exc_info=True)
    if failed:
        logger.error("%d of %d figures failed", failed, len(FIGURES))
        return 1
    logger.info("All %d figures saved to %s", len(FIGURES), config.FIGURES_DIR.relative_to(config.ROOT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
