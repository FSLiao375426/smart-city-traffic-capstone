"""Part 2 Task 4: mini traffic analytics command-line application.

Answers questions about I-94 westbound traffic from the processed dataset.

Run from the repository root (after feature_engineering):

    python -m part2_python.traffic_app query --datetime "2017-06-01 08:00"
    python -m part2_python.traffic_app day --date 2016-07-04
    python -m part2_python.traffic_app peaks --day-type weekday --top 5
    python -m part2_python.traffic_app compare
    python -m part2_python.traffic_app recommend --day-type weekday --earliest 6 --latest 20
    python -m part2_python.traffic_app recommend --day-type weekday --weather Snow --window 2

Every command and its arguments are logged at INFO to part2_python/app.log.
Invalid input (e.g. a malformed date) is logged at ERROR with a clear message,
never a raw traceback. print() is used only for the answers shown to the user.
"""

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from common import config
from common.data import load_features
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

APP_LOG = config.ROOT_DIR / "part2_python" / "app.log"
MIN_HOURS_FOR_WEATHER_PROFILE = 20   # per hour slot, below this the weather filter is not trusted


class UserInputError(Exception):
    """Raised for input the user can fix (bad date, unknown weather, empty range)."""


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------
def parse_datetime(text: str) -> pd.Timestamp:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H"):
        try:
            return pd.Timestamp(datetime.strptime(text.strip(), fmt))
        except ValueError:
            continue
    raise UserInputError(f"Invalid date/time '{text}'. Use the format YYYY-MM-DD HH:MM, e.g. 2017-06-01 08:00")


def parse_date(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(text.strip(), "%Y-%m-%d"))
    except ValueError:
        raise UserInputError(f"Invalid date '{text}'. Use the format YYYY-MM-DD, e.g. 2016-07-04") from None


def check_hour(value: int, name: str) -> int:
    if not 0 <= value <= 23:
        raise UserInputError(f"--{name} must be an hour between 0 and 23, got {value}")
    return value


def fmt_hour(h: int) -> str:
    return f"{h:02d}:00"


def typical_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly averages by day type, excluding flagged closure hours (D9)."""
    usable = df[df["traffic_anomaly"] == 0]
    return usable.groupby(["day_type", "hour"])["traffic_volume"].agg(["mean", "median", "count"])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_query(df: pd.DataFrame, args) -> None:
    """Traffic and weather at one specific hour, compared with a typical hour."""
    ts = parse_datetime(args.datetime).floor("h")
    row = df[df["date_time"] == ts]
    if row.empty:
        first, last = df["date_time"].min(), df["date_time"].max()
        if not first <= ts <= last:
            raise UserInputError(f"{ts:%Y-%m-%d %H:%M} is outside the data range ({first:%Y-%m-%d} to {last:%Y-%m-%d})")
        raise UserInputError(f"No record for {ts:%Y-%m-%d %H:%M}: the sensor has no data for that hour "
                             "(for example the Sep 2014 - May 2015 gap)")
    r = row.iloc[0]
    typical = typical_profile(df).loc[(r["day_type"], r["hour"]), "mean"]
    diff = (r["traffic_volume"] - typical) / typical * 100

    print(f"\n{ts:%A %d %B %Y, %H:%M}")
    print(f"  Traffic volume : {r['traffic_volume']:,} vehicles ({r['traffic_category']} traffic)")
    print(f"  Typical {r['day_type'].lower()} {fmt_hour(r['hour'])}: {typical:,.0f} vehicles "
          f"({diff:+.0f}% compared with typical)")
    print(f"  Weather        : {r['weather_main']} ({r['weather_description']}), {r['temp_c']:.1f} °C")
    if r["holiday"] != "None":
        print(f"  Holiday        : {r['holiday']}")
    if r["traffic_anomaly"] == 1:
        print("  Note           : flagged as an anomaly (likely road closure)")
    logger.info("query answered for %s: volume %d", ts, r["traffic_volume"])


def cmd_day(df: pd.DataFrame, args) -> None:
    """Weather and traffic summary for one calendar day."""
    day = parse_date(args.date)
    rows = df[df["date_time"].dt.normalize() == day].sort_values("date_time")
    if rows.empty:
        raise UserInputError(f"No records for {day:%Y-%m-%d}")
    peak = rows.loc[rows["traffic_volume"].idxmax()]

    print(f"\n{day:%A %d %B %Y}  ({len(rows)} hours recorded)")
    if rows["holiday"].iloc[0] != "None":
        print(f"  Holiday       : {rows['holiday'].iloc[0]}")
    print(f"  Total traffic : {rows['traffic_volume'].sum():,} vehicles")
    print(f"  Busiest hour  : {peak['date_time']:%H:%M} with {peak['traffic_volume']:,} vehicles")
    print(f"  Temperature   : {rows['temp_c'].min():.1f} to {rows['temp_c'].max():.1f} °C")
    print(f"  Weather       : {', '.join(rows['weather_main'].value_counts().index)}")
    print(f"  Congested hrs : {int(rows['congested'].sum())}")
    logger.info("day summary answered for %s: %d hours", day.date(), len(rows))


def cmd_peaks(df: pd.DataFrame, args) -> None:
    """Busiest hours of the day on average."""
    data = df[df["traffic_anomaly"] == 0]
    if args.day_type != "all":
        data = data[data["day_type"] == args.day_type.capitalize()]
    by_hour = data.groupby("hour").agg(avg=("traffic_volume", "mean"), congested=("congested", "mean"))
    top = by_hour.sort_values("avg", ascending=False).head(args.top)

    label = {"weekday": "on weekdays", "weekend": "at weekends", "all": "across all days"}[args.day_type]
    print(f"\nTop {args.top} busiest hours {label} (average vehicles per hour)")
    print(f"  {'Hour':<7}{'Average':>9}{'Congested':>12}")
    for hour, r in top.iterrows():
        print(f"  {fmt_hour(hour):<7}{r['avg']:>9,.0f}{r['congested']:>11.0%}")
    logger.info("peaks answered: top hour %s", fmt_hour(top.index[0]))


def cmd_compare(df: pd.DataFrame, args) -> None:
    """Weekday versus weekend traffic."""
    data = df[df["traffic_anomaly"] == 0]
    summary = data.groupby("day_type").agg(
        avg=("traffic_volume", "mean"), congested=("congested", "mean"), hours=("traffic_volume", "size"))
    profile = data.groupby(["day_type", "hour"])["traffic_volume"].mean().unstack(0)

    print("\nWeekday vs weekend")
    print(f"  {'':<22}{'Weekday':>10}{'Weekend':>10}")
    print(f"  {'Average per hour':<22}{summary.loc['Weekday', 'avg']:>10,.0f}{summary.loc['Weekend', 'avg']:>10,.0f}")
    print(f"  {'Hours congested':<22}{summary.loc['Weekday', 'congested']:>10.1%}"
          f"{summary.loc['Weekend', 'congested']:>10.1%}")
    print(f"  {'Busiest hour':<22}{fmt_hour(profile['Weekday'].idxmax()):>10}{fmt_hour(profile['Weekend'].idxmax()):>10}")
    print(f"  {'Quietest hour':<22}{fmt_hour(profile['Weekday'].idxmin()):>10}{fmt_hour(profile['Weekend'].idxmin()):>10}")
    gap = (profile["Weekday"] - profile["Weekend"])
    print(f"  Largest gap: {fmt_hour(gap.idxmax())}, weekdays carry {gap.max():,.0f} more vehicles per hour")
    logger.info("compare answered")


def cmd_recommend(df: pd.DataFrame, args) -> None:
    """Recommend the lowest-traffic departure window within the user's time range."""
    earliest, latest = check_hour(args.earliest, "earliest"), check_hour(args.latest, "latest")
    if latest - earliest + 1 < args.window:
        raise UserInputError(f"The range {fmt_hour(earliest)}-{fmt_hour(latest)} is shorter than a "
                             f"{args.window}-hour window")

    data = df[(df["traffic_anomaly"] == 0) & (df["day_type"] == args.day_type.capitalize())]
    note = ""
    if args.weather:
        known = sorted(df["weather_main"].unique())
        match = [w for w in known if w.lower() == args.weather.lower()]
        if not match:
            raise UserInputError(f"Unknown weather '{args.weather}'. Choose from: {', '.join(known)}")
        weather_data = data[data["weather_main"] == match[0]]
        slots = weather_data[weather_data["hour"].between(earliest, latest)].groupby("hour").size()
        if len(slots) == latest - earliest + 1 and slots.min() >= MIN_HOURS_FOR_WEATHER_PROFILE:
            data, note = weather_data, f" in {match[0]} conditions"
        else:
            logger.warning("Too little %s history (min %d hours per slot) for a weather-specific profile; "
                           "using all weather", match[0], int(slots.min()) if len(slots) else 0)
            note = f" (too little {match[0].lower()} history, so all weather is used)"

    by_hour = data.groupby("hour")["traffic_volume"].mean()
    windows = (by_hour.rolling(args.window).mean().shift(-(args.window - 1))
               .loc[earliest:latest - args.window + 1].dropna())
    best_start = int(windows.idxmin())
    worst_start = int(windows.idxmax())
    best_end = best_start + args.window

    print(f"\nRecommended travel window for a {args.day_type} journey{note}")
    print(f"  Best : {fmt_hour(best_start)}-{fmt_hour(best_end)}, about {windows.min():,.0f} vehicles per hour")
    print(f"  Avoid: {fmt_hour(worst_start)}-{fmt_hour(worst_start + args.window)}, "
          f"about {windows.max():,.0f} vehicles per hour")
    print(f"\n  For a {args.day_type} journey between {fmt_hour(earliest)} and {fmt_hour(latest + 1)}{note}, "
          f"consider travelling {fmt_hour(best_start)}-{fmt_hour(best_end)}, when historical traffic is typically "
          f"{1 - windows.min() / windows.max():.0%} lower than at the busiest time.")
    logger.info("recommend answered: best window starts %s", fmt_hour(best_start))


# ---------------------------------------------------------------------------
# Argument parser and entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic_app", description="I-94 traffic analytics from the command line")
    parser.add_argument("--debug", action="store_true", help="write DEBUG messages to app.log")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("query", help="traffic and weather at a specific hour")
    p.add_argument("--datetime", required=True, help='e.g. "2017-06-01 08:00"')
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("day", help="weather and traffic summary for one day")
    p.add_argument("--date", required=True, help="e.g. 2016-07-04")
    p.set_defaults(func=cmd_day)

    p = sub.add_parser("peaks", help="busiest hours on average")
    p.add_argument("--day-type", choices=["weekday", "weekend", "all"], default="all")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_peaks)

    p = sub.add_parser("compare", help="weekday versus weekend traffic")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("recommend", help="lowest-traffic travel window")
    p.add_argument("--day-type", choices=["weekday", "weekend"], default="weekday")
    p.add_argument("--earliest", type=int, default=6, help="earliest departure hour (0-23)")
    p.add_argument("--latest", type=int, default=20, help="latest hour the window may cover (0-23)")
    p.add_argument("--window", type=int, default=1, help="length of the travel window in hours")
    p.add_argument("--weather", help="optional weather condition, e.g. Snow, Rain, Clear")
    p.set_defaults(func=cmd_recommend)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(APP_LOG, debug=args.debug, append=True, console_level=logging.WARNING)

    arguments = {k: v for k, v in vars(args).items() if k not in ("func", "command", "debug")}
    logger.info("Command '%s' invoked with arguments %s", args.command, arguments)

    try:
        df = load_features()
    except FileNotFoundError:
        logger.error("Processed data not found. Run the pipeline and feature_engineering first.")
        return 1

    try:
        args.func(df, args)
    except UserInputError as exc:
        logger.error("Invalid input for '%s': %s", args.command, exc)
        return 2
    except Exception:  # last-resort guard: log details to the file, show a short message
        logger.error("Command '%s' failed unexpectedly", args.command, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
