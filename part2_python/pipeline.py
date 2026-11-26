"""Part 2 Task 1: reproducible data-cleaning pipeline.

Turns the raw Metro Interstate Traffic Volume CSV into
data/processed/traffic_clean.csv, the single source of truth (D2) used by
every later step of the capstone.

Run from the repository root:

    python -m part2_python.pipeline            # normal run (INFO and above)
    python -m part2_python.pipeline --debug    # also show DEBUG values

Every cleaning step is its own function and writes its own log message,
so pipeline.log reads as a step-by-step audit trail.
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import config
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline cannot continue (bad file, bad schema...)."""


# ---------------------------------------------------------------------------
# Step 1: load
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, turning file and parsing problems into PipelineError."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise PipelineError(f"Raw file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise PipelineError(f"Raw file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise PipelineError(f"Raw file could not be parsed as CSV: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PipelineError(f"Raw file has an unexpected text encoding: {path}") from exc

    logger.info("Loaded raw file %s: %d rows, %d columns", path.name, df.shape[0], df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Step 2: validate schema (before any other processing)
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Stop the pipeline if any expected column is missing."""
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineError(f"Schema validation failed, missing columns: {missing}")

    extra = [c for c in df.columns if c not in config.EXPECTED_COLUMNS]
    if extra:
        logger.warning("Schema has %d unexpected column(s) that will be ignored: %s", len(extra), extra)

    logger.info("Schema validated: all %d expected columns present", len(config.EXPECTED_COLUMNS))


# ---------------------------------------------------------------------------
# Step 3: standardise categorical values
# ---------------------------------------------------------------------------
def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and fix capitalisation so the same category is spelled one way.

    weather_main  -> Title Case ("clouds " -> "Clouds")
    weather_description -> lower case
    holiday -> "None" for missing values (pandas reads the text "None" as NaN)
    """
    df = df.copy()

    rules = {
        "weather_main": lambda s: s.astype("string").str.strip().str.title(),
        "weather_description": lambda s: s.astype("string").str.strip().str.lower(),
        "holiday": lambda s: s.astype("string").str.strip().fillna("None"),
    }
    for column, rule in rules.items():
        before = df[column].astype("string")
        df[column] = rule(df[column])
        was_missing = before.isna()
        filled = int((was_missing & df[column].notna()).sum())
        reformatted = int((~was_missing & (before != df[column])).sum())
        if filled:
            logger.warning("Filled %d missing value(s) in '%s' with the label 'None'", filled, column)
        if reformatted:
            logger.warning("Standardised spacing or capitalisation of %d value(s) in '%s'", reformatted, column)
        if not filled and not reformatted:
            logger.info("Column '%s' already consistent, no values changed", column)

    unknown = sorted(set(df["weather_main"].dropna()) - set(config.WEATHER_SEVERITY))
    if unknown:
        raise PipelineError(f"Unknown weather_main categories with no severity rank: {unknown}")
    return df


# ---------------------------------------------------------------------------
# Step 4: parse and validate date/time
# ---------------------------------------------------------------------------
def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date_time to a real timestamp and drop rows that cannot be parsed."""
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    bad = int(df["date_time"].isna().sum())
    if bad:
        df = df.dropna(subset=["date_time"])
        logger.warning("Dropped %d row(s) with an unparseable date_time", bad)

    logger.info(
        "Parsed date_time for %d rows, range %s to %s",
        len(df), df["date_time"].min(), df["date_time"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 5: remove exact duplicate rows
# ---------------------------------------------------------------------------
def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are identical in every column."""
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates()
        logger.warning("Dropped %d exact duplicate row(s) (identical in every column)", n_dup)
    else:
        logger.info("No exact duplicate rows found")
    return df


# ---------------------------------------------------------------------------
# Step 6: collapse to one row per hour (D1)
# ---------------------------------------------------------------------------
def collapse_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per timestamp, choosing the most severe weather label.

    The raw file repeats an hour once per weather label observed in it.
    Traffic volume is identical across the copies, so only the weather differs.
    weather_label_count records how many labels the hour originally had.
    """
    inconsistent = int((df.groupby("date_time")["traffic_volume"].nunique() > 1).sum())
    if inconsistent:
        logger.warning("%d hour(s) have conflicting traffic_volume values across duplicates", inconsistent)

    label_count = df.groupby("date_time")["weather_main"].transform("size")
    df = df.assign(
        weather_label_count=label_count,
        _severity=df["weather_main"].map(config.WEATHER_SEVERITY),
    )

    # D21: remember every distinct label of the hour, most severe first
    labels_by_hour = (
        df.sort_values("_severity")
          .groupby("date_time")["weather_main"]
          .agg(lambda s: "|".join(dict.fromkeys(s)))
    )
    df["weather_labels"] = df["date_time"].map(labels_by_hour).astype("string")

    before = len(df)
    df = (
        df.sort_values(["date_time", "_severity"])          # most severe first within each hour
          .drop_duplicates(subset="date_time", keep="first")
          .drop(columns="_severity")
          .reset_index(drop=True)
    )
    removed = before - len(df)
    n_multi = int((df["weather_label_count"] > 1).sum())
    n_distinct_multi = int(df["weather_labels"].str.contains("|", regex=False).sum())

    if not df["date_time"].is_unique:
        raise PipelineError("Hourly collapse failed: date_time is still not unique")

    logger.warning(
        "Collapsed to hourly grain: removed %d row(s); %d hour(s) had several weather rows, "
        "kept the most severe weather_main for each",
        removed, n_multi,
    )
    logger.info(
        "Kept all weather labels in weather_labels: %d hour(s) have two or more different conditions",
        n_distinct_multi,
    )
    logger.info("Hourly table has %d unique hours", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 7: impossible sensor values -> monthly median (D4)
# ---------------------------------------------------------------------------
def impute_with_monthly_median(df: pd.DataFrame, column: str, invalid: pd.Series, reason: str) -> pd.DataFrame:
    """Replace invalid values with the median of valid values in the same year-month.

    An explicit loop over months is used on purpose: each month gets its own
    median, because a single global value ignores Minnesota's large seasonal swing.
    If a month has no valid values, the median of the same calendar month
    across all years is used instead.
    """
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        logger.info("No invalid values found in '%s'", column)
        return df

    df = df.copy()
    year_month = df["date_time"].dt.to_period("M")
    calendar_month = df["date_time"].dt.month

    for period in sorted(year_month[invalid].unique()):
        in_month = year_month == period
        targets = in_month & invalid
        valid_values = df.loc[in_month & ~invalid, column]

        if len(valid_values) > 0:
            median = valid_values.median()
            source = f"{period} median"
        else:
            same_calendar_month = (calendar_month == period.month) & ~invalid
            median = df.loc[same_calendar_month, column].median()
            source = f"calendar-month {period.month} median (no valid data in {period})"

        df.loc[targets, column] = median
        logger.debug("'%s' %s: imputed %d value(s) with %s = %.2f", column, period, int(targets.sum()), source, median)

    logger.warning("Imputed %d value(s) in '%s' with monthly medians: %s", n_invalid, column, reason)
    return df


def impute_with_group_median(
    df: pd.DataFrame, column: str, invalid: pd.Series, group_column: str, reason: str
) -> pd.DataFrame:
    """Replace invalid values with the median of valid values in the same group.

    Used for rainfall (D11): rain depends on the weather condition far more
    than on the month. The 9,831 mm reading is labelled "very heavy rain",
    so a monthly median of 0 mm would contradict its own label.
    Groups with fewer than MIN_GROUP_SIZE valid values fall back to the
    monthly median.
    """
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        logger.info("No invalid values found in '%s'", column)
        return df

    df = df.copy()
    fallback_rows = pd.Series(False, index=df.index)

    for group in sorted(df.loc[invalid, group_column].unique()):
        in_group = df[group_column] == group
        targets = in_group & invalid
        valid_values = df.loc[in_group & ~invalid, column]

        if len(valid_values) >= config.MIN_GROUP_SIZE:
            median = valid_values.median()
            df.loc[targets, column] = median
            logger.debug(
                "'%s' group '%s': imputed %d value(s) with group median = %.2f (from %d valid hours)",
                column, group, int(targets.sum()), median, len(valid_values),
            )
        else:
            fallback_rows |= targets
            logger.debug(
                "'%s' group '%s' has only %d valid hours, falling back to monthly median",
                column, group, len(valid_values),
            )

    if fallback_rows.any():
        df = impute_with_monthly_median(df, column, fallback_rows, reason + " (fallback)")

    logger.warning(
        "Imputed %d value(s) in '%s' with the median of the same %s: %s",
        n_invalid, column, group_column, reason,
    )
    return df


def handle_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect physically impossible readings and repair the input features."""
    temp_bad = (df["temp"] < config.TEMP_MIN_K) | (df["temp"] > config.TEMP_MAX_K)
    df = impute_with_monthly_median(
        df, "temp", temp_bad,
        f"temperature outside {config.TEMP_MIN_K:.0f}-{config.TEMP_MAX_K:.0f} K (e.g. 0 K sensor errors)",
    )

    rain_bad = (df["rain_1h"] < 0) | (df["rain_1h"] > config.RAIN_MAX_MM)
    df = impute_with_group_median(
        df, "rain_1h", rain_bad, "weather_description",
        f"hourly rainfall outside 0-{config.RAIN_MAX_MM:.0f} mm (e.g. 9,831 mm)",
    )

    for column in ["snow_1h", "clouds_all"]:
        negative = df[column] < 0
        df = impute_with_monthly_median(df, column, negative, "negative values are impossible")

    return df


# ---------------------------------------------------------------------------
# Step 8: flag (not impute) anomalous traffic readings
# ---------------------------------------------------------------------------
def flag_traffic_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag near-zero daytime traffic instead of overwriting it.

    traffic_volume is the prediction TARGET, so it is never imputed:
    inventing target values would teach the models fabricated answers.
    Near-zero daytime counts cluster on a few summer weekends (for example
    2016-07-23), which looks like planned road closures rather than sensor
    noise. They are kept, flagged with traffic_anomaly = 1, and later steps
    decide whether to exclude them.
    """
    hour = df["date_time"].dt.hour
    daytime = hour.between(config.DAYTIME_START_HOUR, config.DAYTIME_END_HOUR)
    anomaly = daytime & (df["traffic_volume"] < config.MIN_DAYTIME_VOLUME)

    df = df.assign(traffic_anomaly=anomaly.astype(int))
    n = int(anomaly.sum())
    if n:
        days = sorted(df.loc[anomaly, "date_time"].dt.date.astype(str).unique())
        logger.warning(
            "Flagged %d daytime hour(s) with traffic_volume < %d as traffic_anomaly "
            "(likely road closures, kept but flagged) on %d day(s): %s",
            n, config.MIN_DAYTIME_VOLUME, len(days), ", ".join(days),
        )
    else:
        logger.info("No anomalous daytime traffic readings found")
    return df


# ---------------------------------------------------------------------------
# Step 9: extend the holiday label to the whole day (D3)
# ---------------------------------------------------------------------------
def extend_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Copy each holiday name from its 00:00 row to every hour of that date."""
    df = df.copy()
    before = int((df["holiday"] != "None").sum())

    date = df["date_time"].dt.date
    holiday_by_date = df.loc[df["holiday"] != "None"].groupby(date)["holiday"].first()
    df["holiday"] = date.map(holiday_by_date).fillna("None").astype("string")
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    after = int(df["is_holiday"].sum())
    logger.warning(
        "Extended holiday labels from the midnight hour to the whole day: %d -> %d flagged hours across %d holiday dates",
        before, after, len(holiday_by_date),
    )
    return df


# ---------------------------------------------------------------------------
# Step 10: report time coverage gaps (L2: informative, nothing changed)
# ---------------------------------------------------------------------------
def report_coverage_gaps(df: pd.DataFrame) -> None:
    """Log gaps longer than the configured threshold so analyses can account for them."""
    gaps = df["date_time"].sort_values().diff()
    long_gaps = gaps[gaps > pd.Timedelta(days=config.GAP_WARNING_DAYS)]

    for idx, gap in long_gaps.items():
        end = df.loc[idx, "date_time"]
        logger.warning(
            "Coverage gap of %d days with no data, from %s to %s (yearly totals are affected)",
            gap.days, end - gap, end,
        )
    expected_hours = int((df["date_time"].max() - df["date_time"].min()) / pd.Timedelta(hours=1)) + 1
    logger.info(
        "Time coverage: %d of %d possible hours present (%.1f%%)",
        len(df), expected_hours, 100 * len(df) / expected_hours,
    )


# ---------------------------------------------------------------------------
# Step 11: final checks and save
# ---------------------------------------------------------------------------
def final_checks_and_save(df: pd.DataFrame, output: Path) -> None:
    """Verify the cleaned table meets its contract, then write it to disk."""
    if not df["date_time"].is_unique:
        raise PipelineError("Final check failed: date_time is not unique")
    nulls = df[config.EXPECTED_COLUMNS].isna().sum()
    if nulls.any():
        raise PipelineError(f"Final check failed: missing values remain {nulls[nulls > 0].to_dict()}")

    columns = config.EXPECTED_COLUMNS + ["weather_labels", "weather_label_count", "is_holiday", "traffic_anomaly"]
    df = df.sort_values("date_time")[columns]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output, index=False)
    except OSError as exc:
        raise PipelineError(f"Could not write cleaned file to {output}") from exc

    logger.info("Saved cleaned data to %s: %d rows, %d columns", output, df.shape[0], df.shape[1])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """Fingerprint of a text file: any change to its content changes this value.

    Line endings are normalised first (Windows CRLF -> LF), because Git may
    check the same CSV out with different line endings on different systems.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def changed_cells(before: pd.DataFrame, after: pd.DataFrame, columns: list) -> dict:
    """Number of values each step changed, per column (rows are aligned on date_time)."""
    b, a = before.set_index("date_time"), after.set_index("date_time")
    return {c: int((b[c] != a[c]).sum()) for c in columns}


def write_quality_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved data quality report to %s", path)


def run_pipeline(input_path: Path, output_path: Path) -> dict:
    """Run every step in order and return a data quality report (B2).

    The report records the raw file's SHA-256 fingerprint, so anyone can check
    they started from the same file, and how many rows or values each step
    removed, changed or flagged.
    """
    logger.info("Pipeline started")
    raw_hash = sha256_of(input_path) if input_path.exists() else None
    if raw_hash:
        logger.info("Raw file SHA-256 (line endings normalised): %s", raw_hash)
    df = load_raw(input_path)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "raw_file": {"name": input_path.name, "sha256_normalised_line_endings": raw_hash,
                           "rows": int(df.shape[0]), "columns": int(df.shape[1])},
              "steps": []}

    def record(step: str, rows: int, **details) -> None:
        report["steps"].append({"step": step, "rows_after": int(rows), **details})

    validate_schema(df)
    record("validate_schema", len(df), missing_columns=0)
    df = standardise_categoricals(df)
    record("standardise_categoricals", len(df))
    before = len(df)
    df = parse_datetimes(df)
    record("parse_datetimes", len(df), rows_dropped=before - len(df))
    before = len(df)
    df = drop_exact_duplicates(df)
    record("drop_exact_duplicates", len(df), rows_dropped=before - len(df))
    before = len(df)
    df = collapse_to_hourly(df)
    record("collapse_to_hourly", len(df), rows_merged=before - len(df),
           hours_with_several_conditions=int(df["weather_labels"].str.contains("|", regex=False).sum()))
    before_values = df
    df = handle_impossible_values(df)
    record("handle_impossible_values", len(df),
           values_imputed=changed_cells(before_values, df, ["temp", "rain_1h", "snow_1h", "clouds_all"]))
    df = flag_traffic_anomalies(df)
    record("flag_traffic_anomalies", len(df), hours_flagged=int(df["traffic_anomaly"].sum()),
           traffic_values_changed=0)
    df = extend_holidays(df)
    record("extend_holidays", len(df), holiday_hours=int(df["is_holiday"].sum()))
    report_coverage_gaps(df)
    span = int((df["date_time"].max() - df["date_time"].min()) / pd.Timedelta(hours=1)) + 1
    record("report_coverage_gaps", len(df), hours_present=len(df), hours_possible=span,
           coverage=round(len(df) / span, 4))
    final_checks_and_save(df, output_path)
    report["output_file"] = {"name": output_path.name, "rows": int(len(df)),
                             "period": f"{df['date_time'].min()} to {df['date_time'].max()}"}
    logger.info("Pipeline finished successfully")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the raw traffic CSV into traffic_clean.csv")
    parser.add_argument("--input", type=Path, default=config.RAW_CSV, help="raw CSV path")
    parser.add_argument("--output", type=Path, default=config.CLEAN_CSV, help="cleaned CSV path")
    parser.add_argument("--log-file", type=Path, default=config.LOG_FILE, help="log file path")
    parser.add_argument("--debug", action="store_true", help="show DEBUG messages")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, debug=args.debug)
    try:
        report = run_pipeline(args.input, args.output)
        write_quality_report(report, args.output.parent / config.QUALITY_REPORT_JSON.name)
    except PipelineError:
        logger.error("Pipeline stopped before completion", exc_info=True)
        return 1
    except Exception:  # last-resort guard so the user never sees a raw traceback
        logger.error("Pipeline failed with an unexpected error", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
