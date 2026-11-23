"""Part 1 Task 1.1: load the traffic data into SQLite and verify it.

Creates part1_data_analytics/traffic.db with:

* traffic_raw     the comma-delimited CSV exactly as delivered (48,204 rows)
* traffic_hourly  a VIEW that applies the D1 collapse rule in SQL
                  (one row per hour, most severe weather kept)
* traffic_clean   the Part 2 pipeline output (D2: single source of truth)

The script then checks that the SQL view and the Python pipeline agree on the
number of hours and the total traffic volume, so Part 1 and Part 2 cannot drift.

Run from the repository root (after the Part 2 pipeline):

    python -m part1_data_analytics.load_sqlite
"""

import logging
import sqlite3
import sys

import pandas as pd

from common import config
from common.data import load_clean
from common.logging_setup import setup_logging

logger = logging.getLogger(__name__)

LOG_FILE = config.ROOT_DIR / "part1_data_analytics" / "part1.log"


def severity_case_sql() -> str:
    """Build a SQL CASE expression from the same severity ranking Python uses."""
    whens = "\n".join(
        f"            WHEN '{name}' THEN {rank}" for name, rank in config.WEATHER_SEVERITY.items()
    )
    return f"CASE weather_main\n{whens}\n            ELSE 99\n        END"


HOURLY_VIEW_SQL = f"""
CREATE VIEW traffic_hourly AS
SELECT holiday, temp, rain_1h, snow_1h, clouds_all,
       weather_main, weather_description, date_time, traffic_volume
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY date_time
               ORDER BY {severity_case_sql()}
           ) AS rn
    FROM traffic_raw
)
WHERE rn = 1;
"""


def load_tables(conn: sqlite3.Connection) -> None:
    """Load the raw CSV and the cleaned CSV, then create the hourly view."""
    try:
        # keep_default_na=False keeps the literal text "None" in holiday, as delivered
        raw = pd.read_csv(config.RAW_CSV, keep_default_na=False)
    except FileNotFoundError:
        logger.error("Raw CSV not found at %s", config.RAW_CSV, exc_info=True)
        raise
    raw.to_sql("traffic_raw", conn, if_exists="replace", index=False)
    logger.info("Loaded traffic_raw: %d rows, %d columns", len(raw), raw.shape[1])

    try:
        clean = load_clean()
    except FileNotFoundError:
        logger.error("Cleaned CSV not found. Run 'python -m part2_python.pipeline' first.", exc_info=True)
        raise
    clean["date_time"] = clean["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    clean.to_sql("traffic_clean", conn, if_exists="replace", index=False)
    logger.info("Loaded traffic_clean: %d rows, %d columns", len(clean), clean.shape[1])

    conn.execute("DROP VIEW IF EXISTS traffic_hourly")
    conn.execute(HOURLY_VIEW_SQL)
    logger.info("Created view traffic_hourly (D1 collapse rule in SQL)")


def verify(conn: sqlite3.Connection) -> bool:
    """Run verification queries and log the results. Returns True if all checks pass."""
    q = lambda sql: conn.execute(sql).fetchone()  # noqa: E731

    n_raw, n_cols = q("SELECT COUNT(*) FROM traffic_raw")[0], len(conn.execute("SELECT * FROM traffic_raw LIMIT 1").description)
    first, last = q("SELECT MIN(date_time), MAX(date_time) FROM traffic_raw")
    logger.info("traffic_raw: %d rows, %d columns, %s to %s", n_raw, n_cols, first, last)

    nulls = q("""SELECT SUM(holiday IS NULL), SUM(temp IS NULL), SUM(weather_main IS NULL),
                        SUM(date_time IS NULL), SUM(traffic_volume IS NULL) FROM traffic_raw""")
    logger.info("traffic_raw NULL counts (holiday, temp, weather_main, date_time, traffic_volume): %s", nulls)

    distinct_hours = q("SELECT COUNT(DISTINCT date_time) FROM traffic_raw")[0]
    if distinct_hours < n_raw:
        logger.warning("traffic_raw has %d rows but only %d distinct hours (%d repeated rows)",
                       n_raw, distinct_hours, n_raw - distinct_hours)

    hourly_n, hourly_sum = q("SELECT COUNT(*), SUM(traffic_volume) FROM traffic_hourly")
    clean_n, clean_sum = q("SELECT COUNT(*), SUM(traffic_volume) FROM traffic_clean")
    raw_sum = q("SELECT SUM(traffic_volume) FROM traffic_raw")[0]
    logger.info("Hours: SQL view %d, Python pipeline %d", hourly_n, clean_n)
    logger.info("Total volume: raw %d, SQL view %d, Python pipeline %d", raw_sum, hourly_sum, clean_sum)
    logger.warning("Summing traffic_raw would overstate total volume by %d vehicles (%.1f%%) because of repeated hours",
                   raw_sum - hourly_sum, 100 * (raw_sum - hourly_sum) / hourly_sum)

    ok = hourly_n == clean_n and hourly_sum == clean_sum
    if ok:
        logger.info("Consistency check passed: SQL and Python agree on hours and total volume")
    else:
        logger.error("Consistency check FAILED: SQL view and Python pipeline disagree")
    return ok


def main() -> int:
    setup_logging(LOG_FILE)
    try:
        with sqlite3.connect(config.SQLITE_DB) as conn:
            load_tables(conn)
            ok = verify(conn)
    except (FileNotFoundError, sqlite3.Error, pd.errors.ParserError):
        logger.error("Could not build the SQLite database", exc_info=True)
        return 1
    logger.info("Database ready at %s", config.SQLITE_DB)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
