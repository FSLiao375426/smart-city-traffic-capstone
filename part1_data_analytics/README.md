# Part 1 – Data Analytics

SQLite analysis, descriptive statistics, probability and the Power BI dashboard.

**Read first:** [`INSIGHTS_REPORT.md`](INSIGHTS_REPORT.md), the 1-2 page summary for the mobility team.

## Contents

| Path | Task | Description |
| --- | --- | --- |
| `load_sqlite.py` | 1.1 | Loads the raw CSV and the cleaned data into `traffic.db`, creates the `traffic_hourly` view and checks that SQL and Python agree |
| `sql/traffic_analysis.sql` | 1.1-1.3 | All SQL queries, each marked with `-- name:` |
| `run_sql_analysis.py` | 1.1-1.3 | Runs every named query and saves the results |
| `stats_probability.py` | 2, 3 | Descriptive statistics, correlation, probabilities, independence test, odds ratios |
| `results/` | | Query results (`*.csv`), `sql_results.md`, `stats_probability.md` and `.json` |
| `powerbi/traffic_dashboard.pbix` | 4 | Power BI dashboard |
| `powerbi/traffic_dashboard.pdf` | 4 | PDF export for readers without Power BI |
| `powerbi/screenshots/` | 4 | Data quality profiling and dashboard screenshots |
| `part1.log` | | Log of the three scripts |
| `INSIGHTS_REPORT.md` | | Part 1 deliverable report |

## How to run

From the repository root, with the `capstone` environment active:

```bash
python -m part2_python.pipeline                     # builds data/processed/traffic_clean.csv
python -m part1_data_analytics.load_sqlite          # builds traffic.db and verifies it
python -m part1_data_analytics.run_sql_analysis     # Task 1 queries -> results/
python -m part1_data_analytics.stats_probability    # Tasks 2-3 -> results/
```

`traffic.db` is rebuilt on every run and is not committed. The SQL file can also be opened in any SQLite tool (for example DB Browser for SQLite) against `traffic.db`.

## Tables in traffic.db

| Name | Type | Rows | Purpose |
| --- | --- | --- | --- |
| `traffic_raw` | table | 48,204 | The CSV exactly as delivered |
| `traffic_hourly` | view | 40,575 | One row per hour, most severe weather kept (decision D1, written in SQL) |
| `traffic_clean` | table | 40,575 | Output of the Part 2 pipeline, used for all analysis (D2) |

`load_sqlite.py` stops with an ERROR if the view and the pipeline disagree on the number of hours or total volume.
