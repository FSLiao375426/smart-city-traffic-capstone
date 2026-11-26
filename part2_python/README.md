# Part 2 – Python Traffic Analytics Pipeline

A reproducible, logged workflow that turns the raw CSV into clean data, ML-ready features, figures and a command-line app.

**Read first:** [`REPORT.pdf`](REPORT.pdf), the 2-page methodology and findings report. [`REPORT.md`](REPORT.md) is the full version.

## Contents

| File | Task | Output |
| --- | --- | --- |
| `pipeline.py` | 1 | `data/processed/traffic_clean.csv` (40,575 hours × 13 columns) and `quality_report.json` |
| `feature_engineering.py` | 2 | `data/processed/traffic_features.csv` (40,575 × 51) and `feature_params.json` |
| `visualizations.py` | 3 | Five PNG figures in `figures/` |
| `traffic_app.py` | 4 | Command-line app |
| `pipeline.log` | | Sample log of pipeline, features and figures |
| `app.log` | | Sample log of CLI commands, including invalid input |

Shared code lives in `common/`: `config.py` (paths and decision constants), `data.py` (readers for the processed files) and `logging_setup.py`.

## How to run

From the repository root, with the `capstone` environment active:

```bash
python -m part2_python.pipeline --debug
python -m part2_python.feature_engineering --debug
python -m part2_python.visualizations
```

Drop `--debug` for a normal run. The processed CSV files are not committed because the pipeline regenerates them.

## Command-line app

```bash
python -m part2_python.traffic_app query --datetime "2017-06-01 08:00"
python -m part2_python.traffic_app day --date 2016-07-04
python -m part2_python.traffic_app peaks --day-type weekday --top 5
python -m part2_python.traffic_app compare
python -m part2_python.traffic_app recommend --day-type weekday --earliest 6 --latest 10 --window 2
python -m part2_python.traffic_app recommend --day-type weekday --weather Snow --window 2
python -m part2_python.traffic_app --help
```

| Command | Answers |
| --- | --- |
| `query` | Traffic, weather and category at a given hour, compared with a typical hour of the same day type |
| `day` | Total traffic, busiest hour, temperature range and weather for a date |
| `peaks` | Busiest hours on weekdays, weekends or all days, with the share of congested hours |
| `compare` | Weekday vs weekend averages, congestion share, busiest and quietest hours |
| `recommend` | Lowest-traffic travel window within a time range, optionally for a weather condition, with a plain-language recommendation |

Invalid input (a malformed date, an unknown weather type, an hour outside 0-23, a date in the data gap) produces one clear ERROR line and exit code 2, never a traceback.

## Logging

Every module creates its logger with `logging.getLogger(__name__)`. Handlers are configured only in the entry points, through `common/logging_setup.py`.

| Setting | Value |
| --- | --- |
| Format | `timestamp \| level \| module \| message` |
| Pipeline, features, figures | Console and `part2_python/pipeline.log`. The pipeline starts a fresh log, the next two steps append to it |
| CLI app | `part2_python/app.log` (appended). The console shows only WARNING and ERROR, so the screen shows answers, not progress |
| Default level | INFO. `--debug` adds DEBUG messages |

| Level | Used for | Examples |
| --- | --- | --- |
| DEBUG | Intermediate values | Monthly medians used for imputation, quartile thresholds, scaling parameters |
| INFO | Normal milestones | File loaded with row and column counts, shape before and after feature engineering, figure saved with path, CLI command and arguments |
| WARNING | Recoverable changes | Rows dropped or collapsed, values imputed, anomalies flagged, coverage gaps, class imbalance |
| ERROR | The run cannot continue | Missing file, failed schema validation, invalid CLI input |

`print()` is used only in `traffic_app.py`, for the answers shown to the user.
