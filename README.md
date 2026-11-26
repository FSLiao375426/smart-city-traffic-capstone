# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

Capstone project for the NUS School of Computing **AI, ML and Data Science Programme**.

An end-to-end traffic intelligence solution built on about 48,000 hourly records of westbound I-94 traffic volume near Minneapolis-St Paul (2012-2018), with weather and US federal holiday information.

> **Proxy label notice.** No accident dataset was provided or sourced. The Part 3 classification model predicts a **documented proxy** risk label (high/severe congestion together with severe or low-visibility weather). It demonstrates the classification workflow and must **not** be read as a prediction of real accidents.

## Project structure

```
smart-city-traffic-capstone/
├── common/                   # shared config and logging setup (used by all parts)
│   ├── config.py             # paths, seed, cleaning thresholds, weather sets
│   └── logging_setup.py      # console + file logging for entry points
├── data/
│   ├── raw/                  # original CSV, never modified
│   └── processed/            # outputs of the Part 2 pipeline (regenerated)
├── part1_data_analytics/     # SQLite, statistics, probability, Power BI
├── part2_python/             # pipeline, features, figures, CLI app
├── part3_machine_learning/   # models, clustering, LSTM + SHAP, MLflow, API
└── final_capstone_report/    # final report, bias and fairness report
```

## Setup

Developed with Python 3.11 in a dedicated conda environment (recommended, because TensorFlow supports only some Python versions).

```bash
conda create -n capstone python=3.11 -y
conda activate capstone
pip install -r requirements.txt
```

Without Anaconda, a standard virtual environment also works:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

All scripts are run **from the repository root** with `python -m`, so the shared `common` package is importable.

## Decision log

Decisions that keep the three parts consistent. All constants live in `common/config.py`.

| ID | Decision | Reason |
| --- | --- | --- |
| D1 | One row per hour. When an hour has several weather labels, keep the most severe `weather_main` (order in `WEATHER_SEVERITY`). | The raw file has 48,204 rows but only 40,575 distinct hours. Duplicated hours are mostly bad-weather hours, so leaving them in biases every statistic. |
| D2 | `data/processed/traffic_clean.csv`, produced by the Part 2 pipeline, is the single source of truth. Part 1 SQL applies the same collapse rule in a view. | SQL and Python must report the same numbers. |
| D3 | Holiday flag extended from 00:00 to the whole calendar day. | The raw flag marks only the midnight hour. |
| D4 | Impossible values (0 K temperature, 9,831 mm rain, zero traffic in busy hours) replaced with the median of the same calendar month. | Required by the brief, and Minnesota's seasonal swing makes a global mean misleading. |
| D5 | Three congestion definitions kept under distinct names: `congested` (> 5,500), `traffic_category` (4,500 / 5,500 bands), `congestion_category` (quartiles). | Each is required by a different task. |
| D6 | `SEVERE_WEATHER` = Thunderstorm, Snow, Squall. `LOW_VISIBILITY` = Fog, Mist, Haze, Smoke. | The brief uses both sets without defining them. |
| D7 | Chronological split: train to end of 2017, test on 2018. | Hourly traffic is autocorrelated, so a random split leaks information. |

## Known data limitations

- No data from September 2014 to May 2015. 2012 starts in October and 2018 ends in September.
- New Year's Day 2015 falls inside the gap, which affects Part 1 Task 1.3.
- A single sensor on a single corridor, so results do not generalise to the wider city network.

## How to run

All commands run from the repository root with the environment active. The order matters: each step reads the previous step's output.

```bash
# Part 2 pipeline first: it produces the cleaned data every part uses (D2)
python -m part2_python.pipeline --debug
python -m part2_python.feature_engineering --debug
python -m part2_python.visualizations

# Part 1 analysis
python -m part1_data_analytics.load_sqlite
python -m part1_data_analytics.run_sql_analysis
python -m part1_data_analytics.stats_probability

# Part 2 command-line app (see part2_python/README.md for all commands)
python -m part2_python.traffic_app compare

# Part 3 machine learning
python -m part3_machine_learning.supervised
python -m part3_machine_learning.unsupervised
python -m part3_machine_learning.deep_learning
python -m part3_machine_learning.monitoring
python -m part3_machine_learning.fairness_analysis
python -m part3_machine_learning.api_demo
python -m part3_machine_learning.recommender --date 2018-12-25 --earliest 8 --latest 18 --window 2
uvicorn part3_machine_learning.api:app --port 8000      # API, then open http://127.0.0.1:8000/docs
```

The Power BI dashboard is `part1_data_analytics/powerbi/traffic_dashboard.pbix`, with a PDF export next to it.

### Tests

```bash
python -m unittest discover -s tests -t . -v
```

26 unit tests on small hand-made tables (no data files needed, under a second). They cover the
hourly collapse (traffic is never summed, every weather label is kept), imputation by month,
closure flags, holiday extension, cyclical encoding, training-only quartile thresholds,
the proxy risk label, the three-way chronological split, the model selection rule, and
input checks in the CLI and the recommender.

| Part | Report | Details |
| --- | --- | --- |
| 1 Data analytics | [`part1_data_analytics/INSIGHTS_REPORT.md`](part1_data_analytics/INSIGHTS_REPORT.md) | [`part1_data_analytics/README.md`](part1_data_analytics/README.md) |
| 2 Python pipeline | [`part2_python/REPORT.md`](part2_python/REPORT.md) | [`part2_python/README.md`](part2_python/README.md) |
| 3 Machine learning | [`final_capstone_report/FINAL_REPORT.md`](final_capstone_report/FINAL_REPORT.md) | [`part3_machine_learning/README.md`](part3_machine_learning/README.md) |
| Responsible AI | [`final_capstone_report/BIAS_FAIRNESS_REPORT.md`](final_capstone_report/BIAS_FAIRNESS_REPORT.md) | |

## Logging

Every module uses `logging.getLogger(__name__)`, and only entry-point scripts configure handlers (`common/logging_setup.py`). Messages go to the console and to a log file in the format `timestamp | level | module | message`.

| Log file | Written by |
| --- | --- |
| `part2_python/pipeline.log` | pipeline (fresh log), then feature engineering and figures (appended) |
| `part2_python/app.log` | command-line app (appended, console shows only warnings and errors) |
| `part1_data_analytics/part1.log` | SQLite load (fresh log), then SQL analysis and statistics (appended) |
| `part3_machine_learning/part3.log` | supervised models (fresh log), then clustering, LSTM, monitoring and error analysis (appended) |
| `part3_machine_learning/api.log` | every API request and its prediction |

| Level | Meaning in this project |
| --- | --- |
| DEBUG | Intermediate values such as monthly medians and quartile thresholds. Shown only with `--debug` |
| INFO | Normal milestones: data loaded (rows and columns), step completed, file or figure saved, CLI command invoked |
| WARNING | Recoverable changes: rows dropped or collapsed, values imputed, anomalies flagged, coverage gaps |
| ERROR | The run cannot continue as planned: missing file, failed schema check, invalid user input |

`print()` is used only for answers shown to the user of the command-line app.
