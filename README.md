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

Requires Python 3.10 or later.

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

*Filled in as each part is completed.*

## Logging

*Filled in with Part 2.*
