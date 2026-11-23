# Part 2: Methodology and Findings

**A reproducible Python pipeline for I-94 traffic data**

## 1. Approach

The workflow has four stages, each a separate module with its own log trail:

```
raw CSV ──pipeline.py──> traffic_clean.csv ──feature_engineering.py──> traffic_features.csv
                                                                        ├─ visualizations.py ─> figures/
                                                                        └─ traffic_app.py ───> answers
```

`traffic_clean.csv` is the single source of truth for the whole capstone: Part 1 loads it into SQLite and Power BI, and Part 3 trains on the features built from it. All constants that must stay identical across parts (thresholds, weather groups, the train/test split year) live in `common/config.py`.

Before writing the pipeline, the raw data was profiled (shape, duplicates, uniqueness of timestamps, missing values, physical plausibility, time coverage). Every issue found became a pipeline step that detects it programmatically and logs what it changed.

## 2. Data pipeline (Task 1)

| Step | Result | Decision |
| --- | --- | --- |
| Load with try/except | 48,204 rows, 9 columns | File, empty-file, parse and encoding errors become a clean ERROR |
| Validate schema | 9 expected columns present | Runs before any other step |
| Standardise categories | 1,730 descriptions re-cased, 48,143 missing holiday labels set to "None" | One spelling per category |
| Parse date/time | All rows parsed | Unparseable rows would be dropped and logged |
| Remove exact duplicates | 17 rows | |
| Collapse to one row per hour | 7,612 more rows removed, 40,575 hours | Keep the most severe weather (D1). Traffic was identical across repeats |
| Impossible values | 10 temperatures of 0 K, 1 rainfall of 9,831 mm | Temperature: median of the same month, applied month by month in a loop. Rainfall: median of the same weather description (D11), because its label was "very heavy rain" and the monthly median was 0 mm |
| Traffic anomalies | 20 daytime hours under 100 vehicles on 4 days | Flagged, not imputed (D9). Traffic is the prediction target, and these look like planned road closures |
| Holidays | 53 → 1,203 flagged hours | Extended from the midnight hour to the whole day (D3) |
| Coverage | Gaps of 307 days (Aug 2014 to Jun 2015) and 10 days | Logged as warnings, nothing changed |

Two choices deliberately depart from a literal reading of the brief. Rainfall is imputed by weather condition rather than by month, because a monthly median of 0 mm would contradict the record's own "very heavy rain" label. Zero traffic is flagged rather than imputed, because inventing target values would teach the Part 3 models fabricated answers. Both are documented in the code, the log and the README decision log.

## 3. Feature engineering (Task 2)

The table grows from 12 to 50 columns.

| Group | Features |
| --- | --- |
| Time | `hour`, `day_of_week`, `is_weekend`, `month`, `year`, `time_of_day` band |
| Cyclical | sine and cosine of hour (period 24), day of week (7) and month (12), so that 23:00 sits next to 00:00 |
| Weather | `temp_c`, `weather_severity` (0 = Clear to 10 = Thunderstorm), `is_severe_weather`, `is_low_visibility`, `is_precipitation`, 11 one-hot `wx_*` columns |
| Scaled | `temp_c_z` (z-score, mean 8.23 °C, sd 13.10), `clouds_all_mm` (min-max), `rain_1h_log` (log1p, because 95% of hours have no rain) |
| Targets | `congested` (> 5,500), `traffic_category` (Low/Medium/High bands), `congestion_category` (quartiles at 1,248.5, 3,427 and 4,952), `high_risk` proxy label |

**Target logic.** `congestion_category` uses quartiles of traffic volume, so each class holds about 10,144 hours. `high_risk` marks High or Severe congestion together with severe weather (Thunderstorm, Snow, Squall) or low visibility (Fog, Mist, Haze, Smoke). It is positive for 7.7% of hours, so the log warns that Part 3 must handle class imbalance. Scaling here uses the whole dataset for exploration only. Part 3 refits scalers on the training years to avoid leakage.

## 4. Visual findings (Task 3)

1. **Hourly profile, weekday vs weekend** (`fig1`). Weekdays peak at 07:00 (6,061) and 16:00 (6,241), both above the congestion threshold. Weekends rise slowly to a flat midday plateau near 4,400 and never reach it.
2. **Traffic distribution** (`fig2`). Hourly volume forms three clusters (night below 1,000, shoulder hours near 2,800, daytime near 4,700). The mean (3,291) describes very few real hours, which is why later models need time features rather than a single average.
3. **Congestion by weather, all hours vs peak hours** (`fig3`). Across all hours, clear weather looks less congested than cloudy weather (13% vs 17%). Within peak hours the gap almost vanishes (55% vs 58%). Snow is the exception at 36%. Time of day confounds the raw weather comparison.
4. **Heatmap of hour by day of week** (`fig4`). Monday to Friday share the same two-peak pattern. Friday's evening rush starts earlier (5,778 at 15:00) and lasts longer (4,679 at 18:00, the highest of any day), while Monday has the lowest 16:00 peak (5,967). Saturday and Sunday build slowly to a midday plateau.
5. **Temperature vs traffic** (`fig5`). Every temperature from -30 °C to 35 °C spans the full range of traffic (r = 0.14). Temperature alone is a weak predictor.

## 5. Command-line application (Task 4)

The app answers five kinds of question from the processed data:

```text
$ python -m part2_python.traffic_app compare
Weekday vs weekend
                           Weekday   Weekend
  Average per hour           3,557     2,628
  Hours congested            20.9%      0.4%
  Busiest hour               16:00     13:00

$ python -m part2_python.traffic_app recommend --day-type weekday --earliest 6 --latest 10 --window 2
Recommended travel window for a weekday journey
  Best : 09:00-11:00, about 4,651 vehicles per hour
  Avoid: 07:00-09:00, about 5,785 vehicles per hour

  For a weekday journey between 06:00 and 11:00, consider travelling 09:00-11:00,
  when historical traffic is typically 20% lower than at the busiest time.

$ python -m part2_python.traffic_app query --datetime "2017-13-01 8am"
ERROR | Invalid input for 'query': Invalid date/time '2017-13-01 8am'. Use the format YYYY-MM-DD HH:MM
```

Typical profiles exclude flagged closure hours. When a weather filter has too little history (for example Squall), the app warns and falls back to all weather instead of giving an unreliable answer. The `recommend` command is extended into the Part 3 recommendation system.

## 6. Reproducibility and logging

- **Logging:** every module uses `logging.getLogger(__name__)`. Only entry points configure handlers (console and file). Each cleaning step writes its own message with the number of affected rows and the reason. DEBUG messages (monthly medians, quartile thresholds, scaling parameters) appear only with `--debug`. No `print()` is used for progress. The sample `pipeline.log` and `app.log` are committed.
- **Version control:** the work was committed incrementally, one commit per task, with descriptive messages. A follow-up commit improved rainfall imputation after reviewing the first pipeline run.
- **Environment:** `requirements.txt` and a dedicated Python 3.11 conda environment.

## 7. Limitations

- A single sensor on one corridor. Patterns do not generalise to the city network.
- `rain_1h` is unreliable: 63% of hours labelled Rain record 0 mm, so rainfall features will be weak.
- Weather labels were reduced to one per hour. The original label count is kept in `weather_label_count`, but co-occurring conditions are lost.
- The 2014-2015 gap removes a full winter, so winter patterns rely on fewer years.
