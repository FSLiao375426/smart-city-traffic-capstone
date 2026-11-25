# Model monitoring report

Model: gradient boosting regressor trained on 2012-2017. Baseline MAE = 238.0 (same model type trained to 2016 and validated on 2017, i.e. the error expected at deployment).
Feature drift: PSI vs Jan-Sep 2017, alert above 0.25. Error drift: alert when a month's MAE exceeds 1.25x the baseline.

## Actual 2018: **ALERT / Requires investigation**

Reasons: 2018-01 MAE x1.26, 2018-04 MAE x1.27

### Feature drift

| feature          |   PSI |   KS_statistic |   KS_p_value |   reference_mean |   current_mean | status   |
|:-----------------|------:|---------------:|-------------:|-----------------:|---------------:|:---------|
| temp_c           | 0.136 |          0.096 |         0    |           10.811 |          9.753 | WATCH    |
| clouds_all       | 0.034 |          0.038 |         0    |           42.164 |         39.705 | PASS     |
| rain_1h_log      | 0.229 |          0.061 |         0    |            0     |          0.037 | WATCH    |
| weather_severity | 0.02  |          0.044 |         0    |            2.238 |          2.474 | PASS     |
| traffic_volume   | 0.004 |          0.02  |         0.16 |         3395.13  |       3323.9   | PASS     |

### Monthly prediction error

| month   |     MAE |   ratio_to_baseline | status   |
|:--------|--------:|--------------------:|:---------|
| 2018-01 | 300.912 |               1.264 | ALERT    |
| 2018-02 | 252.463 |               1.061 | PASS     |
| 2018-03 | 217.693 |               0.915 | PASS     |
| 2018-04 | 301.57  |               1.267 | ALERT    |
| 2018-05 | 181.289 |               0.762 | PASS     |
| 2018-06 | 163.976 |               0.689 | PASS     |
| 2018-07 | 191.443 |               0.804 | PASS     |
| 2018-08 | 167.571 |               0.704 | PASS     |
| 2018-09 | 175.521 |               0.737 | PASS     |

## Simulated surge: **ALERT / Requires investigation**

Reasons: 2018-01 MAE x1.26, 2018-04 MAE x1.27, 2018-06 MAE x4.22, 2018-07 MAE x4.17, 2018-08 MAE x4.36, 2018-09 MAE x4.15

### Feature drift

| feature          |   PSI |   KS_statistic |   KS_p_value |   reference_mean |   current_mean | status   |
|:-----------------|------:|---------------:|-------------:|-----------------:|---------------:|:---------|
| temp_c           | 0.136 |          0.096 |            0 |           10.811 |          9.753 | WATCH    |
| clouds_all       | 0.034 |          0.038 |            0 |           42.164 |         39.705 | PASS     |
| rain_1h_log      | 0.229 |          0.061 |            0 |            0     |          0.037 | WATCH    |
| weather_severity | 0.02  |          0.044 |            0 |            2.238 |          2.474 | PASS     |
| traffic_volume   | 0.125 |          0.116 |            0 |         3395.13  |       3773.55  | WATCH    |

### Monthly prediction error

| month   |      MAE |   ratio_to_baseline | status   |
|:--------|---------:|--------------------:|:---------|
| 2018-01 |  300.912 |               1.264 | ALERT    |
| 2018-02 |  252.463 |               1.061 | PASS     |
| 2018-03 |  217.693 |               0.915 | PASS     |
| 2018-04 |  301.57  |               1.267 | ALERT    |
| 2018-05 |  181.289 |               0.762 | PASS     |
| 2018-06 | 1004.29  |               4.219 | ALERT    |
| 2018-07 |  993.023 |               4.172 | ALERT    |
| 2018-08 | 1037.38  |               4.358 | ALERT    |
| 2018-09 |  987.377 |               4.148 | ALERT    |
