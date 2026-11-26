# Model monitoring report

Model: gradient boosting regressor trained on 2012-2017. Baseline MAE = 235.9 (same model type trained to 2016 and validated on 2017, i.e. the error expected at deployment).
Feature drift: PSI vs Jan-Sep 2017, alert above 0.25. Error drift: alert when a month's MAE exceeds 1.25x the baseline.

## Actual 2018: **ALERT / Requires investigation**

Reasons: 2018-04 MAE x1.27

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
| 2018-01 | 289.2   |               1.226 | PASS     |
| 2018-02 | 241.636 |               1.024 | PASS     |
| 2018-03 | 213.451 |               0.905 | PASS     |
| 2018-04 | 298.424 |               1.265 | ALERT    |
| 2018-05 | 181.738 |               0.77  | PASS     |
| 2018-06 | 161.936 |               0.686 | PASS     |
| 2018-07 | 186.86  |               0.792 | PASS     |
| 2018-08 | 172.227 |               0.73  | PASS     |
| 2018-09 | 176.079 |               0.746 | PASS     |

## Simulated surge: **ALERT / Requires investigation**

Reasons: 2018-04 MAE x1.27, 2018-06 MAE x4.29, 2018-07 MAE x4.21, 2018-08 MAE x4.37, 2018-09 MAE x4.18

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
| 2018-01 |  289.2   |               1.226 | PASS     |
| 2018-02 |  241.636 |               1.024 | PASS     |
| 2018-03 |  213.451 |               0.905 | PASS     |
| 2018-04 |  298.424 |               1.265 | ALERT    |
| 2018-05 |  181.738 |               0.77  | PASS     |
| 2018-06 | 1012.92  |               4.294 | ALERT    |
| 2018-07 |  994.12  |               4.214 | ALERT    |
| 2018-08 | 1031.54  |               4.373 | ALERT    |
| 2018-09 |  985.308 |               4.177 | ALERT    |
