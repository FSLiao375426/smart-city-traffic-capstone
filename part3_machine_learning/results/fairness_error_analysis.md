# Error distribution across conditions (2018 test set)

Overall regression MAE: 216.6 vehicles per hour. Groups with fewer than 30 hours are marked unreliable. bias_mean_error > 0 means the model over-predicts traffic for that group.

## Regression error by weather_main

| weather_main   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:---------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Clear          |    2328 |      2989.63 | 189.004 |             2.065 |               0.063 |            0.873 | True       |
| Clouds         |    1996 |      3924.41 | 210.944 |           -19.751 |               0.054 |            0.974 | True       |
| Drizzle        |      46 |      3232.17 | 125.048 |           -14.176 |               0.039 |            0.577 | True       |
| Fog            |      99 |      2522.16 | 107.612 |           -40.453 |               0.043 |            0.497 | True       |
| Haze           |     149 |      3459.31 | 186.225 |           -41.617 |               0.054 |            0.86  | True       |
| Mist           |     354 |      2801.08 | 137.156 |            -4.194 |               0.049 |            0.633 | True       |
| Rain           |     775 |      3468.47 | 162.713 |            -2.189 |               0.047 |            0.751 | True       |
| Smoke          |       2 |      4443    | 117.475 |           117.475 |               0.026 |            0.542 | False      |
| Snow           |     516 |      2809.22 | 540.432 |           296.22  |               0.192 |            2.495 | True       |
| Thunderstorm   |     268 |      3246.82 | 209.278 |            55.096 |               0.064 |            0.966 | True       |

## Regression error by time_of_day

| time_of_day   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:--------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Evening       |    1365 |     2552.76  | 257.41  |             6.627 |               0.101 |            1.188 | True       |
| Evening peak  |    1092 |     5174.09  | 307.289 |            18.47  |               0.059 |            1.419 | True       |
| Midday        |    1365 |     4622.81  | 229.082 |            49.262 |               0.05  |            1.058 | True       |
| Morning peak  |    1087 |     4510.72  | 254.396 |            25.294 |               0.056 |            1.175 | True       |
| Night         |    1624 |      841.827 |  85.523 |            -2.929 |               0.102 |            0.395 | True       |

## Regression error by day_type

| day_type   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:-----------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Weekday    |    4671 |      3586.85 | 195.468 |            23.165 |               0.054 |            0.902 | True       |
| Weekend    |    1862 |      2664.28 | 269.608 |             5.902 |               0.101 |            1.245 | True       |

## Regression error by holiday_flag

| holiday_flag   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:---------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Holiday        |     167 |      2453.41 | 376.398 |           126.497 |               0.153 |            1.738 | True       |
| Not holiday    |    6366 |      3346.74 | 212.407 |            15.405 |               0.063 |            0.981 | True       |

## Regression error by month_name

| month_name   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:-------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| 01 Jan       |     742 |      3111.75 | 300.912 |            23.229 |               0.097 |            1.389 | True       |
| 02 Feb       |     672 |      3261.23 | 252.463 |           -26.202 |               0.077 |            1.166 | True       |
| 03 Mar       |     733 |      3446.92 | 217.693 |            10.006 |               0.063 |            1.005 | True       |
| 04 Apr       |     720 |      3297.24 | 301.57  |           103.719 |               0.091 |            1.392 | True       |
| 05 May       |     743 |      3394.53 | 181.289 |            41.576 |               0.053 |            0.837 | True       |
| 06 Jun       |     719 |      3422.52 | 163.976 |            22.998 |               0.048 |            0.757 | True       |
| 07 Jul       |     744 |      3252.46 | 191.443 |            -7.811 |               0.059 |            0.884 | True       |
| 08 Aug       |     740 |      3436.42 | 167.571 |            -6.325 |               0.049 |            0.774 | True       |
| 09 Sep       |     720 |      3289.26 | 175.521 |             0.86  |               0.053 |            0.81  | True       |

## Proxy risk classifier by weather_main

| weather_main   |   hours |   positives |   recall |   precision |   false_positive_rate |
|:---------------|--------:|------------:|---------:|------------:|----------------------:|
| Clear          |    2328 |           0 |  nan     |     nan     |                 0     |
| Clouds         |    1996 |           0 |  nan     |     nan     |                 0     |
| Drizzle        |      46 |           0 |  nan     |     nan     |                 0     |
| Fog            |      99 |          28 |    1     |       1     |                 0     |
| Haze           |     149 |          86 |    0.988 |       0.988 |                 0.016 |
| Mist           |     354 |         137 |    1     |       0.993 |                 0.005 |
| Rain           |     775 |           0 |  nan     |     nan     |                 0     |
| Smoke          |       2 |           2 |    1     |       1     |                 0     |
| Snow           |     516 |         209 |    0.981 |       0.743 |                 0.231 |
| Thunderstorm   |     268 |         132 |    1     |       0.886 |                 0.125 |

## Proxy risk classifier by time_of_day

| time_of_day   |   hours |   positives |   recall |   precision |   false_positive_rate |
|:--------------|--------:|------------:|---------:|------------:|----------------------:|
| Evening       |    1365 |          15 |    0.8   |       0.324 |                 0.019 |
| Evening peak  |    1092 |         134 |    0.993 |       0.816 |                 0.031 |
| Midday        |    1365 |         195 |    1     |       0.878 |                 0.023 |
| Morning peak  |    1087 |         250 |    0.996 |       0.969 |                 0.01  |
| Night         |    1624 |           0 |  nan     |     nan     |                 0     |

## Proxy risk classifier by day_type

| day_type   |   hours |   positives |   recall |   precision |   false_positive_rate |
|:-----------|--------:|------------:|---------:|------------:|----------------------:|
| Weekday    |    4671 |         498 |    0.994 |       0.92  |                 0.01  |
| Weekend    |    1862 |          96 |    0.979 |       0.667 |                 0.027 |
