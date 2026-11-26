# Error distribution across conditions (2018 test set)

Overall regression MAE: 213.2 vehicles per hour. Groups with fewer than 30 hours are marked unreliable. bias_mean_error > 0 means the model over-predicts traffic for that group.

## Regression error by weather_main

| weather_main   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:---------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Clear          |    2328 |      2989.63 | 189.424 |             4.763 |               0.063 |            0.888 | True       |
| Clouds         |    1996 |      3924.41 | 212.202 |           -17.09  |               0.054 |            0.995 | True       |
| Drizzle        |      46 |      3232.17 | 130.488 |           -24.619 |               0.04  |            0.612 | True       |
| Fog            |      99 |      2522.16 | 108.184 |           -42.538 |               0.043 |            0.507 | True       |
| Haze           |     149 |      3459.31 | 177.006 |           -29.772 |               0.051 |            0.83  | True       |
| Mist           |     354 |      2801.08 | 144.303 |            -6.081 |               0.052 |            0.677 | True       |
| Rain           |     775 |      3468.47 | 158.941 |           -10.544 |               0.046 |            0.745 | True       |
| Smoke          |       2 |      4443    | 108.373 |           108.373 |               0.024 |            0.508 | False      |
| Snow           |     516 |      2809.22 | 497.559 |           281.581 |               0.177 |            2.333 | True       |
| Thunderstorm   |     268 |      3246.82 | 202.101 |            48.384 |               0.062 |            0.948 | True       |

## Regression error by every weather label present (groups overlap)

| weather_label_present   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_vs_overall | reliable   |
|:------------------------|--------:|-------------:|--------:|------------------:|-----------------:|:-----------|
| Clear                   |    2328 |      2989.63 | 189.424 |             4.763 |            0.888 | True       |
| Clouds                  |    1996 |      3924.41 | 212.202 |           -17.09  |            0.995 | True       |
| Drizzle                 |     278 |      3569.06 | 145.354 |             2.561 |            0.682 | True       |
| Fog                     |     192 |      2224.91 | 509.904 |           393.864 |            2.391 | True       |
| Haze                    |     272 |      3213.19 | 254.036 |            32.155 |            1.191 | True       |
| Mist                    |    1099 |      2840.57 | 253.76  |            88.654 |            1.19  | True       |
| Rain                    |     992 |      3391.65 | 182.483 |            17.243 |            0.856 | True       |
| Smoke                   |       2 |      4443    | 108.373 |           108.373 |            0.508 | False      |
| Snow                    |     521 |      2800.6  | 501.656 |           287.203 |            2.353 | True       |
| Thunderstorm            |     268 |      3246.82 | 202.101 |            48.384 |            0.948 | True       |

## Regression error by visibility

| visibility        |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:------------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Low visibility    |    1255 |      2904.89 | 276.472 |           107.741 |               0.095 |            1.297 | True       |
| Normal visibility |    5278 |      3423.53 | 198.187 |            -3.763 |               0.058 |            0.929 | True       |

## Regression error by time_of_day

| time_of_day   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:--------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Evening       |    1365 |     2552.76  | 255.953 |             8.135 |               0.1   |            1.2   | True       |
| Evening peak  |    1092 |     5174.09  | 296.175 |            14.914 |               0.057 |            1.389 | True       |
| Midday        |    1365 |     4622.81  | 223.864 |            45.434 |               0.048 |            1.05  | True       |
| Morning peak  |    1087 |     4510.72  | 252.276 |            29.049 |               0.056 |            1.183 | True       |
| Night         |    1624 |      841.827 |  86.457 |            -3.468 |               0.103 |            0.405 | True       |

## Regression error by day_type

| day_type   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:-----------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Weekday    |    4671 |      3586.85 | 194.101 |            22.983 |               0.054 |            0.91  | True       |
| Weekend    |    1862 |      2664.28 | 261.203 |             4.295 |               0.098 |            1.225 | True       |

## Regression error by holiday_flag

| holiday_flag   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:---------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| Holiday        |     167 |      2453.41 | 369.026 |           134.495 |               0.15  |            1.731 | True       |
| Not holiday    |    6366 |      3346.74 | 209.139 |            14.592 |               0.062 |            0.981 | True       |

## Regression error by month_name

| month_name   |   hours |   avg_volume |     MAE |   bias_mean_error |   MAE_pct_of_volume |   MAE_vs_overall | reliable   |
|:-------------|--------:|-------------:|--------:|------------------:|--------------------:|-----------------:|:-----------|
| 01 Jan       |     742 |      3111.75 | 289.2   |            23.911 |               0.093 |            1.356 | True       |
| 02 Feb       |     672 |      3261.23 | 241.636 |           -31.118 |               0.074 |            1.133 | True       |
| 03 Mar       |     733 |      3446.92 | 213.451 |            12.59  |               0.062 |            1.001 | True       |
| 04 Apr       |     720 |      3297.24 | 298.424 |           104.659 |               0.091 |            1.4   | True       |
| 05 May       |     743 |      3394.53 | 181.738 |            39.928 |               0.054 |            0.852 | True       |
| 06 Jun       |     719 |      3422.52 | 161.936 |            14.363 |               0.047 |            0.759 | True       |
| 07 Jul       |     744 |      3252.46 | 186.86  |           -10.81  |               0.057 |            0.876 | True       |
| 08 Aug       |     740 |      3436.42 | 172.227 |            -0.347 |               0.05  |            0.808 | True       |
| 09 Sep       |     720 |      3289.26 | 176.079 |             3.118 |               0.054 |            0.826 | True       |

## Proxy risk classifier by weather_main

| weather_main   |   hours |   positives |   predicted_positives |   recall |   precision |   false_positive_rate | reliable   |
|:---------------|--------:|------------:|----------------------:|---------:|------------:|----------------------:|:-----------|
| Clear          |    2328 |           0 |                     0 |  n/a     |     n/a     |                 0     | True       |
| Clouds         |    1996 |           0 |                     0 |  n/a     |     n/a     |                 0     | True       |
| Drizzle        |      46 |           4 |                     4 |    1     |       1     |                 0     | True       |
| Fog            |      99 |          28 |                    28 |    1     |       1     |                 0     | True       |
| Haze           |     149 |          87 |                    87 |    0.977 |       0.977 |                 0.032 | True       |
| Mist           |     354 |         140 |                   140 |    0.979 |       0.979 |                 0.014 | True       |
| Rain           |     775 |         116 |                   121 |    0.983 |       0.942 |                 0.011 | True       |
| Smoke          |       2 |           2 |                     0 |    0     |     n/a     |               n/a     | False      |
| Snow           |     516 |         216 |                   278 |    0.995 |       0.773 |                 0.21  | True       |
| Thunderstorm   |     268 |         134 |                   151 |    0.985 |       0.874 |                 0.142 | True       |

## Proxy risk classifier by visibility

| visibility        |   hours |   positives |   predicted_positives |   recall |   precision |   false_positive_rate | reliable   |
|:------------------|--------:|------------:|----------------------:|---------:|------------:|----------------------:|:-----------|
| Low visibility    |    1255 |         537 |                   596 |    0.983 |       0.886 |                 0.095 | True       |
| Normal visibility |    5278 |         190 |                   213 |    0.984 |       0.878 |                 0.005 | True       |

## Proxy risk classifier by time_of_day

| time_of_day   |   hours |   positives |   predicted_positives |   recall |   precision |   false_positive_rate | reliable   |
|:--------------|--------:|------------:|----------------------:|---------:|------------:|----------------------:|:-----------|
| Evening       |    1365 |          19 |                    46 |    0.842 |       0.348 |                 0.022 | True       |
| Evening peak  |    1092 |         159 |                   183 |    0.994 |       0.863 |                 0.027 | True       |
| Midday        |    1365 |         250 |                   273 |    0.984 |       0.901 |                 0.024 | True       |
| Morning peak  |    1087 |         299 |                   307 |    0.987 |       0.961 |                 0.015 | True       |
| Night         |    1624 |           0 |                     0 |  n/a     |     n/a     |                 0     | True       |

## Proxy risk classifier by day_type

| day_type   |   hours |   positives |   predicted_positives |   recall |   precision |   false_positive_rate | reliable   |
|:-----------|--------:|------------:|----------------------:|---------:|------------:|----------------------:|:-----------|
| Weekday    |    4671 |         604 |                   649 |    0.998 |       0.929 |                 0.011 | True       |
| Weekend    |    1862 |         123 |                   160 |    0.911 |       0.7   |                 0.028 | True       |
