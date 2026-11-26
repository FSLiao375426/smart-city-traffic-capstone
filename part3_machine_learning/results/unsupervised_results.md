# Unsupervised learning results

## K-means: choosing k

|   k |   inertia |   silhouette |
|----:|----------:|-------------:|
|   2 |  141288   |        0.305 |
|   3 |  112640   |        0.333 |
|   4 |   95111.5 |        0.338 |
|   5 |   81482.9 |        0.356 |
|   6 |   68952.7 |        0.351 |
|   7 |   58260.1 |        0.377 |
|   8 |   50193.6 |        0.392 |

## K-means: cluster profiles

|   hours |   share_of_hours |   typical_hour | hour_range_p10_p90   |   daytime_share |   weekend_share |   avg_volume |   congested_share |   avg_weather_severity |   severe_or_lowvis_share | top_weather              | name                                                |
|--------:|-----------------:|---------------:|:---------------------|----------------:|----------------:|-------------:|------------------:|-----------------------:|-------------------------:|:-------------------------|:----------------------------------------------------|
|    5425 |             0.13 |            2.8 | 1-22                 |            0.26 |            1    |      1200.46 |              0    |                   2.18 |                     0.25 | Clear, Clouds, Rain      | Weekend off-peak hours, light traffic, fair weather |
|   11732 |             0.29 |            0.2 | 0-23                 |            0.01 |            0    |      1473.46 |              0    |                   1.5  |                     0.16 | Clear, Clouds, Rain      | Weekday off-peak hours, light traffic, fair weather |
|    5820 |             0.14 |           15   | 10-20                |            0.78 |            1    |      3910.24 |              0.01 |                   1.66 |                     0.12 | Clouds, Clear, Rain      | Weekend daytime, busy traffic, fair weather         |
|    5553 |             0.14 |           12.3 | 5-20                 |            0.7  |            0.06 |      4240.78 |              0.24 |                   7.69 |                     0.67 | Rain, Snow, Thunderstorm | Weekday mixed hours, busy traffic, rain or snow     |
|   12025 |             0.3  |           12.3 | 7-17                 |            0.98 |            0    |      5273.36 |              0.4  |                   0.75 |                     0.09 | Clouds, Clear, Mist      | Weekday daytime, congested traffic, fair weather    |

## Association rules: top 10 by lift

| rule                                                                                    |   support |   confidence |   lift |
|:----------------------------------------------------------------------------------------|----------:|-------------:|-------:|
| day=Weekend AND temp=Below 0C AND time=Night => congestion=Low                          |     0.022 |        0.938 |  3.824 |
| day=Weekday AND temp=Above 20C AND time=Night => congestion=Low                         |     0.015 |        0.889 |  3.628 |
| day=Weekend AND time=Night AND weather=Clouds => congestion=Low                         |     0.016 |        0.889 |  3.627 |
| day=Weekend AND temp=Above 20C AND time=Evening peak => congestion=High                 |     0.017 |        0.928 |  3.622 |
| day=Weekend AND time=Night => congestion=Low                                            |     0.063 |        0.882 |  3.597 |
| day=Weekday AND temp=0-20C AND time=Morning peak AND weather=Rain => congestion=Severe  |     0.012 |        0.914 |  3.59  |
| day=Weekday AND temp=0-20C AND time=Morning peak AND weather=Clear => congestion=Severe |     0.018 |        0.907 |  3.565 |
| temp=Below 0C AND time=Night AND weather=Snow => congestion=Low                         |     0.013 |        0.873 |  3.559 |
| time=Night AND weather=Snow => congestion=Low                                           |     0.016 |        0.871 |  3.551 |
| day=Weekend AND temp=0-20C AND time=Night => congestion=Low                             |     0.034 |        0.868 |  3.542 |

### In plain language

- When day is Weekend and temp is Below 0C and time is Night, traffic is Low 94% of the time, 3.8 times more often than usual.
- When day is Weekday and temp is Above 20C and time is Night, traffic is Low 89% of the time, 3.6 times more often than usual.
- When day is Weekend and time is Night and weather is Clouds, traffic is Low 89% of the time, 3.6 times more often than usual.
- When day is Weekend and temp is Above 20C and time is Evening peak, traffic is High 93% of the time, 3.6 times more often than usual.
- When day is Weekend and time is Night, traffic is Low 88% of the time, 3.6 times more often than usual.
- When day is Weekday and temp is 0-20C and time is Morning peak and weather is Rain, traffic is Severe 91% of the time, 3.6 times more often than usual.
- When day is Weekday and temp is 0-20C and time is Morning peak and weather is Clear, traffic is Severe 91% of the time, 3.6 times more often than usual.
- When temp is Below 0C and time is Night and weather is Snow, traffic is Low 87% of the time, 3.6 times more often than usual.
- When time is Night and weather is Snow, traffic is Low 87% of the time, 3.6 times more often than usual.
- When day is Weekend and temp is 0-20C and time is Night, traffic is Low 87% of the time, 3.5 times more often than usual.

## Strongest rule for each congestion level

**Severe**

- When day is Weekday and temp is 0-20C and time is Morning peak and weather is Rain, traffic is Severe 91% of the time, 3.6 times more often than usual.
- When day is Weekday and temp is 0-20C and time is Morning peak and weather is Clear, traffic is Severe 91% of the time, 3.6 times more often than usual.
- When day is Weekday and temp is 0-20C and time is Morning peak, traffic is Severe 90% of the time, 3.5 times more often than usual.

**High**

- When day is Weekend and temp is Above 20C and time is Evening peak, traffic is High 93% of the time, 3.6 times more often than usual.
- When day is Weekend and temp is Above 20C and time is Midday, traffic is High 88% of the time, 3.5 times more often than usual.
- When day is Weekend and time is Evening peak and weather is Clouds, traffic is High 87% of the time, 3.4 times more often than usual.

**Medium**

- When day is Weekday and temp is Below 0C and time is Evening and weather is Clouds, traffic is Medium 80% of the time, 3.3 times more often than usual.
- When temp is Below 0C and time is Evening and weather is Clouds, traffic is Medium 79% of the time, 3.2 times more often than usual.
- When temp is 0-20C and time is Evening and weather is Rain, traffic is Medium 78% of the time, 3.2 times more often than usual.

**Low**

- When day is Weekend and temp is Below 0C and time is Night, traffic is Low 94% of the time, 3.8 times more often than usual.
- When day is Weekday and temp is Above 20C and time is Night, traffic is Low 89% of the time, 3.6 times more often than usual.
- When day is Weekend and time is Night and weather is Clouds, traffic is Low 89% of the time, 3.6 times more often than usual.
