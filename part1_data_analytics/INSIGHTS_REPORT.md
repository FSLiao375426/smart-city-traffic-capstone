# Part 1: Data Analytics Insights Report

**Smart City Traffic Intelligence: westbound I-94, Minneapolis-St Paul, October 2012 to September 2018**

Audience: the Smart City Mobility team. All figures come from `traffic_clean` (one row per hour) unless stated otherwise, so SQL, Python and Power BI report the same numbers.

## Key findings at a glance

1. **Time of day is the dominant driver of traffic.** Weekdays show two commuter peaks (07:00 and 16:00, about 6,000 vehicles per hour). Nights fall below 500. Weather and temperature matter far less once time of day is taken into account.
2. **Congestion is a weekday rush-hour phenomenon.** 15.1% of hours exceed 5,500 vehicles, almost all of them at 06:00-08:00 and 15:00-17:00 on weekdays. At 16:00 on weekdays, 91% of hours are congested.
3. **Apparent weather effects are mostly time-of-day effects.** Clear hours look less congested than cloudy hours (odds ratio 0.74), but after comparing like-for-like hours the difference disappears (odds ratio 1.006). Snow is the one condition that clearly reduces peak-hour congestion (36% vs 55-58%).
4. **Data quality shapes every conclusion.** The raw file repeats hours, has a nine-month gap and contains impossible sensor readings. Ignoring these would overstate yearly traffic by 17.7%.

## 1. Data quality and preparation

| Issue               | Evidence                                                                      | Treatment                                                        |
| ------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Repeated hours      | 48,204 rows but only 40,575 distinct hours (one row per weather label)        | Collapsed to one row per hour, keeping the most severe weather   |
| Coverage gap        | No data from September 2014 to May 2015. 2012 starts in October               | Reported totals next to hours recorded and hourly averages       |
| Impossible readings | 10 temperatures of 0 K, one rainfall of 9,831 mm                              | Imputed with monthly or same-condition medians (Part 2 pipeline) |
| Holiday flag        | Only the midnight hour of each holiday is labelled. 99.9% of rows read "None" | Extended to the whole day                                        |
| Road closures       | 20 daytime hours below 100 vehicles on four summer weekends                   | Kept but flagged (`traffic_anomaly`)                             |

Power Query profiling of the raw file showed no empty cells, but the `holiday` column is a placeholder text ("None") rather than a true value for 99.9% of rows. Data types were appropriate after setting `date_time` to Date/Time.

## 2. Annual trends (Task 1.2)

| Year | Total volume | Hours recorded | Coverage | Average per hour |
| --- | --- | --- | --- | --- |
| 2012 | 6,785,754 | 2,103 | 23.9% | 3,227 |
| 2013 | 24,139,878 | 7,294 | 83.3% | 3,310 |
| 2014 | 14,718,915 | 4,501 | 51.4% | 3,270 |
| 2015 | 11,706,145 | 3,593 | 41.0% | 3,258 |
| 2016 | 25,032,183 | 7,838 | 89.2% | 3,194 |
| 2017 | 29,420,221 | 8,713 | 99.5% | 3,377 |

- **Observation 1: totals measure data coverage, not traffic.** Totals fell 39% in 2014 and 21% in 2015, then rose 114% in 2016. These swings track the number of hours recorded. The hourly average moved by at most 5.7% in any year.
- **Observation 2: underlying demand is stable with a modest rise in 2017.** Average hourly volume stayed between 3,194 and 3,377 vehicles. 2017, the only near-complete year, had the highest average (+5.7% on 2016).
- **Observation 3: repeated hours inflate raw totals.** Summing the raw file overstates volume by 7% to 21% per year (17.7% overall), because hours with several weather labels are counted several times.

## 3. Temperature around holidays (Task 1.3)

| Holiday | Date | Avg temp (°C) | Change vs previous year | Avg traffic |
| --- | --- | --- | --- | --- |
| New Year's Day | 2016-01-01 | -6.1 | +12.6 (vs 2014) | 1,832 |
| New Year's Day | 2017-01-02 (observed) | -1.3 | +4.8 | 2,091 |
| Labor Day | 2015-09-07 | 22.2 | | 2,430 |
| Labor Day | 2016-09-05 | 21.9 | -0.3 | 2,152 |
| Labor Day | 2017-09-04 | 18.1 | -3.7 | 2,395 |

New Year's Day 2015 falls inside the data gap. In 2017 the federal holiday was observed on Monday 2 January. The largest year-on-year change is New Year's Day, which warmed by 12.6 °C from 2014 to 2016 and again by 4.8 °C into 2017.

**Relevance to traffic:** holidays carry far less traffic than ordinary days in the same month (New Year's Day about 42% below an ordinary January day, Labor Day about 30% below an ordinary September day). That holiday effect is much larger than any temperature effect. The warmest New Year's Day (2017) did have the most traffic, but with only three or four holidays per type there is too little evidence to attribute this to temperature.

## 4. Descriptive statistics and correlation (Task 2)

| Statistic          | Traffic volume (vehicles per hour) |
| ------------------ | ---------------------------------- |
| Mean               | 3,290.7                            |
| Median             | 3,427                              |
| Standard deviation | 1,984.8                            |
| Variance           | 3,939,323.5                        |
| Range              | 0 to 7,280 (7,280)                 |

The standard deviation is 60% of the mean, so a "typical hour" barely exists. The distribution has three clusters (quiet nights below 1,000, shoulder hours near 2,800, daytime near 4,700). The mean sits between clusters and describes few real hours.

**Temperature and traffic:** Pearson r = 0.14. The relationship is positive but weak (temperature explains about 2% of the variation). Correlation is not causation here: time of day drives both variables, because afternoons are warmer and busier than nights. After removing the hour-of-day effect, r drops to 0.09.

## 5. Probability of congestion (Task 3)

Congestion = more than 5,500 vehicles per hour. Clear weather = `weather_main` is Clear. High temperature = above 292 K (18.9 °C).

| Quantity                                                         | Value                      |
| ---------------------------------------------------------------- | -------------------------- |
| P(Congestion)                                                    | 0.151                      |
| P(Clear)                                                         | 0.329                      |
| P(Congestion and Clear)                                          | 0.043                      |
| P(Clear given Congestion)                                        | 0.289                      |
| P(High temperature given Congestion)                             | 0.278 (vs 0.259 overall)   |
| P(Congestion) × P(Clear), if independent                         | 0.050                      |
| Odds ratio, clear vs cloudy                                      | 0.74 (95% CI 0.69 to 0.79) |
| Odds ratio, clear vs cloudy, adjusted for hour (Mantel-Haenszel) | 1.006                      |

**Independence:** the observed joint probability (0.043) is 12% below the 0.050 expected under independence, and a chi-square test rejects independence (p < 0.001). Taken at face value, clear weather is associated with *less* congestion.

**Why that is misleading:** clear skies are more common at night (37.6% of night hours vs 28.9% of daytime hours), and congestion never happens at night. Comparing clear and cloudy hours within the same hour of day removes this confounder, and the odds ratio becomes 1.006, meaning no difference. During peak hours, the share of congested hours is 55% when clear, 58% when cloudy, and 36% when snowing.

**Conclusion:** congestion depends on when people travel, not on whether the sky is clear. Snow is the exception that does reduce peak-hour congestion, most likely because some commuters stay home or leave at different times.

## 6. Power BI dashboard (Task 4)

The dashboard (`powerbi/traffic_dashboard.pdf`) shows 40,575 hours, an average of 3,290.7 vehicles and an average temperature of 8.23 °C.

- **Daily trends 2015-2017:** a steady weekly cycle (weekdays high, weekends low) with sharp dips on holidays and a near-zero dip in July 2016 (road closure).
- **Hourly pattern 2017:** the busiest hour is 16:00 (5,834 on average) and the quietest is 03:00 (386).
- **Weather:** Clouds has the highest average (3,617) and Squall the lowest (420), a gap of 3,197. Squall covers a single hour, so this extreme is not reliable. Among conditions with at least 500 hours, the lowest is Fog (2,654), a gap of 963.
- **Temperature:** no visible relationship. Hours at every temperature from -30 °C to 35 °C span the full range of traffic. Most high-traffic hours fall between -5 °C and 25 °C, which is simply the most common temperature range. Outliers are the coldest hours near -30 °C and the closure hours near zero.

## 7. Implications for the mobility team

1. **Target the two weekday peaks.** Measures that shift even a small share of 07:00 and 16:00 trips (flexible hours, transit incentives) address almost all congestion.
2. **Do not plan around clear-sky weather.** Weather-based interventions should focus on snow, which changes traffic behaviour. Other conditions mostly reflect when they occur.
3. **Fix data collection before trusting totals.** Store one record per hour with all weather labels, monitor for multi-month outages, and validate sensor ranges at source.
4. **Use hourly averages, not totals, for year-on-year reporting** until coverage is complete.

*Sources: `results/sql_results.md`, `results/stats_probability.md`, `powerbi/traffic_dashboard.pdf`.*
