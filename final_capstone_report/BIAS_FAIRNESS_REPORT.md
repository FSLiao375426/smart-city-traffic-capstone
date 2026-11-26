# Bias, Fairness, Governance and Sustainability Report

**Smart City Traffic Intelligence: westbound I-94 traffic models (version 2)**

This report covers Part 3 Task 7. It describes where the data and models are limited, who could be served badly by them, and what oversight is needed before they inform real decisions. Error figures come from `part3_machine_learning/results/fairness_error_analysis.md` (2018 test set, gradient boosting regressor and classifier, chosen on 2017 and scored once on 2018).

## Summary

| Risk | Evidence | Severity |
| --- | --- | --- |
| Coverage: one sensor, one direction, gaps | 40,575 of 52,551 possible hours. Nine months missing (Sep 2014 to May 2015) | High for generalisation |
| Faulty weather sensors | Rainfall recorded as 0 for all of 2012 and 2017. Snowfall 0 during the April 2018 blizzard | High for weather features |
| Proxy label is not accidents | Label defined from traffic volume and weather categories, no crash data | High if misread |
| Proxy label ignores plain rain | Rain counts as risky only when visibility is also poor | Medium |
| Errors concentrate in snowstorms | Snow with poor visibility: MAE 616 (2.9× overall), traffic over-predicted by 454 vehicles per hour | High for winter use |
| Errors on holidays | Holiday MAE 369 (1.7× overall) | Medium |

## 1. Sampling and coverage limitations

- **One sensor, one direction.** All data comes from a single westbound counting station on I-94. Eastbound traffic, other corridors, arterials, transit, cyclists and pedestrians are absent. The models describe one road segment and cannot be applied city-wide.
- **Time coverage is uneven.** Only 77% of possible hours exist. September 2014 to May 2015 is missing entirely, so one winter is lost and winter patterns rest on fewer years. 2012 starts in October and 2018 ends in September.
- **Repeated hours.** The raw file repeats an hour for every weather label (48,204 rows for 40,575 hours). Without collapsing them, bad-weather hours would be counted up to six times and dominate every statistic. The pipeline keeps one row per hour, but records every weather label of the hour so that co-occurring conditions such as snow with mist are not lost.
- **Sensor faults in the weather data.** Hourly rainfall is zero for every hour of 2012 and 2017, and 63% of hours labelled Rain record 0 mm overall. During the 14-15 April 2018 blizzard, snowfall was recorded as 0 while the weather label was Snow. The numeric rain and snow features therefore under-represent bad weather, and the models rely mostly on the weather categories.
- **Road closures.** 20 daytime hours with near-zero traffic on four summer weekends were flagged and excluded from training. Closures are real events that a deployed system would face, so the models have no experience of them.
- **Temporal drift.** The data ends in 2018. Commuting changed substantially after 2020 (remote work), so 2012-2018 patterns may no longer hold. The monitoring module is designed to detect this.

## 2. Proxy label and its risks

No accident dataset was available. The classification target is a proxy:

> `high_risk` = 1 when congestion is High or Severe (traffic above the training-period median) **and** any of the hour's weather labels is Thunderstorm, Snow or Squall, or a low-visibility condition (Fog, Mist, Haze, Smoke).

It is positive for 3,987 hours (9.8%). Version 1 looked only at the most severe label of each hour, which hid mist and fog behind rain and missed 784 risky hours.

Risks of this label:

1. **It is not a crash label.** It measures "busy road in bad weather", a plausible risk factor, not an outcome. A model with F1 0.93 on this label says nothing about its ability to predict accidents.
2. **The high scores are partly circular.** The weather part of the label is itself an input feature, so the classifier mainly has to learn when traffic is busy. This explains the near-perfect ROC AUC (0.998) and is not evidence of a strong risk model.
3. **Plain rain is ignored.** Rain and drizzle are in neither the severe nor the low-visibility group. A rainy hour becomes high risk only if mist or fog is reported at the same time (116 of the 775 rainy test hours). Wet roads are a well-known crash factor, so the proxy still under-counts a risky condition.
4. **Thresholds are data-driven, not safety-driven.** "High" congestion means the top half of this sensor's volumes in 2012-2016, not a capacity or speed threshold tied to crash risk.
5. **Risk of misuse.** If presented to decision makers as "accident likelihood", the output could misdirect enforcement or warnings. Every output of this model must carry the proxy caveat.

## 3. How errors are distributed

Overall 2018 regression MAE is 213 vehicles per hour. Errors are not even across conditions:

| Condition | Hours | MAE | vs overall | Mean error | What it means |
| --- | --- | --- | --- | --- | --- |
| **Snow with poor visibility** | 320 | 616 | **2.9×** | +454 | Snowstorms: far less traffic than the model expects, as people stay home |
| Snow, normal visibility | 201 | 319 | 1.5× | +22 | Light snow is predicted reasonably well |
| **Holidays** | 167 | 369 | **1.7×** | +134 | Few holidays per year to learn from, traffic over-predicted |
| April 2018 | 720 | 298 | 1.4× | +105 | Includes the 14-15 April blizzard |
| Evening peak (15-18) | 1,092 | 296 | 1.4× | +15 | Largest volumes, largest absolute errors |
| Weekends | 1,862 | 261 | 1.2× | +4 | Fewer weekend hours in training |
| Clear | 2,328 | 189 | 0.9× | +5 | Well served |
| Poor visibility, no snow | 935 | 160 | 0.75× | -11 | Mist and fog alone are predicted better than average |
| Night (0-5) | 1,624 | 86 | 0.4× | -3 | Small absolute error, but 10% of the low volume |

**Snow and visibility.** Keeping every weather label made it possible to separate snow from the poor visibility that often comes with it. The two together, not snow alone, drive the largest errors. Version 1 could only report "snow, 2.5×" because mist was hidden behind snow.

**Who is affected.** The recommender is least reliable exactly when it matters most: in snowstorms and on holidays. Over-predicting traffic in a snowstorm means a traveller may be told that a quiet road is busy, which is a safe direction of error, but the same bias would mislead a planner sizing snow-clearing or transit services. Errors in relative terms are highest at night (10% of volume), affecting shift workers who travel at those hours.

**Proxy classifier.** For snow, precision is 0.77 with a 21% false positive rate, and for thunderstorms the false positive rate is 14%. On weekends precision falls to 0.70, and in the evening (19 positive hours) to 0.35. Low-visibility conditions are classified almost perfectly. Clear and Clouds hours have no positives by construction, so their recall and precision are reported as n/a rather than 0% or 100%.

**Mitigations.** Report errors by condition with every model release, add more winter and holiday data (or holiday-specific features such as "day before holiday"), repair or replace the faulty rain and snow inputs, show uncertainty in the recommender during snowstorms, and keep the closure flag as a monitored event type. The recommender already marks requests with too little history (fewer than 30 matching past hours, for example squalls) as low confidence.

## 4. Governance: oversight before real-world use

Before any of these models informs a real decision:

1. **Human review of the use case.** A traffic engineer and a data owner agree on the decisions the model may support (travel-time advice: yes. Enforcement or resource allocation: not without further validation).
2. **Label validation.** Replace the proxy with real crash and incident data, joined by time and location, before any safety claim is made.
3. **Honest model selection.** Models are chosen on a validation year with a rule written down in advance, and the test year is used once. Version 1 chose its model on the test year, which would have hidden over-optimism if the choice had been close.
4. **Performance gates.** A model version is promoted only if its error by condition (snowstorm, holiday, peak, night) is within agreed limits, not just its overall MAE. These checks run from MLflow before deployment.
5. **Monitoring with escalation.** The PASS / ALERT monitor runs monthly. An ALERT (as in April 2018) triggers a review by a named owner within an agreed time and can pause the recommender.
6. **Transparency to users.** Recommendations state that they are based on historical patterns for one corridor and may be unreliable in severe weather and on holidays.
7. **Version control and audit trail.** Every model is registered in MLflow with its parameters, metrics, data period and code version (Git commit). The quality report fingerprints the raw data. Logs record every prediction request (`api.log`).
8. **Data governance.** The current data holds no personal information. Adding vehicle-level or mobile-phone data would require a privacy impact assessment.

## 5. Sustainability: resource trade-offs

| Model | Training time (laptop CPU) | Model size | Test MAE |
| --- | --- | --- | --- |
| Linear regression | 0.1 s | 2 KB | 806 |
| Gradient boosting | 4 s | 1.5 MB | 213 |
| Random forest | 10 s | about 105 MB | 220 |
| LSTM (39 epochs, two stages) | about 2.5 minutes | 0.1 MB | 152 (next-hour, uses live traffic) |

- **Choose the smallest model that meets the need.** Gradient boosting matches the random forest's accuracy at a fraction of the size and time, and won on the validation year as well. The random forest's size was also the reason its artifacts are kept out of MLflow.
- **Deep learning has a running cost.** The LSTM is more accurate for next-hour forecasting, but needs live traffic feeds, more compute to retrain, and GPU acceleration at larger scale. Choosing its epochs on a validation year (39 instead of a fixed 15) improved it, but made training about four times longer (35 seconds in v1). That is worth it only if real-time decisions depend on it.
- **Retraining cadence.** Retraining the tree model takes seconds. Retraining should be triggered by monitoring alerts rather than run continuously.
- **The larger sustainability lever is the purpose.** Shifting even a small share of peak-hour trips through better timing advice reduces idling and emissions far more than any saving in model compute.
