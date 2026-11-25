# Bias, Fairness, Governance and Sustainability Report

**Smart City Traffic Intelligence: westbound I-94 traffic models**

This report covers Part 3 Task 7. It describes where the data and models are limited, who could be served badly by them, and what oversight is needed before they inform real decisions. Error figures come from `part3_machine_learning/results/fairness_error_analysis.md` (2018 test set, gradient boosting regressor and classifier).

## Summary

| Risk | Evidence | Severity |
| --- | --- | --- |
| Coverage: one sensor, one direction, gaps | 40,575 of 52,551 possible hours. Nine months missing (Sep 2014 to May 2015) | High for generalisation |
| Faulty weather sensors | Rainfall recorded as 0 for all of 2012 and 2017. Snowfall 0 during the April 2018 blizzard | High for weather features |
| Proxy label is not accidents | Label defined from traffic volume and weather categories, no crash data | High if misread |
| Proxy label excludes rain | Rain and drizzle can never be "high risk" by construction | Medium |
| Errors concentrate in snow | Snow MAE 541 (2.5× overall) with systematic over-prediction | High for winter use |
| Errors on holidays | Holiday MAE 377 (1.7× overall) | Medium |

## 1. Sampling and coverage limitations

- **One sensor, one direction.** All data comes from a single westbound counting station on I-94. Eastbound traffic, other corridors, arterials, transit, cyclists and pedestrians are absent. The models describe one road segment and cannot be applied city-wide.
- **Time coverage is uneven.** Only 77% of possible hours exist. September 2014 to May 2015 is missing entirely, so one winter is lost and winter patterns rest on fewer years. 2012 starts in October and 2018 ends in September.
- **Repeated hours.** The raw file repeats an hour for every weather label (48,204 rows for 40,575 hours). Without collapsing them, bad-weather hours would be counted up to six times and dominate every statistic.
- **Sensor faults in the weather data.** Hourly rainfall is zero for every hour of 2012 and 2017, and 63% of hours labelled Rain record 0 mm overall. During the 14-15 April 2018 blizzard, snowfall was recorded as 0 while the weather label was Snow. The numeric rain and snow features therefore under-represent bad weather, and the models rely mostly on the weather category.
- **Road closures.** 20 daytime hours with near-zero traffic on four summer weekends were flagged and excluded from training. Closures are real events that a deployed system would face, so the models have no experience of them.
- **Temporal drift.** The data ends in 2018. Commuting changed substantially after 2020 (remote work), so 2012-2018 patterns may no longer hold. The monitoring module is designed to detect this.

## 2. Proxy label and its risks

No accident dataset was available. The classification target is a proxy:

> `high_risk` = 1 when congestion is High or Severe (traffic in the top half) **and** the weather is Thunderstorm, Snow or Squall, or visibility is low (Fog, Mist, Haze, Smoke).

Risks of this label:

1. **It is not a crash label.** It measures "busy road in bad weather", a plausible risk factor, not an outcome. A model with F1 0.93 on this label says nothing about its ability to predict accidents.
2. **The high scores are partly circular.** The weather part of the label is itself an input feature, so the classifier only has to learn when traffic is busy. This explains the near-perfect ROC AUC (0.998) and is not evidence of a strong risk model.
3. **Rain is excluded by design.** Rain and drizzle are in neither the severe nor the low-visibility group, so 775 rainy test hours have zero positives. Wet roads are a well-known crash factor, so the proxy systematically ignores a risky condition.
4. **Thresholds are data-driven, not safety-driven.** "High" congestion means the top half of this sensor's volumes, not a capacity or speed threshold tied to crash risk.
5. **Risk of misuse.** If presented to decision makers as "accident likelihood", the output could misdirect enforcement or warnings. Every output of this model must carry the proxy caveat.

## 3. How errors are distributed

Overall 2018 regression MAE is about 217 vehicles per hour. Errors are not even across conditions:

| Condition | Hours | MAE | vs overall | Mean error | What it means |
| --- | --- | --- | --- | --- | --- |
| **Snow** | 516 | 541 | **2.5×** | +291 | The model expects far more traffic than occurs on snowy hours |
| **Holidays** | 167 | 377 | **1.7×** | +121 | Few holidays per year to learn from, traffic over-predicted |
| Evening peak (15-18) | 1,092 | 313 | 1.4× | -13 | Largest volumes, largest absolute errors |
| April 2018 | 720 | 304 | 1.4× | | Includes the 14-15 April blizzard |
| Weekends | 1,862 | 270 | 1.2× | | Fewer weekend hours in training |
| Clear | 2,328 | 190 | 0.9× | -2 | Well served |
| Night (0-5) | 1,624 | 86 | 0.4× | -3 | Small absolute error, but 10% of the low volume |

**Who is affected.** The recommender is least reliable exactly when it matters most: in snow and on holidays. Over-predicting traffic in snow means a traveller may be told that a quiet road is busy, which is a safe direction of error, but the same bias would mislead a planner sizing snow-clearing or transit services. Errors in relative terms are highest at night (10% of volume), affecting shift workers who travel at those hours.

**Proxy classifier.** For snow, precision is 0.74 with a 23% false positive rate, and for thunderstorms the false positive rate is 12.5%. Low-visibility conditions are classified almost perfectly. Rain, Clear and Clouds hours have no positives by construction (see section 2).

**Mitigations.** Report errors by condition with every model release, add more winter and holiday data (or holiday-specific features such as "day before holiday"), repair or replace the faulty rain and snow inputs, widen the recommender's output to show uncertainty in snow, and keep the closure flag as a monitored event type.

## 4. Governance: oversight before real-world use

Before any of these models informs a real decision:

1. **Human review of the use case.** A traffic engineer and a data owner agree on the decisions the model may support (travel-time advice: yes. Enforcement or resource allocation: not without further validation).
2. **Label validation.** Replace the proxy with real crash and incident data, joined by time and location, before any safety claim is made.
3. **Performance gates.** A model version is promoted only if its error by condition (snow, holiday, peak, night) is within agreed limits, not just its overall MAE. These checks run from MLflow before deployment.
4. **Monitoring with escalation.** The PASS / ALERT monitor runs monthly. An ALERT (as in January and April 2018) triggers a review by a named owner within an agreed time and can pause the recommender.
5. **Transparency to users.** Recommendations state that they are based on historical patterns for one corridor and may be unreliable in severe weather and on holidays.
6. **Version control and audit trail.** Every model is registered in MLflow with its parameters, metrics, data period and code version (Git commit). Logs record every prediction request (`api.log`).
7. **Data governance.** The current data holds no personal information. Adding vehicle-level or mobile-phone data would require a privacy impact assessment.

## 5. Sustainability: resource trade-offs

| Model | Training time (laptop CPU) | Model size | Test MAE |
| --- | --- | --- | --- |
| Linear regression | 0.1 s | 2 KB | 806 |
| Gradient boosting | 1.3 s | 1.1 MB | 217 |
| Random forest | 3.5 s | about 105 MB | 222 |
| LSTM (15 epochs) | 35 s | 0.1 MB | 164 (next-hour, uses live traffic) |

- **Choose the smallest model that meets the need.** Gradient boosting matches the random forest's accuracy at a fraction of the size and time, which is why it is deployed. The random forest's size was also the reason its artifacts are kept out of MLflow.
- **Deep learning has a running cost.** The LSTM is more accurate for next-hour forecasting, but needs live traffic feeds, more compute to retrain, and GPU acceleration at larger scale. Its 25% error reduction is worth it only if real-time decisions depend on it.
- **Retraining cadence.** Retraining every month costs seconds for the tree model. Retraining should be triggered by monitoring alerts rather than run continuously.
- **The larger sustainability lever is the purpose.** Shifting even a small share of peak-hour trips through better timing advice reduces idling and emissions far more than any saving in model compute.
