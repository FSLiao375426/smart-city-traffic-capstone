# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

**Final capstone report (version 2), NUS School of Computing AI, ML and Data Science Programme**

Westbound I-94 traffic volume near Minneapolis-St Paul, October 2012 to September 2018, with weather and US federal holidays.

| Part | Detailed report |
| --- | --- |
| 1 Data analytics | [`part1_data_analytics/INSIGHTS_REPORT.md`](../part1_data_analytics/INSIGHTS_REPORT.md) |
| 2 Python pipeline | [`part2_python/REPORT.md`](../part2_python/REPORT.md) |
| 3 Machine learning | This report, sections 4-10, and [`part3_machine_learning/README.md`](../part3_machine_learning/README.md) |
| Bias and fairness | [`BIAS_FAIRNESS_REPORT.md`](BIAS_FAIRNESS_REPORT.md) |
| What changed since v1 | [`V1_V2_COMPARISON.md`](V1_V2_COMPARISON.md) |

> **Accident data:** no accident dataset was provided or sourced. The classification task uses a documented proxy label and does not predict real accidents.

## 1. Executive summary

- **Traffic is driven by the clock.** Two weekday commuter peaks (07:00 and 16:00, over 6,000 vehicles per hour) account for almost all congestion. Every analysis, from correlation to SHAP, confirms that time of day and day of week dominate. Weather matters far less than it first appears, with snow as the clear exception.
- **Data quality had to be solved first.** The raw file repeats hours (48,204 rows, 40,575 hours), has a nine-month gap, impossible readings and faulty rain and snow sensors. Resolving these once, in a logged pipeline, gave every part the same numbers.
- **Models chosen fairly, then tested once.** Models were chosen on 2017 with a rule written down in advance, then scored once on 2018. Gradient boosting predicts hourly traffic for 2018 with a mean error of 213 vehicles (R² 0.969), four times better than a linear model and eight times better than a no-skill baseline. An LSTM using the previous 24 hours reaches 152 for next-hour forecasts.
- **From model to service.** The model powers a travel-time recommender and a FastAPI service. MLflow tracks every experiment and model version. Monitoring flagged the real April 2018 blizzard and a simulated traffic surge.
- **Known limits.** One sensor on one corridor, a proxy risk label, and errors almost three times higher when snow falls with poor visibility. These are documented with mitigations in the bias and fairness report.

## 2. One project in three parts

```
raw CSV ─► Part 2 pipeline ─► traffic_clean.csv ─┬─► Part 1: SQLite, statistics, Power BI
          (cleaning, logged)   (single source)    ├─► Part 2: features, figures, CLI app
                                                   └─► Part 3: models, MLflow, API, monitoring
```

All shared rules (thresholds, weather groups, the training, validation and test periods, the random seed) live in `common/config.py`. The SQL view that rebuilds the hourly table and the Python pipeline agree exactly on 40,575 hours and 133,518,143 vehicles. Findings flowed forward: the Part 1 discovery that time of day confounds weather led to cyclical time features in Part 2, which SHAP confirmed as the strongest predictors in Part 3. The Part 2 CLI `recommend` command grew into the Part 3 recommendation system.

`python run_all.py` reproduces every part in order. 27 unit tests check the cleaning, feature and model-selection rules, and GitHub Actions runs them after every push.

## 3. Data analytics and Python pipeline (Parts 1 and 2)

**Key analytics findings.** Yearly totals swing by up to 114% because coverage varies, but the average per hour stays between 3,194 and 3,377. Summing the raw file would overstate traffic by 17.7%. Holidays carry about 30-42% less traffic than ordinary days in the same month. Temperature and traffic correlate weakly (r = 0.14, falling to 0.09 once hour of day is removed). Clear weather appears to reduce congestion (odds ratio 0.74), but within the same hour of day the effect disappears (Mantel-Haenszel odds ratio 1.006). Snow does reduce peak congestion (36% of peak hours vs 55-58%).

**Pipeline.** Eleven logged steps load, validate, standardise, de-duplicate, collapse to one row per hour, impute impossible values (temperature by month, rainfall by weather description), flag 20 road-closure hours rather than invent target values, extend holiday labels to whole days, and report coverage gaps. Each hour keeps its most severe weather for charts and all of its weather labels for models: 5,349 hours (13%) have two or more conditions, often rain or snow with mist. A quality report records the raw file's fingerprint and what each step changed. Feature engineering expands 13 columns to 51, and the values it learns from data (quartile thresholds, scaling parameters) are fitted on the training years only. Five figures and a five-command CLI complete Part 2.

## 4. Supervised learning (Task 1)

**Setup.** One feature set for every model: 10 time features (including sine and cosine of hour, day of week and month), the holiday flag, 6 weather measures and 11 multi-hot weather columns (28 in total). Road-closure hours are excluded. The data is split by time into three periods:

| Period | Hours | Use |
| --- | --- | --- |
| Training, Oct 2012 to Dec 2016 | 25,309 | Fit every candidate model |
| Validation, 2017 | 8,713 | Choose one model per task |
| Test, Jan to Sep 2018 | 6,533 | Score the chosen model once, after refitting on 2012-2017 |

**Selection rule.** Written in `supervised.py` before any score was seen: the best validation score wins (lowest MAE, highest F1), and models within 2% of the best count as a tie that goes to the simpler one (linear, then gradient boosting, then random forest). The choice is saved to `model_selection.json` before the test year is scored.

| Regression (traffic volume) | Validation MAE | Test MAE | Test RMSE | Test R² |
| --- | --- | --- | --- | --- |
| Dummy (training mean) | 1,744 | 1,728 | 1,975 | 0.000 |
| Linear regression | 802 | 806 | 1,017 | 0.735 |
| Random forest | 239 | 220 | 372 | 0.964 |
| **Gradient boosting (chosen)** | **236** | **213** | **350** | **0.969** |

| Classification (proxy high_risk) | Validation F1 | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy ("not high risk") | 0 | 0.889 | 0 | 0 | 0 | 0.500 |
| Logistic regression | 0.800 | 0.919 | 0.581 | 0.979 | 0.729 | 0.982 |
| Random forest | 0.954 | 0.982 | 0.866 | 0.996 | 0.926 | 0.997 |
| **Gradient boosting (chosen)** | **0.957** | **0.984** | **0.884** | **0.983** | **0.931** | **0.998** |

**Interpretation.** Gradient boosting had the best validation score in both tasks, so it was chosen without needing the tie rule. It is also about 70 times smaller than the random forest (1.5 MB vs about 105 MB). The linear model cannot represent two daily peaks, so tree ensembles cut the error by a factor of four. The dummy classifier is right 88.9% of the time while finding no high-risk hour at all, which is why accuracy is not used to judge classifiers. Validation error is higher than test error because the validation models had one year less data and predicted a full year ahead. The very high classification scores are partly circular, because the weather half of the proxy label is itself a feature.

## 5. Unsupervised learning (Task 2)

**K-means.** Hours were clustered on hour of day (sine and cosine), weekend flag, weather severity and traffic volume, all standardised. Silhouette rose slowly with k and showed no clear elbow, so k = 5 was chosen as the best score among sizes that remain interpretable (0.356, above k = 6).

| Cluster | Share of hours | Avg volume | Congested |
| --- | --- | --- | --- |
| Weekend off-peak hours, light traffic | 13% | 1,200 | 0% |
| Weekday off-peak hours, light traffic | 29% | 1,473 | 0% |
| Weekend daytime, busy traffic | 14% | 3,910 | 1% |
| Weekday mixed hours, busy traffic, rain or snow | 14% | 4,241 | 24% |
| Weekday daytime, congested traffic, fair weather | 30% | 5,273 | 40% |

The algorithm isolated a bad-weather weekday group. Its hours are busy but congested far less often (24%) than fair-weather weekday daytime (40%), echoing the snow effect found in Part 1.

**Association rules.** Time band, day type, weather group, temperature band and holiday were turned into items, and Apriori (support at least 1%, confidence at least 50%) produced 153 rules predicting a congestion quartile. The strongest:

- On weekend nights below 0 °C, traffic is Low 94% of the time (lift 3.8).
- On weekend evenings above 20 °C, traffic is High 93% of the time (lift 3.6), which captures summer leisure traffic.
- On weekday mornings at 0-20 °C, traffic is Severe 91% of the time whether it is raining or clear. Rain does not ease the morning rush.

## 6. Deep learning with explainability (Task 3)

**LSTM.** Each input is the previous 24 hours of traffic, calendar encodings, holiday, temperature and weather severity. The target is the next hour. A sequence is kept only if its 25 hours are consecutive, which skips the data gaps and the hour lost each spring to daylight saving time. Features are standardised with training statistics only. The network (LSTM with 32 units, dense layer with 16 units) follows the same split as the other models: early stopping on 2017 chose 39 epochs (validation MAE 163), then the model was refitted on 2012-2017 (22,604 sequences) and scored once on 2018.

| Next-hour forecast, 2018 | MAE | R² |
| --- | --- | --- |
| Same hour yesterday | 567 | 0.726 |
| Last observed hour | 588 | 0.830 |
| **LSTM** | **152** | **0.987** |

The LSTM beats both naive baselines by a wide margin. It is not directly comparable with gradient boosting (MAE 213), because the LSTM sees live traffic from the previous day. It suits real-time nowcasting, while gradient boosting suits forecasts for any future date.

**SHAP on the LSTM.** GradientExplainer was applied directly to the LSTM on 300 test sequences. Recent traffic is by far the strongest input (mean contribution about 2,000 vehicles), followed by the hour-of-day encodings (about 500 each). The last hour carries 61% of all attribution and the last three hours 80%. The same hour yesterday carries only 0.2%, so the LSTM is essentially a short-term trend follower.

**SHAP on the deployed model.** TreeExplainer was applied to the gradient boosting regressor, which forecasts without live traffic. The largest mean contributions come from hour of day (`hour_cos` 1,391 vehicles, `hour` 420) and day of week (317). Temperature contributes 62 and weather even less. This independently confirms the Part 1 conclusion.

## 7. Advanced technique: MLflow (Task 4)

- **Why.** MLflow ties experiments to the MLOps tasks: tracking, versioning and a path to deployment. It is the most useful technique for a team that will retrain models over time.
- **How.** A local SQLite store (`part3_machine_learning/mlflow.db`) holds three experiments (regression, classification and LSTM). Each run is tagged with its stage (validation or test) and logs the algorithm, hyperparameters, metrics, fit time and model signature. Models are registered in the Model Registry under the same names in every run, so retraining creates new versions: versions 1-2 come from v1 and versions 3-4 from v2.
- **Value.** Every number in this report can be traced to a run. Models can be compared side by side in the MLflow interface and a specific version restored. The v1 and v2 models sit in one registry, which made the version comparison straightforward.
- **Limitations.** A local file store is not shared across a team and has no access control. Random forest artifacts were left out because their size exceeds GitHub's limit. MLflow records what happened but does not enforce good practice, so data versions are tracked through Git and the quality report's fingerprint.

## 8. Recommendation system (Task 5)

The corridor has one route, so the system recommends **when** to travel. For a given date it builds features for each hour (calendar, US federal holidays, expected weather and temperature, defaulting to the month's typical temperature), predicts hourly traffic with the deployed model, finds the lowest-traffic window inside the user's acceptable range and writes a plain-language recommendation. It also counts how many past hours match the request (same weather, day type and hours). Below 30, the recommendation is still given but marked as low confidence.

| Scenario | Recommendation |
| --- | --- |
| Tuesday 2 October 2018, 06:00-11:00, clear | Travel 10:00-11:00, about 4,521 vehicles per hour, 30% below the 07:00 peak |
| Christmas Day 2018, 08:00-19:00, 2-hour trip | Travel 08:00-10:00, about 1,914, 42% below 15:00-17:00 (holiday recognised) |
| Wednesday 14 November 2018, snow, -5 °C | Travel 10:00-11:00, about 4,853, 25% below the 07:00 peak |
| Wednesday 14 November 2018, squall | Same kind of answer, with a caution: no past weekday hours in that range had a squall |

Example output: *"For your Tuesday 25 December 2018 journey between 08:00 and 19:00 (a public holiday), in Clear conditions around -6 °C, consider travelling 08:00-10:00. Expected traffic is about 1,914 vehicles per hour, 42% lower than the busiest time in your range."*

## 9. MLOps and deployment simulation (Task 6)

- **6.1 Versioning.** Registered models, their roles and versions are listed in `part3_machine_learning/README.md`. The deployed model is `traffic_volume_hist_gradient_boosting`. The Git tags `v1.0` and `v2.0` mark the two versions of the whole project.
- **6.2 Tracking.** Parameters, metrics, model versions and experiments are in MLflow (section 7).
- **6.3 Deployment.** A FastAPI service exposes `GET /health`, `POST /predict` and `POST /recommend`, with input validation. `/health` reports the deployed model's test error, read from the results file. A scripted demo sent seven requests: 08:00 on 2 October 2018 returned 5,900 vehicles (High), 03:00 returned 351 (Low), Christmas morning in snow returned 709, and a malformed date or unknown weather returned HTTP 422 with a clear message.
- **6.4 Monitoring.** 2018 is treated as production. Feature drift uses the Population Stability Index and KS test against January-September 2017 (same months, so seasonality is not mistaken for drift). Error drift compares each month's MAE with a baseline of 236, the chosen model's 2017 validation error read from `model_selection.json`.
- **6.5 Alerting.** Each scenario reports PASS / Normal or ALERT / Requires investigation, with a dashboard (`figures/monitoring_dashboard.png`).

| Scenario | Status | Reasons |
| --- | --- | --- |
| Actual 2018 | **ALERT** | April MAE 1.27× baseline (14-15 April blizzard, when traffic fell to about 1,100 per hour). January reached 1.23×, just below the 1.25 threshold |
| Simulated 30% surge from June | **ALERT** | June to September MAE 4.2-4.4× baseline, traffic-volume PSI 0.125 |

No input feature crossed the PSI alert level. Temperature (0.136) and rainfall (0.229) reached the watch level, the latter because the rain gauge reported nothing in 2017. The monitor therefore caught a real event in the actual data and the simulated incident, while ignoring normal seasonal variation.

## 10. Responsible AI (Task 7)

The bias and fairness report details coverage limits (one sensor, a nine-month gap, faulty rain and snow gauges), the risks of the proxy label (not a crash measure, partly circular, ignores rain unless visibility is also poor), and uneven errors. The largest error appears when snow falls with poor visibility: 2.9 times the overall MAE, with traffic over-predicted by about 450 vehicles per hour. Holidays reach 1.7 times. Poor visibility without snow is predicted better than average. The report sets out governance steps before real use (validated crash labels, performance gates by condition, monitored escalation, user-facing caveats) and the resource trade-offs behind choosing gradient boosting over larger models.

## 11. Recommendations for the mobility team

1. **Act on the weekday peaks.** Flexible working hours, staggered shifts and transit incentives targeted at 07:00 and 16:00 address nearly all congestion. The recommender shows that on a typical weekday, travelling at 10:00 instead of 07:00 cuts expected traffic by about 30%.
2. **Plan for snowstorms, not for clear skies.** Snow is the only weather condition that changes traffic materially, and snow with poor visibility is where the models are least reliable. Pair storm-day forecasts with operational judgement.
3. **Fix the data feed.** Store one record per hour with all weather labels, validate sensor ranges at source, and repair the rain and snow gauges.
4. **Collect crash data before making safety claims.** The proxy classifier demonstrates the workflow, not real risk.
5. **Keep the monitor running.** Retrain when an ALERT is confirmed as lasting change, not on a fixed timetable.

## 12. Limitations and future work

One corridor in one direction, data ending in 2018 (before post-2020 commuting changes), unreliable precipitation measurements and a proxy risk label limit how far these results generalise. A single validation year gives one estimate of model error; rolling validation over several years would show how stable it is. Future work: add eastbound and neighbouring sensors, real incident data, hourly weather from a second source, recent years, and probabilistic forecasts that show uncertainty in the recommender.

## Reproducibility

`python run_all.py` runs every step in order; the commands are listed in the root `README.md`. `requirements-lock.txt` pins the exact package versions behind these results, and `data/processed/quality_report.json` fingerprints the raw file. All code uses Python logging (`getLogger(__name__)`, console and file handlers, DEBUG/INFO/WARNING/ERROR used consistently). The project history is a sequence of descriptive Git commits, one per task, with v1 tagged `v1.0` and v2 developed on its own branch.
