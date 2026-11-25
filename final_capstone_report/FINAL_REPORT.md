# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

**Final capstone report, NUS School of Computing AI, ML and Data Science Programme**

Westbound I-94 traffic volume near Minneapolis-St Paul, October 2012 to September 2018, with weather and US federal holidays.

| Part | Detailed report |
| --- | --- |
| 1 Data analytics | [`part1_data_analytics/INSIGHTS_REPORT.md`](../part1_data_analytics/INSIGHTS_REPORT.md) |
| 2 Python pipeline | [`part2_python/REPORT.md`](../part2_python/REPORT.md) |
| 3 Machine learning | This report, sections 4-10, and [`part3_machine_learning/README.md`](../part3_machine_learning/README.md) |
| Bias and fairness | [`BIAS_FAIRNESS_REPORT.md`](BIAS_FAIRNESS_REPORT.md) |

> **Accident data:** no accident dataset was provided or sourced. The classification task uses a documented proxy label and does not predict real accidents.

## 1. Executive summary

- **Traffic is driven by the clock.** Two weekday commuter peaks (07:00 and 16:00, over 6,000 vehicles per hour) account for almost all congestion. Every analysis, from correlation to SHAP, confirms that time of day and day of week dominate. Weather matters far less than it first appears, with snow as the clear exception.
- **Data quality had to be solved first.** The raw file repeats hours (48,204 rows, 40,575 hours), has a nine-month gap, impossible readings and faulty rain and snow sensors. Resolving these once, in a logged pipeline, gave every part the same numbers.
- **Accurate, explainable models.** Gradient boosting predicts hourly traffic for 2018 with a mean error of 217 vehicles (R² 0.966), four times better than a linear baseline. An LSTM using the previous 24 hours reaches 164 for next-hour forecasts.
- **From model to service.** The model powers a travel-time recommender and a FastAPI service. MLflow tracks every experiment and model version. Monitoring flagged real 2018 incidents, including the April blizzard.
- **Known limits.** One sensor on one corridor, a proxy risk label, and errors 2.5 times higher in snow. These are documented with mitigations in the bias and fairness report.

## 2. One project in three parts

```
raw CSV ─► Part 2 pipeline ─► traffic_clean.csv ─┬─► Part 1: SQLite, statistics, Power BI
          (cleaning, logged)   (single source)    ├─► Part 2: features, figures, CLI app
                                                   └─► Part 3: models, MLflow, API, monitoring
```

All shared rules (thresholds, weather groups, the 2018 test year, the random seed) live in `common/config.py`. The SQL view that rebuilds the hourly table and the Python pipeline agree exactly on 40,575 hours and 133,518,143 vehicles. Findings flowed forward: the Part 1 discovery that time of day confounds weather led to cyclical time features in Part 2, which SHAP confirmed as the strongest predictors in Part 3. The Part 2 CLI `recommend` command grew into the Part 3 recommendation system.

## 3. Data analytics and Python pipeline (Parts 1 and 2)

**Key analytics findings.** Yearly totals swing by up to 114% because coverage varies, but the average per hour stays between 3,194 and 3,377. Summing the raw file would overstate traffic by 17.7%. Holidays carry about 30-42% less traffic than ordinary days in the same month. Temperature and traffic correlate weakly (r = 0.14, falling to 0.09 once hour of day is removed). Clear weather appears to reduce congestion (odds ratio 0.74), but within the same hour of day the effect disappears (Mantel-Haenszel odds ratio 1.006). Snow does reduce peak congestion (36% of peak hours vs 55-58%).

**Pipeline.** Eleven logged steps load, validate, standardise, de-duplicate, collapse to one row per hour, impute impossible values (temperature by month, rainfall by weather description), flag 20 road-closure hours rather than invent target values, extend holiday labels to whole days, and report coverage gaps. Feature engineering expands 12 columns to 50 (time, cyclical, weather, scaled, targets). Five figures and a five-command CLI complete Part 2.

## 4. Supervised learning (Task 1)

**Setup.** One feature set for every model: 10 time features (including sine and cosine of hour, day of week and month), the holiday flag, 6 weather measures and 11 weather one-hot columns (28 in total). Training covers October 2012 to December 2017, testing January to September 2018. Road-closure hours are excluded.

| Regression (traffic volume) | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear regression (baseline) | 806 | 1,017 | 0.735 |
| Random forest | 222 | 376 | 0.964 |
| **Gradient boosting** | **217** | **362** | **0.966** |

| Classification (proxy high_risk) | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression (baseline) | 0.959 | 0.691 | 0.992 | 0.814 | 0.994 |
| Random forest | 0.983 | 0.848 | 0.997 | 0.916 | 0.997 |
| **Gradient boosting** | **0.985** | **0.867** | **0.992** | **0.925** | **0.998** |

**Interpretation.** The linear model cannot represent two daily peaks, so tree ensembles cut the error by a factor of four. The two ensembles perform alike, and gradient boosting was deployed because it is about 100 times smaller than the random forest (1.1 MB vs 105 MB) and faster. For classification, accuracy is misleading because always predicting "not high risk" already scores 90.9%. Balanced class weights keep recall near 0.99. The very high scores are partly circular, because the weather half of the proxy label is itself a feature.

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

**Association rules.** Time band, day type, weather group, temperature band and holiday were turned into items, and Apriori (support at least 1%, confidence at least 50%) produced 152 rules predicting a congestion quartile. The strongest:

- On weekend nights below 0 °C, traffic is Low 95% of the time (lift 3.8).
- On weekend evenings above 20 °C, traffic is High 92% of the time (lift 3.7), which captures summer leisure traffic.
- On weekday mornings at 0-20 °C, traffic is Severe about 90% of the time, whether it is raining (91%) or clear (90%). Rain does not ease the morning rush.

## 6. Deep learning with explainability (Task 3)

**LSTM.** Each input is the previous 24 hours of traffic, calendar encodings, holiday, temperature and weather severity. The target is the next hour. A sequence is kept only if its 25 hours are consecutive, which skips the data gaps and the hour lost each spring to daylight saving time. Features are standardised with training statistics only. The network (LSTM with 32 units, dense layer with 16 units) was trained for 15 epochs on 22,604 sequences.

| Next-hour forecast, 2018 | MAE | R² |
| --- | --- | --- |
| Same hour yesterday | 567 | 0.726 |
| Last observed hour | 588 | 0.830 |
| **LSTM** | **164** | **0.985** |

The LSTM beats both naive baselines by a wide margin. It is not directly comparable with gradient boosting (MAE 217), because the LSTM sees live traffic from the previous day. It suits real-time nowcasting, while gradient boosting suits forecasts for any future date.

**SHAP.** An LSTM is hard to explain directly, so, as the brief allows, SHAP TreeExplainer was applied to the gradient boosting model trained on the same target. The largest mean contributions come from hour of day (`hour_cos` 1,392 vehicles, `hour` 418), day of week (314) and the holiday flag, which pushes predictions down by up to 2,000 vehicles on holidays. Temperature and clouds contribute little. This independently confirms the Part 1 conclusion.

## 7. Advanced technique: MLflow (Task 4)

- **Why.** MLflow ties experiments to the MLOps tasks: tracking, versioning and a path to deployment. It is the most useful technique for a team that will retrain models over time.
- **How.** A local SQLite store (`part3_machine_learning/mlflow.db`) holds three experiments (regression, classification and LSTM) and all their runs. Each run logs the algorithm, hyperparameters, metrics, fit time and model signature. Models are registered in the Model Registry, so retraining creates new versions (version 2 already exists from a re-run).
- **Value.** Every number in this report can be traced to a run. Models can be compared side by side in the MLflow interface and a specific version restored. Promotion gates can be scripted against logged metrics.
- **Limitations.** A local file store is not shared across a team and has no access control. Random forest artifacts were left out because their size exceeds GitHub's limit. MLflow records what happened but does not enforce good practice, so data versions must still be tracked through Git.

## 8. Recommendation system (Task 5)

The corridor has one route, so the system recommends **when** to travel. For a given date it builds features for each hour (calendar, US federal holidays, expected weather and temperature, defaulting to the month's typical temperature), predicts hourly traffic with the deployed model, finds the lowest-traffic window inside the user's acceptable range and writes a plain-language recommendation.

| Scenario | Recommendation |
| --- | --- |
| Tuesday 2 October 2018, 06:00-11:00, clear | Travel 10:00-11:00, about 4,513 vehicles per hour, 29% below the 07:00 peak |
| Christmas Day 2018, 08:00-19:00, 2-hour trip | Travel 08:00-10:00, about 2,282, 33% below 16:00-18:00 (holiday recognised) |
| Wednesday 14 November 2018, snow, -5 °C | Travel 10:00-11:00, about 4,673, 24% below the 07:00 peak |

Example output: *"For your Tuesday 25 December 2018 journey between 08:00 and 19:00 (a public holiday), in Clear conditions around -6 °C, consider travelling 08:00-10:00. Expected traffic is about 2,282 vehicles per hour, 33% lower than the busiest time in your range."*

## 9. MLOps and deployment simulation (Task 6)

- **6.1 Versioning.** Registered models and their roles are listed in `part3_machine_learning/README.md`. The deployed model is `traffic_volume_hist_gradient_boosting`.
- **6.2 Tracking.** Parameters, metrics, model versions and experiments are in MLflow (section 7).
- **6.3 Deployment.** A FastAPI service exposes `GET /health`, `POST /predict` and `POST /recommend`, with input validation. A scripted demo sent seven requests: 08:00 on 2 October 2018 returned 5,912 vehicles (High), 03:00 returned 341 (Low), Christmas morning in snow returned 1,064, and a malformed date or unknown weather returned HTTP 422 with a clear message.
- **6.4 Monitoring.** 2018 is treated as production. Feature drift uses the Population Stability Index and KS test against January-September 2017 (same months, so seasonality is not mistaken for drift). Error drift compares each month's MAE with a baseline of 238, the error of the same model type trained to 2016 and validated on 2017.
- **6.5 Alerting.** Each scenario reports PASS / Normal or ALERT / Requires investigation, with a dashboard (`figures/monitoring_dashboard.png`).

| Scenario | Status | Reasons |
| --- | --- | --- |
| Actual 2018 | **ALERT** | January MAE 1.26× baseline (harsh winter), April 1.27× (14-15 April blizzard, when traffic fell to about 1,100 per hour) |
| Simulated 30% surge from June | **ALERT** | June to September MAE 4.2-4.4× baseline, traffic-volume PSI 0.125 |

No input feature crossed the PSI alert level. Temperature (0.136) and rainfall (0.229) reached the watch level, the latter because the rain gauge reported nothing in 2017. The monitor therefore caught a real event in the actual data and the simulated incident, while ignoring normal seasonal variation.

## 10. Responsible AI (Task 7)

The bias and fairness report details coverage limits (one sensor, a nine-month gap, faulty rain and snow gauges), the risks of the proxy label (not a crash measure, partly circular, excludes rain), and uneven errors (snow 2.5× and holidays 1.7× the overall MAE, with over-prediction in both). It sets out governance steps before real use (validated crash labels, performance gates by condition, monitored escalation, user-facing caveats) and the resource trade-offs behind choosing gradient boosting over larger models.

## 11. Recommendations for the mobility team

1. **Act on the weekday peaks.** Flexible working hours, staggered shifts and transit incentives targeted at 07:00 and 16:00 address nearly all congestion. The recommender shows that on a typical weekday, travelling at 10:00 instead of 07:00 cuts expected traffic by about 30%.
2. **Plan for snow, not for clear skies.** Snow is the only weather condition that changes traffic materially, and it is also where the models are least reliable. Pair snow-day forecasts with operational judgement.
3. **Fix the data feed.** Store one record per hour with all weather labels, validate sensor ranges at source, and repair the rain and snow gauges.
4. **Collect crash data before making safety claims.** The proxy classifier demonstrates the workflow, not real risk.
5. **Keep the monitor running.** Retrain when an ALERT is confirmed as lasting change, not on a fixed timetable.

## 12. Limitations and future work

One corridor in one direction, data ending in 2018 (before post-2020 commuting changes), unreliable precipitation measurements and a proxy risk label limit how far these results generalise. Future work: add eastbound and neighbouring sensors, real incident data, hourly weather from a second source, recent years, and probabilistic forecasts that show uncertainty in the recommender.

## Reproducibility

Run order and commands are in the root `README.md`. All code uses Python logging (`getLogger(__name__)`, console and file handlers, DEBUG/INFO/WARNING/ERROR used consistently). The project history is a sequence of descriptive Git commits, one per task.
