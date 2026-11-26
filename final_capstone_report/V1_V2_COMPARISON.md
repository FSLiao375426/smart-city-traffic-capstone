# Version 1 vs version 2: what changed and what it changed

v1 is the first complete submission (Git tag `v1.0`). v2 fixes the method problems found in a review of v1, keeps every v1 deliverable, and re-runs everything on the same data.

**Bottom line.** No main conclusion changed. Time of day still dominates traffic, gradient boosting is still the deployed model, and the recommender still points to mid-morning. What changed is how far the numbers can be trusted: the model is now chosen on a year it is not tested on, labels no longer depend on the test year, and the error analysis can now see that snowstorms with poor visibility, not snow alone, are where the model fails.

## 1. Why a version 2

A side-by-side review of this project and an independent implementation of the same brief found five gaps in v1:

1. **The model was chosen on the test year.** Gradient boosting was deployed because it beat the random forest on 2018, so the 2018 score was no longer an unbiased test.
2. **Quartile thresholds used the test year.** The congestion quartiles (and with them the proxy label) were computed on all years, including 2018.
3. **Only one weather label per hour.** Keeping only the most severe label hid mist and fog behind rain and snow.
4. **No "no-skill" baseline** to show what a model that learns nothing would score.
5. **No automated tests**, so a later change could silently break a cleaning rule.

## 2. What changed

| ID | Area | v1 | v2 |
| --- | --- | --- | --- |
| D19 | Model choice | Train to 2017, test 2018, choose on 2018 | Train to 2016, choose on 2017 with a rule fixed in advance, refit to 2017, test 2018 once |
| D19 | LSTM training | Fixed 15 epochs, early stopping on the last 10% of training data | Epochs chosen by early stopping on 2017 (39), then refitted to 2017 |
| D19 | Baselines | Linear models only | Dummy models added (training mean, always "not high risk") |
| D20 | Learned thresholds | Quartiles and scaling fitted on all years | Fitted on 2012-2016 only, saved to `feature_params.json` |
| D21 | Weather | Most severe label only | All labels kept (`weather_labels`), multi-hot `wx_*`, flags use all labels |
| | Explainability | SHAP on gradient boosting only | Also SHAP directly on the LSTM (GradientExplainer) |
| | Recommender | No check on how much history supports a request | Low-confidence flag when fewer than 30 matching past hours |
| | Error analysis | By the most severe weather | Also by every label present, by visibility, and snow by visibility. Empty denominators reported as n/a |
| | Monitoring | Baseline recomputed separately (238.0) | Baseline read from the frozen model choice (235.9) |
| | Engineering | Fresh-clone check by hand | 27 unit tests, GitHub Actions, `run_all.py`, quality report with raw-file fingerprint, `requirements-lock.txt` |

## 3. Results side by side

All v2 numbers come from the committed result files. The test set is the same 6,533 hours of January to September 2018 in both versions.

### 3.1 Data and labels

| | v1 | v2 | Why it changed |
| --- | --- | --- | --- |
| Hours, columns after cleaning | 40,575 × 12 | 40,575 × 13 | `weather_labels` added |
| Feature table columns | 50 | 51 | Same |
| Congestion quartiles Q1 / Q2 / Q3 | 1,248.5 / 3,427 / 4,952 | 1,204 / 3,339 / 4,932 | Fitted on 2012-2016 only (D20) |
| Low-visibility hours | 3,408 | 6,737 | Mist and fog no longer hidden behind rain or snow (D21) |
| `high_risk` positive hours | 3,133 (7.7%) | 3,987 (9.8%) | +784 from D21, +57 from D20, +13 from both together |

The v2 thresholds and label counts by period (2,257 training, 1,003 validation, 727 test positives) match an independent implementation of the same brief exactly, which cross-checks both.

### 3.2 Regression

The target and test hours are identical, and both final models are trained on 2012-2017, so these numbers are directly comparable.

| Model | v1 test MAE | v2 validation MAE | v2 test MAE | v2 test R² |
| --- | --- | --- | --- | --- |
| Dummy (training mean) | | 1,744 | 1,728 | 0.000 |
| Linear regression | 806 | 802 | 806 | 0.735 |
| Random forest | 222 | 239 | 220 | 0.964 |
| **Gradient boosting** | **217** | **236** | **213** | **0.969** |

- **The v1 choice survives.** Gradient boosting also wins on 2017, so v1 picked the right model, but it could not have known that without a validation year.
- **The honest pre-deployment estimate is 236, not 217.** That is what a team could have expected before seeing 2018. The test year then came in better than expected.
- **Keeping all weather labels helped slightly.** Test MAE fell from 217 to 213 for gradient boosting and from 222 to 220 for the random forest, with the same training period.

### 3.3 Classification

The proxy label itself changed (section 3.1), so these numbers are **not** directly comparable: v2 has 27% more positive hours, many of them harder rain-with-mist cases.

| Model | v1 F1 | v2 validation F1 | v2 test F1 | v2 precision | v2 recall |
| --- | --- | --- | --- | --- | --- |
| Dummy ("not high risk") | | 0 | 0 (accuracy 0.889) | 0 | 0 |
| Logistic regression | 0.814 | 0.800 | 0.729 | 0.581 | 0.979 |
| Random forest | 0.916 | 0.954 | 0.926 | 0.866 | 0.996 |
| **Gradient boosting** | **0.925** | **0.957** | **0.931** | **0.884** | **0.983** |

### 3.4 Deep learning and explainability

| | v1 | v2 |
| --- | --- | --- |
| LSTM epochs | 15 (fixed) | 39 (chosen on 2017) |
| LSTM test MAE, R² | 164, 0.985 | **152, 0.987** |
| Naive baselines (last hour, same hour yesterday) | 588, 567 | 588, 567 |
| SHAP on the LSTM | Not done | Recent traffic dominates; last 3 hours carry 80% of the attribution, the same hour yesterday 0.2% |
| SHAP on gradient boosting | `hour_cos` 1,392, `hour` 418, day of week 314 | `hour_cos` 1,391, `hour` 420, day of week 317 |

v1's fixed 15 epochs stopped the LSTM while it was still improving. Letting a validation year decide cut its error by 7%.

### 3.5 Monitoring

| | v1 | v2 |
| --- | --- | --- |
| Baseline MAE | 238.0 | 235.9 |
| Actual 2018 alerts | January (1.26×), April (1.27×) | April (1.27×) |
| January ratio | 1.26 | 1.23, just below the 1.25 threshold |
| Simulated surge | ALERT June-September | ALERT June-September |

The January alert disappeared because the v2 model is better in winter (January MAE 301 → 289), not because the threshold moved. The real April blizzard is still caught.

### 3.6 Error distribution

| Condition | v1 | v2 |
| --- | --- | --- |
| Snow (most severe label) | MAE 541, 2.5×, +296 | MAE 498, 2.3×, +282 |
| Snow with poor visibility | Could not be measured | **MAE 616, 2.9×, +454** |
| Snow with normal visibility | Could not be measured | MAE 319, 1.5×, +22 |
| Poor visibility without snow | Could not be measured | MAE 160, 0.75× |
| Holidays | MAE 376, 1.7×, +126 | MAE 369, 1.7×, +134 |

This is the most useful new finding of v2. v1 could only say "the model is worse in snow". v2 shows that light snow is handled reasonably, while snowstorms with poor visibility are where traffic collapses far below the forecast. Mist and fog on their own are predicted better than average.

### 3.7 Recommender and API

| Example | v1 | v2 |
| --- | --- | --- |
| 2 Oct 2018, 06:00-11:00, clear | 10:00-11:00, 4,513, 29% below peak | 10:00-11:00, 4,521, 30% below peak |
| Christmas Day, 08:00-19:00, 2 hours | 08:00-10:00, 2,282, 33% below peak | 08:00-10:00, 1,914, 42% below peak |
| 14 Nov 2018, snow, -5 °C | 10:00-11:00, 4,673, 24% below peak | 10:00-11:00, 4,853, 25% below peak |
| API: 2 Oct 08:00 / 03:00 | 5,912 / 341 | 5,900 / 351 |
| API: Christmas 08:00 in snow | 1,064 | 709 |

Every recommended window is the same. The holiday forecasts moved the most, but holiday error is unchanged at 1.7× the overall MAE in both versions, so these shifts sit within the model's known holiday uncertainty rather than showing that either version is better on holidays.

### 3.8 Unsupervised learning

The k-means features did not change, so the five clusters are identical. The association rules use the congestion quartiles, so the new thresholds give 153 rules instead of 152. The strongest rules are the same with confidence within two percentage points (for example weekend nights below 0 °C: Low 95% in v1, 94% in v2).

## 4. Did any conclusion change?

| v1 conclusion | v2 status |
| --- | --- |
| Time of day and day of week dominate traffic | Unchanged, confirmed by SHAP on both models |
| Weather matters far less than it first appears; snow is the exception | Refined: snowstorms with poor visibility are the exception |
| Gradient boosting is the best deployable model | Unchanged, now chosen on a validation year |
| The LSTM is the most accurate next-hour forecaster | Unchanged, and more accurate (152) |
| Travel mid-morning instead of at the 07:00 peak | Unchanged in every example |
| The proxy label excludes rain | Corrected: it excludes plain rain; rain with mist or fog now counts |
| Monitoring catches real incidents | Unchanged for April; January is no longer flagged |

## 5. Lessons

1. **Fix the selection rule before looking at scores.** v1 happened to choose correctly, but only a validation year can show that.
2. **Collapsing data loses information that analysis later needs.** One row per hour was right; one weather label per hour was not.
3. **A label is part of the model.** Thresholds that define a label must follow the same training-only rule as any other learned value.
4. **Tests and a one-command rerun make a second version cheap.** v2 re-ran every part in minutes and could compare every number with v1.

## 6. How to see v1

```bash
git checkout v1.0          # the v1 code and results
git checkout main          # back to the current version
```

The MLflow registry keeps both: versions 1-2 of each registered model are v1 runs, versions 3-4 are v2 runs.
