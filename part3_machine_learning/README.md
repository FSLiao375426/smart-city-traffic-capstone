# Part 3 – Machine Learning and AI

Predictive models, pattern discovery, deep learning with explainability, experiment tracking, a recommendation system, a deployment mock-up and monitoring.

> **Accident data: a documented proxy was used.** No accident dataset was provided or sourced. The classification model predicts a proxy label, `high_risk` = High or Severe congestion (top half of traffic volume) together with severe weather (Thunderstorm, Snow, Squall) or low visibility (Fog, Mist, Haze, Smoke). It demonstrates the classification workflow only and must not be read as a prediction of real accidents. See the [bias and fairness report](../final_capstone_report/BIAS_FAIRNESS_REPORT.md).

## Contents

| File | Task | Output |
| --- | --- | --- |
| `ml_common.py` | all | Shared feature set (28 features), three-way chronological split, MLflow setup |
| `supervised.py` | 1 | Regression and classification: dummy baseline plus three algorithms each, chosen on 2017, scored once on 2018, logged to MLflow |
| `unsupervised.py` | 2 | K-means clustering and Apriori association rules |
| `deep_learning.py` | 3 | LSTM next-hour model (epochs chosen on 2017), SHAP on the LSTM itself and on the tree model |
| `recommender.py` | 5 | Travel-window recommendation for a date, weather and temperature, flagged as low confidence when history is thin |
| `api.py` | 6.3 | FastAPI service: `/health`, `/predict`, `/recommend` |
| `api_demo.py` | 6.3 | Sends sample requests to the API and saves the responses |
| `monitoring.py` | 6.4, 6.5 | Feature drift (PSI, KS) and error drift with PASS / ALERT status |
| `fairness_analysis.py` | 7 | Error breakdown by weather (most severe label and every label), visibility, snow and visibility together, time of day, day type, holiday and month |
| `models/` | | `regression_model.joblib` (deployed), `classification_model.joblib`, `lstm_model.keras` |
| `results/` | | Metrics, `model_selection.json` (the frozen model choice), cluster profiles, rules, monitoring report, API examples, error analysis |
| `figures/` | | K-means, LSTM, SHAP (LSTM and tree model) and monitoring figures |
| `mlflow.db`, `mlruns/` | 4, 6.1, 6.2 | MLflow tracking store and model artifacts |
| `part3.log`, `api.log` | | Logs |

## How to run

From the repository root, after the Part 2 pipeline and feature engineering:

```bash
python -m part3_machine_learning.supervised
python -m part3_machine_learning.unsupervised
python -m part3_machine_learning.deep_learning
python -m part3_machine_learning.monitoring
python -m part3_machine_learning.fairness_analysis
python -m part3_machine_learning.api_demo
```

Recommendation system:

```bash
python -m part3_machine_learning.recommender --date 2018-12-25 --earliest 8 --latest 18 --window 2
python -m part3_machine_learning.recommender --date 2018-11-14 --weather Snow --temp -5 --earliest 6 --latest 10
```

API server (then open http://127.0.0.1:8000/docs):

```bash
uvicorn part3_machine_learning.api:app --port 8000
```

MLflow interface (then open http://127.0.0.1:5000):

```bash
mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db
```

## Results at a glance

Training October 2012 to December 2016, validation 2017 (model choice), test January to September 2018 (scored once, after refitting on 2012-2017).

| Task | Model | Validation 2017 | Test 2018 |
| --- | --- | --- | --- |
| Regression | Dummy (training mean) | MAE 1,744 | MAE 1,728, R² 0.00 |
| | Linear regression | MAE 802 | MAE 806, R² 0.735 |
| | Random forest | MAE 239 | MAE 220, R² 0.964 |
| | **Gradient boosting (chosen, deployed)** | **MAE 236** | **MAE 213, R² 0.969** |
| Classification (proxy) | Dummy (always "not high risk") | F1 0 | F1 0, accuracy 0.889 |
| | Logistic regression | F1 0.80 | F1 0.73, recall 0.98, precision 0.58 |
| | Random forest | F1 0.95 | F1 0.93 |
| | **Gradient boosting (chosen)** | **F1 0.96** | **F1 0.93, ROC AUC 0.998** |
| Clustering | K-means, k = 5 | | Five interpretable traffic regimes |
| Association rules | Apriori | | 153 rules predicting congestion, top lift 3.8 |
| Deep learning | LSTM, 24-hour window, 39 epochs | MAE 163 | MAE 152, R² 0.987 (next-hour) |
| Monitoring | Gradient boosting in 2018 | | ALERT in April (blizzard, 14-15 April) |

## Model versions (Task 6.1)

Every training run registers its model in the MLflow Model Registry, so the registry holds the v1 and v2 history side by side (random forests log metrics and parameters only, to keep the repository under GitHub's file-size limit). Re-running `supervised.py` creates a new version of each registered model.

| Registered model | Purpose | Status |
| --- | --- | --- |
| `traffic_volume_hist_gradient_boosting` | Hourly traffic prediction | **Deployed** in the API and recommender (`models/regression_model.joblib`). Versions 1-2: v1 runs (chosen on the 2018 score). Versions 3-4: v2 runs (chosen on 2017, all weather labels). The deployed file is the latest version |
| `traffic_volume_linear_regression` | Baseline | Reference only |
| `high_risk_hist_gradient_boosting` | Proxy risk classification | Best classifier, not deployed |
| `high_risk_logistic_regression` | Baseline | Reference only |
| LSTM (`traffic_volume_lstm` experiment, artifact `lstm_model.keras`) | Next-hour nowcast | Experimental |

## Design decisions

- **Three-way chronological split (D19).** Hourly traffic is autocorrelated, so a random split would leak neighbouring hours into training. The model is chosen on 2017 with `SELECTION_RULE`, written in `supervised.py` before any score was seen: best validation score, and within 2% of the best the simpler model wins (linear, then gradient boosting, then random forest, which is about 70 times larger). The choice is frozen in `results/model_selection.json` before the 2018 test year is scored.
- **Dummy baselines.** A model that always predicts the training mean (regression) or "not high risk" (classification) shows what "no skill" scores. The dummy classifier reaches 88.9% accuracy with F1 0, which is why accuracy is not used to judge the classifiers.
- **Closure hours excluded.** The 20 flagged road-closure hours are removed from training and evaluation (D9).
- **Same features for every model.** Calendar, cyclical, holiday and weather features. Traffic volume and labels derived from it are never inputs.
- **Class imbalance.** The proxy label is 8.9% positive in training, so classifiers use balanced class weights and are judged on precision, recall, F1 and ROC AUC rather than accuracy.
- **All weather labels (D21).** The `wx_*` features are multi-hot, so an hour with rain and mist counts as both. The proxy label uses every label too (3,987 positive hours instead of 3,133).
- **SHAP on both models.** GradientExplainer explains the LSTM directly: recent traffic dominates, and the last three hours carry 80% of the attribution. TreeExplainer explains the deployed gradient boosting model, where hour of day dominates.
- **LSTM vs gradient boosting are not directly comparable.** The LSTM sees the last 24 hours of actual traffic, so it suits next-hour forecasting. The recommender and API need forecasts for future dates without live traffic, so they use gradient boosting.
- **Low-confidence recommendations.** When fewer than 30 past hours match the requested weather, day type and hours (for example Squall or Smoke), the recommendation is still given but marked `confidence: low` with a caution in the text.
- **Monitoring baseline.** The baseline error (MAE 236) is the 2017 validation error of the chosen model, read from `model_selection.json`, i.e. what was knowable before deployment. Feature drift compares January-September 2018 with the same months of 2017 to avoid confusing seasonality with drift.
