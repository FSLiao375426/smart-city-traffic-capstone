# Part 3 – Machine Learning and AI

Predictive models, pattern discovery, deep learning with explainability, experiment tracking, a recommendation system, a deployment mock-up and monitoring.

> **Accident data: a documented proxy was used.** No accident dataset was provided or sourced. The classification model predicts a proxy label, `high_risk` = High or Severe congestion (top half of traffic volume) together with severe weather (Thunderstorm, Snow, Squall) or low visibility (Fog, Mist, Haze, Smoke). It demonstrates the classification workflow only and must not be read as a prediction of real accidents. See the [bias and fairness report](../final_capstone_report/BIAS_FAIRNESS_REPORT.md).

## Contents

| File | Task | Output |
| --- | --- | --- |
| `ml_common.py` | all | Shared feature set (28 features), chronological split, MLflow setup |
| `supervised.py` | 1 | Regression and classification, three algorithms each, logged to MLflow |
| `unsupervised.py` | 2 | K-means clustering and Apriori association rules |
| `deep_learning.py` | 3 | LSTM next-hour model, SHAP explanation of the tree model |
| `recommender.py` | 5 | Travel-window recommendation for a date, weather and temperature |
| `api.py` | 6.3 | FastAPI service: `/health`, `/predict`, `/recommend` |
| `api_demo.py` | 6.3 | Sends sample requests to the API and saves the responses |
| `monitoring.py` | 6.4, 6.5 | Feature drift (PSI, KS) and error drift with PASS / ALERT status |
| `fairness_analysis.py` | 7 | Error breakdown by weather, time of day, day type, holiday and month |
| `models/` | | `regression_model.joblib` (deployed), `classification_model.joblib`, `lstm_model.keras` |
| `results/` | | Metrics, cluster profiles, rules, monitoring report, API examples, error analysis |
| `figures/` | | K-means, LSTM, SHAP and monitoring figures |
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

Test set: January to September 2018. Training: October 2012 to December 2017.

| Task | Model | Key result |
| --- | --- | --- |
| Regression | Linear regression (baseline) | MAE 806, R² 0.735 |
| | Random forest | MAE 222, R² 0.964 |
| | **Gradient boosting (deployed)** | **MAE 217, R² 0.966** |
| Classification (proxy) | Logistic regression (baseline) | F1 0.81, recall 0.99, precision 0.69 |
| | Random forest | F1 0.92 |
| | **Gradient boosting** | **F1 0.93, ROC AUC 0.998** |
| Clustering | K-means, k = 5 | Five interpretable traffic regimes |
| Association rules | Apriori | 152 rules predicting congestion, top lift 3.8 |
| Deep learning | LSTM, 24-hour window | MAE 164, R² 0.985 (next-hour) |
| Monitoring | Gradient boosting in 2018 | ALERT in January and April (blizzard, 14-15 April) |

## Model versions (Task 6.1)

Every training run registers its model in the MLflow Model Registry (random forests log metrics and parameters only, to keep the repository under GitHub's file-size limit). Re-running `supervised.py` creates a new version of each registered model.

| Registered model | Purpose | Status |
| --- | --- | --- |
| `traffic_volume_hist_gradient_boosting` | Hourly traffic prediction | **Deployed** in the API and recommender (`models/regression_model.joblib`) |
| `traffic_volume_linear_regression` | Baseline | Reference only |
| `high_risk_hist_gradient_boosting` | Proxy risk classification | Best classifier, not deployed |
| `high_risk_logistic_regression` | Baseline | Reference only |
| LSTM (`traffic_volume_lstm` experiment, artifact `lstm_model.keras`) | Next-hour nowcast | Experimental |

## Design decisions

- **Chronological split.** Hourly traffic is autocorrelated, so a random split would leak neighbouring hours into training.
- **Closure hours excluded.** The 20 flagged road-closure hours are removed from training and evaluation (D9).
- **Same features for every model.** Calendar, cyclical, holiday and weather features. Traffic volume and labels derived from it are never inputs.
- **Class imbalance.** The proxy label is 7.5% positive in training, so classifiers use balanced class weights and are judged on precision, recall, F1 and ROC AUC rather than accuracy (always predicting "not high risk" scores 90.9%).
- **SHAP on the tree model.** The LSTM is hard to explain directly, so SHAP explains the gradient boosting model trained on the same target, as the brief allows.
- **LSTM vs gradient boosting are not directly comparable.** The LSTM sees the last 24 hours of actual traffic, so it suits next-hour forecasting. The recommender and API need forecasts for future dates without live traffic, so they use gradient boosting.
- **Monitoring baseline.** The baseline error (MAE 238) is the error of the same model type trained to 2016 and validated on 2017, i.e. what was knowable before deployment. Feature drift compares January-September 2018 with the same months of 2017 to avoid confusing seasonality with drift.
