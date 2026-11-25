"""Part 3 Task 1: supervised models (regression and classification).

Regression     predict traffic_volume for an hour
Classification predict the proxy high_risk label (NOT real accidents)

Each task trains a simple linear baseline and two tree ensembles on the same
features and the same chronological split, logs every run to MLflow and saves
the best model for the API.

Run from the repository root:

    python -m part3_machine_learning.supervised
"""

import json
import logging
import sys
import time
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import config
from common.logging_setup import setup_logging
from part3_machine_learning import ml_common as mc

logger = logging.getLogger(__name__)
SEED = config.RANDOM_SEED

# The feature table has no missing values, so MLflow's integer-schema hint does not apply
warnings.filterwarnings("ignore", message=".*Integer columns in Python cannot represent missing values.*")


def regression_models() -> dict:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=3,
                                               n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                                random_state=SEED),
    }


def classification_models() -> dict:
    return {
        # class_weight="balanced": high_risk is only ~8% positive (imbalanced)
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                                 class_weight="balanced", random_state=SEED),
    }


def model_params(model) -> dict:
    """Readable hyperparameters for MLflow (the last step of a pipeline)."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    keep = ["n_estimators", "max_depth", "min_samples_leaf", "max_iter", "learning_rate", "class_weight"]
    return {k: v for k, v in est.get_params().items() if k in keep}


def log_model_artifact(model, name: str, X_example: pd.DataFrame, pred, registered_name: str) -> None:
    """Log and register the model in MLflow, except random forests.

    A fitted random forest is tens of megabytes, which would push the MLflow
    folder past GitHub's 100 MB file limit. Its parameters and metrics are
    still logged, so the comparison stays complete.
    """
    if name == "random_forest":
        mlflow.set_tag("model_artifact", "skipped (size)")
        logger.debug("Skipped MLflow model artifact for %s to keep the repository small", name)
        return
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                             signature=infer_signature(X_example.head(), pred[:5]),
                             registered_model_name=registered_name)


def run_regression(train: pd.DataFrame, test: pd.DataFrame, features: list) -> pd.DataFrame:
    mc.setup_mlflow("traffic_volume_regression")
    X_tr, y_tr, X_te, y_te = train[features], train["traffic_volume"], test[features], test["traffic_volume"]
    rows = []
    for name, model in regression_models().items():
        with mlflow.start_run(run_name=name):
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - start
            pred = model.predict(X_te)
            metrics = {
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "R2": r2_score(y_te, pred),
                "train_R2": r2_score(y_tr, model.predict(X_tr)),
                "fit_seconds": fit_s,
            }
            mlflow.log_params({"algorithm": name, "n_features": len(features), **model_params(model)})
            mlflow.log_metrics(metrics)
            log_model_artifact(model, name, X_te, pred, f"traffic_volume_{name}")
            logger.info("Regression %-24s MAE %7.1f  R2 %.3f  (train R2 %.3f, %.1fs)",
                        name, metrics["MAE"], metrics["R2"], metrics["train_R2"], fit_s)
            rows.append({"model": name, **metrics, "_model": model})
    results = pd.DataFrame(rows)
    best = results.loc[results["MAE"].idxmin()]
    joblib.dump({"model": best["_model"], "features": features}, mc.MODELS_DIR / "regression_model.joblib")
    logger.info("Best regression model: %s (MAE %.1f), saved for deployment", best["model"], best["MAE"])
    return results.drop(columns="_model")


def run_classification(train: pd.DataFrame, test: pd.DataFrame, features: list) -> pd.DataFrame:
    mc.setup_mlflow("high_risk_classification")
    X_tr, y_tr, X_te, y_te = train[features], train["high_risk"], test[features], test["high_risk"]
    logger.info("high_risk positives: train %.1f%%, test %.1f%%", 100 * y_tr.mean(), 100 * y_te.mean())
    rows = []
    for name, model in classification_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = {
                "accuracy": accuracy_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred),
                "F1": f1_score(y_te, pred),
                "ROC_AUC": roc_auc_score(y_te, proba),
            }
            mlflow.log_params({"algorithm": name, "n_features": len(features), "threshold": 0.5,
                               **model_params(model)})
            mlflow.log_metrics(metrics)
            log_model_artifact(model, name, X_te, pred, f"high_risk_{name}")
            logger.info("Classification %-22s acc %.3f  prec %.3f  rec %.3f  F1 %.3f  AUC %.3f",
                        name, *metrics.values())
            rows.append({"model": name, **metrics, "_model": model})
    results = pd.DataFrame(rows)
    best = results.loc[results["F1"].idxmax()]
    joblib.dump({"model": best["_model"], "features": features}, mc.MODELS_DIR / "classification_model.joblib")
    logger.info("Best classification model: %s (F1 %.3f), saved", best["model"], best["F1"])

    baseline_acc = 1 - y_te.mean()
    logger.info("Accuracy of always predicting 'not high risk' would be %.3f", baseline_acc)
    return results.drop(columns="_model")


def main() -> int:
    setup_logging(mc.LOG_FILE)
    mc.ensure_dirs()
    try:
        df = mc.load_model_data()
        features = mc.feature_columns(df)
        logger.info("Using %d features: %s", len(features), ", ".join(features))
        train, test = mc.time_split(df)
        reg = run_regression(train, test, features)
        clf = run_classification(train, test, features)
    except FileNotFoundError:
        logger.error("Features not found. Run the Part 2 pipeline and feature_engineering first.", exc_info=True)
        return 1
    except (ValueError, KeyError, mlflow.exceptions.MlflowException):
        logger.error("Supervised modelling did not complete", exc_info=True)
        return 1

    out = {"regression": reg.round(4).to_dict("records"), "classification": clf.round(4).to_dict("records"),
           "test_period": "2018-01-01 to 2018-09-30", "train_period": "2012-10-02 to 2017-12-31"}
    (mc.RESULTS_DIR / "supervised_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Supervised model results", "", "Test set: 2018 (chronological split).", "",
          "## Regression: traffic_volume", "", reg.round(3).to_markdown(index=False), "",
          "## Classification: high_risk (proxy label)", "", clf.round(3).to_markdown(index=False), ""]
    (mc.RESULTS_DIR / "supervised_results.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Saved supervised results to %s", mc.RESULTS_DIR.relative_to(config.ROOT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
