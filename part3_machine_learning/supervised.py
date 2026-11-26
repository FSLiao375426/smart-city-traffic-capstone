"""Part 3 Task 1: supervised models (regression and classification).

Regression     predict traffic_volume for an hour
Classification predict the proxy high_risk label (NOT real accidents)

Each task compares a dummy baseline, a simple linear model and two tree
ensembles on the same features, in two stages (D19):

1. Validation  fit on the training years (to 2016), score on 2017 and choose
               ONE model with SELECTION_RULE, which is fixed in this file
               before any score is seen. The choice is saved to
               results/model_selection.json before the test year is touched.
2. Test        refit every candidate on 2012-2017 and score it once on
               Jan-Sep 2018. The test scores are reported for every model,
               but they never change the choice made in stage 1.

Every run is logged to MLflow with a "stage" tag, and the chosen models are
saved for the API, the recommender and the monitoring simulation.

Run from the repository root:

    python -m part3_machine_learning.supervised
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier, DummyRegressor
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

# ---------------------------------------------------------------------------
# Selection rule (D19), written down before any validation score was seen.
# Best validation score wins. Models within TOLERANCE of the best count as a
# tie, and the tie goes to the simplest one in SIMPLICITY_ORDER (cheaper to
# run and store: a fitted random forest is about 100 times larger than
# gradient boosting). The dummy baseline is a reference only, never deployed.
# ---------------------------------------------------------------------------
SELECTION_RULE = {
    "regression": {"metric": "MAE", "better": "lower"},
    "classification": {"metric": "F1", "better": "higher"},
    "tolerance": 0.02,
    "simplicity_order": ["linear", "hist_gradient_boosting", "random_forest"],
    "excluded_from_selection": ["dummy"],
}


def regression_models() -> dict:
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=3,
                                               n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                                random_state=SEED),
    }


def classification_models() -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        # class_weight="balanced": high_risk is only about 10% positive (imbalanced)
        "linear": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                                 class_weight="balanced", random_state=SEED),
    }


ALGORITHM_NAMES = {
    "regression": {"dummy": "Dummy (training mean)", "linear": "Linear regression",
                   "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"},
    "classification": {"dummy": "Dummy (always 'not high risk')", "linear": "Logistic regression",
                       "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"},
}


# Same registered names as v1, so a refitted model becomes a new VERSION of the
# same MLflow registered model (model versioning, Task 6.1)
REGISTERED_NAMES = {
    "regression": {"dummy": "", "linear": "traffic_volume_linear_regression",
                   "random_forest": "traffic_volume_random_forest",
                   "hist_gradient_boosting": "traffic_volume_hist_gradient_boosting"},
    "classification": {"dummy": "", "linear": "high_risk_logistic_regression",
                       "random_forest": "high_risk_random_forest",
                       "hist_gradient_boosting": "high_risk_hist_gradient_boosting"},
}


def model_params(model) -> dict:
    """Readable hyperparameters for MLflow (the last step of a pipeline)."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    keep = ["n_estimators", "max_depth", "min_samples_leaf", "max_iter", "learning_rate",
            "class_weight", "strategy"]
    return {k: v for k, v in est.get_params().items() if k in keep}


def log_model_artifact(model, name: str, X_example: pd.DataFrame, pred, registered_name: str) -> None:
    """Log and register a test-stage model in MLflow, except random forests and dummies.

    A fitted random forest is tens of megabytes, which would push the MLflow
    folder past GitHub's 100 MB file limit, and a dummy has nothing to version.
    Their parameters and metrics are still logged, so the comparison stays complete.
    """
    if name in ("random_forest", "dummy"):
        mlflow.set_tag("model_artifact", "skipped (size)" if name == "random_forest" else "skipped (baseline)")
        logger.debug("Skipped MLflow model artifact for %s", name)
        return
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                             signature=infer_signature(X_example.head(), pred[:5]),
                             registered_model_name=registered_name)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def regression_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {"MAE": mean_absolute_error(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred)}, pred


def classification_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba) if proba.std() > 0 else 0.5   # a constant score ranks nothing
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": auc}, pred


def choose_model(task: str, valid_scores: dict) -> tuple:
    """Apply SELECTION_RULE to validation scores only. Returns (choice, reason)."""
    rule = SELECTION_RULE[task]
    metric, lower_is_better = rule["metric"], rule["better"] == "lower"
    eligible = {k: v[metric] for k, v in valid_scores.items()
                if k not in SELECTION_RULE["excluded_from_selection"]}
    best_name = min(eligible, key=eligible.get) if lower_is_better else max(eligible, key=eligible.get)
    best = eligible[best_name]
    tol = SELECTION_RULE["tolerance"]
    tied = [k for k, v in eligible.items()
            if (v <= best * (1 + tol) if lower_is_better else v >= best * (1 - tol))]
    choice = next(k for k in SELECTION_RULE["simplicity_order"] if k in tied)
    if choice == best_name:
        reason = f"best validation {metric} ({best:.3f})"
    else:
        reason = (f"within {tol:.0%} of the best validation {metric} ({eligible[choice]:.3f} vs "
                  f"{best:.3f} for {best_name}) and simpler, so it wins the tie")
    return choice, reason


# ---------------------------------------------------------------------------
# One task, two stages
# ---------------------------------------------------------------------------
def run_task(task: str, target: str, train, valid, test, features: list) -> tuple:
    """Validation stage, frozen choice, then one test stage. Returns (table, selection)."""
    make_models = regression_models if task == "regression" else classification_models
    score = regression_metrics if task == "regression" else classification_metrics
    experiment = "traffic_volume_regression" if task == "regression" else "high_risk_classification"
    mc.setup_mlflow(experiment)
    names = ALGORITHM_NAMES[task]

    # Stage 1: fit on the training years, score on the validation year
    X_tr, y_tr, X_va, y_va = train[features], train[target], valid[features], valid[target]
    if task == "classification":
        logger.info("high_risk positives: training %.1f%%, validation %.1f%%, test %.1f%%",
                    100 * y_tr.mean(), 100 * y_va.mean(), 100 * test[target].mean())
    valid_scores = {}
    for name, model in make_models().items():
        with mlflow.start_run(run_name=f"{name} (validation)"):
            mlflow.set_tags({"stage": "validation", "train_period": f"to {config.VALID_START}",
                             "eval_period": "2017"})
            start = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - start
            metrics, _ = score(model, X_va, y_va)
            metrics["fit_seconds"] = fit_s
            mlflow.log_params({"algorithm": name, "n_features": len(features), **model_params(model)})
            mlflow.log_metrics(metrics)
            valid_scores[name] = metrics
            logger.info("[validation] %s %-24s %s", task, names[name],
                        "  ".join(f"{k} {v:.3f}" for k, v in metrics.items() if k != "fit_seconds"))

    # Freeze the choice BEFORE the test year is scored
    choice, reason = choose_model(task, valid_scores)
    selection = {"task": task, "chosen": choice, "reason": reason,
                 "rule": {**SELECTION_RULE[task], "tolerance": SELECTION_RULE["tolerance"],
                          "simplicity_order": SELECTION_RULE["simplicity_order"]},
                 "validation_scores": {k: {m: round(v, 4) for m, v in s.items()} for k, s in valid_scores.items()},
                 "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    logger.info("Chosen %s model on validation: %s (%s). Choice frozen before scoring the test year",
                task, names[choice], reason)

    # Stage 2: refit on training + validation years, score every candidate once on the test year
    development = pd.concat([train, valid])
    X_dev, y_dev, X_te, y_te = development[features], development[target], test[features], test[target]
    rows = []
    for name, model in make_models().items():
        with mlflow.start_run(run_name=f"{name} (test)"):
            mlflow.set_tags({"stage": "test", "selected": str(name == choice),
                             "train_period": f"to {config.TEST_START}", "eval_period": "2018 Jan-Sep"})
            start = time.perf_counter()
            model.fit(X_dev, y_dev)
            fit_s = time.perf_counter() - start
            metrics, pred = score(model, X_te, y_te)
            mlflow.log_params({"algorithm": name, "n_features": len(features), **model_params(model)})
            mlflow.log_metrics({**metrics, "fit_seconds": fit_s})
            log_model_artifact(model, name, X_te, pred, REGISTERED_NAMES[task][name])
            logger.info("[test] %s %-30s %s", task, names[name],
                        "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
            key = valid_scores[name]
            row = {"model": names[name], "selected": "yes" if name == choice else ""}
            row.update({f"valid_{m}": key[m] for m in metrics})
            row.update({f"test_{m}": v for m, v in metrics.items()})
            row["fit_seconds"] = fit_s
            rows.append(row)
            if name == choice:
                out = "regression_model.joblib" if task == "regression" else "classification_model.joblib"
                joblib.dump({"model": model, "features": features, "algorithm": name,
                             "trained_to": config.TEST_START}, mc.MODELS_DIR / out)
                logger.info("Saved chosen %s model (%s, refitted on 2012-2017) to %s", task, names[name], out)
    return pd.DataFrame(rows), selection


def save_results(reg: pd.DataFrame, clf: pd.DataFrame, selections: list) -> None:
    (mc.RESULTS_DIR / "model_selection.json").write_text(
        json.dumps({"rule_written_before_scores": True, "selections": selections}, indent=2) + "\n",
        encoding="utf-8")
    out = {"regression": reg.round(4).to_dict("records"), "classification": clf.round(4).to_dict("records"),
           "train_period": "2012-10-02 to 2016-12-31", "validation_period": "2017",
           "test_period": "2018-01-01 to 2018-09-30 (models refitted on 2012-2017)"}
    (mc.RESULTS_DIR / "supervised_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    reg_cols = ["model", "selected", "valid_MAE", "valid_R2", "test_MAE", "test_RMSE", "test_R2", "fit_seconds"]
    clf_cols = ["model", "selected", "valid_F1", "test_accuracy", "test_precision", "test_recall",
                "test_F1", "test_ROC_AUC"]
    md = ["# Supervised model results", "",
          "Training 2012-2016, validation 2017 (model choice), test Jan-Sep 2018 (scored once, "
          "models refitted on 2012-2017). The selection rule and the choice are in model_selection.json.", "",
          "## Regression: traffic_volume", "", reg[reg_cols].round(3).to_markdown(index=False), "",
          "## Classification: high_risk (proxy label)", "", clf[clf_cols].round(3).to_markdown(index=False), ""]
    (mc.RESULTS_DIR / "supervised_results.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Saved supervised results and model_selection.json to %s",
                mc.RESULTS_DIR.relative_to(config.ROOT_DIR))


def main() -> int:
    setup_logging(mc.LOG_FILE)
    mc.ensure_dirs()
    try:
        df = mc.load_model_data()
        features = mc.feature_columns(df)
        logger.info("Using %d features: %s", len(features), ", ".join(features))
        train, valid, test = mc.three_way_split(df)
        reg, reg_sel = run_task("regression", "traffic_volume", train, valid, test, features)
        clf, clf_sel = run_task("classification", "high_risk", train, valid, test, features)
    except FileNotFoundError:
        logger.error("Features not found. Run the Part 2 pipeline and feature_engineering first.", exc_info=True)
        return 1
    except (ValueError, KeyError, mlflow.exceptions.MlflowException):
        logger.error("Supervised modelling did not complete", exc_info=True)
        return 1
    save_results(reg, clf, [reg_sel, clf_sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
