"""Part 3 Task 3: LSTM for traffic behaviour, with SHAP explainability.

LSTM  Reads the previous 24 hours (traffic plus calendar and weather features)
      and predicts the next hour's traffic volume. Sequences never cross a gap
      in the data: every sequence must cover 25 consecutive hours.

SHAP  An LSTM is hard to explain directly, so SHAP is applied to the tree-based
      regression model trained on the same problem (supervised.py). The brief
      allows this, and TreeExplainer gives exact Shapley values for trees.

Run from the repository root (after supervised.py):

    python -m part3_machine_learning.deep_learning
"""

import json
import logging
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up chatter

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

from common import config  # noqa: E402
from common.data import load_features  # noqa: E402
from common.logging_setup import setup_logging  # noqa: E402
from part2_python.visualizations import apply_style, subtitle, BLUE, ORANGE, TEXT_2  # noqa: E402
from part3_machine_learning import ml_common as mc  # noqa: E402

logger = logging.getLogger(__name__)

WINDOW = 24                    # hours of history per sequence
SEQ_FEATURES = ["traffic_volume", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "is_holiday", "temp_c", "weather_severity"]
EPOCHS, BATCH = 15, 256


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame):
    """Return X (n, 24, features), y (n,), target timestamps.

    A sequence is kept only if its 24 input hours and the target hour are 25
    consecutive hours. Targets that are flagged road closures are dropped.
    Besides the long data gaps, this also skips the hour lost each spring when
    daylight saving time starts (e.g. 2018-03-11 02:00 does not exist).
    """
    df = df.sort_values("date_time").reset_index(drop=True)
    t = df["date_time"].to_numpy()
    values = df[SEQ_FEATURES].to_numpy(dtype="float32")
    anomaly = df["traffic_anomaly"].to_numpy()

    span = t[WINDOW:] - t[:-WINDOW]
    ok = (span == np.timedelta64(WINDOW, "h")) & (anomaly[WINDOW:] == 0)
    starts = np.flatnonzero(ok)
    X = np.stack([values[s:s + WINDOW] for s in starts])
    y = values[starts + WINDOW, 0]
    targets = pd.to_datetime(t[starts + WINDOW])
    dropped = len(df) - WINDOW - len(starts)
    logger.warning("Built %d sequences of %d hours; skipped %d positions that cross a data gap or "
                   "target a flagged closure hour", len(starts), WINDOW, dropped)
    return X, y, targets


def scale(X_train, X_test, y_train):
    """Standardise each feature with TRAINING statistics only (no leakage)."""
    mean = X_train.reshape(-1, X_train.shape[2]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[2]).std(axis=0) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std()
    logger.debug("Scaling means %s", dict(zip(SEQ_FEATURES, mean.round(2))))
    return (X_train - mean) / std, (X_test - mean) / std, y_mean, y_std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_lstm(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, n_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mse", metrics=["mae"])
    return model


def train_lstm(X_tr, y_tr, X_te, y_te):
    X_tr_s, X_te_s, y_mean, y_std = scale(X_tr, X_te, y_tr)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    model = build_lstm(X_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    start = time.perf_counter()
    # validation_split takes the LAST 10% of training sequences, i.e. the most recent
    # period, so validation stays chronological too
    history = model.fit(X_tr_s, (y_tr - y_mean) / y_std, validation_split=0.1, epochs=EPOCHS,
                        batch_size=BATCH, callbacks=[stop], verbose=0)
    fit_s = time.perf_counter() - start
    for epoch, (loss, val) in enumerate(zip(history.history["loss"], history.history["val_loss"]), 1):
        logger.debug("Epoch %d: loss %.4f, val_loss %.4f", epoch, loss, val)
    epochs_run = len(history.history["loss"])
    logger.info("LSTM trained for %d epochs in %.0fs (%s)", epochs_run, fit_s,
                "stopped early: validation loss stopped improving" if epochs_run < EPOCHS
                else "all epochs run, validation loss still improving")
    pred = model.predict(X_te_s, verbose=0).ravel() * y_std + y_mean
    return model, pred, history, fit_s


def evaluate(y_true, pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, pred)), "R2": float(r2_score(y_true, pred))}


# ---------------------------------------------------------------------------
# SHAP on the tree model
# ---------------------------------------------------------------------------
def explain_tree_model(test: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(mc.MODELS_DIR / "regression_model.joblib")
    model, features = bundle["model"], bundle["features"]
    sample = test.sample(n=min(1500, len(test)), random_state=config.RANDOM_SEED)[features]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    importance = (pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    for _, r in importance.head(5).iterrows():
        logger.info("SHAP: %-18s mean |contribution| %.0f vehicles", r["feature"], r["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(values, sample, max_display=12, show=False, plot_size=(9, 6))
    fig = plt.gcf()
    fig.axes[0].set_title("How each feature pushes the traffic prediction up or down", loc="left",
                          fontsize=13, fontweight="bold", pad=12)
    save(fig, "shap_summary.png")

    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Mean absolute SHAP value (vehicles per hour)")
    ax.set_title("Time of day dominates the traffic prediction", pad=22)
    subtitle(ax, f"Gradient boosting regressor, {len(sample):,} test hours from 2018")
    save(fig, "shap_importance.png")
    return importance


def plot_lstm(targets, y_true, pred_lstm, pred_naive) -> None:
    week = (targets >= pd.Timestamp("2018-03-05")) & (targets < pd.Timestamp("2018-03-10"))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(targets[week], y_true[week], color=TEXT_2, linewidth=2, label="Actual")
    ax.plot(targets[week], pred_lstm[week], color=BLUE, linewidth=2, label="LSTM prediction")
    ax.plot(targets[week], pred_naive[week], color=ORANGE, linewidth=1.2, linestyle="--",
            label="Same hour yesterday")
    ax.set(ylabel="Vehicles per hour")
    ax.set_title("The LSTM tracks hour-to-hour traffic closely", pad=22)
    subtitle(ax, "Test period, Monday 5 to Friday 9 March 2018")
    ax.legend(loc="upper right", ncol=3)
    fig.autofmt_xdate()
    save(fig, "lstm_test_week.png")


def save(fig, name: str) -> None:
    path = mc.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path.relative_to(config.ROOT_DIR))


def main() -> int:
    setup_logging(mc.LOG_FILE, append=True)
    mc.ensure_dirs()
    apply_style()
    try:
        df = load_features()
        X, y, targets = build_sequences(df)
        test_mask = targets >= pd.Timestamp(config.TEST_START)
        X_tr, y_tr, X_te, y_te = X[~test_mask], y[~test_mask], X[test_mask], y[test_mask]
        logger.info("LSTM data: %d training and %d test sequences", len(X_tr), len(X_te))

        mc.setup_mlflow("traffic_volume_lstm")
        with mlflow.start_run(run_name="lstm_24h"):
            model, pred, history, fit_s = train_lstm(X_tr, y_tr, X_te, y_te)
            m_lstm = evaluate(y_te, pred)
            pred_persist = X_te[:, -1, 0]          # last observed hour
            pred_daily = X_te[:, 0, 0]             # same hour yesterday (24 h before target)
            m_persist, m_daily = evaluate(y_te, pred_persist), evaluate(y_te, pred_daily)
            mlflow.log_params({"window_hours": WINDOW, "lstm_units": 32, "dense_units": 16,
                               "max_epochs": EPOCHS, "epochs_run": len(history.history["loss"]),
                               "batch_size": BATCH, "features": ",".join(SEQ_FEATURES)})
            mlflow.log_metrics({"MAE": m_lstm["MAE"], "R2": m_lstm["R2"], "fit_seconds": fit_s,
                                "baseline_last_hour_MAE": m_persist["MAE"],
                                "baseline_same_hour_yesterday_MAE": m_daily["MAE"]})
            model_path = mc.MODELS_DIR / "lstm_model.keras"
            model.save(model_path)
            mlflow.log_artifact(str(model_path))
        logger.info("LSTM test MAE %.1f, R2 %.3f | last-hour baseline MAE %.1f | same-hour-yesterday MAE %.1f",
                    m_lstm["MAE"], m_lstm["R2"], m_persist["MAE"], m_daily["MAE"])
        plot_lstm(targets[test_mask], y_te, pred, pred_daily)

        _, test = mc.time_split(mc.load_model_data())
        importance = explain_tree_model(test)
    except FileNotFoundError:
        logger.error("Missing input. Run feature_engineering and supervised first.", exc_info=True)
        return 1
    except (ValueError, KeyError, tf.errors.OpError):
        logger.error("Deep learning step did not complete", exc_info=True)
        return 1

    results = {"lstm": m_lstm, "baseline_last_hour": m_persist, "baseline_same_hour_yesterday": m_daily,
               "train_sequences": int(len(X_tr)), "test_sequences": int(len(X_te)),
               "shap_top_features": importance.head(10).round(1).to_dict("records")}
    (mc.RESULTS_DIR / "deep_learning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved deep learning results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
