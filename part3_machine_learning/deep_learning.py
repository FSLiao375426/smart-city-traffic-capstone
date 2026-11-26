"""Part 3 Task 3: LSTM for traffic behaviour, with SHAP explainability.

LSTM  Reads the previous 24 hours (traffic plus calendar and weather features)
      and predicts the next hour's traffic volume. Sequences never cross a gap
      in the data: every sequence must cover 25 consecutive hours.
      Same split as supervised.py (D19): the number of epochs is chosen by
      early stopping on 2017, then the model is refitted on 2012-2017 for that
      many epochs and scored once on 2018.

SHAP  Two explanations:
      1. GradientExplainer directly on the LSTM: which input features, and
         which of the 24 past hours, move its predictions.
      2. TreeExplainer on the gradient boosting regressor (supervised.py),
         which forecasts future dates without live traffic. TreeExplainer
         gives exact Shapley values for trees.

Run from the repository root (after supervised.py):

    python -m part3_machine_learning.deep_learning
"""

import json
import logging
import os
import sys
import time
import warnings

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

# Keras warns that SHAP passes the input as a list; the SHAP values are unaffected
warnings.filterwarnings("ignore", message=".*structure of `inputs` doesn't match.*")
# shap.summary_plot seeds NumPy's global generator internally; the plot is unaffected
warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*", category=FutureWarning)

WINDOW = 24                    # hours of history per sequence
SEQ_FEATURES = ["traffic_volume", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "is_holiday", "temp_c", "weather_severity"]
EPOCHS, BATCH = 40, 256      # upper limit; early stopping on 2017 picks the actual number


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


def fit_scaler(X_train, y_train) -> dict:
    """Standardisation parameters from the training sequences only (no leakage)."""
    flat = X_train.reshape(-1, X_train.shape[2])
    params = {"mean": flat.mean(axis=0), "std": flat.std(axis=0) + 1e-6,
              "y_mean": float(y_train.mean()), "y_std": float(y_train.std())}
    logger.debug("Scaling means %s", dict(zip(SEQ_FEATURES, params["mean"].round(2))))
    return params


def apply_scaler(X, params):
    return (X - params["mean"]) / params["std"]


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


def choose_epochs(X_tr, y_tr, X_va, y_va) -> tuple:
    """Stage 1: train on 2012-2016, early-stop on 2017. Returns (best epochs, validation MAE)."""
    sc = fit_scaler(X_tr, y_tr)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    model = build_lstm(X_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(apply_scaler(X_tr, sc), (y_tr - sc["y_mean"]) / sc["y_std"],
                        validation_data=(apply_scaler(X_va, sc), (y_va - sc["y_mean"]) / sc["y_std"]),
                        epochs=EPOCHS, batch_size=BATCH, callbacks=[stop], verbose=0)
    for epoch, (loss, val) in enumerate(zip(history.history["loss"], history.history["val_loss"]), 1):
        logger.debug("Epoch %d: loss %.4f, val_loss (2017) %.4f", epoch, loss, val)
    run = len(history.history["loss"])
    best = int(np.argmin(history.history["val_loss"])) + 1
    pred = model.predict(apply_scaler(X_va, sc), verbose=0).ravel() * sc["y_std"] + sc["y_mean"]
    valid_mae = float(mean_absolute_error(y_va, pred))
    logger.info("Validation stage: best epoch %d of %d run (%s), validation MAE (2017) %.1f",
                best, run, "stopped early" if run < EPOCHS else "all epochs run", valid_mae)
    return best, valid_mae


def train_final(X_dev, y_dev, X_te, epochs: int):
    """Stage 2: refit on 2012-2017 for the chosen number of epochs, predict 2018 once."""
    sc = fit_scaler(X_dev, y_dev)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    model = build_lstm(X_dev.shape[2])
    start = time.perf_counter()
    model.fit(apply_scaler(X_dev, sc), (y_dev - sc["y_mean"]) / sc["y_std"],
              epochs=epochs, batch_size=BATCH, verbose=0)
    fit_s = time.perf_counter() - start
    logger.info("Final LSTM refitted on 2012-2017 for %d epochs in %.0fs", epochs, fit_s)
    pred = model.predict(apply_scaler(X_te, sc), verbose=0).ravel() * sc["y_std"] + sc["y_mean"]
    return model, sc, pred, fit_s


def evaluate(y_true, pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, pred)), "R2": float(r2_score(y_true, pred))}


# ---------------------------------------------------------------------------
# SHAP directly on the LSTM
# ---------------------------------------------------------------------------
def explain_lstm(model, sc, X_background, X_explain) -> dict:
    """GradientExplainer (expected gradients) on the scaled sequences.

    Each SHAP value is the contribution of one input feature at one past hour
    to one prediction, rescaled to vehicles per hour. Summing absolute values
    over the 24 hours ranks the features; summing over the features shows how
    far back in time the LSTM looks.
    """
    explainer = shap.GradientExplainer(model, apply_scaler(X_background, sc))
    values = np.asarray(explainer.shap_values(apply_scaler(X_explain, sc)))
    values = values.reshape(len(X_explain), WINDOW, len(SEQ_FEATURES)) * sc["y_std"]
    by_feature = (pd.DataFrame({"feature": SEQ_FEATURES,
                                "mean_abs_shap": np.abs(values).sum(axis=1).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    by_lag = np.abs(values).sum(axis=2).mean(axis=0)          # position 0 = 24 hours before target
    share = by_lag / by_lag.sum()
    for _, r in by_feature.head(4).iterrows():
        logger.info("LSTM SHAP: %-16s mean |contribution| %.0f vehicles", r["feature"], r["mean_abs_shap"])
    logger.info("LSTM SHAP: the last hour carries %.0f%% of the attribution, the last 3 hours %.0f%%, "
                "the same hour yesterday %.1f%%", 100 * share[-1], 100 * share[-3:].sum(), 100 * share[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.35})
    top = by_feature.iloc[::-1]
    ax1.barh(top["feature"], top["mean_abs_shap"], color=BLUE, height=0.7)
    ax1.grid(axis="y", visible=False)
    ax1.set(xlabel="Mean |SHAP| summed over 24 hours (vehicles)")
    ax1.set_title("Recent traffic drives the LSTM", pad=22, loc="left")
    subtitle(ax1, f"GradientExplainer, {len(X_explain)} test sequences from 2018")
    lags = np.arange(-WINDOW, 0)
    ax2.bar(lags, by_lag, color=[ORANGE if lag >= -3 else BLUE for lag in lags], width=0.8)
    ax2.set(xlabel="Input hour relative to the predicted hour", ylabel="Mean |SHAP| (vehicles)")
    ax2.set_title(f"Last 3 hours: {100 * share[-3:].sum():.0f}% of the attribution", pad=22, loc="left")
    subtitle(ax2, "All features summed per input hour; orange = last 3 hours")
    save(fig, "lstm_shap.png")
    return {"by_feature": by_feature.round(1).to_dict("records"),
            "share_by_hours_before": {int(-lag): round(float(v), 3) for lag, v in zip(lags, share)},
            "explained_sequences": int(len(X_explain))}


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
    # On native Windows TensorFlow warns that it has no GPU support; training runs on the CPU anyway
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    mc.ensure_dirs()
    apply_style()
    try:
        df = load_features()
        X, y, targets = build_sequences(df)
        valid_start, test_start = pd.Timestamp(config.VALID_START), pd.Timestamp(config.TEST_START)
        tr = np.asarray(targets < valid_start)
        va = np.asarray((targets >= valid_start) & (targets < test_start))
        te = np.asarray(targets >= test_start)
        dev = tr | va
        logger.info("LSTM data: %d training, %d validation and %d test sequences", tr.sum(), va.sum(), te.sum())
        X_te, y_te = X[te], y[te]

        mc.setup_mlflow("traffic_volume_lstm")
        with mlflow.start_run(run_name="lstm_24h"):
            mlflow.set_tags({"stage": "test", "epochs_chosen_on": "2017"})
            epochs, valid_mae = choose_epochs(X[tr], y[tr], X[va], y[va])
            model, sc, pred, fit_s = train_final(X[dev], y[dev], X_te, epochs)
            m_lstm = evaluate(y_te, pred)
            pred_persist = X_te[:, -1, 0]          # last observed hour
            pred_daily = X_te[:, 0, 0]             # same hour yesterday (24 h before target)
            m_persist, m_daily = evaluate(y_te, pred_persist), evaluate(y_te, pred_daily)
            mlflow.log_params({"window_hours": WINDOW, "lstm_units": 32, "dense_units": 16,
                               "max_epochs": EPOCHS, "epochs_chosen": epochs,
                               "batch_size": BATCH, "features": ",".join(SEQ_FEATURES)})
            mlflow.log_metrics({"valid_MAE": valid_mae, "MAE": m_lstm["MAE"], "R2": m_lstm["R2"],
                                "fit_seconds": fit_s, "baseline_last_hour_MAE": m_persist["MAE"],
                                "baseline_same_hour_yesterday_MAE": m_daily["MAE"]})
            model_path = mc.MODELS_DIR / "lstm_model.keras"
            model.save(model_path)
            mlflow.log_artifact(str(model_path))
        logger.info("LSTM test MAE %.1f, R2 %.3f | last-hour baseline MAE %.1f | same-hour-yesterday MAE %.1f",
                    m_lstm["MAE"], m_lstm["R2"], m_persist["MAE"], m_daily["MAE"])
        plot_lstm(targets[te], y_te, pred, pred_daily)

        rng = np.random.default_rng(config.RANDOM_SEED)
        background = X[dev][rng.choice(int(dev.sum()), 200, replace=False)]
        explained = X_te[rng.choice(len(X_te), 300, replace=False)]
        lstm_shap = explain_lstm(model, sc, background, explained)

        _, test = mc.time_split(mc.load_model_data())
        importance = explain_tree_model(test)
    except FileNotFoundError:
        logger.error("Missing input. Run feature_engineering and supervised first.", exc_info=True)
        return 1
    except (ValueError, KeyError, tf.errors.OpError):
        logger.error("Deep learning step did not complete", exc_info=True)
        return 1

    results = {"lstm": {**m_lstm, "valid_MAE_2017": valid_mae, "epochs_chosen": epochs},
               "baseline_last_hour": m_persist, "baseline_same_hour_yesterday": m_daily,
               "train_sequences": int(tr.sum()), "validation_sequences": int(va.sum()),
               "test_sequences": int(te.sum()), "lstm_shap": lstm_shap,
               "shap_top_features": importance.head(10).round(1).to_dict("records")}
    (mc.RESULTS_DIR / "deep_learning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved deep learning results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
