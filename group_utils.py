"""
group_utils.py  -  shared evaluation harness for Group 202604-C5.

Group 202604-C5, evaluation harness. Nobody else edits this file;
ask in the group chat if you need a change.

Every model notebook (01-05) imports from here so that ALL models are:
  * tested on the SAME walk-forward folds (expanding window, never shuffled),
  * compared with the SAME naive forecast (predicted log return = 0),
  * scored with the SAME metrics and the SAME Diebold-Mariano test.

This file does not reinvent anything: it wraps the functions that already
live in tslab (validation.py and metrics.py), so our numbers match the
lecturer's notebooks exactly.

How a model owner uses it (Arsene, Mark M., Bakhita):

    from group_utils import get_splits, save_results
    splits = get_splits(len(dataset))            # horizon=1 by default
    preds = pd.Series(np.nan, index=dataset.index)
    for train_idx, test_idx in splits:
        ... fit on dataset.iloc[train_idx] only ...
        preds.iloc[test_idx] = model.predict(X.iloc[test_idx])
    save_results("ridge", dataset["target_return"], preds)   # -> results/ridge.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tslab import config
from tslab.metrics import (
    diebold_mariano,
    directional_accuracy,
    dm_verdict,
    mae,
    out_of_sample_r2,
    rmse,
)
from tslab.validation import describe_splits, walk_forward_splits

RESULTS_DIR = config.PROJECT_ROOT / "results"
METRICS_FILE = config.DATA_DIR / "metrics.csv"
NAIVE_NAME = "Naive (no change)"
SEED = config.SEED  # 42, use it in every model

# File name in results/  ->  name shown in the master table and in Power BI.
# Plain names, because a portfolio manager reads this table.
MODEL_NAMES = {
    "drift": "Drift (training mean)",
    "ridge": "Ridge regression",
    "gbm": "Gradient boosting",
    "armax": "ARMAX",
    "arima": "ARIMA",
    "lstm": "LSTM",
}


# ---------------------------------------------------------------------------
# 1. Walk-forward splits
# ---------------------------------------------------------------------------
def get_splits(n_rows, horizon=1):
    """
    The ONE set of walk-forward folds the whole group uses.

    Expanding window: fold k trains on rows 0 .. t_k - 1 and tests on the next
    config.TEST_SIZE rows (63 = one quarter). The first training block is
    config.INITIAL_TRAIN rows (about four years).

    gap = horizon - 1. For an h-day target, the last h-1 training labels
    would look into the test period, so they are dropped. For h = 1, gap = 0.
    """
    return list(
        walk_forward_splits(n_rows, config.INITIAL_TRAIN, config.TEST_SIZE, gap=horizon - 1)
    )


def check_splits(splits, index):
    """
    Prove the folds are honest: every training row comes BEFORE every test row,
    and no test row is used twice. Returns the describe_splits table.
    """
    seen = set()
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "a training row is after a test row"
        assert not seen.intersection(test_idx), "a test row appears in two folds"
        seen.update(test_idx.tolist())
    return describe_splits(splits, index)


# ---------------------------------------------------------------------------
# 2. The naive forecast
# ---------------------------------------------------------------------------
def naive_forecast(index):
    """
    'Tomorrow = today'. In log returns this is a predicted return of 0,
    so the predicted price is today's close x exp(0) = today's close.
    """
    return pd.Series(0.0, index=index, name="y_pred")


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def score(y_true, y_pred, close_today=None):
    """
    Score one model's predicted log returns.

    y_true       actual next-day log returns
    y_pred       predicted log returns (same rows)
    close_today  optional: the close on day t, used to also report the error
                 in PRICE terms (next close = close x exp(return)).

    R2_oos is measured against the zero-return (naive) forecast, the same
    definition as Notebook 3/4 (tslab.metrics.out_of_sample_r2).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    row = {
        "RMSE (return)": rmse(y_true, y_pred),
        "MAE (return)": mae(y_true, y_pred),
        "R2 out-of-sample": out_of_sample_r2(y_true, y_pred),
        # NaN for the naive forecast: a forecast of zero has no direction.
        "Directional accuracy": directional_accuracy(y_true, y_pred),
    }
    if close_today is not None:
        close_today = np.asarray(close_today, float)
        actual_close = close_today * np.exp(y_true)
        predicted_close = close_today * np.exp(y_pred)
        row["RMSE (price)"] = rmse(actual_close, predicted_close)
        row["MAE (price)"] = mae(actual_close, predicted_close)
    return row


def dm_vs_naive(y_true, y_pred, horizon=1):
    """
    Diebold-Mariano test of the model against the naive (zero) forecast,
    squared-error loss, Harvey-Leybourne-Newbold correction (tslab version).

    Negative statistic = the model has the smaller errors.
    p < 0.05 = the difference is unlikely to be luck.
    Pass horizon=h for h-day targets (Mark M.'s multi-horizon table).
    """
    y_true = np.asarray(y_true, float)
    stat, p = diebold_mariano(y_true, np.asarray(y_pred, float), np.zeros(len(y_true)), horizon)
    return stat, p, dm_verdict(stat, p)


# ---------------------------------------------------------------------------
# 4. Saving and loading results (the contract between notebooks)
# ---------------------------------------------------------------------------
def save_results(name, y_true, y_pred, results_dir=None):
    """
    Save out-of-sample predictions as results/<name>.csv with columns
    date, y_true, y_pred (log returns). Rows with no prediction are dropped.
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"y_true": pd.Series(y_true), "y_pred": pd.Series(y_pred)}).dropna()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    path = results_dir / f"{name}.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} predictions to {path}")
    return path


def load_results(results_dir=None):
    """
    Read every results/*.csv that has the columns date, y_true, y_pred.
    Other files (e.g. multi_horizon.csv) are skipped. Returns
    {file stem: DataFrame indexed by date}.
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    loaded = {}
    for path in sorted(results_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if not {"date", "y_true", "y_pred"}.issubset(frame.columns) or path.stem == "naive":
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        loaded[path.stem] = frame.set_index("date").sort_index()
    return loaded


# ---------------------------------------------------------------------------
# 5. The master results table
# ---------------------------------------------------------------------------
def build_master_table(results=None, close=None, save_to=METRICS_FILE):
    """
    One row per model, naive FIRST, every model scored on exactly the same
    dates (the dates on which every model has a prediction).

    results  dict from load_results() (loaded automatically if None)
    close    optional Series of the close on day t (dataset["close"]),
             to add price-based RMSE and MAE
    save_to  where to write the table for Power BI (data/metrics.csv)
    """
    results = results if results is not None else load_results()
    if not results:
        raise ValueError("No files in results/. Merge the model PRs first.")

    # Same rows for every model, otherwise the comparison is not fair.
    common = None
    for frame in results.values():
        common = frame.index if common is None else common.intersection(frame.index)
    y_true = next(iter(results.values())).loc[common, "y_true"]
    close_today = close.reindex(common) if close is not None else None

    naive_row = {"Model": NAIVE_NAME, **score(y_true, np.zeros(len(common)), close_today)}
    naive_row.update({"DM statistic": np.nan, "DM p-value": np.nan,
                      "Verdict vs naive": "Benchmark"})
    rows = [naive_row]
    for stem, frame in results.items():
        y_pred = frame.loc[common, "y_pred"]
        stat, p, verdict = dm_vs_naive(y_true, y_pred)
        rows.append({"Model": MODEL_NAMES.get(stem, stem), **score(y_true, y_pred, close_today),
                     "DM statistic": stat, "DM p-value": p, "Verdict vs naive": verdict})

    table = pd.DataFrame(rows).set_index("Model")
    table["Test days"] = len(common)
    table["Test from"] = common.min().strftime("%Y-%m-%d")
    table["Test to"] = common.max().strftime("%Y-%m-%d")

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        table.round(6).reset_index().to_csv(save_to, index=False)
        print(f"Saved the master table to {save_to}")
    return table


def share_of_up_days(y_true):
    """What to report instead of the naive forecast's directional accuracy."""
    return float(np.mean(np.asarray(y_true) > 0))
