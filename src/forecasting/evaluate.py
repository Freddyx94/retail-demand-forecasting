"""Rolling-origin cross-validation windows and forecast accuracy metrics.

Validation design (no look-ahead):

    |------------- training -------------|- fold 1 -|- fold 2 -|- fold 3 -|- hold-out -|
                                          28 days    28 days    28 days    last 28 days

* Each fold trains on everything BEFORE its window and forecasts the next
  28 days. The data is never shuffled.
* The hold-out (last 28 days) is not touched until the final evaluation.
* Only items already on sale at the forecast origin are evaluated: an item
  that did not exist yet cannot be forecast from its history.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One forecast exercise: train up to `train_end`, forecast the window after it."""

    name: str
    train_end: pd.Timestamp     # forecast origin: last day of known sales
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def make_folds(last_date: pd.Timestamp, horizon: int, n_folds: int) -> list[Fold]:
    """CV folds (oldest first) followed by the final hold-out, each `horizon` days long."""
    day = pd.Timedelta(days=1)
    folds = []
    # k = how many windows back from the end: n_folds ... 1 are CV folds, 0 is the hold-out
    for k in range(n_folds, -1, -1):
        valid_end = last_date - k * horizon * day
        valid_start = valid_end - (horizon - 1) * day
        name = "holdout" if k == 0 else f"fold{n_folds - k + 1}"
        folds.append(Fold(name, valid_start - day, valid_start, valid_end))
    return folds


def split(sales: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training history and validation rows for one fold."""
    train = sales[sales["date"] <= fold.train_end]
    valid = sales[sales["date"].between(fold.valid_start, fold.valid_end)]
    # Evaluate only items that existed at the forecast origin
    valid = valid[valid["item_id"].isin(train["item_id"].unique())]
    return train, valid


# --- Metrics -------------------------------------------------------------

def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean absolute error: average miss in units."""
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error: like MAE, but large misses count much more."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def bias(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Total forecast vs total actual, as a share: +0.10 = 10% over-forecast."""
    return float(y_hat.sum() / y.sum() - 1)


def rmsse_scale(train: pd.DataFrame) -> pd.Series:
    """Per-item RMSSE denominator (as in M5).

    Mean squared day-to-day change in the item's training sales, counted from
    its first sale. It measures how much the naive "same as yesterday"
    forecast would miss in-sample, so errors of fast and slow items become
    comparable.
    """
    t = train.sort_values(["item_id", "date"])
    by_item = t.groupby("item_id", observed=True)["units_sold"]
    has_sold = by_item.cummax() > 0
    # A day-to-day change counts only if the previous day was already after the first sale
    previous_has_sold = has_sold.groupby(t["item_id"], observed=True).shift(1, fill_value=False)
    squared_change = by_item.diff().where(previous_has_sold) ** 2
    return squared_change.groupby(t["item_id"], observed=True).mean()


def rmsse(valid: pd.DataFrame, scale: pd.Series) -> float:
    """Root mean squared scaled error, averaged over items (lower is better).

    1.0 means the forecast misses about as much as "same as yesterday" did on
    the training data. Items that never sold in training have no scale and
    are left out.
    """
    sq_err = (valid["units_sold"] - valid["forecast"]) ** 2
    mse_per_item = sq_err.groupby(valid["item_id"], observed=True).mean()
    per_item = np.sqrt(mse_per_item / scale.reindex(mse_per_item.index))
    return float(per_item.replace(np.inf, np.nan).dropna().mean())


def evaluate(valid: pd.DataFrame, train: pd.DataFrame) -> dict[str, float]:
    """All metrics on all days, and on days that are not likely stockouts.

    `valid` needs columns item_id, units_sold, forecast, is_likely_stockout.
    The "_clean" variants drop likely stockout days (finding 6): on those days
    the shelf was probably empty, so zero sales say nothing about demand.
    """
    scale = rmsse_scale(train)
    results = {}
    for suffix, rows in [("", valid), ("_clean", valid[~valid["is_likely_stockout"]])]:
        y, y_hat = rows["units_sold"].to_numpy(float), rows["forecast"].to_numpy(float)
        results[f"mae{suffix}"] = mae(y, y_hat)
        results[f"rmse{suffix}"] = rmse(y, y_hat)
        results[f"rmsse{suffix}"] = rmsse(rows, scale)
        results[f"bias{suffix}"] = bias(y, y_hat)
    return results
