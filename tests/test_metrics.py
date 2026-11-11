"""Metrics and validation windows: small inputs with answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluate import bias, mae, make_folds, rmse, rmsse, rmsse_scale, split


def days(n, start="2016-01-01"):
    return pd.date_range(start, periods=n, freq="D")


# --- Error metrics -------------------------------------------------------------

Y = np.array([1.0, 2.0, 3.0])
Y_HAT = np.array([2.0, 2.0, 5.0])     # misses: +1, 0, +2


def test_mae():
    assert mae(Y, Y_HAT) == pytest.approx(1.0)               # (1 + 0 + 2) / 3


def test_rmse_weights_large_misses_more():
    assert rmse(Y, Y_HAT) == pytest.approx(np.sqrt(5 / 3))   # (1 + 0 + 4) / 3
    assert rmse(Y, Y_HAT) > mae(Y, Y_HAT)


def test_bias_is_total_forecast_vs_total_actual():
    assert bias(Y, Y_HAT) == pytest.approx(0.5)              # 9 / 6 - 1: 50% over
    assert bias(Y, Y) == 0.0


def test_rmsse_scale_counts_changes_only_after_first_sale():
    # Sales 0, 0, 2, 4, 1: the zeros before the first sale are not history yet.
    # Changes counted: 2 -> 4 (+2) and 4 -> 1 (-3): mean of squares = (4 + 9) / 2
    train = pd.DataFrame({"item_id": "A", "date": days(5), "units_sold": [0, 0, 2, 4, 1]})
    assert rmsse_scale(train)["A"] == pytest.approx(6.5)


def test_rmsse_scales_each_item_by_its_own_history():
    scale = pd.Series({"A": 6.5, "B": 2.0})
    valid = pd.DataFrame({
        "item_id": ["A", "A", "B", "B"],
        "units_sold": [3.0, 1.0, 2.0, 2.0],
        "forecast":   [1.0, 1.0, 2.0, 2.0],   # A: MSE (4 + 0) / 2 = 2; B: perfect
    })
    # mean over items of sqrt(MSE / scale): (sqrt(2 / 6.5) + 0) / 2
    assert rmsse(valid, scale) == pytest.approx(np.sqrt(2 / 6.5) / 2)


def test_rmsse_skips_items_that_never_sold_in_training():
    scale = pd.Series({"A": 6.5, "B": np.nan})                 # B: no sales, no scale
    valid = pd.DataFrame({"item_id": ["A", "B"], "units_sold": [3.0, 5.0], "forecast": [1.0, 0.0]})
    assert rmsse(valid, scale) == pytest.approx(np.sqrt(4 / 6.5))


# --- Validation windows (no look-ahead) ----------------------------------------

def test_folds_are_consecutive_windows_ending_with_the_holdout():
    last = pd.Timestamp("2016-05-22")
    folds = make_folds(last, horizon=28, n_folds=3)
    assert [f.name for f in folds] == ["fold1", "fold2", "fold3", "holdout"]
    assert folds[-1].valid_end == last
    for f in folds:
        assert (f.valid_end - f.valid_start).days + 1 == 28
        assert f.train_end == f.valid_start - pd.Timedelta(days=1)    # trains only on the past
    for earlier, later in zip(folds, folds[1:]):
        assert later.valid_start == earlier.valid_end + pd.Timedelta(days=1)


def test_split_uses_only_the_past_and_only_existing_items():
    sales = pd.concat([
        pd.DataFrame({"item_id": "OLD", "date": days(60), "units_sold": 1}),
        # launched after the origin: cannot be forecast from its history
        pd.DataFrame({"item_id": "NEW", "date": days(10, "2016-02-20"), "units_sold": 1}),
    ])
    fold = make_folds(sales["date"].max(), horizon=28, n_folds=0)[0]   # just the hold-out
    train, valid = split(sales, fold)
    assert train["date"].max() < valid["date"].min()
    assert set(valid["item_id"]) == {"OLD"}
    assert len(valid) == 28
