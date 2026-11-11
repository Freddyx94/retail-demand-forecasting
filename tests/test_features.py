"""Leakage guards on the Python side.

The SQL features are checked against the data by the dbt test
tests/assert_feat_uses_only_data_up_to_origin.sql (needs the real data).
These tests check what the models are allowed to see.
"""

import pandas as pd
import pytest

from forecasting.baselines import moving_average, seasonal_naive
from forecasting.models import LGBM_FEATURES

HORIZON = 28


def test_model_features_exclude_the_answer_and_hindsight_columns():
    # units_sold is the target; is_likely_stockout uses the item's whole
    # future history; date and origin_date would let the model memorise time
    for column in ["units_sold", "is_likely_stockout", "is_future", "date", "origin_date"]:
        assert column not in LGBM_FEATURES
    assert len(LGBM_FEATURES) == len(set(LGBM_FEATURES))       # no duplicates


@pytest.fixture
def history():
    """One item, 84 days; sales on day n = n, so every value shows its own date."""
    dates = pd.date_range("2016-01-01", periods=84, freq="D")
    sales = pd.DataFrame({"item_id": "A", "date": dates, "units_sold": range(1, 85)})
    train, valid = sales.iloc[:-HORIZON], sales.iloc[-HORIZON:].reset_index(drop=True)
    return train, valid


def test_seasonal_naive_copies_the_day_28_days_earlier(history):
    train, valid = history
    forecast = seasonal_naive(train, valid)["forecast"]
    # day 57 -> day 29, ..., day 84 -> day 56: always inside the training data
    assert list(forecast) == list(valid["units_sold"] - HORIZON)
    assert forecast.max() <= train["units_sold"].max()


def test_moving_average_is_the_mean_of_the_last_28_training_days(history):
    train, valid = history
    forecast = moving_average(train, valid)["forecast"]
    assert (forecast == train["units_sold"].iloc[-HORIZON:].mean()).all()   # mean(29..56) = 42.5


@pytest.mark.parametrize("baseline", [seasonal_naive, moving_average])
def test_baselines_do_not_look_at_the_answers(history, baseline):
    train, valid = history
    before = baseline(train, valid)["forecast"]
    tampered = valid.assign(units_sold=999)               # change what is being predicted
    after = baseline(train, tampered)["forecast"]
    pd.testing.assert_series_equal(before, after)
