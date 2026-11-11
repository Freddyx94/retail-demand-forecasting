"""Forecasting models: ETS (classical statistics) and LightGBM (machine learning).

Same interface as baselines.py: each function takes the training rows up to
the forecast origin and the validation rows (both from the feature table),
and returns the validation rows with a `forecast` column.
"""

import lightgbm as lgb
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, WindowAverage

# Every feature is known at the forecast origin (see feat_daily_sales.sql):
# sales history ends on the origin; calendar, events and prices are known
# in advance. is_likely_stockout is deliberately NOT here.
LGBM_FEATURES = [
    # forecast position: how far ahead of the origin the day is
    "horizon",
    # sales history up to the origin (findings 1, 2, 6)
    "mean_7d", "mean_28d", "mean_56d", "std_28d", "days_since_sale",
    "same_weekday_last", "same_weekday_mean_4w",
    # calendar (findings 1-3)
    "weekday", "month", "day_of_month", "is_snap", "is_store_closed",
    # events (finding 3)
    "event_name", "event_type", "next_event_name", "days_to_next_event",
    # price (finding 4)
    "sell_price", "price_vs_ref", "price_change_wow",
    # product
    "item_id", "dept_id", "days_since_launch",
]

LGBM_PARAMS = {
    # Tweedie: a distribution for non-negative values with many exact zeros
    # (finding 5). Squared error would treat sales as any real number.
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,   # between Poisson (1) and Gamma (2); close to Poisson for counts
    "learning_rate": 0.05,
    "num_leaves": 63,                # complexity of each tree
    "min_data_in_leaf": 100,         # each leaf must cover >= 100 rows: limits overfitting
    "feature_fraction": 0.8,         # each tree sees a random 80% of features ...
    "bagging_fraction": 0.8,         # ... and a random 80% of rows
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}
LGBM_ROUNDS = 500                    # fixed number of trees: no early stopping on the
                                     # validation window, which would leak it into training


def ets(train: pd.DataFrame, valid: pd.DataFrame, history_days: int = 730) -> pd.DataFrame:
    """Exponential smoothing (AutoETS), one model per item with a weekly season.

    ETS forecasts from a level (and optionally trend) that is updated with a
    "fading memory": recent days weigh more than old ones. With
    season_length=7 it also learns each item's weekly rhythm. AutoETS picks
    the best variant per item by AICc.

    history_days: only the most recent days are used for fitting. Older
    data has almost no weight in ETS anyway (fading memory), and fitting
    1,437 models on 5 years is slow.

    Items where ETS cannot be fitted (e.g. too little history) fall back to
    the 28-day moving average, the same as the baseline.
    """
    horizon = valid["date"].nunique()
    start = train["date"].max() - pd.Timedelta(days=history_days)
    series = (
        train.loc[train["date"] > start, ["item_id", "date", "units_sold"]]
        .rename(columns={"item_id": "unique_id", "date": "ds", "units_sold": "y"})
        .astype({"unique_id": str})          # statsforecast expects plain IDs
    )
    sf = StatsForecast(
        models=[AutoETS(season_length=7)],
        freq="D",
        n_jobs=-1,                          # one item per CPU core
        fallback_model=WindowAverage(window_size=28),
    )
    fc = sf.forecast(df=series, h=horizon).rename(
        columns={"unique_id": "item_id", "ds": "date", "AutoETS": "forecast"}
    )
    fc["item_id"] = fc["item_id"].astype(valid["item_id"].dtype)   # match the category codes
    out = valid.merge(fc[["item_id", "date", "forecast"]], on=["item_id", "date"], how="left")
    out["forecast"] = out["forecast"].clip(lower=0)   # ETS can go below 0; sales cannot
    return out


def lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    drop_stockouts: bool = False,
    params: dict | None = None,
    rounds: int = LGBM_ROUNDS,
) -> pd.DataFrame:
    """One gradient-boosted tree model for all items, using the feature table.

    Global model: learning across all 1,437 items lets a slow seller borrow
    patterns (weekly rhythm, holidays, price response) learned from others.

    drop_stockouts: drop training rows inside likely stockouts (finding 6),
    so the model learns demand rather than sales. In CV this removed the
    under-forecast on in-stock days (-5% -> +1.5%) but scored worse on all
    days, so the main model keeps them (False).
    params / rounds: override LGBM_PARAMS / LGBM_ROUNDS (used by tune.py).
    """
    booster = fit_lightgbm(train, drop_stockouts, params, rounds)
    return valid.assign(forecast=booster.predict(valid[LGBM_FEATURES]))


def fit_lightgbm(
    train: pd.DataFrame,
    drop_stockouts: bool = False,
    params: dict | None = None,
    rounds: int = LGBM_ROUNDS,
) -> lgb.Booster:
    """Train the LightGBM model and return it (see `lightgbm` for the options)."""
    if drop_stockouts:
        train = train[~train["is_likely_stockout"]]
    data = lgb.Dataset(train[LGBM_FEATURES], label=train["units_sold"])
    return lgb.train({**LGBM_PARAMS, **(params or {})}, data, num_boost_round=rounds)
