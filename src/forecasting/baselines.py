"""Simple benchmark forecasts that every model has to beat.

Each function takes the training history (item_id, date, units_sold) up to
the forecast origin and the validation rows (item_id, date), and returns the
validation rows with a `forecast` column. Neither looks past the origin.
"""

import pandas as pd


def moving_average(train: pd.DataFrame, valid: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """"Every day sells the average of the last `window` days" (one flat line per item)."""
    last_date = train["date"].max()
    recent = train[train["date"] > last_date - pd.Timedelta(days=window)]
    level = recent.groupby("item_id")["units_sold"].mean().rename("forecast")
    return valid.join(level, on="item_id")


def seasonal_naive(train: pd.DataFrame, valid: pd.DataFrame, season: int = 28) -> pd.DataFrame:
    """"The next 4 weeks repeat the last 4 weeks", the store manager's rule.

    The forecast for day t is the sales on day t - 28, so each weekday is
    copied from the same weekday four weeks earlier. With a 28-day horizon
    that day is always inside the training history.

    Items launched less than 28 days before the origin have no sales on
    t - 28; they fall back to their average so far.
    """
    past = train[["item_id", "date", "units_sold"]].copy()
    past["date"] = past["date"] + pd.Timedelta(days=season)   # shift history forward
    out = valid.merge(past.rename(columns={"units_sold": "forecast"}),
                      on=["item_id", "date"], how="left")
    fallback = moving_average(train, valid, window=season)["forecast"].to_numpy()
    out["forecast"] = out["forecast"].fillna(pd.Series(fallback, index=out.index))
    return out
