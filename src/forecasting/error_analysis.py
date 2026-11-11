"""Where does LightGBM win or fail? Error analysis on the CV folds.

Usage (from the project root):
    python -m forecasting.error_analysis

For every CV fold, trains LightGBM and the 28-day moving average (the
strongest simple baseline), then compares them by item group, by type of
day and by horizon, and prints LightGBM's feature importance.

Item groups are computed from each fold's TRAINING data (last 365 days), so
an item's group is what a planner would have known at the forecast origin.

Saves the row-level CV predictions to data/cv_predictions.parquet
(git-ignored) for the Stage 5 dashboard.
"""

import numpy as np
import pandas as pd

from forecasting import baselines, models
from forecasting.data import ROOT, connect, load_config, load_features
from forecasting.evaluate import make_folds, rmsse_scale, split

MODELS = ["lightgbm", "moving_average"]


def item_profile(train: pd.DataFrame) -> pd.DataFrame:
    """Per item, as known at the origin: velocity, demand pattern, RMSSE scale, weight."""
    last_year = train[(train["date"] > train["date"].max() - pd.Timedelta(days=365))
                      & ~train["is_store_closed"]]
    units = last_year.groupby("item_id", observed=True)["units_sold"]
    sold = last_year[last_year["units_sold"] > 0].groupby("item_id", observed=True)["units_sold"]

    p = pd.DataFrame({"mean_units": units.mean()})
    # Same cut-offs as the EDA (notebook section 5)
    p["velocity"] = pd.cut(p["mean_units"], [-np.inf, 1, 5, np.inf], right=False,
                           labels=["slow (<1/day)", "medium (1-5/day)", "fast (>=5/day)"])
    # Syntetos-Boylan: ADI = average days between sales, CV2 = variability of non-zero sales
    # reindex: an item with no sales in the last year has no row in `sold`
    adi = (units.size() / sold.size()).reindex(p.index)
    cv2 = ((sold.std() / sold.mean()) ** 2).reindex(p.index)
    p["pattern"] = np.select(
        [(adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
        ["smooth", "erratic", "intermittent"], default="lumpy",
    )
    p.loc[adi.isna() | cv2.isna(), "pattern"] = "lumpy"   # (almost) no sales last year
    p["scale"] = rmsse_scale(train)
    # M5-style weight: the item's share of sales value in the last 28 days
    recent = train[train["date"] > train["date"].max() - pd.Timedelta(days=28)]
    value = (recent["units_sold"] * recent["sell_price"]).groupby(recent["item_id"], observed=True).sum()
    p["weight"] = value / value.sum()
    return p


def day_type(df: pd.DataFrame) -> pd.Series:
    """Event day > 1-2 days before an event > SNAP day > normal day."""
    return pd.Series(np.select(
        [df["event_name"].notna(), df["days_to_next_event"].between(1, 2), df["is_snap"]],
        ["event day", "1-2 days before event", "SNAP day"], default="normal day",
    ), index=df.index)


def cv_predictions(sales: pd.DataFrame, folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-level predictions of both models on every CV fold, plus feature importance."""
    preds, importance = [], []
    for fold in folds:
        train, valid = split(sales, fold)
        booster = models.fit_lightgbm(train)
        out = valid[["item_id", "dept_id", "date", "horizon", "units_sold", "is_likely_stockout"]].copy()
        out["fold"] = fold.name
        out["lightgbm"] = booster.predict(valid[models.LGBM_FEATURES])
        out["moving_average"] = baselines.moving_average(train, valid)["forecast"].to_numpy()
        out["day_type"] = day_type(valid)
        out["horizon_week"] = "week " + ((out["horizon"] - 1) // 7 + 1).astype(str)
        out = out.join(item_profile(train)[["velocity", "pattern", "scale", "weight"]], on="item_id")
        preds.append(out)

        gain = booster.feature_importance(importance_type="gain")
        importance.append(pd.Series(gain / gain.sum(), index=booster.feature_name(), name=fold.name))
        print(f"{fold.name} done", flush=True)
    return pd.concat(preds, ignore_index=True), pd.concat(importance, axis=1)


def rmsse_by(preds: pd.DataFrame, group: str | None, model: str) -> pd.Series:
    """RMSSE per group: per fold, mean over items of sqrt(MSE / scale); then mean over folds."""
    keys = ["fold"] + ([group] if group else [])
    sq = (preds["units_sold"] - preds[model]) ** 2
    per_item = sq.groupby([preds[k] for k in keys + ["item_id"]], observed=True).mean()
    scale = preds.groupby(keys + ["item_id"], observed=True)["scale"].first()
    item_rmsse = np.sqrt(per_item / scale).replace(np.inf, np.nan)
    per_fold = item_rmsse.groupby(level=keys, observed=True).mean()
    return per_fold.groupby(level=group, observed=True).mean() if group else pd.Series({"all": per_fold.mean()})


def compare(preds: pd.DataFrame, group: str | None) -> pd.DataFrame:
    """LightGBM vs moving average for each value of `group` (or overall)."""
    t = pd.DataFrame({f"rmsse_{m}": rmsse_by(preds, group, m) for m in MODELS})
    t["lgbm_better_%"] = 100 * (1 - t["rmsse_lightgbm"] / t["rmsse_moving_average"])
    g = preds.groupby(group, observed=True) if group else preds.groupby(lambda _: "all")
    t["bias_lightgbm_%"] = 100 * (g["lightgbm"].sum() / g["units_sold"].sum() - 1)
    t["bias_moving_average_%"] = 100 * (g["moving_average"].sum() / g["units_sold"].sum() - 1)
    t["share_of_units_%"] = 100 * g["units_sold"].sum() / preds["units_sold"].sum()
    return t.round(3)


def weighted_rmsse(preds: pd.DataFrame, model: str) -> float:
    """Items weighted by recent sales value (as in M5): errors on big sellers count more."""
    sq = (preds["units_sold"] - preds[model]) ** 2
    per_item = sq.groupby([preds["fold"], preds["item_id"]], observed=True).mean()
    info = preds.groupby(["fold", "item_id"], observed=True)[["scale", "weight"]].first()
    item_rmsse = np.sqrt(per_item / info["scale"]).replace(np.inf, np.nan)
    ok = item_rmsse.notna()
    per_fold = (item_rmsse[ok] * info["weight"][ok]).groupby(level="fold").sum() \
        / info["weight"][ok].groupby(level="fold").sum()
    return float(per_fold.mean())


def main() -> None:
    config = load_config()
    horizon, n_folds = config["forecast"]["horizon"], config["forecast"]["cv_folds"]
    with connect(config) as con:
        sales = load_features(con)
    cv_folds = [f for f in make_folds(sales["date"].max(), horizon, n_folds) if f.name != "holdout"]

    preds, importance = cv_predictions(sales, cv_folds)
    preds.to_parquet(ROOT / "data" / "cv_predictions.parquet", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Overall ===\n", compare(preds, None))
    print("\nWeighted RMSSE (by recent sales value):",
          {m: round(weighted_rmsse(preds, m), 3) for m in MODELS})
    for group in ["velocity", "pattern", "dept_id", "day_type", "horizon_week"]:
        print(f"\n=== By {group} ===\n", compare(preds, group))
    print("\n=== Feature importance (share of total gain, mean over folds) ===\n",
          (100 * importance.mean(axis=1)).sort_values(ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
