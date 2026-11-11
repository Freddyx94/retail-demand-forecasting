"""Export the data for the Power BI dashboard as CSV files in dashboard/data/.

Usage (from the project root, after `python run_pipeline.py`):
    python -m forecasting.export

No model is retrained: the hold-out forecasts are recreated with the saved
Stage 3 model (models/lightgbm.txt, trained up to the hold-out origin), and
the script stops if their RMSSE differs from results/holdout_results.csv.
So the dashboard shows exactly the numbers in FINDINGS.md and the README.

Files (git-ignored: they contain item-level M5 sales):
  items.csv          one row per item: department, velocity, demand pattern, value weight
  daily.csv          item x day: actuals, hold-out forecasts and the next 28 days
  item_errors.csv    item x model hold-out errors. RMSSE for any group of items
                     (department, velocity, all) = AVERAGE(rmsse) over its rows,
                     the same definition as the headline number
  model_results.csv  copy of results/holdout_results.csv
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting import baselines, models
from forecasting.data import ROOT, connect, load_config, load_features
from forecasting.error_analysis import item_profile
from forecasting.evaluate import make_folds, rmsse_scale, split

HISTORY_DAYS = 84          # actuals shown before the hold-out window (12 weeks)


def main() -> None:
    config = load_config()
    horizon, n_folds = config["forecast"]["horizon"], config["forecast"]["cv_folds"]
    with connect(config) as con:
        sales = load_features(con)
        future = con.sql("SELECT item_id, date, forecast FROM serving.forecasts").df()

    holdout = make_folds(sales["date"].max(), horizon, n_folds)[-1]
    train, valid = split(sales, holdout)

    # 1. Hold-out forecasts: saved LightGBM model + the two baselines
    booster = lgb.Booster(model_file=str(ROOT / config["paths"]["model_dir"] / "lightgbm.txt"))
    preds = valid[["item_id", "date", "units_sold"]].copy()
    preds["lightgbm"] = booster.predict(valid[models.LGBM_FEATURES])
    preds["seasonal_naive"] = baselines.seasonal_naive(train, valid)["forecast"].to_numpy()
    preds["moving_average_28"] = baselines.moving_average(train, valid)["forecast"].to_numpy()
    model_names = ["lightgbm", "seasonal_naive", "moving_average_28"]

    # 2. Per-item errors (RMSSE as in evaluate.rmsse, kept per item)
    scale = rmsse_scale(train)
    profile = item_profile(train)
    rows = []
    for m in model_names:
        g = preds.assign(sq=(preds["units_sold"] - preds[m]) ** 2).groupby("item_id", observed=True)
        per_item = pd.DataFrame({
            "model": m,
            "rmsse": np.sqrt(g["sq"].mean() / scale.reindex(g["sq"].mean().index)),
            "actual_units": g["units_sold"].sum(),
            "forecast_units": g[m].sum(),
        })
        rows.append(per_item.replace(np.inf, np.nan).dropna(subset=["rmsse"]))  # items with no scale
    item_errors = pd.concat(rows).reset_index()

    # Consistency check against the Stage 3 results table
    expected = pd.read_csv(ROOT / "results" / "holdout_results.csv", index_col="model")["rmsse"]
    for m in model_names:
        got = item_errors.loc[item_errors["model"] == m, "rmsse"].mean()
        assert abs(got - expected[m]) < 1e-3, f"{m}: RMSSE {got:.4f} != results table {expected[m]:.4f}"
        print(f"{m:<18} RMSSE {got:.4f}  (results table {expected[m]:.4f}) OK")

    # 3. Daily series: recent actuals, hold-out forecasts, next 28 days
    history = sales.loc[sales["date"].between(holdout.valid_start - pd.Timedelta(days=HISTORY_DAYS),
                                              holdout.train_end), ["item_id", "date", "units_sold"]]
    future["date"] = pd.to_datetime(future["date"])
    daily = pd.concat([
        history.assign(period="history"),
        preds.assign(period="holdout"),
        future.rename(columns={"forecast": "lightgbm"}).assign(period="future"),
    ], ignore_index=True)
    daily["item_id"] = daily["item_id"].astype(str)
    daily = daily.rename(columns={"units_sold": "actual"})[
        ["item_id", "date", "period", "actual"] + model_names]

    items = (profile[["velocity", "pattern", "weight"]]
             .join(sales.groupby("item_id", observed=True)["dept_id"].first())
             .reset_index())

    out = ROOT / "dashboard" / "data"
    out.mkdir(parents=True, exist_ok=True)
    items[["item_id", "dept_id", "velocity", "pattern", "weight"]].to_csv(out / "items.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False, date_format="%Y-%m-%d")
    item_errors.to_csv(out / "item_errors.csv", index=False)
    pd.read_csv(ROOT / "results" / "holdout_results.csv").to_csv(out / "model_results.csv", index=False)
    print(f"\nWrote to {out.relative_to(ROOT)}: items {len(items):,} rows, daily {len(daily):,}, "
          f"item_errors {len(item_errors):,}, model_results")


if __name__ == "__main__":
    main()
