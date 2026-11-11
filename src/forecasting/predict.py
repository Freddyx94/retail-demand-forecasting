"""Batch scoring: train the final model on all known sales and forecast the next 28 days.

Usage (from the project root, after `dbt build`):
    python -m forecasting.predict

Steps:
  1. Train LightGBM (the Stage 3 winner, same settings) on every day with
     known sales. The hold-out was used for the final test in Stage 3; now
     the model should learn from the most recent data too.
  2. Forecast the future rows of ml.feat_daily_sales (is_future = TRUE):
     every item x the 28 days after the last sales day.
  3. Write them to serving.forecasts, read by the API and the dashboard.
     Each run replaces the table; model_version and created_at say which
     model made the forecasts and when.

Also saves the model to models/<model_version>.txt and logs an MLflow run.
"""

from datetime import datetime, timezone

import mlflow
import pandas as pd

from forecasting import models
from forecasting.data import CATEGORICAL, ROOT, connect, load_config, load_features
from forecasting.train import setup_mlflow


def main() -> None:
    config = load_config()
    with connect(config) as con:
        history = load_features(con)
        future = load_features(con, future=True)

    origin = history["date"].max()
    model_version = f"lightgbm_{origin:%Y%m%d}"   # named after the last day it has seen
    print(f"Training on {len(history):,} rows up to {origin.date()}; "
          f"forecasting {len(future):,} rows ({future['date'].min().date()} .. {future['date'].max().date()})")

    # The two tables were loaded separately, so their category lists differ
    # (e.g. events that only occur in the history). Give the future rows the
    # training categories, so every code means the same thing to the model.
    for col in CATEGORICAL:
        future[col] = future[col].cat.set_categories(history[col].cat.categories)

    booster = models.fit_lightgbm(history)
    model_dir = ROOT / config["paths"]["model_dir"]
    model_dir.mkdir(exist_ok=True)
    booster.save_model(str(model_dir / f"{model_version}.txt"))

    forecasts = pd.DataFrame({
        "item_id": future["item_id"].astype(str),
        "date": future["date"],
        "forecast": booster.predict(future[models.LGBM_FEATURES]),   # expected units sold
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),  # UTC
    })

    with connect(config, read_only=False) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS serving")
        # pandas has no date-only type: cast back to DATE (as in dim_date)
        con.execute("""
            CREATE OR REPLACE TABLE serving.forecasts AS
            SELECT item_id, CAST(date AS DATE) AS date, forecast, model_version, created_at
            FROM forecasts
        """)
        n_rows = con.sql("SELECT count(*) FROM serving.forecasts").fetchone()[0]

    setup_mlflow(config)
    with mlflow.start_run(run_name=f"production_{model_version}"):
        mlflow.set_tag("evaluation", "none (production model, trained on all data)")
        mlflow.log_params({"model": "lightgbm", "model_version": model_version,
                           "train_end": str(origin.date()), "train_rows": len(history),
                           **models.LGBM_PARAMS})
        mlflow.log_metric("forecast_units_28d", forecasts["forecast"].sum())

    daily_total = forecasts.groupby("date")["forecast"].sum()
    print(f"\nWrote {n_rows:,} rows to serving.forecasts (model_version={model_version})")
    print(f"Forecast for the next 28 days: {daily_total.sum():,.0f} units "
          f"({daily_total.mean():,.0f} per day; last 28 days sold "
          f"{history.loc[history['date'] > origin - pd.Timedelta(days=28), 'units_sold'].sum():,})")


if __name__ == "__main__":
    main()
