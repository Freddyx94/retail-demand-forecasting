"""Run every model through rolling-origin cross-validation and log it to MLflow.

Usage (from the project root):
    python -m forecasting.train

For each model: forecast each CV fold, compute the metrics, log them per fold
and averaged to MLflow, and print a results table. The hold-out is NOT used
here; it is evaluated once, at the end of Stage 3.
"""

import mlflow
import pandas as pd

from forecasting import baselines
from forecasting.data import ROOT, connect, load_config, load_sales
from forecasting.evaluate import evaluate, make_folds, split

EXPERIMENT = "m5_ca1_foods"

# name -> function(train, valid) returning valid with a `forecast` column
MODELS = {
    "seasonal_naive": baselines.seasonal_naive,
    "moving_average_28": baselines.moving_average,
}


def setup_mlflow(config: dict) -> None:
    """Store runs in a local SQLite database under mlruns/ (git-ignored)."""
    mlflow_dir = ROOT / config["paths"]["mlflow_dir"]
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(mlflow_dir / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def cross_validate(name, forecast_fn, sales, folds) -> pd.DataFrame:
    """Forecast every CV fold with one model; return one row of metrics per fold."""
    rows = []
    for fold in folds:
        train, valid = split(sales, fold)
        predicted = forecast_fn(train, valid)
        metrics = evaluate(predicted, train)
        rows.append({"model": name, "fold": fold.name, "n_items": valid["item_id"].nunique(), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    horizon, n_folds = config["forecast"]["horizon"], config["forecast"]["cv_folds"]

    with connect(config) as con:
        sales = load_sales(con)

    all_folds = make_folds(sales["date"].max(), horizon, n_folds)
    cv_folds = [f for f in all_folds if f.name != "holdout"]   # hold-out stays untouched
    for f in all_folds:
        print(f"{f.name:8s} train <= {f.train_end.date()}  forecast {f.valid_start.date()} .. {f.valid_end.date()}")

    setup_mlflow(config)
    results = []
    for name, forecast_fn in MODELS.items():
        with mlflow.start_run(run_name=name):
            per_fold = cross_validate(name, forecast_fn, sales, cv_folds)
            mlflow.log_params({"model": name, "horizon": horizon, "cv_folds": n_folds})
            metric_cols = per_fold.columns.drop(["model", "fold", "n_items"])
            for _, row in per_fold.iterrows():
                mlflow.log_metrics({f"{m}_{row['fold']}": row[m] for m in metric_cols})
            mlflow.log_metrics(per_fold[metric_cols].mean().to_dict())   # the headline numbers
        results.append(per_fold)

    results = pd.concat(results)
    summary = results.groupby("model", sort=False)[["mae", "rmse", "rmsse", "bias", "rmsse_clean", "bias_clean"]].mean()
    print("\nPer fold:\n", results.round(3).to_string(index=False))
    print("\nAverage over folds (lower RMSSE is better; bias +0.10 = 10% over-forecast):\n", summary.round(3))


if __name__ == "__main__":
    main()
