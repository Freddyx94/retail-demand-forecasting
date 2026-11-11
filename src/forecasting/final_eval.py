"""Final evaluation on the hold-out (the last 28 days). Run ONCE, at the end of Stage 3.

Usage (from the project root):
    python -m forecasting.final_eval

Every model is trained on all data up to the hold-out origin and forecasts
the last 28 days, which no earlier step has used. Model choices were made on
the CV folds only, so this is an unbiased estimate of real performance.

Outputs:
  * results/holdout_results.csv: model x metric table (for README/FINDINGS)
  * models/lightgbm.txt: the evaluated LightGBM model (git-ignored)
  * MLflow runs named holdout_<model>
"""

from functools import partial

import mlflow
import pandas as pd

from forecasting import baselines, models
from forecasting.data import ROOT, connect, load_config, load_features
from forecasting.error_analysis import item_profile, weighted_rmsse
from forecasting.evaluate import evaluate, make_folds, split
from forecasting.train import setup_mlflow

MODELS = {
    "seasonal_naive": baselines.seasonal_naive,
    "moving_average_28": baselines.moving_average,
    "ets": models.ets,
    "lightgbm_drop_stockouts": partial(models.lightgbm, drop_stockouts=True),
}


def main() -> None:
    config = load_config()
    horizon, n_folds = config["forecast"]["horizon"], config["forecast"]["cv_folds"]
    with connect(config) as con:
        sales = load_features(con)
    holdout = make_folds(sales["date"].max(), horizon, n_folds)[-1]
    assert holdout.name == "holdout"
    print(f"Train <= {holdout.train_end.date()}, forecast {holdout.valid_start.date()} .. {holdout.valid_end.date()}")

    train, valid = split(sales, holdout)
    profile = item_profile(train)

    # The main model: train once, keep it, and save it for Stage 4
    booster = models.fit_lightgbm(train)
    model_dir = ROOT / config["paths"]["model_dir"]
    model_dir.mkdir(exist_ok=True)
    booster.save_model(str(model_dir / "lightgbm.txt"))
    forecasts = {"lightgbm": valid.assign(forecast=booster.predict(valid[models.LGBM_FEATURES]))}
    for name, forecast_fn in MODELS.items():
        forecasts[name] = forecast_fn(train, valid)
        print(f"{name} done", flush=True)

    setup_mlflow(config)
    rows = []
    for name, predicted in forecasts.items():
        metrics = evaluate(predicted, train)
        # weighted_rmsse expects the error-analysis layout: one forecast column per model
        layout = predicted.assign(fold="holdout", **{name: predicted["forecast"]}) \
            .join(profile[["scale", "weight"]], on="item_id")
        metrics["weighted_rmsse"] = weighted_rmsse(layout, name)
        with mlflow.start_run(run_name=f"holdout_{name}"):
            mlflow.set_tag("evaluation", "holdout")
            mlflow.log_params({"model": name, "horizon": horizon})
            mlflow.log_metrics(metrics)
        rows.append({"model": name, **metrics})

    table = pd.DataFrame(rows).set_index("model")[
        ["mae", "rmse", "rmsse", "weighted_rmsse", "bias", "rmsse_clean", "bias_clean"]]
    for base in ["seasonal_naive", "moving_average_28", "ets"]:
        table[f"rmsse_better_than_{base}_%"] = 100 * (1 - table["rmsse"] / table.loc[base, "rmsse"])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    table.round(4).to_csv(out_dir / "holdout_results.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nHold-out results (lower RMSSE is better; bias +0.10 = 10% over-forecast):\n", table.round(3))


if __name__ == "__main__":
    main()
