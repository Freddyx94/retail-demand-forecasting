"""Light tuning of LightGBM on the CV folds (never the hold-out).

Usage (from the project root):
    python -m forecasting.tune

Changes ONE setting at a time from the current defaults (models.LGBM_PARAMS)
and runs the same rolling-origin CV for each. A small, one-at-a-time search
keeps the risk of overfitting to the three CV windows low: the more settings
are tried, the more likely one wins by luck. The hold-out is the check.
"""

from functools import partial

import mlflow
import pandas as pd

from forecasting import models
from forecasting.data import connect, load_config, load_features
from forecasting.evaluate import make_folds
from forecasting.train import cross_validate, setup_mlflow

# name -> (parameter overrides, number of trees)
CANDIDATES = {
    "default": ({}, models.LGBM_ROUNDS),
    # Tweedie power: closer to 1 = Poisson-like counts, higher = heavier zeros/spikes
    "tweedie_1.3": ({"tweedie_variance_power": 1.3}, models.LGBM_ROUNDS),
    "tweedie_1.5": ({"tweedie_variance_power": 1.5}, models.LGBM_ROUNDS),
    # Tree size: smaller trees = simpler rules, larger = more interactions
    "leaves_31": ({"num_leaves": 31}, models.LGBM_ROUNDS),
    "leaves_127": ({"num_leaves": 127}, models.LGBM_ROUNDS),
    # Minimum rows per leaf: higher = rules must hold for more examples
    "min_leaf_500": ({"min_data_in_leaf": 500}, models.LGBM_ROUNDS),
    # Slower learning with more trees: usually a bit better, takes longer
    "lr_0.025_x1000": ({"learning_rate": 0.025}, 1000),
}


def main() -> None:
    config = load_config()
    horizon, n_folds = config["forecast"]["horizon"], config["forecast"]["cv_folds"]
    with connect(config) as con:
        sales = load_features(con)
    cv_folds = [f for f in make_folds(sales["date"].max(), horizon, n_folds) if f.name != "holdout"]

    setup_mlflow(config)
    rows = []
    for name, (params, rounds) in CANDIDATES.items():
        forecast_fn = partial(models.lightgbm, params=params, rounds=rounds)
        with mlflow.start_run(run_name=f"tune_{name}"):
            per_fold = cross_validate(f"lightgbm_{name}", forecast_fn, sales, cv_folds)
            mlflow.log_params({**models.LGBM_PARAMS, **params, "num_boost_round": rounds, "tuning": name})
            mean = per_fold.drop(columns=["model", "fold", "n_items"]).mean()
            mlflow.log_metrics(mean.to_dict())
        rows.append({"candidate": name, **mean.to_dict()})
        print(f"{name:16s} rmsse {mean['rmsse']:.4f}  bias {mean['bias']:+.3f}", flush=True)

    table = pd.DataFrame(rows).set_index("candidate")[["mae", "rmse", "rmsse", "bias", "rmsse_clean", "bias_clean"]]
    print("\nAverage over CV folds (lower RMSSE is better):\n", table.round(4))


if __name__ == "__main__":
    main()
