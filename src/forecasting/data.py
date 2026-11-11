"""Shared helpers: project config and reading data from DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd
import yaml

# Project root = the folder that contains config.yaml (two levels above src/forecasting).
ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    """Read config.yaml, the single source of truth for scope and settings."""
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def connect(config: dict, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the project database, read-only by default.

    dbt writes the staging, marts and ml schemas; predict.py is the only
    Python writer (schema serving). Read-only connections cannot lock others out.
    """
    return duckdb.connect(str(ROOT / config["paths"]["database"]), read_only=read_only)


# Text columns stored as pandas categories: less memory, and LightGBM
# treats them as categorical features. Setting the categories once, on the
# full table, keeps the codes identical in every training and validation split.
CATEGORICAL = ["item_id", "dept_id", "weekday", "event_name", "event_type", "next_event_name"]


def load_features(con: duckdb.DuckDBPyConnection, future: bool = False) -> pd.DataFrame:
    """The feature table (ml.feat_daily_sales): one row per item x date.

    Holds the target (units_sold), all model features and the
    is_likely_stockout flag used to clean training labels and split the
    evaluation. The baselines only use item_id, date and units_sold.

    future=False: days with known sales (training and evaluation).
    future=True:  the next 28 days after the last sales day (scoring);
                  units_sold is empty there.
    """
    df = con.sql(f"""
        SELECT * FROM ml.feat_daily_sales
        WHERE is_future = {future}
        ORDER BY item_id, date
    """).df()
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")
    return df
