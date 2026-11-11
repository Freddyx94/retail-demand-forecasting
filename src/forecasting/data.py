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


def connect(config: dict) -> duckdb.DuckDBPyConnection:
    """Open the project database read-only (dbt is the only writer)."""
    return duckdb.connect(str(ROOT / config["paths"]["database"]), read_only=True)


# Text columns stored as pandas categories: less memory, and LightGBM
# treats them as categorical features. Setting the categories once, on the
# full table, keeps the codes identical in every training and validation split.
CATEGORICAL = ["item_id", "dept_id", "weekday", "event_name", "event_type", "next_event_name"]


def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The feature table (ml.feat_daily_sales): one row per item x date.

    Holds the target (units_sold), all model features and the
    is_likely_stockout flag used to clean training labels and split the
    evaluation. The baselines only use item_id, date and units_sold.
    """
    df = con.sql("SELECT * FROM ml.feat_daily_sales ORDER BY item_id, date").df()
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")
    return df
