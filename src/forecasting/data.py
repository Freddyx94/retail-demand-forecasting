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


def load_sales(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Daily sales per item, with the stockout flag used to split the evaluation.

    One row per item x date from each item's launch to the last sales day.
    """
    return con.sql("""
        SELECT item_id, dept_id, date, units_sold, is_likely_stockout
        FROM ml.feat_daily_sales
        ORDER BY item_id, date
    """).df()
