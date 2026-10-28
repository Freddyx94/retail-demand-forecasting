"""Load the M5 CSV files into the `raw` schema of the DuckDB database.

Raw stays raw: the files are loaded exactly as they are (all stores, all
categories, sales still in wide format). Filtering to the project scope,
unpivoting and cleaning all happen later in dbt, so they are versioned
and tested in one place.

Usage:
    python ingestion/load_raw.py
"""

from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Table name in DuckDB -> source file name in data/raw/
RAW_FILES = {
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sales_train_evaluation": "sales_train_evaluation.csv",
}


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    db_path = PROJECT_ROOT / config["paths"]["database"]

    # Fail early with a clear message if a file is missing.
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {raw_dir}: {', '.join(missing)}")

    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    for table, file_name in RAW_FILES.items():
        csv_path = (raw_dir / file_name).as_posix()
        # CREATE OR REPLACE makes the script safe to re-run (idempotent).
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{table} AS "
            f"SELECT * FROM read_csv('{csv_path}', header = true)"
        )
        n_rows = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        n_cols = len(con.execute(f"DESCRIBE raw.{table}").fetchall())
        print(f"raw.{table:<24} {n_rows:>10,} rows  {n_cols:>5,} columns")

    con.close()
    print(f"\nLoaded into {db_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
