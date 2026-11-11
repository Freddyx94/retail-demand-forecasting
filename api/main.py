"""Forecast API: serves the 28-day forecasts written by `python -m forecasting.predict`.

Run locally (from the project root):
    uvicorn api.main:app --reload
then open http://127.0.0.1:8000/docs for the interactive documentation.

The API only READS the serving.forecasts table; it never runs the model.
Forecasts are computed in batch beforehand, so every request is a quick
database lookup and the service does not need LightGBM installed.

Settings (environment variables):
    FORECAST_DB   path to the DuckDB file (default: data/forecast.duckdb)
"""

import os
from datetime import date, datetime
from pathlib import Path

import duckdb
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "forecast.duckdb"

app = FastAPI(
    title="Retail Demand Forecast API",
    description="28-day daily unit forecasts for Walmart store CA_1, FOODS category (M5 data).",
    version="1.0.0",
)


# --- Response models: the exact shape of every answer -----------------------

class Health(BaseModel):
    status: str = Field(examples=["ok"])
    model_version: str = Field(examples=["lightgbm_20160522"])
    created_at: datetime = Field(description="When the forecasts were made (UTC).")
    n_items: int = Field(description="Number of items with a forecast.")
    first_date: date
    last_date: date


class DailyForecast(BaseModel):
    date: date
    forecast: float = Field(description="Expected units sold that day.")


class ItemForecast(BaseModel):
    item_id: str = Field(examples=["FOODS_3_090"])
    model_version: str
    created_at: datetime
    forecasts: list[DailyForecast]


class ItemSummary(BaseModel):
    item_id: str = Field(examples=["FOODS_3_090"])
    model_version: str
    first_date: date
    units_next_7_days: float
    units_next_28_days: float


# --- Database access ---------------------------------------------------------

def get_db_path() -> Path:
    """Where the forecasts live. A FastAPI dependency, so tests can replace it."""
    return Path(os.environ.get("FORECAST_DB", DEFAULT_DB))


def query(db_path: Path, sql: str, params: list | None = None) -> list[tuple]:
    """Run one query on a short-lived, read-only connection.

    Read-only connections never lock the file, so the batch job can still
    write new forecasts while the API is running. 503 (service unavailable)
    if the database or the forecasts table is missing.
    """
    if not db_path.exists():
        raise HTTPException(503, f"Forecast database not found: {db_path.name}")
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            return con.execute(sql, params or []).fetchall()
    except duckdb.CatalogException:
        raise HTTPException(503, "No forecasts yet: run `python -m forecasting.predict`")


def item_rows(db_path: Path, item_id: str) -> list[tuple]:
    """All forecast rows of one item, by date; 404 if the item is unknown."""
    rows = query(db_path, """
        SELECT date, forecast, model_version, created_at
        FROM serving.forecasts
        WHERE item_id = ?
        ORDER BY date
    """, [item_id])   # "?" placeholder: the item ID is never pasted into the SQL text
    if not rows:
        raise HTTPException(404, f"No forecast for item '{item_id}'")
    return rows


# --- Endpoints -----------------------------------------------------------------

@app.get("/health", response_model=Health)
def health(db_path: Path = Depends(get_db_path)) -> Health:
    """Is the service up, and which forecasts is it serving?"""
    (version, created, n_items, first, last), = query(db_path, """
        SELECT any_value(model_version), max(created_at),
               count(DISTINCT item_id), min(date), max(date)
        FROM serving.forecasts
    """)
    return Health(status="ok", model_version=version, created_at=created,
                  n_items=n_items, first_date=first, last_date=last)


@app.get("/forecast/{item_id}", response_model=ItemForecast)
def forecast(item_id: str, db_path: Path = Depends(get_db_path)) -> ItemForecast:
    """The daily forecast for one item over the next 28 days."""
    rows = item_rows(db_path, item_id)
    return ItemForecast(
        item_id=item_id,
        model_version=rows[0][2],
        created_at=rows[0][3],
        forecasts=[DailyForecast(date=d, forecast=round(f, 2)) for d, f, _, _ in rows],
    )


@app.get("/forecast/{item_id}/summary", response_model=ItemSummary)
def summary(item_id: str, db_path: Path = Depends(get_db_path)) -> ItemSummary:
    """Total forecast units for one item: the next 7 and the next 28 days."""
    rows = item_rows(db_path, item_id)
    units = [f for _, f, _, _ in rows]
    return ItemSummary(
        item_id=item_id,
        model_version=rows[0][2],
        first_date=rows[0][0],
        units_next_7_days=round(sum(units[:7]), 1),
        units_next_28_days=round(sum(units), 1),
    )
