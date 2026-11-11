"""Shared test fixtures.

The tests never touch data/forecast.duckdb: the API tests use a tiny
database built in a temporary folder, so they run anywhere, including
GitHub Actions, where the Kaggle data is not available.
"""

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_db_path

N_DAYS = 28


@pytest.fixture
def forecast_db(tmp_path):
    """A forecasts table with two items and easy-to-check numbers.

    FOODS_1_001: 1.0 unit every day           -> 7 days = 7,  28 days = 28
    FOODS_1_002: day n of the horizon = n units -> 7 days = 28, 28 days = 406
    """
    path = tmp_path / "test.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA serving")
        con.execute(f"""
            CREATE TABLE serving.forecasts AS
            SELECT
                item_id,
                DATE '2016-05-22' + h::INTEGER                AS date,   -- range() gives BIGINT
                CASE WHEN item_id = 'FOODS_1_001' THEN 1.0 ELSE h END::DOUBLE AS forecast,
                'test_model'                                  AS model_version,
                TIMESTAMP '2026-01-01 12:00:00'               AS created_at
            FROM (VALUES ('FOODS_1_001'), ('FOODS_1_002')) AS items(item_id)
            CROSS JOIN range(1, {N_DAYS + 1}) AS horizons(h)
        """)
    return path


@pytest.fixture
def client(forecast_db):
    """An API client that reads the test database instead of the real one."""
    app.dependency_overrides[get_db_path] = lambda: forecast_db
    yield TestClient(app)
    app.dependency_overrides.clear()
