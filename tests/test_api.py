"""API endpoints, against the small test database from conftest.py."""

import duckdb
from fastapi.testclient import TestClient

from api.main import app, get_db_path


def test_health_reports_the_served_forecasts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test_model"
    assert body["n_items"] == 2
    assert body["first_date"] == "2016-05-23"
    assert body["last_date"] == "2016-06-19"


def test_forecast_returns_28_days_in_date_order(client):
    response = client.get("/forecast/FOODS_1_002")
    assert response.status_code == 200
    days = response.json()["forecasts"]
    assert len(days) == 28
    assert [d["date"] for d in days] == sorted(d["date"] for d in days)
    assert days[0] == {"date": "2016-05-23", "forecast": 1.0}
    assert days[-1] == {"date": "2016-06-19", "forecast": 28.0}


def test_summary_totals_7_and_28_days(client):
    flat = client.get("/forecast/FOODS_1_001/summary").json()
    assert flat["units_next_7_days"] == 7.0
    assert flat["units_next_28_days"] == 28.0

    rising = client.get("/forecast/FOODS_1_002/summary").json()
    assert rising["units_next_7_days"] == 28.0       # 1 + 2 + ... + 7
    assert rising["units_next_28_days"] == 406.0     # 1 + 2 + ... + 28
    assert rising["first_date"] == "2016-05-23"


def test_unknown_item_is_404(client):
    for path in ["/forecast/NOT_AN_ITEM", "/forecast/NOT_AN_ITEM/summary"]:
        response = client.get(path)
        assert response.status_code == 404
        assert "NOT_AN_ITEM" in response.json()["detail"]


def test_sql_injection_is_treated_as_an_item_name(client):
    # If the ID were pasted into the SQL, this would match every row
    response = client.get("/forecast/x' OR '1'='1")
    assert response.status_code == 404


def test_missing_database_is_503(tmp_path):
    app.dependency_overrides[get_db_path] = lambda: tmp_path / "does_not_exist.duckdb"
    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503


def test_database_without_forecasts_is_503(tmp_path):
    empty = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty)).close()
    app.dependency_overrides[get_db_path] = lambda: empty
    try:
        response = TestClient(app).get("/forecast/FOODS_1_001")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "forecasting.predict" in response.json()["detail"]
