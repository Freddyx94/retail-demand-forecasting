# Retail Demand Forecasting: From Raw Sales Data to a Deployed Forecast

> **Status:** work in progress

**Business question:** How many units of each food product will one Walmart store sell over the next 28 days, and how much better is a machine learning model than a simple rule a store manager could use?

## Pipeline

```
Raw CSVs ──► DuckDB (raw) ──► dbt star schema ──► Analysis (findings)
                                      │                  │ informs features
                                      ▼                  ▼
                               Feature table ──► ML (baseline → ETS → LightGBM)
                                                         │  MLflow tracking
                                                         ▼
                                Forecast table in DuckDB ◄── batch scoring
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                FastAPI + Docker               Power BI dashboard
```

## Data

[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) (Walmart, Kaggle).
Scope: store `CA_1`, category `FOODS`, set in [`config.yaml`](config.yaml).

Place `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` in `data/raw/` (not committed).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
