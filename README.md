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

## Data model

Raw CSVs are loaded into DuckDB unchanged, then modelled with **dbt** into a tested star schema:

![dbt lineage graph](docs/images/dbt_lineage.png)

| Layer | Models | Purpose |
|---|---|---|
| `raw` | `calendar`, `sell_prices`, `sales_train_evaluation` | CSVs as delivered (all stores, all categories) |
| `staging` (views) | `stg_sales`, `stg_calendar`, `stg_prices` | Scope filter, unpivot of daily sales from wide to long, typed flags |
| `marts` (tables) | `fct_daily_sales`, `dim_date`, `dim_item`, `dim_store` | Star schema. Fact grain: one row per item × store × day |

Key modelling decision: an item's history starts at its **first price**, not its first sale. Before launch the item was not in the store, so those zeros are not real demand and are excluded; zeros after launch are real and kept.

Tests (`dbt build` runs them all): unique/not-null keys, fact-to-dimension relationships, unique fact grain, non-negative units, positive prices, and no missing days per item between launch and the last sales day.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python ingestion/load_raw.py    # CSVs -> DuckDB raw schema
cd dbt_project
dbt build                       # staging + marts, then all tests
dbt docs generate && dbt docs serve   # optional: documentation and lineage graph
```
