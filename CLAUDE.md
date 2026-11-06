# Project context: Retail Demand Forecasting

Working title: **"Retail Demand Forecasting: From Raw Sales Data to a Deployed Forecast"**
GitHub repo name: `retail-demand-forecasting`

## About me
I'm Saheed, a Data Science MSc graduate (UE Germany, Potsdam) based in Berlin. I work as a Warehouse Specialist at Inntralog (Rhenus), where I use SAP daily, so I know first-hand how inventory and order data is created. I'm building this project to strengthen my GitHub and CV for **Data Scientist, ML Engineer and Data Engineer** roles.

I already know: Python (pandas, NumPy, scikit-learn, XGBoost, PyTorch), SQL (CTEs, window functions), DuckDB, Apache Airflow, Power BI, Git, basic AWS.
New to me in this project: **dbt, dimensional modelling (star schema), LightGBM, time-series forecasting methods (ETS/ARIMA), MLflow, FastAPI, Docker, pytest, GitHub Actions.**

**How I want to work:** I want to LEARN, not just receive finished code.
- Explain what each step does and *why* before writing it.
- For SQL and feature engineering, let me attempt it first, then review my version.
- Keep code clean, commented and readable: recruiters will read it.
- Work in small steps and check results after each one (run it, look at the output).
- At the end of each stage, summarise what I learned and what I could say about it in an interview.

My previous project (for reference of style and standard): github.com/Freddyx94/late-delivery-analysis (SQL + DuckDB + Power BI, with a findings write-up).

---

## The goal
Build ONE end-to-end system where each part feeds the next:

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
             (tested in GitHub Actions)     (actual vs forecast, errors)
```

**Business question:** *How many units of each food product will one Walmart store sell over the next 28 days, and how much better is a machine learning model than a simple rule a store manager could use?*

---

## Data
Kaggle competition **"M5 Forecasting - Accuracy"** (Walmart). Downloading requires a Kaggle account and accepting the competition rules.

Files (put them in `data/raw/`):
- `calendar.csv`: one row per day (`d` = d_1 … d_1969, `date`, `wm_yr_wk`, weekday, month, year, `event_name_1/2`, `event_type_1/2`, `snap_CA/TX/WI`)
- `sell_prices.csv`: weekly price per store and item (`store_id`, `item_id`, `wm_yr_wk`, `sell_price`)
- `sales_train_evaluation.csv`: daily unit sales in **wide format**, one row per item × store, columns `d_1` … `d_1941` (2011-01-29 to 2016-05-22)
- Ignore `sales_train_validation.csv` and `sample_submission.csv`.

**Scope (keep it manageable):** store **CA_1**, category **FOODS** only (~1,400 products × ~1,940 days ≈ 2.8M rows). Define this scope ONCE in a config file so it can be changed later.

**Known data characteristics to handle:**
- Sales are wide (one column per day) and must be unpivoted to long format.
- Many products have lots of zero-sales days (intermittent demand).
- Prices are weekly and missing before a product launched, so a product's history only "starts" at its first price.
- Before launch, zeros are not real zeros: exclude pre-launch days from training and evaluation.

---

## Project structure
```
retail-demand-forecasting/
├── CLAUDE.md
├── README.md                  # project overview, results, how to run
├── FINDINGS.md                # stage 2 + 3 write-up for a non-technical reader
├── requirements.txt
├── config.yaml                # store, category, horizon, CV folds, paths
├── Makefile                   # one command per pipeline step (or run_pipeline.py on Windows)
├── data/
│   ├── raw/                   # Kaggle CSVs (git-ignored)
│   └── forecast.duckdb        # database (git-ignored)
├── ingestion/
│   └── load_raw.py            # CSV → DuckDB raw schema
├── dbt_project/               # dbt-duckdb project
│   ├── models/staging/        # stg_sales (unpivoted), stg_calendar, stg_prices
│   ├── models/marts/          # dim_item, dim_store, dim_date, fct_daily_sales
│   ├── models/ml/             # feature table for modelling
│   └── tests/                 # custom data tests
├── analysis/
│   └── 01_eda.ipynb           # stage 2 exploration
├── src/forecasting/
│   ├── features.py
│   ├── baselines.py
│   ├── models.py              # ETS + LightGBM
│   ├── evaluate.py            # metrics + rolling-origin CV
│   ├── train.py               # trains, logs to MLflow, saves best model
│   └── predict.py             # batch scoring → forecast table
├── api/
│   └── main.py                # FastAPI app
├── tests/                     # pytest (features, metrics, API)
├── Dockerfile
├── .github/workflows/ci.yml
├── dashboard/                 # .pbix + screenshot.png
└── .gitignore
```

---

## The process: 5 stages

Mark the current stage below and update it when a stage is finished.

**▶ CURRENT STAGE: Stage 1 — Data engineering** (not started; Stage 0 finished 2026-10-26)

---

### Stage 0: Setup (½ day)
1. Create the repo and folder structure above, a virtual environment, `.gitignore` (data/, *.duckdb, mlruns/, __pycache__/, .venv/, dbt target/ and logs/).
2. `requirements.txt`: duckdb, pandas, numpy, pyyaml, dbt-duckdb, lightgbm, statsforecast (for ETS/ARIMA), scikit-learn, mlflow, fastapi, uvicorn, pydantic, pytest, httpx, matplotlib, jupyter.
3. `config.yaml` with: `store_id: CA_1`, `cat_id: FOODS`, `horizon: 28`, `cv_folds: 3`, paths.
4. Download the M5 files into `data/raw/`.

**Done when:** `pip install -r requirements.txt` works and the three CSVs are in place.

---

### Stage 1: Data engineering (≈ 1 week)
**Goal:** a clean, tested star schema in DuckDB that everything else reads from.

1. **Ingestion** (`ingestion/load_raw.py`): load the three CSVs into a `raw` schema in DuckDB *as they are*. No cleaning here: raw stays raw. Print row counts.
2. **dbt setup:** `dbt init` with the dbt-duckdb adapter, pointing at `data/forecast.duckdb`. Explain to me what dbt is and why teams use it (versioned SQL, dependencies, tests, documentation).
3. **Staging models** (one per source, light cleaning only):
   - `stg_sales`: filter to the scope from config, **UNPIVOT** d_1…d_1941 into (item_id, store_id, d, units_sold).
   - `stg_calendar`: typed dates, event flags, SNAP flag for the store's state.
   - `stg_prices`: filtered to the store.
4. **Mart models (star schema):**
   - `dim_date` (date, weekday, week, month, year, is_event, event_type, is_snap)
   - `dim_item` (item_id, dept_id, cat_id, first_sale_date / launch week)
   - `dim_store` (store_id, state_id)
   - `fct_daily_sales` (date, item_id, store_id, units_sold, sell_price), only from each item's launch date onwards
   - Explain fact vs dimension tables and why a star schema helps analysts and BI tools.
5. **dbt tests:** `not_null`, `unique` on keys, `relationships` from fact to dimensions, plus custom tests: units_sold ≥ 0, no missing dates per item after launch, price > 0.
6. `dbt docs generate` and take a screenshot of the lineage graph for the README.

**Done when:** `dbt build` runs green (all models and tests pass), and I can explain every model.
**Interview point:** "I modelled raw retail data into a tested star schema with dbt."

---

### Stage 2: Analysis (≈ 1 week)
**Goal:** understand demand before modelling. The findings decide which features to build.

Answer these in SQL (against the marts) and visualise in `analysis/01_eda.ipynb`:
1. **Overall pattern:** total daily FOODS sales over time: trend? yearly seasonality? Christmas (store closed: sales = 0)?
2. **Weekly pattern:** sales by weekday. How strong is the weekend effect?
3. **Events and SNAP:** do sales change on event days and SNAP days (food-stamp days)? By how much?
4. **Price:** when an item's price drops, do its sales rise? Look at a few examples, then summarise across items.
5. **Intermittency:** what share of items sell 0 on most days? Group items into fast / medium / slow movers.
6. **Data issues:** anything strange (gaps, outliers, sudden stops)?

Write the key findings at the top of `FINDINGS.md`, each with the feature it leads to (e.g. "Strong weekly pattern → weekday feature and lag 7 multiples").
Check every claim against the data, and run a significance check where a comparison is made. (In my last project, two claims turned out to be wrong when checked, so: verify before writing.)

**Done when:** FINDINGS.md has 4–6 verified findings, each linked to a modelling decision.

---

### Stage 3: Machine learning (≈ 1.5 weeks)
**Goal:** a model that clearly beats a simple baseline, validated without leakage.

**Forecast setup:** daily units per item for the next **28 days** (M5 standard).

1. **Leakage rules (most important part):**
   - Use **rolling-origin (time-series) cross-validation**: e.g. 3 folds, each with a 28-day validation window that comes after all its training data. Never shuffle.
   - Use a **direct** forecast with features available on the forecast date: lags must be ≥ 28 days (lag_28, lag_35, lag_42 …), and rolling means must be computed on data shifted by 28 days. Explain why lag_7 would leak for a 28-day horizon.
   - Price and calendar features for future days are allowed (prices and events are known in advance).
   - Keep a final **hold-out**: the last 28 days are never used until the final evaluation.
2. **Features** (built as a dbt model `ml/feat_daily_sales` or in `features.py`): lags, rolling mean/std, weekday, month, event and SNAP flags, price, price change vs. item's average, item/department IDs as categorical, days since launch.
3. **Models, in order:**
   1. **Seasonal naive baseline:** sales = same weekday last week (for 28 days: repeat the last 4 weeks). This is "what a store manager would do".
   2. **Moving average baseline:** average of the last 28 days.
   3. **ETS / ARIMA** per item (statsforecast). Classical statistics.
   4. **LightGBM global model**: one model across all items, using the features above. Objective: Tweedie or Poisson (explain why: count data with many zeros).
4. **Metrics** (`evaluate.py`): MAE, RMSE, **RMSSE** (scale-free, used in M5, so slow and fast items are comparable) and **bias** (do we over- or under-forecast?). Report per fold and averaged.
5. **MLflow:** log every run (model, parameters, features used, metrics per fold). Use `mlflow ui` to compare runs. Screenshot the comparison for the README.
6. **Light tuning only:** a few LightGBM parameters. Explain overfitting risk.
7. **Error analysis:** where does the model fail? By department, by fast/slow movers, on event days. Feature importance. Add 3–4 findings to FINDINGS.md.
8. **Final evaluation** on the hold-out, once, at the end.

**Done when:** a results table (model × metric) exists, LightGBM's improvement over the seasonal-naive baseline is stated as a % on RMSSE, and the best model is saved.
**Interview point:** "My LightGBM model beat a seasonal-naive baseline by X% RMSSE, validated with rolling-origin CV to avoid look-ahead bias."
Be honest if ETS or a baseline wins for some item groups. That's a finding, not a failure.

---

### Stage 4: Serving the forecast (≈ 1 week)
**Goal:** the model is usable by others, tested and reproducible.

1. **Batch scoring** (`predict.py`): train the final model on all data, forecast the next 28 days for every item, write to a `forecasts` table in DuckDB (item_id, date, forecast, model_version, created_at).
2. **FastAPI** (`api/main.py`):
   - `GET /health`
   - `GET /forecast/{item_id}`: the 28-day forecast for one item (reads from the forecasts table)
   - `GET /forecast/{item_id}/summary`: total forecast units for the next 7 / 28 days
   - Pydantic response models, proper 404 for unknown items.
3. **pytest** (`tests/`): tests for feature functions (e.g. no lag shorter than 28), metric functions (known inputs → known outputs), and API endpoints (using a small fixture database, not the full data).
4. **Docker:** a `Dockerfile` that runs the API. Explain images vs containers. `docker build` + `docker run`, test with the browser at `/docs`.
5. **GitHub Actions** (`.github/workflows/ci.yml`): on every push, install dependencies and run pytest (with the small fixture data).
6. **One-command pipeline:** `make all` (or `python run_pipeline.py` on Windows) runs ingestion → dbt build → train → predict.

**Done when:** `docker run` serves forecasts, CI shows a green tick on GitHub.

---

### Stage 5: Dashboard and write-up (≈ 3–4 days)
1. **Power BI** (reads the marts + forecasts table, exported to CSV if needed):
   - KPI cards: model RMSSE, % improvement vs baseline, forecast units next 28 days.
   - Actual vs forecast line for a selectable item (slicer).
   - Error by department / by fast-slow movers.
   - Forecast for the next 28 days by department.
   - Titles state findings, two colours, consistent definitions across cards (lessons from the last dashboard).
2. **FINDINGS.md**, final version: analysis findings, model results table, error analysis, limitations (one store, one category; M5 is real Walmart data, but results may not generalise), and a "what I'd do next" section.
3. **README.md:** one-paragraph summary, architecture diagram, results table with the % improvement, dashboard screenshot, dbt lineage and MLflow screenshots, how to run (local + Docker).
4. **CV bullet + LinkedIn post** (draft them with me at the end).

**Done when:** README, FINDINGS and the dashboard all show the same numbers.

---

## Rules
- Data files, the .duckdb file, mlruns/ and model binaries must never go into Git (.gitignore handles this).
- **No look-ahead:** every feature must be available at forecast time; every validation window comes after its training data.
- Every number in README/FINDINGS must come from a query or script I can rerun.
- **Saheed commits everything.** Claude never runs `git add`, `git commit` or `git push`, for any reason. Instead, tell me when a stage has ended (or a step is a good commit point) and I will commit and upload to GitHub myself.
- If a step fails, show me the error and explain it before fixing it.
- At the end of each stage: update "CURRENT STAGE" above, summarise what I learned, and remind me what the next stage involves before starting it.
