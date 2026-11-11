# Dashboard

Power BI report on the forecasts: how accurate the model is, where it helps,
and what it forecasts for the next 28 days.

![Dashboard](screenshot.png)

## Data

Created by `python -m forecasting.export` in `dashboard/data/` (git-ignored:
item-level M5 sales may not be redistributed). The script recreates the
hold-out forecasts with the saved Stage 3 model and stops if their RMSSE
differs from [`results/holdout_results.csv`](../results/holdout_results.csv),
so the dashboard shows the same numbers as the README and FINDINGS.

| Table | Grain | Columns |
|---|---|---|
| `items` | item | item_id, dept_id, velocity, pattern, weight |
| `daily` | item × day | item_id, date, period (history / holdout / future), actual, lightgbm, seasonal_naive, moving_average_28 |
| `item_errors` | item × model (hold-out) | item_id, model, rmsse, actual_units, forecast_units |
| `model_results` | model | the hold-out results table |

Relationships: `items[item_id]` 1 → * `daily[item_id]` and `items[item_id]` 1 → * `item_errors[item_id]`.

## Measures (DAX)

RMSSE is computed per item in Python (each item has its own scale). The
RMSSE of any group of items is the **average over its items**, the same
definition as the headline number, so slicing by department or velocity
stays consistent with the cards.

```dax
RMSSE LightGBM        = CALCULATE ( AVERAGE ( item_errors[rmsse] ), item_errors[model] = "lightgbm" )
RMSSE Seasonal naive  = CALCULATE ( AVERAGE ( item_errors[rmsse] ), item_errors[model] = "seasonal_naive" )
RMSSE Moving average  = CALCULATE ( AVERAGE ( item_errors[rmsse] ), item_errors[model] = "moving_average_28" )

% better than seasonal naive = 1 - DIVIDE ( [RMSSE LightGBM], [RMSSE Seasonal naive] )
% better than moving average = 1 - DIVIDE ( [RMSSE LightGBM], [RMSSE Moving average] )

Bias LightGBM =
    DIVIDE (
        CALCULATE ( SUM ( item_errors[forecast_units] ), item_errors[model] = "lightgbm" ),
        CALCULATE ( SUM ( item_errors[actual_units] ),   item_errors[model] = "lightgbm" )
    ) - 1

Forecast units next 28 days = CALCULATE ( SUM ( daily[lightgbm] ), daily[period] = "future" )
Actual units                = SUM ( daily[actual] )
LightGBM forecast           = SUM ( daily[lightgbm] )
Seasonal naive forecast     = SUM ( daily[seasonal_naive] )
```

## Reference values (all items, hold-out 25 Apr – 22 May 2016)

| Measure | Value |
|---|---|
| RMSSE LightGBM | 0.754 |
| % better than seasonal naive | 26.1% |
| % better than moving average | 3.2% |
| Bias LightGBM | −6.8% |
| Forecast units next 28 days | 92,986 |

| Velocity | RMSSE LightGBM | % better than moving average |
|---|---|---|
| fast (≥5/day) | 0.724 | 8.3% |
| medium (1–5/day) | 0.750 | 3.6% |
| slow (<1/day) | 0.763 | 2.1% |

| Department | RMSSE LightGBM | % better than moving average | Forecast units next 28 days |
|---|---|---|---|
| FOODS_1 | 0.829 | 1.6% | 9,657 |
| FOODS_2 | 0.760 | 3.1% | 15,425 |
| FOODS_3 | 0.732 | 3.8% | 67,904 |
