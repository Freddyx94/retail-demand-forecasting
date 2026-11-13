-- 02_forecast_patterns.sql
-- Checks behind "The forecast for the next 28 days" in FINDINGS.md:
-- does the served forecast keep the patterns found in the analysis?
-- Run after `python run_pipeline.py`: paste the queries into the notebook
-- (con.sql("...").df()), or with the DuckDB command-line tool:
--   duckdb -readonly data/forecast.duckdb < analysis/02_forecast_patterns.sql

-- 1. Totals: next 28 days vs the last 28 days of actual sales, and by department
select
    (select sum(forecast) from serving.forecasts)                       as forecast_units_28d,
    (select sum(units_sold) from marts.fct_daily_sales
      where date > (select max(date) from marts.fct_daily_sales) - 28)  as actual_units_last_28d;

select
    i.dept_id,
    round(sum(f.forecast))                                              as forecast_units,
    round(100 * sum(f.forecast) / sum(sum(f.forecast)) over (), 1)      as share_pct
from serving.forecasts as f
join marts.dim_item as i using (item_id)
group by i.dept_id
order by i.dept_id;

-- 2. Daily totals with weekday, event and SNAP flag
--    (weekend vs midweek, Memorial Day vs other Mondays, SNAP Wednesdays)
select
    f.date,
    d.weekday,
    d.event_name,
    d.is_snap,
    round(sum(f.forecast))                                              as forecast_units
from serving.forecasts as f
join marts.dim_date as d using (date)
group by all
order by f.date;

-- 3. Weekend vs midweek (Tuesday-Thursday) ratio of the average daily forecast
with daily as (
    select d.weekday, sum(f.forecast) as units
    from serving.forecasts as f
    join marts.dim_date as d using (date)
    group by f.date, d.weekday
)
select round(
    avg(case when weekday in ('Saturday', 'Sunday') then units end)
  / avg(case when weekday in ('Tuesday', 'Wednesday', 'Thursday') then units end), 2
) as weekend_vs_midweek
from daily;
