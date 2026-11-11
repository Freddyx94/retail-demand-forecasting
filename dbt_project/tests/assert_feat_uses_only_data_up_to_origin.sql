-- Leakage guard for feat_daily_sales: sales-history features may only use
-- data up to the row's forecast origin. Recomputes them from the fact table
-- on a 1% sample and returns every row that breaks a rule:
--   1. the origin is before the day, and the horizon is 1..28;
--   2. mean_28d is the mean of the 28 days ENDING ON the origin;
--   3. same_weekday_last is the sales of the same weekday, at most 7 days
--      before the origin and never after it.
-- If a window frame or join in the model is changed by mistake (e.g. to use
-- the target day instead of the origin), this test fails.

{% set horizon = var('horizon') %}

with feat as (

    select * from {{ ref('feat_daily_sales') }}
    -- a sample is enough: the correlated subquery is slow on 2.3M rows
    where hash(item_id, date) % 100 = 0

),

sales as (

    select * from {{ ref('fct_daily_sales') }}

),

checks as (

    select
        f.item_id,
        f.date,
        f.origin_date,
        f.horizon,
        f.mean_28d,
        f.same_weekday_last,
        (
            select avg(s.units_sold)
            from sales as s
            where s.item_id = f.item_id
              and s.date between f.origin_date - {{ horizon - 1 }} and f.origin_date
        ) as expected_mean_28d,
        (
            select s.units_sold
            from sales as s
            where s.item_id = f.item_id
              and s.date <= f.origin_date
              and s.date >  f.origin_date - 7
              and dayofweek(s.date) = dayofweek(f.date)
        ) as expected_same_weekday_last
    from feat as f

)

select *
from checks
where not (origin_date < date and horizon between 1 and {{ horizon }})
   or abs(coalesce(mean_28d, -1) - coalesce(expected_mean_28d, -1)) > 1e-9
   or same_weekday_last is distinct from expected_same_weekday_last
