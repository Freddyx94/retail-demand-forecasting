-- Leakage guard: lag_28 must be exactly the sales of 28 calendar days
-- earlier, and the 7-day rolling mean must end on that same day.
-- If the window frames in feat_daily_sales were changed by mistake
-- (e.g. "0 preceding" instead of "28 preceding"), this test fails.
-- Returns every row where a feature does not match its definition.

with feat as (

    select * from {{ ref('feat_daily_sales') }}

),

sales as (

    select * from {{ ref('fct_daily_sales') }}

),

expected as (

    select
        f.item_id,
        f.date,
        f.lag_28,
        f.rmean_7_lag28,
        past.units_sold as expected_lag_28,
        (
            select avg(w.units_sold)
            from sales as w
            where w.item_id = f.item_id
              and w.date between f.date - 34 and f.date - 28
        ) as expected_rmean_7
    from feat as f
    left join sales as past
        on  past.item_id = f.item_id
        and past.date    = f.date - 28
    -- a sample is enough: the correlated subquery is slow on 2.3M rows
    where hash(f.item_id, f.date) % 100 = 0

)

select *
from expected
where lag_28 is distinct from expected_lag_28
   or abs(coalesce(rmean_7_lag28, -1) - coalesce(expected_rmean_7, -1)) > 1e-9
