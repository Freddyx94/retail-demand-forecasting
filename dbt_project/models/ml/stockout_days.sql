-- stockout_days.sql
-- Days that are probably out of stock rather than zero demand.
-- Grain: one row per item x date, ONLY for flagged days.
--
-- Why: the analysis (FINDINGS.md, finding 6) found long runs of zero sales
-- that are very unlikely for the item's normal sales rate, e.g. the best
-- seller selling nothing for 227 days. The data has no stock levels, so
-- these zeros are most likely stockouts. If the model trains on them, it
-- learns that best sellers sometimes have zero demand for months.
--
-- Rule (same as notebook section 6): a run of consecutive zero days is
-- flagged when
--   1. it lasts at least 28 days, and
--   2. zero_share ^ run_days < 1e-6, i.e. if each day were independently
--      zero with the item's usual probability, a run this long would
--      happen less than once in a million times, and
--   3. it does not reach the last sales day (a run at the end may be a
--      delisting, which we cannot tell apart from a stockout yet).
--
-- LEAKAGE WARNING: the rule uses the item's whole history (zero_share and
-- the full run length), so this table may only be used to REMOVE training
-- labels and to split the evaluation. It must never be used as a feature.

with sales as (

    select * from {{ ref('fct_daily_sales') }}

),

-- A new run_id starts after every day with a sale, so consecutive zero
-- days share one run_id.
numbered as (

    select
        item_id,
        date,
        units_sold,
        sum((units_sold > 0)::int) over (
            partition by item_id order by date
        ) as run_id
    from sales

),

-- Share of zero days per item. Christmas is excluded because the store is
-- closed then, which says nothing about the item's demand.
item_zero_share as (

    select
        item_id,
        avg((units_sold = 0)::int) as zero_share
    from sales
    where not (month(date) = 12 and day(date) = 25)
    group by item_id

),

zero_runs as (

    select
        item_id,
        run_id,
        min(date) as run_start,
        max(date) as run_end,
        count(*)  as run_days
    from numbered
    where units_sold = 0
    group by item_id, run_id

),

flagged_runs as (

    select r.*
    from zero_runs as r
    join item_zero_share as z
        on r.item_id = z.item_id
    where r.run_days >= 28
      and power(z.zero_share, r.run_days) < 1e-6
      and r.run_end < (select max(date) from sales)

)

-- Expand each flagged run back to one row per day.
select
    n.item_id,
    n.date,
    f.run_start,
    f.run_end,
    f.run_days
from numbered as n
join flagged_runs as f
    on  n.item_id = f.item_id
    and n.run_id  = f.run_id
where n.units_sold = 0
