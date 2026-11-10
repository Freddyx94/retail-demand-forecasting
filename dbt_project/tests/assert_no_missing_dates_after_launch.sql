-- Every item must have a row for every day from its launch_date to the
-- last sales day, with no gaps.
--
-- Approach: gap-based (window function), not count-based.
-- A count-based check compares each item's row count with the expected
-- number of days. On its own, a missing day and a duplicated day would
-- cancel out and pass unnoticed. The gap-based check finds each missing
-- day directly, using LAG to compare every date with the previous one.
--
-- LAG alone can't see gaps at the edges (missing days right after launch
-- or before the end), because there is no "previous row" to compare with.
-- So two boundary checks are added: each item's first date must equal its
-- launch_date, and its last date must equal the last sales day.

with item_days as (

    select
        item_id,
        date,
        lag(date) over (partition by item_id order by date) as prev_date
    from {{ ref('fct_daily_sales') }}

),

last_day as (

    -- Last sales day taken from the raw sales data (via staging), NOT from
    -- fct_daily_sales: otherwise the test would check the fact table against
    -- itself, and a day missing for every item would go unnoticed.
    select max(c.date) as last_sales_date
    from {{ ref('stg_sales') }} as s
    join {{ ref('stg_calendar') }} as c
        on s.d = c.d

),

item_bounds as (

    select
        f.item_id,
        min(f.date) as first_date,
        max(f.date) as last_date,
        i.launch_date
    from {{ ref('fct_daily_sales') }} as f
    join {{ ref('dim_item') }} as i
        on f.item_id = i.item_id
    group by f.item_id, i.launch_date

)

-- 1. gaps in the middle: a day more than 1 day after the previous one
select
    item_id,
    'gap in the middle' as problem,
    prev_date as from_date,
    date as to_date
from item_days
where date_diff('day', prev_date, date) > 1

union all

-- 2. missing days at the start
select
    item_id,
    'does not start at launch_date' as problem,
    launch_date as from_date,
    first_date as to_date
from item_bounds
where first_date <> launch_date

union all

-- 3. missing days at the end
select
    b.item_id,
    'ends before last sales day' as problem,
    b.last_date as from_date,
    l.last_sales_date as to_date
from item_bounds as b
cross join last_day as l
where b.last_date <> l.last_sales_date