-- feat_daily_sales.sql
-- The table the forecasting model learns from.
-- Grain: one row per item x date (fct_daily_sales plus 28 future days).
-- Each row holds the target (units_sold) and everything a planner would
-- have KNOWN on the day the forecast was made.
--
-- FORECAST ORIGIN
-- Forecasts are made every 28 days (var 'horizon') for the next 28 days.
-- The origins are counted back from the last sales day in steps of 28, so
-- they coincide with the cross-validation origins in evaluate.py:
--
--   ... | origin | day 1 ... day 28 | origin | day 1 ... day 28 | last day
--
-- Every row gets the origin of its 28-day block and its horizon
-- (1 = the day after the origin, 28 = the last day of the block).
--
-- THE LEAKAGE RULE
--   * sales-history features use only days up to the origin;
--   * calendar, event and price features may use the day itself:
--     holidays, SNAP days and prices are planned and known in advance.
-- tests/assert_feat_uses_only_data_up_to_origin.sql checks this.
--
-- Why origin-based (an earlier version used lags of >= 28 days per row):
-- shifting every feature by 28 days threw away up to 27 days of the most
-- recent sales. Measured on the CV folds, a 28-day average computed that
-- way scored RMSSE 0.740 instead of 0.717 at the origin.
--
-- FUTURE ROWS (Stage 4: serving)
-- The table also holds the next 28 days after the last sales day, one row
-- per item still on sale, with is_future = TRUE and units_sold = NULL.
-- Their origin is the last sales day. Training and scoring rows come from
-- the SAME SQL, so the features the deployed model sees are computed
-- exactly like the ones it learned from (no "training-serving skew").
--
-- Each feature group links to the finding in FINDINGS.md that justified it.

{% set horizon = var('horizon') %}

with actuals as (

    select * from {{ ref('fct_daily_sales') }}

),

last_day as (

    select max(date) as last_day from actuals

),

-- 0. The days to forecast --------------------------------------------------
-- Every item sold on the last day x the next 28 days. Calendar and prices
-- for these days are already known (stg_calendar and sell_prices run to
-- d_1969); only the sales are unknown.

future as (

    select
        d.date,
        a.item_id,
        a.store_id,
        cast(null as integer) as units_sold,    -- unknown: this is what we forecast
        p.sell_price
    from actuals as a
    cross join last_day as l
    join {{ ref('dim_date') }} as d
        on  d.date >  l.last_day
        and d.date <= l.last_day + {{ horizon }}
    left join {{ ref('stg_prices') }} as p
        on  p.store_id = a.store_id
        and p.item_id  = a.item_id
        and p.wm_yr_wk = d.wm_yr_wk
    where a.date = l.last_day

),

sales as (

    select *, false as is_future from actuals
    union all by name
    select *, true  as is_future from future

),

items as (

    select * from {{ ref('dim_item') }}

),

dates as (

    select * from {{ ref('dim_date') }}

),

stockouts as (

    select * from {{ ref('stockout_days') }}

),

-- 1. Forecast origin and horizon of every row -------------------------------

with_origin as (

    select
        s.*,
        case
            -- the forecast we serve is made on the last sales day
            when s.is_future then l.last_day
            -- (cast: DuckDB subtracts INTEGER days from a DATE, but not BIGINT)
            else l.last_day - ({{ horizon }} * ((l.last_day - s.date) // {{ horizon }} + 1))::int
        end as origin_date
    from sales as s
    cross join last_day as l

),

-- 2. The item's state at the end of each day (the "planner's view") --------
-- Computed for every day; each row later picks the state at its origin.

daily_state as (

    select
        item_id,
        date,

        -- Recent level (finding 1: demand per item drifts, so the model
        -- needs the item's recent level). Windows of whole weeks.
        avg(units_sold)         over (w rows between  6 preceding and current row) as mean_7d,
        avg(units_sold)         over (w rows between 27 preceding and current row) as mean_28d,
        avg(units_sold)         over (w rows between 55 preceding and current row) as mean_56d,
        stddev_samp(units_sold) over (w rows between 27 preceding and current row) as std_28d,

        -- Finding 6: a long run of zeros is a warning sign of an empty shelf
        max(case when units_sold > 0 then date end)
            over (w rows between unbounded preceding and current row) as last_sale_date

    from sales
    window w as (partition by item_id order by date)

),

-- 3. Same-weekday history (finding 2: strong weekly cycle) ------------------
-- For each day: its sales, and the mean of the same weekday over 4 weeks.
-- Partitioning by weekday makes "1 row back" mean "7 days back"; this is
-- safe because tests/assert_no_missing_dates_after_launch.sql guarantees
-- there are no gaps in the dates.

weekday_history as (

    select
        item_id,
        date,
        units_sold,
        avg(units_sold) over (
            partition by item_id, dayofweek(date)
            order by date
            rows between 3 preceding and current row
        ) as mean_4w
    from sales

),

-- 4. Next event (finding 3: the day BEFORE an event often moves sales more
-- than the day itself, e.g. +48% the day before Thanksgiving) -------------

event_days as (

    select date as event_date, event_name as next_event_name
    from dates
    where event_name is not null

),

next_event as (

    select
        d.date,
        e.next_event_name,
        e.event_date - d.date as days_to_next_event
    from dates as d
    asof join event_days as e
        on e.event_date >= d.date

),

-- 5. Past-only reference price (finding 4) ----------------------------------
-- Prices are known in advance, so the day's own price is allowed, but the
-- reference must not use the item's full history (that includes the future).

prices as (

    select
        item_id,
        date,
        avg(sell_price) over (w rows between 364 preceding and 1 preceding) as ref_price_364d,
        lag(sell_price, 7) over w as price_last_week
    from sales
    window w as (partition by item_id order by date)

)

select
    -- Key, target, and when the forecast was made
    o.date,
    o.item_id,
    i.dept_id,
    o.units_sold,
    o.is_future,
    o.origin_date,
    o.date - o.origin_date             as horizon,

    -- Sales history, as known on the origin
    st.mean_7d,
    st.mean_28d,
    st.mean_56d,
    st.std_28d,
    o.origin_date - st.last_sale_date  as days_since_sale,

    -- Same weekday as the target day, most recent week on or before the
    -- origin (horizon 1-7 -> 7 days back, 8-14 -> 14 days back, ...)
    wh.units_sold                      as same_weekday_last,
    wh.mean_4w                         as same_weekday_mean_4w,

    -- Calendar (finding 1: month and Christmas; finding 2: weekday;
    -- finding 3: SNAP and the end-of-month dip)
    d.weekday,
    d.month,
    day(o.date)                        as day_of_month,
    d.is_snap,
    d.is_store_closed,

    -- Events (finding 3: by name, and the days before)
    d.event_name,
    d.event_type,
    n.next_event_name,
    n.days_to_next_event,

    -- Price (finding 4)
    o.sell_price,
    o.sell_price / p.ref_price_364d   as price_vs_ref,
    o.sell_price / p.price_last_week  as price_change_wow,

    -- Product life cycle (finding 1: range expansion)
    o.date - i.launch_date            as days_since_launch,

    -- NOT A FEATURE. Training-label filter only (finding 6): TRUE when the
    -- day is inside a likely stockout. It uses the item's whole history, so
    -- giving it to the model would leak the future.
    (so.date is not null)             as is_likely_stockout

from with_origin as o
join items as i
    on o.item_id = i.item_id
join dates as d
    on o.date = d.date
-- LEFT JOINs: in an item's first block after launch there is no state on
-- the origin yet (the item did not exist), so those features stay NULL.
left join daily_state as st
    on  st.item_id = o.item_id
    and st.date    = o.origin_date
left join weekday_history as wh
    on  wh.item_id = o.item_id
    and wh.date    = o.date - 7 * ceil((o.date - o.origin_date) / 7)::int
left join next_event as n
    on o.date = n.date
left join prices as p
    on  p.item_id = o.item_id
    and p.date    = o.date
left join stockouts as so
    on  so.item_id = o.item_id
    and so.date    = o.date
