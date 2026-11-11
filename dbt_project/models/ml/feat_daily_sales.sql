-- feat_daily_sales.sql
-- The table the forecasting model learns from.
-- Grain: one row per item x date (same as fct_daily_sales).
-- Each row holds the target (units_sold) and everything that would have
-- been KNOWN 28 days before that day, when the forecast is made.
--
-- THE LEAKAGE RULE
-- We forecast 28 days ahead in one go (direct forecast). On the forecast
-- date, the most recent sales we can see are 28 days before the LAST
-- forecast day. So:
--   * every sales-history feature looks back at least 28 days
--     (lag_7 would use sales that have not happened yet);
--   * rolling windows END 28 days back ("rows between N preceding and
--     28 preceding"). Row-based windows equal day-based windows here
--     because tests/assert_no_missing_dates_after_launch.sql guarantees
--     there are no gaps in the dates;
--   * calendar, event and price features may use the day itself:
--     holidays, SNAP days and prices are planned and known in advance.
-- tests/assert_feat_lags_look_back_28_days.sql checks this.
--
-- Each feature group links to the finding in FINDINGS.md that justified it.

with sales as (

    select * from {{ ref('fct_daily_sales') }}

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

-- Finding 3: the day BEFORE an event often moves sales more than the day
-- itself (+48% the day before Thanksgiving). For every day, find the next
-- event on or after it. Events are known in advance, so this is allowed.
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

history as (

    select
        s.date,
        s.item_id,
        s.units_sold,
        s.sell_price,

        -- Sales history (finding 2: weekly cycle -> lags in whole weeks,
        -- so every lag lands on the same weekday as the target day)
        lag(s.units_sold, 28) over w as lag_28,
        lag(s.units_sold, 35) over w as lag_35,
        lag(s.units_sold, 42) over w as lag_42,

        -- Recent level (finding 1: demand per item drifts, so the model
        -- needs the item's recent level, not its lifetime average).
        -- Windows of whole weeks (7, 28, 56 days), all ending 28 days back.
        avg(s.units_sold) over (w rows between 34 preceding and 28 preceding)
            as rmean_7_lag28,
        avg(s.units_sold) over (w rows between 55 preceding and 28 preceding)
            as rmean_28_lag28,
        avg(s.units_sold) over (w rows between 83 preceding and 28 preceding)
            as rmean_56_lag28,
        stddev_samp(s.units_sold) over (w rows between 55 preceding and 28 preceding)
            as rstd_28_lag28,

        -- Finding 6: a long run of zeros is a warning sign. Last day with a
        -- sale, as seen 28 days back (the most recent data we would have).
        max(case when s.units_sold > 0 then s.date end)
            over (w rows between unbounded preceding and 28 preceding)
            as last_sale_date_lag28,

        -- Price (finding 4). Prices are known in advance, so the price of
        -- the day itself is allowed. The reference price uses only PAST
        -- prices (the previous 364 days), never the item's full history.
        avg(s.sell_price) over (w rows between 364 preceding and 1 preceding)
            as ref_price_364d,
        lag(s.sell_price, 7) over w as price_last_week

    from sales as s
    window w as (partition by s.item_id order by s.date)

)

select
    -- Key and target
    h.date,
    h.item_id,
    i.dept_id,
    h.units_sold,

    -- Sales history
    h.lag_28,
    h.lag_35,
    h.lag_42,
    h.rmean_7_lag28,
    h.rmean_28_lag28,
    h.rmean_56_lag28,
    h.rstd_28_lag28,
    h.date - h.last_sale_date_lag28 as days_since_sale_lag28,

    -- Calendar (finding 1: month and Christmas; finding 2: weekday;
    -- finding 3: SNAP and the end-of-month dip)
    d.weekday,
    d.month,
    day(h.date)          as day_of_month,
    d.is_snap,
    d.is_store_closed,

    -- Events (finding 3: by name, and the days before)
    d.event_name,
    d.event_type,
    n.next_event_name,
    n.days_to_next_event,

    -- Price (finding 4)
    h.sell_price,
    h.sell_price / h.ref_price_364d   as price_vs_ref,
    h.sell_price / h.price_last_week  as price_change_wow,

    -- Product life cycle (finding 1: range expansion)
    h.date - i.launch_date            as days_since_launch,

    -- NOT A FEATURE. Training-label filter only (finding 6): TRUE when the
    -- day is inside a likely stockout. It uses the item's whole history, so
    -- giving it to the model would leak the future.
    (so.date is not null)             as is_likely_stockout

from history as h
join items as i
    on h.item_id = i.item_id
join dates as d
    on h.date = d.date
left join next_event as n
    on h.date = n.date
left join stockouts as so
    on  h.item_id = so.item_id
    and h.date    = so.date
