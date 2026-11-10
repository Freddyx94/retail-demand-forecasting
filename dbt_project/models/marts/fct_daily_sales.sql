-- fct_daily_sales.sql
-- The centre of the star schema.
-- Grain: one row per item x store x date
--        ("units of one item sold in one store on one day").
-- Only days on or after each item's launch_date are kept; launch_date is
-- taken from dim_item so "launch" is defined in exactly one place.
--
-- LEFT JOIN to prices, not INNER JOIN:
-- If a week after launch had no price row, an INNER JOIN would silently
-- DROP those sales days. The row count would just be a bit lower, and
-- nobody would notice. A LEFT JOIN keeps the row with sell_price = NULL,
-- so the problem stays VISIBLE and a not_null test on sell_price will
-- catch it. (Same lesson as UNPIVOT silently dropping NULLs: prefer
-- the option that makes data problems loud.)
--
-- Expected row count: for each item, the number of days from launch_date
-- to the last sales day (2016-05-22), inclusive, summed over all items.
-- See the check query in the notes.

with sales as (

    select * from {{ ref('stg_sales') }}

),

calendar as (

    select * from {{ ref('stg_calendar') }}

),

prices as (

    select * from {{ ref('stg_prices') }}

),

items as (

    select * from {{ ref('dim_item') }}

)

select
    c.date,
    s.item_id,
    s.store_id,
    s.units_sold,
    p.sell_price
from sales as s

-- 1. turn 'd_1' into a real date, and get the week for the price join
join calendar as c
    on s.d = c.d

-- 3. keep only days from each item's launch onwards
join items as i
    on s.item_id = i.item_id
   and c.date >= i.launch_date

-- 2. attach the weekly price: a day belongs to one Walmart week,
--    and prices are per store, item and week
left join prices as p
    on  p.store_id = s.store_id
    and p.item_id  = s.item_id
    and p.wm_yr_wk = c.wm_yr_wk