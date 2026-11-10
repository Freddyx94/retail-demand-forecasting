-- dim_item.sql
-- One row per item in scope, with its launch information.
--
-- Which "launch" should the fact table start from: first price or first sale?
-- First PRICE. A price means the item was on the shelf and could be bought.
-- If an item had a price for three weeks but sold nothing, those zeros are
-- REAL: customers could buy it and didn't. The model must learn from them,
-- otherwise it would overestimate demand for slow-selling items.
-- Before the first price, the item did not exist in the store, so those
-- zeros are NOT real demand and must be excluded.
-- Using first sale instead would wrongly drop real zero-demand days.

with items as (

    select distinct
        item_id,
        dept_id,
        cat_id
    from {{ ref('stg_sales') }}

),

launch as (

    -- first week each item had a price
    select
        item_id,
        min(wm_yr_wk) as launch_week
    from {{ ref('stg_prices') }}
    group by item_id

),

week_start as (

    -- first day of each Walmart week
    select
        wm_yr_wk,
        min(date) as week_start_date
    from {{ ref('stg_calendar') }}
    group by wm_yr_wk

),

first_sale as (

    -- first day each item actually sold at least one unit
    select
        s.item_id,
        min(c.date) as first_sale_date
    from {{ ref('stg_sales') }} as s
    join {{ ref('stg_calendar') }} as c
        on s.d = c.d
    where s.units_sold > 0
    group by s.item_id

)

select
    i.item_id,
    i.dept_id,
    i.cat_id,
    l.launch_week,
    w.week_start_date as launch_date,
    f.first_sale_date
from items as i
left join launch     as l on i.item_id = l.item_id
left join week_start as w on l.launch_week = w.wm_yr_wk
left join first_sale as f on i.item_id = f.item_id