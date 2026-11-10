-- stg_prices.sql
-- Weekly sell price per item, for our store and category only.
-- sell_prices has no cat_id column, but every item_id starts with its
-- category (e.g. FOODS_3_090), so we filter on that prefix.

with source as (

    select * from {{ source('raw', 'sell_prices') }}

)

select
    store_id,
    item_id,
    wm_yr_wk,
    sell_price
from source
where store_id = '{{ var("store_id") }}'
  -- starts_with is used instead of LIKE 'FOODS_%', because in LIKE the
  -- underscore is a wildcard meaning "any single character"
  and starts_with(item_id, '{{ var("cat_id") }}_')