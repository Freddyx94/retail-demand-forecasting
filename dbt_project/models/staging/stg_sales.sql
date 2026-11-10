-- stg_sales.sql
-- Staging model for daily unit sales.
-- Source data is wide (one column per day: d_1 ... d_1941).
-- This model keeps only our scope (one store, one category) and
-- unpivots the day columns into long format: one row per item per day.
--
-- Filter BEFORE unpivoting: the raw table has 30,490 rows (all stores and
-- categories). Unpivoting first would create 30,490 x 1,941 = ~59 million
-- rows, and we would then throw away ~95% of them. Filtering first leaves
-- only the CA_1 FOODS rows, so DuckDB unpivots about 2.8 million rows instead.

with source as (

    select * from {{ source('raw', 'sales_train_evaluation') }}

),

scoped as (

    select *
    from source
    where store_id = '{{ var("store_id") }}'
      and cat_id   = '{{ var("cat_id") }}'

),

unpivoted as (

    unpivot scoped
    on columns('^d_[0-9]+$')        -- only the day columns: d_ followed by digits
    into name d value units_sold

)

select
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    units_sold
from unpivoted
