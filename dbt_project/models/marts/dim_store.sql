-- dim_store.sql
-- One row per store in scope. With our current scope this is a single row
-- (CA_1), but building it as a dimension means the model still works if
-- the scope is widened to more stores later.

select distinct
    store_id,
    state_id
from {{ ref('stg_sales') }}