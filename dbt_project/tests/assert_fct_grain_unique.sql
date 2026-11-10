-- The grain is item x store x date: each combination must appear only once.
-- A single-column unique test can't check this, so we group by all three
-- columns and return any combination that appears more than once.

select
    date,
    item_id,
    store_id,
    count(*) as n_rows
from {{ ref('fct_daily_sales') }}
group by date, item_id, store_id
having count(*) > 1