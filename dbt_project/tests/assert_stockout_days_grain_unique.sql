-- The grain of stockout_days is item x date: each combination must appear
-- only once, otherwise joining it to the fact table would duplicate rows.

select
    item_id,
    date,
    count(*) as n_rows
from {{ ref('stockout_days') }}
group by item_id, date
having count(*) > 1
