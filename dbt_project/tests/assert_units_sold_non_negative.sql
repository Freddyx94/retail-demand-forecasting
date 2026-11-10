-- Sales are counts: a negative value would mean a data error.
-- Returns every row that breaks the rule; the test passes if none are returned.

select *
from {{ ref('fct_daily_sales') }}
where units_sold < 0