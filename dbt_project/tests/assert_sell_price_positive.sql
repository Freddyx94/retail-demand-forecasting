-- A price of zero or below would be a data error.
-- (NULL prices are already caught by the not_null test in _marts.yml.)

select *
from {{ ref('fct_daily_sales') }}
where sell_price <= 0