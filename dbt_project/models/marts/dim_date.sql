-- dim_date.sql
-- One row per calendar day (all 1,969 days, including the 28-day forecast
-- horizon after the last sales day).
--
-- event_type on days with two events: we keep the first listed event
-- (event_type_1). Only a handful of days have a second event, and a single
-- category per day keeps grouping and filtering simple (e.g. one slicer
-- value per day in Power BI). Combining both types into new labels would
-- create rare categories with almost no data. is_event still marks every
-- event day, and both events remain available in stg_calendar if needed.
-- event_name follows the same rule. The analysis (FINDINGS.md, finding 3)
-- showed that events of the same type push sales in opposite directions,
-- so the model needs the name, not just the type.
--
-- is_store_closed: the store is closed on 25 December every year
-- (sales = 0 for every item). A flag lets the model and the evaluation
-- treat these days separately instead of learning them as normal zeros.

select
    date,
    d,
    wm_yr_wk,
    weekday,
    wday,
    month,
    year,
    is_event,
    event_name_1 as event_name,
    event_type_1 as event_type,
    is_snap,
    (month = 12 and day(date) = 25) as is_store_closed
from {{ ref('stg_calendar') }}