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

select
    date,
    d,
    wm_yr_wk,
    weekday,
    wday,
    month,
    year,
    is_event,
    event_type_1 as event_type,
    is_snap
from {{ ref('stg_calendar') }}