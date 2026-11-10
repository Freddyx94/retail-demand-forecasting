-- stg_calendar.sql
-- One row per day with calendar attributes, events and the SNAP flag
-- for our store's state.
--
-- Keep all 1,969 days, including the 28 after the last sales day (d_1941).
-- Those future days are exactly the forecast horizon: to predict days
-- d_1942-d_1969 we need their weekday, events and SNAP flags as features.
-- These are known in advance, so using them is not leakage.

with source as (

    select * from {{ source('raw', 'calendar') }}

),

renamed as (

    select
        cast(date as date)  as date,
        d,
        wm_yr_wk,
        weekday,
        wday,
        month,
        year,
        event_name_1,
        event_type_1,
        event_name_2,
        event_type_2,

        -- TRUE when the day has at least one event
        (event_name_1 is not null or event_name_2 is not null) as is_event,

        -- SNAP column for our store's state: 'CA_1' -> 'CA' -> snap_CA
        cast(snap_{{ var("store_id").split("_")[0] }} as boolean) as is_snap

    from source

)

select * from renamed