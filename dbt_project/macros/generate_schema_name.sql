{#
    Use the custom schema name as-is (staging, marts, ml) instead of dbt's
    default "<target_schema>_<custom_schema>" (e.g. main_staging).
    The prefix exists for teams sharing one warehouse; this project has its
    own database file, so clean names are easier to query from Power BI.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
