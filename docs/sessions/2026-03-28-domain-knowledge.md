# Session: Domain Knowledge Module Created

**Date:** 2026-03-28
**Status:** Completed

## Summary

Created `esdc/chat/domain_knowledge.py` - a domain knowledge mapping module that maps Indonesian/English oil & gas terminology to database schema columns and SQL patterns.

Includes table/view hierarchy for query optimization:

```
project_resources (level 1, most detailed)
    ↓ GROUP BY field
field_resources (level 2, pre-aggregated)
    ↓ GROUP BY work area
wa_resources (level 3, pre-aggregated)
    ↓ GROUP BY national
nkri_resources (level 4, most aggregated)
```

## Files Modified

| File | Changes |
|------|---------|
| `esdc/chat/domain_knowledge.py` | Added table hierarchy, query building functions, 88 tests |
| `esdc/chat/prompts.py` | Added Table Selection Guide with examples |
| `tests/test_domain_knowledge.py` | 88 tests, all passing |

## New Functions Added

### Table/View Selection

- `get_table_for_query(entity_type, require_detail)` - Returns optimal table for query
- `can_use_view_for_calculation(uncertainty, table)` - Checks if view supports calculation
- `get_entity_filter_column(entity_type, table)` - Returns column name for filtering
- `build_aggregate_query()` - Builds optimized SQL for aggregate queries
- `get_aggregation_table_info()` - Returns metadata about all tables/views
- `get_recommended_table()` - Helper for LLM table selection

### Key Constants

```python
TABLE_HIERARCHY = {
    "project": "project_resources",
    "field": "field_resources",
    "work_area": "wa_resources",
    "wa": "wa_resources",
    "national": "nkri_resources",
}
```

## Query Optimization Rules

1. **Use views for aggregates**: `field_resources` for field totals, `wa_resources` for work area totals
2. **Use `project_resources` only for**: Project-specific details, project breakdowns
3. **Views support all calculations**: Probable/possible work on views because they have `uncert_level`
4. **National queries**: Use `nkri_resources` for country-wide statistics

## Testing

```bash
python -m pytest tests/test_domain_knowledge.py -v
# 88 passed in 0.85s
```