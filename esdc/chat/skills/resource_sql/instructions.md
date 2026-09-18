# Resource SQL playbook

## Sequence

1. Identify the requested grain (project, field, or WK), report year, and
   uncertainty level before writing SQL.
2. Use the bound simple-query or SQL capability and the column guide when the
   schema or metric is unclear.
3. Preserve the table grain and join keys. Check duplicate rows before
   aggregating; use explicit `GROUP BY` dimensions.
4. For year comparisons, select both years in one query and label each result.
5. State filters, units, NULL handling, and any unresolved entity.

## Evidence boundary

Current values come from SQLite-backed resource tables. GROOVY report is
nullable labelled content, while GROOVY active is status. Do not treat either
as a numeric reserve or resource measure.

## Forbidden shortcuts

- Do not silently choose a report year or uncertainty level when the question
  is ambiguous.
- Do not fabricate P90/P50/P10 values or convert units without a stated basis.
- Do not join project, field, and WK rows without preserving their grain.
