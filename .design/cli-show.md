# CLI Show Interface Design

This document outlines the command-line interface for the `esdc show` command, which queries and displays ESDC (Electronic Submission of Data Center) resource data from the local SQLite database. This design documents the current implementation as of the refactor branch.

---

## Show Command Structure

```
esdc show TABLE [OPTIONS]
```

The `show` command displays data from a specific resource table/view with optional filtering, column selection, and export capabilities.

### Required Arguments

| Argument | Description |
| --- | --- |
| `TABLE` | Name of the table/view to query. Valid values: `project_resources`, `field_resources`, `wa_resources`, `nkri_resources` |

### Available Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--where COLUMN` | str | None | Column name to apply search filter on. If omitted, uses default column based on table type. |
| `--search TEXT` | str | "" | Text pattern to match using SQL LIKE syntax (e.g., "Gas" matches "%Gas%"). Used with `--where`. |
| `--year INTEGER` | int | None | Filter by report year (minimum 2019). When omitted, shows all available years. |
| `--output INTEGER` | int | 0 | Level of detail in output. Range: 0-4, where 0 = minimal, 4 = all columns. |
| `--save / --no-save` | bool | False | Whether to save output to an Excel file. File is named `view_{table}_{YYYYMMDD}.xlsx`. |
| `--columns "COL1 COL2..."` | str | "" | Space-separated list of columns to display. When specified, overrides `--output` level. |

---

## Data Hierarchy & Aggregation Levels

ESDC manages oil and gas resource data at four hierarchical levels:

1. **Project** (`project_resources`): Individual development projects (most granular)
2. **Field** (`field_resources`): Aggregated projects grouped by geological field
3. **Working Area** (`wa_resources`): Aggregated fields grouped by license/concession area
4. **National** (`nkri_resources`): Top-level national aggregation across all working areas

Each level provides progressively aggregated views of reserves, resources, in-place volumes, and production data.

---

## Output Detail Levels

The `--output` option controls how many columns are displayed, from minimal to comprehensive:

| Level | Description | Columns Shown |
| --- | --- | --- |
| `0` | Minimal - Resources and reserves only | report_year, project_name (or field_name/wk_name), uncert_level, resources_mstb, resources_bscf, reserves_mstb, reserves_bscf |
| `1` | Standard - Adds classification | Level 0 + wk_name, field_name, project_stage, project_class, project_level |
| `2` | Extended - Adds in-place volumes | Level 1 + prj_ioip, prj_igip (or ioip, igip for aggregated views) |
| `3` | Comprehensive - Adds production | Level 2 + cprd_sls_oc (sales_cumprod_mstb), cprd_sls_an (sales_cumprod_bscf) |
| `4` | Full - All columns | All columns including metadata, identifiers, and internal fields |

The specific columns at each level are defined in the SQL view files (`esdc/sql/view_*.sql`).

---

## Filtering Behavior

### Default WHERE Columns by Table

When `--where` is not specified, the command uses default columns for filtering:

| Table | Default WHERE Column |
| --- | --- |
| `project_resources` | `project_name` |
| `field_resources` | `field_name` |
| `wa_resources` | `wk_name` (working area name) |
| `nkri_resources` | N/A (no name-based filtering available) |

### Search Pattern Matching

The `--search` option uses SQL LIKE syntax with wildcards automatically added:

- Search term "Gas" becomes `WHERE column LIKE '%Gas%'`
- Pattern matching is case-sensitive (depends on SQLite collation)
- Empty search string (`--search ""`) matches all records

### Year Filtering

The `--year` option:
- Accepts integer values >= 2019
- Filters the `report_year` column
- When omitted, returns data for all available years
- Applied via SQL: `AND report_year = {year}`

### Filter Combination

When multiple filters are provided, they are combined with AND logic:

```sql
WHERE {where_column} LIKE '%{search}%' AND report_year = {year}
```

For `nkri_resources`, only year filtering is supported (no name-based WHERE clause).

---

## Column Selection

The `--columns` option allows custom column selection:

### Syntax

Space-separated column names as a single string:

```bash
esdc show project_resources --columns "project_name reserves_mstb reserves_bscf"
```

### Behavior

- When `--columns` is specified, it overrides the `--output` detail level
- The SELECT clause is rebuilt: `SELECT {col1}, {col2}, ... FROM ...`
- The WHERE and ORDER BY clauses remain unchanged
- Invalid column names will cause a SQLite error
- The FROM clause and table name are preserved from the view definition

### Implementation Note

The current implementation modifies the SQL query by:
1. Loading the base query from the view SQL file
2. Removing everything before the FROM keyword using regex: `.*?(?=FROM)`
3. Prepending `SELECT {columns}` to the modified query
4. Executing the reconstructed query

---

## Output Formats

### Console Table (Default)

When `--save` is not specified, data is printed to stdout in tabulated format:

- **Library**: Uses `tabulate` with `psql` table style
- **Formatting**:
  - Float values formatted with 2 decimal places and thousand separators (e.g., "1,234.56")
  - Right-aligned text for better number readability
  - Headers shown from DataFrame column names
- **No pagination**: Full result set printed (can be very large)

### Excel Export (--save)

When `--save` is specified, output is written to an Excel file:

- **Filename format**: `view_{table}_{YYYYMMDD}.xlsx`
  - Example: `view_project_resources_20241021.xlsx`
- **Location**: Current working directory
- **Worksheet name**: "resources report"
- **No index column**: DataFrame index is not included
- **Raw values**: No special formatting applied (Excel default number formats)

**Note**: There is currently no CSV or JSON export option.

---

## Usage Examples

### Basic Usage

Show all projects (default output level):

```bash
esdc show project_resources
```

Show field resources with standard detail:

```bash
esdc show field_resources --output 1
```

### Filtering by Name

Find projects with "Gas" in the name:

```bash
esdc show project_resources --search Gas
```

Find fields in a specific working area:

```bash
esdc show field_resources --where wk_name --search "Blok A"
```

### Year Filtering

Show 2024 data only:

```bash
esdc show project_resources --year 2024
```

Combine name and year filters:

```bash
esdc show project_resources --search Mahakam --year 2024 --output 2
```

### Column Selection

Show only project names and reserves:

```bash
esdc show project_resources --columns "project_name reserves_mstb reserves_bscf"
```

Custom view of field data:

```bash
esdc show field_resources --columns "field_name wk_name resources_mstb resources_bscf ioip igip" --year 2024
```

### Saving to Excel

Export filtered results:

```bash
esdc show project_resources --search "Gas" --year 2024 --output 3 --save
```

Export national summary:

```bash
esdc show nkri_resources --year 2024 --output 2 --save
```

---

## Table-Specific Behavior

### project_resources

**Default WHERE column**: `project_name`

**Sorting**: Results are ordered by:
- `report_year`
- `project_stage`
- `project_class`
- `project_level`
- `project_name`
- `uncert_level`

**Key columns**:
- `project_name`, `field_name`, `wk_name` (hierarchy)
- `project_stage`, `project_class`, `project_level`, `uncert_level` (classification)
- `rec_oc_risked` (resources_mstb), `rec_an_risked` (resources_bscf)
- `res_oc` (reserves_mstb), `res_an` (reserves_bscf)
- `prj_ioip`, `prj_igip` (in-place volumes)
- `cprd_sls_oc`, `cprd_sls_an` (cumulative production)

### field_resources

**Default WHERE column**: `field_name`

**Sorting**: Results are ordered by:
- `report_year`
- `wk_name`
- `field_name`
- `project_stage`
- `project_class`
- `project_level`
- `uncert_level`

**Key columns**:
- `field_name`, `wk_name` (hierarchy)
- `project_count` (number of constituent projects, level 3+)
- `is_discovered` (discovery status)
- Aggregated reserves, resources, in-place volumes, and production

### wa_resources

**Default WHERE column**: `wk_name`

**Sorting**: Results are ordered by:
- `report_year`
- `wk_name`
- `project_stage`
- `project_class`
- `project_level`
- `uncert_level`

**Key columns**:
- `wk_name` (working area name)
- `project_count` (total projects in WA, level 3+)
- Aggregated reserves, resources, in-place volumes, and production

### nkri_resources

**Default WHERE column**: N/A (national level has no name filtering)

**Filtering**: Only `--year` filter is supported. The `--where` and `--search` options are not applicable.

**Sorting**: Results are ordered by:
- `report_year`
- `project_stage`
- `project_class`
- `uncert_level`

**Key columns**:
- `project_count` (total national projects)
- `is_discovered` (aggregated discovery status)
- National totals for reserves, resources, in-place volumes, and production

---

## Error Handling

### Database Not Found

If the SQLite database doesn't exist (`esdc.db` not found in the configured data directory):

```
ERROR: Database file does not exist. Try to run this command first:

    esdc fetch --save
```

**Exit behavior**: Function returns `None`, warning is logged.

### Invalid Table Name

If an invalid table name is provided:

```
ValueError: 'invalid_table' is not a valid TableName
```

**Exit behavior**: Python exception raised by `TableName(table)` enum conversion.

### Query Execution Error

If the SQL query fails (e.g., invalid column name in `--columns`):

```
ERROR: Cannot query data. Message: {sqlite error message}
```

**Exit behavior**: Function returns `None`, error is logged.

### No Results

If the query returns an empty DataFrame:

```
WARNING: Unable to show data. The query is none.
```

**Note**: This warning only appears if `run_query()` returns `None`, not for empty result sets.

---

## Database Views & SQL Scripts

The `show` command relies on pre-defined SQL views stored in `esdc/sql/`:

| Table | SQL File | Purpose |
| --- | --- | --- |
| `project_resources` | `view_project_resources.sql` | Defines 4 queries (levels 0-3) for project data |
| `field_resources` | `view_field_resources.sql` | Defines 4 queries (levels 0-3) for field aggregations |
| `wa_resources` | `view_wa_resources.sql` | Defines 4 queries (levels 0-3) for WA aggregations |
| `nkri_resources` | `view_nkri_resources.sql` | Defines 4 queries (levels 0-3) for national aggregations |

### Query Selection Logic

The `run_query()` function in `dbmanager.py`:

1. Loads the SQL script file based on table name
2. Splits the script by semicolons to get individual queries
3. Selects the query at index `max(0, output - 1)`:
   - `output=0` → query index 0 (first query)
   - `output=1` → query index 0 (first query)
   - `output=2` → query index 1 (second query)
   - `output=3` → query index 2 (third query)
   - `output=4` → query index 3 (fourth query, usually `SELECT *`)
4. Replaces placeholders (`<where>`, `<like>`, `<year>`) with actual values
5. Executes the query against the SQLite database

### Placeholder System

SQL queries use placeholders for dynamic filtering:

- `<where>`: Replaced with the column name from `--where` option
- `<like>`: Replaced with the search value from `--search` option
- `<year>`: Replaced with the year value from `--year` option

Special handling:
- If `--year` is not provided, the clause `AND report_year = <year>` is removed entirely
- For `nkri_resources`, if `--year` is not provided, `WHERE report_year = <year>` is removed

---

## Column Reference

### Common Resource Columns

| Column | Type | Description | Alias/Note |
| --- | --- | --- | --- |
| `report_year` | INTEGER | Reporting year (e.g., 2024) | |
| `project_stage` | TEXT | Development stage | discovery, appraisal, development, production |
| `project_class` | TEXT | Resource class | contingent, prospective |
| `project_level` | TEXT | Classification level | reservoir, field, prospect |
| `uncert_level` | TEXT | Uncertainty level | P90 (low), P50 (best), P10 (high), U (unrisked) |
| `rec_oc_risked` | REAL | Oil resources (MSTB) | Aliased as `resources_mstb` in some views |
| `rec_an_risked` | REAL | Gas resources (BSCF) | Aliased as `resources_bscf` in some views |
| `res_oc` | REAL | Oil reserves (MSTB) | Aliased as `reserves_mstb` in some views |
| `res_an` | REAL | Gas reserves (BSCF) | Aliased as `reserves_bscf` in some views |

### Project-Specific Columns

| Column | Type | Description |
| --- | --- | --- |
| `project_name` | TEXT | Project identifier |
| `field_name` | TEXT | Parent field name |
| `wk_name` | TEXT | Working area/license name |
| `prj_ioip` | REAL | Project-specific initial oil in place (MMSTB) |
| `prj_igip` | REAL | Project-specific initial gas in place (BSCF) |
| `cprd_sls_oc` | REAL | Cumulative oil sales production (MSTB) |
| `cprd_sls_an` | REAL | Cumulative gas sales production (BSCF) |

### Aggregated View Columns

| Column | Type | Description | Available In |
| --- | --- | --- | --- |
| `project_count` | INTEGER | Number of constituent projects | field, wa, nkri views (level 3+) |
| `is_discovered` | BOOLEAN | Discovery status | field, nkri views |
| `ioip` | REAL | Aggregated initial oil in place | field, wa, nkri views |
| `igip` | REAL | Aggregated initial gas in place | field, wa, nkri views |

**Note**: At output level 3, cumulative production columns may be aliased:
- `cprd_sls_oc` → `sales_cumprod_mstb`
- `cprd_sls_an` → `sales_cumprod_bscf`

---

## Implementation Details

### Code Flow

1. **Command entry**: `esdc.py::show()` function (lines 226-294)
2. **Parameter processing**:
   - Convert table string to `TableName` enum
   - Split `--columns` string by spaces if provided
3. **Query execution**: Call `dbmanager.run_query()` with parameters
4. **Output rendering**:
   - If result is not None:
     - Format floats with 2 decimals and thousand separators
     - Print to console using `tabulate` with psql style
     - If `--save`: Export to Excel with filename `view_{table}_{date}.xlsx`
   - If result is None: Log warning

### Database Connection

- **Location**: Platform-specific user data directory (via `platformdirs`)
  - Windows: `C:\Users\{user}\AppData\Local\esdc\esdc\esdc.db`
  - macOS: `~/Library/Application Support/esdc/esdc.db`
  - Linux: `~/.local/share/esdc/esdc.db`
- **Connection**: Managed via context manager in `dbmanager.run_query()`
- **Query execution**: Uses `pandas.read_sql_query()` for DataFrame results

### Dependencies

- **typer**: CLI argument parsing and type annotations
- **pandas**: DataFrame manipulation and Excel export
- **tabulate**: Console table formatting
- **rich**: Enhanced console output (used for `rich.print`)
- **sqlite3**: Database operations (via pandas)
- **openpyxl**: Excel file format support (pandas dependency)

---

## Related Commands

The `show` command is part of the ESDC data workflow:

1. **Initialize/Fetch**: `esdc fetch --save` - Download data from ESDC API
2. **Reload**: `esdc reload --filetype csv` - Load data from local files into database
3. **Show**: `esdc show {table}` - Query and display data (this command)
4. **Validate**: `esdc validate` - Run validation rules on project_resources

---

## Typical Workflows

### Quick Data Lookup

```bash
# Check a specific project's reserves
esdc show project_resources --search "Banyu Urip" --year 2024

# View all fields in a working area
esdc show field_resources --where wk_name --search "Cepu" --year 2024 --output 1
```

### Report Generation

```bash
# Export national summary for executive report
esdc show nkri_resources --year 2024 --output 2 --save

# Generate field-level report with custom columns
esdc show field_resources --year 2024 --columns "field_name wk_name reserves_mstb reserves_bscf ioip" --save
```

### Data Analysis Preparation

```bash
# Get full dataset for analysis
esdc show project_resources --output 4 --save

# Extract production data for specific fields
esdc show field_resources --output 3 --year 2024 --save
```

---

## Limitations & Considerations

### Current Limitations

1. **No pagination**: Large result sets print entirely to console (can be overwhelming)
2. **No CSV export**: Only Excel export available via `--save`
3. **Limited filtering**: Only simple LIKE matching, no complex SQL expressions
4. **Single table per query**: Cannot join across tables or views
5. **No sorting control**: Sort order is fixed by SQL view definition
6. **No column exclusion**: Can only specify which columns to include, not exclude
7. **No result caching**: Every query hits the database

### Performance Considerations

- **Database size**: SQLite handles datasets up to several hundred thousand rows efficiently
- **Query performance**: Indexes exist on commonly filtered columns (report_year, project_name, field_name, wk_name)
- **Memory usage**: Full result set loaded into pandas DataFrame (can be large for `--output 4`)

### Data Freshness

- Data is only as current as the last `esdc fetch --save` or `esdc reload`
- The `show` command does not fetch new data from the ESDC API
- Run `esdc fetch --save` periodically to update local database

---

## Summary

The `esdc show` command provides a straightforward interface for querying ESDC resource data:

- **Single command** with table name argument (no subcommands)
- **Four tables/views** representing different aggregation levels
- **Flexible filtering** via WHERE column, search pattern, and year
- **Progressive detail levels** (0-4) for controlling column visibility
- **Custom column selection** for ad-hoc queries
- **Dual output modes**: console table or Excel export
- **Integration** with SQLite views for consistent query structure

This design prioritizes simplicity and directness, making common data retrieval tasks quick and memorable while supporting more complex queries through column selection and filtering options.
