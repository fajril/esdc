-- Full-Text Search (FTS) Indexes for ESDC
-- Enables fast text search on common ILIKE query patterns
-- Uses DuckDB's native FTS extension with BM25 ranking

-- Install and load FTS extension
INSTALL fts;
LOAD fts;

-- ============================================================================
-- project_resources: base resources table
-- ============================================================================
PRAGMA create_fts_index(
    'project_resources',
    'uuid',
    'project_name',
    'field_name',
    'wk_name',
    'province',
    'basin128',
    'operator_name',
    'project_remarks',
    'vol_remarks',
    lower=1,
    strip_accents=1,
    overwrite=1
);

-- ============================================================================
-- project_timeseries: base timeseries table
-- ============================================================================
PRAGMA create_fts_index(
    'project_timeseries',
    'uuid',
    'project_name',
    'field_name',
    'wk_name',
    'province',
    'basin128',
    'operator_name',
    'project_remarks',
    lower=1,
    strip_accents=1,
    overwrite=1
);

-- ============================================================================
-- B-tree indexes for non-text queries
-- ============================================================================

-- project_resources: report_year for latest data queries
CREATE INDEX IF NOT EXISTS idx_project_resources_report_year
ON project_resources(report_year DESC);

-- project_resources: compound index for aggregation
CREATE INDEX IF NOT EXISTS idx_project_resources_agg
ON project_resources(report_year, wk_id, field_id, project_class, project_level);

-- project_timeseries: report_year for latest data queries
CREATE INDEX IF NOT EXISTS idx_project_timeseries_report_year
ON project_timeseries(report_year DESC);

-- project_timeseries: compound index for view aggregation
CREATE INDEX IF NOT EXISTS idx_project_timeseries_agg
ON project_timeseries(report_year, wk_id, field_id, project_class, project_level, year);