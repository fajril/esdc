-- Timeseries Views for ESDC
-- Aggregates project_timeseries data hierarchically: field → wa → nkri
-- Each view includes project_class and project_level in aggregation

-- Drop existing views
DROP VIEW IF EXISTS nkri_timeseries;
DROP VIEW IF EXISTS wa_timeseries;
DROP VIEW IF EXISTS field_timeseries;

-- ============================================================================
-- View 1: field_timeseries
-- Aggregate project-level timeseries to field level per forecast year
-- ============================================================================

CREATE VIEW field_timeseries AS
SELECT 
    MIN(report_date) as report_date,
    MIN(report_year) as report_year,
    MIN(report_status) as report_status,
    MAX(is_offshore) as is_offshore,
    MIN(wk_id) as wk_id,
    MIN(wk_name) as wk_name,
    MIN(wk_lat) as wk_lat,
    MIN(wk_long) as wk_long,
    MIN(wk_area) as wk_area,
    MIN(wk_subgroup) as wk_subgroup,
    MIN(psc_eff_start) as psc_eff_start,
    MIN(psc_eff_end) as psc_eff_end,
    MIN(operator_group) as operator_group,
    MIN(operator_name) as operator_name,
    MIN(basin86_id) as basin86_id,
    MIN(basin128) as basin128,
    MIN(province) as province,
    MIN(field_id) as field_id,
    MIN(field_name) as field_name,
    MIN(field_lat) as field_lat,
    MIN(field_long) as field_long,
    MIN(project_class) as project_class,
    MIN(project_level) as project_level,
    MIN(prod_stage) as prod_stage,
    MIN(onstream_year) as onstream_year,
    MIN(onstream_actual) as onstream_actual,
    year,
    COUNT(*) as project_count,
    SUM(project_isactive) as active_project_count,
    AVG(ghv) as ghv_avg,
    
    -- TPF (Total Potential Forecast) - 12 columns
    SUM(tpf_oil) as tpf_oil,
    SUM(tpf_con) as tpf_con,
    SUM(tpf_ga) as tpf_ga,
    SUM(tpf_gn) as tpf_gn,
    SUM(tpf_oc) as tpf_oc,
    SUM(tpf_an) as tpf_an,
    SUM(tpf_risked_oil) as tpf_risked_oil,
    SUM(tpf_risked_con) as tpf_risked_con,
    SUM(tpf_risked_ga) as tpf_risked_ga,
    SUM(tpf_risked_gn) as tpf_risked_gn,
    SUM(tpf_risked_oc) as tpf_risked_oc,
    SUM(tpf_risked_an) as tpf_risked_an,
    
    -- SLF (Sales Forecast) - 6 columns
    SUM(slf_oil) as slf_oil,
    SUM(slf_con) as slf_con,
    SUM(slf_ga) as slf_ga,
    SUM(slf_gn) as slf_gn,
    SUM(slf_oc) as slf_oc,
    SUM(slf_an) as slf_an,
    
    -- SPF (Sales Potential Forecast) - 6 columns
    SUM(spf_oil) as spf_oil,
    SUM(spf_con) as spf_con,
    SUM(spf_ga) as spf_ga,
    SUM(spf_gn) as spf_gn,
    SUM(spf_oc) as spf_oc,
    SUM(spf_an) as spf_an,
    
    -- CRF (Contingent Resources Forecast) - 6 columns
    SUM(crf_oil) as crf_oil,
    SUM(crf_con) as crf_con,
    SUM(crf_ga) as crf_ga,
    SUM(crf_gn) as crf_gn,
    SUM(crf_oc) as crf_oc,
    SUM(crf_an) as crf_an,
    
    -- PRF (Prospective Resources Forecast) - 6 columns
    SUM(prf_oil) as prf_oil,
    SUM(prf_con) as prf_con,
    SUM(prf_ga) as prf_ga,
    SUM(prf_gn) as prf_gn,
    SUM(prf_oc) as prf_oc,
    SUM(prf_an) as prf_an,
    
    -- CIOF (Consumed in Operation Forecast) - 6 columns
    SUM(ciof_oil) as ciof_oil,
    SUM(ciof_con) as ciof_con,
    SUM(ciof_ga) as ciof_ga,
    SUM(ciof_gn) as ciof_gn,
    SUM(ciof_oc) as ciof_oc,
    SUM(ciof_an) as ciof_an,
    
    -- LOSSF (Loss Production Forecast) - 6 columns
    SUM(lossf_oil) as lossf_oil,
    SUM(lossf_con) as lossf_con,
    SUM(lossf_ga) as lossf_ga,
    SUM(lossf_gn) as lossf_gn,
    SUM(lossf_oc) as lossf_oc,
    SUM(lossf_an) as lossf_an,
    
    -- Cumulative Production (cprd_*) - 12 columns
    SUM(cprd_grs_oil) as cprd_grs_oil,
    SUM(cprd_grs_con) as cprd_grs_con,
    SUM(cprd_grs_ga) as cprd_grs_ga,
    SUM(cprd_grs_gn) as cprd_grs_gn,
    SUM(cprd_grs_oc) as cprd_grs_oc,
    SUM(cprd_grs_an) as cprd_grs_an,
    SUM(cprd_sls_oil) as cprd_sls_oil,
    SUM(cprd_sls_con) as cprd_sls_con,
    SUM(cprd_sls_ga) as cprd_sls_ga,
    SUM(cprd_sls_gn) as cprd_sls_gn,
    SUM(cprd_sls_oc) as cprd_sls_oc,
    SUM(cprd_sls_an) as cprd_sls_an,
    
    -- Production Rate (rate_*) - 6 columns
    SUM(rate_oil) as rate_oil,
    SUM(rate_con) as rate_con,
    SUM(rate_ga) as rate_ga,
    SUM(rate_gn) as rate_gn,
    SUM(rate_oc) as rate_oc,
    SUM(rate_an) as rate_an,
    
    -- UUID generation
    (
        lower(hex(randomblob(4))) || '-' ||
        lower(hex(randomblob(2))) || '-' ||
        '4' || substr(hex(randomblob(2)), 2) || '-' ||
        substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
        lower(hex(randomblob(6)))
    ) as uuid
FROM project_timeseries
GROUP BY report_year, wk_id, field_id, project_class, project_level, year
ORDER BY report_year DESC, wk_name, field_name, year;

-- ============================================================================
-- View 2: wa_timeseries
-- Aggregate field-level timeseries to work area level per forecast year
-- ============================================================================

CREATE VIEW wa_timeseries AS
SELECT 
    MIN(report_date) as report_date,
    MIN(report_year) as report_year,
    MIN(report_status) as report_status,
    MAX(is_offshore) as is_offshore,
    MIN(wk_id) as wk_id,
    MIN(wk_name) as wk_name,
    MIN(wk_lat) as wk_lat,
    MIN(wk_long) as wk_long,
    MIN(wk_area) as wk_area,
    MIN(wk_subgroup) as wk_subgroup,
    MIN(psc_eff_start) as psc_eff_start,
    MIN(psc_eff_end) as psc_eff_end,
    MIN(operator_group) as operator_group,
    MIN(operator_name) as operator_name,
    MIN(project_class) as project_class,
    MIN(project_level) as project_level,
    MIN(prod_stage) as prod_stage,
    MIN(onstream_year) as onstream_year,
    MIN(onstream_actual) as onstream_actual,
    year,
    COUNT(*) as field_count,
    SUM(project_count) as project_count,
    SUM(active_project_count) as active_project_count,
    AVG(ghv_avg) as ghv_avg,
    
    -- TPF (Total Potential Forecast) - 12 columns
    SUM(tpf_oil) as tpf_oil,
    SUM(tpf_con) as tpf_con,
    SUM(tpf_ga) as tpf_ga,
    SUM(tpf_gn) as tpf_gn,
    SUM(tpf_oc) as tpf_oc,
    SUM(tpf_an) as tpf_an,
    SUM(tpf_risked_oil) as tpf_risked_oil,
    SUM(tpf_risked_con) as tpf_risked_con,
    SUM(tpf_risked_ga) as tpf_risked_ga,
    SUM(tpf_risked_gn) as tpf_risked_gn,
    SUM(tpf_risked_oc) as tpf_risked_oc,
    SUM(tpf_risked_an) as tpf_risked_an,
    
    -- SLF (Sales Forecast) - 6 columns
    SUM(slf_oil) as slf_oil,
    SUM(slf_con) as slf_con,
    SUM(slf_ga) as slf_ga,
    SUM(slf_gn) as slf_gn,
    SUM(slf_oc) as slf_oc,
    SUM(slf_an) as slf_an,
    
    -- SPF (Sales Potential Forecast) - 6 columns
    SUM(spf_oil) as spf_oil,
    SUM(spf_con) as spf_con,
    SUM(spf_ga) as spf_ga,
    SUM(spf_gn) as spf_gn,
    SUM(spf_oc) as spf_oc,
    SUM(spf_an) as spf_an,
    
    -- CRF (Contingent Resources Forecast) - 6 columns
    SUM(crf_oil) as crf_oil,
    SUM(crf_con) as crf_con,
    SUM(crf_ga) as crf_ga,
    SUM(crf_gn) as crf_gn,
    SUM(crf_oc) as crf_oc,
    SUM(crf_an) as crf_an,
    
    -- PRF (Prospective Resources Forecast) - 6 columns
    SUM(prf_oil) as prf_oil,
    SUM(prf_con) as prf_con,
    SUM(prf_ga) as prf_ga,
    SUM(prf_gn) as prf_gn,
    SUM(prf_oc) as prf_oc,
    SUM(prf_an) as prf_an,
    
    -- CIOF (Consumed in Operation Forecast) - 6 columns
    SUM(ciof_oil) as ciof_oil,
    SUM(ciof_con) as ciof_con,
    SUM(ciof_ga) as ciof_ga,
    SUM(ciof_gn) as ciof_gn,
    SUM(ciof_oc) as ciof_oc,
    SUM(ciof_an) as ciof_an,
    
    -- LOSSF (Loss Production Forecast) - 6 columns
    SUM(lossf_oil) as lossf_oil,
    SUM(lossf_con) as lossf_con,
    SUM(lossf_ga) as lossf_ga,
    SUM(lossf_gn) as lossf_gn,
    SUM(lossf_oc) as lossf_oc,
    SUM(lossf_an) as lossf_an,
    
    -- Cumulative Production (cprd_*) - 12 columns
    SUM(cprd_grs_oil) as cprd_grs_oil,
    SUM(cprd_grs_con) as cprd_grs_con,
    SUM(cprd_grs_ga) as cprd_grs_ga,
    SUM(cprd_grs_gn) as cprd_grs_gn,
    SUM(cprd_grs_oc) as cprd_grs_oc,
    SUM(cprd_grs_an) as cprd_grs_an,
    SUM(cprd_sls_oil) as cprd_sls_oil,
    SUM(cprd_sls_con) as cprd_sls_con,
    SUM(cprd_sls_ga) as cprd_sls_ga,
    SUM(cprd_sls_gn) as cprd_sls_gn,
    SUM(cprd_sls_oc) as cprd_sls_oc,
    SUM(cprd_sls_an) as cprd_sls_an,
    
    -- Production Rate (rate_*) - 6 columns
    SUM(rate_oil) as rate_oil,
    SUM(rate_con) as rate_con,
    SUM(rate_ga) as rate_ga,
    SUM(rate_gn) as rate_gn,
    SUM(rate_oc) as rate_oc,
    SUM(rate_an) as rate_an,
    
    -- UUID generation
    (
        lower(hex(randomblob(4))) || '-' ||
        lower(hex(randomblob(2))) || '-' ||
        '4' || substr(hex(randomblob(2)), 2) || '-' ||
        substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
        lower(hex(randomblob(6)))
    ) as uuid
FROM field_timeseries
GROUP BY report_year, wk_id, project_class, project_level, year
ORDER BY report_year DESC, wk_name, year;

-- ============================================================================
-- View 3: nkri_timeseries
-- Aggregate work area timeseries to national level per forecast year
-- ============================================================================

CREATE VIEW nkri_timeseries AS
SELECT 
    MIN(report_year) as report_year,
    MIN(report_status) as report_status,
    MIN(project_class) as project_class,
    MIN(project_level) as project_level,
    MIN(prod_stage) as prod_stage,
    year,
    COUNT(*) as wa_count,
    SUM(field_count) as field_count,
    SUM(project_count) as project_count,
    SUM(active_project_count) as active_project_count,
    AVG(ghv_avg) as ghv_avg,
    
    -- TPF (Total Potential Forecast) - 12 columns
    SUM(tpf_oil) as tpf_oil,
    SUM(tpf_con) as tpf_con,
    SUM(tpf_ga) as tpf_ga,
    SUM(tpf_gn) as tpf_gn,
    SUM(tpf_oc) as tpf_oc,
    SUM(tpf_an) as tpf_an,
    SUM(tpf_risked_oil) as tpf_risked_oil,
    SUM(tpf_risked_con) as tpf_risked_con,
    SUM(tpf_risked_ga) as tpf_risked_ga,
    SUM(tpf_risked_gn) as tpf_risked_gn,
    SUM(tpf_risked_oc) as tpf_risked_oc,
    SUM(tpf_risked_an) as tpf_risked_an,
    
    -- SLF (Sales Forecast) - 6 columns
    SUM(slf_oil) as slf_oil,
    SUM(slf_con) as slf_con,
    SUM(slf_ga) as slf_ga,
    SUM(slf_gn) as slf_gn,
    SUM(slf_oc) as slf_oc,
    SUM(slf_an) as slf_an,
    
    -- SPF (Sales Potential Forecast) - 6 columns
    SUM(spf_oil) as spf_oil,
    SUM(spf_con) as spf_con,
    SUM(spf_ga) as spf_ga,
    SUM(spf_gn) as spf_gn,
    SUM(spf_oc) as spf_oc,
    SUM(spf_an) as spf_an,
    
    -- CRF (Contingent Resources Forecast) - 6 columns
    SUM(crf_oil) as crf_oil,
    SUM(crf_con) as crf_con,
    SUM(crf_ga) as crf_ga,
    SUM(crf_gn) as crf_gn,
    SUM(crf_oc) as crf_oc,
    SUM(crf_an) as crf_an,
    
    -- PRF (Prospective Resources Forecast) - 6 columns
    SUM(prf_oil) as prf_oil,
    SUM(prf_con) as prf_con,
    SUM(prf_ga) as prf_ga,
    SUM(prf_gn) as prf_gn,
    SUM(prf_oc) as prf_oc,
    SUM(prf_an) as prf_an,
    
    -- CIOF (Consumed in Operation Forecast) - 6 columns
    SUM(ciof_oil) as ciof_oil,
    SUM(ciof_con) as ciof_con,
    SUM(ciof_ga) as ciof_ga,
    SUM(ciof_gn) as ciof_gn,
    SUM(ciof_oc) as ciof_oc,
    SUM(ciof_an) as ciof_an,
    
    -- LOSSF (Loss Production Forecast) - 6 columns
    SUM(lossf_oil) as lossf_oil,
    SUM(lossf_con) as lossf_con,
    SUM(lossf_ga) as lossf_ga,
    SUM(lossf_gn) as lossf_gn,
    SUM(lossf_oc) as lossf_oc,
    SUM(lossf_an) as lossf_an,
    
    -- Cumulative Production (cprd_*) - 12 columns
    SUM(cprd_grs_oil) as cprd_grs_oil,
    SUM(cprd_grs_con) as cprd_grs_con,
    SUM(cprd_grs_ga) as cprd_grs_ga,
    SUM(cprd_grs_gn) as cprd_grs_gn,
    SUM(cprd_grs_oc) as cprd_grs_oc,
    SUM(cprd_grs_an) as cprd_grs_an,
    SUM(cprd_sls_oil) as cprd_sls_oil,
    SUM(cprd_sls_con) as cprd_sls_con,
    SUM(cprd_sls_ga) as cprd_sls_ga,
    SUM(cprd_sls_gn) as cprd_sls_gn,
    SUM(cprd_sls_oc) as cprd_sls_oc,
    SUM(cprd_sls_an) as cprd_sls_an,
    
    -- Production Rate (rate_*) - 6 columns
    SUM(rate_oil) as rate_oil,
    SUM(rate_con) as rate_con,
    SUM(rate_ga) as rate_ga,
    SUM(rate_gn) as rate_gn,
    SUM(rate_oc) as rate_oc,
    SUM(rate_an) as rate_an,
    
    -- UUID generation
    (
        lower(hex(randomblob(4))) || '-' ||
        lower(hex(randomblob(2))) || '-' ||
        '4' || substr(hex(randomblob(2)), 2) || '-' ||
        substr('89ab', 1 + (abs(random()) % 4), 1) || substr(hex(randomblob(2)), 2) || '-' ||
        lower(hex(randomblob(6)))
    ) as uuid
FROM wa_timeseries
GROUP BY report_year, project_class, project_level, year
ORDER BY report_year DESC, year;

-- ============================================================================
-- Indexes for Performance Optimization
-- ============================================================================

-- Index for field_timeseries aggregation
CREATE INDEX IF NOT EXISTS idx_project_timeseries_field_agg 
ON project_timeseries(report_year, wk_id, field_id, project_class, project_level, year);

-- Index for wa_timeseries aggregation
CREATE INDEX IF NOT EXISTS idx_project_timeseries_wa_agg
ON project_timeseries(report_year, wk_id, project_class, project_level, year);

-- Index for common lookup queries
CREATE INDEX IF NOT EXISTS idx_project_timeseries_lookup 
ON project_timeseries(field_name, year, project_name);

-- Index for project-level queries
CREATE INDEX IF NOT EXISTS idx_project_timeseries_project 
ON project_timeseries(project_name, year);
