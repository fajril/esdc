-- Query for FIELD_RESOURCES table -o (resources only)
SELECT report_year, wk_name, field_name,
    project_class, uncert_level,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf'
FROM field_resources 
WHERE <where> LIKE '%<like>%' AND report_year = <year>
ORDER BY report_year, wk_name, field_name, project_class, uncert_level;

-- Query for field_resources table -oo (add in place)
SELECT report_year, wk_name, field_name,
    project_stage, project_class, project_level, uncert_level,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf',
    is_discovered, ioip, igip
FROM field_resources 
WHERE <where> LIKE '%<like>%' AND report_year = <year>
ORDER BY report_year, wk_name, field_name, project_stage, project_class, project_level, uncert_level;

-- Query for field_resources table -ooo (add cumprod)
SELECT report_year, wk_name, field_name, project_count,
    project_stage, project_class, uncert_level,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf',
    cprd_sls_oc as 'sales_cumprod_mstb',
    cprd_sls_an as 'sales_cumprod_bscf',
    is_discovered, ioip, igip
FROM field_resources 
WHERE <where> LIKE '%<like>%' AND report_year = <year>
ORDER BY report_year, wk_name, field_name, project_stage, project_class, uncert_level;

-- Query for field_resources table -oooo (all)
SELECT *
FROM field_resources 
WHERE <where> LIKE '%<like>%' AND report_year = <year>
ORDER BY report_year, wk_name, field_name, project_stage, project_class, project_level, uncert_level;