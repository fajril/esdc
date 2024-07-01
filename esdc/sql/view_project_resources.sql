-- Query for PROJECT_RESOURCES table -o
SELECT report_year, project_name,
    project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an'
FROM project_resources 
WHERE project_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_level, project_name, uncert_lvl;

-- Query for PROJECT_RESOURCES table -oo
SELECT report_year, wk_name, field_name, project_name,
    project_stage, project_class, project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    prj_ioip, prj_igip
FROM project_resources 
WHERE project_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, project_name, uncert_lvl;

-- Query for PROJECT_RESOURCES table -ooo
SELECT report_year, wk_name, field_name, project_name,
    project_stage, project_class, project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    cprd_sls_oc as 'sales_cumprod_oc',
    cprd_sls_an as 'sales_cumprod_an',
    prj_ioip, prj_igip
FROM project_resources 
WHERE project_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, project_name, uncert_lvl;

-- Query for PROJECT_RESOURCES table -oooo
SELECT *
FROM project_resources 
WHERE project_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, wk_name field_name, project_name, uncert_lvl;