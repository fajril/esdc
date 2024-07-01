-- Query for PROJECT_RESOURCES table
SELECT report_year, wk_name, field_name, project_name,
    project_stage, project_class, project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    prj_ioip as ioip, prj_igip as igip
FROM project_resources 
WHERE project_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, uncert_lvl;

-- Query for FIELD_RESOURCES table
SELECT wk_name, field_name, project_count,
    project_stage, project_class, project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    ioip, igip
FROM field_resources 
WHERE field_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, uncert_lvl;

-- Query for WA_RESOURCES table
SELECT wk_name, project_count,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    ioip, igip
FROM wa_resources 
WHERE wk_name LIKE '%{like}%' AND report_year = {year}
ORDER BY project_stage, project_class, project_level, uncert_lvl;

-- Query for other tables
SELECT
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    ioip, igip
FROM other_tables 
WHERE report_year = {year}
ORDER BY project_stage, project_class, project_level, uncert_lvl;


