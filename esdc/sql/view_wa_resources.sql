-- Query for wa_resources table -o (resources only)
SELECT report_year, wk_name, project_stage,
    project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an'
FROM wa_resources 
WHERE wk_name LIKE '%{like}%' AND report_year = {year}
ORDER BY report_year, wk_name, project_stage, project_class, uncert_lvl;

-- Query for wa_resources table -oo (add in place)
SELECT report_year, wk_name,
    project_stage, project_class, project_level, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    ioip, igip
FROM wa_resources 
WHERE wk_name LIKE '%{like}%' AND report_year = {year}
ORDER BY report_year, wk_name, project_stage, project_class, project_level, uncert_lvl;

-- Query for wa_resources table -ooo (add cumprod)
SELECT report_year, wk_name, project_count,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    cprd_sls_oc as 'sales_cumprod_oc',
    cprd_sls_an as 'sales_cumprod_an',
    ioip, igip
FROM wa_resources 
WHERE wk_name LIKE '%{like}%' AND report_year = {year}
ORDER BY report_year, wk_name, project_stage, project_class, uncert_lvl;

-- Query for wa_resources table -oooo (all)
SELECT *
FROM wa_resources 
WHERE wk_name LIKE '%{like}%' AND report_year = {year}
ORDER BY report_year, wk_name, project_stage, project_class, project_level, uncert_lvl;