-- Query for nkri_resources table -o (resources only)
SELECT report_year, project_stage,
    project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an'
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -oo (add in place)
SELECT report_year, wk_name,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    ioip, igip
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -ooo (add cumprod)
SELECT report_year, project_count,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_oc', rec_an_risked as 'resources_an',
    res_oc as 'reserves_oc', res_an as 'reserves_an',
    cprd_sls_oc as 'sales_cumprod_oc',
    cprd_sls_an as 'sales_cumprod_an',
    ioip, igip
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -oooo (all)
SELECT *
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;