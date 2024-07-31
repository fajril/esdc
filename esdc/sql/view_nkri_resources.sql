-- Query for nkri_resources table -o (resources only)
SELECT report_year, project_stage,
    project_class, uncert_lvl,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf'
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -oo (add in place)
SELECT report_year, wk_name,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf',
    ioip, igip
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -ooo (add cumprod)
SELECT report_year, project_count,
    project_stage, project_class, uncert_lvl,
    rec_oc_risked as 'resources_mstb', rec_an_risked as 'resources_bscf',
    res_oc as 'reserves_mstb', res_an as 'reserves_bscf',
    cprd_sls_oc as 'sales_cumprod_mstb',
    cprd_sls_an as 'sales_cumprod_bscf',
    ioip, igip
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;

-- Query for nkri_resources table -oooo (all)
SELECT *
FROM nkri_resources 
WHERE report_year = <year>
ORDER BY report_year, project_stage, project_class, uncert_lvl;