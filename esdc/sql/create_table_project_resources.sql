-- Drop the table if it already exists
DROP TABLE IF EXISTS project_resources;

-- Create the table if it does not exist
CREATE TABLE IF NOT EXISTS project_resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	report_date TEXT,
	report_year INTEGER,
	report_status TEXT,
	is_offshore INTEGER,
	wk_id TEXT,
	wk_name TEXT,
	wk_lat REAL,
	wk_long REAL,
	wk_area_ori REAL,
	wk_area_latest REAL,
	wk_subgroup TEXT,
	psc_start TEXT,
	psc_end TEXT,
	operator_group TEXT,
	operator_name TEXT,
	basin86_id TEXT,
	basin86 TEXT,
	basin128_id TEXT,
	basin128 TEXT,
	prov_id TEXT,
	province TEXT,
	field_id TEXT,
	field_name TEXT,
	field_lat REAL,
	field_long REAL,
	field_area REAL,
	is_unitisasi INTEGER,
	pod_letter_num TEXT,
	pod_name TEXT,
	project_id TEXT,
	project_name TEXT,
	is_discovered INTEGER,
	project_stage TEXT,
	project_eol INTEGER,
	project_isactive INTEGER,
	groovy_isactive INTEGER,
	fusion_isactive INTEGER,
	onstream_year INTEGER,
	onstream_actual TEXT,
	project_remarks TEXT,
	project_class TEXT,
	project_level TEXT,
	prod_stage TEXT,
	uncert_lvl TEXT,
	rec_oil REAL,
	rec_con REAL,
	rec_ga REAL,
	rec_gn REAL,
	rec_oc REAL,
	rec_an REAL,
	rec_oil_risked REAL,
	rec_con_risked REAL,
	rec_ga_risked REAL,
	rec_gn_risked REAL,
	rec_oc_risked REAL,
	rec_an_risked REAL,
	res_oil REAL,
	res_con REAL,
	res_ga REAL,
	res_gn REAL,
	res_oc REAL,
	res_an REAL,
	prj_ioip REAL,
	prj_igip REAL,
	vol_remarks TEXT,
	gcf_srock REAL,
	gcf_res REAL,
	gcf_ts REAL,
	gcf_dyn REAL,
	gcf_total REAL,
	pd REAL,
	eur_rec_oil REAL,
	eur_rec_con REAL,
	eur_rec_ga REAL,
	eur_rec_gn REAL,
	eur_rec_oc REAL,
	eur_rec_an REAL,
	eur_res_oil REAL,
	eur_res_con REAL,
	eur_res_ga REAL,
	eur_res_gn REAL,
	eur_res_oc REAL,
	eur_res_an REAL,
	rf_oil REAL,
	rf_gas REAL,
	urf_rec_oil REAL,
	urf_rec_gas REAL,
	urf_res_oil REAL,
	urf_res_gas REAL,
	rate_grs_oil REAL,
	rate_grs_con REAL,
	rate_grs_ga REAL,
	rate_grs_gn REAL,
	rate_grs_oc REAL,
	rate_grs_an REAL,
	rate_sls_oil REAL,
	rate_sls_con REAL,
	rate_sls_ga REAL,
	rate_sls_gn REAL,
	rate_sls_oc REAL,
	rate_sls_an REAL,
	cprd_grs_oil REAL,
	cprd_grs_con REAL,
	cprd_grs_ga REAL,
	cprd_grs_gn REAL,
	cprd_grs_oc REAL,
	cprd_grs_an REAL,
	cprd_sls_oil REAL,
	cprd_sls_con REAL,
	cprd_sls_ga REAL,
	cprd_sls_gn REAL,
	cprd_sls_oc REAL,
	cprd_sls_an REAL,
	dcpy_um_oil REAL,
	dcpy_um_con REAL,
	dcpy_um_ga REAL,
	dcpy_um_gn REAL,
	dcpy_um_oc REAL,
	dcpy_um_an REAL,
	dcpy_ppa_oil REAL,
	dcpy_ppa_con REAL,
	dcpy_ppa_ga REAL,
	dcpy_ppa_gn REAL,
	dcpy_ppa_oc REAL,
	dcpy_ppa_an REAL,
	dcpy_wi_oil REAL,
	dcpy_wi_con REAL,
	dcpy_wi_ga REAL,
	dcpy_wi_gn REAL,
	dcpy_wi_oc REAL,
	dcpy_wi_an REAL,
	dcpy_cio_oil REAL,
	dcpy_cio_con REAL,
	dcpy_cio_ga REAL,
	dcpy_cio_gn REAL,
	dcpy_cio_oc REAL,
	dcpy_cio_an REAL,
	dcpy_uc_oil REAL,
	dcpy_uc_con REAL,
	dcpy_uc_ga REAL,
	dcpy_uc_gn REAL,
	dcpy_uc_oc REAL,
	dcpy_uc_an REAL,
	ghv_avg REAL
);