# ESDC Database Schema Summary

This document contains the schema description for the ESDC database. Please update with your domain knowledge.

The definition is based on Kerangka Sumber Daya Migas Indonesia (KSMI) which is adapted from Petroleum Resource Management System (PRMS).

---

## Tables and Views

| Name | Description |
|------|-------------|
| `project_resources` | Project-level reserve/resource data |
| `project_timeseries` | Production time series data |
| `field_resources` | View - aggregated by field |
| `wa_resources` | View - aggregated by work area |
| `nkri_resources` | View - national level aggregation |

---

## Table: `project_resources`

### Identification Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `uuid` | TEXT | Unique identifier (UUID v4) |
| `project_id` | TEXT | Project identifier from ESDC |
| `project_name` | TEXT | Name of the project |
| `project_name_previous` | TEXT | Previous project name (if renamed) |
| `field_id` | TEXT | Field identifier |
| `field_name` | TEXT | Name of the field |
| `field_name_previous` | TEXT | Previous field name |
| `wk_id` | TEXT | Work area identifier |
| `wk_name` | TEXT | Work area name |
| `operator_name` | TEXT | Operator company name |
| `operator_group` | TEXT | Operator group/parent company |

### Location Columns

| Column | Type | Description |
|--------|------|-------------|
| `province` | TEXT | Province name |
| `prov_id` | TEXT | Province identifier |
| `basin86` | TEXT | Basin name (86 classification) |
| `basin86_id` | TEXT | Basin identifier (86) |
| `basin128` | TEXT | Basin name (128 classification) |
| `basin128_id` | TEXT | Basin identifier (128) |
| `wk_lat` | REAL | Work area latitude |
| `wk_long` | REAL | Work area longitude |
| `wk_area` | REAL | Work area size (sq km) |
| `wk_area_ori` | REAL | Original work area size (sq km) |
| `wk_area_latest` | REAL | Latest work area size (sq km) |
| `field_lat` | REAL | Field latitude |
| `field_long` | REAL | Field longitude |
| `is_offshore` | INTEGER | Whether location is offshore (0/1) |

### Contract/Regulatory Columns

| Column | Type | Description |
|--------|------|-------------|
| `psc_eff_start` | TEXT | PSC effective start date |
| `psc_eff_end` | TEXT | PSC effective end date |
| `wk_subgroup` | TEXT | Work area subgroup |
| `wk_area_perwakilan_skkmigas` | TEXT | SKKMigas representative area |
| `wk_regionisasi_ngi` | TEXT | NGI regionalization |
| `pod_letter_num` | TEXT | POD letter number |
| `pod_name` | TEXT | POD name |
| `pod_date` | TEXT | POD date |

### Project Classification Columns

GRR: Government of Indonesia Recoverable Resources. The volume that which categorizes as Reserves but does not limited to commercial limits. The purpose of this volume is to let the government know the total recoverable resources if the commercial limits that limiting the reserves are lifted.

low equivalent with P90 uncertainty level under inverted cumulative distribution.
mid equivalent with P50 uncertainty level under inverted cumulative distribution.
high equivalent with P10 uncertainty level under inverted cumulative distribution.

1P means a volume that which classifies as Reserves with low uncertainty level. In order to get it, look at `res_` within the class `reserves`.
2P means a volume that which classifies as Reserves with mid uncertainty level. In order to get it, look at `res_` within the class `reserves`.
3P means a volume that which classifies as Reserves with high uncertainty level. In order to get it, look at `res_` within the class `reserves`.

1R means a volume that which classifies as GRR with low uncertainty level. In order to get it, look at `rec_` within the class `reserves & GRR`.
2R means a volume that which classifies as GRR with mid uncertainty level. In order to get it, look at `rec_` within the class `reserves & GRR`.
3R means a volume that which classifies as GRR with high uncertainty level. In order to get it, look at `rec_` within the class `reserves & GRR`.

1C means a volume that which classifies as Contingent Resources with low uncertainty level. In order to get it, look at `rec_` within the class `Contingent Resources`.
2C means a volume that which classifies as Contingent Resources with mid uncertainty level. In order to get it, look at `rec_` within the class `Contingent Resources`.
3C means a volume that which classifies as Contingent Resources with high uncertainty level. In order to get it, look at `rec_` within the class `Contingent Resources`.

1U means a volume that which classifies as Prospective Resources with low uncertainty level. In order to get it, look at `rec_` within the class `Prospective Resources`.
2U means a volume that which classifies as Prospective Resources with mid uncertainty level. In order to get it, look at `rec_` within the class `Prospective Resources`.
3U means a volume that which classifies as Prospective Resources with high uncertainty level. In order to get it, look at `rec_` within the class `Prospective Resources`.

| Column | Type | Description |
|--------|------|-------------|
| `project_stage` | TEXT | Project stage (Exploration, Exploitation) |
| `project_class` | TEXT | Project classification (Reserves (P) & GRR (R) , Contingent Resources (C), Prospective Resources (U) ) |
| `project_level` | TEXT | Project maturity level (E0, E1, E2, E3 for Reserves & GRR, E4 to E8 and X0 to X3 for Contingent Resources, X4 to X6 for Prospective Resources, A1 and A2 for Abandoned) |
| `project_level_previous` | TEXT | Previous project maturity level |
| `uncert_level` | TEXT | Uncertainty level (Low, Mid, High) |
| `prod_stage` | TEXT | Production stage (Primary, Secondary, Tertiary - EOR or EGR) |
| `project_eol` | INTEGER | End of life (year) |
| `project_isactive` | INTEGER | Whether project is active (0/1) |
| `is_discovered` | INTEGER | Whether discovered (0/1). Undiscovered projects means it is classified as Prospective Resources. Reserves & GRR, Contingent Resources, Prospective Resources. Abandoned is not part of the official classification but is included for recording purposes |
| `is_ltp` | INTEGER | Long Term Plan flag |
| `is_pse_approved` | INTEGER | Penentuan Status Eksplorasi (PSE) approval status |
| `is_pod_approved` | INTEGER | Plan of Development (POD) approval status |
| `is_report_accepted` | INTEGER | Report acceptance status in esdc system |
| `is_unitization` | INTEGER | Unitization flag |
| `onstream_year` | INTEGER | First production year |
| `onstream_actual` | TEXT | Actual onstream date |

### Report Metadata Columns

| Column | Type | Description |
|--------|------|-------------|
| `report_date` | TEXT | Report submission date in esdc system |
| `report_year` | INTEGER | Report year |
| `report_status` | TEXT | Report status |
| `project_remarks` | TEXT | Project remarks/notes. Look here for insights |
| `vol_remarks` | TEXT | Volume remarks |

### Resource Columns (Recoverable) - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `rec_oil` | REAL | Resources oil | MSTB - Thousand Stock Tank Barrels |
| `rec_con` | REAL | Resources condensate | MSTB - Thousand Stock Tank Barrels |
| `rec_ga` | REAL | Resources associated gas | BSCF - Billion Standard Cubic Feet |
| `rec_gn` | REAL | Resources non-associated gas | BSCF - Billion Standard Cubic Feet |
| `rec_oc` | REAL | Resources oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `rec_an` | REAL | Resources total gas (GA + GN) | BSCF - Billion Standard Cubic Feet |
| `rec_mboe` | REAL | Resources (MBOE) | MBOE - Thousand Barrels Oil Equivalent |

### Resource Columns (Risked) - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `rec_oil_risked` | REAL | Risked Resources oil | MSTB - Thousand Stock Tank Barrels |
| `rec_con_risked` | REAL | Risked Resources condensate | MSTB - Thousand Stock Tank Barrels |
| `rec_ga_risked` | REAL | Risked Resources associated gas | BSCF - Billion Standard Cubic Feet |
| `rec_gn_risked` | REAL | Risked Resources non-associated gas | BSCF - Billion Standard Cubic Feet |
| `rec_oc_risked` | REAL | Risked Resources oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `rec_an_risked` | REAL | Risked Resources gas | BSCF - Billion Standard Cubic Feet |
| `rec_mboe_risked` | REAL | Risked Resources (MBOE) | MBOE - Thousand Barrels Oil Equivalent |

### Reserve Columns - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `res_oil` | REAL | Reserves oil | MSTB - Thousand Stock Tank Barrels |
| `res_con` | REAL | Reserves condensate | MSTB - Thousand Stock Tank Barrels |
| `res_ga` | REAL | Reserves associated gas | BSCF - Billion Standard Cubic Feet |
| `res_gn` | REAL | Reserves non-associated gas | BSCF - Billion Standard Cubic Feet |
| `res_oc` | REAL | Reserves oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `res_an` | REAL | Reserves associated + non-associated gas | BSCF - Billion Standard Cubic Feet |

### In-Place Volume Columns - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `prj_ioip` | REAL | Initial Oil In Place | MSTB - Thousand Stock Tank Barrels |
| `prj_igip` | REAL | Initial Gas In Place | BSCF - Billion Standard Cubic Feet |

### EUR Columns (Estimated Ultimate Recovery) - UNITS NEEDED

EUR: Estimated Ultimate Recovery

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `eur_rec_oil` | REAL | EUR Resources oil | MSTB - Thousand Stock Tank Barrels |
| `eur_rec_con` | REAL | EUR Resources condensate | MSTB - Thousand Stock Tank Barrels |
| `eur_rec_ga` | REAL | EUR Resources associated gas | BSCF - Billion Standard Cubic Feet |
| `eur_rec_gn` | REAL | EUR Resources non-associated gas | BSCF - Billion Standard Cubic Feet |
| `eur_rec_oc` | REAL | EUR Resources oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `eur_rec_an` | REAL | EUR Resources associated + non-associated gas | BSCF - Billion Standard Cubic Feet |
| `eur_res_oil` | REAL | EUR reserves oil | MSTB - Thousand Stock Tank Barrels |
| `eur_res_con` | REAL | EUR reserves condensate | MSTB - Thousand Stock Tank Barrels |
| `eur_res_ga` | REAL | EUR reserves associated gas | BSCF - Billion Standard Cubic Feet |
| `eur_res_gn` | REAL | EUR reserves non-associated gas | BSCF - Billion Standard Cubic Feet |
| `eur_res_oc` | REAL | EUR reserves oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `eur_res_an` | REAL | EUR reserves associated + non-associated gas | BSCF - Billion Standard Cubic Feet |

### Recovery Factor Columns - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `rf_oil` | REAL | Recovery factor oil | fraction |
| `rf_gas` | REAL | Recovery factor gas | fraction |
| `urf_rec_oil` | REAL | Ultimate recovery factor resources oil | fraction |
| `urf_rec_gas` | REAL | Ultimate recovery factor resources gas | fraction |
| `urf_res_oil` | REAL | Ultimate recovery factor reserves oil | fraction |
| `urf_res_gas` | REAL | Ultimate recovery factor reserves gas | fraction |

### Production Rate Columns - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `rate_grs_oil` | REAL | Gross production rate oil | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_grs_con` | REAL | Gross production rate condensate | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_grs_ga` | REAL | Gross production rate associated gas | BSCFY - Thousand Stock Tank Barrels per year |
| `rate_grs_gn` | REAL | Gross production rate non-associated gas | BSCFY - Thousand Stock Tank Barrels per year |
| `rate_grs_oc` | REAL | Gross production rate oil + condensate | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_grs_an` | REAL | Gross production rate associated + non-associated gas | BSCFY - Thousand Stock Tank Barrels per year |
| `rate_sls_oil` | REAL | Sales rate oil | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_sls_con` | REAL | Sales rate condensate | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_sls_ga` | REAL | Sales rate associated gas | BSCFY - Thousand Stock Tank Barrels per year |
| `rate_sls_gn` | REAL | Sales rate non-associated gas | BSCFY - Thousand Stock Tank Barrels per year |
| `rate_sls_oc` | REAL | Sales rate oil + condensate | MSTBY - Thousand Stock Tank Barrels per year |
| `rate_sls_an` | REAL | Sales rate associated + non-associated gas | BSCFY - Thousand Stock Tank Barrels per year |

### Cumulative Production Columns - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `cprd_grs_oil` | REAL | Cumulative gross production oil | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_con` | REAL | Cumulative gross production condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_ga` | REAL | Cumulative gross production associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_grs_gn` | REAL | Cumulative gross production non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_grs_oc` | REAL | Cumulative gross production oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_an` | REAL | Cumulative gross production associated + non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_oil` | REAL | Cumulative sales oil | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_con` | REAL | Cumulative sales condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_ga` | REAL | Cumulative sales associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_gn` | REAL | Cumulative sales non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_oc` | REAL | Cumulative sales oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_an` | REAL | Cumulative sales associated + non-associated gas | BSCF - Billion Standard Cubic Feet |

### Development Plan Columns (dcpy) - WHAT DO THESE MEAN?

| Column | Type | Description |
|--------|------|-------------|
| `dcpy_um_oil` | REAL | Discrepancy: Change of EUR oil because of update model |
| `dcpy_um_con` | REAL | Discrepancy: Change of EUR condensate because of update model |
| `dcpy_um_ga` | REAL | Discrepancy: Change of EUR associated gas because of update model |
| `dcpy_um_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of update model |
| `dcpy_um_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of update model |
| `dcpy_um_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of update model |
| `dcpy_ppa_oil` | REAL | Discrepancy: Change of EUR oil because of Production Performance Analysis |
| `dcpy_ppa_con` | REAL | Discrepancy: Change of EUR condensate because of Production Performance Analysis |
| `dcpy_ppa_ga` | REAL | Discrepancy: Change of EUR associated gas because of Production Performance Analysis |
| `dcpy_ppa_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of Production Performance Analysis |
| `dcpy_ppa_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of Production Performance Analysis |
| `dcpy_ppa_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of Production Performance Analysis |
| `dcpy_wi_oil` | REAL | Discrepancy: Change of EUR oil because of Well Intervention |
| `dcpy_wi_con` | REAL | Discrepancy: Change of EUR condensate because of Well Intervention |
| `dcpy_wi_ga` | REAL | Discrepancy: Change of EUR associated gas because of Well Intervention |
| `dcpy_wi_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of Well Intervention |
| `dcpy_wi_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of Well Intervention |
| `dcpy_wi_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of Well Intervention |
| `dcpy_cio_oil` | REAL | Discrepancy: Change of EUR oil because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_cio_con` | REAL | Discrepancy: Change of EUR condensate because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_cio_ga` | REAL | Discrepancy: Change of EUR associated gas because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_cio_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_cio_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_cio_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of Consumed in Operations (Fuel, Flare, Shrinkage) |
| `dcpy_gtr_oil` | REAL | Discrepancy: Change of EUR oil because of Commerciality, GRR to Reserves |
| `dcpy_gtr_con` | REAL | Discrepancy: Change of EUR condensate because of Commerciality, GRR to Reserves |
| `dcpy_gtr_ga` | REAL | Discrepancy: Change of EUR associated gas because of Commerciality, GRR to Reserves |
| `dcpy_gtr_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of Commerciality, GRR to Reserves |
| `dcpy_gtr_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of Commerciality, GRR to Reserves |
| `dcpy_gtr_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of Commerciality, GRR to Reserves |
| `dcpy_uc_oil` | REAL | Discrepancy: Change of EUR oil because of Unaccounted Changes |
| `dcpy_uc_con` | REAL | Discrepancy: Change of EUR condensate because of Unaccounted Changes |
| `dcpy_uc_ga` | REAL | Discrepancy: Change of EUR associated gas because of Unaccounted Changes |
| `dcpy_uc_gn` | REAL | Discrepancy: Change of EUR non-associated gas because of Unaccounted Changes |
| `dcpy_uc_oc` | REAL | Discrepancy: Change of EUR oil + condensate because of Unaccounted Changes |
| `dcpy_uc_an` | REAL | Discrepancy: Change of EUR associated + non-associated gas because of Unaccounted Changes |

### Other Columns

| Column | Type | Description |
|--------|------|-------------|
| `pd` | REAL | Chance of Development (fraction) |
| `ghv_avg` | REAL | Average gross heating value (MBTU/SCF) |
| `gcf_srock` | REAL | Geological Chance Factor of Source Rock |
| `gcf_res` | REAL | Geological Chance Factor of Reservoir |
| `gcf_ts` | REAL | Geological Chance Factor of Trap and Seal |
| `gcf_dyn` | REAL | Geological Chance Factor of Dynamic or Migration |
| `gcf_total` | REAL | The multiplicative of gcf_srock, gcf_res, gcf_ts, gcf_dyn |
| `groovy_isactive` | INTEGER | Long Term Exploration and Development Strategy (GROOVY) report is active flag |
| `fusion_isactive` | INTEGER | Field Maturity Strategic Evaluation (FUSION) report is active flag |

---

## Table: `project_timeseries`

### Identification Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `uuid` | TEXT | Unique identifier |
| `report_date` | TEXT | Report date |
| `report_year` | INTEGER | Report year |
| `report_status` | TEXT | Report status |
| `is_offshore` | INTEGER | Offshore flag (0/1) |
| `basin86_id` | TEXT | Basin ID |
| `basin128` | TEXT | Basin name (128) |
| `province` | TEXT | Province name |
| `operator_group` | TEXT | Operator group |
| `wk_id` | TEXT | Working area ID |
| `wk_name` | TEXT | Working area name |
| `wk_area` | REAL | Working area size |
| `wk_regionisasi_ngi` | TEXT | Neraca Gas Indonesia (NGI) regionalization |
| `wk_area_perwakilan_skkmigas` | TEXT | SKK Migas representative area |
| `psc_eff_start` | TEXT | PSC start date |
| `psc_eff_end` | TEXT | PSC end date |
| `wk_lat` | TEXT | Work area latitude |
| `wk_long` | TEXT | Work area longitude |
| `field_lat` | TEXT | Field latitude |
| `field_long` | TEXT | Field longitude |
| `wk_subgroup` | TEXT | Work area subgroup |
| `operator_name` | TEXT | Operator name |
| `field_id` | TEXT | Field ID |
| `field_name` | TEXT | Field name |
| `project_id` | TEXT | Project ID |
| `project_name` | TEXT | Project name |
| `project_isactive` | INTEGER | Active flag |
| `project_remarks` | TEXT | Project remarks. Look here for insights. |
| `vol_remarks` | TEXT | Volume remarks |
| `frcast_remarks` | TEXT | Forecast remarks |
| `pod_letter_num` | TEXT | POD letter |
| `pod_name` | TEXT | POD name |
| `onstream_year` | INTEGER | Onstream year |
| `onstream_actual` | TEXT | Onstream date |
| `project_class` | TEXT | Project classification. (Reserves & GRR, Contingent Resources, Prospective Resources. Abandoned is not part of the official classification but is included for recording purposes) |
| `project_level` | TEXT | Project level |
| `prod_stage` | TEXT | Production stage |
| `year` | INTEGER | Year of data |
| `hist_year` | INTEGER | Historical year |

### Production Columns (Time Series) - UNITS NEEDED

| Column | Type | Description | Unit? |
|--------|------|-------------|-------|
| `cprd_grs_oil` | REAL | Cumulative gross production oil | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_con` | REAL | Cumulative gross production condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_ga` | REAL | Cumulative gross production associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_grs_gn` | REAL | Cumulative gross production non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_grs_oc` | REAL | Cumulative gross production oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_grs_an` | REAL | Cumulative gross production associated + non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_oil` | REAL | Cumulative sales oil | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_con` | REAL | Cumulative sales condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_ga` | REAL | Cumulative sales associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_gn` | REAL | Cumulative sales non-associated gas | BSCF - Billion Standard Cubic Feet |
| `cprd_sls_oc` | REAL | Cumulative sales oil + condensate | MSTB - Thousand Stock Tank Barrels |
| `cprd_sls_an` | REAL | Cumulative sales associated + non-associated gas | BSCF - Billion Standard Cubic Feet |
| `rate_oil` | REAL | Production rate oil | MSTBY - Million Stock Tank Barrels per Year |
| `rate_con` | REAL | Production rate condensate | MSTBY - Million Stock Tank Barrels per Year |
| `rate_ga` | REAL | Production rate associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `rate_gn` | REAL | Production rate non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `rate_oc` | REAL | Production rate oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `rate_an` | REAL | Production rate total gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_oil` | REAL | Total Potential Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_con` | REAL | Total Potential Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_ga` | REAL | Total Potential Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_gn` | REAL | Total Potential Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_oc` | REAL | Total Potential Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_an` | REAL | Total Potential Forecast of associated + non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_risked_oil` | REAL | Risked Total Potential Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_risked_con` | REAL | Risked Total Potential Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_risked_ga` | REAL | Risked Total Potential Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_risked_gn` | REAL | Risked Total Potential Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `tpf_risked_oc` | REAL | Risked Total Potential Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `tpf_risked_an` | REAL | Risked Total Potential Forecast of associated + non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `slf_oil` | REAL | Sales Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `slf_con` | REAL | Sales Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `slf_ga` | REAL | Sales Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `slf_gn` | REAL | Sales Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `slf_oc` | REAL | Sales Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `slf_an` | REAL | Sales Forecast of associated + non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `spf_oil` | REAL | Sales Potential Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `spf_con` | REAL | Sales Potential Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `spf_ga` | REAL | Sales Potential Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `spf_gn` | REAL | Sales Potential Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `spf_oc` | REAL | Sales Potential Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `spf_an` | REAL | Sales Potential Forecast of associated + non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `crf_oil` | REAL | Contingent Resources Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `crf_con` | REAL | Contingent Resources Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `crf_ga` | REAL | Contingent Resources Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `crf_gn` | REAL | Contingent Resources Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `crf_oc` | REAL | Contingent Resources Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `crf_an` | REAL | Contingent Resources Forecast of total gas | BSCFY - Billion Standard Cubic Feet per Year |
| `prf_oil` | REAL | Prospective Resources Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `prf_con` | REAL | Prospective Resources Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `prf_ga` | REAL | Prospective Resources Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `prf_gn` | REAL | Prospective Resources Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `prf_oc` | REAL | Prospective Resources Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `prf_an` | REAL | Prospective Resources Forecast of total gas | BSCFY - Billion Standard Cubic Feet per Year |
| `ciof_oil` | REAL | Consumed in Operations Forecast of oil | MSTBY - Million Stock Tank Barrels per Year |
| `ciof_con` | REAL | Consumed in Operations Forecast of condensate | MSTBY - Million Stock Tank Barrels per Year |
| `ciof_ga` | REAL | Consumed in Operations Forecast of associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `ciof_gn` | REAL | Consumed in Operations Forecast of non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `ciof_oc` | REAL | Consumed in Operations Forecast of oil + condensate | MSTBY - Million Stock Tank Barrels per Year |
| `lossf_an` | REAL | Consumed in Operations Forecast of associated + non-associated gas | BSCFY - Billion Standard Cubic Feet per Year |
| `ghv` | REAL | Gross heating value |

---

## Open Questions

1. **Units** - Are oil volumes in MMSTB or MSTB? Gas in BCF or MMSCF? What's the convention in ESDC?
2. **dcpy columns** - What do UM, PPA, WI, CIO, GTR, UC stand for? (e.g., Ultimate Mechanical, Pre-Production Agreement, Well Initiate, etc.?)
3. **rec_an vs rec_ga/rec_gn** - Is `an` total gas (associated + non-associated)?
4. **rec_oc** - Is this oil equivalent or just oil + condensate combined?
5. **slf, spf, crf, prf** - What do these abbreviations mean in the context of the time series data?
6. **ciof** - What does this stand for?
7. **gcf columns** - What does GCF stand for (Gas Compression Factor?)?
