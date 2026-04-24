# ruff: noqa: E501
"""Complete database schema definitions for ESDC chat agent.

Provides full schema knowledge to eliminate schema tool calls.
With 256k context window, we include all columns with detailed descriptions.

Tables with unique columns:
- project_resources: Projects, reserves, and resources data
- project_timeseries: Monthly production, injection, and forecast data

Views (field_resources, wa_resources, nkri_resources, field_timeseries,
wa_timeseries, nkri_timeseries) aggregate from these tables and share
the same column names with different grouping levels.
"""

DATABASE_SCHEMA: dict[str, dict] = {
    "project_resources": {
        "description": "Main table for projects, reserves, and resources. One row per project per uncert_level per year. Grouped by project_id + report_year for unique projects.",
        "primary_key": ["project_id", "uncert_level"],
        "grouping": "GROUP BY project_id, report_year for unique projects (ignoring uncert_level duplicates). When aggregating volumes, use SUM() and GROUP BY report_year, then filter by uncert_level = '2. Middle Value' for 2P/2C/2U mid estimates.",
        "columns": {
            # === Identification ===
            "id": {
                "type": "INTEGER",
                "description": "Auto-increment ID. Not useful for queries.",
            },
            "project_id": {
                "type": "TEXT",
                "description": "Unique project identifier. Use for JOINs and deduplication. Same project_id appears 3 times (1P/2P/3P).",
            },
            "project_name": {
                "type": "TEXT",
                "description": "Human-readable project name. E.g., 'Duri Steam Flood Phase 3'. Use for display, NOT for filtering (use field_name or project_id instead).",
            },
            "project_name_previous": {
                "type": "TEXT",
                "description": "Previous project name if renamed.",
            },
            "field_id": {"type": "TEXT", "description": "Field identifier code."},
            "field_name": {
                "type": "TEXT",
                "description": "Field/lapangan name. KEY FILTER COLUMN. E.g., 'DURI', 'RAKAN', 'BELIDA'. Use ILIKE for partial matching.",
            },
            "field_name_previous": {
                "type": "TEXT",
                "description": "Previous field name if renamed.",
            },
            "field_lat": {"type": "REAL", "description": "Field latitude coordinate."},
            "field_long": {
                "type": "REAL",
                "description": "Field longitude coordinate.",
            },
            "field_area": {
                "type": "REAL",
                "description": "Field area in square kilometers.",
            },
            # === Working Area ===
            "wk_id": {"type": "TEXT", "description": "Working area identifier code."},
            "wk_name": {
                "type": "TEXT",
                "description": "Working area name. KEY FILTER COLUMN. E.g., 'ROKAN', 'SIAK', 'PEMATANG'. Use ILIKE for partial matching.",
            },
            "wk_lat": {"type": "REAL", "description": "Working area latitude."},
            "wk_long": {"type": "REAL", "description": "Working area longitude."},
            "wk_area": {"type": "REAL", "description": "Working area size (latest)."},
            "wk_area_ori": {
                "type": "REAL",
                "description": "Original working area size.",
            },
            "wk_area_latest": {
                "type": "REAL",
                "description": "Latest working area size.",
            },
            "wk_subgroup": {
                "type": "TEXT",
                "description": "Working area subgroup classification.",
            },
            "wk_area_perwakilan_skkmigas": {
                "type": "TEXT",
                "description": "SKK Migas regional representative area. KEY FILTER COLUMN.",
            },
            "wk_regionisasi_ngi": {
                "type": "TEXT",
                "description": "NGI regional classification. KEY FILTER COLUMN.",
            },
            # === Operator ===
            "operator_name": {
                "type": "TEXT",
                "description": "Operating company name. E.g., 'Pertamina Hulu Rokan'.",
            },
            "operator_group": {
                "type": "TEXT",
                "description": "Operator parent group. KEY FILTER COLUMN. E.g., 'Pertamina Hulu Mahakam'.",
            },
            # === Basin & Province ===
            "basin86_id": {
                "type": "TEXT",
                "description": "Basin identifier (86 classification).",
            },
            "basin86": {
                "type": "TEXT",
                "description": "Basin name (86 classification). E.g., 'Central Sumatra'.",
            },
            "basin128_id": {
                "type": "TEXT",
                "description": "Basin identifier (128 classification).",
            },
            "basin128": {
                "type": "TEXT",
                "description": "Basin name (128 classification). More detailed than basin86. KEY FILTER COLUMN.",
            },
            "prov_id": {"type": "TEXT", "description": "Province identifier code."},
            "province": {
                "type": "TEXT",
                "description": "Province name. E.g., 'Riau', 'Kalimantan Timur'. KEY FILTER COLUMN.",
            },
            # === PSC ===
            "psc_eff_start": {
                "type": "TEXT",
                "description": "PSC effective start date.",
            },
            "psc_eff_end": {"type": "TEXT", "description": "PSC effective end date."},
            # === POD ===
            "pod_letter_num": {
                "type": "TEXT",
                "description": "POD letter/number identifier.",
            },
            "pod_name": {
                "type": "TEXT",
                "description": (
                    "POD/Asset name. KEY FILTER COLUMN. Values include POD"
                    " variants: POD (Plan of Development), POFD (Plan of"
                    " Further Development), OPL (Optimasi Pengembangan"
                    " Lapangan), OPLL (Optimasi Pengembangan Lapangan -"
                    " Lapangan), POP (Put on Production), POD I (first POD"
                    " in a working area, approved by Minister of ESDM)."
                ),
            },
            "pod_date": {"type": "TEXT", "description": "POD date."},
            # === Project Classification ===
            "project_class": {
                "type": "TEXT",
                "description": "Project classification. VALUES: '1. Reserves & GRR', '2. Contingent Resources', '3. Prospective Resources'. KEY FILTER COLUMN. Use for 'cadangan' (Reserves) vs 'sumber daya' (Resources) queries.",
            },
            "project_stage": {
                "type": "TEXT",
                "description": "Project stage. VALUES: 'Exploration', 'Exploitation'. KEY FILTER COLUMN.",
            },
            "project_level": {
                "type": "TEXT",
                "description": "Project maturity level. VALUES: 'E0' through 'E8'. E0=undiscovered, E3=commercial discovery, E8=onstream. KEY FILTER COLUMN.",
            },
            "project_level_previous": {
                "type": "TEXT",
                "description": "Previous project level if changed.",
            },
            "prod_stage": {"type": "TEXT", "description": "Production stage."},
            "project_eol": {"type": "INTEGER", "description": "End of life indicator."},
            "project_isactive": {
                "type": "INTEGER",
                "description": "Whether project is currently active (0 or 1).",
            },
            "groovy_isactive": {
                "type": "INTEGER",
                "description": "Whether project is in Groovy system (0 or 1).",
            },
            "fusion_isactive": {
                "type": "INTEGER",
                "description": "Whether project is in Fusion system (0 or 1).",
            },
            "onstream_year": {
                "type": "INTEGER",
                "description": "Year project came onstream.",
            },
            "onstream_actual": {
                "type": "TEXT",
                "description": "Actual onstream date string.",
            },
            # === Flags ===
            "is_offshore": {
                "type": "INTEGER",
                "description": "Whether project is offshore (0=onshore, 1=offshore).",
            },
            "is_discovered": {
                "type": "INTEGER",
                "description": "Whether field is discovered (0 or 1).",
            },
            "is_ltp": {
                "type": "INTEGER",
                "description": "Whether project has a Long Term Plan (LTP). LTP = Long Term Plan, a multi-year development plan for oil & gas projects.",
            },
            "is_pse_approved": {
                "type": "INTEGER",
                "description": (
                    "Whether project has PSE (Penentuan Status Eksplorasi)"
                    " approved (0 or 1). PSE = determination of exploration"
                    " status for a project."
                ),
            },
            "is_pod_approved": {
                "type": "INTEGER",
                "description": (
                    "Whether project has POD (Plan of Development) approved"
                    " (0 or 1). POD = regulatory development plan for oil &"
                    " gas projects. Variants: POFD, OPL, OPLL, POP, POD I."
                ),
            },
            "is_report_accepted": {
                "type": "INTEGER",
                "description": "Whether report is accepted (0 or 1).",
            },
            "is_unitization": {
                "type": "INTEGER",
                "description": "Whether project involves unitization (0 or 1).",
            },
            # === Reporting ===
            "report_date": {"type": "TEXT", "description": "Report date string."},
            "report_year": {
                "type": "INTEGER",
                "description": "Report year. KEY FILTER COLUMN. E.g., 2024. Always filter by year for meaningful results.",
            },
            "report_status": {
                "type": "TEXT",
                "description": "Report submission status.",
            },
            "uncert_level": {
                "type": "TEXT",
                "description": "Uncertainty level. VALUES: '1. Low Value' (1P/1C/1U), '2. Middle Value' (2P/2C/2U), '3. High Value' (3P/3C/3U). KEY FILTER COLUMN. Default to '2. Middle Value' for mid estimates.",
            },
            # === Remarks ===
            "project_remarks": {
                "type": "TEXT",
                "description": "CRITICAL: Free-text description of project status, issues, and challenges. Contains economic viability, technical problems, delays. Use for semantic_search and ILIKE queries. E.g., 'Economic issues due to low oil price'. ALWAYS include in results when user asks about issues/problems.",
            },
            "vol_remarks": {
                "type": "TEXT",
                "description": "Volume-related remarks. Notes about production volume changes.",
            },
            # === Resources (rec_*) ===
            "rec_oil": {
                "type": "REAL",
                "description": "Resources crude oil (MSTB). Contingent + Prospective. Use SUM() aggregation.",
            },
            "rec_con": {
                "type": "REAL",
                "description": "Resources condensate (MSTB). Use SUM() aggregation.",
            },
            "rec_ga": {
                "type": "REAL",
                "description": "Resources associated gas (BSCF). Use SUM() aggregation.",
            },
            "rec_gn": {
                "type": "REAL",
                "description": "Resources non-associated gas (BSCF). Use SUM() aggregation.",
            },
            "rec_oc": {
                "type": "REAL",
                "description": "Resources oil+condensate (MSTB). Combined oil and condensate. KEY COLUMN for 'sumber daya minyak'. Use SUM() aggregation.",
            },
            "rec_an": {
                "type": "REAL",
                "description": "Resources total gas (BSCF). Combined associated and non-associated. KEY COLUMN for 'sumber daya gas'. Use SUM() aggregation.",
            },
            "rec_mboe": {
                "type": "REAL",
                "description": "Resources oil equivalent (MBOE). Combined volumes converted to oil equivalent.",
            },
            "rec_oil_risked": {
                "type": "REAL",
                "description": "Resources crude oil risked (MSTB). Identical to rec_oil for GRR and Contingent (GCF=1). Only differs for Prospective Resources. Use SUM() aggregation.",
            },
            "rec_con_risked": {
                "type": "REAL",
                "description": "Resources condensate risked (MSTB). Identical to rec_con for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            "rec_ga_risked": {
                "type": "REAL",
                "description": "Resources associated gas risked (BSCF). Identical to rec_ga for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            "rec_gn_risked": {
                "type": "REAL",
                "description": "Resources non-associated gas risked (BSCF). Identical to rec_gn for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            "rec_oc_risked": {
                "type": "REAL",
                "description": "Resources oil+condensate risked (MSTB). Identical to rec_oc for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            "rec_an_risked": {
                "type": "REAL",
                "description": "Resources total gas risked (BSCF). Identical to rec_an for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            "rec_mboe_risked": {
                "type": "REAL",
                "description": "Resources oil equivalent risked (MBOE). Identical to rec_mboe for GRR and Contingent (GCF=1). Only differs for Prospective Resources.",
            },
            # === Reserves (res_*) ===
            "res_oil": {
                "type": "REAL",
                "description": "Reserves crude oil (MSTB). Commercial volumes only. Use SUM() aggregation.",
            },
            "res_con": {
                "type": "REAL",
                "description": "Reserves condensate (MSTB). Use SUM() aggregation.",
            },
            "res_ga": {
                "type": "REAL",
                "description": "Reserves associated gas (BSCF). Use SUM() aggregation.",
            },
            "res_gn": {
                "type": "REAL",
                "description": "Reserves non-associated gas (BSCF). Use SUM() aggregation.",
            },
            "res_oc": {
                "type": "REAL",
                "description": "Reserves oil+condensate (MSTB). KEY COLUMN for 'cadangan minyak'. Use SUM() aggregation.",
            },
            "res_an": {
                "type": "REAL",
                "description": "Reserves total gas (BSCF). KEY COLUMN for 'cadangan gas'. Use SUM() aggregation.",
            },
            # === In-Place ===
            "prj_ioip": {
                "type": "REAL",
                "description": "Project Initial Oil In Place (MSTB). Use SUM() aggregation.",
            },
            "prj_igip": {
                "type": "REAL",
                "description": "Project Initial Gas In Place (BSCF). Use SUM() aggregation.",
            },
            # === EUR ===
            "eur_rec_oil": {
                "type": "REAL",
                "description": "Estimated Ultimate Recovery - Resources oil (MSTB).",
            },
            "eur_rec_con": {
                "type": "REAL",
                "description": "EUR - Resources condensate (MSTB).",
            },
            "eur_rec_ga": {
                "type": "REAL",
                "description": "EUR - Resources associated gas (BSCF).",
            },
            "eur_rec_gn": {
                "type": "REAL",
                "description": "EUR - Resources non-associated gas (BSCF).",
            },
            "eur_rec_oc": {
                "type": "REAL",
                "description": "EUR - Resources oil+condensate (MSTB).",
            },
            "eur_rec_an": {
                "type": "REAL",
                "description": "EUR - Resources total gas (BSCF).",
            },
            "eur_res_oil": {
                "type": "REAL",
                "description": "EUR - Reserves oil (MSTB).",
            },
            "eur_res_con": {
                "type": "REAL",
                "description": "EUR - Reserves condensate (MSTB).",
            },
            "eur_res_ga": {
                "type": "REAL",
                "description": "EUR - Reserves associated gas (BSCF).",
            },
            "eur_res_gn": {
                "type": "REAL",
                "description": "EUR - Reserves non-associated gas (BSCF).",
            },
            "eur_res_oc": {
                "type": "REAL",
                "description": "EUR - Reserves oil+condensate (MSTB).",
            },
            "eur_res_an": {
                "type": "REAL",
                "description": "EUR - Reserves total gas (BSCF).",
            },
            # === Recovery Factor ===
            "rf_oil": {
                "type": "REAL",
                "description": "Recovery factor oil (fraction). Calculated: cumulative production / IOIP.",
            },
            "rf_gas": {
                "type": "REAL",
                "description": "Recovery factor gas (fraction). Calculated: cumulative production / IGIP.",
            },
            "urf_rec_oil": {
                "type": "REAL",
                "description": "Ultimate recovery factor - Resources oil (fraction).",
            },
            "urf_rec_gas": {
                "type": "REAL",
                "description": "Ultimate recovery factor - Resources gas (fraction).",
            },
            "urf_res_oil": {
                "type": "REAL",
                "description": "Ultimate recovery factor - Reserves oil (fraction).",
            },
            "urf_res_gas": {
                "type": "REAL",
                "description": "Ultimate recovery factor - Reserves gas (fraction).",
            },
            # === GCF (Geological Chance Factor) ===
            "gcf_srock": {"type": "REAL", "description": "GCF for source rock."},
            "gcf_res": {"type": "REAL", "description": "GCF for reservoir."},
            "gcf_ts": {"type": "REAL", "description": "GCF for trap/seal."},
            "gcf_dyn": {"type": "REAL", "description": "GCF for dynamics/charge."},
            "gcf_total": {
                "type": "REAL",
                "description": "Total GCF. Multiply with resources volumes for risked values. For Prospective Resources only.",
            },
            "pd": {
                "type": "REAL",
                "description": "Probability of discovery (0-1). For Prospective Resources.",
            },
            # === Production Rate ===
            "rate_grs_oil": {
                "type": "REAL",
                "description": "Gross production rate oil (MSTB/d).",
            },
            "rate_grs_con": {
                "type": "REAL",
                "description": "Gross production rate condensate (MSTB/d).",
            },
            "rate_grs_ga": {
                "type": "REAL",
                "description": "Gross production rate associated gas (BSCF/d).",
            },
            "rate_grs_gn": {
                "type": "REAL",
                "description": "Gross production rate non-associated gas (BSCF/d).",
            },
            "rate_grs_oc": {
                "type": "REAL",
                "description": "Gross production rate oil+condensate (MSTB/d).",
            },
            "rate_grs_an": {
                "type": "REAL",
                "description": "Gross production rate total gas (BSCF/d).",
            },
            "rate_sls_oil": {
                "type": "REAL",
                "description": "Sales production rate oil (MSTB/d).",
            },
            "rate_sls_con": {
                "type": "REAL",
                "description": "Sales production rate condensate (MSTB/d).",
            },
            "rate_sls_ga": {
                "type": "REAL",
                "description": "Sales production rate associated gas (BSCF/d).",
            },
            "rate_sls_gn": {
                "type": "REAL",
                "description": "Sales production rate non-associated gas (BSCF/d).",
            },
            "rate_sls_oc": {
                "type": "REAL",
                "description": "Sales production rate oil+condensate (MSTB/d).",
            },
            "rate_sls_an": {
                "type": "REAL",
                "description": "Sales production rate total gas (BSCF/d).",
            },
            # === Cumulative Production ===
            "cprd_grs_oil": {
                "type": "REAL",
                "description": "Cumulative gross production oil (MSTB).",
            },
            "cprd_grs_con": {
                "type": "REAL",
                "description": "Cumulative gross production condensate (MSTB).",
            },
            "cprd_grs_ga": {
                "type": "REAL",
                "description": "Cumulative gross production associated gas (BSCF).",
            },
            "cprd_grs_gn": {
                "type": "REAL",
                "description": "Cumulative gross production non-associated gas (BSCF).",
            },
            "cprd_grs_oc": {
                "type": "REAL",
                "description": "Cumulative gross production oil+condensate (MSTB).",
            },
            "cprd_grs_an": {
                "type": "REAL",
                "description": "Cumulative gross production total gas (BSCF).",
            },
            "cprd_sls_oil": {
                "type": "REAL",
                "description": "Cumulative sales production oil (MSTB).",
            },
            "cprd_sls_con": {
                "type": "REAL",
                "description": "Cumulative sales production condensate (MSTB).",
            },
            "cprd_sls_ga": {
                "type": "REAL",
                "description": "Cumulative sales production associated gas (BSCF).",
            },
            "cprd_sls_gn": {
                "type": "REAL",
                "description": "Cumulative sales production non-associated gas (BSCF).",
            },
            "cprd_sls_oc": {
                "type": "REAL",
                "description": "Cumulative sales production oil+condensate (MSTB).",
            },
            "cprd_sls_an": {
                "type": "REAL",
                "description": "Cumulative sales production total gas (BSCF).",
            },
            # === Discrepancy (dcpy_*) ===
            # dcpy = Discrepancy (volume reconciliation between reporting periods)
            "dcpy_um_oil": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies oil (MSTB).",
            },
            "dcpy_um_con": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies condensate (MSTB).",
            },
            "dcpy_um_ga": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies associated gas (BSCF).",
            },
            "dcpy_um_gn": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies non-associated gas (BSCF).",
            },
            "dcpy_um_oc": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies oil+condensate (MSTB).",
            },
            "dcpy_um_an": {
                "type": "REAL",
                "description": "Update Model - volume changes from Geology, Geophysics, and Reservoir studies total gas (BSCF).",
            },
            "dcpy_ppa_oil": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis oil (MSTB).",
            },
            "dcpy_ppa_con": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis condensate (MSTB).",
            },
            "dcpy_ppa_ga": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis associated gas (BSCF).",
            },
            "dcpy_ppa_gn": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis non-associated gas (BSCF).",
            },
            "dcpy_ppa_oc": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis oil+condensate (MSTB).",
            },
            "dcpy_ppa_an": {
                "type": "REAL",
                "description": "Production Performance Analysis - volume changes from production profile analysis total gas (BSCF).",
            },
            "dcpy_wi_oil": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities oil (MSTB).",
            },
            "dcpy_wi_con": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities condensate (MSTB).",
            },
            "dcpy_wi_ga": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities associated gas (BSCF).",
            },
            "dcpy_wi_gn": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities non-associated gas (BSCF).",
            },
            "dcpy_wi_oc": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities oil+condensate (MSTB).",
            },
            "dcpy_wi_an": {
                "type": "REAL",
                "description": "Well Intervention - volume changes from well workover or well service activities total gas (BSCF).",
            },
            "dcpy_cio_oil": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses oil (MSTB).",
            },
            "dcpy_cio_con": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses condensate (MSTB).",
            },
            "dcpy_cio_ga": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses associated gas (BSCF).",
            },
            "dcpy_cio_gn": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses non-associated gas (BSCF).",
            },
            "dcpy_cio_oc": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses oil+condensate (MSTB).",
            },
            "dcpy_cio_an": {
                "type": "REAL",
                "description": "Consumed in Operations - fuel, flare, and shrinkage losses total gas (BSCF).",
            },
            "dcpy_gtr_oil": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions oil (MSTB).",
            },
            "dcpy_gtr_con": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions condensate (MSTB).",
            },
            "dcpy_gtr_ga": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions associated gas (BSCF).",
            },
            "dcpy_gtr_gn": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions non-associated gas (BSCF).",
            },
            "dcpy_gtr_oc": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions oil+condensate (MSTB).",
            },
            "dcpy_gtr_an": {
                "type": "REAL",
                "description": "GRR to Reserves - reserve additions from GRR reclassification or commercial factor reductions total gas (BSCF).",
            },
            "dcpy_uc_oil": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes oil (MSTB).",
            },
            "dcpy_uc_con": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes condensate (MSTB).",
            },
            "dcpy_uc_ga": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes associated gas (BSCF).",
            },
            "dcpy_uc_gn": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes non-associated gas (BSCF).",
            },
            "dcpy_uc_oc": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes oil+condensate (MSTB).",
            },
            "dcpy_uc_an": {
                "type": "REAL",
                "description": "Unaccounted Changes - volume corrections from data reconciliation or record fixes total gas (BSCF).",
            },
            "ghv_avg": {"type": "REAL", "description": "Average Gross Heating Value."},
        },
    },
    "project_timeseries": {
        "description": "Monthly production, injection, and forecast data. One row per project per year per forecast type.",
        "primary_key": ["project_id", "year"],
        "grouping": "For field-level aggregates, use field_timeseries view. For WA-level, use wa_timeseries. For NKRI-level, use nkri_timeseries.",
        "columns": {
            # === Identification ===
            "id": {
                "type": "INTEGER",
                "description": "Auto-increment ID. Not useful for queries.",
            },
            "report_date": {"type": "TEXT", "description": "Report date string."},
            "report_year": {
                "type": "INTEGER",
                "description": "Report year. KEY FILTER COLUMN.",
            },
            "report_status": {
                "type": "TEXT",
                "description": "Report submission status.",
            },
            "is_offshore": {
                "type": "INTEGER",
                "description": "Whether offshore (0 or 1).",
            },
            # === Location ===
            "basin86_id": {
                "type": "TEXT",
                "description": "Basin identifier (86 classification).",
            },
            "basin128": {
                "type": "TEXT",
                "description": "Basin name (128 classification).",
            },
            "province": {"type": "TEXT", "description": "Province name."},
            "operator_group": {"type": "TEXT", "description": "Operator parent group."},
            "operator_name": {"type": "TEXT", "description": "Operating company name."},
            "wk_id": {"type": "TEXT", "description": "Working area identifier."},
            "wk_name": {
                "type": "TEXT",
                "description": "Working area name. KEY FILTER COLUMN.",
            },
            "wk_area": {"type": "REAL", "description": "Working area size."},
            "wk_regionisasi_ngi": {
                "type": "TEXT",
                "description": "NGI regional classification.",
            },
            "wk_area_perwakilan_skkmigas": {
                "type": "TEXT",
                "description": "SKK Migas regional representative area.",
            },
            "wk_subgroup": {"type": "TEXT", "description": "Working area subgroup."},
            "wk_lat": {"type": "TEXT", "description": "Working area latitude."},
            "wk_long": {"type": "TEXT", "description": "Working area longitude."},
            "psc_eff_start": {
                "type": "TEXT",
                "description": "PSC effective start date.",
            },
            "psc_eff_end": {"type": "TEXT", "description": "PSC effective end date."},
            "field_id": {"type": "TEXT", "description": "Field identifier."},
            "field_name": {
                "type": "TEXT",
                "description": "Field name. KEY FILTER COLUMN.",
            },
            "field_lat": {"type": "TEXT", "description": "Field latitude."},
            "field_long": {"type": "TEXT", "description": "Field longitude."},
            "field_area": {"type": "REAL", "description": "Field area size (km^2)."},
            # === Project ===
            "project_id": {
                "type": "TEXT",
                "description": "Project identifier. KEY for JOINs.",
            },
            "project_name": {"type": "TEXT", "description": "Project name."},
            "project_isactive": {
                "type": "INTEGER",
                "description": "Whether project is active (0 or 1).",
            },
            "project_remarks": {
                "type": "TEXT",
                "description": "Free-text project status/issue description. For semantic search.",
            },
            "vol_remarks": {"type": "TEXT", "description": "Volume-related remarks."},
            "frcast_remarks": {
                "type": "TEXT",
                "description": "Forecast-related remarks.",
            },
            "pod_letter_num": {"type": "TEXT", "description": "POD letter/number."},
            "pod_name": {
                "type": "TEXT",
                "description": (
                    "POD/Asset name. Values include POD variants: POD, POFD,"
                    " OPL, OPLL, POP, POD I."
                ),
            },
            "onstream_year": {
                "type": "INTEGER",
                "description": "Year project came onstream.",
            },
            "onstream_actual": {"type": "TEXT", "description": "Actual onstream date."},
            "project_class": {
                "type": "TEXT",
                "description": "Project classification. VALUES: '1. Reserves & GRR', '2. Contingent Resources', '3. Prospective Resources'.",
            },
            "project_level": {
                "type": "TEXT",
                "description": "Project maturity level. VALUES: 'E0' through 'E8'.",
            },
            "prod_stage": {"type": "TEXT", "description": "Production stage."},
            # === Time ===
            "year": {
                "type": "INTEGER",
                "description": "Forecast year. KEY FILTER COLUMN for timeseries queries. Different from report_year.",
            },
            "hist_year": {
                "type": "INTEGER",
                "description": "Historical production year.",
            },
            # === Forecast Columns (per substance suffix pattern: _oil, _con, _ga, _gn, _oc, _an) ===
            "tpf_oil": {
                "type": "REAL",
                "description": "Total Potential Forecast - oil (MSTB). TPF = total recoverable volume profile over years. Sum equals rec_* from project_resources.",
            },
            "tpf_con": {
                "type": "REAL",
                "description": "Total Potential Forecast - condensate (MSTB).",
            },
            "tpf_ga": {
                "type": "REAL",
                "description": "Total Potential Forecast - associated gas (BSCF).",
            },
            "tpf_gn": {
                "type": "REAL",
                "description": "Total Potential Forecast - non-associated gas (BSCF).",
            },
            "tpf_oc": {
                "type": "REAL",
                "description": "Total Potential Forecast - oil+condensate (MSTB).",
            },
            "tpf_an": {
                "type": "REAL",
                "description": "Total Potential Forecast - total gas (BSCF).",
            },
            "tpf_risked_oil": {
                "type": "REAL",
                "description": "TPF risked - oil (MSTB). Prospective only, multiplied by geological chance.",
            },
            "tpf_risked_con": {
                "type": "REAL",
                "description": "TPF risked - condensate (MSTB).",
            },
            "tpf_risked_ga": {
                "type": "REAL",
                "description": "TPF risked - associated gas (BSCF).",
            },
            "tpf_risked_gn": {
                "type": "REAL",
                "description": "TPF risked - non-associated gas (BSCF).",
            },
            "tpf_risked_oc": {
                "type": "REAL",
                "description": "TPF risked - oil+condensate (MSTB).",
            },
            "tpf_risked_an": {
                "type": "REAL",
                "description": "TPF risked - total gas (BSCF).",
            },
            "slf_oil": {
                "type": "REAL",
                "description": "Sales Forecast - oil (MSTB). SLF = reserves production profile. Sum equals res_* from project_resources.",
            },
            "slf_con": {
                "type": "REAL",
                "description": "Sales Forecast - condensate (MSTB).",
            },
            "slf_ga": {
                "type": "REAL",
                "description": "Sales Forecast - associated gas (BSCF).",
            },
            "slf_gn": {
                "type": "REAL",
                "description": "Sales Forecast - non-associated gas (BSCF).",
            },
            "slf_oc": {
                "type": "REAL",
                "description": "Sales Forecast - oil+condensate (MSTB).",
            },
            "slf_an": {
                "type": "REAL",
                "description": "Sales Forecast - total gas (BSCF).",
            },
            "spf_oil": {
                "type": "REAL",
                "description": "Sales Potential Forecast - oil (MSTB). SPF = TPF - SLF.",
            },
            "spf_con": {
                "type": "REAL",
                "description": "Sales Potential Forecast - condensate (MSTB).",
            },
            "spf_ga": {
                "type": "REAL",
                "description": "Sales Potential Forecast - associated gas (BSCF).",
            },
            "spf_gn": {
                "type": "REAL",
                "description": "Sales Potential Forecast - non-associated gas (BSCF).",
            },
            "spf_oc": {
                "type": "REAL",
                "description": "Sales Potential Forecast - oil+condensate (MSTB).",
            },
            "spf_an": {
                "type": "REAL",
                "description": "Sales Potential Forecast - total gas (BSCF).",
            },
            "crf_oil": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - oil (MSTB). For Contingent Resources projects only.",
            },
            "crf_con": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - condensate (MSTB).",
            },
            "crf_ga": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - associated gas (BSCF).",
            },
            "crf_gn": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - non-associated gas (BSCF).",
            },
            "crf_oc": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - oil+condensate (MSTB).",
            },
            "crf_an": {
                "type": "REAL",
                "description": "Contingent Resources Forecast - total gas (BSCF).",
            },
            "prf_oil": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - oil (MSTB). For Prospective Resources projects only.",
            },
            "prf_con": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - condensate (MSTB).",
            },
            "prf_ga": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - associated gas (BSCF).",
            },
            "prf_gn": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - non-associated gas (BSCF).",
            },
            "prf_oc": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - oil+condensate (MSTB).",
            },
            "prf_an": {
                "type": "REAL",
                "description": "Prospective Resources Forecast - total gas (BSCF).",
            },
            "ciof_oil": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - oil (MSTB). Fuel, flare, shrinkage.",
            },
            "ciof_con": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - condensate (MSTB).",
            },
            "ciof_ga": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - associated gas (BSCF).",
            },
            "ciof_gn": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - non-associated gas (BSCF).",
            },
            "ciof_oc": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - oil+condensate (MSTB).",
            },
            "ciof_an": {
                "type": "REAL",
                "description": "Consumed in Operation Forecast - total gas (BSCF).",
            },
            "lossf_oil": {
                "type": "REAL",
                "description": "Loss Production Forecast - oil (MSTB).",
            },
            "lossf_con": {
                "type": "REAL",
                "description": "Loss Production Forecast - condensate (MSTB).",
            },
            "lossf_ga": {
                "type": "REAL",
                "description": "Loss Production Forecast - associated gas (BSCF).",
            },
            "lossf_gn": {
                "type": "REAL",
                "description": "Loss Production Forecast - non-associated gas (BSCF).",
            },
            "lossf_oc": {
                "type": "REAL",
                "description": "Loss Production Forecast - oil+condensate (MSTB).",
            },
            "lossf_an": {
                "type": "REAL",
                "description": "Loss Production Forecast - total gas (BSCF).",
            },
            # === Cumulative Production (in timeseries) ===
            "cprd_grs_oil": {
                "type": "REAL",
                "description": "Cumulative gross production oil (MSTB).",
            },
            "cprd_grs_con": {
                "type": "REAL",
                "description": "Cumulative gross production condensate (MSTB).",
            },
            "cprd_grs_ga": {
                "type": "REAL",
                "description": "Cumulative gross production associated gas (BSCF).",
            },
            "cprd_grs_gn": {
                "type": "REAL",
                "description": "Cumulative gross production non-associated gas (BSCF).",
            },
            "cprd_grs_oc": {
                "type": "REAL",
                "description": "Cumulative gross production oil+condensate (MSTB).",
            },
            "cprd_grs_an": {
                "type": "REAL",
                "description": "Cumulative gross production total gas (BSCF).",
            },
            "cprd_sls_oil": {
                "type": "REAL",
                "description": "Cumulative sales production oil (MSTB).",
            },
            "cprd_sls_con": {
                "type": "REAL",
                "description": "Cumulative sales production condensate (MSTB).",
            },
            "cprd_sls_ga": {
                "type": "REAL",
                "description": "Cumulative sales production associated gas (BSCF).",
            },
            "cprd_sls_gn": {
                "type": "REAL",
                "description": "Cumulative sales production non-associated gas (BSCF).",
            },
            "cprd_sls_oc": {
                "type": "REAL",
                "description": "Cumulative sales production oil+condensate (MSTB).",
            },
            "cprd_sls_an": {
                "type": "REAL",
                "description": "Cumulative sales production total gas (BSCF).",
            },
            # === Production Rate (in timeseries) ===
            "rate_oil": {
                "type": "REAL",
                "description": "Production rate oil (MSTB/d).",
            },
            "rate_con": {
                "type": "REAL",
                "description": "Production rate condensate (MSTB/d).",
            },
            "rate_ga": {
                "type": "REAL",
                "description": "Production rate associated gas (BSCF/d).",
            },
            "rate_gn": {
                "type": "REAL",
                "description": "Production rate non-associated gas (BSCF/d).",
            },
            "rate_oc": {
                "type": "REAL",
                "description": "Production rate oil+condensate (MSTB/d).",
            },
            "rate_an": {
                "type": "REAL",
                "description": "Production rate total gas (BSCF/d).",
            },
            "ghv": {"type": "REAL", "description": "Gross Heating Value."},
        },
    },
    # === Views (aggregated from project_resources) ===
    "field_resources": {
        "description": "Aggregated view of project_resources by field. Groups by field_name, report_year, project_class, uncert_level. Use for field-level queries.",
        "primary_key": ["field_name", "report_year", "project_class", "uncert_level"],
        "grouping": "GROUP BY field_name, report_year, project_class, uncert_level. Contains same volume columns as project_resources (res_*, rec_*, etc.) plus aggregated fields like pod_count, project_count, field_remarks.",
        "unique_columns": {
            "field_id": {"type": "TEXT", "description": "Field identifier."},
            "field_name": {
                "type": "TEXT",
                "description": "Field name. KEY FILTER COLUMN.",
            },
            "field_remarks": {
                "type": "TEXT",
                "description": "Aggregated field-level remarks from project_remarks. Contains issues from all projects in the field.",
            },
            "field_vol_remarks": {
                "type": "TEXT",
                "description": "Aggregated field-level volume remarks.",
            },
            "pod_count": {
                "type": "INTEGER",
                "description": "Number of PODs in this field.",
            },
            "project_count": {
                "type": "INTEGER",
                "description": "Number of projects in this field.",
            },
            "project_active_count": {
                "type": "INTEGER",
                "description": "Number of active projects.",
            },
            "ioip": {
                "type": "REAL",
                "description": "Aggregated Initial Oil In Place (MSTB). Same as prj_ioip.",
            },
            "igip": {
                "type": "REAL",
                "description": "Aggregated Initial Gas In Place (BSCF). Same as prj_igip.",
            },
        },
    },
    "wa_resources": {
        "description": "Aggregated view of project_resources by working area. Groups by wk_name, report_year, project_class, uncert_level. Use for working area-level queries like 'berapa cadangan WK Rokan'.",
        "primary_key": ["wk_name", "report_year", "project_class", "uncert_level"],
        "grouping": "GROUP BY wk_name, report_year, project_class, uncert_level. Contains same volume columns as project_resources plus aggregated fields.",
        "unique_columns": {
            "wk_id": {"type": "TEXT", "description": "Working area identifier."},
            "wk_name": {
                "type": "TEXT",
                "description": "Working area name. KEY FILTER COLUMN. E.g., 'ROKAN'.",
            },
            "wa_remarks": {
                "type": "TEXT",
                "description": "Aggregated WA-level remarks from field_remarks.",
            },
            "wa_vol_remarks": {
                "type": "TEXT",
                "description": "Aggregated WA-level volume remarks.",
            },
            "field_count": {
                "type": "INTEGER",
                "description": "Number of fields in this WA.",
            },
            "unitisasi_count": {
                "type": "INTEGER",
                "description": "Number of unitization projects.",
            },
            "pod_count": {
                "type": "INTEGER",
                "description": "Number of PODs in this WA.",
            },
            "project_count": {
                "type": "INTEGER",
                "description": "Number of projects in this WA.",
            },
            "project_active_count": {
                "type": "INTEGER",
                "description": "Number of active projects.",
            },
        },
    },
    "nkri_resources": {
        "description": "Aggregated view of project_resources at national level. Groups by report_year, project_class, uncert_level. Use for Indonesia-level queries.",
        "primary_key": ["report_year", "project_class", "uncert_level"],
        "grouping": "GROUP BY report_year, project_class, uncert_level.",
        "unique_columns": {
            "wa_count": {"type": "INTEGER", "description": "Number of working areas."},
            "field_count": {"type": "INTEGER", "description": "Number of fields."},
            "project_count": {"type": "INTEGER", "description": "Number of projects."},
            "nkri_remarks": {
                "type": "TEXT",
                "description": "Aggregated national-level remarks.",
            },
        },
    },
}


def get_schema_for_prompt() -> str:
    """Convert full schema definitions to formatted string for system prompt.

    Generates a comprehensive schema reference that eliminates the need
    for schema discovery tools (get_schema, list_tables, get_recommended_table,
    Resources Column Guide, etc.).

    Returns:
        Formatted schema string suitable for injection into system prompt.
    """
    lines = ["## Database Schema\n"]

    for table_name, table_info in DATABASE_SCHEMA.items():
        lines.append(f"\n### {table_name}")
        lines.append(f"{table_info['description']}")
        if "grouping" in table_info:
            lines.append(f"**Aggregation:** {table_info['grouping']}")
        if "primary_key" in table_info:
            lines.append(f"**Primary key:** {', '.join(table_info['primary_key'])}")
        lines.append("")

        columns = table_info.get("columns", {})
        if not columns:
            unique = table_info.get("unique_columns", {})
            if unique:
                lines.append(
                    "**Common columns** (same as project_resources): res_*, rec_*, eur_*, rate_*, cprd_*, dcpy_*, gcf_*, pd, rf_*, urf_*"
                )
                lines.append("")
                lines.append("**Unique columns:**")
                lines.append("| Column | Type | Description |")
                lines.append("|--------|------|-------------|")
                for col_name, col_info in unique.items():
                    col_type = col_info.get("type", "TEXT")
                    desc = col_info.get("description", "")
                    lines.append(f"| {col_name} | {col_type} | {desc} |")
            continue

        lines.append("| Column | Type | Description |")
        lines.append("|--------|------|-------------|")
        for col_name, col_info in columns.items():
            col_type = col_info.get("type", "TEXT")
            desc = col_info.get("description", "")
            agg = col_info.get("aggregation")
            units = col_info.get("units")
            if agg:
                desc = f"{desc} Aggregation: {agg}()."
            if units:
                desc = f"{desc} Units: {units}."
            lines.append(f"| {col_name} | {col_type} | {desc} |")

    return "\n".join(lines)
