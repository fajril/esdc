"""
Domain knowledge mapping for ESDC chat agent.

Maps Indonesian/English oil & gas terminology to database schema columns and SQL patterns.
Based on Kerangka Sumber Daya Migas Indonesia (KSMI) and PRMS standards.
"""

from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass


# =============================================================================
# Table/View Hierarchy
# =============================================================================

TABLE_HIERARCHY: Dict[str, str] = {
    "project": "project_resources",
    "field": "field_resources",
    "lapangan": "field_resources",
    "work_area": "wa_resources",
    "wa": "wa_resources",
    "wilayah_kerja": "wa_resources",
    "wilayah kerja": "wa_resources",
    "national": "nkri_resources",
    "nkri": "nkri_resources",
    "nasional": "nkri_resources",
    # Timeseries views
    "field_timeseries": "field_timeseries",
    "timeseries_field": "field_timeseries",
    "ts_field": "field_timeseries",
    "wa_timeseries": "wa_timeseries",
    "timeseries_wa": "wa_timeseries",
    "ts_wa": "wa_timeseries",
    "nkri_timeseries": "nkri_timeseries",
    "timeseries_nkri": "nkri_timeseries",
    "ts_nkri": "nkri_timeseries",
    "timeseries_national": "nkri_timeseries",
}

AGGREGATION_LEVELS: List[Tuple[str, str, int]] = [
    ("project_resources", "project", 1),
    ("field_resources", "field", 2),
    ("wa_resources", "work_area", 3),
    ("nkri_resources", "national", 4),
    ("field_timeseries", "field_timeseries", 5),
    ("wa_timeseries", "wa_timeseries", 6),
    ("nkri_timeseries", "nkri_timeseries", 7),
]


def get_table_for_query(
    entity_type: Optional[str] = None,
    entity_name: Optional[str] = None,
    require_detail: bool = False,
) -> str:
    """
    Get the recommended table/view for a query.

    Uses pre-aggregated views for efficiency at field/work_area/national levels.

    Args:
        entity_type: Type of entity (field, work_area, wa, national, nkri)
        entity_name: Name of specific entity (not used for selection, but for future use)
        require_detail: If True, use project_resources for detailed analysis

    Returns:
        Table name to query

    Examples:
        >>> get_table_for_query("field")
        'field_resources'

        >>> get_table_for_query("work_area")
        'wa_resources'

        >>> get_table_for_query("field", require_detail=True)
        'project_resources'

        >>> get_table_for_query("project")
        'project_resources'
    """
    if require_detail:
        return "project_resources"

    if entity_type:
        entity_type = entity_type.lower().strip()
        return TABLE_HIERARCHY.get(entity_type, "project_resources")

    return "project_resources"


def can_use_view_for_calculation(uncertainty: str, table: str) -> bool:
    """
    Check if a view can be used for uncertainty calculations.

    Views have pre-aggregated data by uncert_level, so calculated
    values (probable/possible) can be computed from views.

    Args:
        uncertainty: Uncertainty level (probable, possible, 1P, 2P, etc.)
        table: Table name

    Returns:
        True if view can be used for this calculation

    Examples:
        >>> can_use_view_for_calculation("2P", "field_resources")
        True

        >>> can_use_view_for_calculation("probable", "wa_resources")
        True
    """
    valid_tables = [
        "project_resources",
        "field_resources",
        "wa_resources",
        "nkri_resources",
        # Timeseries views
        "field_timeseries",
        "wa_timeseries",
        "nkri_timeseries",
    ]

    if table not in valid_tables:
        return False

    return True


def get_entity_filter_column(entity_type: str, table: str) -> Optional[str]:
    """
    Get the column name to filter by for a given entity type.

    Args:
        entity_type: Type of entity (field, work_area, national, timeseries)
        table: Table name

    Returns:
        Column name for filtering, or None if no filter needed

    Examples:
        >>> get_entity_filter_column("field", "field_resources")
        'field_name'

        >>> get_entity_filter_column("work_area", "wa_resources")
        'wk_name'

        >>> get_entity_filter_column("field", "field_timeseries")
        'field_name'
    """
    entity_type = entity_type.lower().strip() if entity_type else ""

    # Static views
    if entity_type in ["field"]:
        return "field_name"
    elif entity_type in ["work_area", "wa"]:
        return "wk_name"
    elif entity_type in ["national", "nkri"]:
        return None

    # Timeseries views
    if entity_type in ["field_timeseries", "ts_field", "timeseries_field"]:
        return "field_name"
    elif entity_type in ["wa_timeseries", "ts_wa", "timeseries_wa"]:
        return "wk_name"
    elif entity_type in [
        "nkri_timeseries",
        "ts_nkri",
        "timeseries_nkri",
        "timeseries_national",
    ]:
        return None

    if table == "project_resources":
        if "field" in entity_type or entity_type == "":
            return "field_name"
        elif "wk" in entity_type or "work_area" in entity_type:
            return "wk_name"

    # Timeseries tables
    if "timeseries" in table or table.endswith("_timeseries"):
        if "field" in entity_type:
            return "field_name"
        elif "wk" in entity_type or "work_area" in entity_type or "wa" in entity_type:
            return "wk_name"
        elif "national" in entity_type or "nkri" in entity_type:
            return None

    return "field_name"


# =============================================================================
# Column Groups
# =============================================================================

COLUMN_GROUPS: Dict[str, List[str]] = {
    "reserves": ["res_oil", "res_con", "res_ga", "res_gn", "res_oc", "res_an"],
    "resources": [
        "rec_oil",
        "rec_con",
        "rec_ga",
        "rec_gn",
        "rec_oc",
        "rec_an",
        "rec_mboe",
    ],
    "resources_risked": [
        "rec_oil_risked",
        "rec_con_risked",
        "rec_ga_risked",
        "rec_gn_risked",
        "rec_oc_risked",
        "rec_an_risked",
        "rec_mboe_risked",
    ],
    "eur_reserves": [
        "eur_res_oil",
        "eur_res_con",
        "eur_res_ga",
        "eur_res_gn",
        "eur_res_oc",
        "eur_res_an",
    ],
    "eur_resources": [
        "eur_rec_oil",
        "eur_rec_con",
        "eur_rec_ga",
        "eur_rec_gn",
        "eur_rec_oc",
        "eur_rec_an",
    ],
    "in_place": ["prj_ioip", "prj_igip"],
    "recovery_factors": [
        "rf_oil",
        "rf_gas",
        "urf_rec_oil",
        "urf_rec_gas",
        "urf_res_oil",
        "urf_res_gas",
    ],
    "production_rates": [
        "rate_grs_oil",
        "rate_grs_con",
        "rate_grs_ga",
        "rate_grs_gn",
        "rate_grs_oc",
        "rate_grs_an",
        "rate_sls_oil",
        "rate_sls_con",
        "rate_sls_ga",
        "rate_sls_gn",
        "rate_sls_oc",
        "rate_sls_an",
    ],
    "cumulative_production": [
        "cprd_grs_oil",
        "cprd_grs_con",
        "cprd_grs_ga",
        "cprd_grs_gn",
        "cprd_grs_oc",
        "cprd_grs_an",
        "cprd_sls_oil",
        "cprd_sls_con",
        "cprd_sls_ga",
        "cprd_sls_gn",
        "cprd_sls_oc",
        "cprd_sls_an",
    ],
    "discrepancy": [
        "dcpy_um_oil",
        "dcpy_um_con",
        "dcpy_um_ga",
        "dcpy_um_gn",
        "dcpy_um_oc",
        "dcpy_um_an",
        "dcpy_ppa_oil",
        "dcpy_ppa_con",
        "dcpy_ppa_ga",
        "dcpy_ppa_gn",
        "dcpy_ppa_oc",
        "dcpy_ppa_an",
        "dcpy_wi_oil",
        "dcpy_wi_con",
        "dcpy_wi_ga",
        "dcpy_wi_gn",
        "dcpy_wi_oc",
        "dcpy_wi_an",
        "dcpy_cio_oil",
        "dcpy_cio_con",
        "dcpy_cio_ga",
        "dcpy_cio_gn",
        "dcpy_cio_oc",
        "dcpy_cio_an",
        "dcpy_gtr_oil",
        "dcpy_gtr_con",
        "dcpy_gtr_ga",
        "dcpy_gtr_gn",
        "dcpy_gtr_oc",
        "dcpy_gtr_an",
        "dcpy_uc_oil",
        "dcpy_uc_con",
        "dcpy_uc_ga",
        "dcpy_uc_gn",
        "dcpy_uc_oc",
        "dcpy_uc_an",
    ],
    "identification": [
        "id",
        "uuid",
        "project_id",
        "project_name",
        "project_name_previous",
        "field_id",
        "field_name",
        "field_name_previous",
        "wk_id",
        "wk_name",
        "operator_name",
        "operator_group",
    ],
    "location": [
        "province",
        "prov_id",
        "basin86",
        "basin86_id",
        "basin128",
        "basin128_id",
        "wk_lat",
        "wk_long",
        "wk_area",
        "wk_area_ori",
        "wk_area_latest",
        "field_lat",
        "field_long",
        "is_offshore",
    ],
    "contract": [
        "psc_eff_start",
        "psc_eff_end",
        "wk_subgroup",
        "wk_area_perwakilan_skkmigas",
        "wk_regionisasi_ngi",
        "pod_letter_num",
        "pod_name",
        "pod_date",
    ],
    "classification": [
        "project_stage",
        "project_class",
        "project_level",
        "project_level_previous",
        "uncert_level",
        "prod_stage",
        "project_eol",
        "project_isactive",
        "is_discovered",
        "is_ltp",
        "is_pse_approved",
        "is_pod_approved",
        "is_report_accepted",
        "is_unitization",
        "onstream_year",
        "onstream_actual",
    ],
    "report_metadata": [
        "report_date",
        "report_year",
        "report_status",
        "project_remarks",
        "vol_remarks",
    ],
    "other": [
        "pd",
        "ghv_avg",
        "gcf_srock",
        "gcf_res",
        "gcf_ts",
        "gcf_dyn",
        "gcf_total",
        "groovy_isactive",
        "fusion_isactive",
    ],
    # Timeseries forecast columns
    "timeseries_forecast_tpf": [
        "tpf_oil",
        "tpf_con",
        "tpf_ga",
        "tpf_gn",
        "tpf_oc",
        "tpf_an",
    ],
    "timeseries_forecast_tpf_risked": [
        "tpf_risked_oil",
        "tpf_risked_con",
        "tpf_risked_ga",
        "tpf_risked_gn",
        "tpf_risked_oc",
        "tpf_risked_an",
    ],
    "timeseries_forecast_slf": [
        "slf_oil",
        "slf_con",
        "slf_ga",
        "slf_gn",
        "slf_oc",
        "slf_an",
    ],
    "timeseries_forecast_spf": [
        "spf_oil",
        "spf_con",
        "spf_ga",
        "spf_gn",
        "spf_oc",
        "spf_an",
    ],
    "timeseries_forecast_crf": [
        "crf_oil",
        "crf_con",
        "crf_ga",
        "crf_gn",
        "crf_oc",
        "crf_an",
    ],
    "timeseries_forecast_prf": [
        "prf_oil",
        "prf_con",
        "prf_ga",
        "prf_gn",
        "prf_oc",
        "prf_an",
    ],
    "timeseries_forecast_ciof": [
        "ciof_oil",
        "ciof_con",
        "ciof_ga",
        "ciof_gn",
        "ciof_oc",
        "ciof_an",
    ],
    "timeseries_forecast_lossf": [
        "lossf_oil",
        "lossf_con",
        "lossf_ga",
        "lossf_gn",
        "lossf_oc",
        "lossf_an",
    ],
    "timeseries_historical_cumulative": [
        "cprd_grs_oil",
        "cprd_grs_con",
        "cprd_grs_ga",
        "cprd_grs_gn",
        "cprd_grs_oc",
        "cprd_grs_an",
        "cprd_sls_oil",
        "cprd_sls_con",
        "cprd_sls_ga",
        "cprd_sls_gn",
        "cprd_sls_oc",
        "cprd_sls_an",
    ],
    "timeseries_historical_rate": [
        "rate_oil",
        "rate_con",
        "rate_ga",
        "rate_gn",
        "rate_oc",
        "rate_an",
    ],
}


DOMAIN_CONCEPTS: Dict[str, Dict] = {
    "uncertainty_levels": {
        "1P": {
            "db_value": "1. Low Value",
            "description": "Proven reserves - P90 confidence",
        },
        "1R": {
            "db_value": "1. Low Value",
            "description": "Low estimate GRR - P90 confidence",
        },
        "1C": {
            "db_value": "1. Low Value",
            "description": "Low estimate Contingent Resources",
        },
        "1U": {
            "db_value": "1. Low Value",
            "description": "Low estimate Prospective Resources",
        },
        "2P": {
            "db_value": "2. Middle Value",
            "description": "Proven + Probable reserves - P50 confidence",
        },
        "2R": {
            "db_value": "2. Middle Value",
            "description": "Best estimate GRR - P50 confidence",
        },
        "2C": {
            "db_value": "2. Middle Value",
            "description": "Best estimate Contingent Resources",
        },
        "2U": {
            "db_value": "2. Middle Value",
            "description": "Best estimate Prospective Resources",
        },
        "3P": {
            "db_value": "3. High Value",
            "description": "Proven plus Probable plus Possible reserves - P10 confidence",
        },
        "3R": {
            "db_value": "3. High Value",
            "description": "High estimate GRR - P10 confidence",
        },
        "3C": {
            "db_value": "3. High Value",
            "description": "High estimate Contingent Resources",
        },
        "3U": {
            "db_value": "3. High Value",
            "description": "High estimate Prospective Resources",
        },
        "proven": {"db_value": "1. Low Value", "description": "Proven/terbukti"},
        "probable": {
            "calculation": "Middle - Low",
            "description": "Probable/mungkin - difference between middle and low",
        },
        "possible": {
            "calculation": "High - Middle",
            "description": "Possible/harapan - difference between high and middle",
        },
    },
    "project_classes": {
        "reserves": {
            "db_value": None,
            "columns": ["res_*"],
            "description": "Commercial reserves only",
        },
        "grr": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*"],
            "description": "Government Recoverable Resources - reserves plus sales potential",
        },
        "contingent": {
            "db_value": "2. Contingent Resources",
            "columns": ["rec_*"],
            "description": "Contingent Resources - discovered but not commercial",
        },
        "prospective": {
            "db_value": "3. Prospective Resources",
            "columns": ["rec_*", "rec_*_risked"],
            "description": "Prospective Resources - undiscovered potential. Risked means the resources is multiplied by total_gcf column",
        },
        "sales potential": {
            "db_value": "1. Reserves & GRR",
            "columns": ["rec_*", "res_*"],
            "calculation": "rec_* - res_*",
            "description": "The difference between GRR and reserves",
        },
    },
    "volume_types": {
        "cadangan": {
            "columns": ["res_oc", "res_an"],
            "description": "Reserves - commercial volumes",
        },
        "sumber_daya": {
            "columns": ["rec_oc", "rec_an"],
            "description": "Resources - all recoverable volumes",
        },
        "inplace": {
            "columns": ["prj_ioip", "prj_igip"],
            "description": "Initial volumes in place in the project.",
        },
        "eur": {
            "columns": ["eur_res_*", "eur_rec_*"],
            "description": "Estimated Ultimate Recovery",
        },
    },
    "substances": {
        "oil": {"columns": ["*_oil"], "description": "Crude oil"},
        "condensate": {"columns": ["*_con"], "description": "Condensate"},
        "oil_condensate": {
            "columns": ["*_oc"],
            "description": "Oil + Condensate combined",
        },
        "associated_gas": {
            "columns": ["*_ga"],
            "description": "Associated gas",
        },
        "non_associated_gas": {
            "columns": ["*_gn"],
            "description": "Non-associated gas",
        },
        "total_gas": {
            "columns": ["*_an"],
            "description": "Associated + Non-associated gas combined",
        },
    },
}


SYNONYMS: Dict[str, str] = {
    "cadangan": "cadangan",
    "reserves": "cadangan",
    "reserve": "cadangan",
    "sumber daya": "sumber_daya",
    "resources": "sumber_daya",
    "resource": "sumber_daya",
    "sumberdaya": "sumber_daya",
    "terbukti": "proven",
    "proven": "proven",
    "mungkin": "probable",
    "probable": "probable",
    "harapan": "possible",
    "possible": "possible",
    "1p": "1P",
    "2p": "2P",
    "3p": "3P",
    "1r": "1R",
    "2r": "2R",
    "3r": "3R",
    "1c": "1C",
    "2c": "2C",
    "3c": "3C",
    "1u": "1U",
    "2u": "2U",
    "3u": "3U",
    "grr": "grr",
    "government recoverable": "grr",
    "contingent": "contingent",
    "kontijen": "contingent",
    "prospective": "prospective",
    "prospek": "prospective",
    "lapangan": "field",
    "field": "field",
    "wilayah kerja": "wk",
    "wk": "wk",
    "work area": "wk",
    "operator": "operator",
    "basin": "basin",
    "cekungan": "basin",
    "provinsi": "province",
    "province": "province",
    "minyak": "oil",
    "oil": "oil",
    "kondensat": "condensate",
    "condensate": "condensate",
    "gas": "total_gas",
    "inplace": "inplace",
    "in-place": "inplace",
    "in place": "inplace",
    "igip": "inplace",
    "ioip": "inplace",
    "eur": "eur",
    "estimated ultimate recovery": "eur",
    "recovery factor": "recovery_factor",
    "rf": "recovery_factor",
    "produksi": "production",
    "production": "production",
    # Timeseries synonyms
    "perkiraan": "forecast",
    "forecast": "forecast",
    "proyeksi": "forecast",
    "projection": "forecast",
    "produksi kedepan": "forecast",
    "future production": "forecast",
    "ramalan": "forecast",
    "puncak produksi": "peak_production",
    "peak production": "peak_production",
    "produksi maksimum": "peak_production",
    "maximum production": "peak_production",
    "plateau": "plateau",
    "mendatar": "plateau",
    "produksi stabil": "plateau",
    "stable production": "plateau",
    "decline": "decline",
    "penurunan": "decline",
    "produksi menurun": "decline",
    "falling production": "decline",
    "ramp up": "ramp_up",
    "peningkatan": "ramp_up",
    "naik produksi": "ramp_up",
    "increasing production": "ramp_up",
    "end of life": "eol",
    "eol": "eol",
    "masa akhir produksi": "eol",
    "last production year": "eol",
    "tahun akhir produksi": "eol",
    "onstream": "onstream",
    "on stream": "onstream",
    "mulai produksi": "onstream",
    "start production": "onstream",
    "tahun onstream": "onstream",
    "shut in": "shut_in",
    "penutupan sumur": "shut_in",
    "well shut": "shut_in",
    "timeseries": "timeseries",
    "time series": "timeseries",
    "time-series": "timeseries",
    "profil produksi": "timeseries",
    "production profile": "timeseries",
}


@dataclass
class UncertaintySpec:
    """Specification for an uncertainty level."""

    type: Literal["direct", "calculated"]
    db_value: Optional[str] = None
    calculation: Optional[str] = None
    is_cumulative: bool = False
    reserves_only: bool = False
    sql_template: Optional[str] = None
    description: str = ""


UNCERTAINTY_MAP: Dict[str, UncertaintySpec] = {
    "1P": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=False,
        description="Proven reserves - P90 confidence",
    ),
    "1R": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=False,
        description="Low estimate GRR - P90 confidence",
    ),
    "1C": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=False,
        description="Low estimate Contingent Resources",
    ),
    "1U": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=False,
        description="Low estimate Prospective Resources",
    ),
    "2P": UncertaintySpec(
        type="direct",
        db_value="2. Middle Value",
        is_cumulative=True,
        reserves_only=False,
        description="Proved + Probable reserves - P50 confidence",
    ),
    "2R": UncertaintySpec(
        type="direct",
        db_value="2. Middle Value",
        is_cumulative=True,
        reserves_only=False,
        description="Best estimate GRR - P50 confidence",
    ),
    "2C": UncertaintySpec(
        type="direct",
        db_value="2. Middle Value",
        is_cumulative=True,
        reserves_only=False,
        description="Best estimate Contingent Resources",
    ),
    "2U": UncertaintySpec(
        type="direct",
        db_value="2. Middle Value",
        is_cumulative=True,
        reserves_only=False,
        description="Best estimate Prospective Resources",
    ),
    "3P": UncertaintySpec(
        type="direct",
        db_value="3. High Value",
        is_cumulative=True,
        reserves_only=False,
        description="Proved + Probable + Possible reserves - P10 confidence",
    ),
    "3R": UncertaintySpec(
        type="direct",
        db_value="3. High Value",
        is_cumulative=True,
        reserves_only=False,
        description="High estimate GRR - P10 confidence",
    ),
    "3C": UncertaintySpec(
        type="direct",
        db_value="3. High Value",
        is_cumulative=True,
        reserves_only=False,
        description="High estimate Contingent Resources",
    ),
    "3U": UncertaintySpec(
        type="direct",
        db_value="3. High Value",
        is_cumulative=True,
        reserves_only=False,
        description="High estimate Prospective Resources",
    ),
    "PROVEN": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=True,
        description="Proven reserves - same as 1P, for reserves ONLY",
    ),
    "TERBUKTI": UncertaintySpec(
        type="direct",
        db_value="1. Low Value",
        is_cumulative=False,
        reserves_only=True,
        description="Proven reserves (Indonesian) - same as 1P",
    ),
    "PROBABLE": UncertaintySpec(
        type="calculated",
        calculation="2P - 1P",
        reserves_only=True,
        sql_template="""SUM(CASE WHEN uncert_level = '2. Middle Value' THEN {column} ELSE 0 END) -
    SUM(CASE WHEN uncert_level = '1. Low Value' THEN {column} ELSE 0 END)""",
        description="Probable reserves - INCREMENTAL volume between 1P and 2P, reserves ONLY",
    ),
    "MUNGKIN": UncertaintySpec(
        type="calculated",
        calculation="2P - 1P",
        reserves_only=True,
        sql_template="""SUM(CASE WHEN uncert_level = '2. Middle Value' THEN {column} ELSE 0 END) -
    SUM(CASE WHEN uncert_level = '1. Low Value' THEN {column} ELSE 0 END)""",
        description="Probable reserves (Indonesian) - same as probable",
    ),
    "POSSIBLE": UncertaintySpec(
        type="calculated",
        calculation="3P - 2P",
        reserves_only=True,
        sql_template="""SUM(CASE WHEN uncert_level = '3. High Value' THEN {column} ELSE 0 END) -
    SUM(CASE WHEN uncert_level = '2. Middle Value' THEN {column} ELSE 0 END)""",
        description="Possible reserves - INCREMENTAL volume between 2P and 3P, reserves ONLY",
    ),
    "HARAPAN": UncertaintySpec(
        type="calculated",
        calculation="3P - 2P",
        reserves_only=True,
        sql_template="""SUM(CASE WHEN uncert_level = '3. High Value' THEN {column} ELSE 0 END) -
    SUM(CASE WHEN uncert_level = '2. Middle Value' THEN {column} ELSE 0 END)""",
        description="Possible reserves (Indonesian) - same as possible",
    ),
}


@dataclass
class ColumnMetadata:
    column_name: str
    description: str
    units: Optional[str] = None
    category: Optional[str] = None
    volume_type: Optional[str] = None
    substance: Optional[str] = None


COLUMN_METADATA: Dict[str, ColumnMetadata] = {
    # Identification
    "id": ColumnMetadata("id", "Primary key (auto-increment)", None, "identification"),
    "uuid": ColumnMetadata(
        "uuid", "Unique identifier (UUID v4)", None, "identification"
    ),
    "project_id": ColumnMetadata(
        "project_id", "Project identifier from ESDC", None, "identification"
    ),
    "project_name": ColumnMetadata(
        "project_name", "Name of the project", None, "identification"
    ),
    "field_name": ColumnMetadata(
        "field_name", "Name of the field", None, "identification"
    ),
    "wk_name": ColumnMetadata("wk_name", "Work area name", None, "identification"),
    "operator_name": ColumnMetadata(
        "operator_name", "Operator company name", None, "identification"
    ),
    # Location
    "province": ColumnMetadata("province", "Province name", None, "location"),
    "basin128": ColumnMetadata(
        "basin128", "Basin name (128 classification)", None, "location"
    ),
    "is_offshore": ColumnMetadata(
        "is_offshore", "Whether location is offshore (0/1)", None, "location"
    ),
    # Classification
    "project_class": ColumnMetadata(
        "project_class",
        "Project classification (Reserves & GRR, Contingent Resources, Prospective Resources)",
        None,
        "classification",
    ),
    "uncert_level": ColumnMetadata(
        "uncert_level", "Uncertainty level (Low, Middle, High)", None, "classification"
    ),
    "project_level": ColumnMetadata(
        "project_level", "Project maturity level", None, "classification"
    ),
    "report_year": ColumnMetadata(
        "report_year", "Report year", None, "report_metadata"
    ),
    # Reserves columns
    "res_oil": ColumnMetadata(
        "res_oil", "Reserves oil", "MSTB", "reserves", "reserves", "oil"
    ),
    "res_con": ColumnMetadata(
        "res_con", "Reserves condensate", "MSTB", "reserves", "reserves", "condensate"
    ),
    "res_ga": ColumnMetadata(
        "res_ga",
        "Reserves associated gas",
        "BSCF",
        "reserves",
        "reserves",
        "associated_gas",
    ),
    "res_gn": ColumnMetadata(
        "res_gn",
        "Reserves non-associated gas",
        "BSCF",
        "reserves",
        "reserves",
        "non_associated_gas",
    ),
    "res_oc": ColumnMetadata(
        "res_oc",
        "Reserves oil + condensate",
        "MSTB",
        "reserves",
        "reserves",
        "oil_condensate",
    ),
    "res_an": ColumnMetadata(
        "res_an",
        "Reserves total gas (associated + non-associated)",
        "BSCF",
        "reserves",
        "reserves",
        "total_gas",
    ),
    # Resources columns
    "rec_oil": ColumnMetadata(
        "rec_oil", "Resources oil", "MSTB", "resources", "resources", "oil"
    ),
    "rec_con": ColumnMetadata(
        "rec_con",
        "Resources condensate",
        "MSTB",
        "resources",
        "resources",
        "condensate",
    ),
    "rec_ga": ColumnMetadata(
        "rec_ga",
        "Resources associated gas",
        "BSCF",
        "resources",
        "resources",
        "associated_gas",
    ),
    "rec_gn": ColumnMetadata(
        "rec_gn",
        "Resources non-associated gas",
        "BSCF",
        "resources",
        "resources",
        "non_associated_gas",
    ),
    "rec_oc": ColumnMetadata(
        "rec_oc",
        "Resources oil + condensate",
        "MSTB",
        "resources",
        "resources",
        "oil_condensate",
    ),
    "rec_an": ColumnMetadata(
        "rec_an", "Resources total gas", "BSCF", "resources", "resources", "total_gas"
    ),
    "rec_mboe": ColumnMetadata(
        "rec_mboe", "Resources (MBOE)", "MBOE", "resources", "resources", None
    ),
    # Risked Resources columns
    "rec_oil_risked": ColumnMetadata(
        "rec_oil_risked",
        "Risked Resources oil",
        "MSTB",
        "resources_risked",
        "resources",
        "oil",
    ),
    "rec_con_risked": ColumnMetadata(
        "rec_con_risked",
        "Risked Resources condensate",
        "MSTB",
        "resources_risked",
        "resources",
        "condensate",
    ),
    "rec_ga_risked": ColumnMetadata(
        "rec_ga_risked",
        "Risked Resources associated gas",
        "BSCF",
        "resources_risked",
        "resources",
        "associated_gas",
    ),
    "rec_gn_risked": ColumnMetadata(
        "rec_gn_risked",
        "Risked Resources non-associated gas",
        "BSCF",
        "resources_risked",
        "resources",
        "non_associated_gas",
    ),
    "rec_oc_risked": ColumnMetadata(
        "rec_oc_risked",
        "Risked Resources oil + condensate",
        "MSTB",
        "resources_risked",
        "resources",
        "oil_condensate",
    ),
    "rec_an_risked": ColumnMetadata(
        "rec_an_risked",
        "Risked Resources gas",
        "BSCF",
        "resources_risked",
        "resources",
        "total_gas",
    ),
    # In-Place columns
    "prj_ioip": ColumnMetadata(
        "prj_ioip", "Initial Oil In Place", "MSTB", "in_place", "inplace", "oil"
    ),
    "prj_igip": ColumnMetadata(
        "prj_igip", "Initial Gas In Place", "BSCF", "in_place", "inplace", "total_gas"
    ),
    # EUR columns
    "eur_res_oc": ColumnMetadata(
        "eur_res_oc",
        "EUR Reserves oil + condensate",
        "MSTB",
        "eur_reserves",
        "eur",
        "oil_condensate",
    ),
    "eur_res_an": ColumnMetadata(
        "eur_res_an", "EUR Reserves gas", "BSCF", "eur_reserves", "eur", "total_gas"
    ),
    "eur_rec_oc": ColumnMetadata(
        "eur_rec_oc",
        "EUR Resources oil + condensate",
        "MSTB",
        "eur_resources",
        "eur",
        "oil_condensate",
    ),
    "eur_rec_an": ColumnMetadata(
        "eur_rec_an", "EUR Resources gas", "BSCF", "eur_resources", "eur", "total_gas"
    ),
    # Recovery Factors
    "rf_oil": ColumnMetadata(
        "rf_oil", "Recovery factor oil", "fraction", "recovery_factors", None, "oil"
    ),
    "rf_gas": ColumnMetadata(
        "rf_gas",
        "Recovery factor gas",
        "fraction",
        "recovery_factors",
        None,
        "total_gas",
    ),
    # Production Rates
    "rate_grs_oc": ColumnMetadata(
        "rate_grs_oc",
        "Gross production rate oil + condensate",
        "MSTBY",
        "production_rates",
        "rate",
        "oil_condensate",
    ),
    "rate_grs_an": ColumnMetadata(
        "rate_grs_an",
        "Gross production rate gas",
        "BSCFY",
        "production_rates",
        "rate",
        "total_gas",
    ),
    # Cumulative Production
    "cprd_grs_oc": ColumnMetadata(
        "cprd_grs_oc",
        "Cumulative gross production oil + condensate",
        "MSTB",
        "cumulative_production",
        "production",
        "oil_condensate",
    ),
    "cprd_grs_an": ColumnMetadata(
        "cprd_grs_an",
        "Cumulative gross production gas",
        "BSCF",
        "cumulative_production",
        "production",
        "total_gas",
    ),
    # Timeseries Forecast Columns - TPF (Total Potential Forecast)
    "tpf_oil": ColumnMetadata(
        "tpf_oil",
        "Total Potential Forecast oil",
        "MSTB",
        "timeseries_forecast_tpf",
        "forecast",
        "oil",
    ),
    "tpf_con": ColumnMetadata(
        "tpf_con",
        "Total Potential Forecast condensate",
        "MSTB",
        "timeseries_forecast_tpf",
        "forecast",
        "condensate",
    ),
    "tpf_ga": ColumnMetadata(
        "tpf_ga",
        "Total Potential Forecast associated gas",
        "BSCF",
        "timeseries_forecast_tpf",
        "forecast",
        "associated_gas",
    ),
    "tpf_gn": ColumnMetadata(
        "tpf_gn",
        "Total Potential Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_tpf",
        "forecast",
        "non_associated_gas",
    ),
    "tpf_oc": ColumnMetadata(
        "tpf_oc",
        "Total Potential Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_tpf",
        "forecast",
        "oil_condensate",
    ),
    "tpf_an": ColumnMetadata(
        "tpf_an",
        "Total Potential Forecast total gas",
        "BSCF",
        "timeseries_forecast_tpf",
        "forecast",
        "total_gas",
    ),
    "tpf_risked_oil": ColumnMetadata(
        "tpf_risked_oil",
        "Risked Forecast oil",
        "MSTB",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "oil",
    ),
    "tpf_risked_con": ColumnMetadata(
        "tpf_risked_con",
        "Risked Forecast condensate",
        "MSTB",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "condensate",
    ),
    "tpf_risked_ga": ColumnMetadata(
        "tpf_risked_ga",
        "Risked Forecast associated gas",
        "BSCF",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "associated_gas",
    ),
    "tpf_risked_gn": ColumnMetadata(
        "tpf_risked_gn",
        "Risked Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "non_associated_gas",
    ),
    "tpf_risked_oc": ColumnMetadata(
        "tpf_risked_oc",
        "Risked Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "oil_condensate",
    ),
    "tpf_risked_an": ColumnMetadata(
        "tpf_risked_an",
        "Risked Forecast total gas",
        "BSCF",
        "timeseries_forecast_tpf_risked",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - SLF (Sales Forecast)
    "slf_oil": ColumnMetadata(
        "slf_oil",
        "Sales Forecast oil",
        "MSTB",
        "timeseries_forecast_slf",
        "forecast",
        "oil",
    ),
    "slf_con": ColumnMetadata(
        "slf_con",
        "Sales Forecast condensate",
        "MSTB",
        "timeseries_forecast_slf",
        "forecast",
        "condensate",
    ),
    "slf_ga": ColumnMetadata(
        "slf_ga",
        "Sales Forecast associated gas",
        "BSCF",
        "timeseries_forecast_slf",
        "forecast",
        "associated_gas",
    ),
    "slf_gn": ColumnMetadata(
        "slf_gn",
        "Sales Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_slf",
        "forecast",
        "non_associated_gas",
    ),
    "slf_oc": ColumnMetadata(
        "slf_oc",
        "Sales Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_slf",
        "forecast",
        "oil_condensate",
    ),
    "slf_an": ColumnMetadata(
        "slf_an",
        "Sales Forecast total gas",
        "BSCF",
        "timeseries_forecast_slf",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - SPF (Sales Potential Forecast)
    "spf_oil": ColumnMetadata(
        "spf_oil",
        "Sales Potential Forecast oil",
        "MSTB",
        "timeseries_forecast_spf",
        "forecast",
        "oil",
    ),
    "spf_con": ColumnMetadata(
        "spf_con",
        "Sales Potential Forecast condensate",
        "MSTB",
        "timeseries_forecast_spf",
        "forecast",
        "condensate",
    ),
    "spf_ga": ColumnMetadata(
        "spf_ga",
        "Sales Potential Forecast associated gas",
        "BSCF",
        "timeseries_forecast_spf",
        "forecast",
        "associated_gas",
    ),
    "spf_gn": ColumnMetadata(
        "spf_gn",
        "Sales Potential Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_spf",
        "forecast",
        "non_associated_gas",
    ),
    "spf_oc": ColumnMetadata(
        "spf_oc",
        "Sales Potential Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_spf",
        "forecast",
        "oil_condensate",
    ),
    "spf_an": ColumnMetadata(
        "spf_an",
        "Sales Potential Forecast total gas",
        "BSCF",
        "timeseries_forecast_spf",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - CRF (Contingent Resources Forecast)
    "crf_oil": ColumnMetadata(
        "crf_oil",
        "Contingent Resources Forecast oil",
        "MSTB",
        "timeseries_forecast_crf",
        "forecast",
        "oil",
    ),
    "crf_con": ColumnMetadata(
        "crf_con",
        "Contingent Resources Forecast condensate",
        "MSTB",
        "timeseries_forecast_crf",
        "forecast",
        "condensate",
    ),
    "crf_ga": ColumnMetadata(
        "crf_ga",
        "Contingent Resources Forecast associated gas",
        "BSCF",
        "timeseries_forecast_crf",
        "forecast",
        "associated_gas",
    ),
    "crf_gn": ColumnMetadata(
        "crf_gn",
        "Contingent Resources Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_crf",
        "forecast",
        "non_associated_gas",
    ),
    "crf_oc": ColumnMetadata(
        "crf_oc",
        "Contingent Resources Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_crf",
        "forecast",
        "oil_condensate",
    ),
    "crf_an": ColumnMetadata(
        "crf_an",
        "Contingent Resources Forecast total gas",
        "BSCF",
        "timeseries_forecast_crf",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - PRF (Prospective Resources Forecast)
    "prf_oil": ColumnMetadata(
        "prf_oil",
        "Prospective Resources Forecast oil",
        "MSTB",
        "timeseries_forecast_prf",
        "forecast",
        "oil",
    ),
    "prf_con": ColumnMetadata(
        "prf_con",
        "Prospective Resources Forecast condensate",
        "MSTB",
        "timeseries_forecast_prf",
        "forecast",
        "condensate",
    ),
    "prf_ga": ColumnMetadata(
        "prf_ga",
        "Prospective Resources Forecast associated gas",
        "BSCF",
        "timeseries_forecast_prf",
        "forecast",
        "associated_gas",
    ),
    "prf_gn": ColumnMetadata(
        "prf_gn",
        "Prospective Resources Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_prf",
        "forecast",
        "non_associated_gas",
    ),
    "prf_oc": ColumnMetadata(
        "prf_oc",
        "Prospective Resources Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_prf",
        "forecast",
        "oil_condensate",
    ),
    "prf_an": ColumnMetadata(
        "prf_an",
        "Prospective Resources Forecast total gas",
        "BSCF",
        "timeseries_forecast_prf",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - CIOF (Consumed in Operation Forecast)
    "ciof_oil": ColumnMetadata(
        "ciof_oil",
        "Consumed in Operation Forecast oil",
        "MSTB",
        "timeseries_forecast_ciof",
        "forecast",
        "oil",
    ),
    "ciof_con": ColumnMetadata(
        "ciof_con",
        "Consumed in Operation Forecast condensate",
        "MSTB",
        "timeseries_forecast_ciof",
        "forecast",
        "condensate",
    ),
    "ciof_ga": ColumnMetadata(
        "ciof_ga",
        "Consumed in Operation Forecast associated gas",
        "BSCF",
        "timeseries_forecast_ciof",
        "forecast",
        "associated_gas",
    ),
    "ciof_gn": ColumnMetadata(
        "ciof_gn",
        "Consumed in Operation Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_ciof",
        "forecast",
        "non_associated_gas",
    ),
    "ciof_oc": ColumnMetadata(
        "ciof_oc",
        "Consumed in Operation Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_ciof",
        "forecast",
        "oil_condensate",
    ),
    "ciof_an": ColumnMetadata(
        "ciof_an",
        "Consumed in Operation Forecast total gas",
        "BSCF",
        "timeseries_forecast_ciof",
        "forecast",
        "total_gas",
    ),
    # Timeseries Forecast Columns - LOSSF (Loss Production Forecast)
    "lossf_oil": ColumnMetadata(
        "lossf_oil",
        "Loss Production Forecast oil",
        "MSTB",
        "timeseries_forecast_lossf",
        "forecast",
        "oil",
    ),
    "lossf_con": ColumnMetadata(
        "lossf_con",
        "Loss Production Forecast condensate",
        "MSTB",
        "timeseries_forecast_lossf",
        "forecast",
        "condensate",
    ),
    "lossf_ga": ColumnMetadata(
        "lossf_ga",
        "Loss Production Forecast associated gas",
        "BSCF",
        "timeseries_forecast_lossf",
        "forecast",
        "associated_gas",
    ),
    "lossf_gn": ColumnMetadata(
        "lossf_gn",
        "Loss Production Forecast non-associated gas",
        "BSCF",
        "timeseries_forecast_lossf",
        "forecast",
        "non_associated_gas",
    ),
    "lossf_oc": ColumnMetadata(
        "lossf_oc",
        "Loss Production Forecast oil + condensate",
        "MSTB",
        "timeseries_forecast_lossf",
        "forecast",
        "oil_condensate",
    ),
    "lossf_an": ColumnMetadata(
        "lossf_an",
        "Loss Production Forecast total gas",
        "BSCF",
        "timeseries_forecast_lossf",
        "forecast",
        "total_gas",
    ),
    # Timeseries Historical Columns - Cumulative Production
    "cprd_grs_oil": ColumnMetadata(
        "cprd_grs_oil",
        "Cumulative gross production oil",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "oil",
    ),
    "cprd_grs_con": ColumnMetadata(
        "cprd_grs_con",
        "Cumulative gross production condensate",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "condensate",
    ),
    "cprd_grs_ga": ColumnMetadata(
        "cprd_grs_ga",
        "Cumulative gross production associated gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "associated_gas",
    ),
    "cprd_grs_gn": ColumnMetadata(
        "cprd_grs_gn",
        "Cumulative gross production non-associated gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "non_associated_gas",
    ),
    "cprd_grs_oc": ColumnMetadata(
        "cprd_grs_oc",
        "Cumulative gross production oil + condensate",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "oil_condensate",
    ),
    "cprd_grs_an": ColumnMetadata(
        "cprd_grs_an",
        "Cumulative gross production total gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "total_gas",
    ),
    "cprd_sls_oil": ColumnMetadata(
        "cprd_sls_oil",
        "Cumulative sales oil",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "oil",
    ),
    "cprd_sls_con": ColumnMetadata(
        "cprd_sls_con",
        "Cumulative sales condensate",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "condensate",
    ),
    "cprd_sls_ga": ColumnMetadata(
        "cprd_sls_ga",
        "Cumulative sales associated gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "associated_gas",
    ),
    "cprd_sls_gn": ColumnMetadata(
        "cprd_sls_gn",
        "Cumulative sales non-associated gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "non_associated_gas",
    ),
    "cprd_sls_oc": ColumnMetadata(
        "cprd_sls_oc",
        "Cumulative sales oil + condensate",
        "MSTB",
        "timeseries_historical_cumulative",
        "cumulative",
        "oil_condensate",
    ),
    "cprd_sls_an": ColumnMetadata(
        "cprd_sls_an",
        "Cumulative sales total gas",
        "BSCF",
        "timeseries_historical_cumulative",
        "cumulative",
        "total_gas",
    ),
    # Timeseries Historical Columns - Production Rate
    "rate_oil": ColumnMetadata(
        "rate_oil",
        "Production rate oil",
        "MSTB/Y",
        "timeseries_historical_rate",
        "rate",
        "oil",
    ),
    "rate_con": ColumnMetadata(
        "rate_con",
        "Production rate condensate",
        "MSTB/Y",
        "timeseries_historical_rate",
        "rate",
        "condensate",
    ),
    "rate_ga": ColumnMetadata(
        "rate_ga",
        "Production rate associated gas",
        "BSCF/Y",
        "timeseries_historical_rate",
        "rate",
        "associated_gas",
    ),
    "rate_gn": ColumnMetadata(
        "rate_gn",
        "Production rate non-associated gas",
        "BSCF/Y",
        "timeseries_historical_rate",
        "rate",
        "non_associated_gas",
    ),
    "rate_oc": ColumnMetadata(
        "rate_oc",
        "Production rate oil + condensate",
        "MSTB/Y",
        "timeseries_historical_rate",
        "rate",
        "oil_condensate",
    ),
    "rate_an": ColumnMetadata(
        "rate_an",
        "Production rate total gas",
        "BSCF/Y",
        "timeseries_historical_rate",
        "rate",
        "total_gas",
    ),
}


def get_column_group(column_name: str) -> Optional[str]:
    """
    Get the group name for a column.

    Args:
        column_name: Name of the column

    Returns:
        Group name or None if not found
    """
    for group_name, columns in COLUMN_GROUPS.items():
        if column_name in columns:
            return group_name

        for col in columns:
            if col.endswith("*"):
                prefix = col[:-1]
                if column_name.startswith(prefix):
                    return group_name
    return None


def resolve_concept(term: str) -> Optional[Dict]:
    """
    Resolve a term to its domain concept.

    Args:
        term: Term to resolve (e.g., "cadangan", "2P", "GRR")

    Returns:
        Domain concept dictionary or None
    """
    normalized = term.lower().strip()

    if normalized in SYNONYMS:
        normalized = SYNONYMS[normalized]

    if normalized in DOMAIN_CONCEPTS["uncertainty_levels"]:
        return {
            "type": "uncertainty_level",
            **DOMAIN_CONCEPTS["uncertainty_levels"][normalized],
        }

    if normalized in DOMAIN_CONCEPTS["project_classes"]:
        return {
            "type": "project_class",
            **DOMAIN_CONCEPTS["project_classes"][normalized],
        }

    if normalized in DOMAIN_CONCEPTS["volume_types"]:
        return {"type": "volume_type", **DOMAIN_CONCEPTS["volume_types"][normalized]}

    if normalized in DOMAIN_CONCEPTS["substances"]:
        return {"type": "substance", **DOMAIN_CONCEPTS["substances"][normalized]}

    return None


def get_columns_for_concept(
    concept_type: str, concept_name: str, substance: Optional[str] = None
) -> List[str]:
    """
    Get the database columns for a domain concept.

    Args:
        concept_type: Type of concept (uncertainty_level, project_class, volume_type)
        concept_name: Name of the concept
        substance: Optional substance filter (oil, gas, etc.)

    Returns:
        List of column names
    """
    if concept_type == "volume_type":
        if concept_name in DOMAIN_CONCEPTS["volume_types"]:
            return DOMAIN_CONCEPTS["volume_types"][concept_name]["columns"]

    if concept_type == "project_class":
        if concept_name in DOMAIN_CONCEPTS["project_classes"]:
            return DOMAIN_CONCEPTS["project_classes"][concept_name]["columns"]

    return []


def build_sql_pattern(
    concept: str,
    location_filter: Optional[str] = None,
    uncertainty: Optional[str] = None,
    project_class: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build SQL query patterns for common queries.

    Args:
        concept: Domain concept (cadangan, sumber_daya, etc.)
        location_filter: Optional location filter field (field_name, wk_name, etc.)
        uncertainty: Uncertainty level (1P, 2P, 3P, etc.)
        project_class: Project class filter

    Returns:
        Dictionary with SQL patterns
    """
    patterns = {}

    concept_info = resolve_concept(concept)
    if not concept_info:
        return patterns

    if concept_info["type"] == "volume_type":
        columns = concept_info.get("columns", ["res_oc", "res_an"])

        base_query = f"""
        SELECT 
            SUM(pr.{columns[0]}) as oil_condensate,
            SUM(pr.{columns[1]}) as gas
        FROM project_resources pr
        WHERE pr.report_year = (SELECT MAX(report_year) FROM project_resources)
        """

        if location_filter:
            base_query += f"\n        AND pr.{location_filter} LIKE '%{{location}}%'"

        if uncertainty:
            if uncertainty.upper() in ["1P", "1R", "1C", "1U", "PROVEN"]:
                base_query += "\n        AND pr.uncert_level = '1. Low Value'"
            elif uncertainty.upper() in ["2P", "2R", "2C", "2U"]:
                base_query += "\n        AND pr.uncert_level = '2. Middle Value'"
            elif uncertainty.upper() in ["3P", "3R", "3C", "3U"]:
                base_query += "\n        AND pr.uncert_level = '3. High Value'"

        if project_class:
            if project_class.lower() in ["grr", "reserves_grr"]:
                base_query += "\n        AND pr.project_class = '1. Reserves & GRR'"
            elif project_class.lower() in ["contingent"]:
                base_query += (
                    "\n        AND pr.project_class = '2. Contingent Resources'"
                )
            elif project_class.lower() in ["prospective"]:
                base_query += (
                    "\n        AND pr.project_class = '3. Prospective Resources'"
                )

        patterns["base"] = base_query

    return patterns


def get_uncertainty_filter(uncertainty: str) -> str:
    """
    Get the database filter value for an uncertainty level.

    Args:
        uncertainty: Uncertainty level (1P, 2P, 3P, proven, probable, possible)

    Returns:
        Database filter value for uncert_level column

    Note:
        For "probable" and "possible", this returns the default "2. Middle Value".
        Use get_uncertainty_spec() for full specification including SQL templates.
    """
    spec = get_uncertainty_spec(uncertainty)
    if spec and spec.db_value:
        return spec.db_value
    return "2. Middle Value"


def get_uncertainty_spec(
    uncertainty: str, volume_type: Optional[str] = None
) -> Optional[UncertaintySpec]:
    """
    Get the full specification for an uncertainty level.

    This is the preferred function for building SQL queries as it provides
    complete information including whether the value is calculated or direct.

    Args:
        uncertainty: Uncertainty level (1P, 2P, 3P, proven, probable, possible)
        volume_type: Optional volume type for validation ("cadangan", "reserves", etc.)

    Returns:
        UncertaintySpec with full specification, or None if not found

    Raises:
        ValueError: If reserves_only=True and volume_type is not reserves

    Examples:
        >>> get_uncertainty_spec("2P")
        UncertaintySpec(type='direct', db_value='2. Middle Value', ...)

        >>> get_uncertainty_spec("probable")
        UncertaintySpec(type='calculated', calculation='2P - 1P', reserves_only=True, ...)

        >>> get_uncertainty_spec("probable", volume_type="contingent")
        ValueError: 'probable' only applies to Reserves (cadangan), not contingent
    """
    uncertainty = uncertainty.upper().strip()

    spec = UNCERTAINTY_MAP.get(uncertainty)

    if not spec:
        return None

    if spec.reserves_only and volume_type:
        volume_type = volume_type.lower().strip()
        valid_reserve_types = ["cadangan", "reserves", "reserve", "res_oc", "res_an"]
        if volume_type not in valid_reserve_types:
            raise ValueError(
                f"'{uncertainty.lower()}' only applies to Reserves (cadangan), not {volume_type}. "
                f"Use 1C/2C/3C for Contingent Resources or 1U/2U/3U for Prospective Resources."
            )

    return spec


def build_uncertainty_sql(
    uncertainty: str, column: str, table_alias: str = "pr"
) -> str:
    """
    Build the SQL expression for an uncertainty level.

    For direct values (1P, 2P, etc.): Returns a simple WHERE clause condition
    For calculated values (probable, possible): Returns a CASE statement expression

    Args:
        uncertainty: Uncertainty level (1P, 2P, 3P, probable, possible)
        column: Column name (e.g., "res_oc", "res_an")
        table_alias: Table alias (default: "pr")

    Returns:
        SQL expression string

    Examples:
        >>> build_uncertainty_sql("2P", "res_oc")
        "pr.res_oc"

        >>> build_uncertainty_sql("probable", "res_oc")
        "SUM(CASE WHEN uncert_level = '2. Middle Value' THEN pr.res_oc ELSE 0 END) -
         SUM(CASE WHEN uncert_level = '1. Low Value' THEN pr.res_oc ELSE 0 END)"
    """
    spec = get_uncertainty_spec(uncertainty, volume_type=column)

    if not spec:
        return f"{table_alias}.{column}"

    if spec.type == "direct":
        return f"{table_alias}.{column}"

    if spec.type == "calculated" and spec.sql_template:
        return spec.sql_template.format(column=f"{table_alias}.{column}")

    return f"{table_alias}.{column}"


def get_project_class_filter(project_class: str) -> Optional[str]:
    """
    Get the database filter value for a project class.

    Args:
        project_class: Project class (GRR, Contingent, Prospective)

    Returns:
        Database filter value for project_class column, or None if not found
    """
    project_class = project_class.lower().strip()

    mapping = {
        "grr": "1. Reserves & GRR",
        "reserves_grr": "1. Reserves & GRR",
        "reserves & grr": "1. Reserves & GRR",
        "contingent": "2. Contingent Resources",
        "contingent resources": "2. Contingent Resources",
        "kontijen": "2. Contingent Resources",
        "prospective": "3. Prospective Resources",
        "prospective resources": "3. Prospective Resources",
        "prospek": "3. Prospective Resources",
    }

    return mapping.get(project_class, None)


def get_columns_for_substance(substance: str) -> List[str]:
    """
    Get column suffixes for a substance type.

    Args:
        substance: Substance type (oil, gas, oil_condensate, total_gas)

    Returns:
        List of column suffixes
    """
    substance = substance.lower().strip()

    mapping = {
        "oil": ["_oil", "_oc"],
        "condensate": ["_con", "_oc"],
        "oil_condensate": ["_oc"],
        "associated_gas": ["_ga", "_an"],
        "non_associated_gas": ["_gn", "_an"],
        "total_gas": ["_an"],
        "gas": ["_an"],
    }

    return mapping.get(substance, [])


def format_response_value(value: Optional[float], unit: str) -> str:
    """
    Format a numeric value with units for display.

    Args:
        value: Numeric value (can be None)
        unit: Unit (MSTB, BSCF, etc.)

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"

    unit_names = {
        "MSTB": "Thousand Stock Tank Barrels",
        "BSCF": "Billion Standard Cubic Feet",
        "MBOE": "Thousand Barrels Oil Equivalent",
    }

    return f"{value:,.2f} {unit} ({unit_names.get(unit, unit)})"


def get_volume_columns(volume_type: str, is_risked: bool = False) -> Tuple[str, str]:
    """
    Get the primary volume columns for a volume type.

    Args:
        volume_type: Type of volume (cadangan, sumber_daya)
        is_risked: Whether to get risked columns

    Returns:
        Tuple of (oil_condensate_column, gas_column)
    """
    volume_type = volume_type.lower().strip()

    if volume_type in ["cadangan", "reserves", "reserve"]:
        return ("res_oc", "res_an")

    if volume_type in [
        "sumber_daya",
        "sumberdaya",
        "sumber daya",
        "resources",
        "resource",
    ]:
        if is_risked:
            return ("rec_oc_risked", "rec_an_risked")
        return ("rec_oc", "rec_an")

    if volume_type in ["grr"]:
        return ("rec_oc", "rec_an")

    if volume_type in ["eur_reserves", "eur reserves"]:
        return ("eur_res_oc", "eur_res_an")

    if volume_type in ["eur_resources", "eur resources", "eur"]:
        return ("eur_rec_oc", "eur_rec_an")

    return ("res_oc", "res_an")


# =============================================================================
# Query Building Functions
# =============================================================================


def build_aggregate_query(
    entity_type: str,
    entity_name: Optional[str] = None,
    volume_type: str = "cadangan",
    uncertainty: str = "2P",
    project_class: Optional[str] = None,
    use_view: bool = True,
) -> Dict[str, str]:
    """
    Build an optimized SQL query for aggregate-level data.

    Automatically selects the appropriate table/view based on entity_type
    and builds the correct SQL with uncertainty and project_class filters.

    Args:
        entity_type: Type of entity ("field", "work_area", "wa", "national", "nkri", "project")
        entity_name: Name to filter by (e.g., "Duri", "Rokan")
        volume_type: Type of volume ("cadangan", "sumber_daya", "eur")
        uncertainty: Uncertainty level ("1P", "2P", "3P", "probable", "possible")
        project_class: Optional project class filter ("grr", "contingent", "prospective")
        use_view: If True, use pre-aggregated views when possible

    Returns:
        Dict with 'sql' and 'table' keys

    Examples:
        >>> build_aggregate_query("field", "Duri", "cadangan", "2P")
        {'sql': 'SELECT res_oc, res_an FROM field_resources WHERE...', 'table': 'field_resources'}

        >>> build_aggregate_query("field", "Duri", "cadangan", "probable")
        {'sql': 'SELECT CASE WHEN...', 'table': 'field_resources'}
    """
    table = get_table_for_query(entity_type, require_detail=not use_view)
    oc_col, gas_col = get_volume_columns(volume_type)
    entity_col = get_entity_filter_column(entity_type, table)

    base_query = f"FROM {table}"
    filters = ["report_year = (SELECT MAX(report_year) FROM {table})"]

    if entity_col and entity_name:
        filters.append(f"{entity_col} LIKE '%{entity_name}%'")

    spec = get_uncertainty_spec(uncertainty, volume_type=volume_type)

    if spec and spec.type == "calculated":
        sql = f"""SELECT
    {build_uncertainty_sql(uncertainty, oc_col, table_alias=table.split("_")[0][:2])} as {uncertainty.lower()}_oc,
    {build_uncertainty_sql(uncertainty, gas_col, table_alias=table.split("_")[0][:2])} as {uncertainty.lower()}_an
{base_query}
WHERE {" AND ".join(filters)}"""
    else:
        uncert_filter = get_uncertainty_filter(uncertainty)
        filters.append(f"uncert_level = '{uncert_filter}'")

        if project_class:
            class_filter = get_project_class_filter(project_class)
            if class_filter:
                filters.append(f"project_class = '{class_filter}'")

        alias = table.split("_")[0][:2]
        sql = f"""SELECT SUM({alias}.{oc_col}) as oc, SUM({alias}.{gas_col}) as gas
{base_query} {alias}
WHERE {" AND ".join(filters)}"""

    return {"sql": sql.strip(), "table": table}


def get_aggregation_table_info() -> Dict[str, Dict]:
    """
    Get information about all tables/views in the aggregation hierarchy.

    Returns:
        Dict with table names as keys and metadata as values

    Examples:
        >>> info = get_aggregation_table_info()
        >>> info['field_resources']['level']
        2
        >>> info['nkri_resources']['entity_type']
        'national'
    """
    return {
        "project_resources": {
            "level": 1,
            "entity_type": "project",
            "description": "Base table with project-level data",
            "columns": ["all columns"],
            "use_for": [
                "project-specific queries",
                "detailed analysis",
                "drill-down from aggregates",
            ],
        },
        "field_resources": {
            "level": 2,
            "entity_type": "field",
            "description": "Aggregated by field",
            "columns": [
                "field_name",
                "wk_name",
                "res_*",
                "rec_*",
                "uncert_level",
                "project_class",
            ],
            "use_for": ["field-level totals", "field comparisons"],
            "filter_column": "field_name",
        },
        "wa_resources": {
            "level": 3,
            "entity_type": "work_area",
            "description": "Aggregated by work area",
            "columns": ["wk_name", "res_*", "rec_*", "uncert_level", "project_class"],
            "use_for": ["work area totals", "regional summaries"],
            "filter_column": "wk_name",
        },
        "nkri_resources": {
            "level": 4,
            "entity_type": "national",
            "description": "National level aggregation",
            "columns": ["res_*", "rec_*", "uncert_level", "project_class"],
            "use_for": ["national totals", "country-wide statistics"],
            "filter_column": None,
        },
        "field_timeseries": {
            "level": 5,
            "entity_type": "field_timeseries",
            "description": "Timeseries forecast data aggregated by field per year",
            "columns": [
                "field_name",
                "wk_name",
                "year",
                "tpf_*",
                "slf_*",
                "spf_*",
                "crf_*",
                "prf_*",
                "ciof_*",
                "lossf_*",
                "cprd_*",
                "rate_*",
            ],
            "use_for": [
                "field-level production forecasts",
                "peak production analysis",
                "production trends by field",
            ],
            "filter_column": "field_name",
        },
        "wa_timeseries": {
            "level": 6,
            "entity_type": "wa_timeseries",
            "description": "Timeseries forecast data aggregated by work area per year",
            "columns": [
                "wk_name",
                "year",
                "tpf_*",
                "slf_*",
                "spf_*",
                "crf_*",
                "prf_*",
                "ciof_*",
                "lossf_*",
                "cprd_*",
                "rate_*",
            ],
            "use_for": [
                "work area production forecasts",
                "regional production trends",
            ],
            "filter_column": "wk_name",
        },
        "nkri_timeseries": {
            "level": 7,
            "entity_type": "nkri_timeseries",
            "description": "National timeseries forecast data per year",
            "columns": [
                "year",
                "tpf_*",
                "slf_*",
                "spf_*",
                "crf_*",
                "prf_*",
                "ciof_*",
                "lossf_*",
                "cprd_*",
                "rate_*",
            ],
            "use_for": [
                "national production forecasts",
                "country-wide production trends",
                "national peak production analysis",
            ],
            "filter_column": None,
        },
    }


def get_recommended_table(
    entity_type: Optional[str] = None, query_needs_detail: bool = False
) -> str:
    """
    Get the recommended table for a query type.

    This is a helper function to guide the LLM in selecting tables.

    Args:
        entity_type: "field", "work_area", "national", or "project"
        query_needs_detail: True if query needs project-level details

    Returns:
        Recommended table name

    Examples:
        >>> get_recommended_table("field")
        'field_resources'

        >>> get_recommended_table("field", query_needs_detail=True)
        'project_resources'
    """
    if query_needs_detail:
        return "project_resources"

    entity_type = entity_type.lower().strip() if entity_type else ""

    table_map = {
        "project": "project_resources",
        "field": "field_resources",
        "lapangan": "field_resources",
        "work_area": "wa_resources",
        "wa": "wa_resources",
        "wilayah_kerja": "wa_resources",
        "national": "nkri_resources",
        "nkri": "nkri_resources",
        # Timeseries views
        "field_timeseries": "field_timeseries",
        "ts_field": "field_timeseries",
        "timeseries_field": "field_timeseries",
        "wa_timeseries": "wa_timeseries",
        "ts_wa": "wa_timeseries",
        "timeseries_wa": "wa_timeseries",
        "nkri_timeseries": "nkri_timeseries",
        "ts_nkri": "nkri_timeseries",
        "timeseries_nkri": "nkri_timeseries",
        "timeseries_national": "nkri_timeseries",
        # Timeseries keywords
        "forecast": "field_timeseries",
        "perkiraan": "field_timeseries",
        "proyeksi": "field_timeseries",
        "produksi kedepan": "field_timeseries",
        "peak_production": "field_timeseries",
        "puncak produksi": "field_timeseries",
        "trend": "field_timeseries",
        "timeseries": "field_timeseries",
        "time series": "field_timeseries",
        "profil produksi": "field_timeseries",
    }

    return table_map.get(entity_type, "project_resources")


# =============================================================================
# Timeseries-specific Functions
# =============================================================================


def get_peak_production_year_sql(
    table: str = "field_timeseries",
    entity_name: Optional[str] = None,
    substance: str = "oc",
) -> str:
    """
    Build SQL to find the year with peak production.

    Args:
        table: Table name (field_timeseries, wa_timeseries, nkri_timeseries)
        entity_name: Optional entity name to filter (e.g., "Duri")
        substance: Substance suffix (oil, con, ga, gn, oc, an)

    Returns:
        SQL query string

    Examples:
        >>> get_peak_production_year_sql("field_timeseries", "Duri", "oc")
        "SELECT year, tpf_oc FROM field_timeseries WHERE field_name LIKE '%Duri%' ORDER BY tpf_oc DESC LIMIT 1"
    """
    column = f"tpf_{substance}"
    entity_col = get_entity_filter_column(
        "field" if "field" in table else "work_area", table
    )

    sql = f"""SELECT
    year,
    {column} as peak_volume
FROM {table}
WHERE {column} = (
    SELECT MAX({column})
    FROM {table}
    {f"WHERE {entity_col} LIKE '%{entity_name}%'" if entity_name and entity_col else ""}
)"""

    if entity_name and entity_col:
        sql += f" AND {entity_col} LIKE '%{entity_name}%'"

    return sql


def get_last_production_year_sql(
    table: str = "field_timeseries",
    entity_name: Optional[str] = None,
    substance: str = "oc",
) -> str:
    """
    Build SQL to find the last year with production > 0 (EOL - End of Life).

    Args:
        table: Table name (field_timeseries, wa_timeseries, nkri_timeseries)
        entity_name: Optional entity name to filter
        substance: Substance suffix (oil, con, ga, gn, oc, an)

    Returns:
        SQL query string

    Examples:
        >>> get_last_production_year_sql("field_timeseries", "Duri", "oil")
        "SELECT MAX(year) FROM field_timeseries WHERE field_name LIKE '%Duri%' AND tpf_oil > 0"
    """
    column = f"tpf_{substance}"
    entity_col = get_entity_filter_column(
        "field" if "field" in table else "work_area", table
    )

    sql = f"""SELECT
    MAX(year) as last_production_year
FROM {table}
WHERE {column} > 0"""

    if entity_name and entity_col:
        sql += f" AND {entity_col} LIKE '%{entity_name}%'"

    return sql


def get_onstream_year_sql(
    table: str = "field_timeseries",
    entity_name: Optional[str] = None,
    substance: str = "oc",
) -> str:
    """
    Build SQL to find the onstream year.
    Uses onstream_year column if available, otherwise finds MIN(year) with production > 0.

    Args:
        table: Table name
        entity_name: Optional entity name to filter
        substance: Substance suffix

    Returns:
        SQL query string
    """
    column = f"tpf_{substance}"
    entity_col = get_entity_filter_column(
        "field" if "field" in table else "work_area", table
    )

    sql = f"""SELECT
    COALESCE(
        MIN(onstream_year),
        (SELECT MIN(year) FROM {table} AS t2 WHERE t2.{column} > 0 {f"AND t2.{entity_col} = t1.{entity_col}" if entity_col else ""})
    ) as onstream_year
FROM {table} AS t1
WHERE 1=1"""

    if entity_name and entity_col:
        sql += f" AND {entity_col} LIKE '%{entity_name}%'"

    return sql


def convert_volume_units(
    volume: float, from_unit: str, to_unit: str, year: int = 2024
) -> float:
    """
    Convert volume units for timeseries data.

    Args:
        volume: Volume value
        from_unit: Source unit (MSTB, BSCF, MSTBY, BSCFY)
        to_unit: Target unit (BOPD, MMSCFD, MSTBY, BSCFY)
        year: Year for daily calculation (for leap year detection)

    Returns:
        Converted volume value

    Examples:
        >>> convert_volume_units(1000, "MSTB", "BOPD", 2024)
        2732.24  # 1000 * 1000 / 366
    """
    days_in_year = 366 if year % 4 == 0 else 365

    conversions = {
        ("MSTB", "BOPD"): lambda x: x * 1000 / days_in_year,
        ("BSCF", "MMSCFD"): lambda x: x * 1000 / days_in_year,
        ("MSTBY", "BOPD"): lambda x: x / days_in_year,
        ("BSCFY", "MMSCFD"): lambda x: x / days_in_year,
        ("MSTB", "MSTBY"): lambda x: x,  # Same unit, different context
        ("BSCF", "BSCFY"): lambda x: x,
    }

    converter = conversions.get((from_unit, to_unit))
    if converter:
        return round(converter(volume), 2)

    return volume


def build_timeseries_query(
    table: str = "field_timeseries",
    entity_name: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
    substance: str = "oc",
    include_daily: bool = False,
) -> Dict[str, str]:
    """
    Build SQL query for timeseries data with optional unit conversion.

    Args:
        table: Table name (field_timeseries, wa_timeseries, nkri_timeseries)
        entity_name: Entity name to filter
        year: Specific year to query
        year_range: Tuple of (start_year, end_year)
        substance: Substance suffix (oil, con, ga, gn, oc, an)
        include_daily: Whether to include daily rate conversion (BOPD, MMSCFD)

    Returns:
        Dict with 'sql' and 'table' keys

    Examples:
        >>> build_timeseries_query("field_timeseries", "Duri", year=2025)
        {'sql': 'SELECT year, tpf_oc...', 'table': 'field_timeseries'}
    """
    entity_col = get_entity_filter_column(
        "field" if "field" in table else "work_area", table
    )
    forecast_col = f"tpf_{substance}"

    # Determine units
    is_gas = substance in ["ga", "gn", "an"]
    annual_unit = "BSCF" if is_gas else "MSTB"
    daily_unit = "MMSCFD" if is_gas else "BOPD"

    columns = [
        "year",
        f"{forecast_col} as forecast_{annual_unit.lower()}",
    ]

    if include_daily:
        columns.append(
            f"ROUND({forecast_col} * 1000 / CASE WHEN year % 4 = 0 THEN 366 ELSE 365 END, 2) as forecast_{daily_unit.lower()}"
        )

    columns_str = ", ".join(columns)

    sql = f"SELECT {columns_str} FROM {table}"
    filters = []

    if entity_name and entity_col:
        filters.append(f"{entity_col} LIKE '%{entity_name}%'")

    if year:
        filters.append(f"year = {year}")
    elif year_range:
        filters.append(f"year BETWEEN {year_range[0]} AND {year_range[1]}")

    if filters:
        sql += " WHERE " + " AND ".join(filters)

    sql += " ORDER BY year"

    return {"sql": sql, "table": table}


def format_timeseries_response(
    value: Optional[float],
    unit: str,
    year: Optional[int] = None,
    include_daily: bool = False,
) -> str:
    """
    Format timeseries volume with units and optional daily conversion.

    Args:
        value: Volume value
        unit: Unit (MSTB, BSCF)
        year: Year for daily calculation
        include_daily: Whether to include daily rate

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"

    result = format_response_value(value, unit)

    if include_daily and year:
        daily_value = convert_volume_units(
            value, unit, "BOPD" if unit == "MSTB" else "MMSCFD", year
        )
        daily_unit = "BOPD" if unit == "MSTB" else "MMSCFD"
        result += f" ({daily_value:,.2f} {daily_unit})"

    return result
