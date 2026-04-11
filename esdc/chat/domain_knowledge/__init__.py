"""Domain knowledge package for ESDC chat agent.

This package provides domain-specific knowledge mappings for Indonesian oil & gas
data, including table hierarchies, column definitions, and terminology mappings.

All public APIs from the original domain_knowledge.py are re-exported here for
backward compatibility.
"""

# =============================================================================
# Tables Module
# =============================================================================
# =============================================================================
# Columns Module
# =============================================================================
from .columns import (
    COLUMN_GROUPS,
    COLUMN_METADATA,
    ColumnMetadata,
    get_column_group,
)

# =============================================================================
# Concepts Module
# =============================================================================
from .concepts import DOMAIN_CONCEPTS

# =============================================================================
# Functions Module
# =============================================================================
from .functions import (
    build_aggregate_query,
    build_report_year_filter,
    build_sql_pattern,
    build_timeseries_query,
    calculate_eol_year,
    calculate_peak_production_year,
    convert_volume_units,
    detect_report_year_from_query,
    detect_substance_from_query,
    detect_volume_type_from_query,
    enrich_sql_query,
    extract_selected_columns,
    extract_table_from_sql,
    format_response_value,
    format_timeseries_response,
    get_aggregation_table_info,
    get_available_report_year,
    get_columns_for_concept,
    get_columns_for_substance,
    get_forecast_vs_historical_guide,
    get_onstream_year,
    get_project_class_filter,
    get_project_stage_filter,
    get_recommended_table,
    get_resources_columns,
    get_timeseries_columns,
    get_use_case_sql_pattern,
    get_volume_columns,
    is_already_enriched,
    is_forecast_data,
    requires_classification_context,
    resolve_concept,
    should_include_remarks,
    should_use_risked_columns,
)

# =============================================================================
# Problems Module
# =============================================================================
from .problems import (
    PROBLEM_CLUSTER_CODE_PATTERN,
    PROBLEM_CLUSTER_PREFIX_PATTERNS,
    PROBLEM_CLUSTERS,
    enrich_project_with_clusters,
    extract_problem_clusters,
    extract_problem_clusters_from_project,
    format_cluster_for_display,
    get_all_problem_clusters,
    get_cluster_explanation,
    get_cluster_summary,
    get_clusters_by_category,
    get_problem_cluster,
    get_projects_by_cluster,
    search_problem_clusters,
)

# =============================================================================
# Synonyms Module
# =============================================================================
from .synonyms import SYNONYMS

# =============================================================================
# Tables Module
# =============================================================================
from .tables import (
    AGGREGATE_VIEWS,
    AGGREGATION_LEVELS,
    CLASSIFICATION_CONTEXT_COLUMNS,
    DETAIL_TABLES,
    REQUIRES_CLASSIFICATION_PREFIXES,
    TABLE_HIERARCHY,
    TABLE_REMARKS_COLUMNS,
    TABLE_VOL_REMARKS_COLUMNS,
    can_use_view_for_calculation,
    get_classification_context_columns,
    get_entity_filter_column,
    get_remarks_column,
    get_table_for_query,
    is_aggregate_view,
    is_detail_table,
    requires_classification_columns,
)

# =============================================================================
# Uncertainty Module
# =============================================================================
from .uncertainty import (
    DB_VALUE_SYNONYMS,
    UNCERTAINTY_MAP,
    UncertaintySpec,
    build_uncertainty_sql,
    get_uncertainty_filter,
    get_uncertainty_spec,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Problems
    "PROBLEM_CLUSTERS",
    "PROBLEM_CLUSTER_CODE_PATTERN",
    "PROBLEM_CLUSTER_PREFIX_PATTERNS",
    "get_problem_cluster",
    "get_all_problem_clusters",
    "get_clusters_by_category",
    "extract_problem_clusters",
    "extract_problem_clusters_from_project",
    "enrich_project_with_clusters",
    "get_projects_by_cluster",
    "get_cluster_summary",
    "format_cluster_for_display",
    "search_problem_clusters",
    "get_cluster_explanation",
    # Tables
    "TABLE_HIERARCHY",
    "AGGREGATION_LEVELS",
    "TABLE_REMARKS_COLUMNS",
    "TABLE_VOL_REMARKS_COLUMNS",
    "REQUIRES_CLASSIFICATION_PREFIXES",
    "CLASSIFICATION_CONTEXT_COLUMNS",
    "DETAIL_TABLES",
    "AGGREGATE_VIEWS",
    "get_table_for_query",
    "can_use_view_for_calculation",
    "get_entity_filter_column",
    "get_remarks_column",
    "requires_classification_columns",
    "get_classification_context_columns",
    "is_detail_table",
    "is_aggregate_view",
    # Columns
    "ColumnMetadata",
    "COLUMN_GROUPS",
    "COLUMN_METADATA",
    "get_column_group",
    # Concepts
    "DOMAIN_CONCEPTS",
    # Synonyms
    "SYNONYMS",
    # Uncertainty
    "UncertaintySpec",
    "UNCERTAINTY_MAP",
    "DB_VALUE_SYNONYMS",
    "get_uncertainty_filter",
    "get_uncertainty_spec",
    "build_uncertainty_sql",
    # Functions - Query Building
    "resolve_concept",
    "get_columns_for_concept",
    "build_sql_pattern",
    "get_project_class_filter",
    "get_columns_for_substance",
    "format_response_value",
    "calculate_peak_production_year",
    "calculate_eol_year",
    "get_onstream_year",
    "convert_volume_units",
    "build_timeseries_query",
    "format_timeseries_response",
    "get_timeseries_columns",
    "get_resources_columns",
    "get_aggregation_table_info",
    "get_use_case_sql_pattern",
    "get_forecast_vs_historical_guide",
    "is_forecast_data",
    "get_volume_columns",
    "build_aggregate_query",
    "get_recommended_table",
    # Functions - Query Enrichment (New)
    "enrich_sql_query",
    "extract_table_from_sql",
    "extract_selected_columns",
    "is_already_enriched",
    "requires_classification_context",
    "should_include_remarks",
    # Functions - Volume Column Helpers (New)
    "detect_substance_from_query",
    "detect_volume_type_from_query",
    "should_use_risked_columns",
    "get_project_stage_filter",
    # Functions - Report Year Fallback (New)
    "get_available_report_year",
    "build_report_year_filter",
    "detect_report_year_from_query",
]
