"""Domain knowledge package for ESDC chat agent.

This package provides domain-specific knowledge mappings for Indonesian oil & gas
data, including table hierarchies, column definitions, and terminology mappings.

All public APIs from the original domain_knowledge.py are re-exported here for
backward compatibility.
"""

# =============================================================================
# Tables Module
# =============================================================================
from .tables import (
    TABLE_HIERARCHY,
    AGGREGATION_LEVELS,
    get_table_for_query,
    can_use_view_for_calculation,
    get_entity_filter_column,
)

# =============================================================================
# Columns Module
# =============================================================================
from .columns import (
    ColumnMetadata,
    COLUMN_GROUPS,
    COLUMN_METADATA,
    get_column_group,
)

# =============================================================================
# Concepts Module
# =============================================================================
from .concepts import DOMAIN_CONCEPTS

# =============================================================================
# Synonyms Module
# =============================================================================
from .synonyms import SYNONYMS

# =============================================================================
# Uncertainty Module
# =============================================================================
from .uncertainty import (
    UncertaintySpec,
    UNCERTAINTY_MAP,
    DB_VALUE_SYNONYMS,
    get_uncertainty_filter,
    get_uncertainty_spec,
    build_uncertainty_sql,
)

# =============================================================================
# Functions Module
# =============================================================================
from .functions import (
    resolve_concept,
    get_columns_for_concept,
    build_sql_pattern,
    get_project_class_filter,
    get_columns_for_substance,
    format_response_value,
    calculate_peak_production_year,
    calculate_eol_year,
    get_onstream_year,
    convert_volume_units,
    build_timeseries_query,
    format_timeseries_response,
    get_timeseries_columns,
    get_resources_columns,
    get_aggregation_table_info,
    get_use_case_sql_pattern,
    get_forecast_vs_historical_guide,
    is_forecast_data,
    get_volume_columns,
    build_aggregate_query,
    get_recommended_table,
)

# =============================================================================
# Problems Module
# =============================================================================
from .problems import (
    PROBLEM_CLUSTERS,
    PROBLEM_CLUSTER_CODE_PATTERN,
    PROBLEM_CLUSTER_PREFIX_PATTERNS,
    get_problem_cluster,
    get_all_problem_clusters,
    get_clusters_by_category,
    extract_problem_clusters,
    extract_problem_clusters_from_project,
    enrich_project_with_clusters,
    get_projects_by_cluster,
    get_cluster_summary,
    format_cluster_for_display,
    search_problem_clusters,
    get_cluster_explanation,
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
    "get_table_for_query",
    "can_use_view_for_calculation",
    "get_entity_filter_column",
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
    # Functions
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
]
