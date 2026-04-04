"""Table and view hierarchy definitions."""


# =============================================================================
# Table/View Hierarchy
# =============================================================================

TABLE_HIERARCHY: dict[str, str] = {
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
    # Timeseries base table (for detail per project)
    "project_timeseries": "project_timeseries",
    "project_forecast": "project_timeseries",
    "detail": "project_timeseries",
    # Timeseries views (for aggregation)
    "field_timeseries": "field_timeseries",
    "timeseries_field": "field_timeseries",
    "ts_field": "field_timeseries",
    "field_forecast": "field_timeseries",
    "lapangan_forecast": "field_timeseries",
    "wa_timeseries": "wa_timeseries",
    "timeseries_wa": "wa_timeseries",
    "ts_wa": "wa_timeseries",
    "wa_forecast": "wa_timeseries",
    "nkri_timeseries": "nkri_timeseries",
    "timeseries_nkri": "nkri_timeseries",
    "ts_nkri": "nkri_timeseries",
    "timeseries_national": "nkri_timeseries",
    "nkri_forecast": "nkri_timeseries",
    "nasional_forecast": "nkri_timeseries",
}

AGGREGATION_LEVELS: list[tuple[str, str, int]] = [
    ("project_resources", "project", 1),
    ("field_resources", "field", 2),
    ("wa_resources", "work_area", 3),
    ("nkri_resources", "national", 4),
    # Timeseries hierarchy: detail (0) -> aggregate (1-3)
    ("project_timeseries", "project_timeseries", 0),  # Detail level
    ("field_timeseries", "field_timeseries", 1),
    ("wa_timeseries", "wa_timeseries", 2),
    ("nkri_timeseries", "nkri_timeseries", 3),
]

# =============================================================================
# Context Enrichment Metadata
# =============================================================================

# Mapping table to their respective remarks columns
# Used for auto-including context information in queries
TABLE_REMARKS_COLUMNS: dict[str, str | None] = {
    "project_resources": "project_remarks",
    "field_resources": "field_remarks",
    "wa_resources": "wa_remarks",
    "nkri_resources": None,  # No remarks at national level
    "project_timeseries": "project_remarks",
    "field_timeseries": "field_remarks",
    "wa_timeseries": "wa_remarks",
    "nkri_timeseries": None,  # No remarks at national level
}

# Columns that require classification context when queried
# These columns must include project_class and project_stage for proper aggregation
REQUIRES_CLASSIFICATION_PREFIXES: tuple[str, ...] = (
    "rec_",  # Resources columns require classification
    "rec_",  # Risked resources also require classification
)

# Context columns that must be included for proper data interpretation
CLASSIFICATION_CONTEXT_COLUMNS: list[str] = ["project_class", "project_stage"]

# =============================================================================
# Table Categories for Enrichment
# =============================================================================

# Tables that need enrichment (detail level) - enrichment adds GROUP BY
DETAIL_TABLES: set[str] = {
    "project_resources",
    "project_timeseries",
}

# Tables that are pre-aggregated (skip enrichment) - data already grouped
AGGREGATE_VIEWS: set[str] = {
    "field_resources",
    "wa_resources",
    "nkri_resources",
    "field_timeseries",
    "wa_timeseries",
    "nkri_timeseries",
}


def get_remarks_column(table: str) -> str | None:
    """
    Get the remarks column name for a given table.

    Args:
        table: Table name (e.g., "field_resources", "project_timeseries")

    Returns:
        Remarks column name, or None if table has no remarks column

    Examples:
        >>> get_remarks_column("field_resources")
        'field_remarks'
        >>> get_remarks_column("nkri_resources")
        None
    """
    return TABLE_REMARKS_COLUMNS.get(table)


def requires_classification_columns(column: str) -> bool:
    """
    Check if a column requires classification context (project_class, project_stage).

    Resources columns (rec_*) require classification to prevent incorrect aggregation
    across different project classes (Reserves vs Contingent vs Prospective).

    Args:
        column: Column name to check

    Returns:
        True if column requires classification context

    Examples:
        >>> requires_classification_columns("rec_oc")
        True
        >>> requires_classification_columns("res_oc")
        False
        >>> requires_classification_columns("tpf_oc")
        False
    """
    column_lower = column.lower()
    return column_lower.startswith("rec_")


def get_classification_context_columns() -> list[str]:
    """
    Get the list of classification context columns.

    These columns should be included when querying resources data to enable
    proper aggregation by project class and stage.

    Returns:
        List of classification context column names

    Examples:
        >>> get_classification_context_columns()
        ['project_class', 'project_stage']
    """
    return CLASSIFICATION_CONTEXT_COLUMNS.copy()


def is_detail_table(table: str) -> bool:
    """
    Check if table is detail level (needs enrichment with GROUP BY).

    Detail tables contain raw project-level data that needs aggregation
    when querying across multiple projects.

    Args:
        table: Table name to check

    Returns:
        True if table is detail level and needs enrichment

    Examples:
        >>> is_detail_table("project_resources")
        True
        >>> is_detail_table("field_resources")
        False
    """
    return table in DETAIL_TABLES


def is_aggregate_view(table: str) -> bool:
    """
    Check if table is pre-aggregated view (skip enrichment, no GROUP BY needed).

    Aggregate views already contain summarized data grouped by entity
    (field, work_area, or national level).

    Args:
        table: Table name to check

    Returns:
        True if table is pre-aggregated view

    Examples:
        >>> is_aggregate_view("field_resources")
        True
        >>> is_aggregate_view("project_resources")
        False
    """
    return table in AGGREGATE_VIEWS


def get_table_for_query(
    entity_type: str | None = None,
    entity_name: str | None = None,
    require_detail: bool = False,
    timeseries_detail: bool = False,
    prefer_aggregation: bool = True,
) -> str:
    """
    Get the recommended table/view for a query.

    Uses pre-aggregated views for efficiency at field/work_area/national levels.
    For timeseries: uses views for aggregation, project_timeseries for detail per project.
    For static resources: uses views for aggregation, project_resources for detail per project.

    Args:
        entity_type: Type of entity (field, work_area, wa, national, nkri, project)
        entity_name: Name of specific entity
        require_detail: If True, use base table (project_resources) for detailed static analysis
        timeseries_detail: If True, use project_timeseries for detail; False for aggregated views
        prefer_aggregation: If True, prefer aggregated views over detail tables

    Returns:
        Table name to query

    Examples:
        >>> get_table_for_query("field")
        'field_resources'

        >>> get_table_for_query("work_area")
        'wa_resources'

        >>> get_table_for_query("field", require_detail=True)
        'project_resources'

        >>> get_table_for_query("field_timeseries")  # aggregation
        'field_timeseries'

        >>> get_table_for_query("field_timeseries", timeseries_detail=True)  # detail
        'project_timeseries'

        >>> get_table_for_query("field", prefer_aggregation=False)  # prefer detail
        'project_resources'
    """
    if timeseries_detail:
        return "project_timeseries"

    if require_detail:
        return "project_resources"

    if entity_type:
        entity_type = entity_type.lower().strip()
        table = TABLE_HIERARCHY.get(entity_type, "project_resources")

        # If prefer_aggregation is False and the result is an aggregation view,
        # return the detail table instead
        if not prefer_aggregation:
            aggregation_to_detail = {
                "field_resources": "project_resources",
                "wa_resources": "project_resources",
                "nkri_resources": "project_resources",
                "field_timeseries": "project_timeseries",
                "wa_timeseries": "project_timeseries",
                "nkri_timeseries": "project_timeseries",
            }
            return aggregation_to_detail.get(table, table)

        return table

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


def get_entity_filter_column(entity_type: str, table: str) -> str | None:
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
