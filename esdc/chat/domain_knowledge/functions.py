"""Functions for domain knowledge mapping and SQL building."""

from typing import Any

from .concepts import DOMAIN_CONCEPTS
from .synonyms import SYNONYMS


def should_include_remarks(remarks_value: str | None) -> bool:
    """
    Determine if remarks should be shown to user based on content.

    Remarks are only shown if they contain meaningful information.

    Args:
        remarks_value: The remarks column value

    Returns:
        True if remarks should be displayed to user

    Examples:
        >>> should_include_remarks("Field under development planning")
        True
        >>> should_include_remarks("")
        False
        >>> should_include_remarks(None)
        False
    """
    if not remarks_value:
        return False

    # Skip generic/empty values
    meaningless_values = ["", "-", "null", "none", "n/a", "na"]
    return remarks_value.lower().strip() not in meaningless_values


def resolve_concept(term: str) -> dict | None:
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

    if normalized in DOMAIN_CONCEPTS["forecast_types"]:
        return {
            "type": "forecast_type",
            **DOMAIN_CONCEPTS["forecast_types"][normalized],
        }

    if normalized in DOMAIN_CONCEPTS["report_terms"]:
        return {"type": "report_term", **DOMAIN_CONCEPTS["report_terms"][normalized]}

    if normalized in DOMAIN_CONCEPTS["document_types"]:
        return {
            "type": "document_type",
            **DOMAIN_CONCEPTS["document_types"][normalized],
        }

    return None


def get_columns_for_concept(
    concept_type: str, concept_name: str, substance: str | None = None
) -> list[str]:
    """
    Get the database columns for a domain concept.

    Args:
        concept_type: Type of concept (uncertainty_level, project_class, volume_type)
        concept_name: Name of the concept
        substance: Optional substance filter (oil, gas, etc.)

    Returns:
        List of column names
    """
    if (
        concept_type == "volume_type"
        and concept_name in DOMAIN_CONCEPTS["volume_types"]
    ):
        return DOMAIN_CONCEPTS["volume_types"][concept_name]["columns"]

    if (
        concept_type == "project_class"
        and concept_name in DOMAIN_CONCEPTS["project_classes"]
    ):
        return DOMAIN_CONCEPTS["project_classes"][concept_name]["columns"]

    return []


def build_sql_pattern(
    concept: str,
    location_filter: str | None = None,
    uncertainty: str | None = None,
    project_class: str | None = None,
) -> dict[str, str]:
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


def get_project_class_filter(project_class: str) -> str | None:
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

    return mapping.get(project_class)


def get_columns_for_substance(substance: str) -> list[str]:
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


def format_response_value(value: float | None, unit: str) -> str:
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


def calculate_peak_production_year(
    table: str,
    entity_name: str | None = None,
    substance: str = "oc",
    forecast_type: str = "tpf",
) -> dict[str, str]:
    """
    Calculate the year with peak production from a timeseries table.

    This is a Python helper function that can be used with query results.
    For SQL generation, use get_peak_production_year_sql().

    Args:
        table: Table name (field_timeseries, wa_timeseries, nkri_timeseries,
        project_timeseries)
        entity_name: Optional entity name to filter
        substance: Substance suffix (oil, con, ga, gn, oc, an)
        forecast_type: Forecast type (tpf, slf, spf, crf, prf)

    Returns:
        Peak production year or None
    """
    # This function returns metadata about peak production calculation
    # Actual calculation would need to be done with query results
    column = f"{forecast_type}_{substance}"

    # Map table to entity type
    if "field" in table:
        entity_type = "field"
    elif "wa" in table:
        entity_type = "work_area"
    elif "nkri" in table:
        entity_type = "national"
    else:
        entity_type = "project"

    # Return calculation metadata
    return {
        "entity_type": entity_type,
        "column": column,
        "description": f"Year with maximum {forecast_type.upper()} for {substance}",
        "sql_hint": f"ORDER BY {column} DESC LIMIT 1",
    }


def calculate_eol_year(
    table: str,
    entity_name: str | None = None,
    substance: str = "oc",
    forecast_type: str = "tpf",
) -> dict[str, str]:
    """
    Calculate the End of Life (EOL) year from a timeseries table.

    EOL is defined as the last year with production > 0.

    Args:
        table: Table name
        entity_name: Optional entity name to filter
        substance: Substance suffix
        forecast_type: Forecast type

    Returns:
        EOL year or None
    """
    column = f"{forecast_type}_{substance}"

    return {
        "column": column,
        "description": "Last year with production > 0",
        "sql_hint": f"WHERE {column} > 0 ORDER BY year DESC LIMIT 1",
    }


def get_onstream_year(
    table: str,
    entity_name: str | None = None,
    substance: str = "oc",
) -> dict[str, str | None]:
    """
    Get the onstream (start production) year.

    Args:
        table: Table name
        entity_name: Optional entity name to filter
        substance: Substance suffix

    Returns:
        Onstream year or None
    """
    # Map table to entity type for filter column
    if "field" in table:
        filter_col = "field_name"
    elif "wa" in table:
        filter_col = "wk_name"
    else:
        filter_col = None

    return {
        "filter_column": filter_col,
        "description": "First year of production",
        "sql_hint": "MIN(year) WHERE production > 0",
    }


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
    entity_name: str | None = None,
    year: int | None = None,
    year_range: tuple[int, int] | None = None,
    substance: str = "oc",
    include_daily: bool = False,
) -> dict[str, str]:
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
    # Import here to avoid circular import
    from .tables import get_entity_filter_column

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
            f"ROUND({forecast_col} * 1000 / CASE WHEN year % 4 = 0 THEN 366 ELSE 365 END, 2) as forecast_{daily_unit.lower()}"  # noqa: E501
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
    value: float | None,
    unit: str,
    year: int | None = None,
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


def get_timeseries_columns(
    data_type: str = "forecast",
    forecast_type: str = "tpf",
    substance: str = "oc",
) -> dict[str, Any]:
    """Get the correct column names for timeseries queries.

    NOTE: Cumulative production columns (cprd_grs_*, cprd_sls_*)
    should always be queried from *_resources tables, NOT *_timeseries.
    If data_type is 'historical' or 'cumulative', this function returns
    a redirect message pointing to get_resources_columns() instead.
    """
    valid_data_types = ["forecast", "historical", "cumulative", "rate"]
    valid_forecast_types = ["tpf", "slf", "spf", "crf", "prf", "ciof", "lossf"]
    valid_substances = ["oil", "con", "ga", "gn", "oc", "an"]

    data_type = data_type.lower().strip()
    forecast_type = forecast_type.lower().strip()
    substance = substance.lower().strip()

    if data_type not in valid_data_types:
        return {
            "error": f"Invalid data_type '{data_type}'. Must be one of: {valid_data_types}",  # noqa: E501
            "column": None,
        }

    if data_type in ["historical", "cumulative"]:
        cprd_col = f"cprd_grs_{substance}"
        cprd_sls_col = f"cprd_sls_{substance}"
        return {
            "error": (
                f"Cumulative production columns ({cprd_col}, {cprd_sls_col}) "
                f"must ALWAYS be queried from *_resources tables, NOT *_timeseries. "
                f"Use get_resources_columns(volume_type='cumulative_production', "
                f"substance='{substance}') instead."
            ),
            "column": cprd_col,
            "tables": [
                "project_resources",
                "field_resources",
                "wa_resources",
                "nkri_resources",
            ],
            "category": "cumulative_production",
            "redirect_to": "get_resources_columns",
            "warning": (
                "NEVER query cprd_grs_* or cprd_sls_* from *_timeseries. "
                "Always use *_resources tables for cumulative production data."
            ),
            "data_type": data_type,
            "substance": substance,
        }

    if data_type == "forecast" and forecast_type not in valid_forecast_types:
        return {
            "error": f"Invalid forecast_type '{forecast_type}'. Must be one of: {valid_forecast_types}",  # noqa: E501
            "column": None,
        }

    if substance not in valid_substances:
        return {
            "error": f"Invalid substance '{substance}'. Must be one of: {valid_substances}",  # noqa: E501
            "column": None,
        }

    if data_type == "forecast":
        prefix = forecast_type
        category = "forecast"
        unit_suffix = ""
        descriptions = {
            "tpf": "Total Potential Forecast",
            "slf": "Sales Forecast (reserves forecast)",
            "spf": "Sales Potential Forecast (tpf - slf)",
            "crf": "Contingent Resources Forecast",
            "prf": "Prospective Resources Forecast",
            "ciof": "Consumed in Operation Forecast (Fuel, Flare, Shrinkage)",
            "lossf": "Loss Production Forecast",
        }
        description_base = descriptions.get(forecast_type, "Forecast")
    elif data_type == "rate":
        prefix = "rate"
        category = "rate"
        unit_suffix = "/Y"
        description_base = "Production rate"
    else:
        prefix = ""
        category = ""
        unit_suffix = ""
        description_base = ""

    substance_descriptions = {
        "oil": "oil",
        "con": "condensate",
        "ga": "associated gas",
        "gn": "non-associated gas",
        "oc": "oil + condensate",
        "an": "total gas",
    }

    is_gas = substance in ["ga", "gn", "an"]
    unit = "BSCF" + unit_suffix if is_gas else "MSTB" + unit_suffix

    if unit_suffix == "/Y":
        unit_description = (
            "Billion Standard Cubic Feet per Year"
            if is_gas
            else "Thousand Stock Tank Barrels per Year (RATE, not volume)"
        )
    else:
        unit_description = (
            "Billion Standard Cubic Feet (volume)"
            if is_gas
            else "Thousand Stock Tank Barrels (volume)"
        )

    column = f"{prefix}_{substance}"
    substance_desc = substance_descriptions.get(substance, substance)
    description = f"{description_base} for {substance_desc}"
    tables = [
        "project_timeseries",
        "field_timeseries",
        "wa_timeseries",
        "nkri_timeseries",
    ]

    warning = None
    incorrect_alternatives = []

    if category == "forecast":
        warning = f"Use {prefix}_* for forecast volumes. Do NOT use rate_* columns for forecasts."  # noqa: E501
        incorrect_alternatives = [
            f"rate_{substance}",
            f"rate_grs_{substance}",
            f"rate_sls_{substance}",
        ]
    elif category == "rate":
        warning = f"This is a RATE column ({unit}/Y), not a volume."
        incorrect_alternatives = [
            f"tpf_{substance}",
            f"slf_{substance}",
            f"cprd_grs_{substance}",
        ]

    examples = []
    if category == "forecast":
        examples = [
            "SELECT year, tpf_oc FROM field_timeseries WHERE field_name LIKE '%Duri%'",
            "SELECT SUM(tpf_oc) FROM project_timeseries WHERE year = 2025",
            "SELECT year, tpf_oc FROM field_timeseries WHERE field_name LIKE '%Duri%' ORDER BY tpf_oc DESC LIMIT 1",  # noqa: E501
        ]
    elif category == "rate":
        examples = [
            "SELECT year, rate_oc FROM project_timeseries WHERE project_name LIKE '%Duri%' AND year = 2024"  # noqa: E501
        ]

    return {
        "column": column,
        "description": description,
        "unit": unit,
        "unit_description": unit_description,
        "category": category,
        "tables": tables,
        "warning": warning,
        "incorrect_alternatives": incorrect_alternatives,
        "examples": examples,
        "data_type": data_type,
        "forecast_type": forecast_type if data_type == "forecast" else None,
        "substance": substance,
    }


def get_resources_columns(
    volume_type: str = "reserves",
    substance: str = "oc",
) -> dict[str, Any]:
    """Get the correct column names for static resource tables."""
    valid_volume_types = [
        "reserves",
        "resources",
        "risked",
        "cumulative_production",
        "cumulative_sales",
    ]
    valid_substances = ["oil", "con", "ga", "gn", "oc", "an"]

    volume_type = volume_type.lower().strip()
    substance = substance.lower().strip()

    if volume_type not in valid_volume_types:
        return {
            "error": f"Invalid volume_type '{volume_type}'. Must be one of: {valid_volume_types}",  # noqa: E501
            "column": None,
        }

    if substance not in valid_substances:
        return {
            "error": (
                f"Invalid substance '{substance}'. Must be one of: {valid_substances}"
            ),
            "column": None,
        }

    if volume_type == "reserves":
        prefix = "res"
        volume_desc = "Reserves"
        category = "reserves"
    elif volume_type == "resources":
        prefix = "rec"
        volume_desc = "Resources"
        category = "resources"
    elif volume_type == "cumulative_production":
        prefix = "cprd_grs"
        volume_desc = "Cumulative Gross Production"
        category = "cumulative_production"
    elif volume_type == "cumulative_sales":
        prefix = "cprd_sls"
        volume_desc = "Cumulative Sales Production"
        category = "cumulative_sales"
    else:
        prefix = "rec"
        volume_desc = "Risked Resources"
        category = "resources_risked"

    column = f"{prefix}_{substance}"
    if volume_type == "risked":
        column = f"rec_{substance}_risked"

    substance_descriptions = {
        "oil": "oil",
        "con": "condensate",
        "ga": "associated gas",
        "gn": "non-associated gas",
        "oc": "oil + condensate",
        "an": "total gas",
    }

    is_gas = substance in ["ga", "gn", "an"]
    unit = "BSCF" if is_gas else "MSTB"
    unit_description = (
        "Billion Standard Cubic Feet" if is_gas else "Thousand Stock Tank Barrels"
    )

    substance_desc = substance_descriptions.get(substance, substance)
    description = f"{volume_desc} for {substance_desc}"
    tables = ["project_resources", "field_resources", "wa_resources", "nkri_resources"]

    warning = None
    incorrect_alternatives = []

    if volume_type == "reserves":
        warning = "Use res_* columns for reserves. Do NOT use rec_* columns."
        incorrect_alternatives = [f"rec_{substance}", f"rec_{substance}_risked"]
    elif volume_type == "resources":
        warning = "Use rec_* columns for resources. Do NOT use res_* columns."
        incorrect_alternatives = [f"res_{substance}", f"rec_{substance}_risked"]
    elif volume_type == "risked":
        warning = "Use rec_*_risked for risked prospective resources."
        incorrect_alternatives = [f"res_{substance}", f"rec_{substance}"]
    elif volume_type in ["cumulative_production", "cumulative_sales"]:
        warning = (
            f"ALWAYS query {prefix}_* from *_resources tables. "
            f"NEVER use *_timeseries for cumulative production columns."
        )
        incorrect_alternatives = []

    examples = [
        f"SELECT SUM({column}) FROM field_resources WHERE field_name LIKE '%Duri%'",
        f"SELECT {column}, uncert_level FROM project_resources WHERE project_name LIKE '%Duri%'",  # noqa: E501
        f"SELECT SUM({column}) FROM nkri_resources WHERE uncert_level = '2. Middle Value'",  # noqa: E501
    ]

    return {
        "column": column,
        "description": description,
        "unit": unit,
        "unit_description": unit_description,
        "category": category,
        "tables": tables,
        "warning": warning,
        "incorrect_alternatives": incorrect_alternatives,
        "examples": examples,
        "volume_type": volume_type,
        "substance": substance,
    }


def get_aggregation_table_info() -> dict[str, dict]:
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
            "level": 3,
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
        "project_timeseries": {
            "level": 0,
            "entity_type": "project_timeseries",
            "description": "Project-level timeseries forecast data (detail per project per year)",  # noqa: E501
            "columns": [
                "project_id",
                "project_name",
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
                "project-specific production forecasts",
                "detail per project analysis",
                "comparing projects within same field",
            ],
            "filter_column": "project_name",
        },
    }


def get_use_case_sql_pattern(use_case: str) -> dict[str, Any]:
    """
    Get SQL patterns for common timeseries use cases.

    These are templates for common queries that the chat agent can use
    or adapt for user questions.

    Args:
        use_case: Type of use case. Options:
            - "peak_production": Find year with maximum production
            - "last_production_year": Find last year with production > 0
            - "onstream_year": Find first year of production
            - "forecast_volume": Get forecast volume with optional daily conversion
            - "production_trend": Get production trend over time range

    Returns:
        Dict with 'description', 'sql_template', and 'notes'

    Examples:
        >>> pattern = get_use_case_sql_pattern("peak_production")
        >>> print(pattern['description'])
        'Find the year with maximum production for a project/field'
    """
    patterns = {
        "peak_production": {
            "description": "Find the year with maximum production for a project/field",
            "sql_template": """
SELECT
    year,
    {forecast_col}
FROM {table}
WHERE {entity_col} LIKE '%{entity_name}%'
    AND {forecast_col} = (
        SELECT MAX({forecast_col})
        FROM {table}
        WHERE {entity_col} LIKE '%{entity_name}%'
    );
""",
            "notes": "Returns the year and volume at peak production. Use with tpf_oc or tpf_an columns.",  # noqa: E501
        },
        "last_production_year": {
            "description": "Find the last year where production is still expected",
            "sql_template": """
SELECT MAX(year) as last_production_year
FROM {table}
WHERE {entity_col} LIKE '%{entity_name}%'
    AND ({oil_col} > 0 OR {gas_col} > 0);
""",
            "notes": "Returns the final year of production (End of Life). Checks both oil and gas columns.",  # noqa: E501
        },
        "onstream_year": {
            "description": "Find the first year of production (onstream)",
            "sql_template": """
-- Option A: Use onstream_year column if available
SELECT DISTINCT
    COALESCE(onstream_year,
        (SELECT MIN(year)
         FROM {table} AS pt2
         WHERE pt2.{entity_col} = pt1.{entity_col}
         AND ({oil_col} > 0 OR {gas_col} > 0))) as onstream_year
FROM {table} AS pt1
WHERE {entity_col} LIKE '%{entity_name}%';
""",
            "notes": "Uses onstream_year column if available, otherwise finds first year with tpf > 0",  # noqa: E501
        },
        "forecast_volume": {
            "description": "Get forecast volume for a specific year with unit conversion",  # noqa: E501
            "sql_template": """
SELECT
    {entity_col},
    year,
    {oil_col} as forecast_oil_condensate_mstb,
    {gas_col} as forecast_gas_bscf,
    ROUND({oil_col} * 1000 / CASE
        WHEN year % 4 = 0 THEN 366
        ELSE 365
    END, 2) as forecast_oil_condensate_bopd,
    ROUND({gas_col} * 1000 / CASE
        WHEN year % 4 = 0 THEN 366
        ELSE 365
    END, 2) as forecast_gas_mmscfd
FROM {table}
WHERE {entity_col} LIKE '%{entity_name}%'
    AND year = {target_year};
""",
            "notes": "Returns MSTB/BSCF and converts to BOPD/MMSCFD. BOPD = MSTB * 1000 / days_in_year",  # noqa: E501
        },
        "production_trend": {
            "description": "Get production trend over a time range",
            "sql_template": """
SELECT
    year,
    {oil_col},
    {gas_col}
FROM {table}
WHERE {entity_col} LIKE '%{entity_name}%'
    AND year >= (SELECT MAX(report_year) FROM project_timeseries)
    AND year <= (SELECT MAX(report_year) FROM project_timeseries) + {years_forward}
ORDER BY year;
""",
            "notes": "Shows production trend from current report_year forward. Default is 5 years.",  # noqa: E501
        },
    }

    if use_case not in patterns:
        return {
            "error": f"Unknown use case '{use_case}'",
            "available_use_cases": list(patterns.keys()),
        }

    return patterns[use_case]


def get_forecast_vs_historical_guide() -> dict[str, Any]:
    """
    Get guidance on distinguishing forecast vs historical data.

    Returns:
        Dict with column mappings and decision logic for determining
        whether a query refers to historical or forecast data.
    """
    return {
        "historical": {
            "columns": [
                "rate_oil",
                "rate_con",
                "rate_ga",
                "rate_gn",
                "rate_oc",
                "rate_an",
            ],
            "description": "Production rates (rate_*) from timeseries tables",
            "data_type": "Actual/factual historical rate data",
            "year_logic": "year <= hist_year OR year <= report_year",
            "note": (
                "Cumulative production columns (cprd_grs_*, cprd_sls_*) are NOT "
                "included here. They must ALWAYS be queried from *_resources tables, "
                "NEVER from *_timeseries. Use get_resources_columns("
                "volume_type='cumulative_production') for those."
            ),
        },
        "forecast": {
            "columns": [
                "tpf_oil",
                "tpf_con",
                "tpf_ga",
                "tpf_gn",
                "tpf_oc",
                "tpf_an",
                "slf_oil",
                "slf_con",
                "slf_ga",
                "slf_gn",
                "slf_oc",
                "slf_an",
                "spf_oil",
                "spf_con",
                "spf_ga",
                "spf_gn",
                "spf_oc",
                "spf_an",
                "crf_oil",
                "crf_con",
                "crf_ga",
                "crf_gn",
                "crf_oc",
                "crf_an",
                "prf_oil",
                "prf_con",
                "prf_ga",
                "prf_gn",
                "prf_oc",
                "prf_an",
                "ciof_oil",
                "ciof_con",
                "ciof_ga",
                "ciof_gn",
                "ciof_oc",
                "ciof_an",
                "lossf_oil",
                "lossf_con",
                "lossf_ga",
                "lossf_gn",
                "lossf_oc",
                "lossf_an",
            ],
            "description": "Production forecasts and projections",
            "data_type": "Predicted/projected future data",
            "year_logic": "year > hist_year OR year > report_year",
        },
        "decision_rules": {
            "keywords_for_forecast": [
                "forecast",
                "perkiraan",
                "proyeksi",
                "projection",
                "produksi kedepan",
                "future production",
                "ramalan",
            ],
            "keywords_for_historical": [
                "produksi",
                "production",
                "historis",
                "historical",
                "kumulatif",
                "cumulative",
                "actual",
            ],
            "hint": "When in doubt, check if the requested year > report_year",
        },
        "user_guidance": {
            "always_include_report_year": True,
            "rationale": "report_year provides context - e.g., report_year 2023 may have forecast for 2024",  # noqa: E501
            "example": "Say: 'Forecast for 2025 (based on 2024 report)'",
        },
    }


def is_forecast_data(
    year: int, hist_year: int | None = None, report_year: int | None = None
) -> bool:
    """
    Determine if a given year represents forecast data.

    Args:
        year: The year to check
        hist_year: Historical cutoff year (optional)
        report_year: Report year from database (optional)

    Returns:
        True if year represents forecast data, False if historical

    Examples:
        >>> is_forecast_data(2025, hist_year=2024)
        True
        >>> is_forecast_data(2023, report_year=2024)
        False
    """
    return (hist_year is not None and year > hist_year) or (
        report_year is not None and year > report_year
    )


def get_volume_columns(
    volume_type: str, is_risked: bool = False, substance: str | None = None
) -> tuple[str, str]:
    """Get the column names for oil/condensate and gas volumes.

    Volume types and their column mappings:
    - cadangan/reserves → res_* (res_oc, res_an)
    - sumber_daya/grr/resources → rec_* (rec_oc, rec_an)
    - potensi → rec_* (all classified resources)
    - prospective → rec_*_risked (prospective resources only)
    - contingent → rec_* (contingent resources)

    Args:
        volume_type: Type of volume (cadangan, sumber_daya, grr, potensi,
        prospective, contingent)
        is_risked: Whether to return risked columns (only for prospective)
        substance: Specific substance if mentioned ("minyak", "oil", "gas",
        or None for combined)

    Returns:
        Tuple of (oil_condensate_column, gas_column)

    Examples:
        >>> get_volume_columns("cadangan")
        ('res_oc', 'res_an')
        >>> get_volume_columns("cadangan", substance="minyak")
        ('res_oil', 'res_con')
        >>> get_volume_columns("sumber_daya")
        ('rec_oc', 'rec_an')
        >>> get_volume_columns("potensi")  # All classified resources
        ('rec_oc', 'rec_an')
        >>> get_volume_columns("prospective")  # Prospective only - risked
        ('rec_oc_risked', 'rec_an_risked')
        >>> get_volume_columns("contingent")  # Contingent - not risked
        ('rec_oc', 'rec_an')
    """
    volume_type = volume_type.lower().strip()

    # Normalize volume type
    if volume_type in ("cadangan", "reserves", "reservoar"):
        prefix = "res"
    elif (
        volume_type in ("sumber_daya", "sumberdaya", "grr", "resources")
        or volume_type == "potensi"
    ):
        prefix = "rec"
        # Don't force is_risked - potensi can be any classification
    elif volume_type == "prospective":
        prefix = "rec"
        is_risked = True  # Force risked for prospective resources
    elif volume_type == "contingent":
        prefix = "rec"
        # Don't force risked - contingent uses regular rec_*
    else:
        raise ValueError(f"Unknown volume type: {volume_type}")

    # Handle risked suffix
    suffix = "_risked" if is_risked and prefix == "rec" else ""

    # Handle substance-specific columns
    if substance:
        substance = substance.lower().strip()
        if substance in ("minyak", "oil", "crude", "petroleum"):
            return (f"{prefix}_oil{suffix}", f"{prefix}_con{suffix}")
        elif substance in ("gas", "asso", "non-asso", "associated", "non-associated"):
            return (f"{prefix}_ga{suffix}", f"{prefix}_gn{suffix}")

    # Default: combined columns
    return (f"{prefix}_oc{suffix}", f"{prefix}_an{suffix}")


def detect_substance_from_query(query: str) -> str | None:
    """Detect if user mentioned a specific substance in their query.

    Args:
        query: User's query text (natural language)

    Returns:
        "oil" if minyak/oil mentioned, "gas" if gas mentioned, None if neither

    Examples:
        >>> detect_substance_from_query("berapa cadangan minyak lapangan duri?")
        'oil'
        >>> detect_substance_from_query("berapa cadangan gas lapangan duri?")
        'gas'
        >>> detect_substance_from_query("berapa cadangan lapangan duri?")
        None
    """
    query_lower = query.lower()

    # Oil indicators
    oil_terms = [
        "minyak",
        "oil",
        "crude",
        "petroleum",
        "mentah",
        "kondensat",
        "condensate",
    ]
    if any(term in query_lower for term in oil_terms):
        return "oil"

    # Gas indicators
    gas_terms = [
        "gas",
        "asso",
        "non-asso",
        "associated",
        "non-associated",
        "bscfd",
        "mmscfd",
    ]
    if any(term in query_lower for term in gas_terms):
        return "gas"

    return None


def detect_volume_type_from_query(query: str) -> str:
    """Detect volume type (cadangan/sumberdaya/potensi/prospective) from query.

    Distinguishes between:
    - "potensi" → uses rec_* columns (all classified resources)
    - "prospective" or "potensi eksplorasi" → uses rec_*_risked columns
    - "contingent" → uses rec_* columns with Contingent filter

    Args:
        query: User's query text (natural language)

    Returns:
        Volume type: "cadangan", "sumber_daya", "potensi", "prospective",
        or "contingent"

    Examples:
        >>> detect_volume_type_from_query("berapa cadangan lapangan duri?")
        'cadangan'
        >>> detect_volume_type_from_query("berapa potensi lapangan duri?")
        'potensi'
        >>> detect_volume_type_from_query("berapa potensi eksplorasi lapangan duri?")
        'prospective'
        >>> detect_volume_type_from_query("potensi prospective di wa X?")
        'prospective'
        >>> detect_volume_type_from_query("berapa potensi proyek X?")
        'sumber_daya'
    """
    query_lower = query.lower()

    # Special case: "potensi proyek" or "proyek-proyek" → regular resources (GRR)
    if any(
        term in query_lower for term in ["potensi proyek", "proyek-proyek", "proyek di"]
    ):
        return "sumber_daya"

    # Check for prospective resources explicitly
    # "potensi eksplorasi", "potensi prospective", "potensi prospek"
    prospective_indicators = [
        "potensi eksplorasi",
        "potensi prospective",
        "potensi prospek",
        "potensi prospektif",
    ]
    if any(term in query_lower for term in prospective_indicators):
        return "prospective"

    # Check for contingent resources explicitly
    if any(
        term in query_lower
        for term in ["potensi contingent", "potensi kontingen", "contingent"]
    ):
        return "contingent"

    # Check for reserves & GRR explicitly
    if any(term in query_lower for term in ["potensi grr", "potensi reserves", "grr"]):
        return "sumber_daya"

    # General "potensi" without classification → needs project_class filter
    if any(term in query_lower for term in ["potensi", "prospek"]):
        return "potensi"

    # Check for reserves (cadangan)
    if any(term in query_lower for term in ["cadangan", "reserves", "reservoar"]):
        return "cadangan"

    # Check for resources (sumberdaya)
    if any(term in query_lower for term in ["sumber daya", "sumberdaya", "resources"]):
        return "sumber_daya"

    # Default to cadangan
    return "cadangan"


def should_use_risked_columns(query: str, table: str) -> bool:
    """Determine if risked columns should be used based on query and table.

    Risked columns (rec_oc_risked, rec_an_risked) are used ONLY for:
    - Prospective Resources (explicitly: "potensi eksplorasi", "potensi prospective")
    - At aggregate level (field_resources, wa_resources, nkri_resources)

    NOT for:
    - General "potensi" queries (these use rec_* with project_class filter)
    - Contingent Resources (they use rec_*, where risked = unrisked)
    - Reserves & GRR (they use rec_*, where risked = unrisked)
    - Project level queries (project_resources has no risked columns)

    Args:
        query: User's query text
        table: Target table name

    Returns:
        True if risked columns should be used

    Examples:
        >>> should_use_risked_columns("potensi eksplorasi lapangan duri?",
        "field_resources")
        True
        >>> should_use_risked_columns("potensi lapangan duri?", "field_resources")
        False  # General potensi uses rec_*, not risked
        >>> should_use_risked_columns("potensi proyek X?", "project_resources")
        False
        >>> should_use_risked_columns("cadangan lapangan duri?", "field_resources")
        False
    """
    volume_type = detect_volume_type_from_query(query)

    # Only prospective queries use risked columns
    if volume_type != "prospective":
        return False

    # Only at aggregate level (not project level)
    aggregate_tables = [
        "field_resources",
        "wa_resources",
        "nkri_resources",
        "field_timeseries",
        "wa_timeseries",
        "nkri_timeseries",
    ]
    return table in aggregate_tables


def get_project_stage_filter(query: str) -> str | None:
    """Get project_stage SQL filter if eksplorasi/eksploitasi mentioned.

    Args:
        query: User's query text

    Returns:
        "Exploration" or "Exploitation" or None

    Examples:
        >>> get_project_stage_filter("berapa potensi eksplorasi lapangan duri?")
        'Exploration'
        >>> get_project_stage_filter("berapa cadangan lapangan duri?")
        None
    """
    query_lower = query.lower()

    if any(term in query_lower for term in ["eksplorasi", "exploration", "explorasi"]):
        return "Exploration"
    elif any(
        term in query_lower
        for term in ["eksploitasi", "exploitation", "pengembangan", "development"]
    ):
        return "Exploitation"

    return None


def get_available_report_year(
    table: str,
    target_year: int | None = None,
    entity_filter: str | None = None,
    max_fallback_years: int = 10,
) -> dict[str, Any]:
    """
    Get available report year with fallback logic.

    If target_year has no data, check previous years until data found.

    Args:
        table: Table name (field_resources, wa_resources, nkri_resources,
        project_resources)
        target_year: Desired report year (defaults to current year)
        entity_filter: Optional WHERE clause for entity (e.g.,
        "field_name LIKE '%Duri%'")
        max_fallback_years: Maximum years to check backwards (default 10)

    Returns:
        Dict with:
            - requested_year: Year originally requested
            - actual_year: Year to use (may differ due to fallback)
            - years_checked: List of years checked
            - has_data: Whether data exists
            - message: Human-readable message about fallback

    Examples:
        >>> result = get_available_report_year("field_resources", 2024)
        >>> result["years_checked"]
        [2024, 2023, 2022, ...]
    """
    from datetime import datetime

    # Default to current year if not specified
    if target_year is None:
        target_year = datetime.now().year

    # Validate table name to prevent SQL injection
    valid_tables = [
        "field_resources",
        "wa_resources",
        "nkri_resources",
        "project_resources",
        "field_timeseries",
        "wa_timeseries",
        "nkri_timeseries",
        "project_timeseries",
    ]
    if table not in valid_tables:
        raise ValueError(f"Invalid table: {table}. Must be one of {valid_tables}")

    years_checked = []

    # Calculate years to check
    for year in range(target_year, target_year - max_fallback_years - 1, -1):
        years_checked.append(year)

    # Return metadata about fallback strategy
    # The actual SQL execution happens in build_report_year_filter
    result = {
        "requested_year": target_year,
        "actual_year": None,  # Will be set by caller after SQL execution
        "years_checked": years_checked[: max_fallback_years + 1],
        "has_data": False,  # Will be set by caller
        "message": None,
        "fallback_needed": False,
    }

    return result


def build_report_year_filter(
    table: str,
    target_year: int | None = None,
    entity_filter: str | None = None,
    use_subquery: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    Build SQL WHERE clause for report_year with fallback logic.

    Args:
        table: Table name
        target_year: Desired report year (defaults to current year)
        entity_filter: Optional entity filter (e.g., "field_name LIKE '%Duri%'")
        use_subquery: Whether to use subquery approach (recommended for single queries)

    Returns:
        Tuple of (SQL WHERE clause, metadata dict)

    Examples:
        >>> sql_clause, meta = build_report_year_filter("field_resources", 2024)
        >>> print(sql_clause)
        report_year =
        (SELECT MAX(report_year) FROM field_resources WHERE report_year <= 2024)

        >>> sql_clause, meta = build_report_year_filter("project_resources",
        2024, "project_name LIKE '%X%'", use_subquery=False)
        >>> print(sql_clause)
        report_year <= 2024
    """
    metadata = get_available_report_year(table, target_year, entity_filter)

    if use_subquery:
        # Subquery approach: find max year <= target_year
        # This automatically falls back to the most recent available year
        target = metadata["requested_year"]
        subquery = f"SELECT MAX(report_year) FROM {table} WHERE report_year <= {target}"

        if entity_filter:
            subquery += f" AND {entity_filter}"

        where_clause = f"report_year = ({subquery})"
    else:
        # Simple filter: use <= target_year (caller handles fallback via ORDER BY/LIMIT)
        target = metadata["requested_year"]
        where_clause = f"report_year <= {target}"

    return where_clause, metadata


def detect_report_year_from_query(query: str) -> int | None:
    """
    Detect if user mentioned a specific report year in their query.

    Args:
        query: User's query text (natural language)

    Returns:
        Year as int if detected, None otherwise

    Examples:
        >>> detect_report_year_from_query("berapa cadangan tahun 2024?")
        2024
        >>> detect_report_year_from_query("data cadangan 2023?")
        2023
        >>> detect_report_year_from_query("cadangan lapangan duri?")
        None
    """
    import re

    # Match patterns like "tahun 2024", "2024", "tahun 24", "year 2024"
    patterns = [
        r"tahun\s*(\d{4})",  # "tahun 2024"
        r"year\s*(\d{4})",  # "year 2024"
        r"(?:^|\s)(\d{4})(?:\s|$)",  # standalone "2024"
        r"tahun\s*(\d{2})(?!\d)",  # "tahun 24" (2-digit year)
    ]

    query_clean = query.lower().strip()

    for pattern in patterns:
        match = re.search(pattern, query_clean)
        if match:
            year_str = match.group(1)
            year = int(year_str)

            # Handle 2-digit years (24 -> 2024)
            if year < 100:
                # Assume years 00-99 are 2000-2099
                year = 2000 + year

            # Validate year range (reasonable for oil & gas data)
            if 2000 <= year <= 2100:
                return year

    return None


def get_recommended_table(
    entity_type: str | None, query_needs_detail: bool = False
) -> str:
    """
    Get the recommended table for a query.

    Args:
        entity_type: Type of entity (field, work_area, national, project)
        query_needs_detail: Whether query needs project-level details

    Returns:
        Recommended table name
    """
    from .tables import get_table_for_query

    return get_table_for_query(entity_type, require_detail=query_needs_detail)


def build_aggregate_query(
    entity_type: str,
    entity_name: str | None,
    volume_type: str,
    uncertainty: str,
    project_class: str | None = None,
    use_view: bool = True,
    report_year: int | None = None,
) -> dict[str, Any]:
    """
    Build an aggregate query for resource data with automatic report year fallback.

    Args:
        entity_type: Type of entity (field, work_area, national)
        entity_name: Name of the entity (can be None for national)
        volume_type: Type of volume (cadangan, sumber_daya)
        uncertainty: Uncertainty level (1P, 2P, 3P, probable, etc.)
        project_class: Optional project class filter (grr, etc.)
        use_view: Whether to use aggregation views (default True)
        report_year: Optional target report year (defaults to MAX(report_year))

    Returns:
        Dict with 'sql', 'table', and 'report_year_info' keys

    Examples:
        >>> result = build_aggregate_query("field", "Duri", "cadangan", "2P")
        >>> result["table"]
        'field_resources'

        >>> result = build_aggregate_query("field", "Duri", "cadangan", "2P",
        report_year=2023)
        >>> "2023" in result["sql"]
        True
    """
    from .tables import (
        get_entity_filter_column,
    )

    # Get volume columns
    oc_col, gas_col = get_volume_columns(volume_type)

    # Determine table based on entity_type
    if entity_type == "field":
        table = "field_resources" if use_view else "project_resources"
    elif entity_type == "work_area":
        table = "wa_resources" if use_view else "project_resources"
    elif entity_type == "national":
        table = "nkri_resources"
    else:
        table = "project_resources"

    # Get filter column
    filter_col = get_entity_filter_column(entity_type, table) if entity_name else None

    # Handle calculated uncertainties (like probable = 2P - 1P)
    is_calculated = uncertainty.lower() == "probable"

    # Determine uncertainty filter
    uncert_filter = None
    if uncertainty.upper() == "1P" or uncertainty.lower() == "proven":
        uncert_filter = "pr.uncert_level = '1. Low Value'"
    elif uncertainty.upper() == "2P":
        # 2P = 1P + 2P
        uncert_filter = "pr.uncert_level IN ('1. Low Value', '2. Middle Value')"
    elif uncertainty.lower() == "probable":
        # probable = 2P - 1P, but we'll filter for both and use CASE WHEN
        uncert_filter = "pr.uncert_level IN ('1. Low Value', '2. Middle Value')"
    elif uncertainty.upper() == "3P":
        # 3P = 1P + 2P + 3P
        uncert_filter = (
            "pr.uncert_level IN ('1. Low Value', '2. Middle Value', '3. High Value')"
        )
    elif uncertainty.upper() == "1C":
        uncert_filter = "pr.uncert_level = '1. Low Value'"
    elif uncertainty.upper() == "2C":
        uncert_filter = "pr.uncert_level IN ('1. Low Value', '2. Middle Value')"
    elif uncertainty.upper() == "3C":
        uncert_filter = (
            "pr.uncert_level IN ('1. Low Value', '2. Middle Value', '3. High Value')"
        )

    # Build entity filter for report_year subquery
    entity_filter_sql = None
    if filter_col and entity_name:
        entity_filter_sql = f"pr.{filter_col} LIKE '%{entity_name}%'"

    # Build report_year filter with fallback
    report_year_where, report_year_info = build_report_year_filter(
        table, report_year, entity_filter_sql, use_subquery=True
    )

    # Build SQL with CASE WHEN for calculated uncertainties
    if is_calculated:
        sql = f"""SELECT
    SUM(CASE WHEN pr.uncert_level = '2. Middle Value' THEN pr.{oc_col} ELSE 0 END)
    - SUM(CASE WHEN pr.uncert_level = '1.
    Low Value' THEN pr.{oc_col} ELSE 0 END) as oil_condensate,
    SUM(CASE WHEN pr.uncert_level = '2. Middle Value' THEN pr.{gas_col} ELSE 0 END)
    - SUM(CASE WHEN pr.uncert_level = '1.
    Low Value' THEN pr.{gas_col} ELSE 0 END) as gas
FROM {table} pr
WHERE {report_year_where}"""
    else:
        sql = f"""SELECT
    SUM(pr.{oc_col}) as oil_condensate,
    SUM(pr.{gas_col}) as gas
FROM {table} pr
WHERE {report_year_where}"""

    # Add entity filter
    if filter_col and entity_name:
        sql += f"\n    AND pr.{filter_col} LIKE '%{entity_name}%'"

    # Add uncertainty filter
    if uncert_filter:
        sql += f"\n    AND {uncert_filter}"

    # Add project class filter
    if project_class:
        pc_filter = get_project_class_filter(project_class)
        if pc_filter:
            sql += f"\n    AND pr.project_class = '{pc_filter}'"

    return {
        "sql": sql,
        "table": table,
        "report_year_info": report_year_info,
    }
