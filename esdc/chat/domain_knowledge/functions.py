"""Functions for domain knowledge mapping and SQL building."""

from typing import Dict, List, Optional, Tuple, Any

from .concepts import DOMAIN_CONCEPTS
from .synonyms import SYNONYMS


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

    if normalized in DOMAIN_CONCEPTS["forecast_types"]:
        return {
            "type": "forecast_type",
            **DOMAIN_CONCEPTS["forecast_types"][normalized],
        }

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


def calculate_peak_production_year(
    table: str,
    entity_name: Optional[str] = None,
    substance: str = "oc",
    forecast_type: str = "tpf",
) -> Dict[str, str]:
    """
    Calculate the year with peak production from a timeseries table.

    This is a Python helper function that can be used with query results.
    For SQL generation, use get_peak_production_year_sql().

    Args:
        table: Table name (field_timeseries, wa_timeseries, nkri_timeseries, project_timeseries)
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
    entity_name: Optional[str] = None,
    substance: str = "oc",
    forecast_type: str = "tpf",
) -> Dict[str, str]:
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
    entity_name: Optional[str] = None,
    substance: str = "oc",
) -> Dict[str, Optional[str]]:
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


def get_timeseries_columns(
    data_type: str = "forecast",
    forecast_type: str = "tpf",
    substance: str = "oc",
) -> Dict[str, Any]:
    """
    Get the correct column names for timeseries queries.
    """
    valid_data_types = ["forecast", "historical", "cumulative", "rate"]
    valid_forecast_types = ["tpf", "slf", "spf", "crf", "prf", "ciof", "lossf"]
    valid_substances = ["oil", "con", "ga", "gn", "oc", "an"]

    data_type = data_type.lower().strip()
    forecast_type = forecast_type.lower().strip()
    substance = substance.lower().strip()

    if data_type not in valid_data_types:
        return {
            "error": f"Invalid data_type '{data_type}'. Must be one of: {valid_data_types}",
            "column": None,
        }

    if data_type == "forecast" and forecast_type not in valid_forecast_types:
        return {
            "error": f"Invalid forecast_type '{forecast_type}'. Must be one of: {valid_forecast_types}",
            "column": None,
        }

    if substance not in valid_substances:
        return {
            "error": f"Invalid substance '{substance}'. Must be one of: {valid_substances}",
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
    elif data_type in ["historical", "cumulative"]:
        prefix = "cprd_grs"
        category = "historical"
        unit_suffix = ""
        description_base = "Cumulative gross production"
    elif data_type == "rate":
        prefix = "rate"
        category = "rate"
        unit_suffix = "/Y"
        description_base = "Production rate"

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
        warning = f"Use {prefix}_* for forecast volumes. Do NOT use rate_* columns for forecasts."
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
            "SELECT year, tpf_oc FROM field_timeseries WHERE field_name LIKE '%Duri%' ORDER BY tpf_oc DESC LIMIT 1",
        ]
    elif category == "historical":
        examples = [
            "SELECT MAX(cprd_grs_oc) FROM field_timeseries WHERE field_name LIKE '%Duri%'"
        ]
    elif category == "rate":
        examples = [
            "SELECT year, rate_oc FROM project_timeseries WHERE project_name LIKE '%Duri%' AND year = 2024"
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
) -> Dict[str, Any]:
    """
    Get the correct column names for static resource tables.
    """
    valid_volume_types = ["reserves", "resources", "risked"]
    valid_substances = ["oil", "con", "ga", "gn", "oc", "an"]

    volume_type = volume_type.lower().strip()
    substance = substance.lower().strip()

    if volume_type not in valid_volume_types:
        return {
            "error": f"Invalid volume_type '{volume_type}'",
            "column": None,
        }

    if substance not in valid_substances:
        return {
            "error": f"Invalid substance '{substance}'",
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

    examples = [
        f"SELECT SUM({column}) FROM field_resources WHERE field_name LIKE '%Duri%'",
        f"SELECT {column}, uncert_level FROM project_resources WHERE project_name LIKE '%Duri%'",
        f"SELECT SUM({column}) FROM nkri_resources WHERE uncert_level = '2. Middle Value'",
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
            "description": "Project-level timeseries forecast data (detail per project per year)",
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


def get_use_case_sql_pattern(use_case: str) -> Dict[str, Any]:
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
            "notes": "Returns the year and volume at peak production. Use with tpf_oc or tpf_an columns.",
        },
        "last_production_year": {
            "description": "Find the last year where production is still expected",
            "sql_template": """
SELECT MAX(year) as last_production_year
FROM {table}
WHERE {entity_col} LIKE '%{entity_name}%'
    AND ({oil_col} > 0 OR {gas_col} > 0);
""",
            "notes": "Returns the final year of production (End of Life). Checks both oil and gas columns.",
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
            "notes": "Uses onstream_year column if available, otherwise finds first year with tpf > 0",
        },
        "forecast_volume": {
            "description": "Get forecast volume for a specific year with unit conversion",
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
            "notes": "Returns MSTB/BSCF and converts to BOPD/MMSCFD. BOPD = MSTB * 1000 / days_in_year",
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
            "notes": "Shows production trend from current report_year forward. Default is 5 years.",
        },
    }

    if use_case not in patterns:
        return {
            "error": f"Unknown use case '{use_case}'",
            "available_use_cases": list(patterns.keys()),
        }

    return patterns[use_case]


def get_forecast_vs_historical_guide() -> Dict[str, Any]:
    """
    Get guidance on distinguishing forecast vs historical data.

    Returns:
        Dict with column mappings and decision logic for determining
        whether a query refers to historical or forecast data.
    """
    return {
        "historical": {
            "columns": [
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
                "rate_oil",
                "rate_con",
                "rate_ga",
                "rate_gn",
                "rate_oc",
                "rate_an",
            ],
            "description": "Cumulative production (cprd_*) and production rates (rate_*)",
            "data_type": "Actual/factual historical data",
            "year_logic": "year <= hist_year OR year <= report_year",
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
            "rationale": "report_year provides context - e.g., report_year 2023 may have forecast for 2024",
            "example": "Say: 'Forecast for 2025 (based on 2024 report)'",
        },
    }


def is_forecast_data(
    year: int, hist_year: Optional[int] = None, report_year: Optional[int] = None
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
    if hist_year is not None and year > hist_year:
        return True
    if report_year is not None and year > report_year:
        return True
    return False
