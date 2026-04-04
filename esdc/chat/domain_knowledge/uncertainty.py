"""Uncertainty level definitions and utilities."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional


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

# Direct database value synonyms for model-friendly resolution
# These map generic terms like "low", "mid", "high" directly to database values
DB_VALUE_SYNONYMS: Dict[str, str] = {
    "low_value": "1. Low Value",
    "middle_value": "2. Middle Value",
    "high_value": "3. High Value",
}


def get_uncertainty_filter(uncertainty: str) -> str:
    """
    Get the database filter value for an uncertainty level.

    Args:
        uncertainty: Uncertainty level (1P, 2P, 3P, proven, probable, possible,
                     low_value, middle_value, high_value, or generic terms like "low", "mid", "high")

    Returns:
        Database filter value for uncert_level column

    Note:
        For "probable" and "possible", this returns the default "2. Middle Value".
        For direct value synonyms (low_value, middle_value, high_value), returns the
        corresponding database value directly.
        Use get_uncertainty_spec() for full specification including SQL templates.
    """
    uncertainty_normalized = uncertainty.lower().strip()

    # Check direct database value synonyms first (e.g., low_value -> "1. Low Value")
    if uncertainty_normalized in DB_VALUE_SYNONYMS:
        return DB_VALUE_SYNONYMS[uncertainty_normalized]

    # Fall back to UNCERTAINTY_MAP
    spec = UNCERTAINTY_MAP.get(uncertainty.upper().strip())
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
    uncertainty_lower = uncertainty.lower().strip()

    # Check if it's a direct database value synonym first
    if uncertainty_lower in DB_VALUE_SYNONYMS:
        db_value = DB_VALUE_SYNONYMS[uncertainty_lower]
        is_cumulative = db_value in ("2. Middle Value", "3. High Value")
        return UncertaintySpec(
            type="direct",
            db_value=db_value,
            is_cumulative=is_cumulative,
            reserves_only=False,
            description=f"Direct {db_value} uncertainty level",
        )

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
