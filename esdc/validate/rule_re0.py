"""RE0xxx rules: Volumetric validation.

This module registers 66 validation rules (RE0001-RE0066)
covering non-negative checks, ordering constraints, aggregation consistency,
reserves-resources coherence, implication checks, zero-implies-zero checks,
and in-place vs reserves+production bounds.

Each category has an abstract parent class that concrete rules inherit from,
sharing the check/generate_fixes logic while differing only in column names
and uncertainty levels.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import duckdb

from esdc.selection import Severity
from esdc.validate.rule_re0_helpers import (
    AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
    FIELD_IDENTIFIER_COLS,
    UncertLevel,
    _execute_and_build_violations,
    build_aggregation_consistency_sql,
    build_field_non_negative_sql,
    build_field_ordering_sql,
    build_implication_sql,
    build_non_negative_sql,
    build_ordering_sql,
    build_reserve_vs_place_sql,
    build_same_row_ordering_sql,
    build_zero_implication_sql,
)
from esdc.validate.rules import ValidationRule, Violation, register_rule

# ---------------------------------------------------------------------------
# Category A: Non-negative checks (10 rules)
# ---------------------------------------------------------------------------


class RE0NonNegativeRule(ValidationRule):
    """Base: column value must be >= 0 at given uncertainty level.

    Violation when COALESCE(column, 0) < 0 at the given uncert_level.
    Operates on project_resources at the project level.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_non_negative_sql(self.validated_column, self.uncert)
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=[self.validated_column],
            validated_column=self.validated_column,
            compared_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE0FieldNonNegativeRule(ValidationRule):
    """Base: SUM(column) per field must be >= 0 at given uncertainty level.

    Aggregates column values across all projects within each field,
    then flags fields where the total is negative.
    Operates on field_resources.
    """

    is_fixable = False
    applies_to_tables = ["field_resources"]
    severity = Severity.STRICT
    validated_column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_field_non_negative_sql(self.validated_column, self.uncert)
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="field_resources",
            year=year,
            extra_columns=["val_ref"],
            identifier_cols=FIELD_IDENTIFIER_COLS,
            validated_column=self.validated_column,
            compared_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE0FieldOrderingRule(ValidationRule):
    """Base: SUM(validated_col) <= SUM(compared_col) per field across uncert_levels.

    Uses CTE aggregation and self-join to compare totals at different
    uncertainty levels within the same field.
    Operates on field_resources.
    """

    is_fixable = False
    applies_to_tables = ["field_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    low_uncert: UncertLevel
    high_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_field_ordering_sql(
            self.validated_column,
            self.compared_column,
            self.low_uncert,
            self.high_uncert,
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="field_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            identifier_cols=FIELD_IDENTIFIER_COLS,
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0001(RE0FieldNonNegativeRule):
    rule_id = "RE0001"
    description = "IOIP: Low Case must be greater than or equal to zero"
    formal = r"$N^{\text{P90}} \geq 0$"
    validated_column = "ioip"
    uncert = UncertLevel.LOW


@register_rule
class RE0002(RE0FieldNonNegativeRule):
    rule_id = "RE0002"
    description = "IGIP: Low Case must be greater than or equal to zero"
    formal = r"$G^{\text{P90}} \geq 0$"
    validated_column = "igip"
    uncert = UncertLevel.LOW


@register_rule
class RE0007(RE0NonNegativeRule):
    rule_id = "RE0007"
    description = "Oil GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    formal = r"$\Delta N_{pn}^{\text{P90}} \geq 0$"
    validated_column = "rec_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0008(RE0NonNegativeRule):
    rule_id = "RE0008"
    description = "Condensate GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    formal = r"$\Delta N_{pn}^{c \text{ P90}} \geq 0$"
    validated_column = "rec_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0009(RE0NonNegativeRule):
    rule_id = "RE0009"
    description = (
        "Associated Gas GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P90}} \geq 0$"
    validated_column = "rec_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0010(RE0NonNegativeRule):
    rule_id = "RE0010"
    description = (
        "Non Associated Gas GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{pn}^{\text{P90}} \geq 0$"
    validated_column = "rec_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE0011(RE0NonNegativeRule):
    rule_id = "RE0011"
    description = "Oil Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta N_{ps}^{\text{1P}} \geq 0$"
    validated_column = "res_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0012(RE0NonNegativeRule):
    rule_id = "RE0012"
    description = "Condensate Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta N_{ps}^{c\text{ 1P}} \geq 0$"
    validated_column = "res_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0013(RE0NonNegativeRule):
    rule_id = "RE0013"
    description = "Associated Gas Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \geq 0$"
    validated_column = "res_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0014(RE0NonNegativeRule):
    rule_id = "RE0014"
    description = (
        "Non Associated Gas Reserves: 1P must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{ps}^{\text{1P}} \geq 0$"
    validated_column = "res_gn"
    uncert = UncertLevel.LOW


# ---------------------------------------------------------------------------
# Category B: Ordering checks via self-join (20 rules)
# ---------------------------------------------------------------------------


class RE0OrderingRule(ValidationRule):
    """Base: validated_column <= compared_column across uncertainty levels.

    Violation when COALESCE(validated_column, 0) > COALESCE(compared_column, 0)
    across two uncert_level rows for the same project+year.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    low_uncert: UncertLevel
    high_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_ordering_sql(
            self.validated_column,
            self.compared_column,
            self.low_uncert,
            self.high_uncert,
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# --- In-Place ordering (RE0003-RE0006): field-level ---


@register_rule
class RE0003(RE0FieldOrderingRule):
    rule_id = "RE0003"
    description = "IOIP: Low Case must be less than or equal to Mid Case"
    formal = r"$N^{\text{P90}} \leq N^{\text{P50}}$"
    validated_column = "ioip"
    compared_column = "ioip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0004(RE0FieldOrderingRule):
    rule_id = "RE0004"
    description = "IOIP: Mid Case must be less than or equal to High Case"
    formal = r"$N^{\text{P50}} \leq N^{\text{P10}}$"
    validated_column = "ioip"
    compared_column = "ioip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0005(RE0FieldOrderingRule):
    rule_id = "RE0005"
    description = "IGIP: Low Case must be less than or equal to Mid Case"
    formal = r"$G^{\text{P90}} \leq G^{\text{P50}}$"
    validated_column = "igip"
    compared_column = "igip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0006(RE0FieldOrderingRule):
    rule_id = "RE0006"
    description = "IGIP: Mid Case must be less than or equal to High Case"
    formal = r"$G^{\text{P50}} \leq G^{\text{P10}}$"
    validated_column = "igip"
    compared_column = "igip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# --- Resources ordering (RE0015-RE0022) ---


@register_rule
class RE0015(RE0OrderingRule):
    rule_id = "RE0015"
    description = "Oil GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    formal = r"$\Delta N_{pn}^{\text{P90}} \leq \Delta N_{pn}^{\text{P50}}$"
    validated_column = "rec_oil"
    compared_column = "rec_oil"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0016(RE0OrderingRule):
    rule_id = "RE0016"
    description = "Oil GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    formal = r"$\Delta N_{pn}^{\text{P50}} \leq \Delta N_{pn}^{\text{P10}}$"
    validated_column = "rec_oil"
    compared_column = "rec_oil"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0017(RE0OrderingRule):
    rule_id = "RE0017"
    description = (
        "Condensate GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta N_{pn}^{c \text{ P90}} \leq \Delta N_{pn}^{c \text{ P50}}$"
    validated_column = "rec_con"
    compared_column = "rec_con"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0018(RE0OrderingRule):
    rule_id = "RE0018"
    description = (
        "Condensate GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta N_{pn}^{c \text{ P50}} \leq \Delta N_{pn}^{c \text{ P10}}$"
    validated_column = "rec_con"
    compared_column = "rec_con"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0019(RE0OrderingRule):
    rule_id = "RE0019"
    description = (
        "Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P90}} \leq \Delta G_{pn}^{a \text{ P50}}$"
    validated_column = "rec_ga"
    compared_column = "rec_ga"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0020(RE0OrderingRule):
    rule_id = "RE0020"
    description = (
        "Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P50}} \leq \Delta G_{pn}^{a \text{ P10}}$"
    validated_column = "rec_ga"
    compared_column = "rec_ga"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0021(RE0OrderingRule):
    rule_id = "RE0021"
    description = (
        "Non Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta G_{pn}^{\text{P90}} \leq \Delta G_{pn}^{\text{P50}}$"
    validated_column = "rec_gn"
    compared_column = "rec_gn"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0022(RE0OrderingRule):
    rule_id = "RE0022"
    description = (
        "Non Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta G_{pn}^{\text{P50}} \leq \Delta G_{pn}^{\text{P10}}$"
    validated_column = "rec_gn"
    compared_column = "rec_gn"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# --- Reserves ordering (RE0023-RE0030) ---


@register_rule
class RE0023(RE0OrderingRule):
    rule_id = "RE0023"
    description = "Oil Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta N_{ps}^{\text{1P}} \leq \Delta N_{ps}^{\text{2P}}$"
    validated_column = "res_oil"
    compared_column = "res_oil"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0024(RE0OrderingRule):
    rule_id = "RE0024"
    description = "Oil Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta N_{ps}^{\text{2P}} \leq \Delta N_{ps}^{\text{3P}}$"
    validated_column = "res_oil"
    compared_column = "res_oil"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0025(RE0OrderingRule):
    rule_id = "RE0025"
    description = "Condensate Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta N_{ps}^{c \text{ 1P}} \leq \Delta N_{ps}^{c \text{ 2P}}$"
    validated_column = "res_con"
    compared_column = "res_con"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0026(RE0OrderingRule):
    rule_id = "RE0026"
    description = "Condensate Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta N_{ps}^{c \text{ 2P}} \leq \Delta N_{ps}^{c \text{ 3P}}$"
    validated_column = "res_con"
    compared_column = "res_con"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0027(RE0OrderingRule):
    rule_id = "RE0027"
    description = "Associated Gas Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \leq \Delta G_{ps}^{a \text{ 2P}}$"
    validated_column = "res_ga"
    compared_column = "res_ga"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0028(RE0OrderingRule):
    rule_id = "RE0028"
    description = "Associated Gas Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta G_{ps}^{a \text{ 2P}} \leq \Delta G_{ps}^{a \text{ 3P}}$"
    validated_column = "res_ga"
    compared_column = "res_ga"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0029(RE0OrderingRule):
    rule_id = "RE0029"
    description = "Non Associated Gas Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta G_{ps}^{\text{1P}} \leq \Delta G_{ps}^{\text{2P}}$"
    validated_column = "res_gn"
    compared_column = "res_gn"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0030(RE0OrderingRule):
    rule_id = "RE0030"
    description = "Non Associated Gas Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta G_{ps}^{\text{2P}} \leq \Delta G_{ps}^{\text{3P}}$"
    validated_column = "res_gn"
    compared_column = "res_gn"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category C: Reserves <= Resources same-row ordering (12 rules)
# ---------------------------------------------------------------------------


class RE0SameRowOrderingRule(ValidationRule):
    """Base: validated_column <= compared_column within same row.

    Violation when COALESCE(validated_column, 0) > COALESCE(compared_column, 0)
    within the same row (same uncert_level).
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_same_row_ordering_sql(
            self.validated_column, self.compared_column, self.uncert
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# --- Oil ---


@register_rule
class RE0031(RE0SameRowOrderingRule):
    rule_id = "RE0031"
    description = "Oil Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta N_{ps}^{\text{1P}} \leq \Delta N_{pn}^{\text{1R}}$"
    validated_column = "res_oil"
    compared_column = "rec_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0032(RE0SameRowOrderingRule):
    rule_id = "RE0032"
    description = "Oil Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta N_{ps}^{\text{2P}} \leq \Delta N_{pn}^{\text{2R}}$"
    validated_column = "res_oil"
    compared_column = "rec_oil"
    uncert = UncertLevel.MID


@register_rule
class RE0033(RE0SameRowOrderingRule):
    rule_id = "RE0033"
    description = "Oil Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta N_{ps}^{\text{3P}} \leq \Delta N_{pn}^{\text{3R}}$"
    validated_column = "res_oil"
    compared_column = "rec_oil"
    uncert = UncertLevel.HIGH


# --- Condensate ---


@register_rule
class RE0034(RE0SameRowOrderingRule):
    rule_id = "RE0034"
    description = "Condensate Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta N_{ps}^{c \text{ 1P}} \leq \Delta N_{pn}^{c \text{ 1R}}$"
    validated_column = "res_con"
    compared_column = "rec_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0035(RE0SameRowOrderingRule):
    rule_id = "RE0035"
    description = "Condensate Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta N_{ps}^{c \text{ 2P}} \leq \Delta N_{pn}^{c \text{ 2R}}$"
    validated_column = "res_con"
    compared_column = "rec_con"
    uncert = UncertLevel.MID


@register_rule
class RE0036(RE0SameRowOrderingRule):
    rule_id = "RE0036"
    description = "Condensate Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta N_{ps}^{c \text{ 3P}} \leq \Delta N_{pn}^{c \text{ 3R}}$"
    validated_column = "res_con"
    compared_column = "rec_con"
    uncert = UncertLevel.HIGH


# --- Associated Gas ---


@register_rule
class RE0037(RE0SameRowOrderingRule):
    rule_id = "RE0037"
    description = "Associated Gas Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \leq \Delta G_{pn}^{a \text{ 1R}}$"
    validated_column = "res_ga"
    compared_column = "rec_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0038(RE0SameRowOrderingRule):
    rule_id = "RE0038"
    description = "Associated Gas Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta G_{ps}^{a \text{ 2P}} \leq \Delta G_{pn}^{a \text{ 2R}}$"
    validated_column = "res_ga"
    compared_column = "rec_ga"
    uncert = UncertLevel.MID


@register_rule
class RE0039(RE0SameRowOrderingRule):
    rule_id = "RE0039"
    description = "Associated Gas Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta G_{ps}^{a \text{ 3P}} \leq \Delta G_{pn}^{a \text{ 3R}}$"
    validated_column = "res_ga"
    compared_column = "rec_ga"
    uncert = UncertLevel.HIGH


# --- Non-Associated Gas ---


@register_rule
class RE0040(RE0SameRowOrderingRule):
    rule_id = "RE0040"
    description = "Non Associated Gas Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta G_{ps}^{\text{1P}} \leq \Delta G_{pn}^{\text{1R}}$"
    validated_column = "res_gn"
    compared_column = "rec_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE0041(RE0SameRowOrderingRule):
    rule_id = "RE0041"
    description = "Non Associated Gas Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta G_{ps}^{\text{2P}} \leq \Delta G_{pn}^{\text{2R}}$"
    validated_column = "res_gn"
    compared_column = "rec_gn"
    uncert = UncertLevel.MID


@register_rule
class RE0042(RE0SameRowOrderingRule):
    rule_id = "RE0042"
    description = "Non Associated Gas Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta G_{ps}^{\text{3P}} \leq \Delta G_{pn}^{\text{3R}}$"
    validated_column = "res_gn"
    compared_column = "rec_gn"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category D: Aggregation consistency (6 rules)
# ---------------------------------------------------------------------------


class RE0AggregationConsistencyRule(ValidationRule):
    """Base: SUM(compared_column) per field must equal validated_column.

    Violation when ABS(SUM(compared_column) - validated_column) > 0.001.
    Joins field_resources with project_resources on the GROUP BY keys
    (wk_id, field_id, report_year, project_stage, project_class, uncert_level).
    """

    is_fixable = False
    applies_to_tables = ["field_resources", "project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_aggregation_consistency_sql(
            self.validated_column, self.compared_column, self.uncert
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="field_resources",
            year=year,
            extra_columns=["val_sum", "val_field"],
            identifier_cols=AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0043(RE0AggregationConsistencyRule):
    rule_id = "RE0043"
    description = "IOIP Low: Sum of Project IOIP Low must be equal to IOIP Low"
    formal = r"$\sum_{i=1}^n N_{\text{prj},i}^{\text{P90}} = N^{\text{P90}}$"
    validated_column = "ioip"
    compared_column = "prj_ioip"
    uncert = UncertLevel.LOW


@register_rule
class RE0044(RE0AggregationConsistencyRule):
    rule_id = "RE0044"
    description = "IOIP Mid: Sum of Project IOIP Mid must be equal to IOIP Mid"
    formal = r"$\sum_{i=1}^n N_{\text{prj},i}^{\text{P50}} = N^{\text{P50}}$"
    validated_column = "ioip"
    compared_column = "prj_ioip"
    uncert = UncertLevel.MID


@register_rule
class RE0045(RE0AggregationConsistencyRule):
    rule_id = "RE0045"
    description = "IOIP High: Sum of Project IOIP High must be equal to IOIP High"
    formal = r"$\sum_{i=1}^n N_{\text{prj},i}^{\text{P10}} = N^{\text{P10}}$"
    validated_column = "ioip"
    compared_column = "prj_ioip"
    uncert = UncertLevel.HIGH


@register_rule
class RE0046(RE0AggregationConsistencyRule):
    rule_id = "RE0046"
    description = "IGIP Low: Sum of Project IGIP Low must be equal to IGIP Low"
    formal = r"$\sum_{i=1}^n G_{\text{prj},i}^{\text{P90}} = G^{\text{P90}}$"
    validated_column = "igip"
    compared_column = "prj_igip"
    uncert = UncertLevel.LOW


@register_rule
class RE0047(RE0AggregationConsistencyRule):
    rule_id = "RE0047"
    description = "IGIP Mid: Sum of Project IGIP Mid must be equal to IGIP Mid"
    formal = r"$\sum_{i=1}^n G_{\text{prj},i}^{\text{P50}} = G^{\text{P50}}$"
    validated_column = "igip"
    compared_column = "prj_igip"
    uncert = UncertLevel.MID


@register_rule
class RE0048(RE0AggregationConsistencyRule):
    rule_id = "RE0048"
    description = "IGIP High: Sum of Project IGIP High must be equal to IGIP High"
    formal = r"$\sum_{i=1}^n G_{\text{prj},i}^{\text{P10}} = G^{\text{P10}}$"
    validated_column = "igip"
    compared_column = "prj_igip"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category E: Implication checks (4 rules)
# ---------------------------------------------------------------------------


class RE0ImplicationRule(ValidationRule):
    """Base: if validated_column > 0 then compared_column > 0.

    Violation when validated_column > 0 but compared_column == 0 across two
    uncert_level rows for the same project+year.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_implication_sql(
            self.validated_column,
            self.cond_uncert,
            self.compared_column,
            self.result_uncert,
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0049(RE0ImplicationRule):
    rule_id = "RE0049"
    description = (
        "Oil Reserves: 1P should be greater than zero if 3P is greater than zero"
    )
    formal = r"$\Delta N_{ps}^{\text{3P}} > 0  \implies \Delta N_{ps}^{\text{1P}} > 0$"
    validated_column = "res_oil"
    compared_column = "res_oil"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE0050(RE0ImplicationRule):
    rule_id = "RE0050"
    description = (
        "Condensate Reserves: 1P should be greater than zero if 3P is greater than zero"
    )
    formal = (
        r"$\Delta N_{ps}^{c \text{ 3P}} > 0  \implies \Delta N_{ps}^{c \text{ 1P}} > 0$"
    )
    validated_column = "res_con"
    compared_column = "res_con"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE0051(RE0ImplicationRule):
    rule_id = "RE0051"
    description = "Associated Gas Reserves: 1P should be greater than zero if 3P is greater than zero"  # noqa: E501
    formal = (
        r"$\Delta G_{ps}^{a \text{ 3P}} > 0  \implies \Delta G_{ps}^{a \text{ 1P}} > 0$"
    )
    validated_column = "res_ga"
    compared_column = "res_ga"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE0052(RE0ImplicationRule):
    rule_id = "RE0052"
    description = "Non Associated Gas Reserves: 1P should be greater than zero if 3P is greater than zero"  # noqa: E501
    formal = r"$\Delta G_{ps}^{\text{3P}} > 0  \implies \Delta G_{ps}^{\text{1P}} > 0$"
    validated_column = "res_gn"
    compared_column = "res_gn"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


# ---------------------------------------------------------------------------
# Category F: Reserve + CumProd < In-Place (6 rules)
# ---------------------------------------------------------------------------


class RE0ReserveVsPlaceRule(ValidationRule):
    """Base: if place > 0, then reserve + cumprod < place.

    Violation when validated_column > 0 and
    COALESCE(reserve_col, 0) + COALESCE(cumprod_col, 0) >= validated_column.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    reserve_col: str
    cumprod_col: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_reserve_vs_place_sql(
            self.validated_column, self.reserve_col, self.cumprod_col, self.uncert
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp", "val_sum"],
            validated_column=self.validated_column,
            compared_columns=[self.reserve_col, self.cumprod_col],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0053(RE0ReserveVsPlaceRule):
    rule_id = "RE0053"
    description = (
        "Project IOIP Low: if P90 greater than zero then IOIP Low Value "
        "must be greater than sum of Gross Cumulative Production and 1P Reserves"
    )
    formal = r"$N_{\text{prj}}^{\text{P90}} > 0  \implies \Delta N_{ps}^{\text{1P}} + N_{pg} < N_{\text{prj}}^{\text{P90}}$"  # noqa: E501
    validated_column = "prj_ioip"
    reserve_col = "res_oil"
    cumprod_col = "cprd_grs_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0054(RE0ReserveVsPlaceRule):
    rule_id = "RE0054"
    description = (
        "Project IOIP Middle: if P50 greater than zero then IOIP Middle "
        "Value must be greater than sum of Gross Cumulative Production and 2P Reserves"
    )
    formal = r"$N_{\text{prj}}^{\text{P50}} > 0  \implies \Delta N_{ps}^{\text{2P}} + N_{pg} < N_{\text{prj}}^{\text{P50}}$"  # noqa: E501
    validated_column = "prj_ioip"
    reserve_col = "res_oil"
    cumprod_col = "cprd_grs_oil"
    uncert = UncertLevel.MID


@register_rule
class RE0055(RE0ReserveVsPlaceRule):
    rule_id = "RE0055"
    description = (
        "Project IOIP High: if P10 greater than zero then IOIP High Value "
        "must be greater than sum of Gross Cumulative Production and 3P Reserves"
    )
    formal = r"$N_{\text{prj}}^{\text{P10}} > 0  \implies \Delta N_{ps}^{\text{3P}} + N_{pg} < N_{\text{prj}}^{\text{P10}}$"  # noqa: E501
    validated_column = "prj_ioip"
    reserve_col = "res_oil"
    cumprod_col = "cprd_grs_oil"
    uncert = UncertLevel.HIGH


@register_rule
class RE0056(RE0ReserveVsPlaceRule):
    rule_id = "RE0056"
    description = (
        "Project IGIP Low: if P90 greater than zero then IGIP Low Value "
        "must be greater than sum of Gross Cumulative Production and 1P Reserves"
    )
    formal = r"$G_{\text{prj}}^{\text{P90}} > 0  \implies \Delta G_{ps}^{\text{1P}} + G_{pg} < G_{\text{prj}}^{\text{P90}}$"  # noqa: E501
    validated_column = "prj_igip"
    reserve_col = "res_gn"
    cumprod_col = "cprd_grs_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE0057(RE0ReserveVsPlaceRule):
    rule_id = "RE0057"
    description = (
        "Project IGIP Middle: if P50 greater than zero then IGIP Middle "
        "Value must be greater than sum of Gross Cumulative Production and 2P Reserves"
    )
    formal = r"$G_{\text{prj}}^{\text{P50}} > 0  \implies \Delta G_{ps}^{\text{2P}} + G_{pg} < G_{\text{prj}}^{\text{P50}}$"  # noqa: E501
    validated_column = "prj_igip"
    reserve_col = "res_gn"
    cumprod_col = "cprd_grs_gn"
    uncert = UncertLevel.MID


@register_rule
class RE0058(RE0ReserveVsPlaceRule):
    rule_id = "RE0058"
    description = (
        "Project IGIP High: if P10 greater than zero then IGIP High Value "
        "must be greater than sum of Gross Cumulative Production and 3P Reserves"
    )
    formal = r"$G_{\text{prj}}^{\text{P10}} > 0  \implies \Delta G_{ps}^{\text{3P}} + G_{pg} < G_{\text{prj}}^{\text{P10}}$"  # noqa: E501
    validated_column = "prj_igip"
    reserve_col = "res_gn"
    cumprod_col = "cprd_grs_gn"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category G: Zero-implies-zero checks (4 rules)
# ---------------------------------------------------------------------------


class RE0ZeroImplicationRule(ValidationRule):
    """Base: if validated_column is zero, compared_column must also be zero.

    Violation when COALESCE(validated_column, 0) = 0
    but COALESCE(compared_column, 0) != 0,
    across two uncert_level rows for the same project+year.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_zero_implication_sql(
            self.validated_column,
            self.cond_uncert,
            self.compared_column,
            self.result_uncert,
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0059(RE0ZeroImplicationRule):
    rule_id = "RE0059"
    description = "Project IOIP P50: Must equal zero when P90 equals zero"
    formal = (
        r"$N_{\text{prj}}^{\text{P90}} = 0 \implies N_{\text{prj}}^{\text{P50}} = 0$"  # noqa: E501
    )
    validated_column = "prj_ioip"
    compared_column = "prj_ioip"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.MID


@register_rule
class RE0060(RE0ZeroImplicationRule):
    rule_id = "RE0060"
    description = "Project IOIP P10: Must equal zero when P90 equals zero"
    formal = (
        r"$N_{\text{prj}}^{\text{P90}} = 0 \implies N_{\text{prj}}^{\text{P10}} = 0$"  # noqa: E501
    )
    validated_column = "prj_ioip"
    compared_column = "prj_ioip"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.HIGH


@register_rule
class RE0061(RE0ZeroImplicationRule):
    rule_id = "RE0061"
    description = "Project IGIP P50: Must equal zero when P90 equals zero"
    formal = (
        r"$G_{\text{prj}}^{\text{P90}} = 0 \implies G_{\text{prj}}^{\text{P50}} = 0$"  # noqa: E501
    )
    validated_column = "prj_igip"
    compared_column = "prj_igip"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.MID


@register_rule
class RE0062(RE0ZeroImplicationRule):
    rule_id = "RE0062"
    description = "Project IGIP P10: Must equal zero when P90 equals zero"
    formal = (
        r"$G_{\text{prj}}^{\text{P90}} = 0 \implies G_{\text{prj}}^{\text{P10}} = 0$"  # noqa: E501
    )
    validated_column = "prj_igip"
    compared_column = "prj_igip"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Project-level ordering (RE0063-RE0066)
# ---------------------------------------------------------------------------


@register_rule
class RE0063(RE0OrderingRule):
    rule_id = "RE0063"
    description = "Project IOIP P90: Must be less than or equal to P50"
    formal = r"$N_{\text{prj}}^{\text{P90}} \leq N_{\text{prj}}^{\text{P50}}$"
    validated_column = "prj_ioip"
    compared_column = "prj_ioip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0064(RE0OrderingRule):
    rule_id = "RE0064"
    description = "Project IOIP P50: Must be less than or equal to P10"
    formal = r"$N_{\text{prj}}^{\text{P50}} \leq N_{\text{prj}}^{\text{P10}}$"
    validated_column = "prj_ioip"
    compared_column = "prj_ioip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0065(RE0OrderingRule):
    rule_id = "RE0065"
    description = "Project IGIP P90: Must be less than or equal to P50"
    formal = r"$G_{\text{prj}}^{\text{P90}} \leq G_{\text{prj}}^{\text{P50}}$"
    validated_column = "prj_igip"
    compared_column = "prj_igip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0066(RE0OrderingRule):
    rule_id = "RE0066"
    description = "Project IGIP P50: Must be less than or equal to P10"
    formal = r"$G_{\text{prj}}^{\text{P50}} \leq G_{\text{prj}}^{\text{P10}}$"
    validated_column = "prj_igip"
    compared_column = "prj_igip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH
