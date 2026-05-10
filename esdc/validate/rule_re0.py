"""RE0xxx rules: Volumetric validation.

This module registers 52 validation rules (RE0001-RE0042, RE0049-RE0058)
covering non-negative checks, ordering constraints, reserves-resources
coherence, implication checks, and in-place vs reserves+production bounds.

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
    UncertLevel,
    _execute_and_build_violations,
    build_implication_sql,
    build_non_negative_sql,
    build_ordering_sql,
    build_reserve_vs_place_sql,
    build_same_row_ordering_sql,
)
from esdc.validate.rules import ValidationRule, Violation, register_rule

# ---------------------------------------------------------------------------
# Category A: Non-negative checks (10 rules)
# ---------------------------------------------------------------------------


class RE0NonNegativeRule(ValidationRule):
    """Base: column value must be >= 0 at given uncertainty level.

    Violation when COALESCE(column, 0) < 0 at the given uncert_level.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_non_negative_sql(self.column, self.uncert)
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=[self.column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE0001(RE0NonNegativeRule):
    rule_id = "RE0001"
    description = "IOIP: Low Case must be greater than or equal to zero"
    formal = r"$N^{\text{P90}} \geq 0$"
    column = "prj_ioip"
    uncert = UncertLevel.LOW


@register_rule
class RE0002(RE0NonNegativeRule):
    rule_id = "RE0002"
    description = "IGIP: Low Case must be greater than or equal to zero"
    formal = r"$G^{\text{P90}} \geq 0$"
    column = "prj_igip"
    uncert = UncertLevel.LOW


@register_rule
class RE0007(RE0NonNegativeRule):
    rule_id = "RE0007"
    description = "Oil GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    formal = r"$\Delta N_{pn}^{\text{P90}} \geq 0$"
    column = "rec_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0008(RE0NonNegativeRule):
    rule_id = "RE0008"
    description = "Condensate GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    formal = r"$\Delta N_{pn}^{c \text{ P90}} \geq 0$"
    column = "rec_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0009(RE0NonNegativeRule):
    rule_id = "RE0009"
    description = (
        "Associated Gas GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P90}} \geq 0$"
    column = "rec_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0010(RE0NonNegativeRule):
    rule_id = "RE0010"
    description = (
        "Non Associated Gas GRR/CR/PR: 1R/1C/1U must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{pn}^{\text{P90}} \geq 0$"
    column = "rec_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE0011(RE0NonNegativeRule):
    rule_id = "RE0011"
    description = "Oil Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta N_{ps}^{\text{1P}} \geq 0$"
    column = "res_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0012(RE0NonNegativeRule):
    rule_id = "RE0012"
    description = "Condensate Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta N_{ps}^{c\text{ 1P}} \geq 0$"
    column = "res_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0013(RE0NonNegativeRule):
    rule_id = "RE0013"
    description = "Associated Gas Reserves: 1P must be greater than or equal to zero"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \geq 0$"
    column = "res_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0014(RE0NonNegativeRule):
    rule_id = "RE0014"
    description = (
        "Non Associated Gas Reserves: 1P must be greater than or equal to zero"
    )
    formal = r"$\Delta G_{ps}^{\text{1P}} \geq 0$"
    column = "res_gn"
    uncert = UncertLevel.LOW


# ---------------------------------------------------------------------------
# Category B: Ordering checks via self-join (20 rules)
# ---------------------------------------------------------------------------


class RE0OrderingRule(ValidationRule):
    """Base: low_col <= high_col across uncertainty levels.

    Violation when COALESCE(low_col, 0) > COALESCE(high_col, 0)
    across two uncert_level rows for the same project+year.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    low_col: str
    high_col: str
    low_uncert: UncertLevel
    high_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_ordering_sql(
            self.low_col, self.high_col, self.low_uncert, self.high_uncert
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
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# --- In-Place ordering (RE0003-RE0006) ---


@register_rule
class RE0003(RE0OrderingRule):
    rule_id = "RE0003"
    description = "IOIP: Low Case must be less than or equal to Mid Case"
    formal = r"$N^{\text{P90}} \leq N^{\text{P50}}$"
    low_col = "prj_ioip"
    high_col = "prj_ioip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0004(RE0OrderingRule):
    rule_id = "RE0004"
    description = "IOIP: Mid Case must be less than or equal to High Case"
    formal = r"$N^{\text{P50}} \leq N^{\text{P10}}$"
    low_col = "prj_ioip"
    high_col = "prj_ioip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0005(RE0OrderingRule):
    rule_id = "RE0005"
    description = "IGIP: Low Case must be less than or equal to Mid Case"
    formal = r"$G^{\text{P90}} \leq G^{\text{P50}}$"
    low_col = "prj_igip"
    high_col = "prj_igip"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0006(RE0OrderingRule):
    rule_id = "RE0006"
    description = "IGIP: Mid Case must be less than or equal to High Case"
    formal = r"$G^{\text{P50}} \leq G^{\text{P10}}$"
    low_col = "prj_igip"
    high_col = "prj_igip"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# --- Resources ordering (RE0015-RE0022) ---


@register_rule
class RE0015(RE0OrderingRule):
    rule_id = "RE0015"
    description = "Oil GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    formal = r"$\Delta N_{pn}^{\text{P90}} \leq \Delta N_{pn}^{\text{P50}}$"
    low_col = "rec_oil"
    high_col = "rec_oil"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0016(RE0OrderingRule):
    rule_id = "RE0016"
    description = "Oil GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    formal = r"$\Delta N_{pn}^{\text{P50}} \leq \Delta N_{pn}^{\text{P10}}$"
    low_col = "rec_oil"
    high_col = "rec_oil"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0017(RE0OrderingRule):
    rule_id = "RE0017"
    description = (
        "Condensate GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta N_{pn}^{c \text{ P90}} \leq \Delta N_{pn}^{c \text{ P50}}$"
    low_col = "rec_con"
    high_col = "rec_con"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0018(RE0OrderingRule):
    rule_id = "RE0018"
    description = (
        "Condensate GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta N_{pn}^{c \text{ P50}} \leq \Delta N_{pn}^{c \text{ P10}}$"
    low_col = "rec_con"
    high_col = "rec_con"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0019(RE0OrderingRule):
    rule_id = "RE0019"
    description = (
        "Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P90}} \leq \Delta G_{pn}^{a \text{ P50}}$"
    low_col = "rec_ga"
    high_col = "rec_ga"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0020(RE0OrderingRule):
    rule_id = "RE0020"
    description = (
        "Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta G_{pn}^{a \text{ P50}} \leq \Delta G_{pn}^{a \text{ P10}}$"
    low_col = "rec_ga"
    high_col = "rec_ga"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0021(RE0OrderingRule):
    rule_id = "RE0021"
    description = (
        "Non Associated Gas GRR/CR/PR: 1R/1C/1U must be less than or equal to 2R/2C/2U"
    )
    formal = r"$\Delta G_{pn}^{\text{P90}} \leq \Delta G_{pn}^{\text{P50}}$"
    low_col = "rec_gn"
    high_col = "rec_gn"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0022(RE0OrderingRule):
    rule_id = "RE0022"
    description = (
        "Non Associated Gas GRR/CR/PR: 2R/2C/2U must be less than or equal to 3R/3C/3U"
    )
    formal = r"$\Delta G_{pn}^{\text{P50}} \leq \Delta G_{pn}^{\text{P10}}$"
    low_col = "rec_gn"
    high_col = "rec_gn"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# --- Reserves ordering (RE0023-RE0030) ---


@register_rule
class RE0023(RE0OrderingRule):
    rule_id = "RE0023"
    description = "Oil Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta N_{ps}^{\text{1P}} \leq \Delta N_{ps}^{\text{2P}}$"
    low_col = "res_oil"
    high_col = "res_oil"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0024(RE0OrderingRule):
    rule_id = "RE0024"
    description = "Oil Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta N_{ps}^{\text{2P}} \leq \Delta N_{ps}^{\text{3P}}$"
    low_col = "res_oil"
    high_col = "res_oil"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0025(RE0OrderingRule):
    rule_id = "RE0025"
    description = "Condensate Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta N_{ps}^{c \text{ 1P}} \leq \Delta N_{ps}^{c \text{ 2P}}$"
    low_col = "res_con"
    high_col = "res_con"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0026(RE0OrderingRule):
    rule_id = "RE0026"
    description = "Condensate Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta N_{ps}^{c \text{ 2P}} \leq \Delta N_{ps}^{c \text{ 3P}}$"
    low_col = "res_con"
    high_col = "res_con"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0027(RE0OrderingRule):
    rule_id = "RE0027"
    description = "Associated Gas Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \leq \Delta G_{ps}^{a \text{ 2P}}$"
    low_col = "res_ga"
    high_col = "res_ga"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0028(RE0OrderingRule):
    rule_id = "RE0028"
    description = "Associated Gas Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta G_{ps}^{a \text{ 2P}} \leq \Delta G_{ps}^{a \text{ 3P}}$"
    low_col = "res_ga"
    high_col = "res_ga"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


@register_rule
class RE0029(RE0OrderingRule):
    rule_id = "RE0029"
    description = "Non Associated Gas Reserves: 1P must be less than or equal to 2P"
    formal = r"$\Delta G_{ps}^{\text{1P}} \leq \Delta G_{ps}^{\text{2P}}$"
    low_col = "res_gn"
    high_col = "res_gn"
    low_uncert = UncertLevel.LOW
    high_uncert = UncertLevel.MID


@register_rule
class RE0030(RE0OrderingRule):
    rule_id = "RE0030"
    description = "Non Associated Gas Reserves: 2P must be less than or equal to 3P"
    formal = r"$\Delta G_{ps}^{\text{2P}} \leq \Delta G_{ps}^{\text{3P}}$"
    low_col = "res_gn"
    high_col = "res_gn"
    low_uncert = UncertLevel.MID
    high_uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category C: Reserves <= Resources same-row ordering (12 rules)
# ---------------------------------------------------------------------------


class RE0SameRowOrderingRule(ValidationRule):
    """Base: low_col <= high_col within same row.

    Violation when COALESCE(low_col, 0) > COALESCE(high_col, 0)
    within the same row (same uncert_level).
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    low_col: str
    high_col: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_same_row_ordering_sql(self.low_col, self.high_col, self.uncert)
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
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
    low_col = "res_oil"
    high_col = "rec_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE0032(RE0SameRowOrderingRule):
    rule_id = "RE0032"
    description = "Oil Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta N_{ps}^{\text{2P}} \leq \Delta N_{pn}^{\text{2R}}$"
    low_col = "res_oil"
    high_col = "rec_oil"
    uncert = UncertLevel.MID


@register_rule
class RE0033(RE0SameRowOrderingRule):
    rule_id = "RE0033"
    description = "Oil Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta N_{ps}^{\text{3P}} \leq \Delta N_{pn}^{\text{3R}}$"
    low_col = "res_oil"
    high_col = "rec_oil"
    uncert = UncertLevel.HIGH


# --- Condensate ---


@register_rule
class RE0034(RE0SameRowOrderingRule):
    rule_id = "RE0034"
    description = "Condensate Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta N_{ps}^{c \text{ 1P}} \leq \Delta N_{pn}^{c \text{ 1R}}$"
    low_col = "res_con"
    high_col = "rec_con"
    uncert = UncertLevel.LOW


@register_rule
class RE0035(RE0SameRowOrderingRule):
    rule_id = "RE0035"
    description = "Condensate Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta N_{ps}^{c \text{ 2P}} \leq \Delta N_{pn}^{c \text{ 2R}}$"
    low_col = "res_con"
    high_col = "rec_con"
    uncert = UncertLevel.MID


@register_rule
class RE0036(RE0SameRowOrderingRule):
    rule_id = "RE0036"
    description = "Condensate Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta N_{ps}^{c \text{ 3P}} \leq \Delta N_{pn}^{c \text{ 3R}}$"
    low_col = "res_con"
    high_col = "rec_con"
    uncert = UncertLevel.HIGH


# --- Associated Gas ---


@register_rule
class RE0037(RE0SameRowOrderingRule):
    rule_id = "RE0037"
    description = "Associated Gas Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta G_{ps}^{a \text{ 1P}} \leq \Delta G_{pn}^{a \text{ 1R}}$"
    low_col = "res_ga"
    high_col = "rec_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE0038(RE0SameRowOrderingRule):
    rule_id = "RE0038"
    description = "Associated Gas Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta G_{ps}^{a \text{ 2P}} \leq \Delta G_{pn}^{a \text{ 2R}}$"
    low_col = "res_ga"
    high_col = "rec_ga"
    uncert = UncertLevel.MID


@register_rule
class RE0039(RE0SameRowOrderingRule):
    rule_id = "RE0039"
    description = "Associated Gas Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta G_{ps}^{a \text{ 3P}} \leq \Delta G_{pn}^{a \text{ 3R}}$"
    low_col = "res_ga"
    high_col = "rec_ga"
    uncert = UncertLevel.HIGH


# --- Non-Associated Gas ---


@register_rule
class RE0040(RE0SameRowOrderingRule):
    rule_id = "RE0040"
    description = "Non Associated Gas Reserves: 1P must be less than or equal to 1R"
    formal = r"$\Delta G_{ps}^{\text{1P}} \leq \Delta G_{pn}^{\text{1R}}$"
    low_col = "res_gn"
    high_col = "rec_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE0041(RE0SameRowOrderingRule):
    rule_id = "RE0041"
    description = "Non Associated Gas Reserves: 2P must be less than or equal to 2R"
    formal = r"$\Delta G_{ps}^{\text{2P}} \leq \Delta G_{pn}^{\text{2R}}$"
    low_col = "res_gn"
    high_col = "rec_gn"
    uncert = UncertLevel.MID


@register_rule
class RE0042(RE0SameRowOrderingRule):
    rule_id = "RE0042"
    description = "Non Associated Gas Reserves: 3P must be less than or equal to 3R"
    formal = r"$\Delta G_{ps}^{\text{3P}} \leq \Delta G_{pn}^{\text{3R}}$"
    low_col = "res_gn"
    high_col = "rec_gn"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category E: Implication checks (4 rules)
# ---------------------------------------------------------------------------


class RE0ImplicationRule(ValidationRule):
    """Base: if cond_col > 0 then result_col > 0.

    Violation when cond_col > 0 but result_col == 0 across two
    uncert_level rows for the same project+year.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    cond_col: str
    result_col: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_implication_sql(
            self.cond_col, self.cond_uncert, self.result_col, self.result_uncert
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
    cond_col = "res_oil"
    result_col = "res_oil"
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
    cond_col = "res_con"
    result_col = "res_con"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE0051(RE0ImplicationRule):
    rule_id = "RE0051"
    description = "Associated Gas Reserves: 1P should be greater than zero if 3P is greater than zero"  # noqa: E501
    formal = (
        r"$\Delta G_{ps}^{a \text{ 3P}} > 0  \implies \Delta G_{ps}^{a \text{ 1P}} > 0$"
    )
    cond_col = "res_ga"
    result_col = "res_ga"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE0052(RE0ImplicationRule):
    rule_id = "RE0052"
    description = "Non Associated Gas Reserves: 1P should be greater than zero if 3P is greater than zero"  # noqa: E501
    formal = r"$\Delta G_{ps}^{\text{3P}} > 0  \implies \Delta G_{ps}^{\text{1P}} > 0$"
    cond_col = "res_gn"
    result_col = "res_gn"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


# ---------------------------------------------------------------------------
# Category F: Reserve + CumProd < In-Place (6 rules)
# ---------------------------------------------------------------------------


class RE0ReserveVsPlaceRule(ValidationRule):
    """Base: if place > 0, then reserve + cumprod < place.

    Violation when place_col > 0 and
    COALESCE(reserve_col, 0) + COALESCE(cumprod_col, 0) >= place_col.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    place_col: str
    reserve_col: str
    cumprod_col: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_reserve_vs_place_sql(
            self.place_col, self.reserve_col, self.cumprod_col, self.uncert
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
    place_col = "prj_ioip"
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
    place_col = "prj_ioip"
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
    place_col = "prj_ioip"
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
    place_col = "prj_igip"
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
    place_col = "prj_igip"
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
    place_col = "prj_igip"
    reserve_col = "res_gn"
    cumprod_col = "cprd_grs_gn"
    uncert = UncertLevel.HIGH
