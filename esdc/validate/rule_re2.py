"""RE2xxx rules: Material Balance validation.

This module registers 30 validation rules (RE2001-RE2030) covering:

- Material balance for GRR/CR/PR (RE2001-RE2012)
- Material balance for Reserves (RE2013-RE2024)
- Cross-table EUR bounds and implication checks (RE2025-RE2030)

Each category has an abstract parent class that concrete rules inherit
from, sharing the check/generate_fixes logic while differing only in
column names and uncertainty levels.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import duckdb

from esdc.selection import Severity
from esdc.validate.rule_re0_helpers import (
    AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
    UncertLevel,
    _execute_and_build_violations,
)
from esdc.validate.rule_re2_helpers import (
    build_eur_bounds_sql,
    build_eur_greater_than_sql,
    build_eur_implication_sql,
    build_material_balance_sql,
)
from esdc.validate.rules import ValidationRule, Violation, register_rule

# ---------------------------------------------------------------------------
# Category A: Material Balance — GRR/CR/PR (12 rules)
# ---------------------------------------------------------------------------


class RE2MaterialBalanceRule(ValidationRule):
    """Base: current validated_col equals previous + discrepancies - delta.

    Self-joins project_resources on project_id across consecutive
    report years. Violation when ABS(curr - prev - sum(dcpy) +
    curr_prod - prev_prod) > TOLERANCE.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    dcpy_columns: list[str]
    production_column: str
    uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_material_balance_sql(
            self.validated_column,
            self.dcpy_columns,
            self.production_column,
            self.uncert,
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
            compared_columns=self.dcpy_columns + [self.production_column],
            rule_group="RE2",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# --- Oil GRR/CR/PR ---


@register_rule
class RE2001(RE2MaterialBalanceRule):
    rule_id = "RE2001"
    description = (
        "Oil GRR/CR/PR: Current 1R/1C/1U must be consistent with "
        "previous 1R/1C/1U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P90}}"
        r" = \Delta N_{pn,t - 1}^{\text{P90}}"
        r" + \Delta D_{N}^\text{um P90}"
        r" + \Delta D_{N}^\text{ppa P90}"
        r" + \Delta D_{N}^\text{wi P90}"
        r" + \Delta D_{N}^\text{uc P90}"
        r" + \Delta D_{N}^\text{cio P90}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "rec_oil"
    dcpy_columns = [
        "dcpy_um_oil", "dcpy_ppa_oil", "dcpy_wi_oil",
        "dcpy_uc_oil", "dcpy_cio_oil",
    ]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE2005(RE2MaterialBalanceRule):
    rule_id = "RE2005"
    description = (
        "Oil GRR/CR/PR: Current 2R/2C/2U must be consistent with "
        "previous 2R/2C/2U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P50}}"
        r" = \Delta N_{pn,t - 1}^{\text{P50}}"
        r" + \Delta D_{N}^\text{um P50}"
        r" + \Delta D_{N}^\text{ppa P50}"
        r" + \Delta D_{N}^\text{wi P50}"
        r" + \Delta D_{N}^\text{uc P50}"
        r" + \Delta D_{N}^\text{cio P50}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "rec_oil"
    dcpy_columns = [
        "dcpy_um_oil", "dcpy_ppa_oil", "dcpy_wi_oil",
        "dcpy_uc_oil", "dcpy_cio_oil",
    ]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.MID


@register_rule
class RE2009(RE2MaterialBalanceRule):
    rule_id = "RE2009"
    description = (
        "Oil GRR/CR/PR: Current 3R/3C/3U must be consistent with "
        "previous 3R/3C/3U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P10}}"
        r" = \Delta N_{pn,t - 1}^{\text{P10}}"
        r" + \Delta D_{N}^\text{um P10}"
        r" + \Delta D_{N}^\text{ppa P10}"
        r" + \Delta D_{N}^\text{wi P10}"
        r" + \Delta D_{N}^\text{uc P10}"
        r" + \Delta D_{N}^\text{cio P10}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "rec_oil"
    dcpy_columns = [
        "dcpy_um_oil", "dcpy_ppa_oil", "dcpy_wi_oil",
        "dcpy_uc_oil", "dcpy_cio_oil",
    ]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.HIGH


# --- Condensate GRR/CR/PR ---


@register_rule
class RE2002(RE2MaterialBalanceRule):
    rule_id = "RE2002"
    description = (
        "Condensate GRR/CR/PR: Current 1R/1C/1U must be consistent with "
        "previous 1R/1C/1U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P90}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P90}}"
        r" + \Delta D_{N^c}^\text{um P90}"
        r" + \Delta D_{N^c}^\text{ppa P90}"
        r" + \Delta D_{N^c}^\text{wi P90}"
        r" + \Delta D_{N^c}^\text{uc P90}"
        r" + \Delta D_{N^c}^\text{cio P90}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "rec_con"
    dcpy_columns = [
        "dcpy_um_con", "dcpy_ppa_con", "dcpy_wi_con",
        "dcpy_uc_con", "dcpy_cio_con",
    ]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.LOW


@register_rule
class RE2006(RE2MaterialBalanceRule):
    rule_id = "RE2006"
    description = (
        "Condensate GRR/CR/PR: Current 2R/2C/2U must be consistent with "
        "previous 2R/2C/2U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P50}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P50}}"
        r" + \Delta D_{N^c}^\text{um P50}"
        r" + \Delta D_{N^c}^\text{ppa P50}"
        r" + \Delta D_{N^c}^\text{wi P50}"
        r" + \Delta D_{N^c}^\text{uc P50}"
        r" + \Delta D_{N^c}^\text{cio P50}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "rec_con"
    dcpy_columns = [
        "dcpy_um_con", "dcpy_ppa_con", "dcpy_wi_con",
        "dcpy_uc_con", "dcpy_cio_con",
    ]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.MID


@register_rule
class RE2010(RE2MaterialBalanceRule):
    rule_id = "RE2010"
    description = (
        "Condensate GRR/CR/PR: Current 3R/3C/3U must be consistent with "
        "previous 3R/3C/3U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P10}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P10}}"
        r" + \Delta D_{N^c}^\text{um P10}"
        r" + \Delta D_{N^c}^\text{ppa P10}"
        r" + \Delta D_{N^c}^\text{wi P10}"
        r" + \Delta D_{N^c}^\text{uc P10}"
        r" + \Delta D_{N^c}^\text{cio P10}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "rec_con"
    dcpy_columns = [
        "dcpy_um_con", "dcpy_ppa_con", "dcpy_wi_con",
        "dcpy_uc_con", "dcpy_cio_con",
    ]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.HIGH


# --- Associated Gas GRR/CR/PR ---


@register_rule
class RE2003(RE2MaterialBalanceRule):
    rule_id = "RE2003"
    description = (
        "Associated Gas GRR/CR/PR: Current 1R/1C/1U must be consistent with "
        "previous 1R/1C/1U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P90}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P90}}"
        r" + \Delta D_{G^a}^\text{um P90}"
        r" + \Delta D_{G^a}^\text{ppa P90}"
        r" + \Delta D_{G^a}^\text{wi P90}"
        r" + \Delta D_{G^a}^\text{uc P90}"
        r" + \Delta D_{G^a}^\text{cio P90}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "rec_ga"
    dcpy_columns = [
        "dcpy_um_ga", "dcpy_ppa_ga", "dcpy_wi_ga",
        "dcpy_uc_ga", "dcpy_cio_ga",
    ]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE2007(RE2MaterialBalanceRule):
    rule_id = "RE2007"
    description = (
        "Associated Gas GRR/CR/PR: Current 2R/2C/2U must be consistent with "
        "previous 2R/2C/2U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P50}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P50}}"
        r" + \Delta D_{G^a}^\text{um P50}"
        r" + \Delta D_{G^a}^\text{ppa P50}"
        r" + \Delta D_{G^a}^\text{wi P50}"
        r" + \Delta D_{G^a}^\text{uc P50}"
        r" + \Delta D_{G^a}^\text{cio P50}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "rec_ga"
    dcpy_columns = [
        "dcpy_um_ga", "dcpy_ppa_ga", "dcpy_wi_ga",
        "dcpy_uc_ga", "dcpy_cio_ga",
    ]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.MID


@register_rule
class RE2011(RE2MaterialBalanceRule):
    rule_id = "RE2011"
    description = (
        "Associated Gas GRR/CR/PR: Current 3R/3C/3U must be consistent with "
        "previous 3R/3C/3U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P10}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P10}}"
        r" + \Delta D_{G^a}^\text{um P10}"
        r" + \Delta D_{G^a}^\text{ppa P10}"
        r" + \Delta D_{G^a}^\text{wi P10}"
        r" + \Delta D_{G^a}^\text{uc P10}"
        r" + \Delta D_{G^a}^\text{cio P10}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "rec_ga"
    dcpy_columns = [
        "dcpy_um_ga", "dcpy_ppa_ga", "dcpy_wi_ga",
        "dcpy_uc_ga", "dcpy_cio_ga",
    ]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.HIGH


# --- Non Associated Gas GRR/CR/PR ---


@register_rule
class RE2004(RE2MaterialBalanceRule):
    rule_id = "RE2004"
    description = (
        "Non Associated Gas GRR/CR/PR: Current 1R/1C/1U must be consistent "
        "with previous 1R/1C/1U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P90}}"
        r" = \Delta G_{pn,t - 1}^{\text{P90}}"
        r" + \Delta D_{G}^\text{um P90}"
        r" + \Delta D_{G}^\text{ppa P90}"
        r" + \Delta D_{G}^\text{wi P90}"
        r" + \Delta D_{G}^\text{uc P90}"
        r" + \Delta D_{G}^\text{cio P90}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "rec_gn"
    dcpy_columns = [
        "dcpy_um_gn", "dcpy_ppa_gn", "dcpy_wi_gn",
        "dcpy_uc_gn", "dcpy_cio_gn",
    ]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE2008(RE2MaterialBalanceRule):
    rule_id = "RE2008"
    description = (
        "Non Associated Gas GRR/CR/PR: Current 2R/2C/2U must be consistent "
        "with previous 2R/2C/2U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P50}}"
        r" = \Delta G_{pn,t - 1}^{\text{P50}}"
        r" + \Delta D_{G}^\text{um P50}"
        r" + \Delta D_{G}^\text{ppa P50}"
        r" + \Delta D_{G}^\text{wi P50}"
        r" + \Delta D_{G}^\text{uc P50}"
        r" + \Delta D_{G}^\text{cio P50}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "rec_gn"
    dcpy_columns = [
        "dcpy_um_gn", "dcpy_ppa_gn", "dcpy_wi_gn",
        "dcpy_uc_gn", "dcpy_cio_gn",
    ]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.MID


@register_rule
class RE2012(RE2MaterialBalanceRule):
    rule_id = "RE2012"
    description = (
        "Non Associated Gas GRR/CR/PR: Current 3R/3C/3U must be consistent "
        "with previous 3R/3C/3U, all discrepancies, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P10}}"
        r" = \Delta G_{pn,t - 1}^{\text{P10}}"
        r" + \Delta D_{G}^\text{um P10}"
        r" + \Delta D_{G}^\text{ppa P10}"
        r" + \Delta D_{G}^\text{wi P10}"
        r" + \Delta D_{G}^\text{uc P10}"
        r" + \Delta D_{G}^\text{cio P10}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "rec_gn"
    dcpy_columns = [
        "dcpy_um_gn", "dcpy_ppa_gn", "dcpy_wi_gn",
        "dcpy_uc_gn", "dcpy_cio_gn",
    ]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category B: Material Balance — Reserves (12 rules)
# ---------------------------------------------------------------------------


# --- Oil Reserves ---


@register_rule
class RE2013(RE2MaterialBalanceRule):
    rule_id = "RE2013"
    description = (
        "Oil Reserves: Current 1P must be consistent with previous 1P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P90}}"
        r" = \Delta N_{pn,t - 1}^{\text{P90}}"
        r" + \Delta D_{N}^\text{gtr P90}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "res_oil"
    dcpy_columns = ["dcpy_gtr_oil"]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.LOW


@register_rule
class RE2017(RE2MaterialBalanceRule):
    rule_id = "RE2017"
    description = (
        "Oil Reserves: Current 2P must be consistent with previous 2P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P50}}"
        r" = \Delta N_{pn,t - 1}^{\text{P50}}"
        r" + \Delta D_{N}^\text{gtr P50}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "res_oil"
    dcpy_columns = ["dcpy_gtr_oil"]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.MID


@register_rule
class RE2021(RE2MaterialBalanceRule):
    rule_id = "RE2021"
    description = (
        "Oil Reserves: Current 3P must be consistent with previous 3P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{\text{P10}}"
        r" = \Delta N_{pn,t - 1}^{\text{P10}}"
        r" + \Delta D_{N}^\text{gtr P10}"
        r" - \left(N_{ps,t} - N_{ps,t - 1}\right)$"
    )
    validated_column = "res_oil"
    dcpy_columns = ["dcpy_gtr_oil"]
    production_column = "cprd_sls_oil"
    uncert = UncertLevel.HIGH


# --- Condensate Reserves ---


@register_rule
class RE2014(RE2MaterialBalanceRule):
    rule_id = "RE2014"
    description = (
        "Condensate Reserves: Current 1P must be consistent with previous 1P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P90}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P90}}"
        r" + \Delta D_{N^c}^\text{gtr P90}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "res_con"
    dcpy_columns = ["dcpy_gtr_con"]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.LOW


@register_rule
class RE2018(RE2MaterialBalanceRule):
    rule_id = "RE2018"
    description = (
        "Condensate Reserves: Current 2P must be consistent with previous 2P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P50}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P50}}"
        r" + \Delta D_{N^c}^\text{gtr P50}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "res_con"
    dcpy_columns = ["dcpy_gtr_con"]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.MID


@register_rule
class RE2022(RE2MaterialBalanceRule):
    rule_id = "RE2022"
    description = (
        "Condensate Reserves: Current 3P must be consistent with previous 3P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta N_{pn,t}^{c \text{P10}}"
        r" = \Delta N_{pn,t - 1}^{c \text{P10}}"
        r" + \Delta D_{N^c}^\text{gtr P10}"
        r" - \left(N_{ps,t}^c - N_{ps,t - 1}^c\right)$"
    )
    validated_column = "res_con"
    dcpy_columns = ["dcpy_gtr_con"]
    production_column = "cprd_sls_con"
    uncert = UncertLevel.HIGH


# --- Associated Gas Reserves ---


@register_rule
class RE2015(RE2MaterialBalanceRule):
    rule_id = "RE2015"
    description = (
        "Associated Gas Reserves: Current 1P must be consistent with previous 1P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P90}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P90}}"
        r" + \Delta D_{G^a}^\text{gtr P90}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "res_ga"
    dcpy_columns = ["dcpy_gtr_ga"]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.LOW


@register_rule
class RE2019(RE2MaterialBalanceRule):
    rule_id = "RE2019"
    description = (
        "Associated Gas Reserves: Current 2P must be consistent with previous 2P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P50}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P50}}"
        r" + \Delta D_{G^a}^\text{gtr P50}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "res_ga"
    dcpy_columns = ["dcpy_gtr_ga"]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.MID


@register_rule
class RE2023(RE2MaterialBalanceRule):
    rule_id = "RE2023"
    description = (
        "Associated Gas Reserves: Current 3P must be consistent with previous 3P, "
        "Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{a \text{P10}}"
        r" = \Delta G_{pn,t - 1}^{a \text{P10}}"
        r" + \Delta D_{G^a}^\text{gtr P10}"
        r" - \left(G_{ps,t}^a - G_{ps,t - 1}^a\right)$"
    )
    validated_column = "res_ga"
    dcpy_columns = ["dcpy_gtr_ga"]
    production_column = "cprd_sls_ga"
    uncert = UncertLevel.HIGH


# --- Non Associated Gas Reserves ---


@register_rule
class RE2016(RE2MaterialBalanceRule):
    rule_id = "RE2016"
    description = (
        "Non Associated Gas Reserves: Current 1P must be consistent with "
        "previous 1P, Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P90}}"
        r" = \Delta G_{pn,t - 1}^{\text{P90}}"
        r" + \Delta D_{G}^\text{gtr P90}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "res_gn"
    dcpy_columns = ["dcpy_gtr_gn"]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.LOW


@register_rule
class RE2020(RE2MaterialBalanceRule):
    rule_id = "RE2020"
    description = (
        "Non Associated Gas Reserves: Current 2P must be consistent with "
        "previous 2P, Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P50}}"
        r" = \Delta G_{pn,t - 1}^{\text{P50}}"
        r" + \Delta D_{G}^\text{gtr P50}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "res_gn"
    dcpy_columns = ["dcpy_gtr_gn"]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.MID


@register_rule
class RE2024(RE2MaterialBalanceRule):
    rule_id = "RE2024"
    description = (
        "Non Associated Gas Reserves: Current 3P must be consistent with "
        "previous 3P, Change from Commerciality, and production"
    )
    formal = (
        r"$\Delta G_{pn,t}^{\text{P10}}"
        r" = \Delta G_{pn,t - 1}^{\text{P10}}"
        r" + \Delta D_{G}^\text{gtr P10}"
        r" - \left(G_{ps,t} - G_{ps,t - 1}\right)$"
    )
    validated_column = "res_gn"
    dcpy_columns = ["dcpy_gtr_gn"]
    production_column = "cprd_sls_gn"
    uncert = UncertLevel.HIGH


# ---------------------------------------------------------------------------
# Category C: Cross-table EUR bounds and implication checks (6 rules)
# ---------------------------------------------------------------------------


class RE2EurBoundsRule(ValidationRule):
    """Base: field in-place > EUR when EUR > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). fr_result at result_uncert, fr_cond at cond_uncert.
    EUR = rec + cprd_sls (computed).
    """

    is_fixable = False
    applies_to_tables = ["field_resources"]
    severity = Severity.WARNING
    field_column: str
    eur_rec_column: str
    eur_cprd_column: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_eur_bounds_sql(
            self.field_column,
            self.eur_rec_column,
            self.eur_cprd_column,
            self.cond_uncert,
            self.result_uncert,
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
            identifier_cols=AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
            validated_column=self.field_column,
            compared_columns=[self.eur_rec_column, self.eur_cprd_column],
            rule_group="RE2",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE2EurImplicationRule(ValidationRule):
    """Base: if EUR > 0 then field > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). fr_result at result_uncert, fr_cond at cond_uncert.
    Violation when EUR > 0 but field_column = 0.
    """

    is_fixable = False
    applies_to_tables = ["field_resources"]
    severity = Severity.STRICT
    field_column: str
    eur_rec_column: str
    eur_cprd_column: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_eur_implication_sql(
            self.field_column,
            self.eur_rec_column,
            self.eur_cprd_column,
            self.cond_uncert,
            self.result_uncert,
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
            identifier_cols=AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
            validated_column=self.field_column,
            compared_columns=[self.eur_rec_column, self.eur_cprd_column],
            rule_group="RE2",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE2EurGreaterThanRule(ValidationRule):
    """Base: field > EUR when EUR > 0.

    Self-joins field_resources on (wk_id, field_id, report_year, project_stage,
    project_class). Both sides at the same uncert_level (row joins with itself).
    Violation when field_column <= EUR given EUR > 0.
    """

    is_fixable = False
    applies_to_tables = ["field_resources"]
    severity = Severity.STRICT
    field_column: str
    eur_rec_column: str
    eur_cprd_column: str
    cond_uncert: UncertLevel
    result_uncert: UncertLevel

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_eur_greater_than_sql(
            self.field_column,
            self.eur_rec_column,
            self.eur_cprd_column,
            self.cond_uncert,
            self.result_uncert,
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
            identifier_cols=AGGREGATION_CONSISTENCY_IDENTIFIER_COLS,
            validated_column=self.field_column,
            compared_columns=[self.eur_rec_column, self.eur_cprd_column],
            rule_group="RE2",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE2025(RE2EurBoundsRule):
    rule_id = "RE2025"
    description = (
        "IOIP Mid: IOIP Mid should be greater than Sum of all projects "
        "Oil Ultimate GRR/CR/PR 3R/3C/3U if Sum greater than zero"
    )
    formal = (
        r"$\sum_{i=1}^n \left(\Delta N_{pn,i}^{\text{P10}}"
        r" + N_{ps,i}\right) > 0"
        r" \implies N^{\text{P50}}"
        r" > \sum_{i=1}^n \left(\Delta N_{pn,i}^{\text{P10}}"
        r" + N_{ps,i}\right)$"
    )
    field_column = "ioip"
    eur_rec_column = "rec_oil"
    eur_cprd_column = "cprd_sls_oil"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.MID


@register_rule
class RE2026(RE2EurBoundsRule):
    rule_id = "RE2026"
    description = (
        "IGIP Mid: IGIP Mid should be greater than Sum of all projects "
        "Non Associated Gas Ultimate 3R/3C/3U if Sum greater than zero"
    )
    formal = (
        r"$\sum_{i=1}^n \left(\Delta G_{pn,i}^{\text{P10}}"
        r" + G_{ps,i}\right) > 0"
        r" \implies G^{\text{P50}}"
        r" > \sum_{i=1}^n \left(\Delta G_{pn,i}^{\text{P10}}"
        r" + G_{ps,i}\right)$"
    )
    field_column = "igip"
    eur_rec_column = "rec_gn"
    eur_cprd_column = "cprd_sls_gn"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.MID


@register_rule
class RE2027(RE2EurImplicationRule):
    rule_id = "RE2027"
    description = (
        "IGIP Low: IGIP Low Case must be greater than zero "
        "if Sum of all projects Condensate GRR/CR/PR 3R/3C/3U greater than zero"
    )
    formal = (
        r"$\sum_{i=1}^n \left(\Delta N_{pn,i}^{c \text{P10}}"
        r" + N_{ps,i}^c\right) > 0"
        r" \implies G^{\text{P90}} > 0$"
    )
    field_column = "igip"
    eur_rec_column = "rec_con"
    eur_cprd_column = "cprd_sls_con"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE2028(RE2EurImplicationRule):
    rule_id = "RE2028"
    description = (
        "IOIP Low: IOIP Low Case must be greater than zero "
        "if Sum of all projects Associated Gas GRR/CR/PR 3R/3C/3U greater than zero"
    )
    formal = (
        r"$\sum_{i=1}^n \left(\Delta G_{pn,i}^{a \text{P10}}"
        r" + G_{ps,i}^a\right) > 0"
        r" \implies N^{\text{P90}} > 0$"
    )
    field_column = "ioip"
    eur_rec_column = "rec_ga"
    eur_cprd_column = "cprd_sls_ga"
    cond_uncert = UncertLevel.HIGH
    result_uncert = UncertLevel.LOW


@register_rule
class RE2029(RE2EurGreaterThanRule):
    rule_id = "RE2029"
    description = (
        "IOIP Low: IOIP Low Case must be greater than "
        "sum of all projects Oil EUR GRR/CR/PR 1R/1C/1U"
    )
    formal = (
        r"$N^{\text{P90}}"
        r" > \sum_{i=1}^n \left(\Delta N_{pn,i}^{\text{P90}}"
        r" + N_{ps,i}\right) > 0$"
    )
    field_column = "ioip"
    eur_rec_column = "rec_oil"
    eur_cprd_column = "cprd_sls_oil"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.LOW


@register_rule
class RE2030(RE2EurGreaterThanRule):
    rule_id = "RE2030"
    description = (
        "IGIP Low: IGIP Low Case must be greater than "
        "sum of all projects Non Associated Gas EUR GRR/CR/PR 1R/1C/1U"
    )
    formal = (
        r"$G^{\text{P90}}"
        r" > \sum_{i=1}^n \left(\Delta G_{pn,i}^{\text{P90}}"
        r" + G_{ps,i}\right) > 0$"
    )
    field_column = "igip"
    eur_rec_column = "rec_gn"
    eur_cprd_column = "cprd_sls_gn"
    cond_uncert = UncertLevel.LOW
    result_uncert = UncertLevel.LOW
