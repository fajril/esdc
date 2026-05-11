"""RE1xxx rules: Production and Forecast validation.

This module registers 32 validation rules (RE1001-RE1044, skipping Net cumprod
rules and WP&B cross-checks) covering:

- Non-negative checks for cumulative production (RE1001-RE1004, RE1009-RE1012)
- Monotonic ordering: current cumprod >= previous cumprod (RE1013-RE1016,
  RE1021-RE1024)
- Sales volume <= Gross volume (RE1029-RE1032)
- Sales Forecast <= Total Potential Forecast per year (RE1033-RE1036)
- Sum of Sales Forecast = 2P Reserves (RE1037-RE1040)
- Sum of Total Potential Forecast = 2R GRR/CR/PR (RE1041-RE1044)

Each category has an abstract parent class that concrete rules inherit from,
sharing the check/generate_fixes logic while differing only in column names.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import duckdb

from esdc.selection import Severity
from esdc.validate.rule_re0_helpers import (
    UncertLevel,
    _execute_and_build_violations,
    build_non_negative_sql,
    build_same_row_ordering_sql,
)
from esdc.validate.rule_re1_helpers import (
    TS_IDENTIFIER_COLS,
    build_forecast_sum_equals_reserve_sql,
    build_forecast_sum_equals_resource_sql,
    build_monotonic_sql,
    build_timeseries_sales_le_tpf_sql,
)
from esdc.validate.rules import ValidationRule, Violation, register_rule

# ---------------------------------------------------------------------------
# Category A: Non-negative checks (8 rules)
# ---------------------------------------------------------------------------


class RE1NonNegativeRule(ValidationRule):
    """Cumulative production value must be >= 0 at MID uncertainty.

    Violation when COALESCE(column, 0) < -TOLERANCE at MID uncert_level.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    uncert: UncertLevel = UncertLevel.MID

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
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1001(RE1NonNegativeRule):
    rule_id = "RE1001"
    description = "Oil Gross Cumprod: Must be greater than or equal to zero"
    formal = r"$N_{pg} \geq 0$"
    validated_column = "cprd_grs_oil"


@register_rule
class RE1002(RE1NonNegativeRule):
    rule_id = "RE1002"
    description = "Condensate Gross Cumprod: Must be greater than or equal to zero"
    formal = r"$N_{pg}^c \geq 0$"
    validated_column = "cprd_grs_con"


@register_rule
class RE1003(RE1NonNegativeRule):
    rule_id = "RE1003"
    description = "Associated Gas Gross Cumprod: Must be greater than or equal to zero"
    formal = r"$G_{pg}^a \geq 0$"
    validated_column = "cprd_grs_ga"


@register_rule
class RE1004(RE1NonNegativeRule):
    rule_id = "RE1004"
    description = (
        "Non Associated Gas Gross Cumprod: Must be greater than or equal to zero"
    )
    formal = r"$G_{pg} \geq 0$"
    validated_column = "cprd_grs_gn"


@register_rule
class RE1009(RE1NonNegativeRule):
    rule_id = "RE1009"
    description = "Oil Sales Cumprod: Must be greater than or equal to zero"
    formal = r"$N_{ps} \geq 0$"
    validated_column = "cprd_sls_oil"


@register_rule
class RE1010(RE1NonNegativeRule):
    rule_id = "RE1010"
    description = "Condensate Sales Cumprod: Must be greater than or equal to zero"
    formal = r"$N_{ps}^c \geq 0$"
    validated_column = "cprd_sls_con"


@register_rule
class RE1011(RE1NonNegativeRule):
    rule_id = "RE1011"
    description = "Associated Gas Sales Cumprod: Must be greater than or equal to zero"
    formal = r"$G_{ps}^a \geq 0$"
    validated_column = "cprd_sls_ga"


@register_rule
class RE1012(RE1NonNegativeRule):
    rule_id = "RE1012"
    description = (
        "Non Associated Gas Sales Cumprod: Must be greater than or equal to zero"
    )
    formal = r"$G_{ps} \geq 0$"
    validated_column = "cprd_sls_gn"


# ---------------------------------------------------------------------------
# Category B: Monotonic ordering — current >= previous (8 rules)
# ---------------------------------------------------------------------------


class RE1MonotonicRule(ValidationRule):
    """Cumulative production must be monotonically non-decreasing across years.

    Violation when previous cumprod - current cumprod > TOLERANCE.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    uncert: UncertLevel = UncertLevel.MID

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_monotonic_sql(self.validated_column, self.uncert)
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
            compared_columns=[self.validated_column],
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1013(RE1MonotonicRule):
    rule_id = "RE1013"
    description = "Oil Gross Cumprod: Must be greater than or equal to previous Cumprod"
    formal = r"$N_{pg, t} \geq N_{pg, t - 1}$"
    validated_column = "cprd_grs_oil"


@register_rule
class RE1014(RE1MonotonicRule):
    rule_id = "RE1014"
    description = (
        "Condensate Gross Cumprod: Must be greater than or equal to previous Cumprod"
    )
    formal = r"$N_{pg, t}^c \geq N_{pg, t - 1}^c$"
    validated_column = "cprd_grs_con"


@register_rule
class RE1015(RE1MonotonicRule):
    rule_id = "RE1015"
    description = "Associated Gas Gross Cumprod: Must be greater than or equal to previous Cumprod"  # noqa: E501
    formal = r"$G_{pg, t}^a \geq G_{pg, t - 1}^a$"
    validated_column = "cprd_grs_ga"


@register_rule
class RE1016(RE1MonotonicRule):
    rule_id = "RE1016"
    description = "Non Associated Gas Gross Cumprod: Must be greater than or equal to previous Cumprod"  # noqa: E501
    formal = r"$G_{pg, t} \geq G_{pg, t - 1}$"
    validated_column = "cprd_grs_gn"


@register_rule
class RE1021(RE1MonotonicRule):
    rule_id = "RE1021"
    description = "Oil Sales Cumprod: Must be greater than or equal to previous Cumprod"
    formal = r"$N_{ps, t} \geq N_{ps, t - 1}$"
    validated_column = "cprd_sls_oil"


@register_rule
class RE1022(RE1MonotonicRule):
    rule_id = "RE1022"
    description = (
        "Condensate Sales Cumprod: Must be greater than or equal to previous Cumprod"
    )
    formal = r"$N_{ps, t}^c \geq N_{ps, t - 1}^c$"
    validated_column = "cprd_sls_con"


@register_rule
class RE1023(RE1MonotonicRule):
    rule_id = "RE1023"
    description = "Associated Gas Sales Cumprod: Must be greater than or equal to previous Cumprod"  # noqa: E501
    formal = r"$G_{ps, t}^a \geq G_{ps, t - 1}^a$"
    validated_column = "cprd_sls_ga"


@register_rule
class RE1024(RE1MonotonicRule):
    rule_id = "RE1024"
    description = "Non Associated Gas Sales Cumprod: Must be greater than or equal to previous Cumprod"  # noqa: E501
    formal = r"$G_{ps, t} \geq G_{ps, t - 1}$"
    validated_column = "cprd_sls_gn"


# ---------------------------------------------------------------------------
# Category C: Sales <= Gross (4 rules)
# ---------------------------------------------------------------------------


class RE1SalesVsGrossRule(ValidationRule):
    """Sales cumprod must be <= Gross cumprod at MID uncertainty.

    Violation when COALESCE(sales, 0) - COALESCE(gross, 0) > TOLERANCE.
    """

    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    uncert: UncertLevel = UncertLevel.MID

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_same_row_ordering_sql(
            self.validated_column, self.compared_column, self.uncert,
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
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1029(RE1SalesVsGrossRule):
    rule_id = "RE1029"
    description = "Oil Cumprod: Sales Volume must be less than or equal to Gross Volume"
    formal = r"$N_{ps} \leq N_{pg}$"
    validated_column = "cprd_sls_oil"
    compared_column = "cprd_grs_oil"


@register_rule
class RE1030(RE1SalesVsGrossRule):
    rule_id = "RE1030"
    description = (
        "Condensate Cumprod: Sales Volume must be less than or equal to Gross Volume"
    )
    formal = r"$N_{ps}^c \leq N_{pg}^c$"
    validated_column = "cprd_sls_con"
    compared_column = "cprd_grs_con"


@register_rule
class RE1031(RE1SalesVsGrossRule):
    rule_id = "RE1031"
    description = "Associated Gas Cumprod: Sales Volume must be less than or equal to Gross Volume"  # noqa: E501
    formal = r"$G_{ps}^a \leq G_{pg}^a$"
    validated_column = "cprd_sls_ga"
    compared_column = "cprd_grs_ga"


@register_rule
class RE1032(RE1SalesVsGrossRule):
    rule_id = "RE1032"
    description = "Non Associated Gas Cumprod: Sales Volume must be less than or equal to Gross Volume"  # noqa: E501
    formal = r"$G_{ps} \leq G_{pg}$"
    validated_column = "cprd_sls_gn"
    compared_column = "cprd_grs_gn"


# ---------------------------------------------------------------------------
# Category D: Sales Forecast <= Total Potential Forecast (4 rules)
# ---------------------------------------------------------------------------


class RE1TimeseriesSalesVsTpfRule(ValidationRule):
    """Yearly Sales Forecast must be <= Yearly Total Potential Forecast.

    Violation when COALESCE(sales, 0) - COALESCE(tpf, 0) > TOLERANCE.
    """

    is_fixable = False
    applies_to_tables = ["project_timeseries"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_timeseries_sales_le_tpf_sql(
            self.validated_column, self.compared_column,
        )
        return _execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_timeseries",
            year=year,
            extra_columns=["val_ref", "val_cmp"],
            identifier_cols=TS_IDENTIFIER_COLS,
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1033(RE1TimeseriesSalesVsTpfRule):
    rule_id = "RE1033"
    description = "Oil Sales Forecast: Yearly Sales Volume must be less than or equal to Yearly Total Potential Volume"  # noqa: E501
    formal = (
        r"$\forall t \in \lbrace t_R + 1, \dots , t_m \rbrace \mid "
        r"q_{o, t}^{s} \leq q_{o, t}^{\text{tp}}$"
    )
    validated_column = "slf_oil"
    compared_column = "tpf_oil"


@register_rule
class RE1034(RE1TimeseriesSalesVsTpfRule):
    rule_id = "RE1034"
    description = "Condensate Sales Forecast: Yearly Sales Volume must be less than or equal to Yearly Total Potential Volume"  # noqa: E501
    formal = (
        r"$\forall t \in \lbrace t_R + 1, \dots , t_m \rbrace \mid "
        r"q_{c, t}^{s} \leq q_{c, t}^{\text{tp}}$"
    )
    validated_column = "slf_con"
    compared_column = "tpf_con"


@register_rule
class RE1035(RE1TimeseriesSalesVsTpfRule):
    rule_id = "RE1035"
    description = "Associated Gas Sales Forecast: Yearly Sales Volume must be less than or equal to Yearly Total Potential Volume"  # noqa: E501
    formal = (
        r"$\forall t \in \lbrace t_R + 1, \dots , t_m \rbrace \mid "
        r"q_{a, t}^{s} \leq q_{a, t}^{\text{tp}}$"
    )
    validated_column = "slf_ga"
    compared_column = "tpf_ga"


@register_rule
class RE1036(RE1TimeseriesSalesVsTpfRule):
    rule_id = "RE1036"
    description = "Non Associated Gas Sales Forecast: Yearly Sales Volume must be less than or equal to Yearly Total Potential Volume"  # noqa: E501
    formal = (
        r"$\forall t \in \lbrace t_R + 1, \dots , t_m \rbrace \mid "
        r"q_{n, t}^{s} \leq q_{n, t}^{\text{tp}}$"
    )
    validated_column = "slf_gn"
    compared_column = "tpf_gn"


# ---------------------------------------------------------------------------
# Category E: Sum of Sales Forecast = 2P Reserves (4 rules)
# ---------------------------------------------------------------------------


class RE1ForecastSumReserveRule(ValidationRule):
    """Sum of yearly Sales Forecast must equal 2P Reserves."""

    is_fixable = False
    applies_to_tables = ["project_resources", "project_timeseries"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    uncert: UncertLevel = UncertLevel.MID

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_forecast_sum_equals_reserve_sql(
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
            extra_columns=["val_sum", "val_ref"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1037(RE1ForecastSumReserveRule):
    rule_id = "RE1037"
    description = (
        "Oil Sales Forecast: Sum of Yearly Forecast must be equal to 2P Reserves"
    )
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{o, t}^{s} = "
        r"\Delta N_{ps}^{\text{2P}}$"
    )
    validated_column = "slf_oil"
    compared_column = "res_oil"


@register_rule
class RE1038(RE1ForecastSumReserveRule):
    rule_id = "RE1038"
    description = (
        "Condensate Sales Forecast: Sum of Yearly Forecast must be equal to 2P Reserves"
    )
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{c, t}^{s} = "
        r"\Delta N_{ps}^{c \text{2P}}$"
    )
    validated_column = "slf_con"
    compared_column = "res_con"


@register_rule
class RE1039(RE1ForecastSumReserveRule):
    rule_id = "RE1039"
    description = "Associated Gas Sales Forecast: Sum of Yearly Forecast must be equal to 2P Reserves"  # noqa: E501  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{a, t}^{s} = "
        r"\Delta G_{ps}^{a \text{2P}}$"
    )
    validated_column = "slf_ga"
    compared_column = "res_ga"


@register_rule
class RE1040(RE1ForecastSumReserveRule):
    rule_id = "RE1040"
    description = "Non Associated Gas Sales Forecast: Sum of Yearly Forecast must be equal to 2P Reserves"  # noqa: E501  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{n, t}^{s} = "
        r"\Delta G_{ps}^{\text{2P}}$"
    )
    validated_column = "slf_gn"
    compared_column = "res_gn"


# ---------------------------------------------------------------------------
# Category F: Sum of TPF = 2R GRR/CR/PR (4 rules)
# ---------------------------------------------------------------------------


class RE1ForecastSumResourceRule(ValidationRule):
    """Sum of yearly Total Potential Forecast must equal 2R resources."""

    is_fixable = False
    applies_to_tables = ["project_resources", "project_timeseries"]
    severity = Severity.STRICT
    validated_column: str
    compared_column: str
    uncert: UncertLevel = UncertLevel.MID

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        sql = build_forecast_sum_equals_resource_sql(
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
            extra_columns=["val_sum", "val_ref"],
            validated_column=self.validated_column,
            compared_columns=[self.compared_column],
            rule_group="RE1",
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE1041(RE1ForecastSumResourceRule):
    rule_id = "RE1041"
    description = "Oil Total Potential Forecast: Sum of Yearly Forecast must be equal to 2R GRR/CR/PR"  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{o, t}^{\text{tp}} = "
        r"\Delta N_{pn}^{\text{P50}}$"
    )
    validated_column = "tpf_oil"
    compared_column = "rec_oil"


@register_rule
class RE1042(RE1ForecastSumResourceRule):
    rule_id = "RE1042"
    description = "Condensate Total Potential Forecast: Sum of Yearly Forecast must be equal to 2R GRR/CR/PR"  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{c, t}^{\text{tp}} = "
        r"\Delta N_{pn}^{c \text{P50}}$"
    )
    validated_column = "tpf_con"
    compared_column = "rec_con"


@register_rule
class RE1043(RE1ForecastSumResourceRule):
    rule_id = "RE1043"
    description = "Associated Gas Total Potential Forecast: Sum of Yearly Forecast must be equal to 2R GRR/CR/PR"  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{a, t}^{\text{tp}} = "
        r"\Delta G_{pn}^{a \text{P50}}$"
    )
    validated_column = "tpf_ga"
    compared_column = "rec_ga"


@register_rule
class RE1044(RE1ForecastSumResourceRule):
    rule_id = "RE1044"
    description = "Non Associated Gas Total Potential Forecast: Sum of Yearly Forecast must be equal to 2R GRR/CR/PR"  # noqa: E501
    formal = (
        r"$\sum_{t=t_R + 1}^{t_m} q_{n, t}^{\text{tp}} = "
        r"\Delta G_{pn}^{\text{P50}}$"
    )
    validated_column = "tpf_gn"
    compared_column = "rec_gn"
