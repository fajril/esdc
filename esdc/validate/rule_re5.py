"""RE5xxx rules: Maturity Level validation.

69 rules (RE5001-RE5069) covering:
- Production vs maturity level implications
- Maturity level state machine transitions
- Multi-year history transitions
- GCF constraints and monotonicity
- Sales/reserves vs level consistency
- Remarks requirements for discrepancies/changes
- Onstream actual constraints
- Complex reserve exhaustion checks
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import duckdb

from esdc.selection import Severity
from esdc.validate.rule_re5_helpers import (
    ABANDONED_LEVELS,
    E0_E1_E4_E7,
    E0_E7_E8_A1_A2,
    IDENTIFIER_COLS,
    REC_COLUMNS,
    ProjectLevel,
    _add_year_filter,
    _add_year_filter_self_join,
    build_discrepancy_implies_remarks_sql,
    build_gcf_abandoned_binary_sql,
    build_gcf_element_not_neutral_sql,
    build_gcf_monotonic_sql,
    build_gcf_must_be_filled_sql,
    build_gcf_neutral_implies_lead_sql,
    build_gcf_range_sql,
    build_gcf_total_implies_level_sql,
    build_gcf_transition_sql,
    build_groovy_transition_sql,
    build_ioip_igip_change_implies_remarks_sql,
    build_level_implies_1p_forbidden_sql,
    build_level_implies_1p_required_sql,
    build_level_implies_sales_positive_sql,
    build_level_mandatory_sql,
    build_multi_year_transition_sql,
    build_no_reserves_implies_level_sql,
    build_no_sales_implies_not_level_sql,
    build_onstream_before_report_year_sql,
    build_onstream_required_sql,
    build_reserves_implies_not_abandoned_sql,
    build_sales_implies_level_set_sql,
    build_sales_implies_level_sql,
    build_transition_sql,
    execute_and_build_violations,
)
from esdc.validate.rules import TOLERANCE, ValidationRule, Violation, register_rule

# ---------------------------------------------------------------------------
# Category A: Production → Maturity Level (RE5001–RE5002)
# ---------------------------------------------------------------------------


class RE5SalesImpliesE0Rule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_sales_implies_level_sql([ProjectLevel.E0])
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5001(RE5SalesImpliesE0Rule):
    rule_id = "RE5001"
    description = "If sales production > 0, project level must be E0. On Production"
    formal = r"$(q_{o,t_R} > 0) \lor (q_{c,t_R} > 0) \lor (q_{a,t_R} > 0) \lor (q_{n,t_R} > 0) \implies M_{t_R} = E_0$"


class RE5NoSalesNotE0Rule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_no_sales_implies_not_level_sql([ProjectLevel.E0])
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5002(RE5NoSalesNotE0Rule):
    rule_id = "RE5002"
    description = "If no sales production, project level cannot be E0. On Production"
    formal = r"$(q_{o,t_R} = q_{c,t_R} = q_{a,t_R} = q_{n,t_R} = 0) \implies M_{t_R} \neq E_0$"


# ---------------------------------------------------------------------------
# Category B: Groovy + Production Transition Rules (RE5003, RE5005, RE5006)
# ---------------------------------------------------------------------------


class RE5GroovyProductionTransitionRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    previous_level: str
    groovy_value: bool
    required_level: str
    require_no_production: bool = False

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_groovy_transition_sql(
            self.previous_level,
            self.groovy_value,
            self.required_level,
            require_no_production=self.require_no_production,
        )
        if self.require_no_production:
            sql = _add_year_filter_self_join(sql, year)
        else:
            sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=["groovy_isactive"],
            extra_columns=["project_level", "groovy_isactive"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5003(RE5GroovyProductionTransitionRule):
    rule_id = "RE5003"
    description = (
        "If previous level E1 and no production and no GROOVY, level must be E4"
    )
    formal = (
        r"$(q = 0) \land (M_{t_R-1} = E_1) \land (G_r = \bot) \implies M_{t_R} = E_4$"
    )
    previous_level = ProjectLevel.E1
    groovy_value = False
    required_level = ProjectLevel.E4
    require_no_production = True


@register_rule
class RE5006(RE5GroovyProductionTransitionRule):
    rule_id = "RE5006"
    description = (
        "If previous level E4 and no production and has GROOVY, level must be E4"
    )
    formal = (
        r"$(q = 0) \land (M_{t_R-1} = E_4) \land (G_r = \top) \implies M_{t_R} = E_4$"
    )
    previous_level = ProjectLevel.E4
    groovy_value = True
    required_level = ProjectLevel.E4
    require_no_production = True


# ---------------------------------------------------------------------------
# Category C: Multi-Year Transition Rules (RE5004, RE5007, RE5008, RE5018)
# ---------------------------------------------------------------------------


class RE5MultiYearTransitionRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    previous_levels: list[str]
    n_years: int
    required_level: str
    severity: Severity = Severity.STRICT
    groovy_condition: bool | None = None
    no_production: bool = True

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_multi_year_transition_sql(
            self.previous_levels,
            self.n_years,
            self.required_level,
            groovy_condition=self.groovy_condition,
            no_production=self.no_production,
        )
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5004(RE5MultiYearTransitionRule):
    rule_id = "RE5004"
    description = "If E1 for last 3 years and no production, level must be E4"
    formal = r"$(q = 0) \land (M_{t_R-1} = M_{t_R-2} = M_{t_R-3} = E_1) \implies M_{t_R} = E_4$"
    previous_levels = [ProjectLevel.E1] * 3
    n_years = 3
    required_level = ProjectLevel.E4


@register_rule
class RE5007(RE5MultiYearTransitionRule):
    rule_id = "RE5007"
    description = (
        "If E2 for last 3 years and no production and no GROOVY, level should be E5"
    )
    formal = r"$(q = 0) \land (M_{t_R-1}=M_{t_R-2}=M_{t_R-3} = E_2) \land (G_r = \bot) \implies M_{t_R} = E_5$"
    previous_levels = [ProjectLevel.E2] * 3
    n_years = 3
    required_level = ProjectLevel.E5
    severity = Severity.WARNING
    groovy_condition = False


@register_rule
class RE5008(RE5MultiYearTransitionRule):
    rule_id = "RE5008"
    description = (
        "If E3 for last 3 years and no production and no GROOVY, level should be E5"
    )
    formal = r"$(q = 0) \land (M_{t_R-1}=M_{t_R-2}=M_{t_R-3} = E_3) \land (G_r = \bot) \implies M_{t_R} = E_5$"
    previous_levels = [ProjectLevel.E3] * 3
    n_years = 3
    required_level = ProjectLevel.E5
    severity = Severity.WARNING
    groovy_condition = False


@register_rule
class RE5005(RE5MultiYearTransitionRule):
    rule_id = "RE5005"
    description = (
        "If E4 for last 3 years and no production and no GROOVY, level must be E7"
    )
    formal = r"$(q = 0) \land (M_{t_R-1}=M_{t_R-2}=M_{t_R-3}=E_4) \land (G_r = \bot) \implies M_{t_R} = E_7$"
    previous_levels = [ProjectLevel.E4] * 3
    n_years = 3
    required_level = ProjectLevel.E7
    groovy_condition = False


@register_rule
class RE5018(ValidationRule):
    rule_id = "RE5018"
    description = "If X1 for last 2 years, level cannot be X1"
    formal = r"$M_{t_R-1} = M_{t_R-2} = X_1 \implies M_{t_R} \neq X_1$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
        sql = (
            f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident}, curr.project_level AS val_ref"
            f" FROM project_resources curr"
            f" JOIN project_resources prev1 ON prev1.project_id = curr.project_id AND prev1.report_year = curr.report_year - 1 AND prev1.uncert_level = curr.uncert_level"
            f" JOIN project_resources prev2 ON prev2.project_id = curr.project_id AND prev2.report_year = curr.report_year - 2 AND prev2.uncert_level = curr.uncert_level"
            f" WHERE prev1.project_level = '{ProjectLevel.X1.value}'"
            f" AND prev2.project_level = '{ProjectLevel.X1.value}'"
            f" AND curr.project_level = '{ProjectLevel.X1.value}'"
        )
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# ---------------------------------------------------------------------------
# Category D: Maturity Level State Machine — Single Transition (RE5009–RE5023)
# ---------------------------------------------------------------------------


class RE5TransitionRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    previous_level: str
    allowed_levels: list[str]
    severity: Severity = Severity.WARNING

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_transition_sql(self.previous_level, self.allowed_levels)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=["project_level_previous"],
            extra_columns=["project_level", "project_level_previous"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5009(RE5TransitionRule):
    rule_id = "RE5009"
    description = "If previous E2, current must be E0, E2, or E5"
    formal = r"$M_{t_R-1} = E_2 \implies M_{t_R} \in \{E_0, E_2, E_5\}$"
    previous_level = ProjectLevel.E2
    allowed_levels = [ProjectLevel.E0, ProjectLevel.E2, ProjectLevel.E5]


@register_rule
class RE5010(RE5TransitionRule):
    rule_id = "RE5010"
    description = "If previous E3, current must be E0, E2, E3, or E5"
    formal = r"$M_{t_R-1} = E_3 \implies M_{t_R} \in \{E_0, E_2, E_3, E_5\}$"
    previous_level = ProjectLevel.E3
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.E5,
    ]


@register_rule
class RE5011(RE5TransitionRule):
    rule_id = "RE5011"
    description = "If previous E5, current must be E0, E2, E3, or E5"
    formal = r"$M_{t_R-1} = E_5 \implies M_{t_R} \in \{E_0, E_2, E_3, E_5\}$"
    previous_level = ProjectLevel.E5
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.E5,
    ]


@register_rule
class RE5012(RE5TransitionRule):
    rule_id = "RE5012"
    description = "If previous E6, current must be E0, E2, E3, or E8"
    formal = r"$M_{t_R-1} = E_6 \implies M_{t_R} \in \{E_0, E_2, E_3, E_8\}$"
    previous_level = ProjectLevel.E6
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.E8,
    ]
    severity = Severity.STRICT


@register_rule
class RE5013(ValidationRule):
    rule_id = "RE5013"
    description = "If previous E7 and has GROOVY, level must be E4"
    formal = r"$(M_{t_R-1} = E_7) \land (G_r = \top) \implies M_{t_R} = E_4$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_groovy_transition_sql(ProjectLevel.E7, True, ProjectLevel.E4)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=["groovy_isactive"],
            extra_columns=["project_level", "groovy_isactive"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5014(RE5TransitionRule):
    rule_id = "RE5014"
    description = "If previous E7, current must be E0, E4, or E7"
    formal = r"$M_{t_R-1} = E_7 \implies M_{t_R} \in \{E_0, E_4, E_7\}$"
    previous_level = ProjectLevel.E7
    allowed_levels = [ProjectLevel.E0, ProjectLevel.E4, ProjectLevel.E7]


@register_rule
class RE5015(RE5TransitionRule):
    rule_id = "RE5015"
    description = "If previous E8, current must be E0, E2, E3, or E8"
    formal = r"$M_{t_R-1} = E_8 \implies M_{t_R} \in \{E_0, E_2, E_3, E_8\}$"
    previous_level = ProjectLevel.E8
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.E8,
    ]


@register_rule
class RE5016(RE5TransitionRule):
    rule_id = "RE5016"
    description = "If previous X0, current must be E0, E2, E3, X0, X2, or X3"
    formal = r"$M_{t_R-1} = X_0 \implies M_{t_R} \in \{E_0, E_2, E_3, X_0, X_2, X_3\}$"
    previous_level = ProjectLevel.X0
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.X0,
        ProjectLevel.X2,
        ProjectLevel.X3,
    ]


@register_rule
class RE5017(RE5TransitionRule):
    rule_id = "RE5017"
    description = "If previous X1, current must be E0, E2, E3, X0, X1, or X2"
    formal = r"$M_{t_R-1} = X_1 \implies M_{t_R} \in \{E_0, E_2, E_3, X_0, X_1, X_2\}$"
    previous_level = ProjectLevel.X1
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.X0,
        ProjectLevel.X1,
        ProjectLevel.X2,
    ]


@register_rule
class RE5019(RE5TransitionRule):
    rule_id = "RE5019"
    description = "If previous X2, current must be E0, E2, E3, X0, or X2"
    formal = r"$M_{t_R-1} = X_2 \implies M_{t_R} \in \{E_0, E_2, E_3, X_0, X_2\}$"
    previous_level = ProjectLevel.X2
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.X0,
        ProjectLevel.X2,
    ]


@register_rule
class RE5020(RE5TransitionRule):
    rule_id = "RE5020"
    description = "If previous X3, current must be E0, E2, E3, or X3"
    formal = r"$M_{t_R-1} = X_3 \implies M_{t_R} \in \{E_0, E_2, E_3, X_3\}$"
    previous_level = ProjectLevel.X3
    allowed_levels = [
        ProjectLevel.E0,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.X3,
    ]


@register_rule
class RE5021(RE5TransitionRule):
    rule_id = "RE5021"
    description = "If previous X4, current must be E3, X0, X1, or X4"
    formal = r"$M_{t_R-1} = X_4 \implies M_{t_R} \in \{E_3, X_0, X_1, X_4\}$"
    previous_level = ProjectLevel.X4
    allowed_levels = [
        ProjectLevel.E3,
        ProjectLevel.X0,
        ProjectLevel.X1,
        ProjectLevel.X4,
    ]


@register_rule
class RE5022(RE5TransitionRule):
    rule_id = "RE5022"
    description = "If previous X5, current must be X0, X1, X4, or X5"
    formal = r"$M_{t_R-1} = X_5 \implies M_{t_R} \in \{X_0, X_1, X_4, X_5\}$"
    previous_level = ProjectLevel.X5
    allowed_levels = [
        ProjectLevel.X0,
        ProjectLevel.X1,
        ProjectLevel.X4,
        ProjectLevel.X5,
    ]


@register_rule
class RE5023(RE5TransitionRule):
    rule_id = "RE5023"
    description = "If previous X6, current must be X1, X4, X5, or X6"
    formal = r"$M_{t_R-1} = X_6 \implies M_{t_R} \in \{X_1, X_4, X_5, X_6\}$"
    previous_level = ProjectLevel.X6
    allowed_levels = [
        ProjectLevel.X1,
        ProjectLevel.X4,
        ProjectLevel.X5,
        ProjectLevel.X6,
    ]


# ---------------------------------------------------------------------------
# Category E: GCF → Maturity Level (RE5024–RE5026)
# ---------------------------------------------------------------------------


class RE5GcfTotalImpliesLevelRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    condition: str
    required_levels: list[str]

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_total_implies_level_sql(self.condition, self.required_levels)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=["gcf_total"],
            extra_columns=["project_level", "gcf_total"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5024(RE5GcfTotalImpliesLevelRule):
    rule_id = "RE5024"
    description = "If 0 < GCF total < 1, level must be X5 or X6"
    formal = r"$0 < P_g < 1 \implies M_{t_R} \in \{X_5, X_6\}$"
    condition = "gcf_total > 0 AND gcf_total < 1"
    required_levels = [ProjectLevel.X5, ProjectLevel.X6]


@register_rule
class RE5025(RE5GcfTotalImpliesLevelRule):
    rule_id = "RE5025"
    description = "If GCF total = 1, level must be X4 or greater"
    formal = r"$P_g = 1 \implies M_{t_R} \in M_E \cup \{X_0, \dots, X_4\}$"
    condition = "gcf_total = 1"
    required_levels = [
        ProjectLevel.E0,
        ProjectLevel.E1,
        ProjectLevel.E2,
        ProjectLevel.E3,
        ProjectLevel.E4,
        ProjectLevel.E5,
        ProjectLevel.E6,
        ProjectLevel.E7,
        ProjectLevel.E8,
        ProjectLevel.X0,
        ProjectLevel.X1,
        ProjectLevel.X2,
        ProjectLevel.X3,
        ProjectLevel.X4,
    ]


@register_rule
class RE5026(RE5GcfTotalImpliesLevelRule):
    rule_id = "RE5026"
    description = "If GCF total = 0, level must be A1 or A2"
    formal = r"$P_g = 0 \implies M_{t_R} \in \{A_1, A_2\}$"
    condition = "gcf_total = 0"
    required_levels = ABANDONED_LEVELS


# ---------------------------------------------------------------------------
# Category F: GCF Element Not Neutral (RE5027–RE5030)
# ---------------------------------------------------------------------------


class RE5GcfElementNotNeutralRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.WARNING
    gcf_column: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_element_not_neutral_sql(self.gcf_column)
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column=self.gcf_column,
            compared_columns=[self.gcf_column],
            extra_columns=[self.gcf_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5027(RE5GcfElementNotNeutralRule):
    rule_id = "RE5027"
    description = "If previous GCF Source Rock != 0.5, current must not be 0.5"
    formal = r"$P_{g,s,t_R-1} \neq 0.5 \implies P_{g,s,t_R} \neq 0.5$"
    gcf_column = "gcf_srock"


@register_rule
class RE5028(RE5GcfElementNotNeutralRule):
    rule_id = "RE5028"
    description = "If previous GCF Reservoir != 0.5, current must not be 0.5"
    formal = r"$P_{g,r,t_R-1} \neq 0.5 \implies P_{g,r,t_R} \neq 0.5$"
    gcf_column = "gcf_res"


@register_rule
class RE5029(RE5GcfElementNotNeutralRule):
    rule_id = "RE5029"
    description = "If previous GCF Trap and Seal != 0.5, current must not be 0.5"
    formal = r"$P_{g,ts,t_R-1} \neq 0.5 \implies P_{g,ts,t_R} \neq 0.5$"
    gcf_column = "gcf_ts"


@register_rule
class RE5030(RE5GcfElementNotNeutralRule):
    rule_id = "RE5030"
    description = "If previous GCF Dynamic != 0.5, current must not be 0.5"
    formal = r"$P_{g,m,t_R-1} \neq 0.5 \implies P_{g,m,t_R} \neq 0.5$"
    gcf_column = "gcf_dyn"


# ---------------------------------------------------------------------------
# Category G: GCF Combination Rules (RE5031–RE5040)
# ---------------------------------------------------------------------------


@register_rule
class RE5031(ValidationRule):
    rule_id = "RE5031"
    description = "If previous X6 and current X5, GCF total must be >= previous"
    formal = (
        r"$(M_{t_R-1} = X_6) \land (M_{t_R} = X_5) \implies P_{g,t_R-1} \leq P_{g,t_R}$"
    )
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.WARNING

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_transition_sql()
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="gcf_total",
            compared_columns=["gcf_total"],
            extra_columns=["gcf_total"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5032(ValidationRule):
    rule_id = "RE5032"
    description = "If any GCF element = 0.5, level must be X6"
    formal = r"$(P_{g,s} = 0.5) \lor (P_{g,r} = 0.5) \lor (P_{g,ts} = 0.5) \lor (P_{g,m} = 0.5) \implies M_{t_R} = X_6$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.WARNING

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_neutral_implies_lead_sql()
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE5GcfAbandonedBinaryRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.WARNING
    gcf_column: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_abandoned_binary_sql(self.gcf_column)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column=self.gcf_column,
            compared_columns=[],
            extra_columns=[self.gcf_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5033(RE5GcfAbandonedBinaryRule):
    rule_id = "RE5033"
    description = "If project is abandoned, GCF Source Rock must be 0 or 1"
    formal = r"$M_{t_R} \in \{A_1, A_2\} \implies P_{g,s} \in \{0, 1\}$"
    gcf_column = "gcf_srock"


@register_rule
class RE5034(RE5GcfAbandonedBinaryRule):
    rule_id = "RE5034"
    description = "If project is abandoned, GCF Reservoir must be 0 or 1"
    formal = r"$M_{t_R} \in \{A_1, A_2\} \implies P_{g,r} \in \{0, 1\}$"
    gcf_column = "gcf_res"


@register_rule
class RE5035(RE5GcfAbandonedBinaryRule):
    rule_id = "RE5035"
    description = "If project is abandoned, GCF Trap and Seal must be 0 or 1"
    formal = r"$M_{t_R} \in \{A_1, A_2\} \implies P_{g,ts} \in \{0, 1\}$"
    gcf_column = "gcf_ts"


@register_rule
class RE5036(RE5GcfAbandonedBinaryRule):
    rule_id = "RE5036"
    description = "If project is abandoned, GCF Dynamic must be 0 or 1"
    formal = r"$M_{t_R} \in \{A_1, A_2\} \implies P_{g,m} \in \{0, 1\}$"
    gcf_column = "gcf_dyn"


class RE5GcfMonotonicRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.WARNING
    gcf_column: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_monotonic_sql(self.gcf_column)
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column=self.gcf_column,
            compared_columns=[self.gcf_column],
            extra_columns=[self.gcf_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5037(RE5GcfMonotonicRule):
    rule_id = "RE5037"
    description = "If previous GCF Source Rock > 0.5 and not abandoned, current must be >= previous"
    formal = r"$(P_{g,s,t_R-1} > 0.5) \land M_{t_R} \notin \{A_1,A_2\} \implies P_{g,s,t_R} \geq P_{g,s,t_R-1}$"
    gcf_column = "gcf_srock"


@register_rule
class RE5038(RE5GcfMonotonicRule):
    rule_id = "RE5038"
    description = (
        "If previous GCF Reservoir > 0.5 and not abandoned, current must be >= previous"
    )
    formal = r"$(P_{g,r,t_R-1} > 0.5) \land M_{t_R} \notin \{A_1,A_2\} \implies P_{g,r,t_R} \geq P_{g,r,t_R-1}$"
    gcf_column = "gcf_res"


@register_rule
class RE5039(RE5GcfMonotonicRule):
    rule_id = "RE5039"
    description = "If previous GCF Trap and Seal > 0.5 and not abandoned, current must be >= previous"
    formal = r"$(P_{g,ts,t_R-1} > 0.5) \land M_{t_R} \notin \{A_1,A_2\} \implies P_{g,ts,t_R} \geq P_{g,ts,t_R-1}$"
    gcf_column = "gcf_ts"


@register_rule
class RE5040(RE5GcfMonotonicRule):
    rule_id = "RE5040"
    description = (
        "If previous GCF Dynamic > 0.5 and not abandoned, current must be >= previous"
    )
    formal = r"$(P_{g,m,t_R-1} > 0.5) \land M_{t_R} \notin \{A_1,A_2\} \implies P_{g,m,t_R} \geq P_{g,m,t_R-1}$"
    gcf_column = "gcf_dyn"


# ---------------------------------------------------------------------------
# Category H: GCF Validity (RE5041–RE5051)
# ---------------------------------------------------------------------------


@register_rule
class RE5041(ValidationRule):
    rule_id = "RE5041"
    description = "If sales cumulative production > 0, level must be E0, E1, E4, or E7"
    formal = r"$(N_{ps} > 0) \lor (N_{ps}^c > 0) \lor (G_{ps}^a > 0) \lor (G_{ps} > 0) \implies M_{t_R} \in \{E_0, E_1, E_4, E_7\}$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_sales_implies_level_set_sql(E0_E1_E4_E7)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5042(ValidationRule):
    rule_id = "RE5042"
    description = "Project level must be filled"
    formal = r"$M_{t_R} \notin \emptyset$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_level_mandatory_sql()
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


class RE5ReservesImpliesNotAbandonedRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    uncert_level: str
    columns: list[str]

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_reserves_implies_not_abandoned_sql(self.uncert_level, self.columns)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5043(RE5ReservesImpliesNotAbandonedRule):
    rule_id = "RE5043"
    description = "If project has hydrocarbon volume, level cannot be A1 or A2"
    formal = r"$\Delta N_{pn}^{P10} + \Delta N_{pn}^{cP10} + \Delta G_{pn}^{aP10} + \Delta G_{pn}^{P10} > 0 \implies M_{t_R} \notin \{A_1, A_2\}$"
    uncert_level = "3. High Value"
    columns = REC_COLUMNS


@register_rule
class RE5065(RE5ReservesImpliesNotAbandonedRule):
    rule_id = "RE5065"
    description = "If project has hydrocarbon in-place volume, level cannot be A1 or A2"
    formal = r"$N_{prj}^{P10} + G_{prj}^{P10} > 0 \implies M_{t_R} \notin \{A_1, A_2\}$"
    uncert_level = "3. High Value"
    columns = ["prj_ioip", "prj_igip"]


class RE5GcfMustBeFilledRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    gcf_column: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_must_be_filled_sql(self.gcf_column)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column=self.gcf_column,
            compared_columns=[],
            extra_columns=[self.gcf_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5044(RE5GcfMustBeFilledRule):
    rule_id = "RE5044"
    description = "GCF Source Rock must be filled"
    formal = r"$P_{g,s,t_R} \notin \emptyset$"
    gcf_column = "gcf_srock"


@register_rule
class RE5045(RE5GcfMustBeFilledRule):
    rule_id = "RE5045"
    description = "GCF Reservoir must be filled"
    formal = r"$P_{g,r,t_R} \notin \emptyset$"
    gcf_column = "gcf_res"


@register_rule
class RE5046(RE5GcfMustBeFilledRule):
    rule_id = "RE5046"
    description = "GCF Trap and Seal must be filled"
    formal = r"$P_{g,ts,t_R} \notin \emptyset$"
    gcf_column = "gcf_ts"


@register_rule
class RE5047(RE5GcfMustBeFilledRule):
    rule_id = "RE5047"
    description = "GCF Dynamic must be filled"
    formal = r"$P_{g,m,t_R} \notin \emptyset$"
    gcf_column = "gcf_dyn"


class RE5GcfRange01Rule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    gcf_column: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_gcf_range_sql(self.gcf_column)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column=self.gcf_column,
            compared_columns=[],
            extra_columns=[self.gcf_column],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5048(RE5GcfRange01Rule):
    rule_id = "RE5048"
    description = "GCF Source Rock must be within range 0 to 1"
    formal = r"$0 \leq P_{g,s,t_R} \leq 1$"
    gcf_column = "gcf_srock"


@register_rule
class RE5049(RE5GcfRange01Rule):
    rule_id = "RE5049"
    description = "GCF Reservoir must be within range 0 to 1"
    formal = r"$0 \leq P_{g,r,t_R} \leq 1$"
    gcf_column = "gcf_res"


@register_rule
class RE5050(RE5GcfRange01Rule):
    rule_id = "RE5050"
    description = "GCF Trap and Seal must be within range 0 to 1"
    formal = r"$0 \leq P_{g,ts,t_R} \leq 1$"
    gcf_column = "gcf_ts"


@register_rule
class RE5051(RE5GcfRange01Rule):
    rule_id = "RE5051"
    description = "GCF Dynamic must be within range 0 to 1"
    formal = r"$0 \leq P_{g,m,t_R} \leq 1$"
    gcf_column = "gcf_dyn"


# ---------------------------------------------------------------------------
# Category I: Level ↔ Sales/Reserves (RE5052–RE5055)
# ---------------------------------------------------------------------------


@register_rule
class RE5052(ValidationRule):
    rule_id = "RE5052"
    description = "If level is E0/E1/E4/E7, sales cumulative production must be > 0"
    formal = r"$M_{t_R} \in \{E_0, E_1, E_4, E_7\} \implies N_{ps} > 0$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_level_implies_sales_positive_sql(E0_E1_E4_E7)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5053(ValidationRule):
    rule_id = "RE5053"
    description = "If no hydrocarbon volume, level must be E0/E7/E8/A1/A2"
    formal = r"$\Delta N_{pn}^{P10} = \Delta N_{pn}^{cP10} = \Delta G_{pn}^{aP10} = \Delta G_{pn}^{P10} = 0 \implies M_{t_R} \in \{E_0, E_7, E_8, A_1, A_2\}$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_no_reserves_implies_level_sql(E0_E7_E8_A1_A2)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5054(ValidationRule):
    rule_id = "RE5054"
    description = "Level E1/E2/E3 must have 1P reserves > 0"
    formal = r"$M_{t_R} \in \{E_1, E_2, E_3\} \implies \Delta N_{ps}^{1P} > 0$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_level_implies_1p_required_sql(
            [ProjectLevel.E1, ProjectLevel.E2, ProjectLevel.E3]
        )
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5055(ValidationRule):
    rule_id = "RE5055"
    description = "Levels E4 through X6 must not have 1P reserves"
    formal = r"$M_{t_R} \in \{E_4, \dots, X_6\} \implies \Delta N_{ps}^{1P} = 0$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        forbidden_levels = [
            ProjectLevel.E4,
            ProjectLevel.E5,
            ProjectLevel.E6,
            ProjectLevel.E7,
            ProjectLevel.E8,
            ProjectLevel.X0,
            ProjectLevel.X1,
            ProjectLevel.X2,
            ProjectLevel.X3,
            ProjectLevel.X4,
            ProjectLevel.X5,
            ProjectLevel.X6,
        ]
        sql = build_level_implies_1p_forbidden_sql(forbidden_levels)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# ---------------------------------------------------------------------------
# Category J: Remarks Requirements (RE5056–RE5064)
# ---------------------------------------------------------------------------


class RE5DiscrepancyImpliesRemarksRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    uncert_level: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_discrepancy_implies_remarks_sql(self.uncert_level)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_remarks",
            compared_columns=[],
            extra_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5056(RE5DiscrepancyImpliesRemarksRule):
    rule_id = "RE5056"
    description = "If discrepancy in resources low, project must have remarks"
    formal = r"$\sum \Delta D^{P90} > 0 \implies M_{remarks} \notin \emptyset$"
    uncert_level = "1. Low Value"


@register_rule
class RE5057(RE5DiscrepancyImpliesRemarksRule):
    rule_id = "RE5057"
    description = "If discrepancy in resources mid, project must have remarks"
    formal = r"$\sum \Delta D^{P50} > 0 \implies M_{remarks} \notin \emptyset$"
    uncert_level = "2. Middle Value"


@register_rule
class RE5058(RE5DiscrepancyImpliesRemarksRule):
    rule_id = "RE5058"
    description = "If discrepancy in resources high, project must have remarks"
    formal = r"$\sum \Delta D^{P10} > 0 \implies M_{remarks} \notin \emptyset$"
    uncert_level = "3. High Value"


class RE5IOIPChangeImpliesRemarksRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    uncert_level: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_ioip_igip_change_implies_remarks_sql("prj_ioip", self.uncert_level)
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_remarks",
            compared_columns=[],
            extra_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5059(RE5IOIPChangeImpliesRemarksRule):
    rule_id = "RE5059"
    description = "If project IOIP low changed, project must have remarks"
    formal = r"$N_{prj,t_R}^{P90} \neq N_{prj,t_{R-1}}^{P90} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "1. Low Value"


@register_rule
class RE5060(RE5IOIPChangeImpliesRemarksRule):
    rule_id = "RE5060"
    description = "If project IOIP mid changed, project must have remarks"
    formal = r"$N_{prj,t_R}^{P50} \neq N_{prj,t_{R-1}}^{P50} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "2. Middle Value"


@register_rule
class RE5061(RE5IOIPChangeImpliesRemarksRule):
    rule_id = "RE5061"
    description = "If project IOIP high changed, project must have remarks"
    formal = r"$N_{prj,t_R}^{P10} \neq N_{prj,t_{R-1}}^{P10} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "3. High Value"


class RE5IGIPChangeImpliesRemarksRule(ValidationRule):
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT
    uncert_level: str

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_ioip_igip_change_implies_remarks_sql("prj_igip", self.uncert_level)
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_remarks",
            compared_columns=[],
            extra_columns=[],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5062(RE5IGIPChangeImpliesRemarksRule):
    rule_id = "RE5062"
    description = "If project IGIP low changed, project must have remarks"
    formal = r"$G_{prj,t_R}^{P90} \neq G_{prj,t_{R-1}}^{P90} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "1. Low Value"


@register_rule
class RE5063(RE5IGIPChangeImpliesRemarksRule):
    rule_id = "RE5063"
    description = "If project IGIP mid changed, project must have remarks"
    formal = r"$G_{prj,t_R}^{P50} \neq G_{prj,t_{R-1}}^{P50} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "2. Middle Value"


@register_rule
class RE5064(RE5IGIPChangeImpliesRemarksRule):
    rule_id = "RE5064"
    description = "If project IGIP high changed, project must have remarks"
    formal = r"$G_{prj,t_R}^{P10} \neq G_{prj,t_{R-1}}^{P10} \implies M_{remarks} \notin \emptyset$"
    uncert_level = "3. High Value"


# ---------------------------------------------------------------------------
# Category K: Onstream Actual (RE5066–RE5067)
# ---------------------------------------------------------------------------


@register_rule
class RE5066(ValidationRule):
    rule_id = "RE5066"
    description = "Levels E0/E1/E4/E7 require onstream actual"
    formal = r"$M_{t_R} \in \{E_0, E_1, E_4, E_7\} \implies t_{ons} \notin \emptyset$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_onstream_required_sql(E0_E1_E4_E7)
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="onstream_actual",
            compared_columns=[],
            extra_columns=["project_level", "onstream_actual"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5067(ValidationRule):
    rule_id = "RE5067"
    description = "Onstream actual must be less than reporting year"
    formal = r"$t_{ons} < t_R$"
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        sql = build_onstream_before_report_year_sql()
        sql = _add_year_filter(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="onstream_actual",
            compared_columns=[],
            extra_columns=["report_year", "onstream_actual"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


# ---------------------------------------------------------------------------
# Category L: Complex Reserve Rules (RE5068–RE5069)
# ---------------------------------------------------------------------------


@register_rule
class RE5068(ValidationRule):
    rule_id = "RE5068"
    description = (
        "At E0 with 1P reserves, production + dcpy_gtr must not equal previous 1P"
    )
    formal = (
        r"$(\Delta N_{ps}^{1P} > 0) \land (M_{t_R} = E_0) \implies "
        r"(q_o + \Delta D_N^{gtr P90} \neq \Delta N_{ps,t_R-1}^{1P}) "
        r"\land (q_c + \Delta D_{N^c}^{gtr P90} \neq \Delta N_{ps,t_R-1}^{c 1P}) "
        r"\land (q_n + \Delta D_G^{gtr P90} \neq \Delta G_{ps,t_R-1}^{1P}) "
        r"\land (q_a + \Delta D_{G^a}^{gtr P90} \neq \Delta G_{ps,t_R-1}^{a 1P})$"
    )
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
        t = TOLERANCE
        sql = (
            f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
            f" curr.project_level AS val_ref"
            f" FROM project_resources curr"
            f" JOIN project_resources prev"
            f" ON curr.project_id = prev.project_id"
            f" AND curr.report_year = prev.report_year + 1"
            f" AND prev.uncert_level = '1. Low Value'"
            f" WHERE curr.uncert_level = '1. Low Value'"
            f" AND curr.project_level = '{ProjectLevel.E0.value}'"
            f" AND ("
            f"  COALESCE(curr.res_oil, 0) > {t}"
            f"  OR COALESCE(curr.res_con, 0) > {t}"
            f"  OR COALESCE(curr.res_ga, 0) > {t}"
            f"  OR COALESCE(curr.res_gn, 0) > {t}"
            f" )"
            f" AND ("
            f"  (COALESCE(prev.res_oil, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_oil, 0) - COALESCE(prev.cprd_sls_oil, 0))"
            f"    + COALESCE(curr.dcpy_gtr_oil, 0)"
            f"    - COALESCE(prev.res_oil, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_con, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_con, 0) - COALESCE(prev.cprd_sls_con, 0))"
            f"    + COALESCE(curr.dcpy_gtr_con, 0)"
            f"    - COALESCE(prev.res_con, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_ga, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_ga, 0) - COALESCE(prev.cprd_sls_ga, 0))"
            f"    + COALESCE(curr.dcpy_gtr_ga, 0)"
            f"    - COALESCE(prev.res_ga, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_gn, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_gn, 0) - COALESCE(prev.cprd_sls_gn, 0))"
            f"    + COALESCE(curr.dcpy_gtr_gn, 0)"
            f"    - COALESCE(prev.res_gn, 0)"
            f"   ) <= {t})"
            f" )"
        )
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []


@register_rule
class RE5069(ValidationRule):
    rule_id = "RE5069"
    description = "If 2P = 0 and production exhausted previous 2P, level must be E0"
    formal = (
        r"$(\Delta N_{ps}^{2P} = 0) \land "
        r"((q_o + \Delta D_N^{gtr P50} = \Delta N_{ps,t_R-1}^{2P}) \lor "
        r"(q_c + \Delta D_{N^c}^{gtr P50} = \Delta N_{ps,t_R-1}^{c 2P}) \lor "
        r"(q_n + \Delta D_G^{gtr P50} = \Delta G_{ps,t_R-1}^{2P}) \lor "
        r"(q_a + \Delta D_{G^a}^{gtr P50} = \Delta G_{ps,t_R-1}^{a 2P})) "
        r"\implies M_{t_R} = E_0$"
    )
    is_fixable = False
    applies_to_tables = ["project_resources"]
    severity = Severity.STRICT

    def check(
        self, conn: duckdb.DuckDBPyConnection, year: list[int] | None = None
    ) -> list[Violation]:
        ident = ", ".join(f"curr.{c}" for c in IDENTIFIER_COLS)
        t = TOLERANCE
        sql = (
            f"SELECT DISTINCT ON (curr.project_id, curr.report_year) {ident},"
            f" curr.project_level AS val_ref"
            f" FROM project_resources curr"
            f" JOIN project_resources prev"
            f" ON curr.project_id = prev.project_id"
            f" AND curr.report_year = prev.report_year + 1"
            f" AND prev.uncert_level = '2. Middle Value'"
            f" WHERE curr.uncert_level = '2. Middle Value'"
            f" AND curr.project_level != '{ProjectLevel.E0.value}'"
            f" AND ("
            f"  ABS(COALESCE(curr.res_oil, 0)) <= {t}"
            f"  OR ABS(COALESCE(curr.res_con, 0)) <= {t}"
            f"  OR ABS(COALESCE(curr.res_ga, 0)) <= {t}"
            f"  OR ABS(COALESCE(curr.res_gn, 0)) <= {t}"
            f" )"
            f" AND ("
            f"  (COALESCE(prev.res_oil, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_oil, 0) - COALESCE(prev.cprd_sls_oil, 0))"
            f"    + COALESCE(curr.dcpy_gtr_oil, 0)"
            f"    - COALESCE(prev.res_oil, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_con, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_con, 0) - COALESCE(prev.cprd_sls_con, 0))"
            f"    + COALESCE(curr.dcpy_gtr_con, 0)"
            f"    - COALESCE(prev.res_con, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_ga, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_ga, 0) - COALESCE(prev.cprd_sls_ga, 0))"
            f"    + COALESCE(curr.dcpy_gtr_ga, 0)"
            f"    - COALESCE(prev.res_ga, 0)"
            f"   ) <= {t})"
            f"  OR (COALESCE(prev.res_gn, 0) > {t}"
            f"   AND ABS("
            f"    (COALESCE(curr.cprd_sls_gn, 0) - COALESCE(prev.cprd_sls_gn, 0))"
            f"    + COALESCE(curr.dcpy_gtr_gn, 0)"
            f"    - COALESCE(prev.res_gn, 0)"
            f"   ) <= {t})"
            f" )"
        )
        sql = _add_year_filter_self_join(sql, year)
        return execute_and_build_violations(
            conn,
            sql,
            rule_id=self.rule_id,
            description=self.description,
            severity=self.severity,
            table="project_resources",
            validated_column="project_level",
            compared_columns=[],
            extra_columns=["project_level"],
        )

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        return []
