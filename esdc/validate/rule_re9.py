"""RE9xxx rules: Metadata / active-inactive state consistency."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import duckdb

from esdc.selection import Severity
from esdc.validate.rules import ValidationRule, Violation, register_rule


@register_rule
class RE9001(ValidationRule):
    """Inactive project must have zero volumetric values.

    If project_isactive = 0 then all reserve, resource, and in-place
    volume columns must also be zero.  A non-zero value on an inactive
    project indicates stale or inconsistent data.
    """

    rule_id = "RE9001"
    description = "Inactive project must have zero volumetric values"
    formal = (
        r"$\text{project\_isactive} = 0 \implies "
        r"N_{\text{project}} = 0 \;\land\; "
        r"G_{\text{project}} = 0 \;\land\; "
        r"\Delta N_{pn}^{f} = 0 \;\land\; "
        r"\Delta G_{pn}^{f} = 0 \;\land\; "
        r"\Delta N_{ps}^{f} = 0 \;\land\; "
        r"\Delta G_{ps}^{f} = 0 "
        r"\quad \forall f \in \{\text{oil, con, ga, gn, oc, an}\}$"
    )
    severity = Severity.STRICT
    is_fixable = True
    applies_to_tables = ["project_resources"]

    ZERO_COLUMNS: list[str] = [
        "rec_oil",
        "rec_con",
        "rec_ga",
        "rec_gn",
        "rec_oc",
        "rec_an",
        "rec_oil_risked",
        "rec_con_risked",
        "rec_ga_risked",
        "rec_gn_risked",
        "rec_oc_risked",
        "rec_an_risked",
        "res_oil",
        "res_con",
        "res_ga",
        "res_gn",
        "res_oc",
        "res_an",
        "prj_ioip",
        "prj_igip",
    ]

    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        nonzero = " OR ".join(f"COALESCE({c}, 0) != 0" for c in self.ZERO_COLUMNS)
        cols = ", ".join(
            ["report_year", "project_name", "wk_name", "field_name", "project_isactive"]
            + self.ZERO_COLUMNS
        )
        sql = (
            f"SELECT {cols} FROM project_resources"
            f" WHERE project_isactive = 0 AND ({nonzero})"
        )
        if year:
            year_list = ", ".join(str(y) for y in year)
            sql += f" AND report_year IN ({year_list})"

        rows = conn.execute(sql).fetchall()
        violations: list[Violation] = []

        for row in rows:
            report_year = row[0]
            project_name = row[1]
            wk_name = row[2]
            field_name = row[3]
            is_active = row[4]
            volumes = row[5:]

            current: dict[str, object] = {"project_isactive": is_active}
            for i, col in enumerate(self.ZERO_COLUMNS):
                current[col] = volumes[i]

            violations.append(
                Violation(
                    rule_id=self.rule_id,
                    rule_group="RE9",
                    description=self.description,
                    severity=self.severity,
                    table=self.applies_to_tables[0],
                    validated_column="project_isactive",
                    compared_columns=[],
                    identifiers={
                        "report_year": str(report_year),
                        "project_name": str(project_name),
                        "wk_name": str(wk_name),
                        "field_name": str(field_name),
                    },
                    current_values=current,
                )
            )

        return violations

    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        set_clause = ", ".join(f"{c} = 0" for c in self.ZERO_COLUMNS)
        fixes: list[tuple[str, list[object]]] = []
        for v in violations:
            name = v.identifiers["project_name"]
            report_year = int(v.identifiers["report_year"])
            sql = (
                f"UPDATE project_resources SET {set_clause}"
                " WHERE project_name = ?"
                " AND report_year = ?"
                " AND project_isactive = 0"
            )
            fixes.append((sql, [name, report_year]))
        return fixes
