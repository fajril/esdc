"""Extensible validation rule system for ESDC data."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import duckdb
import rich

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection
from esdc.selection import Severity

# Simple LaTeX tokens that do NOT take arguments.
_LATEX_TOKEN_MAP: dict[str, str] = {
    r"\implies": " -> ",
    r"\land": " & ",
    r"\lor": " | ",
    r"\forall": " for all ",
    r"\in": " in ",
    r"\exists": " exists ",
    r"\neg": " not ",
    r"\quad": "  ",
    r"\;": " ",
    r"\,": " ",
    r"\Delta": "delta ",
    r"\sum": "sum ",
    r"\times": " x ",
    r"\leq": " <= ",
    r"\geq": " >= ",
    r"\neq": " != ",
    r"\\": "",
    r"$": "",
}

# Regex to strip LaTeX commands that take a single braced argument,
# e.g. \text{project_name}, \mathbf{x}, \mathrm{val}.
_COMMAND_ARG_RE: re.Pattern[str] = re.compile(
    r"\\(?:text|mathbf|mathrm|mathit|mathcal)\{([^}]*)\}"
)


def render_formal(formal: str) -> str:
    """Render LaTeX formal notation as plain-text approximation.

    Processing order matters: commands with arguments are stripped
    first so that the surrounding braces are handled correctly.
    Remaining ``{`` and ``}`` characters (LaTeX grouping) are removed
    after all command processing.
    """
    result = formal.replace("  ", " ")
    result = _COMMAND_ARG_RE.sub(r"\1", result)
    for token, replacement in _LATEX_TOKEN_MAP.items():
        result = result.replace(token, replacement)
    result = result.replace("{", "").replace("}", "")
    return result.strip()


@dataclass
class Violation:
    """A single rule violation found in the database."""

    rule_id: str
    rule_group: str
    description: str
    severity: Severity
    table: str
    identifiers: dict[str, str]
    current_values: dict[str, object]
    fix_sql: str | None = None
    fix_applied: bool = False


@dataclass
class ValidationResult:
    """Result of running a single validation rule."""

    rule_id: str
    rule_group: str
    description: str
    formal: str
    severity: Severity
    is_fixable: bool
    total_violations: int
    violations: list[Violation]
    fix_applied_count: int = 0


class ValidationRule(ABC):
    """Base class for all validation rules.

    Subclasses must set class attributes and implement check() and
    generate_fixes().  The @register_rule decorator auto-registers them.
    """

    rule_id: str = ""
    description: str = ""
    formal: str = ""
    severity: Severity = Severity.WARNING
    is_fixable: bool = True
    applies_to_tables: list[str] = ["project_resources"]

    @abstractmethod
    def check(
        self,
        conn: duckdb.DuckDBPyConnection,
        year: list[int] | None = None,
    ) -> list[Violation]:
        """Run the rule against the DB. Return all violations."""
        ...

    @abstractmethod
    def generate_fixes(
        self, violations: list[Violation]
    ) -> list[tuple[str, list[object]]]:
        """Build parameterized SQL UPDATE statements for each violation.

        Returns a list of (sql, params) tuples so that callers can use
        ``conn.execute(sql, params)`` safely.
        """
        ...


# --- Rule Registry ---

_RULE_REGISTRY: dict[str, type[ValidationRule]] = {}


def register_rule(cls: type[ValidationRule]) -> type[ValidationRule]:
    """Decorator to register a validation rule class."""
    _RULE_REGISTRY[cls.rule_id] = cls
    return cls


def get_rule(rule_id: str) -> type[ValidationRule] | None:
    return _RULE_REGISTRY.get(rule_id)


def get_rules_by_group(group: str) -> list[type[ValidationRule]]:
    prefix = group.upper()
    return [
        cls for rid, cls in sorted(_RULE_REGISTRY.items()) if rid.startswith(prefix)
    ]


def get_all_rules() -> list[type[ValidationRule]]:
    return [cls for _, cls in sorted(_RULE_REGISTRY.items())]


def run_validation(
    rule_ids: list[str] | None = None,
    groups: list[str] | None = None,
    force_fix: bool = False,
    year: list[int] | None = None,
) -> list[ValidationResult]:
    """Run validation rules and optionally apply fixes.

    Parameters
    ----------
    rule_ids
        Specific rule IDs to run, e.g. ``["RE9001"]``.
    groups
        Rule groups, e.g. ``["RE9"]``. Ignored when *rule_ids* is given.
    force_fix
        If ``True``, try to execute fix SQL for each violation. Only
        works when the rule declares ``is_fixable=True``.
    year
        Optional list of report years to scope the check.

    Returns:
    -------
    list[ValidationResult]
        One result per rule that was executed.
    """
    db_path = Config.get_db_file()
    if not db_path.exists():
        rich.print("[red]Database not found. Run 'esdc fetch --save' first.[/red]")
        return []

    # Resolve rule classes from rule_ids, groups, or all
    if rule_ids:
        rule_classes = [
            cls
            for rid in rule_ids
            if (cls := get_rule(rid)) is not None
        ]
    elif groups:
        rule_classes: list[type[ValidationRule]] = []
        for g in groups:
            rule_classes.extend(get_rules_by_group(g))
    else:
        rule_classes = get_all_rules()

    if not rule_classes:
        rich.print("[yellow]No matching validation rules found.[/yellow]")
        return []

    results: list[ValidationResult] = []
    conn = get_duckdb_connection(db_path, read_only=not force_fix)

    try:
        any_fix_applied = False
        for rule_cls in rule_classes:
            rule = rule_cls()
            violations = rule.check(conn, year=year)
            fix_applied_count = 0

            if force_fix and violations:
                if not rule.is_fixable:
                    rich.print(
                        f"[yellow]  {rule.rule_id}: Fixable: No — "
                        f"skipping auto-fix (manual review required)[/yellow]"
                    )
                else:
                    fix_items = rule.generate_fixes(violations)
                    for i, (fix_sql, fix_params) in enumerate(fix_items):
                        try:
                            conn.execute(fix_sql, fix_params)
                            fix_applied_count += 1
                            violations[i].fix_applied = True
                            any_fix_applied = True
                        except duckdb.Error:
                            logging.exception("Fix failed for %s", rule.rule_id)

            result = ValidationResult(
                rule_id=rule.rule_id,
                rule_group=rule.rule_id[:3],
                description=rule.description,
                formal=rule.formal,
                severity=rule.severity,
                is_fixable=rule.is_fixable,
                total_violations=len(violations),
                violations=violations,
                fix_applied_count=fix_applied_count,
            )
            results.append(result)

        if any_fix_applied:
            conn.execute("CHECKPOINT")
    finally:
        conn.close()

    return results
