"""Transactional POD v2 API read-model rebuilds from canonical local inputs."""

from __future__ import annotations

import importlib.resources
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from functools import lru_cache
from typing import Any

from esdc.api_snapshot import iter_project_resources_rows
from esdc.pod_registry.contract import ColumnContract, load_pod_contract
from esdc.timeseries_snapshot import iter_timeseries_rows

_MIDDLE_CASE = "2. Middle Value"
_CENT = Decimal("0.01")
_NO_ROUND_MONITORING = frozenset(
    {
        "irr_pf_ctr_pod",
        "pv_ratio_pf_ctr_pod",
        "pi_pf_ctr_pod",
        "irr_pf_ctr_real",
        "pv_ratio_pf_ctr_real",
        "pi_pf_ctr_real",
        "irr_pf_ctr_otlk",
        "pv_ratio_pf_ctr_otlk",
        "pi_pf_ctr_otlk",
    }
)
_FORECAST_INPUT_DECIMALS = frozenset(
    {"usf_oil_pod", "usf_con_pod", "usf_ga_pod", "usf_gn_pod"}
)
_TS_COMPONENTS = {
    "usf_oil_otlk": ("cprd_sls_oil", "slf_oil"),
    "usf_con_otlk": ("cprd_sls_con", "slf_con"),
    "usf_ga_otlk": ("cprd_sls_ga", "slf_ga"),
    "usf_gn_otlk": ("cprd_sls_gn", "slf_gn"),
}
_RESOURCE_COMPONENTS = {
    "res_oil_2p": "res_oil",
    "res_con_2p": "res_con",
    "res_ga_2p": "res_ga",
    "res_gn_2p": "res_gn",
    "res_oc_2p": "res_oc",
    "res_an_2p": "res_an",
    "rec_oil_mid": "rec_oil",
    "rec_con_mid": "rec_con",
    "rec_ga_mid": "rec_ga",
    "rec_gn_mid": "rec_gn",
    "rec_oc_mid": "rec_oc",
    "rec_an_mid": "rec_an",
    "ioip_mid": "prj_ioip",
    "igip_mid": "prj_igip",
}
_RESOURCE_SALES_COMPONENTS = (
    "cprd_sls_oil",
    "cprd_sls_con",
    "cprd_sls_ga",
    "cprd_sls_gn",
    "cprd_sls_oc",
    "cprd_sls_an",
)
_RESOURCE_LIST_FIELDS = {
    "wk_name_list": "wk_name",
    "field_name_list": "field_name",
    "project_name_list": "project_name",
}


@dataclass(frozen=True, slots=True)
class RebuildDiagnostic:
    """Actionable reason a read-model value or row could not be derived."""

    code: str
    message: str
    pod_id_seq: int | None = None
    reported_date: str | None = None
    report_year: int | None = None
    year_frcast: int | None = None
    project_id: str | None = None
    component: str | None = None


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """Result counts and the exact source snapshot basis used for a rebuild."""

    monitoring_rows: int
    forecast_rows: int
    source_generation: int
    resource_revision: str | None
    resource_sha256: str | None
    timeseries_revision: str | None
    timeseries_sha256: str | None
    diagnostics: tuple[RebuildDiagnostic, ...]


@dataclass(slots=True)
class _RepeatedValue:
    count: int = 0
    value: Decimal | None = None
    missing: bool = False
    conflicting: bool = False

    def add(self, value: object) -> None:
        number = _as_decimal(value)
        self.count += 1
        if number is None:
            self.missing = True
        elif self.value is None:
            self.value = number
        elif number != self.value:
            self.conflicting = True

    def resolve(self) -> Decimal | None:
        if self.count == 0 or self.missing or self.conflicting:
            return None
        return self.value


@dataclass(slots=True)
class _TimeseriesGroup:
    components: dict[str, _RepeatedValue]


@lru_cache(maxsize=1)
def _output_columns() -> dict[str, tuple[ColumnContract, ...]]:
    contract = load_pod_contract()
    return {
        name: contract.tables[name].columns
        for name in ("pod_monitoring_api", "pod_forecast_api")
    }


@lru_cache(maxsize=1)
def _forecast_input_columns() -> dict[str, ColumnContract]:
    table = load_pod_contract().tables["pod_forecast"]
    return {column.name: column for column in table.columns}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _require_v2_tables(conn: sqlite3.Connection) -> int:
    if not conn.in_transaction:
        raise RuntimeError(
            "POD read-model rebuild requires an active caller transaction"
        )
    required = (
        "pod_registry_state",
        "pod_monitoring_api",
        "pod_forecast_api",
        "m_pod",
        "pod_project",
        "pod_monitoring",
        "pod_forecast",
    )
    missing = [name for name in required if not _table_exists(conn, name)]
    if missing:
        raise RuntimeError(
            "POD read-model rebuild requires initialized v2 tables: "
            + ", ".join(missing)
        )
    state = conn.execute(
        "SELECT schema_version, contract_id, contract_version, dictionary_version, "
        "source_generation "
        "FROM pod_registry_state WHERE singleton_id = 1"
    ).fetchone()
    expected = (2, "esdc.pod", 1, load_pod_contract().dictionary_version)
    if state is None or tuple(state[:4]) != expected:
        raise RuntimeError("POD read-model rebuild requires the supported v2 registry")
    return int(state[4])


def _install_read_model_indexes(conn: sqlite3.Connection) -> None:
    script = (
        importlib.resources.files("esdc")
        .joinpath("sql")
        .joinpath("pod_read_models.sql")
        .read_text(encoding="utf-8")
    )
    for statement in script.split(";"):
        sql = statement.strip()
        if sql:
            conn.execute(sql)


def _snapshot_basis(
    conn: sqlite3.Connection, source_table: str
) -> tuple[str | None, str | None]:
    if not _table_exists(conn, "api_snapshot"):
        return None, None
    row = conn.execute(
        "SELECT revision, source_sha256 FROM api_snapshot WHERE source_table = ?",
        (source_table,),
    ).fetchone()
    if row is None:
        return None, None
    return (str(row[0]), str(row[1]))


def _fetch_dicts(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    cursor = conn.execute(f'SELECT * FROM "{table}"')
    names = tuple(column[0] for column in cursor.description or ())
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if number != number.to_integral_value():
        return None
    return int(number)


def _as_decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        number = value
    else:
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            return None
    if not number.is_finite():
        return None
    return number


def _decimal_text(value: Decimal, *, scale: Decimal | None = None) -> str:
    if scale is not None:
        value = value.quantize(scale, rounding=ROUND_HALF_UP)
    return format(value, "f")


def _validate_decimal_precision(value: Decimal, column: ColumnContract) -> None:
    if column.precision is None or column.scale is None:
        return
    quantum = Decimal(1).scaleb(-column.scale)
    try:
        quantized = value.quantize(quantum)
    except InvalidOperation as exc:
        raise ValueError(
            f"{column.name} exceeds precision {column.precision}, scale {column.scale}"
        ) from exc
    if quantized != value:
        raise ValueError(
            f"{column.name} has more than scale {column.scale} fractional digits"
        )
    integer_digits = column.precision - column.scale
    if abs(value) >= Decimal(10) ** integer_digits:
        raise ValueError(
            f"{column.name} exceeds precision {column.precision}, scale {column.scale}"
        )


def _quantized_decimal_text(value: Decimal, column: ColumnContract) -> str:
    if column.scale is None:
        return _decimal_text(value)
    quantum = Decimal(1).scaleb(-column.scale)
    try:
        quantized = value.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError(
            f"{column.name} exceeds precision {column.precision}, scale {column.scale}"
        ) from exc
    _validate_decimal_precision(quantized, column)
    return _decimal_text(quantized)


def _add_diagnostic(
    diagnostics: set[RebuildDiagnostic],
    code: str,
    message: str,
    *,
    pod_id_seq: int | None = None,
    reported_date: str | None = None,
    report_year: int | None = None,
    year_frcast: int | None = None,
    project_id: str | None = None,
    component: str | None = None,
) -> None:
    diagnostics.add(
        RebuildDiagnostic(
            code,
            message,
            pod_id_seq,
            reported_date,
            report_year,
            year_frcast,
            project_id,
            component,
        )
    )


def _load_resource_groups(
    conn: sqlite3.Connection,
    project_ids: set[str],
    diagnostics: set[RebuildDiagnostic],
) -> tuple[
    dict[tuple[str, int], list[object]],
    dict[str, set[int]],
]:
    middle: dict[tuple[str, int], list[object]] = defaultdict(list)
    contexts: dict[str, set[int]] = defaultdict(set)
    if not _table_exists(conn, "api_project_resources"):
        _add_diagnostic(
            diagnostics,
            "resource_snapshot_missing",
            "Local project-resource snapshot rows are not available.",
        )
        return middle, contexts
    for row in iter_project_resources_rows(conn):
        project_id = row.project_id
        report_year = _as_int(row.report_year)
        if project_id not in project_ids:
            continue
        if report_year is None:
            _add_diagnostic(
                diagnostics,
                "resource_report_year_invalid",
                "A mapped project-resource row has no usable report year.",
                project_id=project_id,
            )
            continue
        contexts[project_id].add(report_year)
        if row.uncert_level == _MIDDLE_CASE:
            middle[(project_id, report_year)].append(row)
    return middle, contexts


def _load_timeseries_groups(
    conn: sqlite3.Connection,
    project_ids: set[str],
    diagnostics: set[RebuildDiagnostic],
) -> tuple[
    dict[tuple[str, int, int], _TimeseriesGroup],
    dict[str, set[int]],
    dict[tuple[str, int], set[int]],
]:
    groups: dict[tuple[str, int, int], _TimeseriesGroup] = {}
    contexts: dict[str, set[int]] = defaultdict(set)
    series_years: dict[tuple[str, int], set[int]] = defaultdict(set)
    if not _table_exists(conn, "api_project_timeseries"):
        _add_diagnostic(
            diagnostics,
            "timeseries_snapshot_missing",
            "Local project-timeseries snapshot rows are not available.",
        )
        return groups, contexts, series_years
    fields = tuple(value[1] for value in _TS_COMPONENTS.values())
    for row in iter_timeseries_rows(conn):
        try:
            payload = json.loads(row.row_json, parse_float=Decimal)
        except (json.JSONDecodeError, TypeError):
            _add_diagnostic(
                diagnostics,
                "timeseries_json_invalid",
                "A canonical project-timeseries row is not valid JSON.",
            )
            continue
        if not isinstance(payload, dict):
            continue
        project_id = payload.get("project_id")
        if not isinstance(project_id, str) or project_id not in project_ids:
            continue
        report_year = row.report_year
        series_year = _as_int(payload.get("year"))
        if report_year is None or series_year is None:
            _add_diagnostic(
                diagnostics,
                "timeseries_year_missing",
                "A mapped project-timeseries row lacks a usable report or series year.",
                project_id=project_id,
                report_year=report_year,
            )
            continue
        key = (project_id, report_year, series_year)
        group = groups.setdefault(
            key, _TimeseriesGroup({field: _RepeatedValue() for field in fields})
        )
        for field in fields:
            group.components[field].add(payload.get(field))
        contexts[project_id].add(report_year)
        series_years[(project_id, report_year)].add(series_year)
    return groups, contexts, series_years


def _resource_value(
    middle: dict[tuple[str, int], list[object]],
    project_id: str,
    year: int,
    component: str,
    diagnostics: set[RebuildDiagnostic],
    *,
    pod_id_seq: int | None,
    reported_date: str | None,
    report_year: int | None,
    year_frcast: int | None,
) -> Decimal | None:
    rows = middle.get((project_id, year), [])
    if len(rows) != 1:
        _add_diagnostic(
            diagnostics,
            "resource_group_incomplete",
            "A mapped project must have exactly one middle-case row for this year.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=year_frcast,
            project_id=project_id,
            component=component,
        )
        return None
    value = _as_decimal(getattr(rows[0], component, None))
    if value is None:
        _add_diagnostic(
            diagnostics,
            "resource_component_missing",
            "A required middle-case resource component is NULL or nonnumeric.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=year_frcast,
            project_id=project_id,
            component=component,
        )
    return value


def _aggregate_resource(
    project_ids: tuple[str, ...],
    middle: dict[tuple[str, int], list[object]],
    year: int,
    component: str,
    diagnostics: set[RebuildDiagnostic],
    *,
    pod_id_seq: int,
    reported_date: str,
) -> Decimal | None:
    if not project_ids:
        _add_diagnostic(
            diagnostics,
            "project_mapping_missing",
            "No project mapping is available for this POD.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=year,
            component=component,
        )
        return None
    values = [
        _resource_value(
            middle,
            project_id,
            year,
            component,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=year,
            year_frcast=None,
        )
        for project_id in project_ids
    ]
    if any(value is None for value in values):
        return None
    return sum((value for value in values if value is not None), Decimal(0))


def _resource_names(
    project_ids: tuple[str, ...],
    all_rows: dict[tuple[str, int], list[object]],
    year: int,
    field: str,
    *,
    require_nonblank: bool = False,
) -> str | None:
    names: set[str] = set()
    for project_id in project_ids:
        for row in all_rows.get((project_id, year), ()):
            value = getattr(row, field, None)
            if value is None:
                continue
            name = str(value)
            if require_nonblank and not name.strip():
                continue
            names.add(name)
    return ";".join(sorted(names)) if names else None


def _project_level(
    project_ids: tuple[str, ...],
    middle: dict[tuple[str, int], list[object]],
    year: int,
    diagnostics: set[RebuildDiagnostic],
    *,
    pod_id_seq: int,
    reported_date: str,
) -> str | None:
    levels: list[str] = []
    for project_id in project_ids:
        rows = middle.get((project_id, year), [])
        if len(rows) != 1:
            _add_diagnostic(
                diagnostics,
                "project_level_incomplete",
                "A project level requires exactly one middle-case row per mapping.",
                pod_id_seq=pod_id_seq,
                reported_date=reported_date,
                report_year=year,
                project_id=project_id,
                component="project_level",
            )
            return None
        level = getattr(rows[0], "project_level", None)
        if not isinstance(level, str) or not _known_level(level):
            _add_diagnostic(
                diagnostics,
                "project_level_unknown",
                "A mapped project has a missing or unrecognized historical level.",
                pod_id_seq=pod_id_seq,
                reported_date=reported_date,
                report_year=year,
                project_id=project_id,
                component="project_level",
            )
            return None
        levels.append(level)
    if not levels:
        return None
    domains = {level[0] for level in levels}
    if not domains.intersection({"E", "X"}) and len(set(levels)) > 1:
        return None
    return min(levels, key=_level_rank)


def _known_level(level: str) -> bool:
    return (
        len(level) >= 4
        and level[0] in "EXA"
        and level[1].isdigit()
        and level[2:4] == ". "
        and (
            (level[0] == "E" and int(level[1]) <= 8)
            or (level[0] == "X" and int(level[1]) <= 6)
            or (level[0] == "A" and int(level[1]) in (1, 2))
        )
    )


def _level_rank(level: str) -> tuple[int, int, str]:
    return ({"E": 0, "X": 1, "A": 2}[level[0]], int(level[1]), level)


def _resolve_series_component(
    project_id: str,
    report_year: int,
    series_year: int,
    component: str,
    groups: dict[tuple[str, int, int], _TimeseriesGroup],
    middle: dict[tuple[str, int], list[object]],
    diagnostics: set[RebuildDiagnostic],
    *,
    pod_id_seq: int,
    reported_date: str,
) -> Decimal | None:
    if series_year <= report_year:
        this_year = _resource_value(
            middle,
            project_id,
            series_year,
            component,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=series_year,
        )
        previous_year = _resource_value(
            middle,
            project_id,
            series_year - 1,
            component,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=series_year,
        )
        if this_year is None or previous_year is None:
            return None
        return this_year - previous_year

    group = groups.get((project_id, report_year, series_year))
    field = _TS_COMPONENTS[
        next(name for name, pair in _TS_COMPONENTS.items() if pair[0] == component)
    ][1]
    repeated = group.components[field] if group is not None else None
    value = repeated.resolve() if repeated is not None else None
    if value is None:
        code = (
            "timeseries_value_conflicting"
            if repeated and repeated.conflicting
            else "timeseries_value_missing"
        )
        _add_diagnostic(
            diagnostics,
            code,
            "A future outlook component is missing, NULL, or disagrees "
            "across duplicate rows.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=series_year,
            project_id=project_id,
            component=field,
        )
    return value


def _aggregate_series_component(
    project_ids: tuple[str, ...],
    report_year: int,
    series_year: int,
    component: str,
    groups: dict[tuple[str, int, int], _TimeseriesGroup],
    middle: dict[tuple[str, int], list[object]],
    diagnostics: set[RebuildDiagnostic],
    *,
    pod_id_seq: int,
    reported_date: str,
) -> Decimal | None:
    if not project_ids:
        _add_diagnostic(
            diagnostics,
            "project_mapping_missing",
            "No project mapping is available for this POD.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
            report_year=report_year,
            year_frcast=series_year,
            component=component,
        )
        return None
    values = [
        _resolve_series_component(
            project_id,
            report_year,
            series_year,
            component,
            groups,
            middle,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
        )
        for project_id in project_ids
    ]
    if any(value is None for value in values):
        return None
    return sum((value for value in values if value is not None), Decimal(0))


def _sum_required(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None or right is None:
        return None
    return left + right


def _sum_nullable_as_zero(left: object, right: object) -> Decimal | None:
    if left is not None and _as_decimal(left) is None:
        return None
    if right is not None and _as_decimal(right) is None:
        return None
    return (_as_decimal(left) or Decimal(0)) + (_as_decimal(right) or Decimal(0))


def _monitoring_row(
    master: dict[str, Any],
    monitoring: dict[str, Any],
    projects: tuple[str, ...],
    middle: dict[tuple[str, int], list[object]],
    groups: dict[tuple[str, int, int], _TimeseriesGroup],
    diagnostics: set[RebuildDiagnostic],
    references: dict[str, dict[int, str]],
    predecessors: dict[int, tuple[str, ...]],
) -> dict[str, Any]:
    pod_id_seq = int(master["pod_id_seq"])
    reported_date = str(monitoring["reported_date"])
    year = _as_int(reported_date[:4])
    row: dict[str, Any] = {
        column.name: None for column in _output_columns()["pod_monitoring_api"]
    }
    sources = {"m_pod": master, "pod_monitoring": monitoring}
    source_aliases = {
        "approval_institution": ("r_institution", "institution_id"),
        "pod_type": ("r_pod_type", "pod_type_id"),
        "pod_stage": ("r_pod_stage", "pod_stage_id"),
    }
    for column in _output_columns()["pod_monitoring_api"]:
        if column.source in sources and column.name in sources[column.source]:
            row[column.name] = sources[column.source][column.name]
        elif column.source == "m_pod" and column.name == "onstream_date_pod":
            row[column.name] = master.get("onstream_date")
        elif column.name in source_aliases:
            ref_table, id_name = source_aliases[column.name]
            source_name = "pod_monitoring" if ref_table == "r_pod_stage" else "m_pod"
            reference_id = _as_int(sources[source_name].get(id_name))
            row[column.name] = (
                references[ref_table].get(reference_id)
                if reference_id is not None
                else None
            )
        elif column.name == "preceded_by_pod_name":
            row[column.name] = ";".join(predecessors.get(pod_id_seq, ())) or None
    row["reported_date"] = reported_date

    if year is None:
        _add_diagnostic(
            diagnostics,
            "monitoring_report_year_invalid",
            "A monitoring submission has no usable reporting year.",
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
        )
        return row
    for output_name, source_name in _RESOURCE_LIST_FIELDS.items():
        row[output_name] = _resource_names(
            projects,
            middle,
            year,
            source_name,
            require_nonblank=output_name == "wk_name_list",
        )
    row["project_level_highest"] = _project_level(
        projects,
        middle,
        year,
        diagnostics,
        pod_id_seq=pod_id_seq,
        reported_date=reported_date,
    )
    for output_name, component in _RESOURCE_COMPONENTS.items():
        row[output_name] = _aggregate_resource(
            projects,
            middle,
            year,
            component,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
        )
    for component in _RESOURCE_SALES_COMPONENTS:
        row[component] = _aggregate_resource(
            projects,
            middle,
            year,
            component,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
        )
    row["res_oc_2p_pod"] = _sum_nullable_as_zero(
        master.get("res_oil_2p_pod"), master.get("res_con_2p_pod")
    )
    row["res_an_2p_pod"] = _sum_nullable_as_zero(
        master.get("res_ga_2p_pod"), master.get("res_gn_2p_pod")
    )
    row["res_oc_2p_risked_pod"] = _sum_required(
        _as_decimal(master.get("res_oil_2p_risked_pod")),
        _as_decimal(master.get("res_con_2p_risked_pod")),
    )
    row["res_an_2p_risked_pod"] = _sum_required(
        _as_decimal(master.get("res_ga_2p_risked_pod")),
        _as_decimal(master.get("res_gn_2p_risked_pod")),
    )
    return row


def _monitoring_inputs(
    conn: sqlite3.Connection,
    pods: dict[int, dict[str, Any]],
    all_pods: dict[int, dict[str, Any]],
    diagnostics: set[RebuildDiagnostic],
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "pod_monitoring"):
        return []
    result = []
    for row in _fetch_dicts(conn, "pod_monitoring"):
        pod_id_seq = _as_int(row.get("pod_id_seq"))
        master = pods.get(pod_id_seq) if pod_id_seq is not None else None
        if master is None:
            continue
        date_value = row.get("reported_date")
        date_text = str(date_value) if date_value is not None else ""
        approval_date = str(master.get("approval_date") or "")
        successor_id = _as_int(master.get("superseded_by"))
        successor = all_pods.get(successor_id) if successor_id is not None else None
        if not date_text or approval_date > date_text or (
            successor is not None
            and str(successor.get("approval_date") or "") <= date_text
        ):
            _add_diagnostic(
                diagnostics,
                "monitoring_not_temporally_eligible",
                "The POD approval interval does not cover this monitoring date.",
                pod_id_seq=pod_id_seq,
                reported_date=date_text or None,
            )
            continue
        result.append(row)
    result.sort(key=lambda item: (int(item["pod_id_seq"]), str(item["reported_date"])))
    return result


def _forecast_inputs(
    conn: sqlite3.Connection,
    pod_ids: set[int],
    diagnostics: set[RebuildDiagnostic],
) -> dict[tuple[int, int], dict[str, Any]]:
    forecasts: dict[tuple[int, int], dict[str, Any]] = {}
    if not _table_exists(conn, "pod_forecast"):
        return forecasts
    for row in _fetch_dicts(conn, "pod_forecast"):
        pod_id_seq = _as_int(row.get("pod_id_seq"))
        year = _as_int(row.get("year_frcast"))
        if pod_id_seq is None or year is None or pod_id_seq not in pod_ids:
            continue
        for name in _FORECAST_INPUT_DECIMALS:
            raw_value = row.get(name)
            if raw_value is None:
                continue
            number = _as_decimal(raw_value)
            if number is None:
                raise ValueError(f"{name} must be a finite numeric value")
            _validate_decimal_precision(number, _forecast_input_columns()[name])
        forecasts[(pod_id_seq, year)] = row
    return forecasts


def _forecast_row(
    master: dict[str, Any],
    report_year: int,
    series_year: int,
    approved: dict[str, Any] | None,
    monpod_id: int | None,
    projects: tuple[str, ...],
    groups: dict[tuple[str, int, int], _TimeseriesGroup],
    middle: dict[tuple[str, int], list[object]],
    diagnostics: set[RebuildDiagnostic],
) -> dict[str, Any]:
    pod_id_seq = int(master["pod_id_seq"])
    reported_date = f"{report_year:04d}-12-31"
    row: dict[str, Any] = {
        column.name: None for column in _output_columns()["pod_forecast_api"]
    }
    if approved is not None:
        for name in (
            "frcast_id",
            "usf_oil_pod",
            "usf_con_pod",
            "usf_ga_pod",
            "usf_gn_pod",
            "frcast_remarks",
            "created_at",
            "created_by",
            "updated_at",
            "updated_by",
        ):
            row[name] = approved.get(name)
    row.update(
        {
            "reported_date": reported_date,
            "monpod_id": monpod_id,
            "pod_id_seq": pod_id_seq,
            "pod_id": master.get("pod_id"),
            "pod_name": master.get("pod_name"),
            "year_frcast": series_year,
        }
    )
    approved_oil = _as_decimal(approved.get("usf_oil_pod")) if approved else None
    approved_con = _as_decimal(approved.get("usf_con_pod")) if approved else None
    approved_ga = _as_decimal(approved.get("usf_ga_pod")) if approved else None
    approved_gn = _as_decimal(approved.get("usf_gn_pod")) if approved else None
    row["usf_oc_pod"] = _sum_required(approved_oil, approved_con)
    row["usf_an_pod"] = _sum_required(approved_ga, approved_gn)
    for output_name, (resource_component, _) in _TS_COMPONENTS.items():
        row[output_name] = _aggregate_series_component(
            projects,
            report_year,
            series_year,
            resource_component,
            groups,
            middle,
            diagnostics,
            pod_id_seq=pod_id_seq,
            reported_date=reported_date,
        )
    row["usf_oc_otlk"] = _sum_required(row["usf_oil_otlk"], row["usf_con_otlk"])
    row["usf_an_otlk"] = _sum_required(row["usf_ga_otlk"], row["usf_gn_otlk"])
    return row


def _active_at(
    master: dict[str, Any], all_pods: dict[int, dict[str, Any]], date_text: str
) -> bool:
    if str(master.get("approval_date") or "") > date_text:
        return False
    successor_id = _as_int(master.get("superseded_by"))
    if successor_id is None:
        return True
    successor = all_pods.get(successor_id)
    return successor is None or str(successor.get("approval_date") or "") > date_text


def _prepare_references(conn: sqlite3.Connection) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for table, id_column, value_column in (
        ("r_institution", "id", "institution"),
        ("r_pod_type", "id", "pod_type"),
        ("r_pod_stage", "id", "stage_name"),
    ):
        result[table] = {
            int(row[0]): str(row[1])
            for row in conn.execute(
                f'SELECT "{id_column}", "{value_column}" FROM "{table}"'
            )
        }
    return result


def _prepare_predecessors(
    pods: dict[int, dict[str, Any]],
) -> dict[int, tuple[str, ...]]:
    predecessors: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for pod_id_seq, master in pods.items():
        successor = _as_int(master.get("superseded_by"))
        if successor is not None and master.get("pod_name") is not None:
            predecessors[successor].append((pod_id_seq, str(master["pod_name"])))
    return {
        pod_id: tuple(name for _, name in sorted(items))
        for pod_id, items in predecessors.items()
    }


def _encode_output_rows(
    table: str,
    rows: list[dict[str, Any]],
    *,
    monitoring: bool,
) -> list[tuple[Any, ...]]:
    columns = _output_columns()[table]
    values = []
    for row in rows:
        encoded = []
        for column in columns:
            value = row.get(column.name)
            if column.logical_type == "decimal" and value is not None:
                number = _as_decimal(value)
                if number is None:
                    raise ValueError(
                        f"{column.name} must be a finite numeric value"
                    )
                if monitoring:
                    try:
                        finite_double = math.isfinite(float(number))
                    except (OverflowError, ValueError) as exc:
                        raise ValueError(
                            f"{column.name} exceeds finite output precision"
                        ) from exc
                    if not finite_double:
                        raise ValueError(
                            f"{column.name} exceeds finite output precision"
                        )
                if not monitoring and column.name in _FORECAST_INPUT_DECIMALS:
                    _validate_decimal_precision(
                        number, _forecast_input_columns()[column.name]
                    )
                elif (
                    not monitoring
                    and column.name.startswith("usf_")
                    and column.scale is not None
                ):
                    value = _quantized_decimal_text(number, column)
                elif monitoring and column.name in _NO_ROUND_MONITORING:
                    value = (
                        _decimal_text(number)
                        if column.scale is not None
                        else float(number)
                    )
                elif monitoring:
                    try:
                        with localcontext() as context:
                            context.prec = max(context.prec, number.adjusted() + 3)
                            number = number.quantize(_CENT, rounding=ROUND_HALF_UP)
                    except InvalidOperation as exc:
                        raise ValueError(
                            f"{column.name} exceeds finite output precision"
                        ) from exc
                    value = (
                        _decimal_text(number)
                        if column.scale is not None
                        else float(number)
                    )
                elif column.scale is not None:
                    scale = Decimal(1).scaleb(-column.scale)
                    value = _decimal_text(number, scale=scale)
                else:
                    value = float(number)
                persisted_number = _as_decimal(value)
                if persisted_number is None:
                    raise ValueError(
                        f"{column.name} cannot be represented as a finite numeric value"
                    )
                _validate_decimal_precision(persisted_number, column)
            encoded.append(value)
        values.append(tuple(encoded))
    return values


def _write_output_rows(
    conn: sqlite3.Connection, table: str, values: list[tuple[Any, ...]]
) -> None:
    if not values:
        return
    names = tuple(column.name for column in _output_columns()[table])
    sql = (
        f'INSERT INTO "{table}" ('
        + ", ".join(f'"{name}"' for name in names)
        + ") VALUES ("
        + ", ".join("?" for _ in names)
        + ")"
    )
    conn.executemany(sql, values)


def _delete_existing(
    conn: sqlite3.Connection, table: str, pod_ids: tuple[int, ...] | None
) -> None:
    if pod_ids is None:
        conn.execute(f'DELETE FROM "{table}"')
        return
    for start in range(0, len(pod_ids), 500):
        chunk = pod_ids[start : start + 500]
        if not chunk:
            continue
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(
            f'DELETE FROM "{table}" WHERE pod_id_seq IN ({placeholders})', chunk
        )


def _persist_rebuild_diagnostics(
    conn: sqlite3.Connection,
    report: RebuildReport,
    selected: tuple[int, ...] | None,
) -> None:
    """Persist diagnostics with the exact generation and source snapshot used."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pod_read_model_snapshot (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            source_generation INTEGER NOT NULL,
            resource_revision TEXT,
            resource_sha256 TEXT,
            timeseries_revision TEXT,
            timeseries_sha256 TEXT,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pod_read_model_diagnostic (
            diagnostic_id INTEGER PRIMARY KEY,
            source_generation INTEGER NOT NULL,
            resource_revision TEXT,
            resource_sha256 TEXT,
            timeseries_revision TEXT,
            timeseries_sha256 TEXT,
            code TEXT NOT NULL,
            message TEXT NOT NULL,
            pod_id_seq INTEGER,
            reported_date TEXT,
            report_year INTEGER,
            year_frcast INTEGER,
            project_id TEXT,
            component TEXT
        )
        """
    )
    previous_snapshot = conn.execute(
        "SELECT source_generation, resource_revision, resource_sha256, "
        "timeseries_revision, timeseries_sha256 "
        "FROM pod_read_model_snapshot WHERE singleton_id=1"
    ).fetchone()
    current_basis = (
        report.resource_revision,
        report.resource_sha256,
        report.timeseries_revision,
        report.timeseries_sha256,
    )
    if selected is not None:
        previous_basis = (
            None
            if previous_snapshot is None
            else tuple(previous_snapshot[index] for index in range(1, 5))
        )
        if previous_basis != current_basis:
            raise RuntimeError(
                "scoped POD rebuild cannot carry diagnostics across a changed "
                "resource/timeseries snapshot basis"
            )

    conn.execute(
        """
        INSERT INTO pod_read_model_snapshot (
            singleton_id, source_generation, resource_revision, resource_sha256,
            timeseries_revision, timeseries_sha256
        ) VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT (singleton_id) DO UPDATE SET
            source_generation=excluded.source_generation,
            resource_revision=excluded.resource_revision,
            resource_sha256=excluded.resource_sha256,
            timeseries_revision=excluded.timeseries_revision,
            timeseries_sha256=excluded.timeseries_sha256,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
        """,
        (
            report.source_generation,
            report.resource_revision,
            report.resource_sha256,
            report.timeseries_revision,
            report.timeseries_sha256,
        ),
    )
    if selected is None:
        conn.execute("DELETE FROM pod_read_model_diagnostic")
    else:
        assert previous_snapshot is not None
        previous_generation = int(previous_snapshot[0])
        if selected:
            placeholders = ",".join("?" for _ in selected)
            conn.execute(
                "UPDATE pod_read_model_diagnostic SET source_generation=? "
                f"WHERE source_generation=? AND pod_id_seq IS NOT NULL "
                f"AND pod_id_seq NOT IN ({placeholders})",
                (report.source_generation, previous_generation, *selected),
            )
            conn.execute(
                "DELETE FROM pod_read_model_diagnostic "
                f"WHERE pod_id_seq IN ({placeholders})",
                selected,
            )
        else:
            conn.execute(
                "UPDATE pod_read_model_diagnostic SET source_generation=? "
                "WHERE source_generation=? AND pod_id_seq IS NOT NULL",
                (report.source_generation, previous_generation),
            )
        conn.execute("DELETE FROM pod_read_model_diagnostic WHERE pod_id_seq IS NULL")
    conn.executemany(
        """
        INSERT INTO pod_read_model_diagnostic (
            source_generation, resource_revision, resource_sha256,
            timeseries_revision, timeseries_sha256, code, message, pod_id_seq,
            reported_date, report_year, year_frcast, project_id, component
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                report.source_generation,
                report.resource_revision,
                report.resource_sha256,
                report.timeseries_revision,
                report.timeseries_sha256,
                diagnostic.code,
                diagnostic.message,
                diagnostic.pod_id_seq,
                diagnostic.reported_date,
                diagnostic.report_year,
                diagnostic.year_frcast,
                diagnostic.project_id,
                diagnostic.component,
            )
            for diagnostic in report.diagnostics
        ],
    )


def rebuild_pod_read_models(
    conn: sqlite3.Connection, *, pod_ids: tuple[int, ...] | None = None
) -> RebuildReport:
    """Rebuild physical API output tables inside the caller's source transaction.

    Resource and timeseries inputs are consumed through their typed SQLite
    iterators. No generation is advanced here; callers use
    :func:`refresh_pod_after_source_change` once after a logical source update.
    """
    source_generation = _require_v2_tables(conn)
    from esdc.pod_registry.migrations import _ensure_registry_writer

    _ensure_registry_writer(conn)
    selected = None if pod_ids is None else tuple(dict.fromkeys(pod_ids))
    if selected is not None and any(_as_int(value) is None for value in selected):
        raise ValueError("pod_ids must contain integer POD sequence values")
    selected_set = None if selected is None else set(selected)
    diagnostics: set[RebuildDiagnostic] = set()

    resource_revision, resource_sha256 = _snapshot_basis(conn, "project_resources")
    timeseries_revision, timeseries_sha256 = _snapshot_basis(
        conn, "project_timeseries"
    )
    if resource_revision is None:
        _add_diagnostic(
            diagnostics,
            "resource_snapshot_basis_missing",
            "No committed project-resource snapshot revision is recorded.",
        )
    if timeseries_revision is None:
        _add_diagnostic(
            diagnostics,
            "timeseries_snapshot_basis_missing",
            "No committed project-timeseries snapshot revision is recorded.",
        )

    if selected is not None:
        previous_snapshot = conn.execute(
            "SELECT resource_revision, resource_sha256, timeseries_revision, "
            "timeseries_sha256 FROM pod_read_model_snapshot WHERE singleton_id=1"
        ).fetchone()
        current_basis = (
            resource_revision,
            resource_sha256,
            timeseries_revision,
            timeseries_sha256,
        )
        if previous_snapshot is None or tuple(previous_snapshot) != current_basis:
            selected = None
            selected_set = None

    pods = {
        int(row["pod_id_seq"]): row
        for row in _fetch_dicts(conn, "m_pod")
        if selected_set is None or int(row["pod_id_seq"]) in selected_set
    }
    all_pods = {
        int(row["pod_id_seq"]): row for row in _fetch_dicts(conn, "m_pod")
    }
    mapping_rows = [
        row
        for row in _fetch_dicts(conn, "pod_project")
        if int(row["pod_id_seq"]) in pods
    ]
    projects_by_pod: dict[int, tuple[str, ...]] = {}
    project_lists: dict[int, list[str]] = defaultdict(list)
    for row in mapping_rows:
        project_lists[int(row["pod_id_seq"])].append(str(row["project_id"]))
    for pod_id_seq in pods:
        projects_by_pod[pod_id_seq] = tuple(sorted(set(project_lists[pod_id_seq])))
        if not projects_by_pod[pod_id_seq]:
            _add_diagnostic(
                diagnostics,
                "project_mapping_missing",
                "The POD has no project mapping for read-model aggregation.",
                pod_id_seq=pod_id_seq,
            )
    project_ids = {
        project_id for projects in projects_by_pod.values() for project_id in projects
    }
    middle, resource_contexts = _load_resource_groups(conn, project_ids, diagnostics)
    timeseries_groups, timeseries_contexts, timeseries_series_years = (
        _load_timeseries_groups(conn, project_ids, diagnostics)
    )
    references = _prepare_references(conn)
    predecessors = _prepare_predecessors(all_pods)
    monitoring_inputs = _monitoring_inputs(conn, pods, all_pods, diagnostics)
    monitoring_lookup = {
        (int(row["pod_id_seq"]), str(row["reported_date"])): row
        for row in monitoring_inputs
    }
    forecast_inputs = _forecast_inputs(conn, set(pods), diagnostics)

    monitoring_output: list[dict[str, Any]] = []
    for monitoring in monitoring_inputs:
        pod_id_seq = int(monitoring["pod_id_seq"])
        master = pods[pod_id_seq]
        monitoring_output.append(
            _monitoring_row(
                master,
                monitoring,
                projects_by_pod[pod_id_seq],
                middle,
                timeseries_groups,
                diagnostics,
                references,
                predecessors,
            )
        )

    report_contexts: dict[int, set[int]] = defaultdict(set)
    series_by_pod_context: dict[tuple[int, int], set[int]] = defaultdict(set)
    for pod_id_seq, projects in projects_by_pod.items():
        for project_id in projects:
            report_contexts[pod_id_seq].update(resource_contexts.get(project_id, set()))
            report_contexts[pod_id_seq].update(
                timeseries_contexts.get(project_id, set())
            )
            for report_year in timeseries_contexts.get(project_id, set()):
                series_by_pod_context[(pod_id_seq, report_year)].update(
                    timeseries_series_years.get((project_id, report_year), set())
                )
    forecasts_by_pod: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (pod_id_seq, year), forecast in forecast_inputs.items():
        forecasts_by_pod[pod_id_seq].append((year, forecast))
        if not report_contexts.get(pod_id_seq):
            _add_diagnostic(
                diagnostics,
                "forecast_reporting_context_missing",
                "An approved forecast has no annual resource or timeseries "
                "report-year context.",
                pod_id_seq=pod_id_seq,
                year_frcast=year,
            )

    forecast_output: list[dict[str, Any]] = []
    for pod_id_seq, master in pods.items():
        projects = projects_by_pod[pod_id_seq]
        approved_by_year = dict(forecasts_by_pod.get(pod_id_seq, ()))
        for report_year in sorted(report_contexts.get(pod_id_seq, set())):
            reported_date = f"{report_year:04d}-12-31"
            if not _active_at(master, all_pods, reported_date):
                continue
            series_years = set(
                series_by_pod_context.get((pod_id_seq, report_year), set())
            )
            series_years.update(approved_by_year)
            mon = monitoring_lookup.get((pod_id_seq, reported_date))
            monpod_id = _as_int(mon.get("monpod_id")) if mon is not None else None
            for series_year in sorted(series_years):
                forecast_output.append(
                    _forecast_row(
                        master,
                        report_year,
                        series_year,
                        approved_by_year.get(series_year),
                        monpod_id,
                        projects,
                        timeseries_groups,
                        middle,
                        diagnostics,
                    )
                )

    monitoring_values = _encode_output_rows(
        "pod_monitoring_api", monitoring_output, monitoring=True
    )
    forecast_values = _encode_output_rows(
        "pod_forecast_api", forecast_output, monitoring=False
    )
    _install_read_model_indexes(conn)
    _delete_existing(conn, "pod_monitoring_api", selected)
    _delete_existing(conn, "pod_forecast_api", selected)
    _write_output_rows(conn, "pod_monitoring_api", monitoring_values)
    _write_output_rows(conn, "pod_forecast_api", forecast_values)
    report = RebuildReport(
        len(monitoring_output),
        len(forecast_output),
        source_generation,
        resource_revision,
        resource_sha256,
        timeseries_revision,
        timeseries_sha256,
        tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.code,
                    item.pod_id_seq or -1,
                    item.report_year or -1,
                    item.year_frcast or -1,
                    item.project_id or "",
                    item.component or "",
                ),
            )
        ),
    )
    _persist_rebuild_diagnostics(conn, report, selected)
    return report


def refresh_pod_after_source_change(
    conn: sqlite3.Connection,
    *,
    actor: str,
    pod_ids: tuple[int, ...] | None = None,
) -> RebuildReport | None:
    """Advance one source generation and rebuild before the caller commits.

    Legacy databases have no v2 state row and remain on their existing writer
    path. On v2, the same connection and transaction own the source generation,
    dependency-basis invalidation, diagnostics, and physical output replacement.
    """
    if not conn.in_transaction:
        raise RuntimeError("POD source changes require an active caller transaction")
    if not _table_exists(conn, "pod_registry_state"):
        return None
    _require_v2_tables(conn)
    from esdc.pod_registry.migrations import _ensure_registry_writer

    _ensure_registry_writer(conn)
    if not actor.strip():
        raise ValueError("source-change actor cannot be empty")
    state = conn.execute(
        "SELECT source_generation, knowledge_generation "
        "FROM pod_registry_state WHERE singleton_id=1"
    ).fetchone()
    if state is None:
        raise RuntimeError("POD registry state row is missing")
    previous_generation = int(state[0])
    previous_knowledge_generation = (
        None if state[1] is None else int(state[1])
    )
    source_generation = previous_generation + 1
    conn.execute(
        "UPDATE pod_registry_state SET source_generation = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'), updated_by = ? "
        "WHERE singleton_id = 1",
        (source_generation, actor),
    )
    from esdc.knowledge.freshness import reconcile_knowledge_after_pod_change

    reconcile_knowledge_after_pod_change(
        conn,
        previous_generation=previous_generation,
        previous_knowledge_generation=previous_knowledge_generation,
        source_generation=source_generation,
        actor=actor,
    )
    return rebuild_pod_read_models(conn, pod_ids=pod_ids)
