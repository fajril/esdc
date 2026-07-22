"""Canonical entity registry for ESDC entity recognition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntitySpec:
    """Database-backed entity definition used by the resolver."""

    key: str
    label: str
    level: str
    lookup_table: str
    default_table: str
    timeseries_table: str | None
    name_column: str
    id_column: str | None
    hints: tuple[str, ...]
    priority: int


ENTITY_REGISTRY: dict[str, EntitySpec] = {
    "project_name": EntitySpec(
        key="project_name",
        label="Project",
        level="project",
        lookup_table="project_resources",
        default_table="project_resources",
        timeseries_table="project_timeseries",
        name_column="project_name",
        id_column="project_id",
        hints=("proyek", "project"),
        priority=30,
    ),
    "field_name": EntitySpec(
        key="field_name",
        label="Field",
        level="field",
        lookup_table="project_resources",
        default_table="field_resources",
        timeseries_table="field_timeseries",
        name_column="field_name",
        id_column="field_id",
        hints=("lapangan", "field"),
        priority=20,
    ),
    "wk_name": EntitySpec(
        key="wk_name",
        label="WorkingArea",
        level="work_area",
        lookup_table="project_resources",
        default_table="wa_resources",
        timeseries_table="wa_timeseries",
        name_column="wk_name",
        id_column="wk_id",
        hints=("wk", "wilayah kerja", "working area"),
        priority=10,
    ),
    "operator_name": EntitySpec(
        key="operator_name",
        label="Operator",
        level="project",
        lookup_table="project_resources",
        default_table="project_resources",
        timeseries_table="project_timeseries",
        name_column="operator_name",
        id_column=None,
        hints=("operator", "perusahaan", "oleh"),
        priority=40,
    ),
}

ENTITY_LABEL_TO_KEY = {spec.label: key for key, spec in ENTITY_REGISTRY.items()}
