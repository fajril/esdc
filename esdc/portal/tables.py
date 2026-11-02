"""Current contract-driven portal grid configuration."""

from typing import Any

from esdc.pod_registry.contract import ColumnContract, TableContract, load_pod_contract


def build_grid(table: dict[str, Any]) -> dict[str, Any]:
    role = table.get("role", "")
    key_fields = table.get("key_fields", [])
    columns = table.get("columns", [])
    groups = list(dict.fromkeys(c.get("group") for c in columns if c.get("group")))

    return {
        "title": table.get("title", ""),
        "role": role,
        "readonly": role != "input",
        "pk": key_fields,
        "columns": [build_column(c, role) for c in columns],
        "groups": groups,
        "columnChooser": True,
    }


def build_column(col: dict[str, Any], role: str) -> dict[str, Any]:
    generated = col.get("generated", False)
    new_only = col.get("new_only", False)
    nullable = col.get("nullable", True)
    ownership = col.get("ownership", "")
    kind = col.get("kind", "")
    storage_type = col.get("storage_type", "")

    editable = (
        role == "input" and ownership == "input" and not generated and not new_only
    )
    editable_on_new = role == "input" and ownership == "input" and not generated
    readonly = not editable and not editable_on_new
    required = not nullable and editable_on_new
    editor = pick_editor(storage_type, kind)

    return {
        "field": col.get("name", ""),
        "title": col.get("title", ""),
        "logical_type": kind,
        "data_type": storage_type,
        "nullable": nullable,
        "unit": col.get("unit"),
        "group": col.get("group"),
        "description": col.get("description"),
        "readonly": readonly,
        "editable": editable,
        "editableOnNew": editable_on_new,
        "required": required,
        "editor": editor,
        "ref": col.get("ref"),
    }


def pick_editor(storage_type: str, kind: str) -> str:
    if kind in ("decimal", "integer"):
        return "text"
    if kind in ("uuid", "text", "datetime"):
        return "text"
    if kind == "date":
        return "date"
    if kind == "boolean":
        return "tickCross"
    return "text"


_NEW_ONLY = {
    "m_project": {"project_id"},
    "m_pod": {"approval_date", "institution_id", "pod_type_id", "rev_num"},
}
_GRAINS = {
    "pod_monitoring_api": ["pod_id_seq", "reported_date"],
    "pod_forecast_api": ["pod_id_seq", "reported_date", "year_frcast"],
}
_REFS = {"m_pod": "pods", "m_project": "projects"}


def _contract_column(column: ColumnContract, table: TableContract) -> dict[str, Any]:
    reference = next((r for r in table.references if r.column == column.name), None)
    return {
        "name": column.name,
        "title": column.formal_name,
        "kind": column.logical_type,
        "storage_type": column.data_type,
        "nullable": column.nullable,
        "unit": column.unit,
        "group": column.group,
        "description": column.description,
        "ownership": column.ownership,
        "generated": column.ownership != "input",
        "new_only": column.name in _NEW_ONLY.get(table.name, set()),
        "ref": None
        if reference is None
        else _REFS.get(reference.target_table, reference.target_table),
    }


def _static_column(
    name: str, kind: str = "text", *, editable: bool = False
) -> dict[str, Any]:
    return {
        "name": name,
        "title": name.replace("_", " ").title(),
        "kind": kind,
        "storage_type": "BIGINT" if kind == "integer" else "TEXT",
        "nullable": True,
        "ownership": "input" if editable else "derived",
        "generated": not editable,
    }


def _build_configs() -> dict[str, dict]:
    configs = {}
    for table in load_pod_contract().tables.values():
        configs[table.name] = build_grid(
            {
                "title": table.name.replace("_", " ").title(),
                "role": table.role,
                "columns": [
                    _contract_column(column, table) for column in table.columns
                ],
                "key_fields": _GRAINS.get(table.name, list(table.primary_key)),
            }
        )
    for name, labels in {
        "r_institution": ["institution", "description"],
        "r_pod_type": ["pod_type", "description"],
        "r_pod_stage": ["stage_name", "stage_definition"],
    }.items():
        columns = [_static_column("id", "integer")] + [
            _static_column(c)
            for c in [*labels, "created_at", "created_by", "updated_at", "updated_by"]
        ]
        for column in columns:
            column["nullable"] = column["name"] == "description"
        configs[name] = build_grid(
            {
                "title": name,
                "role": "reference",
                "columns": columns,
                "key_fields": ["id"],
            }
        )
    entities = {"wk_name", "field_name", "project_name"}
    columns = [
        _static_column(name, editable=name in entities)
        for name in (
            "doc_id",
            "file_name",
            "doc_type",
            "doc_date",
            "subject",
            "wk_name",
            "field_name",
            "project_name",
            "pod_name",
            "suggested_pod_ids",
        )
    ]
    configs["documents"] = build_grid(
        {
            "title": "Documents",
            "role": "input",
            "columns": columns,
            "key_fields": ["doc_id"],
        }
    )
    for column in configs["documents"]["columns"]:
        if column["field"] in entities:
            column["entityList"] = True
    columns = [
        _static_column("pod_id_seq", "integer", editable=True),
        _static_column("doc_id", editable=True),
    ]
    for column in columns:
        column.update(
            nullable=False,
            new_only=True,
            ref="pods" if column["name"] == "pod_id_seq" else "documents",
        )
    configs["pod_document"] = build_grid(
        {
            "title": "Document Links",
            "role": "input",
            "columns": columns,
            "key_fields": ["pod_id_seq", "doc_id"],
        }
    )
    return configs


TABLE_CONFIGS: dict[str, dict] = _build_configs()
