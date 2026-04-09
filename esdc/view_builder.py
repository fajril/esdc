import re
from dataclasses import dataclass, field

from esdc.selection import TableName


@dataclass
class ViewColumn:
    """A column in a view query with optional alias."""

    name: str
    alias: str | None = None

    def to_sql(self) -> str:
        """Return SQL representation, e.g. 'col' or 'col AS alias'."""
        if self.alias:
            return f"{self.name} AS '{self.alias}'"
        return self.name


@dataclass
class ViewDefinition:
    """Defines a view's structure for dynamic query building."""

    table_name: str
    default_where_column: str | None = None
    outputs: dict[int, list[ViewColumn]] = field(default_factory=dict)
    order_by: list[str] = field(default_factory=list)


VIEW_DEFINITIONS: dict[TableName, ViewDefinition] = {
    TableName.PROJECT_RESOURCES: ViewDefinition(
        table_name="project_resources",
        default_where_column="project_name",
        outputs={
            1: [
                ViewColumn("report_year"),
                ViewColumn("project_name"),
                ViewColumn("project_level"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
            ],
            2: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("field_name"),
                ViewColumn("project_name"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("project_level"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("prj_ioip"),
                ViewColumn("prj_igip"),
            ],
            3: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("field_name"),
                ViewColumn("project_name"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("project_level"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
                ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
                ViewColumn("prj_ioip"),
                ViewColumn("prj_igip"),
            ],
        },
        order_by=[
            "report_year",
            "project_stage",
            "project_class",
            "project_level",
            "wk_name",
            "field_name",
            "project_name",
            "uncert_level",
        ],
    ),
    TableName.FIELD_RESOURCES: ViewDefinition(
        table_name="field_resources",
        default_where_column="field_name",
        outputs={
            1: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("field_name"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
            ],
            2: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("field_name"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("project_level"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("is_discovered"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
            3: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("field_name"),
                ViewColumn("project_count"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
                ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
                ViewColumn("is_discovered"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
        },
        order_by=[
            "report_year",
            "wk_name",
            "field_name",
            "project_stage",
            "project_class",
            "project_level",
            "uncert_level",
        ],
    ),
    TableName.WA_RESOURCES: ViewDefinition(
        table_name="wa_resources",
        default_where_column="wk_name",
        outputs={
            1: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
            ],
            2: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("project_level"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
            3: [
                ViewColumn("report_year"),
                ViewColumn("wk_name"),
                ViewColumn("project_count"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
                ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
        },
        order_by=[
            "report_year",
            "wk_name",
            "project_stage",
            "project_class",
            "project_level",
            "uncert_level",
        ],
    ),
    TableName.NKRI_RESOURCES: ViewDefinition(
        table_name="nkri_resources",
        default_where_column=None,
        outputs={
            1: [
                ViewColumn("report_year"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
            ],
            2: [
                ViewColumn("report_year"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("is_discovered"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
            3: [
                ViewColumn("report_year"),
                ViewColumn("project_count"),
                ViewColumn("project_stage"),
                ViewColumn("project_class"),
                ViewColumn("uncert_level"),
                ViewColumn("rec_oc_risked", "resources_mstb"),
                ViewColumn("rec_an_risked", "resources_bscf"),
                ViewColumn("res_oc", "reserves_mstb"),
                ViewColumn("res_an", "reserves_bscf"),
                ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
                ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
                ViewColumn("is_discovered"),
                ViewColumn("ioip"),
                ViewColumn("igip"),
            ],
        },
        order_by=["report_year", "project_stage", "project_class", "uncert_level"],
    ),
}

_LIKE_SPECIAL_CHARS = re.compile(r"([%_])")


def _escape_like(value: str) -> str:
    escaped = _LIKE_SPECIAL_CHARS.sub(r"\\\1", value)
    return f"%{escaped}%"


def build_view_query(
    table: TableName,
    output: int = 0,
    where: str | None = None,
    like: str | None = None,
    year: int | None = None,
    columns: str | list[str] = "",
) -> tuple[str, list[str | int]]:
    """Build a parameterized SQL query for the specified view.

    Dynamically constructs SELECT, WHERE, and ORDER BY clauses
    based on the view definition and provided filters.

    Args:
        table: The table/view to query.
        output: Output detail level (0=ALL, 1-3=specific columns, 4+=ALL).
        where: Column name for WHERE LIKE clause (overrides default).
        like: Value for LIKE pattern matching (substring search).
        year: Year filter value.
        columns: Specific columns to select instead of default.

    Returns:
        Tuple of (parameterized_query, params_list).
    """
    view_def = VIEW_DEFINITIONS[table]
    output = min(output, 4)

    if columns:
        col_list = list(columns) if isinstance(columns, list) else [columns]
        select_clause = ", ".join(col_list)
    elif output == 0 or output >= 4 or output not in view_def.outputs:
        select_clause = "*"
    else:
        select_clause = ", ".join(col.to_sql() for col in view_def.outputs[output])

    query = f"SELECT {select_clause}\nFROM {view_def.table_name}"

    conditions: list[str] = []
    params: list[str | int] = []

    if like is not None and view_def.default_where_column is not None:
        where_col = where or view_def.default_where_column
        conditions.append(f"{where_col} LIKE ?")
        params.append(_escape_like(like))

    if year is not None:
        conditions.append("report_year = ?")
        params.append(year)

    if conditions:
        query += "\nWHERE " + " AND ".join(conditions)

    if view_def.order_by:
        query += "\nORDER BY " + ", ".join(view_def.order_by)

    return query, params
