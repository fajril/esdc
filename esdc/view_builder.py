import re
from dataclasses import dataclass, field
from enum import Enum

from esdc.selection import TableName


class DetailLevel(str, Enum):
    RESERVES = "reserves"
    RESOURCES = "resources"
    RESOURCES_RISKED = "resources_risked"
    INPLACE = "inplace"
    CUMPROD = "cumprod"
    RATE = "rate"
    ALL = "all"


VALID_DETAILS = {d.value for d in DetailLevel}


@dataclass
class ViewColumn:
    name: str
    alias: str | None = None

    def to_sql(self) -> str:
        if self.alias:
            return f"{self.name} AS '{self.alias}'"
        return self.name


@dataclass
class ViewDefinition:
    table_name: str
    default_where_column: str | None = None
    mandatory_columns: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)


VIEW_DEFINITIONS: dict[TableName, ViewDefinition] = {
    TableName.PROJECT_RESOURCES: ViewDefinition(
        table_name="project_resources",
        default_where_column="project_name",
        mandatory_columns=[
            "report_year",
            "project_name",
            "field_name",
            "wk_name",
            "project_level",
            "uncert_level",
        ],
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
        mandatory_columns=[
            "report_year",
            "field_name",
            "wk_name",
            "project_stage",
            "project_class",
            "uncert_level",
        ],
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
        mandatory_columns=[
            "report_year",
            "wk_name",
            "project_stage",
            "project_class",
            "uncert_level",
        ],
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
        mandatory_columns=[
            "report_year",
            "project_stage",
            "project_class",
            "uncert_level",
        ],
        order_by=["report_year", "project_stage", "project_class", "uncert_level"],
    ),
}

DETAIL_COLUMNS: dict[str, dict[TableName, list[ViewColumn]]] = {
    "reserves": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
    },
    "resources": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
    },
    "resources_risked": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("rec_oc_risked", "resources_mstb"),
            ViewColumn("rec_an_risked", "resources_bscf"),
            ViewColumn("res_oc", "reserves_mstb"),
            ViewColumn("res_an", "reserves_bscf"),
        ],
    },
    "inplace": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("prj_ioip", "ioip"),
            ViewColumn("prj_igip", "igip"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
    },
    "cumprod": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
            ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
            ViewColumn("cprd_grs_oc", "gross_cumprod_mstb"),
            ViewColumn("cprd_grs_an", "gross_cumprod_bscf"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
            ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
            ViewColumn("is_discovered"),
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
            ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("cprd_sls_oc", "sales_cumprod_mstb"),
            ViewColumn("cprd_sls_an", "sales_cumprod_bscf"),
            ViewColumn("is_discovered"),
            ViewColumn("ioip", "ioip"),
            ViewColumn("igip", "igip"),
        ],
    },
    "rate": {
        TableName.PROJECT_RESOURCES: [
            ViewColumn("rate_sls_oc", "sales_rate_mstb"),
            ViewColumn("rate_sls_an", "sales_rate_bscf"),
            ViewColumn("rate_grs_oc", "gross_rate_mstb"),
            ViewColumn("rate_grs_an", "gross_rate_bscf"),
        ],
        TableName.FIELD_RESOURCES: [
            ViewColumn("rate_sls_oc", "sales_rate_mstb"),
            ViewColumn("rate_sls_an", "sales_rate_bscf"),
            ViewColumn("rate_grs_oc", "gross_rate_mstb"),
            ViewColumn("rate_grs_an", "gross_rate_bscf"),
        ],
        TableName.WA_RESOURCES: [
            ViewColumn("rate_sls_oc", "sales_rate_mstb"),
            ViewColumn("rate_sls_an", "sales_rate_bscf"),
            ViewColumn("rate_grs_oc", "gross_rate_mstb"),
            ViewColumn("rate_grs_an", "gross_rate_bscf"),
        ],
        TableName.NKRI_RESOURCES: [
            ViewColumn("rate_sls_oc", "sales_rate_mstb"),
            ViewColumn("rate_sls_an", "sales_rate_bscf"),
            ViewColumn("rate_grs_oc", "gross_rate_mstb"),
            ViewColumn("rate_grs_an", "gross_rate_bscf"),
        ],
    },
}

_LIKE_SPECIAL_CHARS = re.compile(r"([%_])")


def _escape_like(value: str) -> str:
    escaped = _LIKE_SPECIAL_CHARS.sub(r"\\\1", value)
    return f"%{escaped}%"


def build_view_query(
    table: TableName,
    details: list[str] | None = None,
    where: str | None = None,
    like: str | None = None,
    years: list[int] | None = None,
    columns: str | list[str] = "",
) -> tuple[str, list[str | int]]:
    """Build a parameterized SQL query for the specified view.

    Args:
        table: The table/view to query.
        details: List of detail levels (reserves, resources, resources_risked,
            inplace, cumprod, rate, all). Defaults to ["resources"].
        where: Column name for WHERE ILIKE clause (overrides default).
        like: Value for ILIKE pattern matching (case-insensitive substring).
        years: List of year filter values.
        columns: Specific columns to select instead of default.

    Returns:
        Tuple of (parameterized_query, params_list).

    Raises:
        ValueError: If an invalid detail name is provided.
    """
    view_def = VIEW_DEFINITIONS[table]

    if details is None:
        details = ["resources"]

    for detail in details:
        if detail not in VALID_DETAILS:
            raise ValueError(
                f"Invalid detail '{detail}'. "
                f"Valid options: {', '.join(sorted(VALID_DETAILS))}"
            )

    if columns:
        col_list = list(columns) if isinstance(columns, list) else [columns]
        select_clause = ", ".join(col_list)
    elif "all" in details:
        select_clause = "*"
    else:
        seen: set[str] = set()
        col_parts: list[str] = []

        for col in view_def.mandatory_columns:
            if col not in seen:
                col_parts.append(col)
                seen.add(col)

        for detail in details:
            if detail == "all":
                continue
            for vc in DETAIL_COLUMNS[detail][table]:
                if vc.name not in seen:
                    col_parts.append(vc.to_sql())
                    seen.add(vc.name)

        select_clause = ", ".join(col_parts)

    query = f"SELECT {select_clause}\nFROM {view_def.table_name}"

    conditions: list[str] = []
    params: list[str | int] = []

    if like is not None and view_def.default_where_column is not None:
        where_col = where or view_def.default_where_column
        conditions.append(f"{where_col} ILIKE ?")
        params.append(_escape_like(like))

    if years:
        if len(years) == 1:
            conditions.append("report_year = ?")
            params.append(years[0])
        else:
            placeholders = ", ".join("?" for _ in years)
            conditions.append(f"report_year IN ({placeholders})")
            params.extend(years)

    if conditions:
        query += "\nWHERE " + " AND ".join(conditions)

    if view_def.order_by:
        query += "\nORDER BY " + ", ".join(view_def.order_by)

    return query, params
