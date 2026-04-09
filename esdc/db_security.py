import re
from pathlib import Path

from esdc.selection import TableName

LIKE_SPECIAL_CHARS = re.compile(r"([%_])")


def _load_sql_script(script_file: str) -> str:
    """Load a SQL script from the esdc/sql/ directory."""
    file = Path(__file__).parent / "sql" / script_file
    with open(file, encoding="utf-8") as f:
        return f.read()


class SQLSanitizer:
    """Build parameterized queries for ESDC database views.

    Replaces <where>, <like>, <year> placeholders in SQL view scripts
    with ? parameter markers to prevent SQL injection.
    """

    _WHERE_COLUMNS: dict[TableName, str] = {
        TableName.PROJECT_RESOURCES: "project_name",
        TableName.FIELD_RESOURCES: "field_name",
        TableName.WA_RESOURCES: "wk_name",
    }

    _SQL_SCRIPT_MAP: dict[TableName, str] = {
        TableName.PROJECT_RESOURCES: "view_project_resources.sql",
        TableName.FIELD_RESOURCES: "view_field_resources.sql",
        TableName.WA_RESOURCES: "view_wa_resources.sql",
        TableName.NKRI_RESOURCES: "view_nkri_resources.sql",
    }

    @classmethod
    def sanitize_like(cls, value: str) -> str:
        """Escape LIKE special characters and wrap with wildcards.

        Escapes % and _ characters in the input value, then wraps
        the result with % wildcards for LIKE matching.

        Args:
            value: The raw search string to sanitize.

        Returns:
            Sanitized string like '%value%' with special chars escaped.
        """
        escaped = LIKE_SPECIAL_CHARS.sub(r"\\\1", value)
        return f"%{escaped}%"

    @classmethod
    def build_query(
        cls,
        table: TableName,
        where: str | None = None,
        like: str | None = None,
        year: int | None = None,
        output: int = 0,
        columns: str | list[str] = "",
    ) -> tuple[str, list[str | int]]:
        """Build a parameterized query from a SQL view script.

        Args:
            table: The table/view to query.
            where: Column name for WHERE clause (NKRI ignores this).
            like: Value for LIKE pattern matching.
            year: Year filter value.
            output: Output format index (selects which query in the script).
            columns: Specific columns to select instead of default.

        Returns:
            Tuple of (parameterized_query, params_list).
        """
        output = min(output, 4)
        script = _load_sql_script(cls._SQL_SCRIPT_MAP[table])
        queries = script.split(";")
        query = queries[max(0, output - 1)].strip()

        if columns:
            pattern = r".*?(?=FROM)"
            query_match = re.search(pattern, query, re.IGNORECASE)
            if query_match is None:
                raise ValueError(f"Could not parse query: {query}")
            col_list = list(columns) if isinstance(columns, list) else [columns]
            select_clause = "SELECT " + ", ".join(col_list) + " "
            query = select_clause + query[query_match.end() :]

        params: list[str | int] = []

        if table == TableName.NKRI_RESOURCES:
            query, params = cls._apply_nkri_filters(query, year)
        else:
            query, params = cls._apply_resource_filters(query, table, where, like, year)

        return query, params

    @classmethod
    def _apply_nkri_filters(
        cls, query: str, year: int | None
    ) -> tuple[str, list[str | int]]:
        """Apply filters for NKRI resources (year only, no LIKE)."""
        params: list[str | int] = []
        if year is None:
            query = re.sub(r"\s*WHERE\s+report_year\s*=\s*<year>", "", query)
        else:
            query = query.replace("<year>", "?")
            params.append(year)
        return query, params

    @classmethod
    def _apply_resource_filters(
        cls,
        query: str,
        table: TableName,
        where: str | None,
        like: str | None,
        year: int | None,
    ) -> tuple[str, list[str | int]]:
        """Apply filters for project/field/WA resources (WHERE LIKE + year)."""
        params: list[str | int] = []

        where_column = cls._WHERE_COLUMNS.get(table, "wk_name")
        if where is not None:
            where_column = where

        if like is not None:
            sanitized_like = cls.sanitize_like(like)
            if year is not None:
                query = query.replace(
                    "WHERE <where> LIKE '%<like>%' AND report_year = <year>",
                    f"WHERE {where_column} LIKE ? AND report_year = ?",
                )
                params.extend([sanitized_like, year])
            else:
                like_year_pat = (
                    r"WHERE\s+<where>\s+LIKE\s+"
                    r"'%<like>%'\s+AND\s+report_year\s*=\s*<year>"
                )
                query = re.sub(
                    like_year_pat,
                    f"WHERE {where_column} LIKE ?",
                    query,
                    flags=re.IGNORECASE,
                )
                params.append(sanitized_like)
        else:
            if year is not None:
                year_pat = (
                    r"WHERE\s+<where>\s+LIKE\s+"
                    r"'%<like>%'\s+AND\s+(report_year\s*=\s*)<year>"
                )
                query = re.sub(
                    year_pat,
                    rf"WHERE {where_column} AND \1?",
                    query,
                    flags=re.IGNORECASE,
                )
                params.append(year)
            else:
                no_filter_pat = (
                    r"WHERE\s+<where>\s+LIKE\s+"
                    r"'%<like>%'\s+AND\s+report_year\s*=\s*<year>"
                )
                query = re.sub(no_filter_pat, "", query, flags=re.IGNORECASE)

        return query, params
