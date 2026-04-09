import re
from pathlib import Path

from esdc.selection import TableName
from esdc.view_builder import build_view_query as _build_view_query

LIKE_SPECIAL_CHARS = re.compile(r"([%_])")


def _load_sql_script(script_file: str) -> str:
    file = Path(__file__).parent / "sql" / script_file
    with open(file, encoding="utf-8") as f:
        return f.read()


class SQLSanitizer:
    """Build parameterized queries for ESDC database views."""

    @staticmethod
    def sanitize_like(value: str) -> str:
        """Escape LIKE special characters and wrap with wildcards."""
        escaped = LIKE_SPECIAL_CHARS.sub(r"\\\1", value)
        return f"%{escaped}%"

    @classmethod
    def build_query(
        cls,
        table: TableName,
        where: str | None = None,
        like: str | None = None,
        years: list[int] | None = None,
        details: list[str] | None = None,
        columns: str | list[str] = "",
    ) -> tuple[str, list[str | int]]:
        """Build a parameterized query for the specified view."""
        return _build_view_query(
            table,
            details=details,
            where=where,
            like=like,
            years=years,
            columns=columns,
        )
