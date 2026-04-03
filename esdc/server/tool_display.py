"""Tool display configuration for user-friendly tool call formatting."""

# Standard library
import json
import logging

# Tool name mappings with user-friendly display info
TOOL_DISPLAY_INFO = {
    "execute_sql": {
        "emoji": "📝",
        "title": "Query Database",
        "description": "Menjalankan query SQL untuk mengambil data dari database",
        "language": "sql",
    },
    "list_tables": {
        "emoji": "📊",
        "title": "Lihat Daftar Tabel",
        "description": "Melihat daftar tabel yang tersedia dalam database",
        "language": "json",
    },
    "get_recommended_table": {
        "emoji": "🔍",
        "title": "Cari Tabel Rekomendasi",
        "description": "Mencari tabel yang sesuai dengan kebutuhan analisis",
        "language": "json",
    },
    "get_table_schema": {
        "emoji": "📋",
        "title": "Lihat Struktur Tabel",
        "description": "Mendapatkan informasi kolom dan tipe data tabel",
        "language": "json",
    },
    "search_problem_cluster": {
        "emoji": "🔎",
        "title": "Cari Cluster Masalah",
        "description": "Mencari cluster atau kelompok masalah dalam data",
        "language": "json",
    },
}


def get_tool_display_info(tool_name: str) -> dict:
    """Get display information for a tool.

    Args:
        tool_name: The internal tool name

    Returns:
        Dictionary with emoji, title, description, and language
    """
    return TOOL_DISPLAY_INFO.get(
        tool_name,
        {
            "emoji": "🔧",
            "title": tool_name.replace("_", " ").title(),
            "description": f"Mengeksekusi operasi {tool_name}",
            "language": "json",
        },
    )


logger = logging.getLogger("esdc.server.tool_display")


def generate_dynamic_description(tool_name: str, tool_args: dict) -> str:
    """Generate dynamic description based on tool and args.

    Args:
        tool_name: The internal tool name
        tool_args: The tool arguments

    Returns:
        Dynamic description string
    """
    display_info = get_tool_display_info(tool_name)

    # Custom logic for specific tools
    if tool_name == "execute_sql":
        query = tool_args.get("query", "").strip()

        # Log full SQL query to file
        logger.info(f"SQL Query executed: {query}")

        # Extract table name from query
        table_keywords = ["FROM", "JOIN", "INTO"]
        tables = []
        for keyword in table_keywords:
            if keyword in query.upper():
                # Simple extraction - could be improved
                parts = query.upper().split(keyword)
                if len(parts) > 1:
                    table_part = parts[1].split()[0].strip(";,()")
                    if table_part and table_part not in ["SELECT", "WHERE", "AND"]:
                        tables.append(table_part)

        if tables:
            return f"Mengambil data dari tabel {', '.join(tables)}"
        else:
            return display_info["description"]

    elif tool_name == "get_recommended_table":
        entity_type = tool_args.get("entity_type", "")
        if entity_type:
            return f"Mencari tabel rekomendasi untuk entitas {entity_type}"
        return display_info["description"]

    elif tool_name == "search_problem_cluster":
        keywords = tool_args.get("keywords", [])
        if keywords:
            if isinstance(keywords, list):
                return f"Mencari masalah terkait {', '.join(keywords)}"
            else:
                return f"Mencari masalah terkait {keywords}"
        return display_info["description"]

    # Log tool args for other tools
    if tool_args:
        logger.info(f"Tool {tool_name} called with args: {tool_args}")

    return display_info["description"]


def format_tool_args(tool_name: str, tool_args: dict) -> str:
    """Format tool arguments as plain text for collapsible display.

    Args:
        tool_name: The internal tool name
        tool_args: The tool arguments

    Returns:
        Plain text formatted arguments
    """
    if tool_name == "execute_sql":
        sql = tool_args.get("query", "")
        return f"Query:\n{sql.strip()}"
    else:
        # Format as simple key-value pairs without JSON markers
        lines = []
        for key, value in tool_args.items():
            if isinstance(value, str):
                lines.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)):
                lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) if lines else "Tidak ada parameter"
