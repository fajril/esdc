"""Integration tests for hybrid tool passthrough flow.

Tests the complete flow of external tool categorization, LangChain
conversion, and event detection.
"""

from esdc.chat.external_tools import (
    INTERNAL_TOOL_NAMES,
    categorize_tools,
    convert_external_specs_to_langchain,
    is_external_tool_marker,
    parse_external_tool_name,
)


class TestHybridToolFlow:
    """Test the complete hybrid tool execution flow."""

    def test_categorize_mixed_tools(self):
        """Internal and external tools are correctly separated."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        ]
        internal, external_names, external_specs = categorize_tools(tools=tools)
        assert "execute_sql" in internal
        assert "run_command" in external_names
        assert "write_file" in external_names
        assert "execute_sql" not in external_names
        assert len(external_specs) == 2

    def test_convert_external_specs_to_langchain(self):
        """External tool specs can be converted to LangChain tool format."""
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to run",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
        ]
        tools = convert_external_specs_to_langchain(specs)
        assert len(tools) == 1
        assert tools[0].name == "run_command"
        assert tools[0].description == "Run a shell command"

    def test_no_external_tools(self):
        """When no tools are provided, only internal tools are available."""
        internal, external_names, external_specs = categorize_tools(tools=None)
        assert len(external_names) == 0
        assert len(external_specs) == 0
        assert "execute_sql" in internal

    def test_all_internal_tools_present(self):
        """All ESDC internal tools are always in the internal set."""
        internal, _, _ = categorize_tools(tools=None)
        for tool_name in INTERNAL_TOOL_NAMES:
            assert tool_name in internal, f"{tool_name} missing from internal tools"

    def test_external_tool_marker_detection(self):
        """External tool markers are correctly detected and parsed."""
        assert is_external_tool_marker("[EXTERNAL_TOOL_CALL:run_command]")
        assert is_external_tool_marker("[EXTERNAL_TOOL_CALL:write_file]")
        assert not is_external_tool_marker("some normal sql result")
        assert not is_external_tool_marker("")
        assert not is_external_tool_marker("[INTERNAL_TOOL_RESULT]")

        assert (
            parse_external_tool_name("[EXTERNAL_TOOL_CALL:run_command]")
            == "run_command"
        )
        assert (
            parse_external_tool_name("[EXTERNAL_TOOL_CALL:write_file]") == "write_file"
        )
        assert parse_external_tool_name("some normal sql result") is None

    def test_convert_multiple_external_specs(self):
        """Multiple external specs are all converted."""
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ]
        tools = convert_external_specs_to_langchain(specs)
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "run_command" in names
        assert "write_file" in names
        assert "read_file" in names

    def test_categorize_with_only_openterminal_tools(self):
        """Only OpenTerminal tools, no ESDC tools in request."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
        ]
        internal, external_names, external_specs = categorize_tools(tools=tools)
        assert len(internal) == len(INTERNAL_TOOL_NAMES)
        assert "run_command" in external_names
        assert "read_file" in external_names
        assert len(external_specs) == 2
