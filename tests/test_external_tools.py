"""Tests for external tool categorization and LangChain conversion."""

from esdc.chat.external_tools import (
    INTERNAL_TOOL_NAMES,
    categorize_tools,
    convert_external_specs_to_langchain,
    is_external_tool_marker,
    parse_external_tool_name,
)


class TestCategorizeTools:
    def test_no_external_tools_none(self):
        internal, external_names, external_specs = categorize_tools(tools=None)
        assert len(external_names) == 0
        assert len(external_specs) == 0
        assert internal == INTERNAL_TOOL_NAMES

    def test_no_external_tools_empty_list(self):
        internal, external_names, external_specs = categorize_tools(tools=[])
        assert len(external_names) == 0
        assert len(external_specs) == 0
        assert internal == INTERNAL_TOOL_NAMES

    def test_mixed_internal_and_external(self):
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
                    "description": "Run a command",
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
                    "description": "Write a file",
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
        assert "run_command" not in internal
        assert "write_file" not in internal
        assert "run_command" in external_names
        assert "write_file" in external_names
        assert "execute_sql" not in external_names
        assert len(external_specs) == 2

    def test_all_external(self):
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
        ]
        internal, external_names, external_specs = categorize_tools(tools=tools)
        assert len(internal) == len(INTERNAL_TOOL_NAMES)
        assert "run_command" in external_names
        assert len(external_specs) == 1

    def test_internal_tool_not_in_external(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
        ]
        internal, external_names, external_specs = categorize_tools(tools=tools)
        assert "execute_sql" not in external_names
        assert len(external_specs) == 0


class TestConvertExternalSpecsToLangchain:
    def test_basic_conversion(self):
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

    def test_multiple_tools(self):
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
        ]
        tools = convert_external_specs_to_langchain(specs)
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "run_command" in names
        assert "write_file" in names

    def test_empty_specs(self):
        tools = convert_external_specs_to_langchain([])
        assert len(tools) == 0

    def test_optional_parameters(self):
        specs = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
        ]
        tools = convert_external_specs_to_langchain(specs)
        assert len(tools) == 1
        assert tools[0].name == "read_file"


class TestExternalToolMarker:
    def test_is_external_tool_marker(self):
        assert is_external_tool_marker("[EXTERNAL_TOOL_CALL:run_command]") is True
        assert is_external_tool_marker("[EXTERNAL_TOOL_CALL:write_file]") is True
        assert is_external_tool_marker("some normal sql result") is False
        assert is_external_tool_marker("") is False

    def test_parse_external_tool_name(self):
        assert (
            parse_external_tool_name("[EXTERNAL_TOOL_CALL:run_command]")
            == "run_command"
        )
        assert (
            parse_external_tool_name("[EXTERNAL_TOOL_CALL:write_file]") == "write_file"
        )
        assert parse_external_tool_name("some normal sql result") is None
        assert parse_external_tool_name("") is None

    def test_marker_format(self):
        for name in ["run_command", "write_file", "read_file"]:
            marker = f"[EXTERNAL_TOOL_CALL:{name}]"
            assert is_external_tool_marker(marker) is True
            assert parse_external_tool_name(marker) == name
