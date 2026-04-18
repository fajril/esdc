def test_get_schema_for_prompt():
    """Test that schema for prompt generates valid output."""
    from esdc.chat.domain_knowledge.schema_definitions import get_schema_for_prompt

    schema = get_schema_for_prompt()

    assert "project_resources" in schema
    assert "province" in schema
    assert "project_timeseries" in schema
    assert "field_resources" in schema
    assert "wa_resources" in schema
    assert "nkri_resources" in schema
    assert "rec_oc" in schema
    assert "tpf_oc" in schema


def test_database_schema_structure():
    """Test DATABASE_SCHEMA has expected structure."""
    from esdc.chat.domain_knowledge.schema_definitions import DATABASE_SCHEMA

    assert "project_resources" in DATABASE_SCHEMA
    assert "project_timeseries" in DATABASE_SCHEMA
    assert "field_resources" in DATABASE_SCHEMA
    assert "wa_resources" in DATABASE_SCHEMA
    assert "nkri_resources" in DATABASE_SCHEMA

    pr = DATABASE_SCHEMA["project_resources"]
    assert "columns" in pr
    assert "description" in pr
    assert "primary_key" in pr
    assert "rec_oc" in pr["columns"]
    assert "tpf_oc" in DATABASE_SCHEMA["project_timeseries"]["columns"]


def test_system_prompt_has_no_inline_schema():
    """System prompt should NOT contain inline schema.

    Schema is available via get_schema tool.
    """
    from esdc.chat.prompts import get_system_prompt

    prompt = get_system_prompt()

    assert "res_oc" in prompt
    assert "rec_an" in prompt
    assert "field_resources" in prompt

    assert "project_id | TEXT" not in prompt
    assert "field_lat | REAL" not in prompt
    assert "rec_oil | REAL" not in prompt

    assert len(prompt) < 22000, (
        f"System prompt is {len(prompt)} chars, expected < 22000"
    )
