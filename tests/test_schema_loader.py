def test_load_core_schema():
    from esdc.chat.schema_loader import SchemaLoader

    loader = SchemaLoader()
    schema = loader.get_core_schema()

    assert "project_resources" in schema
    assert "province" in schema
