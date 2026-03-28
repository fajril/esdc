def test_text_to_sql_generation():
    from esdc.chat.text_to_sql import TextToSQL

    engine = TextToSQL(provider=None)  # Mock provider
    sql = engine.generate("Show oil reserves in East Java")

    assert "SELECT" in sql.upper()
    assert "project_resources" in sql.lower()
