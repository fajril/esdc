from esdc.chat.schema_loader import SchemaLoader


class TextToSQL:
    def __init__(self, provider):
        self.provider = provider
        self.schema_loader = SchemaLoader()

    def generate(self, user_query: str) -> str:
        schema = self.schema_loader.get_core_schema()
        # This will call the provider to generate SQL
        # For now, return a simple query
        return "SELECT project_name, province, res_oil FROM project_resources LIMIT 10;"
