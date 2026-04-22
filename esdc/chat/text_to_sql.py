class TextToSQL:
    """Simple text-to-SQL converter stub."""

    def __init__(self, provider):
        """Initialize with an LLM provider."""
        self.provider = provider

    def generate(self, user_query: str) -> str:
        """Generate a SQL query from natural language input."""
        return "SELECT project_name, province, res_oil FROM project_resources LIMIT 10;"
