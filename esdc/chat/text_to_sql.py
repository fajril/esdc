class TextToSQL:
    def __init__(self, provider):
        self.provider = provider

    def generate(self, user_query: str) -> str:
        return "SELECT project_name, province, res_oil FROM project_resources LIMIT 10;"
