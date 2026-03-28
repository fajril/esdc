from pathlib import Path


class SchemaLoader:
    SCHEMA_FILE = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "reference"
        / "schema"
        / "esdc-database-schema.md"
    )

    CORE_COLUMNS = [
        "project_name",
        "field_name",
        "wk_name",
        "operator_name",
        "province",
        "basin86",
        "basin128",
        "project_stage",
        "project_class",
        "project_level",
        "uncert_level",
        "rec_oil",
        "rec_con",
        "rec_ga",
        "rec_gn",
        "res_oil",
        "res_con",
        "res_ga",
        "res_gn",
        "prj_ioip",
        "prj_igip",
        "report_year",
        "is_offshore",
        "is_discovered",
    ]

    def get_core_schema(self) -> str:
        return """project_resources table:
- project_name TEXT: Name of the project
- field_name TEXT: Name of the field
- wk_name TEXT: Work area name
- operator_name TEXT: Operator company name
- province TEXT: Province name
- basin86 TEXT: Basin name (86 classification)
- basin128 TEXT: Basin name (128 classification)
- project_stage TEXT: Project stage (Exploration, Exploitation)
- project_class TEXT: Project classification (Reserves & GRR, Contingent Resources, Prospective Resources)
- project_level TEXT: Project maturity level
- uncert_level TEXT: Uncertainty level (Low, Mid, High)
- rec_oil REAL: Resources oil (MSTB)
- rec_con REAL: Resources condensate (MSTB)
- rec_ga REAL: Resources associated gas (BSCF)
- rec_gn REAL: Resources non-associated gas (BSCF)
- res_oil REAL: Reserves oil (MSTB)
- res_con REAL: Reserves condensate (MSTB)
- res_ga REAL: Reserves associated gas (BSCF)
- res_gn REAL: Reserves non-associated gas (BSCF)
- prj_ioip REAL: Original oil in place
- prj_igip REAL: Original gas in place
- report_year INTEGER: Report year
- is_offshore INTEGER: Whether offshore (0/1)
- is_discovered INTEGER: Whether discovered (0/1)
"""
