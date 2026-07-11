from pathlib import Path

from esdc.chat.domain_knowledge import doc_schema
from esdc.corpus.store import CorpusStore


class FakeEmbedder:
    """Deterministic 3-dim embedder, same shape as tests/corpus/test_store.py."""

    model = "fake-embed"

    def generate_embedding(self, text: str) -> list[float]:
        return [1.0, 1.0, 1.0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(t) for t in texts]


def test_llm_field_names_returns_eleven_fields_in_order():
    names = doc_schema.llm_field_names()
    assert names == (
        "doc_type",
        "doc_number",
        "doc_date",
        "subject",
        "sender",
        "recipient",
        "doc_level",
        "wk_name",
        "field_name",
        "project_name",
        "extras",
    )


def test_enum_values_doc_type():
    assert doc_schema.enum_values("doc_type") == ("surat", "mom", "ba", "other")


def test_enum_values_doc_level():
    assert doc_schema.enum_values("doc_level") == ("wk", "field", "project", "unknown")


def test_render_prompt_definitions_starts_with_doc_type():
    assert doc_schema.render_prompt_definitions().startswith("- doc_type:")


def test_render_tool_context_mentions_sender_and_organization():
    ctx = doc_schema.render_tool_context()
    assert "sender" in ctx
    assert "organization" in ctx.lower()

    assert "doc_level" in ctx


def test_schema_field_names_are_columns_of_documents_table(tmp_path: Path):
    """Sync guard: every doc_schema field must map to a real `documents` column.

    Catches drift between doc_schema.yaml and the CorpusStore table schema.
    `extras` -> `metadata` and `source_file` -> `file_name` are known,
    intentional renames (see esdc/corpus/pipeline.py's `doc = {...}` dict
    build in run_commit) so they're allowed via an explicit mapping rather
    than treated as schema/table drift.
    """
    store = CorpusStore(db_path=tmp_path / "sync.duckdb", embedder=FakeEmbedder())
    store.ensure_tables()
    try:
        columns = {
            row[0]
            for row in store._get_connection()
            .execute(f"DESCRIBE {store.DOC_TABLE}")
            .fetchall()
        }
    finally:
        store.close()

    explicit_mapping = {"extras": "metadata", "source_file": "file_name"}

    schema = doc_schema.load_doc_schema()
    all_field_names = doc_schema.llm_field_names() + doc_schema.housekeeping_field_names()
    assert set(all_field_names) == {
        f["name"] for f in schema["llm_fields"] + schema["housekeeping_fields"]
    }

    for name in all_field_names:
        mapped = explicit_mapping.get(name, name)
        assert mapped in columns, (
            f"doc_schema field {name!r} (-> {mapped!r}) is not a column of "
            f"the documents table; columns={sorted(columns)}"
        )
