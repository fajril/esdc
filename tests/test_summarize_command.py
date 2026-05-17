import json

import duckdb
from langchain_core.messages import AIMessage

from esdc.chat.token_counter import TokenUsage
from esdc.configs import Config
from esdc.db_security import _load_sql_script
from esdc.esdc import app
from esdc.summarizer import (
    SummaryEntityResult,
    _commit_live_tokens,
    _preview_live_tokens,
    _summarize_entity,
    ensure_summary_table,
)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "headline": "Peluang produksi dan cadangan perlu diprioritaskan",
                "executive_summary": (
                    "Terdapat kendala subsurface dan fasilitas, namun remarks "
                    "menunjukkan peluang optimasi produksi dan penambahan cadangan."
                ),
                "current_situation": "Beberapa proyek masih memerlukan tindak lanjut.",
                "key_challenges": ["Kendala fasilitas", "Ketidakpastian subsurface"],
                "solution_proposals": ["Lanjutkan workover dan evaluasi reservoir"],
                "production_or_reserve_opportunities": [
                    "Optimasi produksi dan maturation resources ke reserves"
                ],
                "management_attention": ["Perlu prioritas keputusan eksekusi"],
                "ksmi_context": {
                    "resource_classes": ["Contingent Resources"],
                    "project_levels": ["E2"],
                    "constraint_types": ["technical"],
                    "mentions_groovy": False,
                    "mentions_pod_or_pse": False,
                },
                "source_coverage": {
                    "source_items_reviewed": 1,
                    "source_items_with_material_issues": 1,
                },
            }
        )


class MetadataLLM(FakeLLM):
    last_provider_name = "deepseek"
    last_model_name = "deepseek-v4-flash"


class UsageMetadataLLM(FakeLLM):
    def invoke(self, prompt):
        content = super().invoke(prompt)
        message = AIMessage(content=content)
        message.usage_metadata = {  # type: ignore[attr-defined]
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        }
        return message


class ZeroUsageMetadataLLM(FakeLLM):
    """Simulates vLLM-style provider that returns all-zero usage."""

    def invoke(self, prompt):
        content = super().invoke(prompt)
        message = AIMessage(content=content)
        message.usage_metadata = {  # type: ignore[attr-defined]
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        return message


def _patch_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(
            lambda cls: {
                "name": "test-provider",
                "provider_type": "openai",
                "model": "test-model",
            }
        ),
    )
    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda config: llm)
    return llm


def _create_minimal_project_resources():
    Config.init_config()
    Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            CREATE TABLE project_resources (
                report_year INTEGER,
                project_id TEXT,
                project_name TEXT,
                field_id TEXT,
                field_name TEXT,
                wk_id TEXT,
                wk_name TEXT,
                project_class TEXT,
                project_stage TEXT,
                project_level TEXT,
                uncert_level TEXT,
                project_remarks TEXT,
                res_oc REAL,
                res_an REAL,
                rec_oc REAL,
                rec_an REAL,
                rec_oc_risked REAL,
                rec_an_risked REAL,
                prj_ioip REAL,
                prj_igip REAL,
                rate_sls_oc REAL,
                rate_sls_an REAL,
                cprd_sls_oc REAL,
                cprd_sls_an REAL
            )
            """
        )
        rows = [
            (
                2025,
                "P-1",
                "Project Alpha",
                "F-1",
                "Field Alpha",
                "WK-1",
                "WK Alpha",
                "Contingent Resources",
                "Development",
                "E2",
                "2. Middle Value",
                "Ada peluang workover untuk menaikkan produksi.",
                100.0,
                50.0,
                200.0,
                100.0,
                150.0,
                75.0,
                1000.0,
                500.0,
                10.0,
                5.0,
                300.0,
                150.0,
            ),
            (
                2025,
                "P-2",
                "Project Beta",
                "F-2",
                "Field Beta",
                "WK-1",
                "WK Alpha",
                "Reserves & GRR",
                "Production",
                "E0",
                "2. Middle Value",
                "Perlu debottlenecking fasilitas untuk menjaga produksi.",
                300.0,
                150.0,
                400.0,
                200.0,
                350.0,
                175.0,
                2000.0,
                1000.0,
                20.0,
                10.0,
                600.0,
                300.0,
            ),
            (
                2025,
                "P-3",
                "Project Gamma",
                "F-3",
                "Field Gamma",
                "WK-2",
                "WK Beta",
                "Prospective Resources",
                "Exploration",
                "X5",
                "2. Middle Value",
                "Prospek membutuhkan data tambahan untuk unlock resources.",
                0.0,
                0.0,
                500.0,
                250.0,
                200.0,
                100.0,
                3000.0,
                1500.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO project_resources VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )
    finally:
        conn.close()


def _summary_rows():
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        return conn.execute(
            """
            SELECT entity_level, report_year, entity_id, summary_json, source_hash
            FROM resource_summaries
            ORDER BY entity_level, entity_id
            """
        ).fetchall()
    finally:
        conn.close()


def _summary_metadata():
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        return conn.execute(
            """
            SELECT entity_level, entity_id, provider, model
            FROM resource_summaries
            ORDER BY entity_level, entity_id
            """
        ).fetchall()
    finally:
        conn.close()


def _summary_token_rows():
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        return conn.execute(
            """
            SELECT entity_level, entity_id, input_tokens_processed,
                   output_tokens_processed, total_tokens_processed,
                   token_count_source, token_count_confidence
            FROM resource_summaries
            ORDER BY entity_level, entity_id
            """
        ).fetchall()
    finally:
        conn.close()


def test_summarize_field_creates_one_field_summary(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    assert "fields 1 created/0 skipped" in result.stdout
    rows = _summary_rows()
    assert len(rows) == 1
    assert rows[0][0] == "field"
    assert rows[0][2] == "F-1"
    assert "tokens " in result.stdout
    assert " processed" in result.stdout


def test_summarize_stores_actual_llm_metadata(runner, isolated_config, monkeypatch):
    llm = MetadataLLM()
    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(
            lambda cls: {
                "name": "ollama_cloud",
                "provider_type": "ollama_cloud",
                "model": "kimi-k2.5",
            }
        ),
    )
    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda config: llm)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    assert _summary_metadata() == [
        ("field", "F-1", "deepseek", "deepseek-v4-flash")
    ]


def test_summarize_stores_actual_token_usage(runner, isolated_config, monkeypatch):
    llm = UsageMetadataLLM()
    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(
            lambda cls: {
                "name": "test-provider",
                "provider_type": "openai",
                "model": "gpt-4o-mini",
            }
        ),
    )
    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda config: llm)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    assert "tokens 125 processed" in result.stdout
    assert _summary_token_rows() == [
        ("field", "F-1", 100, 25, 125, "provider_usage", "exact")
    ]


def test_summarize_estimates_token_usage_without_provider_usage(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    token_rows = _summary_token_rows()
    assert len(token_rows) == 1
    _, _, input_tokens, output_tokens, total_tokens, source, confidence = token_rows[0]
    assert input_tokens > 0
    assert output_tokens > 0
    assert total_tokens == input_tokens + output_tokens
    assert source == "tiktoken"
    assert confidence == "estimated"


def test_summarize_falls_back_when_provider_returns_zero_usage(
    runner, isolated_config, monkeypatch
):
    """When provider returns usage_metadata with all 0s, fall back to estimation."""
    llm = ZeroUsageMetadataLLM()
    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(
            lambda cls: {
                "name": "test-provider",
                "provider_type": "openai",
                "model": "gpt-4o-mini",
            }
        ),
    )
    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda config: llm)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    token_rows = _summary_token_rows()
    assert len(token_rows) == 1
    _, _, input_tokens, output_tokens, total_tokens, source, confidence = token_rows[0]
    assert input_tokens > 0
    assert output_tokens > 0
    assert total_tokens == input_tokens + output_tokens
    assert source == "tiktoken"
    assert confidence == "estimated"


def test_summarize_field_skips_unchanged_hash_and_force_regenerates(
    runner, isolated_config, monkeypatch
):
    llm = _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    first = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )
    second = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )
    forced = runner.invoke(
        app,
        [
            "summarize",
            "field",
            "Field Alpha",
            "--year",
            "2025",
            "--force",
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert forced.exit_code == 0
    assert "fields 0 created/1 skipped" in second.stdout
    assert "tokens 0 processed" in second.stdout
    assert "fields 1 created/0 skipped" in forced.stdout
    assert len(llm.prompts) == 2


def test_summarize_field_with_empty_remarks_does_not_call_llm(
    runner, isolated_config, monkeypatch
):
    llm = _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            UPDATE project_resources
            SET project_remarks = ''
            WHERE field_id = 'F-1'
            """
        )
    finally:
        conn.close()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    assert "fields 1 created/0 skipped" in result.stdout
    assert llm.prompts == []
    rows = _summary_rows()
    summary = json.loads(rows[0][3])
    assert _summary_token_rows() == [
        ("field", "F-1", 0, 0, 0, None, None)
    ]
    assert summary["headline"] == "Tidak ada remarks material untuk field Field Alpha."
    assert summary["data_quality_notes"] == [
        "Source tidak memiliki remarks yang cukup informatif untuk diringkas."
    ]


def test_summarize_prompt_includes_low_quality_remark_context(
    runner, isolated_config, monkeypatch
):
    llm = _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            UPDATE project_resources
            SET project_remarks = 'asdf'
            WHERE field_id = 'F-1'
            """
        )
    finally:
        conn.close()

    result = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )

    assert result.exit_code == 0
    assert len(llm.prompts) == 1
    assert "Source quality:" in llm.prompts[0]
    assert '"source_items_with_low_quality_text": 1' in llm.prompts[0]
    assert "remarks yang cukup informatif" in llm.prompts[0]


def test_summary_hides_benign_data_quality_note(runner, isolated_config, monkeypatch):
    class BenignQualityLLM(FakeLLM):
        def invoke(self, prompt):
            payload = json.loads(super().invoke(prompt))
            payload["data_quality_notes"] = ["Tidak ada isu kualitas data."]
            return json.dumps(payload)

    llm = BenignQualityLLM()
    monkeypatch.setattr(
        Config,
        "get_provider_config",
        classmethod(
            lambda cls: {
                "name": "test-provider",
                "provider_type": "openai",
                "model": "test-model",
            }
        ),
    )
    monkeypatch.setattr("esdc.providers.create_llm_from_config", lambda config: llm)
    _create_minimal_project_resources()

    generate = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )
    result = runner.invoke(app, ["summary", "field", "Field Alpha", "--year", "2025"])

    assert generate.exit_code == 0
    assert result.exit_code == 0
    assert "Data Quality Notes" not in result.output
    assert "Tidak ada isu kualitas data" not in result.output


def test_summarize_field_ambiguous_fallback_fails(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            INSERT INTO project_resources VALUES (
                2025, 'P-4', 'Project Delta', 'F-4', 'Field Alpha East',
                'WK-3', 'WK Gamma', 'Contingent Resources', 'Development', 'E2',
                '2. Middle Value', 'Remark tambahan untuk alpha east.', 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1
            )
            """
        )
    finally:
        conn.close()

    result = runner.invoke(
        app, ["summarize", "field", "Alpha", "--year", "2025"]
    )

    assert result.exit_code == 1
    assert "Ambiguous field name 'Alpha'" in result.stdout
    assert "Field Alpha (F-1)" in result.stdout
    assert "Field Alpha East (F-4)" in result.stdout


def test_summarize_progress_text_is_visible(runner, isolated_config, monkeypatch):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(app, ["summarize", "field", "--year", "2025"])

    assert result.exit_code == 0
    assert "Summarizing fields" in result.output
    assert "processed " in result.output
    assert " tokens" in result.output


def test_summarize_updates_prompt_tokens_before_invoke(isolated_config):
    token_updates: list[int] = []

    class CallbackAwareLLM(FakeLLM):
        def invoke(self, prompt):
            assert token_updates
            assert token_updates[-1] > 0
            return super().invoke(prompt)

    conn = duckdb.connect(":memory:")
    try:
        ensure_summary_table(conn)
        result = _summarize_entity(
            conn=conn,
            llm=CallbackAwareLLM(),
            level="field",
            year=2025,
            entity_id="F-1",
            entity_name="Field Alpha",
            source_level="project_remarks",
            source_items=[
                {
                    "project_id": "P-1",
                    "project_name": "Project Alpha",
                    "project_remarks": "Peluang workover untuk menaikkan produksi.",
                }
            ],
            metrics={},
            provider="test-provider",
            provider_type="openai",
            model="gpt-4o-mini",
            base_url="",
            force=False,
            retry=1,
            progress_token_callback=token_updates.append,
        )
    finally:
        conn.close()

    assert result.created
    assert token_updates[0] > 0


def test_live_token_counter_is_global_and_monotonic():
    live_tokens = {"actual": 100, "display": 100}

    preview = _preview_live_tokens(live_tokens, 75)
    committed = _commit_live_tokens(
        live_tokens,
        SummaryEntityResult(
            created=True,
            token_usage=TokenUsage(
                input_tokens=30,
                output_tokens=10,
                total_tokens=40,
                source="provider_usage",
                confidence="exact",
            ),
        ),
    )
    next_preview = _preview_live_tokens(live_tokens, 20)

    assert preview == 175
    assert committed == 175
    assert next_preview == 175
    assert live_tokens["actual"] == 140


def test_summary_field_displays_generated_summary(runner, isolated_config, monkeypatch):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    generate = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )
    assert generate.exit_code == 0

    result = runner.invoke(app, ["summary", "field", "Field Alpha", "--year", "2025"])

    assert result.exit_code == 0
    assert "Peluang produksi dan cadangan perlu diprioritaskan" in result.output
    assert "Key Challenges" in result.output
    assert "Production / Reserve Opportunities" in result.output
    assert "Field Alpha" in result.output


def test_summary_json_outputs_raw_payload(runner, isolated_config, monkeypatch):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    generate = runner.invoke(
        app, ["summarize", "field", "Field Alpha", "--year", "2025"]
    )
    assert generate.exit_code == 0

    result = runner.invoke(
        app, ["summary", "field", "Field Alpha", "--year", "2025", "--json"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["entity_level"] == "field"
    assert payload["entity_id"] == "F-1"
    assert payload["summary"]["headline"] == (
        "Peluang produksi dan cadangan perlu diprioritaskan"
    )


def test_summary_missing_gives_generate_command(runner, isolated_config):
    _create_minimal_project_resources()

    result = runner.invoke(app, ["summary", "field", "Field Alpha", "--year", "2025"])

    assert result.exit_code == 1
    assert "No summaries have been generated yet" in result.stdout
    assert "esdc summarize all --year 2025" in result.stdout


def test_summary_requires_year(runner, isolated_config):
    _create_minimal_project_resources()

    result = runner.invoke(app, ["summary", "field", "Field Alpha"])

    assert result.exit_code == 1
    assert "--year is required" in result.stdout


def test_summarize_fields_only_creates_field_summaries(
    runner, isolated_config, monkeypatch
):
    llm = _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(app, ["summarize", "field", "--year", "2025"])

    assert result.exit_code == 0
    assert "fields 3 created/0 skipped" in result.stdout
    rows = _summary_rows()
    assert {row[0] for row in rows} == {"field"}
    assert len(rows) == 3
    summary = json.loads(rows[0][3])
    assert summary["production_or_reserve_opportunities"]
    assert "Technical context key metrics" in llm.prompts[0]
    assert "res_oc" in llm.prompts[0]
    assert "Peluang" not in llm.prompts[0]


def test_summarize_working_areas_requires_field_summaries(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "wk", "--year", "2025"]
    )

    assert result.exit_code == 1
    assert "Missing field summaries" in result.stdout


def test_summarize_default_runs_field_wk_and_nkri(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(app, ["summarize", "all", "--year", "2025"])

    assert result.exit_code == 0
    rows = _summary_rows()
    assert [row[0] for row in rows].count("field") == 3
    assert [row[0] for row in rows].count("working_area") == 2
    assert [row[0] for row in rows].count("nkri") == 1
    assert "working areas 2 created/0 skipped" in result.stdout
    assert "nkri 1 created/0 skipped" in result.stdout
    assert "Summarizing fields" in result.output
    assert "Summarizing working areas" in result.output
    assert "Summarizing NKRI" in result.output


def test_summarize_skips_unchanged_hash_and_force_regenerates(
    runner, isolated_config, monkeypatch
):
    llm = _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    first = runner.invoke(app, ["summarize", "field", "--year", "2025"])
    second = runner.invoke(app, ["summarize", "field", "--year", "2025"])
    forced = runner.invoke(
        app, ["summarize", "field", "--year", "2025", "--force"]
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert forced.exit_code == 0
    assert "fields 0 created/3 skipped" in second.stdout
    assert "fields 3 created/0 skipped" in forced.stdout
    assert len(llm.prompts) == 6


def test_create_esdc_view_exposes_summary_columns(isolated_config):
    Config.init_config()
    Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        for script_name in (
            "create_table_project_resources.sql",
            "create_column_uuid.sql",
            "create_project_resources_uuid.sql",
            "create_project_resources_is_discovered.sql",
            "create_project_resources_project_stage.sql",
        ):
            sql_script = _load_sql_script(script_name).replace(
                "{table_name}", "project_resources"
            )
            for statement in [
                s.strip()
                for s in sql_script.split(";")
                if s.strip()
            ]:
                conn.execute(statement)
        conn.execute(
            """
            INSERT INTO project_resources (
                report_year, report_status, wk_id, wk_name, field_id, field_name,
                project_id, project_name, project_stage, project_class, uncert_level,
                project_remarks, pod_name, project_isactive, groovy_isactive,
                fusion_isactive, is_unitization, is_discovered
            )
            VALUES (
                2025, 'ACTIVE', 'WK-1', 'WK Alpha', 'F-1', 'Field Alpha',
                'P-1', 'Project Alpha', 'Production', 'Reserves & GRR',
                '2. Middle Value', 'Remark', 'POD', 1, 0, 0, 0, 1
            )
            """
        )
        for statement in [
            s.strip()
            for s in _load_sql_script("create_esdc_view.sql").split(";")
            if s.strip()
        ]:
            conn.execute(statement)
        conn.execute(
            """
            INSERT INTO resource_summaries (
                entity_level, report_year, entity_id, entity_name, summary_json,
                summary_text, source_hash, source_level, provider, model, generated_at
            )
            VALUES
                ('field', 2025, 'F-1', 'Field Alpha', '{}', 'Field summary', 'h1',
                 'project_remarks', 'test', 'model', 'now'),
                ('working_area', 2025, 'WK-1', 'WK Alpha', '{}', 'WK summary', 'h2',
                 'field_summary', 'test', 'model', 'now'),
                ('nkri', 2025, 'NKRI', 'NKRI', '{}', 'NKRI summary', 'h3',
                 'wk_summary', 'test', 'model', 'now')
            """
        )
        field_summary = conn.execute(
            "SELECT field_summary FROM field_resources WHERE field_id = 'F-1'"
        ).fetchone()[0]
        wk_summary = conn.execute(
            "SELECT wk_summary FROM wa_resources WHERE wk_id = 'WK-1'"
        ).fetchone()[0]
        nkri_summary = conn.execute(
            "SELECT nkri_summary FROM nkri_resources WHERE report_year = 2025"
        ).fetchone()[0]
    finally:
        conn.close()

    assert field_summary == "Field summary"
    assert wk_summary == "WK summary"
    assert nkri_summary == "NKRI summary"


def test_ensure_summary_table_migrates_token_columns(isolated_config):
    Config.init_config()
    Config.get_db_dir().mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            CREATE TABLE resource_summaries (
                entity_level TEXT NOT NULL,
                report_year INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT,
                summary_json TEXT NOT NULL,
                summary_text TEXT,
                source_hash TEXT NOT NULL,
                source_level TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (entity_level, report_year, entity_id)
            )
            """
        )
        ensure_summary_table(conn)
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(resource_summaries)").fetchall()
        }
    finally:
        conn.close()

    assert "input_tokens_processed" in columns
    assert "output_tokens_processed" in columns
    assert "total_tokens_processed" in columns
    assert "token_count_source" in columns
    assert "token_count_confidence" in columns
