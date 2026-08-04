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
    _parse_summary_response,
    _preview_live_tokens,
    _strategic_analysis_data,
    _summarize_entity,
    build_summary_prompt,
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


class StrategicAnalysisLLM(FakeLLM):
    def invoke(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(
            {
                "report_year": 2025,
                "summary": {
                    "total_oil_mbopd": 1.0,
                    "total_gas_mmscfd": 1.0,
                    "total_field_mboe": 100.0,
                    "total_exploration_mboe": 80.0,
                    "key_findings": [
                        "Peluang utama berada pada Big Resource.",
                        "Pematangan POD menjadi penentu onstream.",
                    ],
                    "recommendations": [
                        "Prioritaskan persetujuan POD dan kesiapan fasilitas."
                    ],
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


def test_parse_summary_response_strips_think_block_at_start():
    """Baseline: the old ad-hoc code already handled this exact shape."""
    content = '<think>reasoning here</think>\n{"status": "ok"}'
    assert _parse_summary_response(content) == {"status": "ok"}


def test_parse_summary_response_strips_thinking_spelling():
    """The "<thinking>" spelling was never recognized by the old code.

    The old code only checked the literal "<think>" prefix, so a
    "<thinking>" block (a real spelling used by some reasoning models) was
    never stripped. Because the block below contains an unmatched "{" of
    its own, the old code's brace-slicing grabbed content starting inside
    the thinking block, producing invalid JSON and raising
    JSONDecodeError. With strip_thinking_tags removing the whole block
    first, only the real JSON object remains.
    """
    content = (
        "<thinking>value should be like {invalid} but let's see</thinking>\n"
        '{"status": "ok", "count": 1}'
    )
    assert _parse_summary_response(content) == {"status": "ok", "count": 1}


def test_parse_summary_response_strips_think_block_after_preamble():
    """A thinking block preceded by a preamble was never recognized either.

    The old code only stripped when the thinking tag was the literal
    first characters of the stripped content. A short preamble before the
    tag meant the ``startswith("<think>")`` check never fired. Because the
    thinking block here also contains a stray "{", the old code's
    brace-slicing spanned from inside the thinking block to the real
    closing brace, yielding invalid multi-object JSON and raising
    JSONDecodeError. strip_thinking_tags removes the tagged block
    regardless of position, leaving the preamble text and the real JSON;
    brace-slicing then isolates the JSON object correctly.
    """
    content = (
        "Sure, here is the analysis.\n"
        "<think>Let me consider the numbers {a: 1}</think>\n"
        '{"status": "ok"}'
    )
    assert _parse_summary_response(content) == {"status": "ok"}


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
                onstream_year INTEGER,
                operator_name TEXT,
                rec_mboe REAL,
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
                2025, "P-1", "Project Alpha", "F-1", "Field Alpha",
                "WK-1", "WK Alpha", "Contingent Resources", "Development",
                "E2", "2. Middle Value",
                "Ada peluang workover untuk menaikkan produksi.",
                2026, "Op A", 150.0,
                100.0, 50.0, 200.0, 100.0, 150.0, 75.0,
                1000.0, 500.0, 10.0, 5.0, 300.0, 150.0,
            ),
            (
                2025, "P-2", "Project Beta", "F-2", "Field Beta",
                "WK-1", "WK Alpha", "Reserves & GRR", "Production",
                "E0", "2. Middle Value",
                "Perlu debottlenecking fasilitas untuk menjaga produksi.",
                2026, "Op A", 300.0,
                300.0, 150.0, 400.0, 200.0, 350.0, 175.0,
                2000.0, 1000.0, 20.0, 10.0, 600.0, 300.0,
            ),
            (
                2025, "P-3", "Project Gamma", "F-3", "Field Gamma",
                "WK-2", "WK Beta", "Prospective Resources", "Exploration",
                "X5", "2. Middle Value",
                "Prospek membutuhkan data tambahan untuk unlock resources.",
                2027, "Op B", 350.0,
                0.0, 0.0, 500.0, 250.0, 200.0, 100.0,
                3000.0, 1500.0, 0.0, 0.0, 0.0, 0.0,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO project_resources VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )
    finally:
        conn.close()


def _create_project_timeseries():
    conn = duckdb.connect(str(Config.get_db_file()))
    try:
        conn.execute(
            """
            CREATE TABLE project_timeseries (
                project_id TEXT,
                report_year INTEGER,
                year INTEGER,
                onstream_year INTEGER,
                project_level TEXT,
                tpf_oc REAL,
                tpf_an REAL,
                project_remarks TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO project_timeseries
            (project_id, report_year, year, onstream_year, project_level,
             tpf_oc, tpf_an, project_remarks)
            VALUES
            ('P-1', 2025, 2026, 2026, 'E2', 365.0, 0.0, 'Workover to increase production'),
            ('P-2', 2025, 2026, 2026, 'E0', 730.0, 0.5, 'Facility debottlenecking')
            """
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


def _create_strategic_analysis_tables(conn):
    conn.execute(
        """
        CREATE TABLE project_resources (
            report_year INTEGER,
            project_id TEXT,
            project_name TEXT,
            wk_name TEXT,
            operator_name TEXT,
            field_name TEXT,
            project_level TEXT,
            uncert_level TEXT,
            onstream_year INTEGER,
            rec_oc REAL,
            rec_an REAL,
            rec_mboe REAL,
            project_remarks TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO project_resources VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            (
                2025,
                "P-BIG",
                "Big Resource",
                "WK Alpha",
                "Op A",
                "Field A",
                "E6. Further Development",
                "2. Middle Value",
                2026,
                1000.0,
                1000.0,
                100.0,
                "Big project requires POD finalization.",
            ),
            (
                2025,
                "P-X1",
                "Major Discovery",
                "WK Alpha",
                "Op A",
                "Field B",
                "X1. Discovery under Evaluation",
                "2. Middle Value",
                2027,
                800.0,
                800.0,
                80.0,
                "Discovery requires appraisal.",
            ),
            (
                2025,
                "P-X0",
                "Pending Development",
                "WK Alpha",
                "Op A",
                "Field C",
                "X0. Development Pending",
                "2. Middle Value",
                2026,
                600.0,
                600.0,
                60.0,
                "Development decision pending.",
            ),
            (
                2025,
                "P-MID",
                "Middle Development",
                "WK Alpha",
                "Op A",
                "Field D",
                "E6. Further Development",
                "2. Middle Value",
                2027,
                400.0,
                400.0,
                40.0,
                "Needs facility optimization.",
            ),
            (
                2025,
                "P-LOW",
                "Low Pending",
                "WK Alpha",
                "Op A",
                "Field E",
                "X0. Development Pending",
                "2. Middle Value",
                2026,
                200.0,
                200.0,
                20.0,
                "Economics under review.",
            ),
            (
                2025,
                "P-SMALL-FORECAST",
                "Small High Forecast",
                "WK Alpha",
                "Op A",
                "Field F",
                "E2. Under Development",
                "2. Middle Value",
                2026,
                100.0,
                100.0,
                10.0,
                "Forecast is high but resources are small.",
            ),
            (
                2025,
                "P-TINY-FORECAST",
                "Tiny High Forecast",
                "WK Alpha",
                "Op A",
                "Field G",
                "E0. On Production",
                "2. Middle Value",
                2026,
                50.0,
                50.0,
                5.0,
                "Very high forecast but lowest resources.",
            ),
            (
                2025,
                "P-X2",
                "Exploration Prospect",
                "WK Alpha",
                "Op A",
                "Field H",
                "X2. Exploration Prospect",
                "2. Middle Value",
                2026,
                300.0,
                700.0,
                70.0,
                "Prospect requires seismic maturation.",
            ),
            (
                2025,
                "P-X3",
                "Exploration Lead",
                "WK Alpha",
                "Op A",
                "Field I",
                "X3. Exploration Lead",
                "2. Middle Value",
                2027,
                250.0,
                650.0,
                65.0,
                "Lead requires prospect maturation.",
            ),
        ],
    )
    conn.execute(
        """
        CREATE TABLE project_timeseries (
            project_id TEXT,
            report_year INTEGER,
            year INTEGER,
            tpf_oc REAL,
            tpf_an REAL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO project_timeseries VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("P-BIG", 2025, 2026, 365.0, 0.365),
            ("P-X1", 2025, 2027, 730.0, 0.730),
            ("P-X0", 2025, 2026, 1095.0, 1.095),
            ("P-MID", 2025, 2027, 1460.0, 1.460),
            ("P-LOW", 2025, 2026, 1825.0, 1.825),
            ("P-SMALL-FORECAST", 2025, 2026, 36500.0, 36.5),
            ("P-TINY-FORECAST", 2025, 2026, 73000.0, 73.0),
        ],
    )


def test_strategic_analysis_data_uses_cumulative_80_percent_contributors():
    conn = duckdb.connect(":memory:")
    try:
        _create_strategic_analysis_tables(conn)

        data = _strategic_analysis_data(conn, 2025)

        assert set(data) == {
            "report_year",
            "analysis_year",
            "outlook_year",
            "oil_analysis",
            "gas_analysis",
            "field_development",
            "exploration_highlights",
        }
        oil_names = {
            project["project_name"]
            for project in data["oil_analysis"]["priority_projects"]
        }
        gas_names = {
            project["project_name"]
            for project in data["gas_analysis"]["priority_projects"]
        }
        assert data["analysis_year"] == 2026
        assert data["outlook_year"] == 2027
        assert oil_names == {"Tiny High Forecast", "Small High Forecast"}
        assert gas_names == {"Tiny High Forecast", "Small High Forecast"}
        assert data["oil_analysis"]["top3_projects"][0]["project_name"] == (
            "Tiny High Forecast"
        )
        assert data["oil_analysis"]["outlook_top3_projects"][0]["project_name"] == (
            "Middle Development"
        )
        assert data["oil_analysis"]["total_projects_reviewed"] == 5
        assert data["oil_analysis"]["total_priority_projects"] == 2
        assert data["oil_analysis"]["total_mbopd"] == 309.0
        assert data["gas_analysis"]["total_mmscfd"] == 309.0

        field_development = data["field_development"]
        assert field_development["total_projects_reviewed"] == 3
        assert field_development["priority_projects"][0]["project_level"] == (
            "E6. Further Development"
        )
        assert field_development["priority_projects"][0]["project_name"] == "Big Resource"

        exploration = data["exploration_highlights"]
        assert exploration["total_projects_reviewed"] == 1
        assert exploration["priority_projects"][0]["project_level"] == (
            "X2. Exploration Prospect"
        )
        assert [
            project["project_level"]
            for project in exploration["outlook_top3_projects"]
        ] == ["X1. Discovery under Evaluation", "X3. Exploration Lead"]
    finally:
        conn.close()


def test_strategic_summary_prompt_matches_reference_contributor_rule():
    prompt = build_summary_prompt(
        level="nkri",
        year=2025,
        entity_name="NKRI",
        source_level="strategic_analysis",
        source_items=[],
        metrics={"project_count": 1},
        strategic_data={
            "report_year": 2025,
            "analysis_year": 2026,
            "outlook_year": 2027,
            "oil_analysis": {
                "total_projects_reviewed": 1,
                "total_priority_projects": 1,
                "total_mbopd": 1.0,
                "priority_projects": [
                    {
                        "project_name": "Big Resource",
                        "mbopd": 1.0,
                        "project_remarks": "Requires POD finalization.",
                    }
                ],
            }
        },
    )

    assert "80% kontribusi" in prompt
    assert "{report_year} + 1" in prompt
    assert "top 3 proyek" in prompt
    assert "Jangan menggunakan Markdown table" in prompt
    assert "hanya memiliki 4 sub-header" in prompt
    assert "rec_mboe >= percentile 80" not in prompt
    assert "pct_rank >= 0.80" not in prompt
    assert "oil_analysis" in prompt
    assert "gas_analysis" in prompt
    assert "field_development" in prompt
    assert "exploration_highlights" in prompt
    assert "key_findings" in prompt
    assert "recommendations" in prompt
    assert "Jangan keluarkan daftar proyek" in prompt
    assert '"current_situation"' not in prompt
    assert '"key_challenges"' not in prompt
    assert '"solution_proposals"' not in prompt
    assert '"management_attention"' not in prompt
    assert '"data_quality_notes"' not in prompt


def test_strategic_summary_stores_reference_schema_and_text():
    conn = duckdb.connect(":memory:")
    try:
        ensure_summary_table(conn)
        result = _summarize_entity(
            conn=conn,
            llm=StrategicAnalysisLLM(),
            level="nkri",
            year=2025,
            entity_id="NKRI",
            entity_name="NKRI",
            source_level="strategic_analysis",
            source_items=[],
            metrics={},
            provider="test-provider",
            provider_type="openai",
            model="gpt-4o-mini",
            base_url="",
            force=False,
            retry=1,
            strategic_data={
                "report_year": 2025,
                "analysis_year": 2026,
                "outlook_year": 2027,
                "oil_analysis": {
                    "total_projects_reviewed": 1,
                    "total_priority_projects": 1,
                    "total_mbopd": 1.0,
                    "priority_projects": [
                        {
                            "wk_name": "WK Alpha",
                            "project_name": "Big Resource",
                            "project_level": "E6. Further Development",
                            "onstream_year": 2026,
                            "rec_oc": 1000.0,
                            "rec_mboe": 100.0,
                            "mbopd": 1.0,
                            "scale": "Besar",
                            "project_remarks": "Requires POD finalization.",
                            "issues": ["POD belum final"],
                            "mitigation": ["Percepat persetujuan POD"],
                        }
                    ],
                    "top3_projects": [
                        {
                            "wk_name": "WK Alpha",
                            "project_name": "Big Resource",
                            "project_level": "E6. Further Development",
                            "onstream_year": 2026,
                            "rec_oc": 1000.0,
                            "rec_mboe": 100.0,
                            "mbopd": 1.0,
                            "scale": "Besar",
                            "issues": ["POD belum final"],
                            "mitigation": ["Percepat persetujuan POD"],
                        }
                    ],
                    "outlook_top3_projects": [],
                },
                "gas_analysis": {"priority_projects": []},
                "field_development": {"priority_projects": []},
                "exploration_highlights": {"priority_projects": []},
            },
        )

        assert result.created
        row = conn.execute(
            """
            SELECT summary_json, summary_text
            FROM resource_summaries
            WHERE entity_level = 'nkri'
              AND report_year = 2025
              AND entity_id = 'NKRI'
            """
        ).fetchone()
        assert row is not None
        summary = json.loads(row[0])
        summary_text = row[1]
        assert "oil_analysis" in summary
        assert "summary" in summary
        assert "current_situation" not in summary
        assert "key_challenges" not in summary
        assert summary["oil_analysis"]["total_projects_reviewed"] == 1
        assert summary["oil_analysis"]["priority_projects"][0]["issues"] == [
            "POD belum final"
        ]
        assert summary_text.startswith("# Strategic Evaluation")
        assert "Pada tahun 2026" in summary_text
        assert "Pada tahun analisis" not in summary_text
        assert "## Potensi Peningkatan Produksi Minyak" in summary_text
        assert "## Potensi Peningkatan Produksi Gas" in summary_text
        assert "## Potensi Pengembangan Lapangan" in summary_text
        assert "## Exploration Highlight" in summary_text
        assert "## Ringkasan Eksekutif" not in summary_text
        assert "## Arahan Manajemen" not in summary_text
        assert "| Proyek |" not in summary_text
        assert "**Big Resource**" in summary_text
        assert "**WK Alpha**" in summary_text
        assert "POD belum final" in summary_text
        assert "Mitigasi yang relevan mencakup percepatan persetujuan POD." in (
            summary_text
        )
        assert "Mitigasi yang relevan adalah" not in summary_text
        assert ".." not in summary_text
        assert "Recommendations:" not in summary_text
        assert "`rec_" not in summary_text
        assert "Terdapat kendala subsurface" not in summary_text
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
            INSERT INTO project_resources (
                report_year, project_id, project_name, field_id, field_name,
                wk_id, wk_name, project_class, project_stage, project_level,
                uncert_level, project_remarks, onstream_year, operator_name,
                rec_mboe, res_oc, res_an, rec_oc, rec_an, rec_oc_risked,
                rec_an_risked, prj_ioip, prj_igip, rate_sls_oc, rate_sls_an,
                cprd_sls_oc, cprd_sls_an
            ) VALUES (
                2025, 'P-4', 'Project Delta', 'F-4', 'Field Alpha East',
                'WK-3', 'WK Gamma', 'Contingent Resources', 'Development', 'E2',
                '2. Middle Value', 'Remark tambahan untuk alpha east.',
                2026, 'Op A', 1.0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
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


def test_summary_nkri_displays_strategic_summary_text(
    runner, isolated_config, monkeypatch
):
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
    monkeypatch.setattr(
        "esdc.providers.create_llm_from_config",
        lambda config: StrategicAnalysisLLM(),
    )
    _create_minimal_project_resources()
    _create_project_timeseries()
    generate = runner.invoke(app, ["summarize", "nkri", "--year", "2025"])
    assert generate.exit_code == 0

    result = runner.invoke(app, ["summary", "nkri", "--year", "2025"])

    assert result.exit_code == 0
    assert "Potensi Peningkatan Produksi Minyak" in result.output
    assert "Potensi Peningkatan Produksi Gas" in result.output
    assert "Potensi Pengembangan Lapangan" in result.output
    assert "Exploration Highlight" in result.output
    assert "source=strategic_analysis" in result.output


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


def test_summarize_working_areas_requires_project_timeseries(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()

    result = runner.invoke(
        app, ["summarize", "wk", "--year", "2025"]
    )

    assert result.exit_code == 1
    assert result.exception is not None
    assert "project_timeseries" in str(result.exception).lower()


def test_summarize_default_runs_field_wk_and_nkri(
    runner, isolated_config, monkeypatch
):
    _patch_llm(monkeypatch)
    _create_minimal_project_resources()
    _create_project_timeseries()

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
