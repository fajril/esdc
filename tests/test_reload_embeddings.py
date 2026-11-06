# tests/test_reload_embeddings.py
"""esdc reload resolves its service, reports a retired value, and checks the client."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import typer

from esdc import esdc as cli
from esdc.configs import Config


def test_generate_embeddings_accepts_backend():
    assert "embed_backend" in inspect.signature(cli._generate_embeddings).parameters


def test_retired_value_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    """A stored `local` value stops the run with its message, not an exception."""
    db = tmp_path / "esdc.duckdb"
    db.write_bytes(b"")
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db))

    with patch("esdc.search.semantic_resolver.SemanticResolver") as res_cls:
        try:
            cli._generate_embeddings(embed_backend="local")
        except typer.Exit as exit_code:
            assert exit_code.exit_code == 1
        else:
            raise AssertionError("a retired value must stop the run")

    out = capsys.readouterr().out
    assert "in-process" in out.lower()
    assert "embedding_backend" in out
    # Nothing was written, and no client was built from the retired value.
    res_cls.assert_not_called()


def test_unhealthy_service_reports_its_own_message_and_skips(tmp_path, monkeypatch):
    """The check's diagnosis is what the user sees, and generation is skipped."""
    db = tmp_path / "esdc.duckdb"
    db.write_bytes(b"")
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db))

    printed: list[str] = []

    class CapturingConsole:
        def print(self, text: str) -> None:
            printed.append(str(text))

    fake = MagicMock()
    fake.model = "qwen3-embedding-0.6b-q8_0"
    fake.check_available.return_value = (
        "[Embedding] backend 'ollama' service at http://box returned HTTP 503."
    )
    monkeypatch.setattr("esdc.esdc.console", CapturingConsole())
    with (
        patch("esdc.embedders.get_build_embedder", return_value=fake),
        patch("esdc.search.semantic_resolver.SemanticResolver") as res_cls,
    ):
        cli._generate_embeddings(embed_backend="ollama")

    fake.check_available.assert_called_once()
    assert any("503" in line for line in printed)
    assert not any("Ollama not available" in line for line in printed)
    res_cls.assert_not_called()


def test_a_service_that_answers_runs_the_writer(tmp_path, monkeypatch):
    """The check gates the writer only when it reports a failure."""
    db = tmp_path / "esdc.duckdb"
    db.write_bytes(b"")
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db))

    fake = MagicMock()
    fake.model = "qwen3-embedding-0.6b-q8_0"
    fake.check_available.return_value = None
    with (
        patch("esdc.embedders.get_build_embedder", return_value=fake),
        patch("esdc.search.semantic_resolver.SemanticResolver") as res_cls,
    ):
        res_cls.return_value.count_documents_with_remarks.return_value = 0
        cli._generate_embeddings(embed_backend="ollama")

    assert res_cls.called
