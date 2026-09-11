# tests/test_reload_embeddings.py
"""esdc reload picks its generation backend and only health-checks daemons."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from esdc import esdc as cli
from esdc.configs import Config


def test_generate_embeddings_accepts_backend():
    assert "embed_backend" in inspect.signature(cli._generate_embeddings).parameters


def test_local_backend_skips_health_check(tmp_path, monkeypatch):
    # _generate_embeddings imports Config inside the function body, so patch
    # the class itself rather than a name bound in esdc.esdc.
    db = tmp_path / "esdc.duckdb"
    db.write_bytes(b"")
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db))

    fake = MagicMock(spec=["model", "generate_embedding", "generate_embeddings_batch"])
    fake.model = "qwen3-embedding-0.6b-q8_0"
    with (
        patch("esdc.embedders.get_build_embedder", return_value=fake),
        patch("esdc.search.semantic_resolver.SemanticResolver") as res_cls,
    ):
        res_cls.return_value.count_documents_with_remarks.return_value = 0
        cli._generate_embeddings(embed_backend="local")

    # spec= above means touching a health_check attribute would raise, so
    # reaching the resolver at all proves no health check was attempted.
    assert res_cls.called


def test_unhealthy_daemon_skips_without_raising(tmp_path, monkeypatch):
    db = tmp_path / "esdc.duckdb"
    db.write_bytes(b"")
    monkeypatch.setattr(Config, "get_db_file", classmethod(lambda cls: db))

    fake = MagicMock()
    fake.model = "qwen3-embedding-0.6b-q8_0"
    fake.health_check.return_value = False
    with (
        patch("esdc.embedders.get_build_embedder", return_value=fake),
        patch("esdc.search.semantic_resolver.SemanticResolver") as res_cls,
    ):
        cli._generate_embeddings(embed_backend="ollama")

    fake.health_check.assert_called_once()
    res_cls.assert_not_called()
