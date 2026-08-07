# tests/corpus/test_pipeline_backend.py
"""Generation commands route through get_build_embedder; others do not."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from esdc.corpus import pipeline


def test_run_commit_accepts_embed_backend():
    assert "embed_backend" in inspect.signature(pipeline.run_commit).parameters


def test_run_reembed_accepts_embed_backend():
    assert "embed_backend" in inspect.signature(pipeline.run_reembed).parameters


def test_run_export_has_no_embed_backend():
    """Export never opens an embedder; a flag would imply a dependency."""
    assert "embed_backend" not in inspect.signature(pipeline.run_export).parameters


def test_run_reembed_passes_backend_to_factory(tmp_path):
    fake = MagicMock()
    fake.model = "qwen3-embedding-0.6b-q8_0"
    fake.generate_embedding.return_value = [1.0, 0.0]
    with (
        patch.object(pipeline, "get_build_embedder", return_value=fake) as factory,
        patch.object(pipeline, "CorpusStore") as store_cls,
    ):
        store = store_cls.return_value
        store._embedder = fake
        store.list_documents.return_value = []
        pipeline.run_reembed(embed_backend="local")

    factory.assert_called_once_with("local")
    assert store_cls.call_args.kwargs["embedder"] is fake


def test_run_commit_passes_backend_to_factory(tmp_path):
    fake = MagicMock()
    fake.model = "qwen3-embedding-0.6b-q8_0"
    with (
        patch.object(pipeline, "get_build_embedder", return_value=fake) as factory,
        patch.object(pipeline, "CorpusStore") as store_cls,
        patch.object(pipeline, "_collect_sidecars", return_value=[]),
    ):
        pipeline.run_commit([], embed_backend="openai")

    factory.assert_called_once_with("openai")
    assert store_cls.call_args.kwargs["embedder"] is fake
