# tests/corpus/test_llama_backend.py
"""llama.cpp backend: GGUF resolution + Llama construction."""

from __future__ import annotations

from unittest.mock import patch

from esdc.corpus import llama_backend as lb


def test_resolve_gguf_downloads_into_config_models_dir(tmp_path):
    with (
        patch.object(lb.Config, "get_config_dir", return_value=tmp_path),
        patch.object(
            lb, "hf_hub_download", return_value=str(tmp_path / "x.gguf")
        ) as dl,
    ):
        out = lb.resolve_gguf("some/repo", "file.gguf")
    assert out == str(tmp_path / "x.gguf")
    _, kwargs = dl.call_args
    assert kwargs["repo_id"] == "some/repo"
    assert kwargs["filename"] == "file.gguf"
    assert kwargs["cache_dir"] == str(tmp_path / "models")


def test_load_llama_passes_gpu_layers_and_pooling():
    fake_cfg = {"n_gpu_layers": 5}
    with (
        patch.object(lb.Config, "get_corpus_config", return_value=fake_cfg),
        patch.object(lb, "Llama") as LlamaCls,
    ):
        lb.load_llama("/tmp/m.gguf", pooling_type=4, n_ctx=2048)
    _, kwargs = LlamaCls.call_args
    assert kwargs["model_path"] == "/tmp/m.gguf"
    assert kwargs["embedding"] is True
    assert kwargs["pooling_type"] == 4
    assert kwargs["n_ctx"] == 2048
    assert kwargs["n_gpu_layers"] == 5
    # embed() truncates every input at n_batch tokens (NOT n_ctx), and
    # non-causal embedding models need the whole sequence in one physical
    # batch — both must track n_ctx or chunks are silently cut at 512.
    assert kwargs["n_batch"] == 2048
    assert kwargs["n_ubatch"] == 2048
