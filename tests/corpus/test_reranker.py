# tests/corpus/test_reranker.py
"""Reranker on llama.cpp RANK pooling (mocked model)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from esdc.corpus import reranker as rr_mod


def setup_function(_):
    # reset singleton between tests
    rr_mod.Reranker._instance = None
    rr_mod.Reranker._failed = False


def test_resolve_model_name_default():
    with patch.object(rr_mod.Config, "get_corpus_config", return_value={}):
        assert rr_mod._resolve_model_name() == rr_mod.DEFAULT_RERANKER


def test_rerank_score_is_index_zero_of_rank_output():
    model = MagicMock()
    # relevant -> high P(yes); irrelevant -> low
    model.embed.side_effect = lambda pair: (
        [0.98, 0.02] if "450 juta" in pair else [0.30, 0.70]
    )
    r = rr_mod.Reranker(model)
    scores = r.rerank("cadangan?", ["Banyu Urip 450 juta barel", "keselamatan kerja"])
    assert scores[0] > scores[1]
    assert scores[0] == 0.98


def test_rerank_input_uses_official_qwen3_template():
    model = MagicMock()
    model.embed.return_value = [0.5, 0.5]
    rr_mod.Reranker(model).rerank("cadangan?", ["some doc"])
    prompt = model.embed.call_args[0][0]
    # GGUF does not bake the template; we must build it. Official format:
    # system prefix + <Instruct>/<Query>/<Document> + assistant/<think> suffix.
    assert "<Instruct>:" in prompt
    assert "<Query>: cadangan?" in prompt
    assert "<Document>: some doc" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_get_returns_none_and_caches_failure_on_load_error():
    with patch.object(rr_mod, "_load_reranker", side_effect=RuntimeError("boom")):
        assert rr_mod.Reranker.get() is None
        assert rr_mod.Reranker.get() is None  # cached, no second attempt
