"""Tests for corpus configuration."""

from esdc.configs import KEY_DESCRIPTIONS, Config


def test_corpus_config_defaults_are_processing_only(monkeypatch):
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: None))
    cfg = Config.get_corpus_config()
    assert cfg["chunk_size"] == 3000
    assert cfg["chunk_overlap"] == 300
    assert cfg["ocr_dpi"] == 200
    assert cfg["num_ctx"] == 16384
    assert cfg["min_chars_per_page"] == 50
    assert cfg["min_image_area"] == 0.05
    for key in (
        "rerank",
        "rerank_backend",
        "rerank_provider",
        "rerank_model",
        "ocr_model",
        "ocr_provider",
        "metadata_model",
        "metadata_provider",
        "cleanup_model",
        "cleanup_provider",
        "ollama_host",
        "n_gpu_layers",
    ):
        assert key not in cfg, key


def test_corpus_config_reads_processing_and_ignores_old_model_fields(monkeypatch):
    yaml_cfg = {
        "corpus": {
            "chunk_size": 2000,
            "ocr_model": "qwen2.5vl:7b",
            "metadata_model": "main",
            "ollama_host": "http://old:11434",
        }
    }
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: yaml_cfg))
    cfg = Config.get_corpus_config()
    assert cfg["chunk_size"] == 2000
    assert cfg["chunk_overlap"] == 300  # default survives partial override
    assert "ocr_model" not in cfg
    assert "metadata_model" not in cfg
    assert "ollama_host" not in cfg


def test_corpus_keys_documented():
    assert "corpus.chunk_size" in KEY_DESCRIPTIONS
    assert "corpus.min_image_area" in KEY_DESCRIPTIONS
    for legacy in (
        "corpus.ocr_model",
        "corpus.metadata_model",
        "corpus.cleanup_model",
        "corpus.ollama_host",
    ):
        assert legacy not in KEY_DESCRIPTIONS, legacy


def test_corpus_model_keys_leave_the_persistent_defaults():
    for key in (
        "ocr_model",
        "ocr_provider",
        "metadata_model",
        "metadata_provider",
        "cleanup_model",
        "cleanup_provider",
        "ollama_host",
        "rerank",
        "rerank_backend",
        "rerank_provider",
        "rerank_model",
        "n_gpu_layers",
    ):
        assert key not in Config.CORPUS_DEFAULTS, key


def test_corpus_n_gpu_layers_is_conversion_only():
    # The retired llama.cpp key is interpreted only by the one-time converter.
    assert "corpus.n_gpu_layers" not in Config.INT_KEYS
    assert "corpus.n_gpu_layers" not in KEY_DESCRIPTIONS


def test_corpus_queries_path_under_config_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "get_config_dir", classmethod(lambda cls: tmp_path))
    assert Config.get_corpus_queries_path() == tmp_path / "corpus_queries.jsonl"
