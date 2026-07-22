from esdc.configs import Config


def test_corpus_config_defaults(monkeypatch):
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: None))
    cfg = Config.get_corpus_config()
    assert cfg["chunk_size"] == 3000
    assert cfg["chunk_overlap"] == 300
    assert cfg["ocr_model"] == "glm-ocr"
    assert cfg["metadata_model"] == "main"   # default chat provider
    assert cfg["cleanup_model"] == "main"    # default chat provider
    assert cfg["ocr_dpi"] == 200
    assert cfg["num_ctx"] == 16384
    assert cfg["min_chars_per_page"] == 50
    assert cfg["min_image_area"] == 0.05


def test_corpus_config_from_yaml(monkeypatch):
    yaml_cfg = {"corpus": {"chunk_size": 2000, "ocr_model": "qwen2.5vl:7b"}}
    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: yaml_cfg))
    cfg = Config.get_corpus_config()
    assert cfg["chunk_size"] == 2000
    assert cfg["chunk_overlap"] == 300          # default survives partial override
    assert cfg["ocr_model"] == "qwen2.5vl:7b"


def test_corpus_keys_documented():
    from esdc.configs import KEY_DESCRIPTIONS

    assert "corpus.ocr_model" in KEY_DESCRIPTIONS
    assert "corpus.metadata_model" in KEY_DESCRIPTIONS
    assert "corpus.cleanup_model" in KEY_DESCRIPTIONS
    assert "corpus.min_image_area" in KEY_DESCRIPTIONS


def test_corpus_ollama_host_default_and_documented(monkeypatch):
    from esdc.configs import KEY_DESCRIPTIONS

    monkeypatch.setattr(Config, "_load_config", classmethod(lambda cls: None))
    assert Config.get_corpus_config()["ollama_host"] == ""
    assert "corpus.ollama_host" in KEY_DESCRIPTIONS
