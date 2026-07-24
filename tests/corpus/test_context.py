from esdc.corpus.context import build_context_prefix, build_embed_text


def _doc(**over):
    doc = {
        "doc_type": "POD",
        "doc_topic": ["Produksi", "Cadangan"],
        "subject": "Rencana Pengembangan Lapangan Merak",
        "wk_name": ["WK Alpha"],
        "field_name": ["Merak"],
        "project_name": ["Merak Phase 2"],
        "pod_name": ["POD Merak I"],
    }
    doc.update(over)
    return doc


def test_prefix_contains_type_subject_and_entities():
    prefix = build_context_prefix(_doc())
    assert "POD" in prefix
    assert "Rencana Pengembangan Lapangan Merak" in prefix
    assert "Merak" in prefix
    assert "WK Alpha" in prefix


def test_prefix_accepts_json_string_arrays():
    prefix = build_context_prefix(
        _doc(field_name='["Merak"]', wk_name='["WK Alpha"]',
             project_name=None, pod_name=None, doc_topic='["Produksi"]')
    )
    assert "Merak" in prefix
    assert "WK Alpha" in prefix
    assert "[" not in prefix  # JSON syntax must not leak into the prefix


def test_prefix_skips_empty_fields():
    prefix = build_context_prefix(
        {"doc_type": None, "subject": "", "field_name": []}
    )
    assert prefix == ""


def test_prefix_dedupes_entities():
    prefix = build_context_prefix(
        _doc(field_name=["Merak"], project_name=["Merak"], pod_name=None,
             wk_name=None)
    )
    assert prefix.count("Merak") == 2  # once in subject, once in entities


def test_embed_text_prepends_prefix_and_section():
    out = build_embed_text("POD | Merak", "4.3 Analisis", "isi chunk")
    assert out == "POD | Merak | 4.3 Analisis\n\nisi chunk"


def test_embed_text_without_prefix_or_section_is_chunk_only():
    assert build_embed_text("", None, "isi chunk") == "isi chunk"
