import pytest

from esdc.corpus.chunker import Chunk, chunk_markdown


def test_short_doc_single_chunk():
    chunks = chunk_markdown("# Judul\nisi pendek", chunk_size=3000, overlap=300)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].section == "Judul"
    assert "isi pendek" in chunks[0].text


def test_page_markers_stripped():
    md = "<!-- page 1: llm_ocr -->\n# A\nisi"
    chunks = chunk_markdown(md, chunk_size=3000, overlap=300)
    assert "<!--" not in chunks[0].text


def test_splits_on_headings_and_size():
    md = "# A\n" + "x" * 2500 + "\n# B\n" + "y" * 2500
    chunks = chunk_markdown(md, chunk_size=3000, overlap=100)
    assert len(chunks) == 2
    assert chunks[0].section == "A"
    assert chunks[1].section == "B"


def test_oversized_section_splits_with_overlap():
    md = "# Besar\n" + "z" * 7000
    chunks = chunk_markdown(md, chunk_size=3000, overlap=300)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 3000 for c in chunks)
    assert chunks[0].text[-100:] in chunks[1].text


def test_small_sections_pack_into_one_chunk():
    md = "# A\nisi a\n# B\nisi b\n# C\nisi c"
    chunks = chunk_markdown(md, chunk_size=3000, overlap=300)
    assert len(chunks) == 1
    assert chunks[0].section == "A"
    assert "isi a" in chunks[0].text
    assert "isi b" in chunks[0].text
    assert "isi c" in chunks[0].text


def test_h2_h4_headings_split():
    md = "## Keputusan\n" + "x" * 2500 + "\n#### Tindak Lanjut\n" + "y" * 2500
    chunks = chunk_markdown(md, chunk_size=3000, overlap=100)
    assert len(chunks) == 2
    assert chunks[0].section == "Keputusan"
    assert chunks[1].section == "Tindak Lanjut"


def test_overlap_with_realistic_text():
    md = "# Rapat\n" + ("kalimat panjang dengan spasi " * 300)
    chunks = chunk_markdown(md, chunk_size=3000, overlap=300)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 3000 for c in chunks)
    for prev, nxt in zip(chunks[:-1], chunks[1:], strict=True):
        shared = prev.text[-50:].strip()
        assert shared
        assert shared in nxt.text


def test_empty_markdown_returns_no_chunks():
    assert chunk_markdown("", chunk_size=3000, overlap=300) == []


def test_whitespace_only_markdown_returns_no_chunks():
    assert chunk_markdown("   \n\n  \t\n", chunk_size=3000, overlap=300) == []


def test_only_page_marker_returns_no_chunks():
    assert (
        chunk_markdown("<!-- page 1: native -->\n", chunk_size=3000, overlap=300) == []
    )


def test_chunk_indexes_are_consecutive():
    md = "# A\n" + "x" * 2500 + "\n# B\n" + "y" * 2500
    chunks = chunk_markdown(md, chunk_size=3000, overlap=100)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_markdown("# A\nbody", chunk_size=1000, overlap=1000)


def test_image_ocr_markers_stripped():
    md = (
        "<!-- page 1: native -->\n# Judul\nteks asli\n\n"
        "<!-- page 1 image 1: llm_ocr -->\n| A | B |\n|---|---|\n| 1 | 2 |"
    )
    chunks = chunk_markdown(md, chunk_size=500, overlap=50)
    joined = " ".join(c.text for c in chunks)
    assert "image 1" not in joined
    assert "page 1" not in joined
    assert "| 1 | 2 |" in joined
