"""Tests for corpus cleanup and deduplication."""

from esdc.corpus.cleanup import CLEANUP_PROMPT, cleanup_markdown

NATIVE_DOC = (
    "<!-- page 1: native -->\n"
    "# **Judul dokumen**\n"
    "baris yang ter-\npotong dengan angka 8,146,475\n\n"
    "<!-- page 1 image 1: llm_ocr -->\n"
    "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
    "<!-- page 2: llm_ocr -->\n"
    "teks hasil OCR halaman 2\n"
)


def test_only_native_segments_sent_to_llm():
    prompts = []

    def caller(prompt: str) -> str:
        prompts.append(prompt)
        return "# Judul dokumen\nbaris yang terpotong dengan angka 8,146,475"

    cleaned, n_cleaned, n_rejected = cleanup_markdown(NATIVE_DOC, caller)
    assert len(prompts) == 1  # native page only; image + OCR segments untouched
    assert "ter-\npotong" in prompts[0]
    assert n_cleaned == 1 and n_rejected == 0
    # markers all survive
    assert "<!-- page 1: native -->" in cleaned
    assert "<!-- page 1 image 1: llm_ocr -->" in cleaned
    assert "<!-- page 2: llm_ocr -->" in cleaned
    # cleaned text swapped in, other segments verbatim
    assert "baris yang terpotong" in cleaned
    assert "| 1 | 2 |" in cleaned
    assert "teks hasil OCR halaman 2" in cleaned


def test_invented_number_rejected():
    def caller(prompt: str) -> str:
        return "# Judul\nangka baru 9,999,999"

    cleaned, n_cleaned, n_rejected = cleanup_markdown(NATIVE_DOC, caller)
    assert n_cleaned == 0 and n_rejected == 1
    assert "ter-\npotong" in cleaned  # original kept


def test_gross_length_change_rejected():
    def caller(prompt: str) -> str:
        return "ok"

    cleaned, n_cleaned, n_rejected = cleanup_markdown(NATIVE_DOC, caller)
    assert n_cleaned == 0 and n_rejected == 1
    assert "ter-\npotong" in cleaned


def test_caller_exception_keeps_original():
    def caller(prompt: str) -> str:
        raise RuntimeError("provider down")

    cleaned, n_cleaned, n_rejected = cleanup_markdown(NATIVE_DOC, caller)
    assert n_cleaned == 0 and n_rejected == 1
    assert "ter-\npotong" in cleaned


def test_code_fence_stripped():
    def caller(prompt: str) -> str:
        return (
            "```markdown\n# Judul dokumen\n"
            "baris yang terpotong dengan angka 8,146,475\n```"
        )

    cleaned, n_cleaned, n_rejected = cleanup_markdown(NATIVE_DOC, caller)
    assert n_cleaned == 1
    assert "```" not in cleaned.split("<!-- page 1 image 1")[0]


def test_prompt_forbids_rewording():
    assert "Do NOT reword" in CLEANUP_PROMPT
    assert "EXACTLY" in CLEANUP_PROMPT


def test_reasoning_block_stripped_before_guards():
    """A reasoning-model response must not leak into committed markdown.

    Pre-fix, the caller's raw output only went through fence-stripping, not
    strip_thinking_tags. On a long page, a short `<think>` block prefixing
    an otherwise faithful reformat keeps the combined length within the
    0.5x-1.5x guard and introduces no invented digits, so `_guard_ok`
    accepts it and the thinking prose lands verbatim in committed markdown
    (verified: pre-fix, len ratio ~1.09, `_guard_ok` True). The page below
    is long enough to reproduce that "guard passes" case rather than the
    more common "guard rejects" case a short page would hit.
    """
    long_doc = (
        "<!-- page 1: native -->\n"
        "Bagian ini menjelaskan rincian teknis proyek pengembangan lapangan "
        "yang mencakup studi subsurface, fasilitas produksi, dan jadwal "
        "implementasi tahun 2024 dengan estimasi biaya 8,146,475 dolar AS "
        "serta evaluasi risiko operasional yang harus dipertimbangkan "
        "secara menyeluruh oleh tim manajemen proyek sebelum keputusan "
        "final diambil.\n\n"
    )
    reformatted = (
        "Bagian ini menjelaskan rincian teknis proyek pengembangan lapangan "
        "yang mencakup studi subsurface, fasilitas produksi, dan jadwal "
        "implementasi tahun 2024 dengan estimasi biaya 8,146,475 dolar AS "
        "serta evaluasi risiko operasional yang harus dipertimbangkan "
        "secara menyeluruh oleh tim manajemen proyek sebelum keputusan "
        "final diambil."
    )

    def caller(prompt: str) -> str:
        return "<think>Fixing wrap only.</think>\n" + reformatted

    cleaned, n_cleaned, n_rejected = cleanup_markdown(long_doc, caller)
    assert n_cleaned == 1 and n_rejected == 0
    assert "<think>" not in cleaned
    assert "Fixing wrap only" not in cleaned
    assert reformatted in cleaned


def test_native_docx_and_native_md_segments_are_cleanable():
    doc = (
        "<!-- page 1: native_docx -->\n"
        "baris yang ter-\npotong dari docx\n\n"
        "<!-- page 1: native_md -->\n"
        "baris yang ter-\npotong dari md\n"
    )
    prompts = []

    def caller(prompt: str) -> str:
        prompts.append(prompt)
        if "docx" in prompt:
            return "baris yang terpotong dari docx"
        return "baris yang terpotong dari md"

    cleaned, n_cleaned, n_rejected = cleanup_markdown(doc, caller)
    assert len(prompts) == 2
    assert n_cleaned == 2 and n_rejected == 0
    assert "baris yang terpotong dari docx" in cleaned
    assert "baris yang terpotong dari md" in cleaned
    assert "<!-- page 1: native_docx -->" in cleaned
    assert "<!-- page 1: native_md -->" in cleaned
