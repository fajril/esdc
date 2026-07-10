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
