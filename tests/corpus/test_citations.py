from esdc.corpus.citations import extract_letter_numbers, normalize_letter_number


def test_normalize_strips_all_whitespace_and_uppercases():
    assert normalize_letter_number("SRT- 0204 /SKKIE1000/2025/S1") == (
        "SRT-0204/SKKIE1000/2025/S1"
    )
    assert normalize_letter_number("srt-0204/skkie1000/2025/s1") == (
        "SRT-0204/SKKIE1000/2025/S1"
    )
    assert normalize_letter_number(None) == ""


def test_erratic_spacing_compares_equal():
    a = normalize_letter_number("SRT- 0204 /SKKIE1000/2025/S1")
    b = normalize_letter_number("SRT-0204/SKKIE1000/2025/S1")
    assert a == b


def test_extract_real_corpus_formats():
    body = (
        "Menindaklanjuti surat Kepala SKK Migas No. SRT-0184/SKKO0000/2016/S1 "
        "tanggal 3 Mei 2016 dan surat 0433 /BPA0000/2011/S1, serta "
        "surat T-37/MG.04/MEM.M/2025."
    )
    assert extract_letter_numbers(body) == [
        "SRT-0184/SKKO0000/2016/S1",
        "0433/BPA0000/2011/S1",
        "T-37/MG.04/MEM.M/2025",
    ]


def test_prefix_word_is_not_swallowed():
    # "surat 0433 /..." must not yield "SURAT0433/..." — the word before the
    # number is Indonesian prose, not a letter-series prefix.
    assert extract_letter_numbers("surat 0433 /BPA0000/2011/S1") == [
        "0433/BPA0000/2011/S1"
    ]


def test_plain_ratios_are_not_letter_numbers():
    assert extract_letter_numbers("rasio 1.500/2 tidak berlaku") == []


def test_duplicates_collapse_preserving_order():
    body = "No. 1347/SKKA0000/2013/S1 ... lihat 1347 / SKKA0000 / 2013 / S1"
    assert extract_letter_numbers(body) == ["1347/SKKA0000/2013/S1"]


def test_empty_input():
    assert extract_letter_numbers("") == []
