import datetime

from esdc.corpus import rename


def test_format_doc_date_from_iso_string():
    assert rename.format_doc_date("2024-01-15") == "2024.01.15"


def test_format_doc_date_from_date_object():
    assert rename.format_doc_date(datetime.date(2024, 3, 5)) == "2024.03.05"


def test_format_doc_date_none_and_garbage():
    assert rename.format_doc_date(None) is None
    assert rename.format_doc_date("not-a-date") is None
    assert rename.format_doc_date("") is None


def test_sanitize_title_strips_illegal_chars_and_collapses_space():
    assert rename.sanitize_title("Persetujuan  POD/OPL: Duri?") == "Persetujuan POD OPL Duri"


def test_sanitize_title_none_and_empty():
    assert rename.sanitize_title(None) is None
    assert rename.sanitize_title("   ") is None
    assert rename.sanitize_title("///") is None


def test_sanitize_title_caps_length():
    out = rename.sanitize_title("x" * 300)
    assert out is not None and len(out) <= 150
