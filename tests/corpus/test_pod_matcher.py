import sqlite3

import pytest

from esdc.corpus.pod_matcher import PodMatcher


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE m_pod (pod_id TEXT, pod_name TEXT, pod_letter_num TEXT)")
    c.executemany(
        "INSERT INTO m_pod VALUES (?, ?, ?)",
        [
            ("PL-2003-0005-3-2-0", "POD Mengoepeh", "294/BP"),
            ("PL-2019-0300-4-1-1", "Revisi POD I Lapangan Abadi", "SRT-0368-SKKMA0000"),
            ("PL-2016-0502-3-4-0", "POFD Lapangan Puncak", None),
        ],
    )
    return c


def test_letter_num_exact_match_wins(conn):
    ids, reasons = PodMatcher(conn).suggest("SRT-0368-SKKMA0000", None)
    assert ids == ["PL-2019-0300-4-1-1"]
    assert any("letter" in r for r in reasons)


def test_letter_num_match_is_whitespace_and_case_insensitive(conn):
    ids, _ = PodMatcher(conn).suggest("  srt-0368-skkma0000 ", None)
    assert ids == ["PL-2019-0300-4-1-1"]


def test_pod_name_exact_casefold(conn):
    ids, _ = PodMatcher(conn).suggest(None, ["pod mengoepeh"])
    assert ids == ["PL-2003-0005-3-2-0"]


def test_pod_name_substring(conn):
    # Letter says more than the registry name; containment either way counts.
    ids, _ = PodMatcher(conn).suggest(
        None, ["Persetujuan Revisi POD I Lapangan Abadi (Juli 2019)"]
    )
    assert "PL-2019-0300-4-1-1" in ids


def test_pod_name_fuzzy(conn):
    ids, _ = PodMatcher(conn).suggest(None, ["POD Mengupeh"])  # typo'd vowel
    assert ids == ["PL-2003-0005-3-2-0"]


def test_no_match_returns_empty(conn):
    ids, reasons = PodMatcher(conn).suggest("SRT-9999", ["Lapangan Antah"])
    assert ids == [] and reasons == []


def test_short_names_never_substring_match(conn):
    ids, _ = PodMatcher(conn).suggest(None, ["POD"])
    assert ids == []


def test_missing_table_suggests_nothing():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    ids, reasons = PodMatcher(c).suggest("294/BP", ["POD Mengoepeh"])
    assert ids == [] and reasons == []


def test_dedupe_and_cap(conn):
    ids, _ = PodMatcher(conn).suggest("294/BP", ["POD Mengoepeh"])
    assert ids == ["PL-2003-0005-3-2-0"]  # both signals, one suggestion
