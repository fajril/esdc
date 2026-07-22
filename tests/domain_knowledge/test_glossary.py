from esdc.chat.domain_knowledge.glossary import glossary_lookup, glossary_terms


def test_glossary_lookup_by_code():
    result = glossary_lookup("TBS")
    assert result is not None
    assert "Trustee Borrowing Scheme" in result


def test_glossary_lookup_by_code_case_insensitive():
    result = glossary_lookup("tbs")
    assert result is not None
    assert "Trustee Borrowing Scheme" in result


def test_glossary_lookup_by_alias_case_insensitive():
    result = glossary_lookup("trustee borrowing")
    assert result is not None
    assert "Trustee Borrowing Scheme" in result


def test_glossary_lookup_by_key():
    result = glossary_lookup("TrusteeBorrowingScheme")
    assert result is not None
    assert "Trustee Borrowing Scheme" in result


def test_glossary_lookup_miss_returns_none():
    """GROOVY is a KSMI entity, not a glossary term — must miss here."""
    assert glossary_lookup("GROOVY") is None


def test_glossary_terms_contains_tbs():
    assert "TBS" in glossary_terms()
