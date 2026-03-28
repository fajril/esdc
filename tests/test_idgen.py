from esdc.idgen import (
    _luhn_mod16,
    gen_field_id,
    gen_project_id,
    gen_zone_id,
    verify_field_id,
    verify_zone_id,
)


def test_luhn_mod16_known_payloads():
    assert _luhn_mod16("2C0123") == "B"
    assert _luhn_mod16("2C0124") == "9"
    assert _luhn_mod16("AA9876") == "4"


def test_gen_field_id_from_grid_only():
    generated = gen_field_id(["2C"], [], total_id=3)
    assert generated == [
        "F-2C-0123-B",
        "F-2C-0124-9",
        "F-2C-0125-7",
    ]


def test_gen_field_id_advances_existing_suffixes():
    existing_ids = ["F-2C-0123-B", "F-2C-0124-9"]
    generated = gen_field_id(["2C"], existing_ids, total_id=2)
    assert generated == [
        "F-2C-0125-7",
        "F-2C-0126-5",
    ]


def test_gen_project_id_from_field():
    generated = gen_project_id("F-2C-0123-B", [], total_id=3)
    assert generated == [
        "P-2C0123B-01",
        "P-2C0123B-02",
        "P-2C0123B-03",
    ]


def test_gen_project_id_advances_existing():
    existing = [
        "P-2C0123B-01",
        "P-2C0123B-02",
        "P-2C0123B-03",
    ]
    generated = gen_project_id("F-2C-0123-B", existing, total_id=2)
    assert generated == [
        "P-2C0123B-04",
        "P-2C0123B-05",
    ]


def test_gen_zone_id_from_field():
    generated = gen_zone_id("F-2C-0123-B", [], total_id=3)
    assert generated == [
        "Z-2C0123B-012-B",
        "Z-2C0123B-013-9",
        "Z-2C0123B-014-7",
    ]


def test_gen_zone_id_advances_existing():
    existing = [
        "Z-2C0123B-012-B",
        "Z-2C0123B-013-9",
    ]
    generated = gen_zone_id("F-2C-0123-B", existing, total_id=2)
    assert generated == [
        "Z-2C0123B-014-7",
        "Z-2C0123B-015-5",
    ]


def test_verify_field_id():
    assert verify_field_id("F-2C-0123-B")
    assert not verify_field_id("F-2C-0123-C")


def test_verify_zone_id():
    assert verify_zone_id("Z-2C0123B-012-B")
    assert not verify_zone_id("Z-2C0123B-012-C")
