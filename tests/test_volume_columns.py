"""Tests for volume column detection functions."""

import pytest

from esdc.chat.domain_knowledge import (
    detect_substance_from_query,
    detect_volume_type_from_query,
    get_project_stage_filter,
    get_volume_columns,
    should_use_risked_columns,
)


class TestGetVolumeColumns:
    """Tests for get_volume_columns function."""

    def test_cadangan_combined(self):
        """Cadangan without substance should use combined columns."""
        oc, an = get_volume_columns("cadangan")
        assert oc == "res_oc"
        assert an == "res_an"

    def test_cadangan_minyak(self):
        """Cadangan minyak should use specific columns."""
        oc, an = get_volume_columns("cadangan", substance="minyak")
        assert oc == "res_oil"
        assert an == "res_con"

    def test_cadangan_gas(self):
        """Cadangan gas should use specific columns."""
        oc, an = get_volume_columns("cadangan", substance="gas")
        assert oc == "res_ga"
        assert an == "res_gn"

    def test_sumberdaya_combined(self):
        """Sumberdaya without substance should use combined columns."""
        oc, an = get_volume_columns("sumber_daya")
        assert oc == "rec_oc"
        assert an == "rec_an"

    def test_potensi_risked(self):
        """Potensi should use risked columns."""
        oc, an = get_volume_columns("potensi")
        assert oc == "rec_oc_risked"
        assert an == "rec_an_risked"

    def test_potensi_minyak_risked(self):
        """Potensi minyak should use risked oil columns."""
        oc, an = get_volume_columns("potensi", substance="minyak")
        assert oc == "rec_oil_risked"
        assert an == "rec_con_risked"

    def test_potensi_gas_risked(self):
        """Potensi gas should use risked gas columns."""
        oc, an = get_volume_columns("potensi", substance="gas")
        assert oc == "rec_ga_risked"
        assert an == "rec_gn_risked"

    def test_explicit_risked_flag(self):
        """is_risked parameter should force risked columns."""
        oc, an = get_volume_columns("sumber_daya", is_risked=True)
        assert oc == "rec_oc_risked"
        assert an == "rec_an_risked"

    def test_grr_same_as_sumberdaya(self):
        """GRR should behave like sumber_daya."""
        oc1, an1 = get_volume_columns("grr")
        oc2, an2 = get_volume_columns("sumber_daya")
        assert oc1 == oc2 == "rec_oc"
        assert an1 == an2 == "rec_an"


class TestDetectSubstanceFromQuery:
    """Tests for detect_substance_from_query function."""

    def test_detect_minyak(self):
        """Should detect minyak."""
        assert (
            detect_substance_from_query("berapa cadangan minyak lapangan duri?")
            == "oil"
        )

    def test_detect_oil(self):
        """Should detect oil."""
        assert detect_substance_from_query("how much oil reserves in duri?") == "oil"

    def test_detect_gas(self):
        """Should detect gas."""
        assert (
            detect_substance_from_query("berapa cadangan gas lapangan duri?") == "gas"
        )

    def test_detect_asso(self):
        """Should detect associated gas."""
        assert detect_substance_from_query("berapa associated gas?") == "gas"

    def test_no_substance(self):
        """Should return None when no substance specified."""
        assert detect_substance_from_query("berapa cadangan lapangan duri?") is None

    def test_detect_kondensat(self):
        """Should detect kondensat/condensate as oil."""
        assert detect_substance_from_query("berapa kondensat?") == "oil"


class TestDetectVolumeTypeFromQuery:
    """Tests for detect_volume_type_from_query function."""

    def test_detect_cadangan(self):
        """Should detect cadangan."""
        assert (
            detect_volume_type_from_query("berapa cadangan lapangan duri?")
            == "cadangan"
        )

    def test_detect_potensi(self):
        """Should detect potensi."""
        assert (
            detect_volume_type_from_query("berapa potensi lapangan duri?") == "potensi"
        )

    def test_detect_potensi_proyek(self):
        """Potensi proyek should return sumber_daya (GRR), not potensi."""
        assert (
            detect_volume_type_from_query(
                "berapa potensi proyek-proyek di lapangan duri?"
            )
            == "sumber_daya"
        )

    def test_detect_sumberdaya(self):
        """Should detect sumberdaya."""
        assert (
            detect_volume_type_from_query("berapa sumberdaya proyek X?")
            == "sumber_daya"
        )

    def test_detect_eksplorasi_as_potensi(self):
        """Eksplorasi should be treated as potensi."""
        assert detect_volume_type_from_query("berapa potensi eksplorasi?") == "potensi"

    def test_default_to_cadangan(self):
        """Should default to cadangan if no type detected."""
        assert detect_volume_type_from_query("berapa lapangan duri?") == "cadangan"


class TestShouldUseRiskedColumns:
    """Tests for should_use_risked_columns function."""

    def test_potensi_field_resources(self):
        """Potensi at field level should use risked columns."""
        assert should_use_risked_columns("potensi lapangan duri?", "field_resources")

    def test_potensi_wa_resources(self):
        """Potensi at WA level should use risked columns."""
        assert should_use_risked_columns("potensi wilayah kerja rokan?", "wa_resources")

    def test_potensi_nkri_resources(self):
        """Potensi at national level should use risked columns."""
        assert should_use_risked_columns("potensi nasional?", "nkri_resources")

    def test_potensi_project_resources(self):
        """Potensi at project level should NOT use risked columns."""
        assert not should_use_risked_columns("potensi proyek X?", "project_resources")

    def test_cadangan_no_risked(self):
        """Cadangan should never use risked columns."""
        assert not should_use_risked_columns(
            "cadangan lapangan duri?", "field_resources"
        )

    def test_sumberdaya_no_risked(self):
        """Sumberdaya should not use risked columns."""
        assert not should_use_risked_columns(
            "sumberdaya proyek X?", "project_resources"
        )


class TestGetProjectStageFilter:
    """Tests for get_project_stage_filter function."""

    def test_detect_eksplorasi(self):
        """Should detect eksplorasi."""
        assert (
            get_project_stage_filter("berapa potensi eksplorasi lapangan duri?")
            == "Exploration"
        )

    def test_detect_exploration(self):
        """Should detect exploration."""
        assert (
            get_project_stage_filter("exploration potential in field Duri?")
            == "Exploration"
        )

    def test_detect_eksploitasi(self):
        """Should detect eksploitasi."""
        assert get_project_stage_filter("berapa potensi eksploitasi?") == "Exploitation"

    def test_detect_pengembangan(self):
        """Should detect pengembangan as exploitation."""
        assert get_project_stage_filter("development projects?") == "Exploitation"

    def test_no_filter(self):
        """Should return None if no stage mentioned."""
        assert get_project_stage_filter("berapa cadangan lapangan duri?") is None


class TestIntegration:
    """Integration tests combining all functions."""

    def test_scenario_1_cadangan_lapangan(self):
        """Test 1: Berapa cadangan lapangan duri?"""
        query = "berapa cadangan lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "field_resources")

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "cadangan"
        assert substance is None
        assert not is_risked
        assert oc == "res_oc"
        assert an == "res_an"

    def test_scenario_2_potensi_lapangan(self):
        """Test 2: Berapa potensi lapangan duri?"""
        query = "berapa potensi lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "field_resources")

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "potensi"
        assert substance is None
        assert is_risked
        assert oc == "rec_oc_risked"
        assert an == "rec_an_risked"

    def test_scenario_3_potensi_proyek(self):
        """Test 3: Berapa potensi proyek-proyek di lapangan duri?"""
        query = "berapa potensi proyek-proyek di lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "project_resources")

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "sumber_daya"  # Not potensi because it's at project level
        assert substance is None
        assert not is_risked  # Project level, not risked
        assert oc == "rec_oc"
        assert an == "rec_an"

    def test_scenario_4_cadangan_minyak(self):
        """Test 4: Berapa cadangan minyak lapangan duri?"""
        query = "berapa cadangan minyak lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "field_resources")

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "cadangan"
        assert substance == "oil"
        assert not is_risked
        assert oc == "res_oil"
        assert an == "res_con"

    def test_scenario_5_cadangan_gas(self):
        """Test 5: Berapa cadangan gas lapangan duri?"""
        query = "berapa cadangan gas lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "field_resources")

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "cadangan"
        assert substance == "gas"
        assert not is_risked
        assert oc == "res_ga"
        assert an == "res_gn"

    def test_scenario_6_eksplorasi_wa(self):
        """Test 6: Berapa potensi eksplorasi wilayah kerja rokan?"""
        query = "berapa potensi eksplorasi wilayah kerja rokan?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "wa_resources")
        stage = get_project_stage_filter(query)

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "potensi"
        assert substance is None
        assert is_risked
        assert stage == "Exploration"
        assert oc == "rec_oc_risked"
        assert an == "rec_an_risked"

    def test_scenario_7_eksplorasi_field(self):
        """Test 7: Berapa potensi eksplorasi di lapangan duri?"""
        query = "berapa potensi eksplorasi di lapangan duri?"

        volume_type = detect_volume_type_from_query(query)
        substance = detect_substance_from_query(query)
        is_risked = should_use_risked_columns(query, "field_resources")
        stage = get_project_stage_filter(query)

        oc, an = get_volume_columns(volume_type, is_risked, substance)

        assert volume_type == "potensi"
        assert substance is None
        assert is_risked
        assert stage == "Exploration"
        assert oc == "rec_oc_risked"
        assert an == "rec_an_risked"
