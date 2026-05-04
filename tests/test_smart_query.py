"""Tests for build_smart_query() in view_builder.py."""

import pytest

from esdc.selection import TableName
from esdc.view_builder import build_smart_query


class TestBuildSmartQuery:
    """Tests for smart aggregate query builder."""

    def test_reserves_work_area(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.WA_RESOURCES,
            entity_name="Rokan",
        )
        sql = res["sql"]
        assert "SUM(res_oc)" in sql
        assert "reserves_mstb" in sql
        assert "SUM(res_an)" in sql
        assert "reserves_bscf" in sql
        assert "wa_resources" in sql
        assert "wk_name ILIKE" in sql
        assert "uncert_level IN" in sql
        assert "report_year" in sql
        assert "GROUP BY" not in sql

    def test_resources_work_area_grouped(self):
        res = build_smart_query(
            query_type="resources",
            table=TableName.WA_RESOURCES,
            entity_name="Rokan",
        )
        sql = res["sql"]
        assert "SUM(rec_oc_risked)" in sql
        assert "SUM(rec_an_risked)" in sql
        assert "GROUP BY project_class, project_stage" in sql

    def test_contingent_national(self):
        res = build_smart_query(
            query_type="contingent",
            table=TableName.NKRI_RESOURCES,
        )
        sql = res["sql"]
        assert "nkri_resources" in sql
        assert "project_class LIKE '%Contingent%'" in sql
        assert "GROUP BY" not in sql

    def test_prospective_field(self):
        res = build_smart_query(
            query_type="prospective",
            table=TableName.FIELD_RESOURCES,
            entity_name="Duri",
        )
        sql = res["sql"]
        assert "rec_oc_risked" in sql
        assert "rec_an_risked" in sql
        assert "project_class LIKE '%Prospective%'" in sql

    def test_cumprod_field(self):
        res = build_smart_query(
            query_type="cumprod",
            table=TableName.FIELD_RESOURCES,
            entity_name="Duri",
        )
        sql = res["sql"]
        # field_resources cumprod detail does NOT include gross cumprod columns
        assert "SUM(cprd_sls_oc)" in sql
        assert "SUM(cprd_sls_an)" in sql
        assert "cprd_grs" not in sql

    def test_prodrate_work_area(self):
        res = build_smart_query(
            query_type="prodrate",
            table=TableName.WA_RESOURCES,
            entity_name="Rokan",
        )
        sql = res["sql"]
        assert "rate" in sql

    def test_comparison_two_years(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.WA_RESOURCES,
            entity_name="Rokan",
            report_years=[2023, 2024],
        )
        sql = res["sql"]
        assert "report_year IN (?, ?)" in sql
        assert "GROUP BY report_year" in sql
        assert res["params"] == ["%Rokan%", 2023, 2024]

    def test_trend_three_years(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.WA_RESOURCES,
            entity_name="Rokan",
            report_years=[2022, 2023, 2024],
        )
        sql = res["sql"]
        assert "report_year IN (?, ?, ?)" in sql
        assert "GROUP BY report_year" in sql

    def test_uncertainty_1p(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.FIELD_RESOURCES,
            entity_name="Duri",
            uncertainty="1P",
        )
        sql = res["sql"]
        assert "uncert_level = '1. Low Value'" in sql

    def test_default_uncertainty_2p(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.FIELD_RESOURCES,
            entity_name="Duri",
        )
        sql = res["sql"]
        assert "uncert_level IN ('1. Low Value', '2. Middle Value')" in sql

    def test_national_no_entity_filter(self):
        res = build_smart_query(
            query_type="reserves",
            table=TableName.NKRI_RESOURCES,
        )
        sql = res["sql"]
        assert "ILIKE" not in sql
        assert "nkri_resources" in sql

    def test_invalid_query_type_raises(self):
        with pytest.raises(ValueError, match="Invalid query_type"):
            build_smart_query(
                query_type="invalid",
                table=TableName.WA_RESOURCES,
                entity_name="Rokan",
            )

    def test_project_level_is_ignored(self):
        """PROJECT_RESOURCES is valid but entity filtering still works."""
        res = build_smart_query(
            query_type="reserves",
            table=TableName.PROJECT_RESOURCES,
            entity_name="Abadi",
        )
        sql = res["sql"]
        assert "project_name ILIKE" in sql
