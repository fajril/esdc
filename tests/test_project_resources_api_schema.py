"""Current project-resource API payload compatibility regressions."""

from esdc.api_project_resources import normalize_project_resources


def test_current_api_row_without_legacy_geometry_is_accepted() -> None:
    result = normalize_project_resources(
        [{"project_id": "api-no-geometry", "report_year": 2025}]
    )

    assert len(result.rows) == 1
    assert result.rows[0].field_geom is None
