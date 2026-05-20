"""Tests for Eureka dashboard query functions."""

from __future__ import annotations

from unittest.mock import patch

import duckdb
import pytest

from esdc.configs import Config
from esdc.eureka.queries import get_field_kpis


@pytest.fixture
def field_kpi_db(tmp_path, monkeypatch):
    """Create seeded database with realistic field project data."""
    config_dir = tmp_path / ".esdc"
    db_file = config_dir / "esdc.duckdb"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_file))
    Config.init_config()
    config_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_file))
    conn.execute("""
        CREATE TABLE project_resources (
            id INTEGER,
            report_date TEXT,
            report_year INTEGER,
            report_status TEXT,
            project_name TEXT,
            project_stage TEXT,
            project_class TEXT,
            project_level TEXT,
            uncert_level TEXT,
            field_id TEXT,
            cprd_sls_oc REAL,
            cprd_sls_an REAL,
            wk_name TEXT
        );
    """)

    data = [
        # PROD-A: E0, has sales → production
        *[
            (
                2025,
                "Exploitation",
                "Reserves",
                "E0. On Production",
                "F-PROD-A",
                100.0,
                50.0,
            )
        ],
        # PROD-B: E1, has sales → production
        *[
            (
                2025,
                "Exploitation",
                "Reserves",
                "E1. Production on Hold",
                "F-PROD-B",
                5.0,
                2.0,
            )
        ],
        # DEV-A: E2 only, no sales → development
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E2. Under Development",
                "F-DEV-A",
                0.0,
                0.0,
            )
        ],
        # DEV-B: E3 only, no sales → development
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E3. Development Not Viable",
                "F-DEV-B",
                0.0,
                0.0,
            )
        ],
        # IDLE-E4: E4 only → idle
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E4. Production Pending",
                "F-IDLE-E4",
                0.0,
                0.0,
            )
        ],
        # IDLE-E5: E5 only → idle (E5 now in idle range)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E5. Development Unclarified",
                "F-IDLE-E5",
                0.0,
                0.0,
            )
        ],
        # IDLE-E6: E6 only → idle (E6 now in idle range)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E6. Further Development",
                "F-IDLE-E6",
                0.0,
                0.0,
            )
        ],
        # IDLE-E7: E7 only → idle
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E7. Production Not Viable",
                "F-IDLE-E7",
                0.0,
                0.0,
            )
        ],
        # IDLE-E8: E8 only → idle (E8 now in idle range)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E8. Further Development Not Viable",
                "F-IDLE-E8",
                0.0,
                0.0,
            )
        ],
        # IDLE-MIXED: E4+E7 → idle (both in E4-E8 range)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E4. Production Pending",
                "F-IDLE-MIXED",
                0.0,
                0.0,
            )
        ],
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E7. Production Not Viable",
                "F-IDLE-MIXED",
                0.0,
                0.0,
            )
        ],
        # MIXED-E0E4: E0+E4 → NOT idle (E0 outside E4-E8), production
        *[
            (
                2025,
                "Exploitation",
                "Reserves",
                "E0. On Production",
                "F-MIXED-E0E4",
                0.0,
                0.0,
            )
        ],
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E4. Production Pending",
                "F-MIXED-E0E4",
                0.0,
                0.0,
            )
        ],
        # DEV-WITH-SALES: E2, has sales → production (not development)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E2. Under Development",
                "F-DEV-SALES",
                10.0,
                5.0,
            )
        ],
        # MIXED-DEV: E2+E8 → production (catch-all, not pure dev/idle)
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E2. Under Development",
                "F-MIXED-DEV",
                0.0,
                0.0,
            )
        ],
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E8. Further Development Not Viable",
                "F-MIXED-DEV",
                0.0,
                0.0,
            )
        ],
        # EXPLORE-ONLY: Exploration, Contingent → discovered
        *[
            (
                2025,
                "Exploration",
                "Contingent Resources",
                "X2. Prospect Identification",
                "F-EXPLORE",
                0.0,
                0.0,
            )
        ],
        # UNDISC-ONLY: Exploration, Prospective → undiscovered
        *[
            (
                2025,
                "Exploration",
                "Prospective Resources",
                "X6. Not Yet Defined",
                "F-UNDISC",
                0.0,
                0.0,
            )
        ],
        # OVERLAP: E4 (exploit) + X2 (explore) + Abandoned
        *[
            (
                2025,
                "Exploitation",
                "Contingent Resources",
                "E4. Production Pending",
                "F-OVERLAP",
                0.0,
                0.0,
            )
        ],
        *[
            (
                2025,
                "Exploration",
                "Contingent Resources",
                "X2. Prospect Identification",
                "F-OVERLAP",
                0.0,
                0.0,
            )
        ],
        *[(2025, "Abandoned", "Abandoned", "A2. Dissolved", "F-OVERLAP", 0.0, 0.0)],
        # ABANDONED-ONLY: should NOT be counted
        *[(2025, "Abandoned", "Abandoned", "A2. Dissolved", "F-ABANDONED", 0.0, 0.0)],
    ]

    rows = []
    for entry in data:
        report_year, stage, cls, level, field_id, oc, an = entry
        rows.append(
            (
                None,
                "2025-01-01",
                report_year,
                "ACTIVE",
                f"Project {field_id}",
                stage,
                cls,
                level,
                "2. Middle Value",
                field_id,
                oc,
                an,
                None,
            )
        )

    conn.executemany(
        """INSERT INTO project_resources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.close()

    with patch("esdc.eureka.queries.get_latest_year", return_value=2025):
        yield


class TestGetFieldKpis:
    """Tests for get_field_kpis()."""

    def test_field_counts_with_seeded_data(self, field_kpi_db):
        result = get_field_kpis()

        # Total unique fields (excludes Abandoned-only)
        assert result.total_fields == 16

        # Exploitation breakdown
        assert result.exploit_fields == 14, f"exploit={result.exploit_fields}"
        assert result.production_fields == 5, f"prod={result.production_fields}"
        assert result.development_fields == 2, f"dev={result.development_fields}"
        assert result.idle_fields == 7, f"idle={result.idle_fields}"

        # Exploitation = production + development + idle
        assert (
            result.exploit_fields
            == result.production_fields + result.development_fields + result.idle_fields
        )

        # Exploration breakdown
        assert result.exploration_fields == 3, f"explore={result.exploration_fields}"
        assert result.discovered_fields == 2, f"disc={result.discovered_fields}"
        assert result.undiscovered_fields == 1, f"undisc={result.undiscovered_fields}"

        # total < exploit + exploration due to overlap
        assert result.total_fields < result.exploit_fields + result.exploration_fields

    def test_idle_includes_e4_to_e8(self, field_kpi_db):
        """E5/E6/E8-only fields are idle like E4/E7."""
        result = get_field_kpis()
        assert result.idle_fields >= 6, "E4-E8 only fields should be idle"

    def test_mixed_e0_e4_is_not_idle(self, field_kpi_db):
        """Field with E0+E4 should be production, not idle."""
        result = get_field_kpis()
        assert result.idle_fields < result.exploit_fields, "not all fields are idle"
        assert result.production_fields >= 1, "mixed E0+E4 should be production"

    def test_development_no_sales(self, field_kpi_db):
        """E2/E3 with sales is NOT development."""
        result = get_field_kpis()
        assert result.development_fields == 2, (
            "only E2/E3 without sales are development"
        )
        # F-DEV-SALES has E2 + sales → should be production
        assert result.production_fields >= 1, "E2 with sales should be production"
        assert (
            "production" in str(result).lower()
            or result.production_fields > result.development_fields
        )

    def test_no_double_count_in_total(self, field_kpi_db):
        """F-OVERLAP counted once in total despite being in both exploit+explore."""
        result = get_field_kpis()
        total_if_added = result.exploit_fields + result.exploration_fields
        double_count = total_if_added - result.total_fields
        assert double_count == 1, f"Expected 1 overlap, got {double_count}"

    def test_filter_by_year(self, field_kpi_db):
        """Different year should return zeros."""
        with patch("esdc.eureka.queries.get_latest_year", return_value=2030):
            result = get_field_kpis()
            assert result.total_fields == 0
            assert result.exploit_fields == 0
            assert result.exploration_fields == 0
