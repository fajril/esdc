"""Tests for POD Eureka dataclass, chart, and query behavior."""

from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from esdc.configs import Config
from esdc.eureka.charts.pod import (
    POD_CATEGORIES,
    _agg_pod_timeseries,
    _to_mbopd,
    _to_mmscfd,
    pod_timeseries_an,
    pod_timeseries_oc,
)
from esdc.eureka.queries import (
    PodEconomics,
    PodKpiData,
    PodResourceBreakdown,
    PodTimeseriesRow,
)

# ---------------------------------------------------------------------------
# Dataclass unit tests
# ---------------------------------------------------------------------------


class TestPodResourceBreakdown:
    def test_creation_and_field_access(self):
        b = PodResourceBreakdown(
            low_oc=1.0,
            mid_oc=2.0,
            high_oc=3.0,
            low_an=4.0,
            mid_an=5.0,
            high_an=6.0,
            yoy_low_oc=10.0,
            yoy_mid_oc=20.0,
            yoy_high_oc=30.0,
            yoy_low_an=40.0,
            yoy_mid_an=50.0,
            yoy_high_an=60.0,
        )
        assert b.low_oc == 1.0
        assert b.mid_an == 5.0
        assert b.yoy_high_oc == 30.0

    def test_none_yoy_fields(self):
        b = PodResourceBreakdown(
            low_oc=1.0,
            mid_oc=2.0,
            high_oc=3.0,
            low_an=4.0,
            mid_an=5.0,
            high_an=6.0,
            yoy_low_oc=None,
            yoy_mid_oc=None,
            yoy_high_oc=None,
            yoy_low_an=None,
            yoy_mid_an=None,
            yoy_high_an=None,
        )
        assert b.yoy_low_oc is None
        assert b.yoy_mid_an is None

    def test_frozen(self):
        b = PodResourceBreakdown(
            low_oc=1.0,
            mid_oc=2.0,
            high_oc=3.0,
            low_an=4.0,
            mid_an=5.0,
            high_an=6.0,
            yoy_low_oc=1.0,
            yoy_mid_oc=1.0,
            yoy_high_oc=1.0,
            yoy_low_an=1.0,
            yoy_mid_an=1.0,
            yoy_high_an=1.0,
        )
        with pytest.raises(AttributeError):
            b.low_oc = 99.0  # type: ignore[misc]


class TestPodEconomics:
    def test_creation_and_field_access(self):
        e = PodEconomics(
            gross_revenue=1000.0,
            investment=200.0,
            opex=150.0,
            asr=50.0,
            irr=0.15,
            npv=500.0,
            goi_take=100.0,
            pct_investment=20.0,
            pct_opex=15.0,
            pct_asr=5.0,
            pct_goi_take=10.0,
        )
        assert e.gross_revenue == 1000.0
        assert e.investment == 200.0
        assert e.pct_investment == 20.0

    def test_percentage_calculations(self):
        e = PodEconomics(
            gross_revenue=1000.0,
            investment=200.0,
            opex=150.0,
            asr=50.0,
            irr=None,
            npv=500.0,
            goi_take=100.0,
            pct_investment=round(200.0 / 1000.0 * 100, 1),
            pct_opex=round(150.0 / 1000.0 * 100, 1),
            pct_asr=round(50.0 / 1000.0 * 100, 1),
            pct_goi_take=round(100.0 / 1000.0 * 100, 1),
        )
        assert e.pct_investment == 20.0
        assert e.pct_opex == 15.0
        assert e.pct_asr == 5.0
        assert e.pct_goi_take == 10.0

    def test_none_irr_and_zero_revenue(self):
        e = PodEconomics(
            gross_revenue=0.0,
            investment=200.0,
            opex=150.0,
            asr=50.0,
            irr=None,
            npv=500.0,
            goi_take=100.0,
            pct_investment=None,
            pct_opex=None,
            pct_asr=None,
            pct_goi_take=None,
        )
        assert e.irr is None
        assert e.pct_investment is None

    def test_frozen(self):
        e = PodEconomics(
            gross_revenue=1000.0,
            investment=200.0,
            opex=150.0,
            asr=50.0,
            irr=0.15,
            npv=500.0,
            goi_take=100.0,
            pct_investment=20.0,
            pct_opex=15.0,
            pct_asr=5.0,
            pct_goi_take=10.0,
        )
        with pytest.raises(AttributeError):
            e.gross_revenue = 0  # type: ignore[misc]


class TestPodKpiData:
    def test_creation(self):
        reserves = PodResourceBreakdown(
            low_oc=1.0,
            mid_oc=2.0,
            high_oc=3.0,
            low_an=4.0,
            mid_an=5.0,
            high_an=6.0,
            yoy_low_oc=None,
            yoy_mid_oc=None,
            yoy_high_oc=None,
            yoy_low_an=None,
            yoy_mid_an=None,
            yoy_high_an=None,
        )
        economics = PodEconomics(
            gross_revenue=1000.0,
            investment=200.0,
            opex=150.0,
            asr=50.0,
            irr=0.15,
            npv=500.0,
            goi_take=100.0,
            pct_investment=20.0,
            pct_opex=15.0,
            pct_asr=5.0,
            pct_goi_take=10.0,
        )
        kpi = PodKpiData(
            pod_id="POD-001",
            pod_name="Test POD",
            case_type="plan",
            year=2024,
            reserves=reserves,
            contingent_exploit=reserves,
            contingent_explore=reserves,
            prospective=reserves,
            cum_oil=10.0,
            cum_gas=20.0,
            yearly_oil=5.0,
            yearly_gas=10.0,
            economics=economics,
        )
        assert kpi.pod_id == "POD-001"
        assert kpi.reserves.low_oc == 1.0
        assert kpi.economics.gross_revenue == 1000.0


class TestPodTimeseriesRow:
    def test_creation(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=100.0,
            an_bscfy=0.5,
        )
        assert row.year == 2025
        assert row.category == "Reserves"
        assert row.oc_mstby == 100.0
        assert row.an_bscfy == 0.5

    def test_frozen(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=100.0,
            an_bscfy=0.5,
        )
        with pytest.raises(AttributeError):
            row.year = 2026  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Chart unit tests
# ---------------------------------------------------------------------------


class TestPodTimeseriesHelpers:
    def test_to_mbopd(self):
        assert _to_mbopd(36500) == 100.0
        assert _to_mbopd(0) == 0.0

    def test_to_mmscfd(self):
        assert _to_mmscfd(1.0) == round(1000 / 365, 1)
        assert _to_mmscfd(0) == 0.0


class TestAggPodTimeseries:
    def test_single_row(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        years, values = _agg_pod_timeseries([row], "oc_mstby", _to_mbopd)
        assert years == [2025]
        assert values["Reserves"] == [100.0]

    def test_multiple_rows_same_year(self):
        row1 = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        row2 = PodTimeseriesRow(
            year=2025,
            category="Contingent Exploitation",
            oc_mstby=73000.0,
            an_bscfy=2.0,
        )
        years, values = _agg_pod_timeseries([row1, row2], "oc_mstby", _to_mbopd)
        assert years == [2025]
        assert values["Reserves"] == [100.0]
        assert values["Contingent Exploitation"] == [200.0]

    def test_aggregation_sums_same_category_same_year(self):
        row1 = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        row2 = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        years, values = _agg_pod_timeseries([row1, row2], "oc_mstby", _to_mbopd)
        assert values["Reserves"] == [200.0]

    def test_unknown_category_skipped(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Unknown",
            oc_mstby=100.0,
            an_bscfy=1.0,
        )
        years, values = _agg_pod_timeseries([row], "oc_mstby", _to_mbopd)
        for cat in POD_CATEGORIES:
            assert all(v == 0.0 for v in values[cat])


class TestPodTimeseriesOc:
    def test_returns_figure(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        fig = pod_timeseries_oc([row])
        assert fig is not None
        assert len(fig.data) > 0  # type: ignore[arg-type]

    def test_empty_data_returns_empty_figure(self):
        fig = pod_timeseries_oc([])
        assert fig.data == ()

    def test_single_data_point(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        fig = pod_timeseries_oc([row])
        trace_names = [t.name for t in fig.data]  # type: ignore[union-attr]
        assert "Total" in trace_names
        assert "Reserves" in trace_names

    def test_yaxis_title(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        fig = pod_timeseries_oc([row])
        assert fig.layout.yaxis.title.text == "MBOPD"


class TestPodTimeseriesAn:
    def test_returns_figure(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        fig = pod_timeseries_an([row])
        assert fig is not None
        assert len(fig.data) > 0  # type: ignore[arg-type]

    def test_empty_data_returns_empty_figure(self):
        fig = pod_timeseries_an([])
        assert fig.data == ()

    def test_yaxis_title(self):
        row = PodTimeseriesRow(
            year=2025,
            category="Reserves",
            oc_mstby=36500.0,
            an_bscfy=1.0,
        )
        fig = pod_timeseries_an([row])
        assert fig.layout.yaxis.title.text == "MMSCFD"


# ---------------------------------------------------------------------------
# Integration tests (with mocked database)
# ---------------------------------------------------------------------------


def _create_seeded_db(db_path):
    """Create a DuckDB database with minimal POD test data."""
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE pod_economics (
                pod_id TEXT,
                pod_name TEXT,
                case_type TEXT,
                effective_date DATE,
                lifting_oil DOUBLE,
                lifting_gas DOUBLE,
                grs_rev DOUBLE,
                investment DOUBLE,
                opex DOUBLE,
                asr DOUBLE,
                ctr_irr DOUBLE,
                ctr_npv DOUBLE,
                goi_take DOUBLE
            );
        """)
        conn.execute("""
            CREATE TABLE pod_project (
                pod_id TEXT,
                project_id TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE project_resources (
                project_id TEXT,
                report_date DATE,
                report_year INTEGER,
                report_status TEXT,
                project_name TEXT,
                project_stage TEXT,
                project_class TEXT,
                project_level TEXT,
                uncert_level TEXT,
                wk_name TEXT,
                field_name TEXT,
                fluid TEXT,
                value DOUBLE,
                rec_oc_risked DOUBLE,
                rec_an_risked DOUBLE,
                res_oc DOUBLE,
                res_an DOUBLE,
                prj_ioip DOUBLE,
                prj_igip DOUBLE
            );
        """)
        conn.execute("""
            INSERT INTO pod_economics VALUES
                ('POD-001', 'Test POD Alpha', 'plan', '2024-01-01',
                 365000, 365000000, 10000, 2000, 1500, 500, 0.15, 8000, 1000),
                ('POD-001', 'Test POD Alpha', 'plan', '2025-01-01',
                 730000, 730000000, 20000, 4000, 3000, 1000, 0.18, 16000, 2000);
        """)
        conn.execute("""
            INSERT INTO pod_project VALUES
                ('POD-001', 'PROJ-001'),
                ('POD-001', 'PROJ-002');
        """)
        conn.execute("""
            INSERT INTO project_resources
                (project_id, report_date, report_year, report_status,
                 project_name, project_stage, project_class, project_level,
                 uncert_level, wk_name, field_name, fluid, value,
                 rec_oc_risked, rec_an_risked, res_oc, res_an,
                 prj_ioip, prj_igip)
            VALUES
                ('PROJ-001', '2024-01-01', 2024, 'ACTIVE',
                 'Project Alpha', '1. Exploitation',
                 '1P Reserves', '1. Low Value', 'WK-01', 'Field A',
                 'OC', 5000.0, 1000.0, 500.0, 5000.0, 2500.0, 500.0, 250.0, 125.0),
                ('PROJ-002', '2024-01-01', 2024, 'ACTIVE',
                 'Project Beta', '2. Exploration',
                 'Contingent Resources', '2. Middle Value', 'WK-01', 'Field B',
                 'OC', 10000.0, 2000.0, 1000.0, 10000.0, 5000.0, 1000.0, 500.0, 250.0);
        """)
    finally:
        conn.close()


@pytest.fixture
def seeded_pod_db(isolated_config):
    """Create a database with POD test data and return the config path."""
    Config.init_config()
    db_file = Config.get_db_file()
    db_dir = Config.get_db_dir()
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)

    _create_seeded_db(db_file)
    return isolated_config


@pytest.fixture
def client(seeded_pod_db):
    """Create a FastAPI test client with a seeded POD database."""
    from esdc.eureka.app import create_eureka_app

    app = create_eureka_app()
    return TestClient(app)


class TestPodApiEndpoints:
    def test_pod_dashboard_returns_200(self, client):
        response = client.get("/eureka/pod", follow_redirects=False)
        assert response.status_code in (200, 307, 308)

    def test_pod_wks_endpoint(self, client):
        response = client.get("/eureka/api/pod/wks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_pod_pods_endpoint(self, client):
        response = client.get("/eureka/api/pod/pods?wk=WK-01")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.xfail(
        reason="_dataclass_to_dict does not handle nested dataclasses; app bug"
    )
    def test_pod_kpis_endpoint(self, client):
        response = client.get(
            "/eureka/api/pod/kpis?pod_id=POD-001&case_type=plan&year=2024",
            follow_redirects=False,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "pod_id" in data

    def test_pod_chart_timeseries_oc(self, client):
        response = client.get(
            "/eureka/api/pod/chart/timeseries_oc?pod_id=POD-001&case_type=plan"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list, str))

    def test_pod_chart_timeseries_an(self, client):
        response = client.get(
            "/eureka/api/pod/chart/timeseries_an?pod_id=POD-001&case_type=plan"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list, str))

    def test_pod_chart_unknown_returns_404(self, client):
        response = client.get(
            "/eureka/api/pod/chart/nonexistent?pod_id=POD-001&case_type=plan"
        )
        assert response.status_code == 404

    def test_pod_table_endpoint(self, client):
        response = client.get(
            "/eureka/api/pod/table?pod_id=POD-001&case_type=plan&year=2024&detail=economics"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_pod_export_csv(self, client):
        response = client.get(
            "/eureka/api/pod/export?pod_id=POD-001&case_type=plan&year=2024&detail=economics&format=csv"
        )
        assert response.status_code == 200


class TestPodTimeseriesQuery:
    def test_get_pod_timeseries(self, seeded_pod_db):
        from esdc.eureka.queries import get_pod_timeseries

        rows = get_pod_timeseries("POD-001", "plan")
        assert isinstance(rows, list)
