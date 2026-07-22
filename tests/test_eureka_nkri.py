"""Tests for NKRI Eureka chart and query behavior."""

from __future__ import annotations

import duckdb

from esdc.configs import Config
from esdc.eureka.charts.nkri import _agg_timeseries_an, _agg_timeseries_oc
from esdc.eureka.queries import NKRITimeseriesRow, get_nkri_timeseries


def test_agg_timeseries_oc_uses_spf_for_sales_potential_resources():
    row = NKRITimeseriesRow(
        year=2025,
        project_class="Reserves",
        project_level="E1. Production on Hold",
        tpf_oc=100.0,
        tpf_an=0.0,
        slf_oc=200.0,
        slf_an=0.0,
        spf_oc=365.0,
        spf_an=0.0,
        tpf_risked_oc=0.0,
        tpf_risked_an=0.0,
    )

    years, values = _agg_timeseries_oc([row])

    assert years == [2025]
    assert values["Reserves"] == [0.5]
    assert values["Sales Potential Resources"] == [1.0]


def test_agg_timeseries_an_uses_spf_for_sales_potential_resources():
    row = NKRITimeseriesRow(
        year=2025,
        project_class="Reserves",
        project_level="E1. Production on Hold",
        tpf_oc=0.0,
        tpf_an=100.0,
        slf_oc=0.0,
        slf_an=200.0,
        spf_oc=0.0,
        spf_an=0.365,
        tpf_risked_oc=0.0,
        tpf_risked_an=0.0,
    )

    years, values = _agg_timeseries_an([row])

    assert years == [2025]
    assert values["Reserves"] == [547.9]
    assert values["Sales Potential Resources"] == [1.0]


def test_get_nkri_timeseries_returns_spf_fields(tmp_path, monkeypatch):
    config_dir = tmp_path / ".esdc"
    db_file = config_dir / "esdc.duckdb"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_file))
    Config._config_cache = None
    Config.init_config()

    conn = duckdb.connect(str(db_file))
    conn.execute("""
        CREATE TABLE nkri_timeseries (
            report_year INTEGER,
            year INTEGER,
            project_class TEXT,
            project_level TEXT,
            tpf_oc REAL,
            tpf_an REAL,
            slf_oc REAL,
            slf_an REAL,
            spf_oc REAL,
            spf_an REAL,
            tpf_risked_oc REAL,
            tpf_risked_an REAL
        );
    """)
    conn.executemany(
        "INSERT INTO nkri_timeseries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                2024,
                2025,
                "Reserves",
                "E1. Production on Hold",
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
            ),
            (
                2024,
                2025,
                "Reserves",
                "E2. Under Development",
                None,
                None,
                None,
                None,
                50.0,
                60.0,
                None,
                None,
            ),
            (
                2024,
                2056,
                "Reserves",
                "E1. Production on Hold",
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ],
    )
    conn.close()

    rows = get_nkri_timeseries(2024)

    assert rows == [
        NKRITimeseriesRow(
            year=2025,
            project_class="Reserves",
            project_level="E",
            tpf_oc=100.0,
            tpf_an=200.0,
            slf_oc=300.0,
            slf_an=400.0,
            spf_oc=550.0,
            spf_an=660.0,
            tpf_risked_oc=700.0,
            tpf_risked_an=800.0,
        )
    ]
