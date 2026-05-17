"""DuckDB query functions for Eureka dashboard data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from esdc.configs import Config
from esdc.dbmanager import get_duckdb_connection

logger = logging.getLogger(__name__)


# --- Classification normalization SQL fragments ---

PROJECT_STAGE_NORM = """
CASE
    WHEN project_stage IN ('1. Exploitation', 'Exploitation') THEN 'Exploitation'
    WHEN project_stage IN ('2. Exploration', 'Exploration') THEN 'Exploration'
    WHEN project_stage LIKE '%Abandoned%' THEN 'Abandoned'
    ELSE project_stage
END
"""

PROJECT_CLASS_NORM = """
CASE
    WHEN project_class LIKE '%Reserves%' THEN 'Reserves & GRR'
    WHEN project_class LIKE '%Contigent%'
        OR project_class LIKE '%Contingent%'
        THEN 'Contingent Resources'
    WHEN project_class LIKE '%Prospective%' THEN 'Prospective Resources'
    WHEN project_class LIKE '%Abandoned%' THEN 'Abandoned'
    ELSE project_class
END
"""

PROD_STAGE_NORM = """
CASE
    WHEN prod_stage IN ('1. Primary') THEN 'Primary'
    WHEN prod_stage IN ('2. Waterflood') THEN 'Waterflood'
    WHEN prod_stage IN ('3. EOR/EGR', 'EOR/GR') THEN 'EOR/EGR'
    ELSE prod_stage
END
"""


@dataclass(frozen=True)
class NKRIKpiData:
    """NKRI-level KPI card data."""

    # Reserves (MSTB / MSCF → divide by 1000 for display)
    res_1p_oc: float
    res_2p_oc: float
    res_3p_oc: float
    res_1p_an: float
    res_2p_an: float
    res_3p_an: float
    # GRR — total recoverable (rec_* at Reserves & GRR class)
    grr_1r_oc: float
    grr_1r_an: float
    grr_2r_oc: float
    grr_2r_an: float
    grr_3r_oc: float
    grr_3r_an: float
    # Contingent Exploitation (risked, 1C/2C/3C)
    cont_exploit_1c_oc: float
    cont_exploit_1c_an: float
    cont_exploit_2c_oc: float
    cont_exploit_2c_an: float
    cont_exploit_3c_oc: float
    cont_exploit_3c_an: float
    # Contingent Exploration (risked, 1C/2C/3C)
    cont_explore_1c_oc: float
    cont_explore_1c_an: float
    cont_explore_2c_oc: float
    cont_explore_2c_an: float
    cont_explore_3c_oc: float
    cont_explore_3c_an: float
    # Prospective (risked, 1U/2U/3U)
    prospective_1u_oc: float
    prospective_1u_an: float
    prospective_2u_oc: float
    prospective_2u_an: float
    prospective_3u_oc: float
    prospective_3u_an: float
    # Sales cumulative
    cumprod_sls_oc: float
    cumprod_sls_an: float
    # Yearly production
    yearly_sls_oc: float
    yearly_sls_an: float
    # YoY changes (percent)
    yoy_res_2p_oc: float | None = None
    yoy_res_2p_an: float | None = None
    yoy_grr_oc: float | None = None
    yoy_grr_an: float | None = None
    yoy_cont_exploit_oc: float | None = None
    yoy_cont_exploit_an: float | None = None
    yoy_cont_explore_oc: float | None = None
    yoy_cont_explore_an: float | None = None
    yoy_prospective_oc: float | None = None
    yoy_prospective_an: float | None = None


@dataclass(frozen=True)
class WKKpiData:
    """WK-level KPI counts."""

    total_wk: int
    exploit_wk: int
    production_wk: int
    development_wk: int
    exploration_wk: int


@dataclass(frozen=True)
class FieldKpiData:
    """Field-level KPI counts."""

    total_fields: int
    exploit_fields: int
    production_fields: int
    development_fields: int
    idle_fields: int
    exploration_fields: int
    discovered_fields: int
    undiscovered_fields: int


@dataclass(frozen=True)
class ProjectKpiData:
    """Project-level KPI counts."""

    total_projects: int
    exploit_projects: int
    primary_projects: int
    waterflood_projects: int
    eor_egr_projects: int
    exploration_projects: int


@dataclass(frozen=True)
class InPlaceKpiData:
    """In-Place (IOIP/IGIP) KPI data at P50 level."""

    disc_exploit_ioip: float
    disc_exploit_igip: float
    disc_explore_ioip: float
    disc_explore_igip: float
    undisc_risked_ioip: float
    undisc_risked_igip: float


@dataclass(frozen=True)
class NKRIResourcesRow:
    """One row of NKRI resources data (for chart/table)."""

    project_stage_norm: str
    project_class_norm: str
    uncert_level: str
    rec_oc: float
    rec_an: float
    res_oc: float
    res_an: float


@dataclass(frozen=True)
class NKRITimeseriesRow:
    """One row of NKRI timeseries forecast data."""

    year: int
    project_class: str
    project_level: str
    tpf_oc: float
    tpf_an: float
    slf_oc: float
    slf_an: float
    spf_oc: float
    spf_an: float
    tpf_risked_oc: float
    tpf_risked_an: float


@dataclass(frozen=True)
class OnstreamRow:
    """One row of onstream year project counts."""

    onstream_year: int
    project_class: str
    level_prefix: str
    project_count: int
    total_sales: float


def _get_conn(read_only: bool = True):
    """Get a DuckDB connection using the configured database path."""
    db_path = Config.get_db_file()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return get_duckdb_connection(db_path, read_only=read_only)


def get_available_years() -> list[int]:
    """Get distinct report years from nkri_resources, sorted descending."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT report_year FROM nkri_resources ORDER BY report_year DESC"
        ).fetchall()
        return [r[0] for r in rows if r[0] is not None]
    finally:
        conn.close()


def get_latest_year() -> int:
    """Get the latest report year."""
    years = get_available_years()
    if not years:
        raise ValueError("No report years found in nkri_resources")
    return years[0]


def _to_mmstb(val: float | None) -> float:
    """Convert MSTB → MMSTB (divide by 1000)."""
    if val is None:
        return 0.0
    return round(val / 1000, 0)


def _to_tscf(val: float | None) -> float:
    """Convert MSCF → TSCF (divide by 1000)."""
    if val is None:
        return 0.0
    return round(val / 1000, 0)


def _pct_change(current: float, previous: float) -> float | None:
    """Calculate percentage change, return None if previous is 0."""
    if previous == 0:
        return None
    return round(((current - previous) / previous) * 100, 1)


def get_nkri_kpis(year: int | None = None) -> NKRIKpiData:
    """Get NKRI-level KPI card data for a given year.

    If year is None, uses the latest available year.
    Also computes YoY change vs previous year.
    """
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        # --- Reserves (res_*) + GRR (rec_*) from nkri_resources ---
        class_res_sql = f"""
        SELECT
            uncert_level,
            SUM(res_oc) as sum_res_oc,
            SUM(res_an) as sum_res_an,
            SUM(rec_oc) as sum_rec_oc,
            SUM(rec_an) as sum_rec_an
        FROM nkri_resources
        WHERE {PROJECT_CLASS_NORM.strip()} = 'Reserves & GRR'
          AND report_year = {year}
        GROUP BY uncert_level
        """
        class_res_rows = conn.execute(class_res_sql).fetchall()
        res_map: dict[str, tuple[float, float]] = {}
        rec_map: dict[str, tuple[float, float]] = {}
        for r in class_res_rows:
            res_map[r[0]] = (r[1], r[2])
            rec_map[r[0]] = (r[3], r[4])

        res_1p = res_map.get("1. Low Value", (0, 0))
        res_2p = res_map.get("2. Middle Value", (0, 0))
        res_3p = res_map.get("3. High Value", (0, 0))
        rec_1p = rec_map.get("1. Low Value", (0, 0))
        rec_2p = rec_map.get("2. Middle Value", (0, 0))
        rec_3p = rec_map.get("3. High Value", (0, 0))

        # --- Contingent Exploitation (1C/2C/3C) from nkri_resources ---
        cont_exploit_sql = f"""
        SELECT
            uncert_level,
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM nkri_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND report_year = {year}
        GROUP BY uncert_level
        """
        cont_ex_rows = conn.execute(cont_exploit_sql).fetchall()
        cont_ex_map: dict[str, tuple[float, float]] = {}
        for r in cont_ex_rows:
            cont_ex_map[r[0]] = (r[1], r[2])
        cont_exploit_1c = cont_ex_map.get("1. Low Value", (0, 0))
        cont_exploit_2c = cont_ex_map.get("2. Middle Value", (0, 0))
        cont_exploit_3c = cont_ex_map.get("3. High Value", (0, 0))

        # --- Contingent Exploration (1C/2C/3C) from nkri_resources ---
        cont_explore_sql = f"""
        SELECT
            uncert_level,
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM nkri_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
          AND report_year = {year}
        GROUP BY uncert_level
        """
        cont_ex_rows = conn.execute(cont_explore_sql).fetchall()
        cont_exr_map: dict[str, tuple[float, float]] = {}
        for r in cont_ex_rows:
            cont_exr_map[r[0]] = (r[1], r[2])
        cont_explore_1c = cont_exr_map.get("1. Low Value", (0, 0))
        cont_explore_2c = cont_exr_map.get("2. Middle Value", (0, 0))
        cont_explore_3c = cont_exr_map.get("3. High Value", (0, 0))

        # --- Prospective (1U/2U/3U) from nkri_resources ---
        prospect_sql = f"""
        SELECT
            uncert_level,
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM nkri_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Prospective Resources'
          AND report_year = {year}
        GROUP BY uncert_level
        """
        prospect_rows = conn.execute(prospect_sql).fetchall()
        prospect_map: dict[str, tuple[float, float]] = {}
        for r in prospect_rows:
            prospect_map[r[0]] = (r[1], r[2])
        prospective_1u = prospect_map.get("1. Low Value", (0, 0))
        prospective_2u = prospect_map.get("2. Middle Value", (0, 0))
        prospective_3u = prospect_map.get("3. High Value", (0, 0))

        # --- Sales cumulative ---
        cumprod_sql = f"""
        SELECT
            SUM(COALESCE(cprd_sls_oc, 0)) as sum_oc,
            SUM(COALESCE(cprd_sls_an, 0)) as sum_an
        FROM nkri_resources
        WHERE report_year = {year}
          AND uncert_level = '2. Middle Value'
        """
        cumprod = conn.execute(cumprod_sql).fetchone()
        cumprod_sls_oc = cumprod[0] if cumprod else 0
        cumprod_sls_an = cumprod[1] if cumprod else 0

        # --- Yearly production rate ---
        yearly_sql = f"""
        SELECT
            SUM(COALESCE(rate_sls_oc, 0)) as sum_oc,
            SUM(COALESCE(rate_sls_an, 0)) as sum_an
        FROM nkri_resources
        WHERE report_year = {year}
          AND uncert_level = '2. Middle Value'
        """
        yearly = conn.execute(yearly_sql).fetchone()
        yearly_sls_oc = yearly[0] if yearly else 0
        yearly_sls_an = yearly[1] if yearly else 0

        # --- YoY: compute previous year values ---
        prev_years = [y for y in get_available_years() if y < year]
        prev_year = max(prev_years) if prev_years else None

        yoy_data = {}
        if prev_year is not None:
            yoy_data = _compute_yoy(
                conn,
                year,
                prev_year,
                res_map,
                rec_map,
                cont_exploit_2c[0],
                cont_exploit_2c[1],
                cont_explore_2c[0],
                cont_explore_2c[1],
                prospective_2u[0],
                prospective_2u[1],
            )

        return NKRIKpiData(
            res_1p_oc=_to_mmstb(res_1p[0]),
            res_2p_oc=_to_mmstb(res_2p[0]),
            res_3p_oc=_to_mmstb(res_3p[0]),
            res_1p_an=_to_tscf(res_1p[1]),
            res_2p_an=_to_tscf(res_2p[1]),
            res_3p_an=_to_tscf(res_3p[1]),
            grr_1r_oc=_to_mmstb(rec_1p[0]),
            grr_1r_an=_to_tscf(rec_1p[1]),
            grr_2r_oc=_to_mmstb(rec_2p[0]),
            grr_2r_an=_to_tscf(rec_2p[1]),
            grr_3r_oc=_to_mmstb(rec_3p[0]),
            grr_3r_an=_to_tscf(rec_3p[1]),
            cont_exploit_1c_oc=_to_mmstb(cont_exploit_1c[0]),
            cont_exploit_1c_an=_to_tscf(cont_exploit_1c[1]),
            cont_exploit_2c_oc=_to_mmstb(cont_exploit_2c[0]),
            cont_exploit_2c_an=_to_tscf(cont_exploit_2c[1]),
            cont_exploit_3c_oc=_to_mmstb(cont_exploit_3c[0]),
            cont_exploit_3c_an=_to_tscf(cont_exploit_3c[1]),
            cont_explore_1c_oc=_to_mmstb(cont_explore_1c[0]),
            cont_explore_1c_an=_to_tscf(cont_explore_1c[1]),
            cont_explore_2c_oc=_to_mmstb(cont_explore_2c[0]),
            cont_explore_2c_an=_to_tscf(cont_explore_2c[1]),
            cont_explore_3c_oc=_to_mmstb(cont_explore_3c[0]),
            cont_explore_3c_an=_to_tscf(cont_explore_3c[1]),
            prospective_1u_oc=_to_mmstb(prospective_1u[0]),
            prospective_1u_an=_to_tscf(prospective_1u[1]),
            prospective_2u_oc=_to_mmstb(prospective_2u[0]),
            prospective_2u_an=_to_tscf(prospective_2u[1]),
            prospective_3u_oc=_to_mmstb(prospective_3u[0]),
            prospective_3u_an=_to_tscf(prospective_3u[1]),
            cumprod_sls_oc=_to_mmstb(cumprod_sls_oc),
            cumprod_sls_an=_to_tscf(cumprod_sls_an),
            yearly_sls_oc=_to_mmstb(yearly_sls_oc),
            yearly_sls_an=_to_tscf(yearly_sls_an),
            **yoy_data,
        )
    finally:
        conn.close()


def _compute_yoy(
    conn,
    year: int,
    prev_year: int,
    curr_res_map: dict,
    curr_rec_map: dict,
    curr_ce_oc: float,
    curr_ce_an: float,
    curr_cx_oc: float,
    curr_cx_an: float,
    curr_pr_oc: float,
    curr_pr_an: float,
) -> dict:
    """Compute YoY percentage changes for KPIs."""
    # Reserves + GRR (combined query, 4 columns per uncert_level)
    prev_class_sql = f"""
    SELECT
        uncert_level,
        SUM(res_oc) as sum_res_oc,
        SUM(res_an) as sum_res_an,
        SUM(rec_oc) as sum_rec_oc,
        SUM(rec_an) as sum_rec_an
    FROM nkri_resources
    WHERE {PROJECT_CLASS_NORM.strip()} = 'Reserves & GRR'
      AND report_year = {prev_year}
    GROUP BY uncert_level
    """
    prev_rows = conn.execute(prev_class_sql).fetchall()
    prev_res_map: dict[str, tuple] = {}
    prev_rec_map: dict[str, tuple] = {}
    for r in prev_rows:
        prev_res_map[r[0]] = (r[1], r[2])
        prev_rec_map[r[0]] = (r[3], r[4])

    prev_res_2p = prev_res_map.get("2. Middle Value", (0, 0))
    prev_rec_2p = prev_rec_map.get("2. Middle Value", (0, 0))
    curr_res_2p = curr_res_map.get("2. Middle Value", (0, 0))
    curr_rec_2p = curr_rec_map.get("2. Middle Value", (0, 0))

    # Contingent Exploitation YoY
    prev_cont_exploit_sql = f"""
    SELECT
        SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
        SUM(COALESCE(rec_an_risked, 0)) as sum_an
    FROM nkri_resources
    WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
      AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
      AND uncert_level = '2. Middle Value'
      AND report_year = {prev_year}
    """
    prev_ce = conn.execute(prev_cont_exploit_sql).fetchone()
    prev_ce_oc = prev_ce[0] if prev_ce else 0
    prev_ce_an = prev_ce[1] if prev_ce else 0

    # Contingent Exploration YoY
    prev_cont_explore_sql = f"""
    SELECT
        SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
        SUM(COALESCE(rec_an_risked, 0)) as sum_an
    FROM nkri_resources
    WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
      AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
      AND uncert_level = '2. Middle Value'
      AND report_year = {prev_year}
    """
    prev_cx = conn.execute(prev_cont_explore_sql).fetchone()
    prev_cx_oc = prev_cx[0] if prev_cx else 0
    prev_cx_an = prev_cx[1] if prev_cx else 0

    # Prospective YoY
    prev_prospect_sql = f"""
    SELECT
        SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
        SUM(COALESCE(rec_an_risked, 0)) as sum_an
    FROM nkri_resources
    WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Prospective Resources'
      AND uncert_level = '2. Middle Value'
      AND report_year = {prev_year}
    """
    prev_pr = conn.execute(prev_prospect_sql).fetchone()
    prev_pr_oc = prev_pr[0] if prev_pr else 0
    prev_pr_an = prev_pr[1] if prev_pr else 0

    def _yoy_pct(current: float, prev: float, to_unit) -> float | None:
        return _pct_change(to_unit(current), to_unit(prev))

    return {
        "yoy_res_2p_oc": _yoy_pct(curr_res_2p[0], prev_res_2p[0], _to_mmstb),
        "yoy_res_2p_an": _yoy_pct(curr_res_2p[1], prev_res_2p[1], _to_tscf),
        "yoy_grr_oc": _yoy_pct(curr_rec_2p[0], prev_rec_2p[0], _to_mmstb),
        "yoy_grr_an": _yoy_pct(curr_rec_2p[1], prev_rec_2p[1], _to_tscf),
        "yoy_cont_exploit_oc": _yoy_pct(curr_ce_oc, prev_ce_oc, _to_mmstb),
        "yoy_cont_exploit_an": _yoy_pct(curr_ce_an, prev_ce_an, _to_tscf),
        "yoy_cont_explore_oc": _yoy_pct(curr_cx_oc, prev_cx_oc, _to_mmstb),
        "yoy_cont_explore_an": _yoy_pct(curr_cx_an, prev_cx_an, _to_tscf),
        "yoy_prospective_oc": _yoy_pct(curr_pr_oc, prev_pr_oc, _to_mmstb),
        "yoy_prospective_an": _yoy_pct(curr_pr_an, prev_pr_an, _to_tscf),
    }


def get_nkri_resources(
    year: int | None = None,
) -> list[NKRIResourcesRow]:
    """Get NKRI resources data grouped by classification."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        sql = f"""
        SELECT
            {PROJECT_STAGE_NORM.strip()} AS project_stage_norm,
            {PROJECT_CLASS_NORM.strip()} AS project_class_norm,
            uncert_level,
            SUM(COALESCE(rec_oc, 0)) as rec_oc,
            SUM(COALESCE(rec_an, 0)) as rec_an,
            SUM(COALESCE(res_oc, 0)) as res_oc,
            SUM(COALESCE(res_an, 0)) as res_an
        FROM nkri_resources
        WHERE report_year = {year}
        GROUP BY project_stage_norm, project_class_norm, uncert_level
        ORDER BY project_class_norm, project_stage_norm, uncert_level
        """
        rows = conn.execute(sql).fetchall()
        return [
            NKRIResourcesRow(
                project_stage_norm=r[0],
                project_class_norm=r[1],
                uncert_level=r[2],
                rec_oc=r[3],
                rec_an=r[4],
                res_oc=r[5],
                res_an=r[6],
            )
            for r in rows
        ]
    finally:
        conn.close()


def get_wk_kpis(year: int | None = None) -> WKKpiData:
    """Get Working Area KPI counts."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        # Total WK count from project_resources
        total_sql = f"""
        SELECT COUNT(DISTINCT wk_name) FROM project_resources
        WHERE report_year = {year} AND wk_name IS NOT NULL AND wk_name != ''
        """
        total_row = conn.execute(total_sql).fetchone()
        total = total_row[0] if total_row else 0

        # Exploitation WK count
        exploit_sql = f"""
        SELECT COUNT(DISTINCT wk_name) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND wk_name IS NOT NULL AND wk_name != ''
        """
        exploit_row = conn.execute(exploit_sql).fetchone()
        exploit = exploit_row[0] if exploit_row else 0

        # Production vs Development within Exploitation
        prod_dev_sql = f"""
        SELECT
            SUM(CASE WHEN total_sales > 0 THEN 1 ELSE 0 END)
                as production_wk,
            SUM(CASE WHEN total_sales = 0 THEN 1 ELSE 0 END)
                as development_wk
        FROM (
            SELECT wk_name,
                SUM(COALESCE(cprd_sls_oc, 0)
                    + COALESCE(cprd_sls_an, 0)) as total_sales
            FROM project_resources
            WHERE report_year = {year}
              AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
              AND wk_name IS NOT NULL AND wk_name != ''
            GROUP BY wk_name
        )
        """
        prod_dev = conn.execute(prod_dev_sql).fetchone()
        production = prod_dev[0] if prod_dev else 0
        development = prod_dev[1] if prod_dev else 0

        exploration = total - exploit

        return WKKpiData(
            total_wk=total,
            exploit_wk=exploit,
            production_wk=production,
            development_wk=development,
            exploration_wk=exploration,
        )
    finally:
        conn.close()


def get_field_kpis(year: int | None = None) -> FieldKpiData:
    """Get Field KPI counts."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        # Exploitation breakdown by field maturity
        # Priority: idle (E4-E8 only) → development (E2/E3 only, no sales)
        # → production (catch-all)
        exploit_brk_sql = f"""
        SELECT
            SUM(CASE WHEN all_idle THEN 1 ELSE 0 END) as idle,
            SUM(CASE WHEN all_development AND NOT has_sales
                THEN 1 ELSE 0 END
            ) as development,
            SUM(CASE WHEN NOT all_idle
                AND NOT (all_development AND NOT has_sales)
                THEN 1 ELSE 0 END
            ) as production
        FROM (
            SELECT field_id,
                BOOL_AND(LEFT(project_level, 2)
                    IN ('E4','E5','E6','E7','E8')
                ) as all_idle,
                BOOL_AND(LEFT(project_level, 2)
                    IN ('E2','E3')
                ) as all_development,
                BOOL_OR(COALESCE(cprd_sls_oc, 0)
                    + COALESCE(cprd_sls_an, 0) > 0
                ) as has_sales
            FROM project_resources
            WHERE report_year = {year}
              AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
              AND field_id IS NOT NULL AND field_id != ''
            GROUP BY field_id
        )
        """
        brk = conn.execute(exploit_brk_sql).fetchone()
        idle = brk[0] if brk and brk[0] is not None else 0
        development = brk[1] if brk and brk[1] is not None else 0
        production = brk[2] if brk and brk[2] is not None else 0
        exploit = development + idle + production

        # Exploration fields split by class
        discovered_sql = f"""
        SELECT COUNT(DISTINCT field_id) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
          AND ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
          AND field_id IS NOT NULL AND field_id != ''
        """
        disc_row = conn.execute(discovered_sql).fetchone()
        discovered = disc_row[0] if disc_row else 0

        undiscovered_sql = f"""
        SELECT COUNT(DISTINCT field_id) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
          AND ({PROJECT_CLASS_NORM.strip()}) = 'Prospective Resources'
          AND field_id IS NOT NULL AND field_id != ''
        """
        undisc_row = conn.execute(undiscovered_sql).fetchone()
        undiscovered = undisc_row[0] if undisc_row else 0
        exploration = discovered + undiscovered

        total_sql = f"""
        SELECT COUNT(DISTINCT field_id) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) IN ('Exploitation', 'Exploration')
          AND field_id IS NOT NULL AND field_id != ''
        """
        total_row = conn.execute(total_sql).fetchone()
        total = total_row[0] if total_row else 0

        return FieldKpiData(
            total_fields=total,
            exploit_fields=exploit,
            production_fields=production,
            development_fields=development,
            idle_fields=idle,
            exploration_fields=exploration,
            discovered_fields=discovered,
            undiscovered_fields=undiscovered,
        )
    finally:
        conn.close()


def get_project_kpis(year: int | None = None) -> ProjectKpiData:
    """Get Project KPI counts."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        total_sql = f"""
        SELECT COUNT(*) FROM project_resources WHERE report_year = {year}
        """
        total_row = conn.execute(total_sql).fetchone()
        total = total_row[0] if total_row else 0

        exploit_sql = f"""
        SELECT COUNT(*) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
        """
        exploit_row = conn.execute(exploit_sql).fetchone()
        exploit = exploit_row[0] if exploit_row else 0

        primary_sql = f"""
        SELECT COUNT(*) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND ({PROD_STAGE_NORM.strip()}) = 'Primary'
        """
        primary_row = conn.execute(primary_sql).fetchone()
        primary = primary_row[0] if primary_row else 0

        waterflood_sql = f"""
        SELECT COUNT(*) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND ({PROD_STAGE_NORM.strip()}) = 'Waterflood'
        """
        waterflood_row = conn.execute(waterflood_sql).fetchone()
        waterflood = waterflood_row[0] if waterflood_row else 0

        eor_sql = f"""
        SELECT COUNT(*) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND ({PROD_STAGE_NORM.strip()}) = 'EOR/EGR'
        """
        eor_row = conn.execute(eor_sql).fetchone()
        eor = eor_row[0] if eor_row else 0

        exploration = total - exploit

        return ProjectKpiData(
            total_projects=total,
            exploit_projects=exploit,
            primary_projects=primary,
            waterflood_projects=waterflood,
            eor_egr_projects=eor,
            exploration_projects=exploration,
        )
    finally:
        conn.close()


def get_nkri_table_data(
    year: int | None = None,
    detail: str = "resources",
) -> list[dict]:
    """Get NKRI table data with optional detail level filter.

    Args:
        year: Report year (defaults to latest)
        detail: One of 'resources', 'reserves', 'inplace', 'cumprod', 'rate'
    """
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        select_cols = {
            "resources": f"""
                {PROJECT_STAGE_NORM.strip()} AS project_stage,
                {PROJECT_CLASS_NORM.strip()} AS project_class,
                uncert_level,
                SUM(COALESCE(rec_oc,0)) as rec_oc,
                SUM(COALESCE(rec_an,0)) as rec_an,
                SUM(COALESCE(rec_oc_risked,0)) as rec_oc_risked,
                SUM(COALESCE(rec_an_risked,0)) as rec_an_risked
            """,
            "reserves": f"""
                {PROJECT_STAGE_NORM.strip()} AS project_stage,
                {PROJECT_CLASS_NORM.strip()} AS project_class,
                uncert_level,
                SUM(COALESCE(res_oc,0)) as res_oc,
                SUM(COALESCE(res_an,0)) as res_an
            """,
            "inplace": f"""
                {PROJECT_STAGE_NORM.strip()} AS project_stage,
                {PROJECT_CLASS_NORM.strip()} AS project_class,
                SUM(COALESCE(ioip,0)) as ioip,
                SUM(COALESCE(igip,0)) as igip
            """,
            "cumprod": f"""
                {PROJECT_STAGE_NORM.strip()} AS project_stage,
                {PROJECT_CLASS_NORM.strip()} AS project_class,
                SUM(COALESCE(cprd_sls_oc,0)) as cprd_sls_oc,
                SUM(COALESCE(cprd_sls_an,0)) as cprd_sls_an
            """,
            "rate": f"""
                {PROJECT_STAGE_NORM.strip()} AS project_stage,
                {PROJECT_CLASS_NORM.strip()} AS project_class,
                SUM(COALESCE(rate_sls_oc,0)) as rate_sls_oc,
                SUM(COALESCE(rate_sls_an,0)) as rate_sls_an
            """,
        }

        cols = select_cols.get(detail, select_cols["resources"])

        group_clause = (
            ", uncert_level"
            if detail
            in (
                "resources",
                "reserves",
            )
            else ""
        )
        sql = f"""
        SELECT {cols}
        FROM nkri_resources
        WHERE report_year = {year}
        GROUP BY project_stage, project_class{group_clause}
        ORDER BY project_class, project_stage{group_clause}
        """
        rows = conn.execute(sql).fetchall()
        col_names = [desc[0] for desc in conn.description]
        return [dict(zip(col_names, row, strict=False)) for row in rows]
    finally:
        conn.close()


def get_inplace_kpis(year: int | None = None) -> InPlaceKpiData:
    """Get NKRI In-Place (IOIP/IGIP) KPIs at P50 level."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        # Discovered — Exploitation: Reserves & GRR + Contingent Resources
        disc_exploit_sql = f"""
        SELECT
            SUM(COALESCE(ioip, 0)) as ioip,
            SUM(COALESCE(igip, 0)) as igip
        FROM nkri_resources
        WHERE report_year = {year}
          AND uncert_level = '2. Middle Value'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND ({PROJECT_CLASS_NORM.strip()})
            IN ('Reserves & GRR', 'Contingent Resources')
        """
        de = conn.execute(disc_exploit_sql).fetchone()
        disc_exploit_ioip = de[0] if de else 0
        disc_exploit_igip = de[1] if de else 0

        # Discovered — Exploration: Contingent Resources (Exploration)
        disc_explore_sql = f"""
        SELECT
            SUM(COALESCE(ioip, 0)) as ioip,
            SUM(COALESCE(igip, 0)) as igip
        FROM nkri_resources
        WHERE report_year = {year}
          AND uncert_level = '2. Middle Value'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
          AND ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
        """
        dx = conn.execute(disc_explore_sql).fetchone()
        disc_explore_ioip = dx[0] if dx else 0
        disc_explore_igip = dx[1] if dx else 0

        # Undiscovered — Risked: Prospective Resources × gcf_total
        #   query project_resources directly (gcf_total only available there)
        undisc_sql = f"""
        SELECT
            SUM(COALESCE(prj_ioip, 0) * COALESCE(gcf_total, 0)) as ioip,
            SUM(COALESCE(prj_igip, 0) * COALESCE(gcf_total, 0)) as igip
        FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_CLASS_NORM.strip()}) = 'Prospective Resources'
          AND uncert_level = '2. Middle Value'
        """
        un = conn.execute(undisc_sql).fetchone()
        undisc_risked_ioip = un[0] if un else 0
        undisc_risked_igip = un[1] if un else 0

        return InPlaceKpiData(
            disc_exploit_ioip=_to_mmstb(disc_exploit_ioip),
            disc_exploit_igip=_to_tscf(disc_exploit_igip),
            disc_explore_ioip=_to_mmstb(disc_explore_ioip),
            disc_explore_igip=_to_tscf(disc_explore_igip),
            undisc_risked_ioip=_to_mmstb(undisc_risked_ioip),
            undisc_risked_igip=_to_tscf(undisc_risked_igip),
        )
    finally:
        conn.close()


def get_nkri_summary(year: int | None = None) -> str | None:
    """Get the NKRI-level AI-generated summary text for a given year.

    Returns the headline + executive_summary text from the summarizer,
    or None if no summary exists for that year.
    """
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT nkri_summary FROM nkri_resources"
            " WHERE report_year = ? AND nkri_summary IS NOT NULL LIMIT 1",
            [year],
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


def get_nkri_timeseries(
    year: int | None = None,
) -> list[NKRITimeseriesRow]:
    """Get NKRI timeseries forecast data from nkri_timeseries.

    Returns rows for forecast years from year+1 to year+31,
    grouped by year, project_class, and project_level.
    """
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        start_yr = year + 1
        end_yr = year + 31
        sql = f"""
        SELECT
            year as forecast_year,
            project_class,
            LEFT(project_level, 1) as lvl,
            SUM(COALESCE(tpf_oc, 0)) as tpf_oc,
            SUM(COALESCE(tpf_an, 0)) as tpf_an,
            SUM(COALESCE(slf_oc, 0)) as slf_oc,
            SUM(COALESCE(slf_an, 0)) as slf_an,
            SUM(COALESCE(spf_oc, 0)) as spf_oc,
            SUM(COALESCE(spf_an, 0)) as spf_an,
            SUM(COALESCE(tpf_risked_oc, 0)) as tpf_risked_oc,
            SUM(COALESCE(tpf_risked_an, 0)) as tpf_risked_an
        FROM nkri_timeseries
        WHERE report_year = {year}
          AND year BETWEEN {start_yr} AND {end_yr}
        GROUP BY forecast_year, project_class, lvl
        ORDER BY forecast_year
        """
        rows = conn.execute(sql).fetchall()
        return [
            NKRITimeseriesRow(
                year=r[0],
                project_class=r[1],
                project_level=r[2],
                tpf_oc=float(r[3]),
                tpf_an=float(r[4]),
                slf_oc=float(r[5]),
                slf_an=float(r[6]),
                spf_oc=float(r[7]),
                spf_an=float(r[8]),
                tpf_risked_oc=float(r[9]),
                tpf_risked_an=float(r[10]),
            )
            for r in rows
        ]
    finally:
        conn.close()


def get_onstream_data(
    year: int | None = None,
) -> list[OnstreamRow]:
    """Get onstream year project counts by class for a report year."""
    if year is None:
        year = get_latest_year()

    conn = _get_conn()
    try:
        sql = f"""
        SELECT
            onstream_year,
            {PROJECT_CLASS_NORM.strip()} as class,
            LEFT(project_level, 1) as lvl,
            COUNT(DISTINCT project_id) as cnt,
            SUM(COALESCE(cprd_sls_oc, 0)
                + COALESCE(cprd_sls_an, 0)) as total_sales
        FROM project_resources
        WHERE report_year = {year}
          AND onstream_year IS NOT NULL
          AND onstream_year > 1900
          AND onstream_year <= 2050
        GROUP BY onstream_year, class, lvl
        ORDER BY onstream_year
        """
        rows = conn.execute(sql).fetchall()
        return [
            OnstreamRow(
                onstream_year=r[0],
                project_class=r[1],
                level_prefix=r[2],
                project_count=r[3],
                total_sales=float(r[4] if r[4] else 0),
            )
            for r in rows
        ]
    finally:
        conn.close()
