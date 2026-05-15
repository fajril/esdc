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
    # GRR — total recoverable (rec_* at Reserves & GRR class, 2P)
    grr_oc: float
    grr_an: float
    # Contingent Exploitation (risked, 2C)
    cont_exploit_oc: float
    cont_exploit_an: float
    # Contingent Exploration (risked, 2C)
    cont_explore_oc: float
    cont_explore_an: float
    # Prospective (risked, 2U)
    prospective_oc: float
    prospective_an: float
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
    exploration_fields: int


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
class NKRIResourcesRow:
    """One row of NKRI resources data (for chart/table)."""

    project_stage_norm: str
    project_class_norm: str
    uncert_level: str
    rec_oc: float
    rec_an: float
    res_oc: float
    res_an: float


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
        rec_2p = rec_map.get("2. Middle Value", (0, 0))

        # --- Contingent Exploitation (2C) from project_resources ---
        cont_exploit_sql = f"""
        SELECT
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM project_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND uncert_level = '2. Middle Value'
          AND report_year = {year}
        """
        cont_ex = conn.execute(cont_exploit_sql).fetchone()
        cont_exploit_oc = cont_ex[0] if cont_ex else 0
        cont_exploit_an = cont_ex[1] if cont_ex else 0

        # --- Contingent Exploration (2C) from project_resources ---
        cont_explore_sql = f"""
        SELECT
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM project_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Contingent Resources'
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploration'
          AND uncert_level = '2. Middle Value'
          AND report_year = {year}
        """
        cont_ex2 = conn.execute(cont_explore_sql).fetchone()
        cont_explore_oc = cont_ex2[0] if cont_ex2 else 0
        cont_explore_an = cont_ex2[1] if cont_ex2 else 0

        # --- Prospective (risked 2U) from project_resources ---
        prospect_sql = f"""
        SELECT
            SUM(COALESCE(rec_oc_risked, 0)) as sum_oc,
            SUM(COALESCE(rec_an_risked, 0)) as sum_an
        FROM project_resources
        WHERE ({PROJECT_CLASS_NORM.strip()}) = 'Prospective Resources'
          AND uncert_level = '2. Middle Value'
          AND report_year = {year}
        """
        prospect = conn.execute(prospect_sql).fetchone()
        prospective_oc = prospect[0] if prospect else 0
        prospective_an = prospect[1] if prospect else 0

        # --- Sales cumulative ---
        cumprod_sql = f"""
        SELECT
            SUM(COALESCE(cprd_sls_oc, 0)) as sum_oc,
            SUM(COALESCE(cprd_sls_an, 0)) as sum_an
        FROM project_resources
        WHERE report_year = {year}
        """
        cumprod = conn.execute(cumprod_sql).fetchone()
        cumprod_sls_oc = cumprod[0] if cumprod else 0
        cumprod_sls_an = cumprod[1] if cumprod else 0

        # --- Yearly production rate ---
        yearly_sql = f"""
        SELECT
            SUM(COALESCE(rate_sls_oc, 0)) as sum_oc,
            SUM(COALESCE(rate_sls_an, 0)) as sum_an
        FROM project_resources
        WHERE report_year = {year}
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
                conn, year, prev_year, res_map, rec_map,
                cont_exploit_oc, cont_exploit_an,
                cont_explore_oc, cont_explore_an,
                prospective_oc, prospective_an,
            )

        return NKRIKpiData(
            res_1p_oc=_to_mmstb(res_1p[0]),
            res_2p_oc=_to_mmstb(res_2p[0]),
            res_3p_oc=_to_mmstb(res_3p[0]),
            res_1p_an=_to_tscf(res_1p[1]),
            res_2p_an=_to_tscf(res_2p[1]),
            res_3p_an=_to_tscf(res_3p[1]),
            grr_oc=_to_mmstb(rec_2p[0]),
            grr_an=_to_tscf(rec_2p[1]),
            cont_exploit_oc=_to_mmstb(cont_exploit_oc),
            cont_exploit_an=_to_tscf(cont_exploit_an),
            cont_explore_oc=_to_mmstb(cont_explore_oc),
            cont_explore_an=_to_tscf(cont_explore_an),
            prospective_oc=_to_mmstb(prospective_oc),
            prospective_an=_to_tscf(prospective_an),
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
    FROM project_resources
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
    FROM project_resources
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
    FROM project_resources
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
        total_sql = f"""
        SELECT COUNT(DISTINCT field_id) FROM project_resources
        WHERE report_year = {year}
          AND field_id IS NOT NULL AND field_id != ''
        """
        total_row = conn.execute(total_sql).fetchone()
        total = total_row[0] if total_row else 0

        exploit_sql = f"""
        SELECT COUNT(DISTINCT field_id) FROM project_resources
        WHERE report_year = {year}
          AND ({PROJECT_STAGE_NORM.strip()}) = 'Exploitation'
          AND field_id IS NOT NULL AND field_id != ''
        """
        exploit_row = conn.execute(exploit_sql).fetchone()
        exploit = exploit_row[0] if exploit_row else 0
        exploration = total - exploit

        return FieldKpiData(
            total_fields=total,
            exploit_fields=exploit,
            exploration_fields=exploration,
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
