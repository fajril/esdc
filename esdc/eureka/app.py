"""Eureka FastAPI application — Resources Knowledge Pages dashboard."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import markdown as md
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from esdc.eureka.charts.nkri import (
    nkri_onstream_chart,
    nkri_timeseries_an,
    nkri_timeseries_oc,
)
from esdc.eureka.queries import (
    get_available_years,
    get_field_kpis,
    get_inplace_kpis,
    get_latest_year,
    get_nkri_kpis,
    get_nkri_summary,
    get_nkri_table_data,
    get_nkri_timeseries,
    get_onstream_data,
    get_project_kpis,
    get_wk_kpis,
)

logger = logging.getLogger("esdc.eureka")

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _markdown_filter(text: str) -> str:
    """Convert Markdown to HTML, supporting LaTeX via pymdownx.arithmatex."""
    if not text:
        return ""
    return md.markdown(
        text,
        extensions=["pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )


templates.env.filters["markdown"] = _markdown_filter

# Default year from CLI (set by run_eureka)
_default_year: int | None = None


def create_eureka_app(default_year: int | None = None) -> FastAPI:
    """Create and configure the Eureka FastAPI application.

    Args:
        default_year: Default report year to pre-select in the dashboard.
    """
    app = FastAPI(
        title="ESDC Eureka",
        description="Resources Knowledge Pages Dashboard",
        version="0.1.0",
    )

    # --- Exception handler (consistent with esdc serve) ---
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle generic exceptions with full context."""
        error_msg = f"Unhandled exception: {type(exc).__name__}: {str(exc)}"
        stack_trace = traceback.format_exc()

        logger.error(f"[EXCEPTION] {error_msg}")
        logger.error(f"[STACK TRACE]\n{stack_trace}")

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "detail": "Check server logs for full traceback",
                }
            },
        )

    # --- Redirects ---
    @app.get("/")
    async def root_redirect() -> RedirectResponse:
        """Redirect root to Eureka dashboard."""
        return RedirectResponse(url="/eureka/", status_code=307)

    @app.get("/eureka")
    async def slash_eureka_redirect() -> RedirectResponse:
        """Redirect /eureka -> /eureka/ (trailing slash)."""
        return RedirectResponse(url="/eureka/", status_code=307)

    # --- Dashboard page ---
    @app.get("/eureka/", response_class=HTMLResponse)
    async def dashboard(request: Request, year: int | None = None) -> HTMLResponse:
        """Main NKRI dashboard page."""
        try:
            years = get_available_years()
        except FileNotFoundError:
            return HTMLResponse(
                "<h1>Database not found</h1><p>Run <code>esdc fetch</code> first.</p>",
                status_code=500,
            )

        if not years:
            return HTMLResponse(
                "<h1>No data available</h1><p>No report years found in database.</p>",
                status_code=404,
            )

        # Use CLI default_year if no query param given
        if year is None:
            year = default_year
        selected_year = year if (year and year in years) else years[0]
        prev_year = _prev_year(years, selected_year)

        kpis = get_nkri_kpis(selected_year)
        wk = get_wk_kpis(selected_year)
        fields = get_field_kpis(selected_year)
        projects = get_project_kpis(selected_year)
        inplace = get_inplace_kpis(selected_year)
        nkri_summary = get_nkri_summary(selected_year)

        # Generate charts
        ts_data = get_nkri_timeseries(selected_year)
        onstream_data = get_onstream_data(selected_year)
        ts_oc_fig = nkri_timeseries_oc(ts_data)
        ts_an_fig = nkri_timeseries_an(ts_data)
        onstream_fig = nkri_onstream_chart(onstream_data, selected_year)

        ctx = {
            "years": years,
            "selected_year": selected_year,
            "prev_year": prev_year,
            "kpis": kpis,
            "wk": wk,
            "fields": fields,
            "projects": projects,
            "inplace": inplace,
            "nkri_summary": nkri_summary,
            "ts_oc_fig": ts_oc_fig,
            "ts_an_fig": ts_an_fig,
            "onstream_fig": onstream_fig,
        }
        return templates.TemplateResponse(request, "nkri.html", ctx)

    # --- API Routes ---
    @app.get("/eureka/api/nkri/kpis")
    async def nkri_kpis(year: int | None = None) -> JSONResponse:
        """API endpoint for NKRI KPI card data."""
        try:
            kpis = get_nkri_kpis(year)
            wk = get_wk_kpis(year or get_latest_year())
            fields = get_field_kpis(year or get_latest_year())
            projects = get_project_kpis(year or get_latest_year())
            return JSONResponse(
                {
                    "kpis": _dataclass_to_dict(kpis),
                    "wk": _dataclass_to_dict(wk),
                    "fields": _dataclass_to_dict(fields),
                    "projects": _dataclass_to_dict(projects),
                }
            )
        except FileNotFoundError:
            return JSONResponse({"error": "Database not found"}, status_code=500)

    @app.get("/eureka/api/nkri/chart/{chart_name}")
    async def nkri_chart(chart_name: str, year: int | None = None) -> JSONResponse:
        """API endpoint for NKRI chart data as Plotly JSON."""
        try:
            years = get_available_years()
            selected_year = year if year and year in years else years[0]

            chart_map = {
                "timeseries_oc": lambda _: nkri_timeseries_oc(
                    get_nkri_timeseries(selected_year)
                ),
                "timeseries_an": lambda _: nkri_timeseries_an(
                    get_nkri_timeseries(selected_year)
                ),
                "onstream": lambda _: nkri_onstream_chart(
                    get_onstream_data(selected_year), selected_year
                ),
            }

            if chart_name not in chart_map:
                return JSONResponse(
                    {"error": f"Unknown chart: {chart_name}"},
                    status_code=404,
                )

            fig = chart_map[chart_name](None)
            return JSONResponse(fig.to_json(), media_type="application/json")
        except FileNotFoundError:
            return JSONResponse({"error": "Database not found"}, status_code=500)

    @app.get("/eureka/api/nkri/table")
    async def nkri_table(
        year: int | None = None, detail: str = "resources"
    ) -> JSONResponse:
        """API endpoint for NKRI data table."""
        try:
            data = get_nkri_table_data(year, detail)
            return JSONResponse(data)
        except FileNotFoundError:
            return JSONResponse({"error": "Database not found"}, status_code=500)

    @app.get("/eureka/api/nkri/export", response_model=None)
    async def nkri_export(
        year: int | None = None,
        format: str = Query("csv", alias="format"),
    ):
        """Export NKRI data as CSV or Excel."""
        import io

        import pandas as pd

        try:
            data = get_nkri_table_data(year)
            df = pd.DataFrame(data)

            if format == "csv":
                output = io.StringIO()
                df.to_csv(output, index=False)
                fname = f"nkri_resources_{year or 'latest'}.csv"
                return HTMLResponse(
                    content=output.getvalue(),
                    media_type="text/csv",
                    headers={"Content-Disposition": (f"attachment; filename={fname}")},
                )
            else:
                output = io.BytesIO()
                df.to_excel(output, index=False, sheet_name="NKRI Resources")
                fname = f"nkri_resources_{year or 'latest'}.xlsx"
                return HTMLResponse(
                    content=output.getvalue(),
                    media_type=(
                        "application/"
                        "vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    headers={"Content-Disposition": (f"attachment; filename={fname}")},
                )
        except FileNotFoundError:
            return JSONResponse({"error": "Database not found"}, status_code=500)

    # Mount static files AFTER routes to prevent path conflicts
    if STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="static",
        )

    return app


def run_eureka(
    host: str = "0.0.0.0",
    port: int = 2030,
    log_level: str = "info",
    year: int | None = None,
) -> None:
    """Run the Eureka dashboard server."""
    from esdc.configs import Config

    Config.init_config()

    # Pass default_year to the app factory so it wires to dashboard
    app = create_eureka_app(default_year=year)

    logger.info(f"Starting Eureka dashboard on http://{host}:{port}/eureka/")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def _prev_year(years: list[int], current: int) -> int | None:
    """Find the previous year in the sorted years list."""
    sorted_years = sorted(years)
    idx = sorted_years.index(current) if current in sorted_years else -1
    if idx > 0:
        return sorted_years[idx - 1]
    return None


def _dataclass_to_dict(obj: object) -> dict | object:
    """Convert a dataclass instance to a dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj
