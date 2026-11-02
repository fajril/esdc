"""Current POD editor: SQLite truth, bounded reads and explicit publication."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from esdc.configs import Config
from esdc.pod_registry.api_context import read_only_connection
from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.contract import load_pod_contract
from esdc.pod_registry.migrations import (
    MigrationRefusedError,
    _validate_existing_v2_registry,
)
from esdc.pod_registry.publication import get_pod_registry_status
from esdc.pod_registry.publish import publish_pod_registry
from esdc.pod_registry.store import get_esdc_sqlite_path, get_sqlite_connection
from esdc.portal.data import (
    iso_date,
    literal_contains,
    lookup_projects,
    read_reference_options,
    read_table_page,
    validate_page_size,
)
from esdc.portal.document_entities import (
    _refresh_mirror_after_save,
    apply_document_entity_changeset,
)
from esdc.portal.tables import TABLE_CONFIGS
from esdc.source_publication import (
    committed_pod_source_generation,
    get_fetch_publication_status,
    publish_committed_pod_sources,
)

logger = logging.getLogger(__name__)
_PUBLISH_LOCK = threading.Lock()
_COMPONENTS = {
    "oil": "Oil",
    "con": "Condensate",
    "ga": "Associated gas",
    "gn": "Non-associated gas",
    "oc": "Oil and condensate",
    "an": "All gas",
}
_REFERENCE_LABELS = {
    "pods": ("pod_id_seq", "pod_name"),
    "r_institution": ("id", "institution"),
    "r_pod_type": ("id", "pod_type"),
    "r_pod_stage": ("id", "stage_name"),
    "projects": ("value", "label"),
    "documents": ("value", "label"),
}


def _grid_payload(table: str) -> dict[str, Any]:
    config = copy.deepcopy(TABLE_CONFIGS[table])
    config.update(
        table=table,
        allowCreate=not config["readonly"] and table != "documents",
        allowDelete=not config["readonly"] and table != "documents",
    )
    for col in config["columns"]:
        if col.get("ref"):
            col["refValue"], col["refLabel"] = _REFERENCE_LABELS[col["ref"]]
        if table == "m_pod" and col["field"] == "rev_num":
            col["required"] = False
    config["defaults"] = {
        "m_project": {"is_active": True},
        "pod_project": {"is_main_project": False},
    }.get(table, {})
    config["creation_inputs"] = (
        [
            {
                "field": "predecessors",
                "title": "Superseded approvals",
                "ref": "pods",
                "refValue": "pod_id_seq",
                "refLabel": "pod_name",
                "type": "multiselect",
                "optional": True,
            }
        ]
        if table == "m_pod"
        else []
    )
    return {"table": table, "title": config["title"], "config": config}


def _chart_config() -> dict:
    columns = {
        c.name: c for c in load_pod_contract().tables["pod_forecast_api"].columns
    }
    return {
        "components": [
            {"value": key, "label": label, "unit": columns[f"usf_{key}_pod"].unit}
            for key, label in _COMPONENTS.items()
        ]
    }


@contextmanager
def _snapshot() -> Iterator[sqlite3.Connection]:
    path = get_esdc_sqlite_path()
    if not path.is_file():
        raise HTTPException(409, "Load a validated current POD dataset first")
    try:
        with closing(read_only_connection(path)) as conn:
            _validate_existing_v2_registry(conn)
            yield conn
    except (sqlite3.Error, MigrationRefusedError) as exc:
        raise HTTPException(409, f"Current POD registry required: {exc}") from exc


def _safe(value: Any) -> Any:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) > 2**53 - 1
    ):
        return str(value)
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _query(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        code = 409 if "POD registry" in str(exc) else 422
        raise HTTPException(code, str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(409, f"Current POD source unavailable: {exc}") from exc


def _expected(body: object) -> int:
    if not isinstance(body, dict):
        raise HTTPException(422, "Request body must be a JSON object")
    value = body.get("source_generation")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HTTPException(422, "source_generation must be an integer >= 0")
    return value


def _pending(
    code: int, generation: int | None, error: str, table: str | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "ok": False,
            "publication": "pending",
            "source_committed": generation is not None,
            "source_generation": generation,
            "error": error,
            "recovery_command": "esdc corpus sync"
            if table == "documents"
            else "esdc rebuild-api-analytics",
        },
    )


def _publish(generation: int, table: str | None = None):
    source, cache = get_esdc_sqlite_path(), Path(Config.get_db_file())
    # The lock lives entirely in a worker thread, never across an async await.
    with _PUBLISH_LOCK:
        current = committed_pod_source_generation(source)
        if current != generation:
            return _pending(
                409, current, "Source changed; reload before publishing", table
            )
        try:
            if table == "documents":
                warnings = _refresh_mirror_after_save(cache, source)
                if warnings:
                    return _pending(503, generation, "; ".join(warnings), table)
            else:
                with _snapshot() as conn:
                    has_raw = (
                        conn.execute(
                            "SELECT 1 FROM sqlite_master WHERE "
                            "type='table' AND name='api_snapshot'"
                        ).fetchone()
                        is not None
                        and conn.execute(
                            "SELECT 1 FROM api_snapshot WHERE "
                            "source_table='project_resources'"
                        ).fetchone()
                        is not None
                    )
                report = (
                    publish_committed_pod_sources(
                        source, cache, expected_generation=generation
                    )
                    if has_raw
                    else publish_pod_registry(source, cache)
                )
                error = getattr(report, "projection_error", None)
                if error:
                    return _pending(503, generation, str(error))
            current = committed_pod_source_generation(source)
            if current != generation:
                return _pending(
                    409,
                    current,
                    "Source changed during publication; publish current data",
                    table,
                )
            if table != "documents":
                status = get_pod_registry_status(source, cache)
                if status.state != "fresh" or status.published_generation != generation:
                    return _pending(
                        503,
                        generation,
                        status.message or "POD projection remains pending",
                    )
            return {
                "ok": True,
                "publication": "published",
                "source_generation": generation,
                "published_generation": generation,
                "scope": "documents" if table == "documents" else "pod",
            }
        except Exception as exc:
            logger.exception("Portal publication failed")
            return _pending(503, generation, str(exc), table)


def create_portal_app() -> FastAPI:
    app = FastAPI(title="ESDC POD Portal")
    base = Path(__file__).resolve().parent
    app.mount("/static", StaticFiles(directory=base / "static"), name="static")
    templates = Jinja2Templates(directory=base / "templates")

    @app.get("/")
    @app.get("/revisions")
    def root():
        return RedirectResponse("/pods", status_code=307)

    def page(request: Request, route: str):
        pages = {
            "pods": ("PODs", ["m_pod"]),
            "projects": ("Projects", ["m_project"]),
            "links": ("Project links", ["pod_project"]),
            "monitoring": ("Monitoring", ["pod_monitoring"]),
            "forecast": ("Forecast", ["pod_forecast"]),
            "results": ("Results", ["pod_monitoring_api", "pod_forecast_api"]),
            "references": (
                "References",
                ["r_institution", "r_pod_type", "r_pod_stage"],
            ),
            "documents": ("Documents", ["documents", "pod_document"]),
        }
        title, tables = pages[route]
        return templates.TemplateResponse(
            request,
            "grid.html",
            {
                "title": title,
                "grids": [_grid_payload(t) for t in tables],
                "chart": _chart_config() if route == "results" else None,
                "asset_version": int(
                    max(
                        (base / "static" / name).stat().st_mtime
                        for name in (
                            "portal.js",
                            "portal.css",
                            "editors.js",
                            "chart.js",
                            "results.js",
                        )
                    )
                ),
            },
        )

    for route in (
        "pods",
        "projects",
        "links",
        "monitoring",
        "forecast",
        "results",
        "references",
        "documents",
    ):

        def make_endpoint(route_name: str):
            def endpoint(request: Request):
                return page(request, route_name)

            return endpoint

        app.add_api_route(
            "/" + route, make_endpoint(route), methods=["GET"], name=route
        )

    @app.get("/api/tables/{table}")
    def get_table(
        table: str,
        page: int = 1,
        size: int = 100,
        filters: str | None = None,
        sort: str | None = None,
        direction: str = "asc",
    ):
        if table not in TABLE_CONFIGS:
            raise HTTPException(404, "Unknown table")
        try:
            parsed = None if filters is None else json.loads(filters)
        except ValueError as exc:
            raise HTTPException(422, "filters must be JSON") from exc
        _query(validate_page_size, page, size)
        if table == "documents" and get_esdc_sqlite_path().is_file():
            with _snapshot() as conn:
                if (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='documents'"
                    ).fetchone()
                    is None
                ):
                    return {
                        "rows": [],
                        "total": 0,
                        "page": page,
                        "size": size,
                        "last_page": 1,
                        "source_generation": conn.execute(
                            "SELECT source_generation FROM "
                            "pod_registry_state WHERE singleton_id=1"
                        ).fetchone()[0],
                        "initialized": True,
                        "available": False,
                    }
        return _query(
            read_table_page,
            table,
            page=page,
            size=size,
            filters=parsed,
            sort=sort,
            direction=direction,
        )

    @app.get("/api/references")
    def references():
        return _query(read_reference_options)

    @app.get("/api/projects")
    def projects(q: str = ""):
        rows = _query(lookup_projects, q=q)
        return [
            {
                "value": r["project_id"],
                "label": f"{r['project_id']} — {r['project_name'] or ''}"
                + (" (inactive)" if not r["is_active"] else ""),
            }
            for r in rows
        ]

    @app.get("/api/documents")
    def documents(q: str = ""):
        if len(q) > 200:
            raise HTTPException(422, "Search must be <= 200 characters")
        with _snapshot() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                    "name='documents'"
                ).fetchone()
                is None
            ):
                return []
            pattern = literal_contains(q)
            rows = conn.execute(
                "SELECT doc_id,file_name FROM documents WHERE doc_id "
                "LIKE ? ESCAPE '\\' OR file_name LIKE ? ESCAPE '\\' "
                "ORDER BY file_name,doc_id LIMIT 20",
                (pattern, pattern),
            )
            return [{"value": r[0], "label": f"{r[1] or ''} — {r[0]}"} for r in rows]

    @app.get("/api/status")
    def status():
        source, cache = get_esdc_sqlite_path(), Path(Config.get_db_file())
        try:
            return {
                "fetch": asdict(
                    get_fetch_publication_status(sqlite_path=source, duckdb_path=cache)
                ),
                "pod": asdict(get_pod_registry_status(source, cache)),
            }
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/tables/{table}/save")
    async def save(table: str, request: Request):
        if table not in TABLE_CONFIGS:
            raise HTTPException(404, "Unknown table")
        if TABLE_CONFIGS[table]["readonly"]:
            raise HTTPException(405, "Read-only table")
        try:
            changes = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "Request body must be JSON") from exc
        expected = _expected(changes)
        if table == "documents":
            current = await run_in_threadpool(
                committed_pod_source_generation, get_esdc_sqlite_path()
            )
            if current != expected:
                return JSONResponse(
                    status_code=409,
                    content={
                        "ok": False,
                        "conflict": True,
                        "source_committed": False,
                        "source_generation": current,
                        "errors": [
                            {
                                "kind": "conflict",
                                "index": -1,
                                "field": None,
                                "message": "Source changed; preserve this "
                                "draft and reload",
                            }
                        ],
                    },
                )
            result = await run_in_threadpool(apply_document_entity_changeset, changes)
            if result.source_generation is None:
                result.source_generation = current
        else:
            result = await run_in_threadpool(apply_changeset, table, changes)
        body = asdict(result)
        body["publication"] = (
            "published"
            if table == "documents" and result.ok and not result.warnings
            else "pending"
        )
        return JSONResponse(
            status_code=200 if result.ok else 409 if result.conflict else 422,
            content=_safe(body),
        )

    @app.post("/api/publish")
    async def publish(request: Request):
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "Request body must be JSON") from exc
        expected = _expected(body)
        table = body.get("table")
        if table is not None and table not in TABLE_CONFIGS:
            raise HTTPException(422, "Unknown publication scope")
        return await run_in_threadpool(_publish, expected, table)

    @app.get("/api/chart")
    def chart(pod_id_seq: int, reported_date: str, component: str):
        if component not in _COMPONENTS or not -(2**63) <= pod_id_seq < 2**63:
            raise HTTPException(422, "Invalid POD or component")
        reported_date = _query(iso_date, reported_date)
        with _snapshot() as conn:
            rows = conn.execute(
                f"SELECT year_frcast,usf_{component}_pod,usf_{component}_otlk "
                "FROM pod_forecast_api WHERE pod_id_seq=? AND reported_date=? "
                "ORDER BY year_frcast",
                (pod_id_seq, reported_date),
            )
            series = [
                {
                    "year": r[0],
                    "approved": None if r[1] is None else str(r[1]),
                    "outlook": None if r[2] is None else str(r[2]),
                }
                for r in rows
            ]
            gen = conn.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
        unit = next(
            c["unit"] for c in _chart_config()["components"] if c["value"] == component
        )
        return _safe(
            {
                "series": series,
                "unit": unit,
                "source_generation": gen,
                "pod_id_seq": pod_id_seq,
                "reported_date": reported_date,
            }
        )

    @app.get("/api/diagnostics")
    def diagnostics(
        pod_id_seq: int | None = None,
        reported_date: str | None = None,
        page: int = 1,
        size: int = 200,
    ):
        _query(validate_page_size, page, size)
        if pod_id_seq is not None and not -(2**63) <= pod_id_seq < 2**63:
            raise HTTPException(422, "Invalid POD")
        if reported_date is not None:
            reported_date = _query(iso_date, reported_date)
        with _snapshot() as conn:
            generation = conn.execute(
                "SELECT source_generation FROM pod_registry_state WHERE singleton_id=1"
            ).fetchone()[0]
            basis = conn.execute(
                "SELECT * FROM pod_read_model_snapshot WHERE singleton_id=1"
            ).fetchone()
            where, params = "source_generation=?", [generation]
            if pod_id_seq is not None:
                where += " AND (pod_id_seq IS NULL OR pod_id_seq=?)"
                params.append(pod_id_seq)
            if reported_date is not None:
                where += " AND (reported_date IS NULL OR reported_date=?)"
                params.append(reported_date)
            count = conn.execute(
                "SELECT COUNT(*) FROM pod_read_model_diagnostic WHERE " + where, params
            ).fetchone()[0]
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM pod_read_model_diagnostic WHERE "
                    + where
                    + " ORDER BY diagnostic_id LIMIT ? OFFSET ?",
                    [*params, size, (page - 1) * size],
                )
            ]
            return _safe(
                {
                    "source_generation": generation,
                    "basis": None if basis is None else dict(basis),
                    "total": count,
                    "page": page,
                    "size": size,
                    "last_page": max(1, (count + size - 1) // size),
                    "rows": rows,
                }
            )

    @app.get("/api/backup")
    def backup():
        from esdc.portal.backup import export_pod_backup

        temp = TemporaryDirectory(prefix="esdc-pod-backup-")
        try:
            output = export_pod_backup(
                get_esdc_sqlite_path(), Path(temp.name) / "pod.xlsx"
            )
        except Exception as exc:
            temp.cleanup()
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(
            output,
            filename="POD-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=BackgroundTask(temp.cleanup),
        )

    return app


def run_portal(
    host: str = "127.0.0.1", port: int = 13334, log_level: str = "info"
) -> None:
    Config.init_config()
    if not get_esdc_sqlite_path().exists():
        get_sqlite_connection().close()
    uvicorn.run(create_portal_app(), host=host, port=port, log_level=log_level)
