"""POD registry portal — Excel-like editor over the SQLite registry."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from esdc.pod_registry.changesets import apply_changeset
from esdc.pod_registry.publish import publish_pod_registry
from esdc.pod_registry.store import get_sqlite_connection
from esdc.portal.tables import TABLE_CONFIGS

logger = logging.getLogger(__name__)

_M_POD_ROWS_SQL = """
SELECT m.*,
    (SELECT group_concat(predecessor_id, '; ') FROM pod_revision
      WHERE successor_id = m.pod_id) AS preceded_by,
    (SELECT group_concat(successor_id, '; ') FROM pod_revision
      WHERE predecessor_id = m.pod_id) AS superseded_by
FROM m_pod m ORDER BY m.approval_seq
"""

# r_institution/r_pod_type grid columns use "institution"/"pod_type" as the
# editable field name, but the changeset service's insert/update contract for
# these tables expects the key "name" (see pod_registry/changesets.py
# _apply_inserts / _apply_updates for r_institution and r_pod_type).
_R_TABLE_NAME_FIELD = {"r_institution": "institution", "r_pod_type": "pod_type"}


def _normalize_r_table_rows(table: str, rows: list[dict]) -> list[dict]:
    """Rename the grid's institution/pod_type field to "name" for apply_changeset."""
    field = _R_TABLE_NAME_FIELD.get(table)
    if field is None:
        return rows
    normalized = []
    for row in rows:
        row = dict(row)
        if field in row:
            row["name"] = row.pop(field)
        normalized.append(row)
    return normalized


def _fetch_rows(table: str) -> list[dict]:
    conn = get_sqlite_connection()
    try:
        sql = _M_POD_ROWS_SQL if table == "m_pod" else f"SELECT * FROM {table}"
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def _fetch_refs() -> dict:
    conn = get_sqlite_connection()
    try:
        return {
            "institutions": [dict(r) for r in conn.execute(
                "SELECT code, institution FROM r_institution ORDER BY code")],
            "pod_types": [dict(r) for r in conn.execute(
                "SELECT code, pod_type FROM r_pod_type ORDER BY code")],
            "pods": [dict(r) for r in conn.execute(
                "SELECT id, pod_id, pod_name FROM m_pod ORDER BY approval_seq")],
            "pod_ids": [r["pod_id"] for r in conn.execute(
                "SELECT pod_id FROM m_pod ORDER BY approval_seq")],
        }
    finally:
        conn.close()


# project/document lookups back autocomplete datalists, save-time
# validation, and the grid's id — name cell formatters; opening DuckDB
# + scanning on every request makes typing stall, so cache the id→name
# maps briefly (keyed by db path so tests with distinct tmp dirs don't
# share entries).
_PROJECT_IDS_TTL_S = 60.0
_projects_cache: dict[str, tuple[float, dict[str, str] | None]] = {}
_documents_cache: dict[str, tuple[float, dict[str, str] | None]] = {}

# publish rewrites the DuckDB snapshot tables; serialize concurrent requests
# (save-triggered publish racing the Publish button) instead of letting two
# writers collide.
_publish_lock = threading.Lock()


def _known_projects() -> dict[str, str] | None:
    """Project id → name map from DuckDB; None if unavailable."""
    from esdc.configs import Config

    db_path = Config.get_db_file()
    cached = _projects_cache.get(str(db_path))
    if cached is not None and time.monotonic() - cached[0] < _PROJECT_IDS_TTL_S:
        return cached[1]
    projects = _load_projects(db_path)
    _projects_cache[str(db_path)] = (time.monotonic(), projects)
    return projects


def _known_project_ids() -> set[str] | None:
    """Project ids for typo checking; None if unavailable."""
    projects = _known_projects()
    return set(projects) if projects is not None else None


def _load_projects(db_path) -> dict[str, str] | None:
    return _load_id_name_map(
        db_path,
        "SELECT project_id, any_value(project_name)"
        " FROM project_resources GROUP BY project_id",
        "project id validation unavailable",
    )


def _known_documents() -> dict[str, str] | None:
    """Corpus doc_id → file_name map from DuckDB; None if unavailable."""
    from esdc.configs import Config

    db_path = Config.get_db_file()
    cached = _documents_cache.get(str(db_path))
    if cached is not None and time.monotonic() - cached[0] < _PROJECT_IDS_TTL_S:
        return cached[1]
    documents = _load_id_name_map(
        db_path,
        "SELECT doc_id, file_name FROM documents",
        "corpus doc id validation unavailable",
    )
    _documents_cache[str(db_path)] = (time.monotonic(), documents)
    return documents


def _known_doc_ids() -> set[str] | None:
    """Corpus doc ids for typo checking; None if unavailable."""
    documents = _known_documents()
    return set(documents) if documents is not None else None


def _load_id_name_map(db_path, sql: str, warn: str) -> dict[str, str] | None:
    from esdc.dbmanager import get_duckdb_connection

    if not Path(db_path).exists():
        return None
    try:
        conn = get_duckdb_connection(db_path, read_only=True)
        try:
            rows = conn.execute(sql).fetchall()
            return {r[0]: r[1] or "" for r in rows}
        finally:
            conn.close()
    except Exception:
        logger.warning(warn, exc_info=True)
        return None


def create_portal_app() -> FastAPI:
    app = FastAPI(title="ESDC POD Portal")
    base = Path(__file__).resolve().parent
    app.mount("/static", StaticFiles(directory=base / "static"), name="static")
    templates = Jinja2Templates(directory=base / "templates")

    @app.get("/api/tables/{table}")
    def get_table(table: str):
        if table not in TABLE_CONFIGS:
            raise HTTPException(status_code=404, detail="unknown table")
        refs = _fetch_refs()
        if table == "project_pod":
            # id → name map for the grid's "id — name" project cell formatter
            refs["projects"] = _known_projects() or {}
        if table == "pod_document":
            # doc_id → file_name map for the documents datalist + formatter
            refs["documents"] = _known_documents() or {}
        return {"rows": _fetch_rows(table), "refs": refs}

    @app.post("/api/tables/{table}/save")
    async def save_table(table: str, request: Request):
        if table not in TABLE_CONFIGS:
            raise HTTPException(status_code=404, detail="unknown table")
        changes = await request.json()
        if table in _R_TABLE_NAME_FIELD:
            changes = dict(changes)
            changes["inserts"] = _normalize_r_table_rows(
                table, changes.get("inserts") or []
            )
            changes["updates"] = _normalize_r_table_rows(
                table, changes.get("updates") or []
            )
        known = _known_project_ids() if table == "project_pod" else None
        known_docs = _known_doc_ids() if table == "pod_document" else None
        result = apply_changeset(
            table, changes, known_project_ids=known, known_doc_ids=known_docs
        )
        payload = {
            "ok": result.ok,
            "applied": result.applied,
            "generated": result.generated,
            "errors": [asdict(e) for e in result.errors],
        }
        if not result.ok:
            return JSONResponse(status_code=422, content=payload)
        # Publish is NOT run here: it drops/recreates the DuckDB snapshot,
        # checkpoints, and invalidates chat caches — too slow to block the
        # save round-trip on. The client fires POST /api/publish right after
        # a successful save and surfaces any failure with a retry path.
        return payload

    @app.post("/api/publish")
    def publish():
        try:
            with _publish_lock:
                results = publish_pod_registry()
        except Exception as exc:  # sqlite state is intact — surface, allow retry
            logger.error("publish failed", exc_info=True)
            return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
        return {"ok": True, "tables": {r.table_name: r.row_count for r in results}}

    @app.get("/api/projects")
    def projects(q: str = ""):
        known = _known_projects()
        if not known:
            return []
        ql = q.lower()
        matches = sorted(
            (pid, name) for pid, name in known.items()
            if ql in pid.lower() or ql in name.lower()
        )
        return [
            {"value": pid, "label": f"{pid} — {name}" if name else pid}
            for pid, name in matches[:20]
        ]

    # HTML pages added in Task 8
    _register_pages(app, templates)
    return app


def _grid_payload(table: str) -> dict:
    cfg = TABLE_CONFIGS[table]
    return {"table": table, "title": cfg["title"],
            "config": {"table": table, **cfg}}


def _register_pages(app: FastAPI, templates: Jinja2Templates) -> None:
    @app.get("/")
    def root():
        return RedirectResponse("/pods")

    static_dir = Path(__file__).resolve().parent / "static"

    def _asset_version() -> int:
        # mtime-based cache buster: StaticFiles sends no Cache-Control, so
        # browsers heuristically cache portal.js/css and keep serving stale
        # code after an upgrade. A changed ?v= forces a refetch.
        return int(max(
            (static_dir / name).stat().st_mtime for name in ("portal.js", "portal.css")
        ))

    def _page(request: Request, title: str, tables: list[str]):
        return templates.TemplateResponse(
            request, "grid.html",
            {"title": title, "grids": [_grid_payload(t) for t in tables],
             "asset_version": _asset_version()},
        )

    @app.get("/pods")
    def pods(request: Request):
        return _page(request, "PODs", ["m_pod"])

    @app.get("/links")
    def links(request: Request):
        return _page(request, "Project Links", ["project_pod"])

    @app.get("/documents")
    def documents(request: Request):
        return _page(request, "Document Links", ["pod_document"])

    @app.get("/revisions")
    def revisions(request: Request):
        return _page(request, "Revisions", ["pod_revision"])

    @app.get("/references")
    def references(request: Request):
        return _page(request, "References", ["r_institution", "r_pod_type"])


def run_portal(
    host: str = "127.0.0.1", port: int = 13334, log_level: str = "info"
) -> None:
    from esdc.configs import Config

    Config.init_config()
    app = create_portal_app()
    logger.info(f"Starting POD portal on http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
