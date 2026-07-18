// POD Portal grid logic — built on Tabulator 6.3.1 (vendored, MIT license).
// One Tabulator instance per <section class="grid-section"> on the page,
// driven by the JSON config embedded in <script id="config-<table>">.

const gridState = {}; // table -> { table: Tabulator, refs, config, updates: Map, deletes: Map }

let activeGrid = null; // state of the grid the user last interacted with

// Fields the changeset API requires as real integers. Tabulator's "input"
// editor always yields a string, and "list" editor values keyed by an
// object always have string keys (JS coerces object keys to strings), so
// both need coercing back to Number before hitting the API.
const NUMERIC_FIELD_NAMES = new Set(["id", "rev_num", "code"]);
const NUMERIC_REFS = new Set(["institutions", "pod_types", "pods"]);

function pkKey(cfg, data) {
  return cfg.pk.map((k) => data[k]).join(" ");
}

function fieldConfig(cfg, field) {
  return cfg.columns.find((c) => c.field === field);
}

function isNumericField(cfg, field) {
  if (NUMERIC_FIELD_NAMES.has(field)) return true;
  const col = fieldConfig(cfg, field);
  return !!col && NUMERIC_REFS.has(col.ref);
}

// Excel copies dates in the sheet's display format — commonly day-first
// (dd/mm/yyyy) on Indonesian locales — while the changeset API wants ISO.
function normalizeDateInput(value) {
  if (typeof value !== "string") return value;
  const m = value.trim().match(/^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$/);
  if (m) return `${m[3]}-${m[2].padStart(2, "0")}-${m[1].padStart(2, "0")}`;
  return value;
}

function coerceValue(cfg, field, value) {
  const col = fieldConfig(cfg, field);
  if (col && col.editor === "date") return normalizeDateInput(value);
  if (isNumericField(cfg, field) && typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    if (!Number.isNaN(n)) return n;
  }
  return value;
}

function refLookup(refRows, codeField, labelField) {
  const map = {};
  for (const row of refRows) map[row[codeField]] = row[labelField];
  return map;
}

function buildColumn(col, refs) {
  const column = {
    field: col.field,
    title: col.required ? `${col.title} *` : col.title,
  };

  const dimmed = col.field === "preceded_by" || col.field === "superseded_by";
  if (dimmed) {
    column.formatter = (cell) => {
      const el = document.createElement("span");
      el.style.opacity = "0.55";
      el.textContent = cell.getValue() ?? "";
      return el;
    };
  }

  if (col.readonly) {
    column.editable = false;
    column.cssClass = "cell-readonly";
  } else if (col.editableOnNew) {
    column.editable = (cell) => cell.getRow().getData()._new === true;
    column.editor = "input"; // default; ref/date/autocomplete below override
  } else if (col.editable) {
    column.editor = "input";
  }

  if (col.editor === "date") {
    column.editor = "date";
  }

  if (col.ref === "institutions" || col.ref === "pod_types") {
    const rows = col.ref === "institutions" ? refs.institutions : refs.pod_types;
    const labelField = col.ref === "institutions" ? "institution" : "pod_type";
    const values = refLookup(rows, "code", labelField);
    column.editor = "list";
    column.editorParams = {
      values,
      autocomplete: true, // typing filters the options, Sheets-style
      listOnEmpty: true,
      placeholderEmpty: "No entries — add them on the References page",
    };
    column.formatter = (cell) => {
      const v = cell.getValue();
      return values[v] !== undefined ? `${v} — ${values[v]}` : v;
    };
  } else if (col.ref === "pods") {
    const values = {};
    for (const p of refs.pods) values[p.id] = `${p.pod_id} — ${p.pod_name}`;
    column.editor = "list";
    column.editorParams = {
      values,
      autocomplete: true,
      listOnEmpty: true,
      placeholderEmpty: "No PODs yet — add them on the PODs page",
    };
    column.formatter = (cell) => {
      const v = cell.getValue();
      return values[v] !== undefined ? values[v] : v;
    };
  } else if (col.ref === "pod_ids") {
    column.editor = "list";
    column.editorParams = { values: refs.pod_ids, autocomplete: true };
  }

  if (col.autocomplete) {
    const url = col.autocomplete;
    column.editor = "list";
    column.editorParams = {
      valuesLookup: async (_cell, filterTerm) => {
        // server returns [{value, label}] — label is "id — name"
        const resp = await fetch(`${url}?q=${encodeURIComponent(filterTerm || "")}`);
        return resp.ok ? await resp.json() : [];
      },
      autocomplete: true,
      freetext: true,
      filterRemote: true, // re-query per term; filterDelay debounces keystrokes
      filterDelay: 300,
    };
    const projects = refs.projects || {};
    column.formatter = (cell) => {
      const v = cell.getValue();
      return v && projects[v] ? `${v} — ${projects[v]}` : (v ?? "");
    };
  }

  if (!col.readonly) {
    column.cellEdited = (cell) => onCellEdited(cell);
  }

  return column;
}

function buildColumns(cfg, refs) {
  const columns = cfg.columns.map((col) => buildColumn(col, refs));
  columns.push({
    title: "",
    field: "_delete",
    width: 40,
    headerSort: false,
    formatter: () => "🗑",
    cellClick: (e, cell) => markRowDeleted(cell.getTable(), cell.getRow()),
  });
  return columns;
}

function onCellEdited(cell) {
  const table = cell.getTable();
  const state = table.pod_state;
  const data = cell.getRow().getData();
  cell.getElement().classList.remove("cell-error");
  cell.getRow().getElement().classList.remove("row-error");
  if (data._new) return; // inserts are captured wholesale on save
  const key = pkKey(state.cfg, data);
  const field = cell.getField();
  const value = coerceValue(state.cfg, field, cell.getValue());
  state.updates.set(key, { ...(state.updates.get(key) || {}), ...pkFields(state.cfg, data), [field]: value });
  persistState(state);
}

function pkFields(cfg, data) {
  const out = {};
  for (const k of cfg.pk) out[k] = data[k];
  return out;
}

// Fields a user can actually fill on a row (excludes readonly per config;
// preceded_by/superseded_by are readonly in tables.py so no extra set needed).
function rowEditableFields(cfg) {
  return cfg.columns.filter((c) => !c.readonly).map((c) => c.field);
}

// A _new row the user never touched — all editable fields blank.
function isBlankNewRow(cfg, data) {
  return rowEditableFields(cfg).every((f) => {
    const v = data[f];
    return v === undefined || v === null || String(v).trim() === "";
  });
}

// Empty grid gets one blank _new row so the user can type/paste immediately.
function seedIfEmpty(state) {
  if (state.table.getRows().length === 0) {
    state.table.addRow({ _new: true });
  }
}

// Can a pasted value be written into this column for this row?
// Existing rows: only always-editable columns (structural fields stay locked).
// New rows: editable or editableOnNew. Readonly columns are never written.
function isColWritable(col, isNew) {
  if (col.readonly) return false;
  if (isNew) return !!(col.editable || col.editableOnNew);
  return !!col.editable;
}

// Excel-style block paste: fill from the top-left of the current range
// selection across columns/rows, overwriting editable cells and auto-adding
// new rows for overflow. The anchor is resolved live at paste time from the
// range module, so sorts/reloads between click and paste cannot misdirect it.
async function handlePaste(e, state) {
  const clip = e.clipboardData || window.clipboardData;
  const text = clip ? clip.getData("text/plain") : "";
  if (!text) return;
  e.preventDefault();

  const lines = text.replace(/\r?\n$/, "").split(/\r?\n/).map((l) => l.split("\t"));
  const cfg = state.cfg;
  const cols = cfg.columns; // ordered data columns (excludes gutter/_delete)
  const tableRows = state.table.getRows(); // snapshot once — not per line

  let anchorRowPos = 0;
  let anchorColIndex = 0;
  const ranges = state.table.getRanges ? state.table.getRanges() : [];
  if (ranges && ranges.length) {
    const rg = ranges[0];
    const rgRows = rg.getRows();
    const rgCols = rg.getColumns();
    if (rgRows.length) {
      const pos = tableRows.indexOf(rgRows[0]);
      if (pos >= 0) anchorRowPos = pos;
    }
    if (rgCols.length) {
      const ci = cols.findIndex((c) => c.field === rgCols[0].getField());
      if (ci >= 0) anchorColIndex = ci;
    }
  }

  // Batch all mutations behind one redraw: per-row update/addRow otherwise
  // triggers a full redraw + column re-measure per pasted line.
  state.table.blockRedraw();
  try {
    const newRowsData = [];
    for (let r = 0; r < lines.length; r++) {
      const line = lines[r];
      const targetPos = anchorRowPos + r;
      const row = targetPos < tableRows.length ? tableRows[targetPos] : null;
      const isNew = row ? row.getData()._new === true : true;

      const patch = {};
      const updatedFields = {};
      for (let c = 0; c < line.length; c++) {
        const fi = anchorColIndex + c;
        if (fi >= cols.length) break; // pasted past the last column
        const col = cols[fi];
        if (!isColWritable(col, isNew)) continue;
        const val = coerceValue(cfg, col.field, line[c]);
        patch[col.field] = val;
        if (row && !isNew) updatedFields[col.field] = val;
      }

      if (!row) {
        newRowsData.push({ _new: true, ...patch });
      } else if (Object.keys(patch).length) {
        row.update(patch);
        if (!isNew) {
          const d = row.getData();
          const key = pkKey(cfg, d);
          state.updates.set(key, {
            ...(state.updates.get(key) || {}),
            ...pkFields(cfg, d),
            ...updatedFields,
          });
        }
      }
    }
    if (newRowsData.length) await state.table.addData(newRowsData);
  } finally {
    state.table.restoreRedraw();
  }
  persistState(state);
}

function rowFormatter(row) {
  const data = row.getData();
  const el = row.getElement();
  el.classList.toggle("row-new", data._new === true);
  el.classList.toggle("row-deleted", data._deleted === true);
}

// Insert payload: only fields declared in the grid config and not readonly.
// Readonly-per-config covers server-generated (pod_id/approval_seq on m_pod)
// and computed (preceded_by/superseded_by) fields per table — a field like
// pod_id that is readonly on m_pod but user-selected on project_pod is kept
// where the config says it is editable.
function stripForInsert(cfg, data) {
  const out = {};
  for (const [k, v] of Object.entries(data)) {
    if (k.startsWith("_")) continue;
    const col = fieldConfig(cfg, k);
    if (!col || col.readonly) continue;
    out[k] = coerceValue(cfg, k, v);
  }
  return out;
}

function buildChangeset(state) {
  const inserts = [];
  const insertRows = []; // parallel to inserts — maps server error index → row
  for (const row of state.table.getRows()) {
    const data = row.getData();
    if (data._new && !data._deleted && !isBlankNewRow(state.cfg, data)) {
      inserts.push(stripForInsert(state.cfg, data));
      insertRows.push(row);
    }
  }
  const updates = Array.from(state.updates.values());
  const deletes = Array.from(state.deletes.values());
  return { changeset: { inserts, updates, deletes }, insertRows };
}

// Staged changes are client-side only and each nav link is a full page load,
// so they are snapshotted to sessionStorage (per table) on every mutation and
// on pagehide, then restored after the grid rebuilds. A successful save
// clears the snapshot (reloadGrid).
const STORAGE_PREFIX = "podportal:";

function persistState(state) {
  const newRows = [];
  for (const row of state.table.getRows()) {
    const d = row.getData();
    if (d._new && !d._deleted && !isBlankNewRow(state.cfg, d)) newRows.push(d);
  }
  const key = STORAGE_PREFIX + state.cfg.table;
  if (!newRows.length && !state.updates.size && !state.deletes.size) {
    sessionStorage.removeItem(key);
    return;
  }
  sessionStorage.setItem(key, JSON.stringify({
    newRows,
    updates: Array.from(state.updates.entries()),
    deletes: Array.from(state.deletes.entries()),
  }));
}

function restoreState(state) {
  const raw = sessionStorage.getItem(STORAGE_PREFIX + state.cfg.table);
  if (!raw) return false;
  let saved;
  try {
    saved = JSON.parse(raw);
  } catch {
    return false;
  }
  const byPk = new Map(
    state.table.getRows().map((r) => [pkKey(state.cfg, r.getData()), r])
  );
  for (const [key, patch] of saved.updates || []) {
    const row = byPk.get(key);
    if (!row) continue; // row changed server-side since the snapshot
    row.update(patch);
    state.updates.set(key, patch);
  }
  for (const [key, pk] of saved.deletes || []) {
    const row = byPk.get(key);
    if (!row) continue;
    const d = row.getData();
    d._deleted = true;
    row.update(d);
    state.deletes.set(key, pk);
  }
  if ((saved.newRows || []).length) state.table.addData(saved.newRows);
  return !!(
    (saved.newRows || []).length ||
    (saved.updates || []).length ||
    (saved.deletes || []).length
  );
}

function hasPendingChanges() {
  return Object.values(gridState).some((state) => {
    if (state.updates.size || state.deletes.size) return true;
    return state.table.getRows().some((row) => {
      const data = row.getData();
      return data._new === true && !isBlankNewRow(state.cfg, data);
    });
  });
}

function markRowDeleted(table, row) {
  const data = row.getData();
  const state = table.pod_state;
  if (data._new) {
    row.delete();
    return;
  }
  data._deleted = true;
  row.update(data);
  const key = pkKey(state.cfg, data);
  state.deletes.set(key, pkFields(state.cfg, data));
  state.updates.delete(key);
  persistState(state);
}

function setStatus(msg, isError) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.color = isError ? "#b00020" : "";
}

async function fetchTableData(tableName) {
  const resp = await fetch(`/api/tables/${tableName}`);
  if (!resp.ok) throw new Error(`loading ${tableName} failed (HTTP ${resp.status})`);
  return await resp.json();
}

function buildGrid(tableName, cfg, payload) {
  const tableOptions = {
    data: payload.rows,
    columns: buildColumns(cfg, payload.refs),
    layout: "fitDataStretch",
    rowFormatter,
    // Data-entry order is server-defined (approval_seq); header sorting is
    // disabled so visual row order always matches data order — paste targets
    // and the row-number gutter stay truthful.
    columnDefaults: { headerSort: false },
    // Spreadsheet interaction model (Sheets/Excel convention): single click
    // selects a range, double click edits, gutter shows row numbers, Ctrl+C
    // copies the selected range, Ctrl+Z undoes.
    rowHeader: {
      title: "",
      field: "_rownum",
      formatter: "rownum",
      hozAlign: "center",
      frozen: true,
      width: 40,
      resizable: false,
      editable: false,
      headerSort: false,
    },
    selectableRange: 1,
    selectableRangeColumns: true,
    selectableRangeRows: true,
    editTriggerEvent: "dblclick",
    history: true, // undo/redo via Ctrl+Z / Ctrl+Y (built-in keybindings)
    clipboard: true,
    clipboardCopyRowRange: "range",
    clipboardCopyConfig: { rowHeaders: false, columnHeaders: false },
    clipboardCopyStyled: false,
    clipboardPasteAction: false, // paste handled by our handlePaste
  };
  // Only a single-column pk is a valid, unique Tabulator row index; composite
  // pks (project_pod, pod_revision) fall back to Tabulator's internal index.
  if (cfg.pk.length === 1) tableOptions.index = cfg.pk[0];

  const table = new Tabulator(`#grid-${tableName}`, tableOptions);

  const state = {
    table,
    cfg,
    refs: payload.refs,
    updates: new Map(),
    deletes: new Map(),
  };
  table.pod_state = state;
  gridState[tableName] = state;
  table.on("tableBuilt", () => {
    const restored = restoreState(state);
    seedIfEmpty(state);
    if (restored) {
      setStatus("Restored unsaved changes from this session — Save to keep them", false);
    }
  });

  // Paste itself is handled at document level (initKeyboard) — Cmd+V's
  // ClipboardEvent fires on whatever has focus (often <body>), which never
  // bubbles through this holder. Here we only track which grid is active.
  const holder = document.getElementById(`grid-${tableName}`);
  if (holder) holder.addEventListener("mousedown", () => { activeGrid = state; });

  return table;
}

function editorIsOpen() {
  return !!document.querySelector(".tabulator-editing");
}

// Top-left cell of the current range selection in the active grid.
function anchorCell(state) {
  const ranges = state.table.getRanges ? state.table.getRanges() : [];
  if (!ranges.length) return null;
  const rows = ranges[0].getRows();
  const cols = ranges[0].getColumns();
  if (!rows.length || !cols.length) return null;
  return rows[0].getCell(cols[0].getField());
}

// Sheets keyboard model, document-level because Tabulator's own keybindings
// listen for Ctrl only (not Cmd on macOS) and require table focus:
//   Cmd/Ctrl+C copies the selected range, Cmd/Ctrl+V block-pastes at the
//   range anchor, Cmd/Ctrl+Z / Shift+Z / Y undo/redo, Enter opens the editor,
//   and typing a character starts editing the anchor cell (type-to-edit).
function initKeyboard() {
  document.addEventListener("paste", (e) => {
    const state = activeGrid;
    if (!state) return;
    const el = document.activeElement;
    // A cell editor (or any other form control) owns the paste natively.
    if (el && el.closest && el.closest(".tabulator-editing")) return;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
    handlePaste(e, state);
  });

  document.addEventListener("keydown", (e) => {
    const state = activeGrid;
    if (!state || editorIsOpen()) return;
    const mod = e.metaKey || e.ctrlKey;
    const key = e.key.toLowerCase();
    if (mod && key === "c") {
      state.table.copyToClipboard("range");
      e.preventDefault();
      return;
    }
    if (mod && (key === "z" || key === "y")) {
      if (key === "y" || e.shiftKey) state.table.redo();
      else state.table.undo();
      e.preventDefault();
      return;
    }
    if (mod || e.altKey) return; // Cmd+V arrives via the paste event above

    const isPrintable = e.key.length === 1;
    if (e.key !== "Enter" && !isPrintable) return;
    const cell = anchorCell(state);
    if (!cell) return;
    const col = fieldConfig(state.cfg, cell.getField());
    if (!col) return;
    const isNew = cell.getRow().getData()._new === true;
    if (!isColWritable(col, isNew)) return;
    if (e.key === "Enter") {
      e.preventDefault(); // keep the range module from also moving the range
      cell.edit(true);
    } else {
      // Type-to-edit: open the editor during keydown so focus lands on its
      // input before the browser inserts the character; the editor selects
      // the existing content, so the keystroke replaces it. No
      // preventDefault — the character must reach the input.
      cell.edit(true);
    }
  });
}

async function reloadGrid(tableName) {
  const state = gridState[tableName];
  const payload = await fetchTableData(tableName);
  state.refs = payload.refs;
  state.updates.clear();
  state.deletes.clear();
  sessionStorage.removeItem(STORAGE_PREFIX + tableName); // staged work is saved
  // Rebuild columns so list editors/formatters capture the fresh reference
  // data — their value maps are closures over refs from build time.
  state.table.setColumns(buildColumns(state.cfg, payload.refs));
  await state.table.setData(payload.rows);
  seedIfEmpty(state);
}

function clearRowErrors(table) {
  for (const row of table.getRows()) {
    row.getElement().classList.remove("row-error");
    for (const cell of row.getCells()) {
      cell.getElement().classList.remove("cell-error");
    }
  }
}

function applyRowErrors(state, changeset, insertRows, errors) {
  const messages = [];
  for (const err of errors) {
    let row = null;
    if (err.kind === "insert") {
      // inserts and insertRows are built in lockstep — index maps directly.
      row = insertRows[err.index] || null;
    } else {
      const bucket = changeset[`${err.kind}s`];
      const rowData = bucket && err.index in bucket ? bucket[err.index] : null;
      // updates/deletes always carry their pk fields. Compare stringified:
      // the changeset coerces numeric pks to Number while live row data from
      // "input"/"list" editors holds strings.
      if (rowData) {
        row = state.table.getRows().find((r) => {
          const d = r.getData();
          return state.cfg.pk.every((k) => String(d[k]) === String(rowData[k]));
        });
      }
    }
    if (row) {
      row.getElement().classList.add("row-error");
      messages.push(`Row ${row.getPosition()}: ${err.message}`);
    } else {
      messages.push(err.message);
    }
  }
  return messages;
}

function showErrors(tableName, messages) {
  const el = document.getElementById(`errors-${tableName}`);
  if (!el) return;
  el.textContent = messages.join("\n");
  el.hidden = messages.length === 0;
}

// Client-side required check so an incomplete row fails fast with a clear
// message instead of a server round-trip (the server still re-validates).
function validateRequired(state, changeset, insertRows) {
  const required = state.cfg.columns.filter((c) => c.required);
  const messages = [];
  changeset.inserts.forEach((insert, i) => {
    const missing = required.filter((c) => {
      const v = insert[c.field];
      return v === undefined || v === null || String(v).trim() === "";
    });
    if (!missing.length) return;
    const row = insertRows[i];
    if (row) row.getElement().classList.add("row-error");
    const pos = row ? row.getPosition() : i + 1;
    messages.push(`Row ${pos}: missing ${missing.map((c) => c.title).join(", ")}`);
  });
  return messages;
}

async function saveTable(tableName) {
  const state = gridState[tableName];
  if (!state) return;
  const { changeset, insertRows } = buildChangeset(state);
  clearRowErrors(state.table);
  showErrors(tableName, []);
  const clientErrors = validateRequired(state, changeset, insertRows);
  if (clientErrors.length) {
    showErrors(tableName, clientErrors);
    setStatus("Not saved — fix the highlighted rows, then Save again", true);
    return;
  }
  setStatus("Saving…", false);
  let resp;
  let body;
  try {
    resp = await fetch(`/api/tables/${tableName}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(changeset),
    });
    body = await resp.json();
  } catch (err) {
    setStatus(`Save failed (${err.message || "network error"}) — nothing was saved`, true);
    return;
  }
  if (resp.status === 422) {
    const messages = applyRowErrors(state, changeset, insertRows, body.errors || []);
    showErrors(tableName, messages);
    setStatus("Not saved — fix the errors listed above the grid", true);
    return;
  }
  if (!resp.ok) {
    setStatus(`Save failed (HTTP ${resp.status}) — nothing was saved`, true);
    return;
  }
  setStatus("Saved ✓ · publishing…", false);
  await reloadGrid(tableName);
  await publishRegistry("Saved ✓ · ");
}

async function publishRegistry(prefix = "") {
  let resp;
  let body;
  try {
    resp = await fetch("/api/publish", { method: "POST" });
    body = await resp.json();
  } catch (err) {
    setStatus(`${prefix}publish failed (${err.message || "network error"}) — use Publish to retry`, true);
    return;
  }
  if (!resp.ok || body.ok === false) {
    setStatus(`${prefix}publish failed: ${body.error || `HTTP ${resp.status}`} — use Publish to retry`, true);
    return;
  }
  const counts = Object.entries(body.tables || {})
    .map(([t, n]) => `${t}: ${n}`)
    .join(", ");
  setStatus(`${prefix}published ✓ ${counts}`, false);
}

function initGrids() {
  document.querySelectorAll('script[id^="config-"]').forEach((script) => {
    const cfg = JSON.parse(script.textContent);
    const tableName = cfg.table;
    fetchTableData(tableName)
      .then((payload) => buildGrid(tableName, cfg, payload))
      .catch((err) => setStatus(err.message || `loading ${tableName} failed`, true));
  });

  document.querySelectorAll("button.add-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      const state = gridState[btn.dataset.table];
      if (state) state.table.addRow({ _new: true }, true);
    });
  });

  document.querySelectorAll("button.save").forEach((btn) => {
    btn.addEventListener("click", () => saveTable(btn.dataset.table));
  });

  const publishBtn = document.getElementById("publish-btn");
  if (publishBtn) publishBtn.addEventListener("click", () => publishRegistry());

  initKeyboard();

  // Snapshot staged edits before the page goes away (nav/refresh/close);
  // restoreState brings them back on the next load in this tab.
  window.addEventListener("pagehide", () => {
    for (const state of Object.values(gridState)) persistState(state);
  });

  // sessionStorage dies with the tab — warn before closing with staged edits.
  window.addEventListener("beforeunload", (e) => {
    if (hasPendingChanges()) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}
