// POD Portal grid logic — built on Tabulator 6.3.1 (vendored, MIT license).
// One Tabulator instance per <section class="grid-section"> on the page,
// driven by the JSON config embedded in <script id="config-<table>">.

const gridState = {}; // table -> { table: Tabulator, refs, config, updates: Map, deletes: [] }

const COMPUTED_FIELDS = new Set(["_new", "preceded_by", "superseded_by"]);
const READONLY_INSERT_FIELDS = new Set(["pod_id", "approval_seq"]);
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

function coerceValue(cfg, field, value) {
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
    title: col.title,
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
    column.editorParams = { values };
    column.formatter = (cell) => {
      const v = cell.getValue();
      return values[v] !== undefined ? `${v} — ${values[v]}` : v;
    };
  } else if (col.ref === "pods") {
    const values = {};
    for (const p of refs.pods) values[p.id] = `${p.pod_id} — ${p.pod_name}`;
    column.editor = "list";
    column.editorParams = { values };
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
        const resp = await fetch(`${url}?q=${encodeURIComponent(filterTerm || "")}`);
        return resp.ok ? await resp.json() : [];
      },
      autocomplete: true,
      freetext: true,
    };
  }

  if (!col.readonly) {
    column.cellEdited = (cell) => onCellEdited(cell);
  }

  return column;
}

function onCellEdited(cell) {
  const table = cell.getTable();
  const state = table.pod_state;
  const data = cell.getRow().getData();
  cell.getElement().classList.remove("cell-error");
  if (data._new) return; // inserts are captured wholesale on save
  const key = pkKey(state.cfg, data);
  const field = cell.getField();
  const value = coerceValue(state.cfg, field, cell.getValue());
  state.updates.set(key, { ...(state.updates.get(key) || {}), ...pkFields(state.cfg, data), [field]: value });
}

function pkFields(cfg, data) {
  const out = {};
  for (const k of cfg.pk) out[k] = data[k];
  return out;
}

function rowFormatter(row) {
  const data = row.getData();
  const el = row.getElement();
  el.classList.toggle("row-new", data._new === true);
  el.classList.toggle("row-deleted", data._deleted === true);
}

function stripForInsert(cfg, data) {
  const out = {};
  for (const [k, v] of Object.entries(data)) {
    if (k.startsWith("_")) continue;
    if (COMPUTED_FIELDS.has(k)) continue;
    if (READONLY_INSERT_FIELDS.has(k)) continue;
    out[k] = coerceValue(cfg, k, v);
  }
  return out;
}

function buildChangeset(state) {
  const inserts = [];
  for (const row of state.table.getRows()) {
    const data = row.getData();
    if (data._new && !data._deleted) inserts.push(stripForInsert(state.cfg, data));
  }
  const updates = Array.from(state.updates.values());
  const deletes = Array.from(state.deletes.values());
  return { inserts, updates, deletes };
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
}

function setStatus(msg, isError) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.color = isError ? "#b00020" : "";
}

async function fetchTableData(tableName) {
  const resp = await fetch(`/api/tables/${tableName}`);
  return await resp.json();
}

function buildGrid(tableName, cfg, payload) {
  const columns = cfg.columns.map((col) => buildColumn(col, payload.refs));
  columns.push({
    title: "",
    field: "_delete",
    width: 40,
    headerSort: false,
    formatter: () => "🗑",
    cellClick: (e, cell) => markRowDeleted(cell.getTable(), cell.getRow()),
  });

  const tableOptions = {
    data: payload.rows,
    columns,
    layout: "fitDataStretch",
    rowFormatter,
    clipboard: true,
    clipboardPasteAction: "range",
    selectableRange: 1,
    selectableRangeColumns: true,
    selectableRangeRows: true,
    // With range selection enabled, the default edit trigger ("focus")
    // conflicts with the range-selection module: single click only selects
    // the range and editors never open. Tabulator 6.x's documented pairing
    // for spreadsheet-style UX is single click = select, double click = edit.
    editTriggerEvent: "dblclick",
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
  return table;
}

async function reloadGrid(tableName) {
  const state = gridState[tableName];
  const payload = await fetchTableData(tableName);
  state.refs = payload.refs;
  state.updates.clear();
  state.deletes.clear();
  await state.table.setData(payload.rows);
}

function clearRowErrors(table) {
  for (const row of table.getRows()) {
    for (const cell of row.getCells()) {
      cell.getElement().classList.remove("cell-error");
    }
  }
}

function applyRowErrors(state, changeset, errors) {
  const messages = [];
  for (const err of errors) {
    messages.push(err.message);
    const bucket = changeset[`${err.kind}s`];
    if (!bucket || !(err.index in bucket)) continue;
    const rowData = bucket[err.index];
    // For inserts pk may not be fully known (e.g. pod_id not yet issued);
    // fall back to matching on whatever pk fields are present. Compare
    // stringified: the changeset coerces numeric pks (id, pod_id) to Number
    // via stripForInsert, while live row data from "input"/"list" editors
    // holds strings — strict === would never match and no cell turns red.
    const row = state.table.getRows().find((r) => {
      const d = r.getData();
      return state.cfg.pk.every(
        (k) => rowData[k] === undefined || String(d[k]) === String(rowData[k])
      );
    });
    if (row) {
      row.getElement().classList.add("cell-error");
      for (const cell of row.getCells()) cell.getElement().classList.add("cell-error");
    }
  }
  return messages;
}

async function saveTable(tableName) {
  const state = gridState[tableName];
  const changeset = buildChangeset(state);
  clearRowErrors(state.table);
  const resp = await fetch(`/api/tables/${tableName}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changeset),
  });
  const body = await resp.json();
  if (resp.status === 422) {
    const messages = applyRowErrors(state, changeset, body.errors || []);
    setStatus(messages[0] || "Save failed", true);
    return;
  }
  if (body.publish_error) {
    setStatus(`Saved · publish failed: ${body.publish_error}`, true);
  } else {
    setStatus("Saved · published ✓", false);
  }
  await reloadGrid(tableName);
}

async function publishRegistry() {
  const resp = await fetch("/api/publish", { method: "POST" });
  const body = await resp.json();
  if (!resp.ok) {
    setStatus("Publish failed", true);
    return;
  }
  const counts = Object.entries(body.tables || {})
    .map(([t, n]) => `${t}: ${n}`)
    .join(", ");
  setStatus(`Published ✓ ${counts}`, false);
}

function initGrids() {
  document.querySelectorAll('script[id^="config-"]').forEach((script) => {
    const cfg = JSON.parse(script.textContent);
    const tableName = cfg.table;
    fetchTableData(tableName).then((payload) => buildGrid(tableName, cfg, payload));
  });

  document.querySelectorAll("button.add-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tableName = btn.dataset.table;
      gridState[tableName].table.addRow({ _new: true }, true);
    });
  });

  document.querySelectorAll("button.save").forEach((btn) => {
    btn.addEventListener("click", () => saveTable(btn.dataset.table));
  });

  const publishBtn = document.getElementById("publish-btn");
  if (publishBtn) publishBtn.addEventListener("click", () => publishRegistry());
}
