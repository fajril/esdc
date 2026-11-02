/* Current POD workspace: exact values, local drafts, explicit source publication. */
(() => {
  for (const link of document.querySelectorAll("nav a")) {
    if (new URL(link.href).pathname === location.pathname) link.setAttribute("aria-current", "page");
  }
  const states = new Map();
  const refs = {};
  let nextDraft = 0;
  const node = (tag, text, cls) => {
    const value = document.createElement(tag);
    if (text !== undefined) value.textContent = text;
    if (cls) value.className = cls;
    return value;
  };
  const option = (value, label) => {
    const item = node("option", label); item.value = value; return item;
  };
  async function json(url, options = {}) {
    const response = await fetch(url, options);
    let body;
    try { body = await response.json(); }
    catch (_) { throw new Error(`Server returned an unreadable response (${response.status})`); }
    if (!response.ok) {
      const error = new Error(typeof body.detail === "string" ? body.detail : body.error || `Request failed (${response.status})`);
      error.body = body; error.status = response.status; throw error;
    }
    return body;
  }
  const dirty = s => s.edits.size > 0 || s.deletes.size > 0 || (s.table?.getRows().some(r => r.getData()._new) || false);
  const blocked = s => s.loading || s.saving || dirty(s);
  function status(s, message, error = false) {
    s.status.textContent = message;
    s.status.classList.toggle("error", error);
  }
  function refresh(s) {
    const changed = dirty(s);
    s.count.textContent = `${s.total} records · version ${s.generation ?? "unavailable"}`;
    s.legend.textContent = changed ? "Unsaved changes" : "All changes saved";
    s.save.disabled = s.saving || s.loading || !changed;
    s.discard.disabled = s.saving || s.loading || !changed;
    s.add.disabled = s.saving || s.loading || s.generation === null || !s.cfg.allowCreate;
    s.remove.disabled = s.saving || s.loading || !s.cfg.allowDelete;
    s.clear.disabled = s.saving || s.loading || s.cfg.readonly || !s.activeCell;
    for (const control of s.navigation) control.disabled = blocked(s);
    s.prev.disabled ||= s.page <= 1;
    s.next.disabled ||= s.page >= s.lastPage;
    s.retry.hidden = !s.pending;
    s.retry.disabled = s.publishing || s.loading || s.saving || changed;
  }
  function markRow(s, row) {
    const data = row.getData();
    row.getElement().classList.toggle("row-dirty", data._new || s.edits.has(data._row_key));
    row.getElement().classList.toggle("row-deleted", s.deletes.has(data._row_key));
  }
  function edited(s, cell) {
    const data = cell.getRow().getData(), key = data._row_key;
    if (!data._new) {
      const original = s.original.get(key), field = cell.getField();
      const patch = s.edits.get(key) || {pk: Object.fromEntries(s.cfg.pk.map(k => [k, original[k]])), fields: {}};
      const col = s.cfg.columns.find(c => c.field === field);
      const same = col?.logical_type === "integer" && data[field] !== null && original[field] !== null
        ? String(data[field]) === String(original[field]) : Object.is(data[field], original[field]);
      if (same) delete patch.fields[field]; else patch.fields[field] = data[field];
      if (Object.keys(patch.fields).length) s.edits.set(key, patch); else s.edits.delete(key);
    }
    markRow(s, cell.getRow()); refresh(s);
  }
  function errors(s, issues, submission) {
    s.errors.replaceChildren();
    for (const row of s.table.getRows()) {
      row.getElement().classList.remove("row-error");
      for (const cell of row.getCells()) cell.getElement().classList.remove("cell-error");
    }
    for (const issue of issues || []) {
      const message = node("p", `${issue.kind} ${issue.index >= 0 ? issue.index + 1 : ""}${issue.field ? " · " + issue.field : ""}: ${issue.message}`);
      s.errors.append(message);
      const key = submission?.[issue.kind]?.[issue.index];
      const row = key === undefined ? null : s.table.getRow(key);
      if (row) {
        row.getElement().classList.add("row-error");
        if (issue.field) row.getCell(issue.field)?.getElement().classList.add("cell-error");
      }
    }
  }
  async function load(s) {
    s.abort?.abort(); s.abort = new AbortController();
    const seq = ++s.readSeq;
    s.loading = true; refresh(s);
    const params = new URLSearchParams({page: String(s.page), size: String(s.size), filters: JSON.stringify(s.filters), direction: s.direction});
    if (s.sort) params.set("sort", s.sort);
    try {
      const result = await json(`/api/tables/${s.name}?${params}`, {signal: s.abort.signal});
      if (seq !== s.readSeq) return;
      s.generation = result.source_generation; s.total = result.total;
      s.lastPage = result.last_page; s.page = result.page;
      s.original = new Map(result.rows.map(row => [row._row_key, structuredClone(row)]));
      s.edits.clear(); s.deletes.clear(); s.activeCell = null;
      await s.table.setData(result.rows);
      if (seq !== s.readSeq) return;
      status(s, result.initialized ? (result.available === false ? "Ingest documents before editing their metadata." : "Current data loaded.") : "Load a POD dataset to begin.");
      return true;
    } catch (error) {
      if (seq === s.readSeq && error.name !== "AbortError") status(s, error.message, true);
    } finally { if (seq === s.readSeq) { s.loading = false; refresh(s); } }
  }
  function guard(s, action) {
    if (blocked(s)) { status(s, "Save or discard this draft before changing the view.", true); return; }
    action();
  }
  function payload(s) {
    const body = {source_generation: s.generation, inserts: [], updates: [], deletes: []};
    const keys = {insert: [], update: [], delete: []};
    for (const row of s.table.getRows()) {
      const data = row.getData();
      if (!data._new) continue;
      const value = {};
      for (const col of s.cfg.columns) {
        if ((col.editable || col.editableOnNew) && Object.hasOwn(data, col.field)) value[col.field] = data[col.field];
      }
      for (const input of s.cfg.creation_inputs || []) {
        if (Object.hasOwn(data, input.field)) value[input.field] = data[input.field];
      }
      body.inserts.push(value); keys.insert.push(data._row_key);
    }
    for (const [key, edit] of s.edits) {
      if (s.deletes.has(key)) continue;
      body.updates.push({...edit.pk, ...edit.fields}); keys.update.push(key);
    }
    for (const [key, value] of s.deletes) { body.deletes.push(value); keys.delete.push(key); }
    return {body, keys};
  }
  async function publicationStatus() {
    try {
      const result = await json("/api/status");
      const pod = result.pod;
      if (!pod || pod.source_generation === null) return;
      document.getElementById("status").textContent = pod.state === "fresh" ? "" : "Publication pending";
      for (const s of states.values()) {
        if (s.name === "documents" || s.generation !== pod.source_generation || dirty(s)) continue;
        s.pending = pod.state === "fresh" ? null : {generation: s.generation};
        if (s.pending) status(s, "Source saved · publication pending. Retry to update analytics.");
        refresh(s);
      }
    } catch (error) { document.getElementById("status").textContent = "Publication status unavailable"; }
  }
  async function publish(s) {
    if (!s.pending || s.publishing || s.saving || dirty(s)) return;
    const generation = s.pending.generation;
    s.publishing = true; refresh(s);
    try {
      await json("/api/publish", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({source_generation: generation, table: s.name})});
      if (s.pending?.generation === generation) s.pending = null;
      s.publicationDetails.replaceChildren();
      document.getElementById("status").textContent = "";
      status(s, "Saved and published.");
    } catch (error) {
      status(s, "Saved · publication pending. Resolve the cache error, then retry.", true);
      const detail = node("details"), summary = node("summary", "Publication details");
      detail.append(summary, node("p", error.message));
      if (error.body?.recovery_command) detail.append(node("code", error.body.recovery_command));
      s.publicationDetails.replaceChildren(detail);
    }
    finally { s.publishing = false; refresh(s); }
  }
  async function save(s) {
    if (s.saving || s.loading || !dirty(s)) return;
    s.table.getEditedCells().forEach(cell => cell.clearEdited());
    const submitted = payload(s);
    s.saving = true; refresh(s);
    let result;
    try {
      result = await json(`/api/tables/${s.name}/save`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(submitted.body)});
    } catch (error) {
      errors(s, error.body?.errors || [{kind: "save", index: -1, message: error.message}], submitted.keys);
      status(s, error.status === 409 ? "Source changed. Draft retained; discard and reload before resubmitting." : "Save failed. Draft retained.", true);
    } finally { s.saving = false; refresh(s); }
    if (!result) return;
    errors(s, []);
    // Remove committed inserts/deletes immediately, even if the next GET fails.
    // They must never remain as a resubmittable draft after a durable source save.
    for (const key of [...submitted.keys.insert, ...submitted.keys.delete]) {
      const row = s.table.getRow(key); if (row) await row.delete();
    }
    for (const key of submitted.keys.update) {
      const row = s.table.getRow(key); if (row) s.original.set(key, structuredClone(row.getData()));
    }
    s.edits.clear(); s.deletes.clear();
    s.pending = result.publication === "published" ? null : {generation: result.source_generation};
    // The source is committed now. Refresh or publication failures must never resubmit it.
    s.generation = result.source_generation;
    const loaded = await load(s);
    if (!loaded) { status(s, "Saved. Reload failed; publication can be retried without saving again.", true); refresh(s); return; }
    if (s.name === "m_pod" || s.name === "m_project") {
      try { Object.assign(refs, await json("/api/references")); }
      catch (error) { status(s, `Saved. Reference refresh failed: ${error.message}`, true); }
    }
    if (s.pending) await publish(s);
    else status(s, "Saved and published.");
  }
  function button(holder, label, handler) {
    const value = node("button", label); value.type = "button";
    value.addEventListener("click", handler); holder.append(value); return value;
  }
  function chooser(s, holder) {
    const details = node("details"), summary = node("summary", "Columns"); details.append(summary);
    const groups = new Map();
    for (const col of s.cfg.columns) {
      const name = col.group || "General";
      if (!groups.has(name)) { const group = node("fieldset"); group.append(node("legend", name)); groups.set(name, group); details.append(group); }
      const label = node("label"), input = node("input"); input.type = "checkbox";
      input.checked = !(/_uuid$|^created_|^updated_/.test(col.field));
      input.addEventListener("change", () => s.table.getColumn(col.field).toggle());
      label.append(input, document.createTextNode(col.title || col.field)); groups.get(name).append(label);
    }
    holder.append(details);
  }
  function toolbar(s, holder) {
    const tools = node("div", undefined, "mw-toolbar"), actions = node("div", undefined, "mw-controls");
    s.count = node("span"); s.legend = node("span");
    s.add = button(actions, "Add row", async () => {
      if (s.loading || s.saving || !s.cfg.allowCreate) return;
      await s.table.addRow({...structuredClone(s.cfg.defaults || {}), _new: true, _row_key: `draft:${++nextDraft}`}, true); refresh(s);
    });
    s.remove = button(actions, "Delete / restore selected", () => {
      if (s.saving || s.loading || !s.cfg.allowDelete) return;
      for (const row of s.table.getSelectedRows()) {
        const data = row.getData(), key = data._row_key;
        if (data._new) { row.delete(); continue; }
        if (s.deletes.has(key)) s.deletes.delete(key);
        else s.deletes.set(key, Object.fromEntries(s.cfg.pk.map(k => [k, data[k]])));
        markRow(s, row);
      }
      refresh(s);
    });
    s.clear = button(actions, "Set cell to NULL", () => {
      const cell = s.activeCell;
      if (!cell || s.saving || s.loading) return;
      const col = s.cfg.columns.find(c => c.field === cell.getField());
      if (!col || !col.nullable || s.cfg.readonly || !(col.editable || (cell.getRow().getData()._new && col.editableOnNew))) {
        status(s, "Select an editable nullable cell.", true); return;
      }
      cell.setValue(null); edited(s, cell);
    });
    s.save = button(actions, "Save", () => save(s));
    s.discard = button(actions, "Discard draft", async () => {
      if (s.saving || s.loading) return;
      await load(s); errors(s, []); await publicationStatus();
    });
    s.retry = button(actions, "Retry publication", () => publish(s));
    const reload = button(actions, "Reload", () => guard(s, async () => { await load(s); await publicationStatus(); }));
    tools.append(actions, s.count, s.legend);
    const view = node("div", undefined, "mw-controls");
    const field = node("select"), op = node("select"), value = node("input"), nullValue = node("select");
    field.setAttribute("aria-label", "Filter field"); op.setAttribute("aria-label", "Filter operator"); value.setAttribute("aria-label", "Filter value"); nullValue.setAttribute("aria-label", "NULL condition");
    nullValue.append(option("true", "Is NULL"), option("false", "Is not NULL")); nullValue.hidden = true;
    field.append(option("", "All records"));
    for (const col of s.cfg.columns) field.append(option(col.field, col.title || col.field));
    for (const key of ["eq", "ne", "contains", "gt", "gte", "lt", "lte", "is_null"]) op.append(option(key, key));
    op.addEventListener("change", () => { value.hidden = op.value === "is_null"; nullValue.hidden = !value.hidden; });
    const sort = node("select"), direction = node("select");
    sort.setAttribute("aria-label", "Sort field"); direction.setAttribute("aria-label", "Sort direction");
    sort.append(option("", "Default order"));
    for (const col of s.cfg.columns) sort.append(option(col.field, col.title || col.field));
    direction.append(option("asc", "Ascending"), option("desc", "Descending"));
    view.append(field, op, value, nullValue, sort, direction);
    const apply = button(view, "Apply view", () => guard(s, () => {
      let typed = value.value;
      if (s.cfg.columns.find(c => c.field === field.value)?.logical_type === "boolean") typed = typed === "true" ? true : typed === "false" ? false : typed;
      s.filters = field.value ? [{field: field.value, op: op.value, value: op.value === "is_null" ? nullValue.value === "true" : typed}] : [];
      s.sort = sort.value; s.direction = direction.value; s.page = 1; load(s);
    }));
    s.prev = button(view, "Previous", () => guard(s, () => { s.page--; load(s); }));
    s.next = button(view, "Next", () => guard(s, () => { s.page++; load(s); }));
    const size = node("select"); size.setAttribute("aria-label", "Page size");
    for (const count of [25, 100, 200]) size.append(option(String(count), `${count} per page`));
    size.value = "100"; size.addEventListener("change", () => guard(s, () => { s.size = Number(size.value); s.page = 1; load(s); }));
    view.append(size); tools.append(view); s.navigation = [reload, field, op, value, nullValue, sort, direction, apply, s.prev, s.next, size];
    chooser(s, tools); holder.append(tools);
    s.status = node("p", "Loading…", "mw-status"); s.status.setAttribute("role", "status");
    s.errors = node("div", undefined, "mw-errors"); s.errors.setAttribute("role", "alert"); s.publicationDetails = node("div", undefined, "publication-details"); holder.append(s.status, s.errors, s.publicationDetails);
    if (s.cfg.readonly) for (const control of [s.add, s.remove, s.clear, s.save, s.discard]) control.hidden = true;
  }
  async function grid(script) {
    const name = script.id.slice(7), cfg = JSON.parse(script.textContent), holder = document.getElementById(`grid-${name}`);
    const s = {name, cfg, edits: new Map(), deletes: new Map(), original: new Map(), page: 1, size: 100, lastPage: 1, total: 0, generation: null, filters: [], sort: "", direction: "asc", readSeq: 0, saving: false, publishing: false, loading: false, pending: null, activeCell: null};
    states.set(name, s); holder.replaceChildren(); toolbar(s, holder);
    const host = node("div", undefined, "tabulator-host"); holder.append(host);
    const columns = PortalEditors.columns(cfg, refs, cell => edited(s, cell), cell => { s.activeCell = cell; refresh(s); });
    for (const col of columns) {
      const permitted = col.editable;
      col.headerSort = false;
      col.editable = cell => !s.loading && !s.saving && !s.deletes.has(cell.getRow().getData()._row_key) && (typeof permitted === "function" ? permitted(cell) : permitted === true);
    }
    s.table = new Tabulator(host, {data: [], index: "_row_key", columns, layout: "fitData", height: "520px", selectableRows: true, rowHeader: {formatter: "rowSelection", titleFormatter: "rowSelection", headerSort: false, hozAlign: "center", width: 42, frozen: true}, rowFormatter: row => markRow(s, row)});
    s.table.on("cellEdited", cell => edited(s, cell));
    await new Promise(resolve => s.table.on("tableBuilt", resolve));
    await load(s); return s;
  }
  window.PortalWorkspace = {
    json, refs, states,
    async setFilters(name, filters) {
      const s = states.get(name); if (!s || s.saving || dirty(s)) return false;
      s.filters = filters; s.page = 1; return Boolean(await load(s));
    }
  };
  let initialized = false;
  window.initGrids = async () => {
    if (initialized) return; initialized = true;
    try {
      Object.assign(refs, await json("/api/references"));
      await Promise.all(Array.from(document.querySelectorAll('script[id^="config-"]'), grid));
      await publicationStatus();
      if (window.initResults) await window.initResults();
    } catch (error) { document.getElementById("status").textContent = error.message; }
  };
  window.addEventListener("beforeunload", event => { if (Array.from(states.values()).some(s => s.table && (dirty(s) || s.saving))) { event.preventDefault(); event.returnValue = ""; } });
})();
