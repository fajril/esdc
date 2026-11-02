(function () {
  'use strict';

  var CONFIG = null;
  var seq = 0;
  var controller = null;
  var initialized = false;

  function readConfig() {
    var node = document.getElementById('chart-config');
    if (!node) return null;
    try {
      var parsed = JSON.parse(node.textContent);
      if (parsed && Array.isArray(parsed.components)) return parsed;
    } catch (e) { /* malformed config: exit quietly */ }
    return null;
  }

  function pick(id) { return document.getElementById(id); }

  async function fetchJson(url, signal) {
    return PortalWorkspace.json(url, { signal: signal });
  }

  function isIntNonNegative(v) { return Number.isInteger(v) && v >= 0; }

  function buildChartWrapper(chart, scope) {
    var wrap = document.createElement('div');
    var heading = document.createElement('p');
    heading.textContent = 'POD ' + scope.pod + ' / ' + scope.date + ' / ' + scope.component;
    wrap.appendChild(heading);
    var plot = document.createElement('div');
    wrap.appendChild(plot);
    renderComparisonSeries(plot, chart.series, chart.unit);
    return wrap;
  }

  function validDiagnostics(diag) {
    return diag && Array.isArray(diag.rows) && isIntNonNegative(diag.total) &&
      Number.isInteger(diag.page) && diag.page > 0 &&
      Number.isInteger(diag.last_page) && diag.last_page > 0;
  }

  function buildDiagnosticsWrapper(diag, scope, generation, requestSeq, signal) {
    var wrap = document.createElement('div');
    var head = document.createElement('p');
    head.textContent = 'Page ' + diag.page + ' of ' + diag.last_page + ' / ' + diag.total + ' diagnostics';
    wrap.appendChild(head);
    diag.rows.forEach(function (row) {
      var line = document.createElement('p');
      var context = ['project_id', 'report_year', 'year_frcast', 'component']
        .filter(function (key) { return row[key] !== null && row[key] !== undefined; })
        .map(function (key) { return key + ': ' + row[key]; });
      line.textContent = row.code + ': ' + row.message + (context.length ? ' [' + context.join(', ') + ']' : '');
      wrap.appendChild(line);
    });
    var basis = document.createElement('p');
    basis.textContent = diag.basis ? 'Resources: ' + (diag.basis.resource_revision || 'unavailable') +
      ' / Timeseries: ' + (diag.basis.timeseries_revision || 'unavailable') : 'Basis: unavailable';
    wrap.appendChild(basis);
    if (diag.last_page > 1) {
      var nav = document.createElement('div');
      var busy = false;
      var buttons = [];
      function pagingButton(label, page, disabled) {
        var button = document.createElement('button');
        button.type = 'button'; button.textContent = label; button.disabled = disabled;
        button.addEventListener('click', async function () {
          if (busy || requestSeq !== seq) return;
          busy = true; buttons.forEach(function (item) { item.button.disabled = true; });
          try {
            var query = '?pod_id_seq=' + encodeURIComponent(scope.pod) +
              '&reported_date=' + encodeURIComponent(scope.date) + '&page=' + page;
            var result = await fetchJson('/api/diagnostics' + query, signal);
            if (requestSeq !== seq) return;
            if (!validDiagnostics(result)) throw new Error('Invalid diagnostics page.');
            if (result.source_generation !== generation) throw new Error('Source changed; show results again.');
            pick('result-diagnostics').replaceChildren(buildDiagnosticsWrapper(result, scope, generation, requestSeq, signal));
            pick('result-status').classList.toggle('error', false);
            pick('result-status').textContent = 'Diagnostics loaded: page ' + result.page + ' of ' + result.last_page + '.';
          } catch (error) {
            if (requestSeq !== seq || error.name === 'AbortError') return;
            pick('result-status').textContent = 'Diagnostics could not be loaded: ' + error.message;
            pick('result-status').classList.toggle('error', true);
          } finally {
            busy = false;
            buttons.forEach(function (item) { item.button.disabled = item.disabled; });
          }
        });
        buttons.push({button: button, disabled: disabled}); nav.appendChild(button);
      }
      pagingButton('Previous diagnostics', diag.page - 1, diag.page <= 1);
      pagingButton('Next diagnostics', diag.page + 1, diag.page >= diag.last_page);
      wrap.appendChild(nav);
    }
    return wrap;
  }

  async function apply() {
    var podSel = pick('result-pod');
    var dateInput = pick('result-date');
    var compSel = pick('result-component');
    var btn = pick('result-apply');
    var status = pick('result-status');
    var chartHost = pick('result-chart');
    var diagHost = pick('result-diagnostics');

    var pod = podSel.value;
    var date = dateInput.value;
    var component = compSel.value;
    if (!pod || !date) {
      status.textContent = 'Select a POD and a date first.';
      return;
    }

    seq++;
    var mySeq = seq;
    if (controller) controller.abort();
    controller = new AbortController();
    var signal = controller.signal;
    btn.disabled = true;
    status.textContent = 'Loading...';

    try {
      var q = '?pod_id_seq=' + encodeURIComponent(pod) + '&reported_date=' + encodeURIComponent(date);
      var results = await Promise.all([
        fetchJson('/api/chart' + q + '&component=' + encodeURIComponent(component), signal),
        fetchJson('/api/diagnostics' + q, signal)
      ]);
      if (mySeq !== seq) return;

      var chart = results[0];
      var diag = results[1];
      if (!Array.isArray(chart.series) || !validDiagnostics(diag)) {
        throw new Error('Invalid payload: series/rows must be arrays and total a non-negative integer.');
      }
      if (chart.source_generation !== diag.source_generation) {
        throw new Error('Invalid payload: source_generation mismatch between chart and diagnostics.');
      }

      var filters = [
        { field: 'pod_id_seq', op: 'eq', value: pod },
        { field: 'reported_date', op: 'eq', value: date }
      ];
      var ok = await Promise.all([
        PortalWorkspace.setFilters('pod_monitoring_api', filters),
        PortalWorkspace.setFilters('pod_forecast_api', filters)
      ]);
      if (mySeq !== seq) return;
      if (!ok[0] || !ok[1]) throw new Error('Grid read failed for one or both result grids.');

      var gen = chart.source_generation;
      var g1 = PortalWorkspace.states.get('pod_monitoring_api');
      var g2 = PortalWorkspace.states.get('pod_forecast_api');
      if (!g1 || !g2 || g1.generation !== gen || g2.generation !== gen) {
        throw new Error('Grid generation does not match chart source_generation.');
      }

      var scope = { pod: pod, date: date, component: component };
      var chartWrap = buildChartWrapper(chart, scope);
      var diagWrap = buildDiagnosticsWrapper(diag, scope, gen, mySeq, signal);
      if (mySeq !== seq) return;

      chartHost.replaceChildren(chartWrap);
      diagHost.replaceChildren(diagWrap);
      status.classList.toggle('error', false);
      status.textContent = 'POD ' + pod + ' / ' + date + ' / ' + component +
        ': ' + chart.series.length + ' points, unit ' + chart.unit + '.';
    } catch (e) {
      if (mySeq !== seq || (e && e.name === 'AbortError')) return;
      status.classList.toggle('error', true);
      status.textContent = 'Error: ' + e.message;
    } finally {
      if (mySeq === seq) btn.disabled = false;
    }
  }

  window.initResults = async function () {
    CONFIG = readConfig();
    if (!CONFIG || initialized) return;
    initialized = true;

    var podSel = pick('result-pod');
    var compSel = pick('result-component');
    var btn = pick('result-apply');

    (PortalWorkspace.refs.pods || []).forEach(function (p) {
      var opt = document.createElement('option');
      opt.value = p.pod_id_seq;
      opt.textContent = p.pod_name + ' (' + p.approval_date + ', letter ' + p.pod_letter_num + ')';
      podSel.appendChild(opt);
    });

    CONFIG.components.forEach(function (c) {
      var opt = document.createElement('option');
      opt.value = c.value;
      opt.textContent = c.label + ' (' + c.unit + ')';
      compSel.appendChild(opt);
    });

    function invalidate() {
      seq++;
      if (controller) controller.abort();
      controller = null;
      btn.disabled = false;
    }

    podSel.addEventListener('change', invalidate);
    pick('result-date').addEventListener('change', invalidate);
    compSel.addEventListener('change', invalidate);
    btn.addEventListener('click', apply);
  };
})();
