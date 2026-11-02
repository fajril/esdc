// PortalEditors - Tabulator 6.3.1 column definitions for the portal grid.
// Parent owns portal.js state/network; this module only builds column defs.
(function (global) {
  'use strict';

  var HIDDEN_RE = /(_uuid$|^created_|^updated_)/;

  function isRefCol(col) { return !!col.ref; }

  function isEntityList(col) {
    if (col.entityList === true) return true;
    var t = String(col.logical_type || '').toLowerCase();
    return t === 'entity_list' || t === 'entitylist';
  }

  function span(s) {
    var el = document.createElement('span');
    el.textContent = s;
    return el;
  }

  function displaySpan(s) {
    if (s === '') return span('(empty string)');
    return span(s);
  }

  function refValues(col, refs) {
    var arr = (refs && refs[col.ref]) || [];
    return arr.map(function (raw) {
      var value = String(raw[col.refValue]);
      var label = raw[col.refLabel];
      if (col.ref === 'pods') {
        label = [raw.pod_name, raw.approval_date, raw.pod_letter_num]
          .filter(function (p) { return p !== null && p !== undefined && p !== ''; })
          .join(' · ');
      }
      return { value: value, label: String(label) };
    });
  }

  function refFormatter(col, refs) {
    var map = {};
    refValues(col, refs).forEach(function (o) { map[o.value] = o.label; });
    return function (cell) {
      var v = cell.getValue();
      if (v === null || v === undefined) return span('\u2014');
      var raw = String(v);
      return displaySpan(map[raw] !== undefined ? map[raw] : raw);
    };
  }

  function entityDisplay(col, value) {
    if (value === null || value === undefined) return '';
    if (Array.isArray(value)) return value.join('; ');
    if (typeof value === 'string') {
      try {
        var arr = JSON.parse(value);
        if (Array.isArray(arr)) return arr.join('; ');
      } catch (e) { /* not a JSON array, keep raw string */ }
    }
    return String(value);
  }

  function textEditor(col) {
    return function (cell, onRendered, success, cancel) {
      var input = document.createElement('input');
      input.type = col.logical_type === 'date' ? 'date' : 'text';
      var v = cell.getValue();
      if (isEntityList(col)) {
        input.value = entityDisplay(col, v);
      } else {
        input.value = v === null || v === undefined ? '' : String(v);
      }
      var done = false;
      function commit() {
        if (done) return;
        done = true;
        success(input.value);
      }
      input.addEventListener('blur', commit);
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') commit();
        else if (e.key === 'Escape') { done = true; cancel(); }
      });
      onRendered(function () { input.focus(); input.select(); });
      return input;
    };
  }

  function titleWithUnit(col) {
    return col.unit ? col.title + ' (' + col.unit + ')' : col.title;
  }

  function editableCallback(col, config) {
    return function (cell) {
      if (config.readonly) return false;
      var row = cell.getRow().getData();
      return row._new ? !!(col.editable || col.editableOnNew) : !!col.editable;
    };
  }

  function columns(config, refs, onEdit, onActiveCell) {
    var defs = [];
    var pk = config.pk || [];

    (config.columns || []).forEach(function (col, i) {
      var def = {
        field: col.field,
        title: titleWithUnit(col),
        visible: !HIDDEN_RE.test(col.field),
        editable: editableCallback(col, config),
        cellClick: function (event, cell) { if (onActiveCell) onActiveCell(cell); }
      };
      if (col.description) def.headerTooltip = col.description;
      if (i === 0 && pk.indexOf(col.field) >= 0) def.frozen = true;

      if (isRefCol(col)) {
        def.formatter = refFormatter(col, refs);
        def.editor = 'list';
        if (col.ref === 'projects' || col.ref === 'documents') {
          def.editorParams = function () {
            return {
              autocomplete: true,
              listOnEmpty: true,
              freetext: false,
              filterRemote: true,
              valuesLookup: async function (cell, filterTerm) {
                var response = await fetch('/api/' + col.ref + '?q=' + encodeURIComponent(filterTerm || ''));
                if (!response.ok) throw new Error('Reference lookup failed: ' + col.ref + ' (' + response.status + ')');
                return await response.json();
              }
            };
          };
        } else {
          def.editorParams = function () {
            return {
              autocomplete: true,
              listOnEmpty: true,
              freetext: false,
              values: refValues(col, refs)
            };
          };
        }
      } else if (String(col.logical_type || '').toLowerCase() === 'boolean') {
        def.formatter = function (cell) {
          var v = cell.getValue();
          if (v === null || v === undefined) return span('\u2014');
          return displaySpan(v ? 'True' : 'False');
        };
        def.editor = 'list';
        def.editorParams = {
          values: [{ label: 'NULL', value: null }, { label: 'True', value: true }, { label: 'False', value: false }],
          listOnEmpty: true
        };
      } else {
        def.formatter = function (cell) {
          var v = cell.getValue();
          if (v === null || v === undefined) return span('\u2014');
          return displaySpan(isEntityList(col) ? entityDisplay(col, v) : String(v));
        };
        def.editor = textEditor(col);
      }

      defs.push(def);
    });

    if (!config.readonly) {
      (config.creation_inputs || []).forEach(function (ci) {
        var staticOptions = refValues({ ref: ci.ref, refValue: ci.refValue, refLabel: ci.refLabel }, refs);
        defs.push({
          field: ci.field,
          title: ci.title,
          editor: 'list',
          editorParams: {
            values: staticOptions,
            multiselect: true,
            listOnEmpty: true
          },
          formatter: function (cell) {
            var v = cell.getValue();
            if (v === null || v === undefined) return span('\u2014');
            var map = {};
            staticOptions.forEach(function (o) { map[o.value] = o.label; });
            var parts = Array.isArray(v) ? v : [v];
            return displaySpan(parts.map(function (id) {
              var key = String(id);
              return map[key] !== undefined ? map[key] : key;
            }).join('; '));
          },
          editable: function (cell) {
            return !config.readonly && !!cell.getRow().getData()._new;
          },
          cellClick: function (event, cell) { if (onActiveCell) onActiveCell(cell); }
        });
      });
    }

    return defs;
  }

  global.PortalEditors = { columns: columns };
})(window);
