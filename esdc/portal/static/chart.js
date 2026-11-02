// Render approved vs outlook comparison: accessible SVG chart + exact value table.
(function () {
  const SERIES = [
    { key: "approved", label: "Approved", color: "#1f6fb2" },
    { key: "outlook", label: "Outlook", color: "#0f8a8d" }
  ];
  const SVG_NS = "http://www.w3.org/2000/svg";

  function isFiniteNumeric(value) {
    if (typeof value === "number") return Number.isFinite(value);
    if (typeof value === "string") {
      if (value.trim() === "") return false; // empty/whitespace-only strings are NOT numeric
      const n = Number(value);
      return Number.isFinite(n);
    }
    return false;
  }

  function normalizeRows(rows) {
    return (Array.isArray(rows) ? rows : []).map(function (row) {
      const year = Number(row && row.year);
      const point = { year: year, approved: null, outlook: null };
      SERIES.forEach(function (s) {
        const raw = row && row[s.key];
        // Preserve the exact original value for display; NULL only when not a finite number.
        point[s.key] = raw === undefined ? null : raw;
      });
      return point;
    }).filter(function (p) { return Number.isFinite(p.year); })
      .sort(function (a, b) { return a.year - b.year; });
  }

  function el(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (k) {
      const v = Number(attrs[k]);
      if (!Number.isFinite(v)) throw new Error("non-finite SVG attribute " + k);
      node.setAttribute(k, String(v));
    });
    return node;
  }

  function text(content, x, y, anchor) {
    const t = el("text", { x: x, y: y });
    t.setAttribute("text-anchor", anchor || "start");
    t.textContent = String(content);
    return t;
  }

  function buildSvg(points, unit) {
    const W = 760, H = 320, padL = 72, padR = 16, padT = 24, padB = 48;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const years = points.map(function (p) { return p.year; });
    const values = [];
    points.forEach(function (p) {
      SERIES.forEach(function (s) {
        if (isFiniteNumeric(p[s.key])) values.push(Number(p[s.key]));
      });
    });
    const yearMin = Math.min.apply(null, years), yearMax = Math.max.apply(null, years);
    let vMin = values.length ? Math.min.apply(null, values) : 0;
    let vMax = values.length ? Math.max.apply(null, values) : 1;
    if (vMin === vMax) {
      // Scale-relative padding so representable values actually change for huge magnitudes.
      const mag = Math.abs(vMin);
      const pad = Math.max(mag * 0.05, 1);
      vMin = vMin - pad;
      vMax = vMax + pad;
    }
    const spanY = yearMax - yearMin || 1;
    const spanV = vMax - vMin;
    const xOf = function (year) { return padL + ((year - yearMin) / spanY) * plotW; };
    const yOf = function (value) { return padT + ((vMax - value) / spanV) * plotH; };

    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Comparison of approved and outlook values by year" + (unit ? " in " + unit : ""));

    const axis = el("line", { x1: padL, y1: padT, x2: padL, y2: padT + plotH });
    axis.setAttribute("stroke", "#666");
    svg.appendChild(axis);
    const base = el("line", { x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH });
    base.setAttribute("stroke", "#666");
    svg.appendChild(base);

    [vMin, (vMin + vMax) / 2, vMax].forEach(function (v) {
      const y = yOf(v);
      const grid = el("line", { x1: padL, y1: y, x2: padL + plotW, y2: y });
      grid.setAttribute("stroke", "#e4e4e4");
      svg.appendChild(grid);
      const label = v === 0 ? "0" : (Math.abs(v) >= 1e6 || Math.abs(v) < 0.001) ? v.toExponential(2) : Number(v.toPrecision(4)).toString();
      svg.appendChild(text(label, padL - 8, y + 4, "end"));
    });

    // Decimate year tick labels to about 10 (including the last) while keeping all points.
    const maxLabels = 10;
    const step = Math.max(1, Math.ceil(years.length / maxLabels));
    years.forEach(function (year, i) {
      if (i % step !== 0 && i !== years.length - 1) return;
      const x = xOf(year);
      svg.appendChild(text(String(year), x, padT + plotH + 18, "middle"));
    });
    svg.appendChild(text(unit || "", padL + plotW / 2, H - 6, "middle"));

    SERIES.forEach(function (s) {
      let d = "";
      let prevYear = null;
      points.forEach(function (p) {
        if (!isFiniteNumeric(p[s.key])) { prevYear = null; return; }
        const x = xOf(p.year), y = yOf(Number(p[s.key]));
        const coord = x.toFixed(3) + " " + y.toFixed(3);
        if (!d || prevYear === null || prevYear !== p.year - 1) {
          d += (d ? " M" : "M") + coord;
        } else {
          d += " L" + coord;
        }
        prevYear = p.year;
      });
      if (d) {
        const path = el("path", {});
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", s.color);
        path.setAttribute("stroke-width", "2");
        svg.appendChild(path);
      }
      points.forEach(function (p) {
        if (!isFiniteNumeric(p[s.key])) return;
        const x = xOf(p.year), y = yOf(Number(p[s.key]));
        const circle = el("circle", { cx: x, cy: y, r: 4 });
        circle.setAttribute("fill", s.color);
        const title = document.createElementNS(SVG_NS, "title");
        title.textContent = p.year + ": " + p[s.key] + (unit ? " " + unit : "") + " (" + s.label + ")";
        circle.appendChild(title);
        svg.appendChild(circle);
      });
    });

    SERIES.forEach(function (s, i) {
      const lx = padL + 10 + i * 120, ly = 12;
      const swatch = el("rect", { x: lx, y: ly - 8, width: 12, height: 12 });
      swatch.setAttribute("fill", s.color);
      svg.appendChild(swatch);
      svg.appendChild(text(s.label, lx + 16, ly));
    });
    return svg;
  }

  function buildTable(points, unit) {
    const table = document.createElement("table");
    const head = table.insertRow();
    ["Year", "Approved" + (unit ? " (" + unit + ")" : ""), "Outlook" + (unit ? " (" + unit + ")" : "")]
      .forEach(function (h) {
        const th = document.createElement("th");
        th.textContent = h;
        head.appendChild(th);
      });
    points.forEach(function (p) {
      const row = table.insertRow();
      const yearCell = row.insertCell();
      yearCell.textContent = String(p.year);
      SERIES.forEach(function (s) {
        const cell = row.insertCell();
        const raw = p[s.key];
        if (raw === null) {
          cell.textContent = "\u2014";
        } else if (raw === "") {
          const mark = document.createElement("span");
          mark.textContent = "(empty string)";
          cell.appendChild(mark);
        } else {
          cell.textContent = String(raw);
        }
      });
    });
    return table;
  }

  window.renderComparisonSeries = function (container, rows, unit) {
    container.replaceChildren();
    const points = normalizeRows(rows);
    if (!points.length) {
      const p = document.createElement("p");
      p.textContent = "No data available.";
      container.appendChild(p);
      return;
    }
    const hasValues = points.some(function (p) {
      return SERIES.some(function (s) { return isFiniteNumeric(p[s.key]); });
    });
    if (!hasValues) {
      const p = document.createElement("p");
      p.textContent = "No data available for either series.";
      container.appendChild(p);
    } else {
      container.appendChild(buildSvg(points, unit));
    }
    container.appendChild(buildTable(points, unit));
  };
})();
