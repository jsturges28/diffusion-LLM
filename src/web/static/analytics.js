// Analytics Suite: client-side logic.

"use strict";

// ---- DOM refs ----

var groupByMount =
  document.getElementById("group-by-mount");
var groupBySelect = null;
var btnCompare =
  document.getElementById("btn-compare");
var btnRefresh =
  document.getElementById("btn-refresh");
var runsTbody =
  document.getElementById("runs-tbody");
var runsEmpty =
  document.getElementById("runs-empty");
var selectAllCb =
  document.getElementById("select-all");

var detailPanel =
  document.getElementById("detail-modal");
var detailTitle =
  document.getElementById("detail-title");
var detailMeta =
  document.getElementById("detail-meta");
var btnCloseDetail =
  document.getElementById("btn-close-detail");
var timingSection =
  document.getElementById("timing-section");
var gpuLabel =
  document.getElementById("gpu-label");

var overlayViewer =
  document.getElementById("overlay-viewer");
var overlaySelectGroup =
  document.getElementById("overlay-select-group");
var overlayDrawerHandle =
  document.getElementById("overlay-drawer-handle");
var overlaySelectMount =
  document.getElementById("overlay-select-mount");
var overlayOutput =
  document.getElementById("overlay-output");
var overlayReadout =
  document.getElementById("overlay-readout");
var overlayLegend =
  document.getElementById("overlay-legend");
var overlayEmpty =
  document.getElementById("overlay-empty");
var overlaySelect = null;
// Cached frames payload and current overlay mode for the open run.
var overlayData = null;
var overlayMode = "none";

// Layered "Diff vs Original" controls (mirror the generator): two
// opacity sliders plus a difference-blend toggle. State is kept here
// so re-rendering the diff (on slider input) is cheap.
var overlayDiffControls =
  document.getElementById("overlay-diff-controls");
var overlayDiffOrigInput =
  document.getElementById("overlay-diff-original-opacity");
var overlayDiffEditInput =
  document.getElementById("overlay-diff-edited-opacity");
var overlayDiffBlendInput =
  document.getElementById("overlay-diff-blend");
var overlayDiffOrigOpacity = 50;
var overlayDiffEditOpacity = 100;
var overlayDiffBlendOn = false;

var comparePanel =
  document.getElementById("compare-panel");
var btnCloseCompare =
  document.getElementById("btn-close-compare");

var modalDelete =
  document.getElementById("modal-delete");
var deleteRunLabel =
  document.getElementById("delete-run-label");
var btnDeleteConfirm =
  document.getElementById("btn-delete-confirm");
var btnDeleteCancel =
  document.getElementById("btn-delete-cancel");
var btnDeleteClose =
  document.getElementById("btn-delete-close");
var pendingDeleteId = null;

// ---- Chart.js defaults ----

Chart.defaults.color = "#888888";
Chart.defaults.borderColor = "#1e1e1e";
Chart.defaults.font.family =
  "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;

// Fixed-position tooltip: anchored to the top-left
// of the chart area so it never obscures data lines.
Chart.Tooltip.positioners.topLeft =
  function (elements, eventPosition) {
    var chart = this.chart;
    return {
      x: chart.chartArea.left + 8,
      y: chart.chartArea.top + 8,
    };
  };

// Smart positioner: places the tooltip in the chart-area corner
// diagonally opposite the hovered point, so the box steers clear of
// the data as the cursor moves along the curve.
// Places the tooltip in the chart-area corner diagonally opposite
// the hovered point, and returns the box's intended top-left corner
// (paired with forced xAlign:"left"/yAlign:"top"). The point is
// clamped so the box stays fully inside the plotting area on all
// four sides, so it never spills onto the x-axis.
Chart.Tooltip.positioners.smart =
  function (elements, eventPosition) {
    var chart = this.chart;
    var area = chart.chartArea;
    var pad = 10;
    // Box size from the previous frame (0 on the very first hover,
    // corrected on the next frame as it fades in).
    var w = this.width || 120;
    var h = this.height || 44;
    if (!elements || elements.length === 0) {
      return { x: area.left + pad, y: area.top + pad };
    }
    var el = elements[0].element;
    var midX = (area.left + area.right) / 2;
    var midY = (area.top + area.bottom) / 2;
    var x = (el.x > midX)
      ? area.left + pad
      : area.right - pad - w;
    var y = (el.y > midY)
      ? area.top + pad
      : area.bottom - pad - h;
    x = Math.max(
      area.left + pad, Math.min(x, area.right - pad - w)
    );
    y = Math.max(
      area.top + pad, Math.min(y, area.bottom - pad - h)
    );
    return { x: x, y: y };
  };

// Inline plugin: once the tooltip box is drawn, "burn" the data
// through it. Any trendline segment the box covers is redrawn,
// clipped to the box rect, with a glow (so a box trapped over the
// line stays legible), and the active point(s) are re-drawn glowing
// on top.
var burnThroughPlugin = {
  id: "burnThrough",
  afterDraw: function (chart) {
    var tt = chart.tooltip;
    if (!tt || tt.opacity === 0) { return; }
    var ctx = chart.ctx;
    var bx = tt.x;
    var by = tt.y;
    var bw = tt.width;
    var bh = tt.height;

    if (bw > 0 && bh > 0) {
      for (var di = 0; di < chart.data.datasets.length; di++) {
        var meta = chart.getDatasetMeta(di);
        if (
          meta.hidden || !meta.data || meta.data.length === 0
        ) {
          continue;
        }
        var ds = chart.data.datasets[di];
        var color = (typeof ds.borderColor === "string")
          ? ds.borderColor : "#ffffff";
        ctx.save();
        ctx.beginPath();
        ctx.rect(bx, by, bw, bh);
        ctx.clip();
        ctx.beginPath();
        var started = false;
        for (var pi = 0; pi < meta.data.length; pi++) {
          var p = meta.data[pi];
          if (!p || p.skip) { continue; }
          if (!started) {
            ctx.moveTo(p.x, p.y);
            started = true;
          } else {
            ctx.lineTo(p.x, p.y);
          }
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.restore();
      }
    }

    var active = chart.getActiveElements();
    if (active && active.length) {
      for (var i = 0; i < active.length; i++) {
        var ael = active[i].element;
        if (!ael) { continue; }
        var ads = chart.data.datasets[active[i].datasetIndex];
        var acolor = (ads && typeof ads.borderColor === "string")
          ? ads.borderColor : "#ffffff";
        ctx.save();
        ctx.beginPath();
        ctx.arc(ael.x, ael.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = acolor;
        ctx.shadowColor = acolor;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.restore();
      }
    }
  },
};

// Per-chart tooltip-box visibility (the eye toggle in each header).
var tooltipEnabled = {
  convergence: true,
  timing: true,
  confidence: true,
};

// ---- State ----

var allRuns = [];
var sortKey = "created_at";
var sortAsc = false;
var checkedIds = {};
var activeRunId = null;
var gpuName = null;

var chartConvergence = null;
var chartTiming = null;
var chartConfidence = null;
var chartCompareConv = null;

// Map of chart name to Chart instance for zoom.
var chartInstances = {};

var COMPARE_COLORS = [
  "#00ff41", "#00aaff", "#ff9f1c",
  "#ff4444", "#aa66ff", "#ffee00",
  "#ff66aa", "#66ffcc",
];

// Colors for resumed timing segments.
var TIMING_COLOR = "#00aaff";
var TIMING_RESUMED = "#66ccff";

// ---- Data fetching ----

function fetchRuns() {
  return fetch("/api/analytics/runs")
    .then(function (r) { return r.json(); });
}

function fetchMetrics(runId) {
  var url = "/api/analytics/runs/"
    + encodeURIComponent(runId) + "/metrics";
  return fetch(url)
    .then(function (r) { return r.json(); });
}

function fetchCompare(ids) {
  var url = "/api/analytics/compare?ids="
    + ids.map(encodeURIComponent).join(",");
  return fetch(url)
    .then(function (r) { return r.json(); });
}

function fetchFrames(runId) {
  var url = "/api/analytics/runs/"
    + encodeURIComponent(runId) + "/frames";
  return fetch(url)
    .then(function (r) { return r.json(); });
}

function fetchSystemInfo() {
  return fetch("/api/analytics/system")
    .then(function (r) { return r.json(); });
}

// ---- Helpers ----

function paramVal(run, key) {
  if (key === "prompt") {
    return run.prompt || "";
  }
  if (key === "model") {
    return run.backend || run.model || "";
  }
  if (key === "elapsed_seconds") {
    return run.elapsed_seconds;
  }
  if (key === "created_at") {
    return run.created_at || run.run_id || "";
  }
  if (key === "has_diff") {
    return run.has_diff ? "Yes" : "No";
  }
  if (run.params && run.params[key] !== undefined) {
    return run.params[key];
  }
  return "";
}

function displayVal(run, key) {
  var v = paramVal(run, key);
  if (v === undefined || v === null) {
    return "N/A";
  }
  if (key === "prompt") {
    var s = String(v);
    if (s.length > 40) {
      return s.substring(0, 37) + "...";
    }
    return s;
  }
  if (key === "elapsed_seconds") {
    return Number(v).toFixed(1) + "s";
  }
  if (key === "created_at") {
    return String(v).replace("T", " ");
  }
  return String(v);
}

function checkedRunIds() {
  var ids = [];
  var keys = Object.keys(checkedIds);
  for (var i = 0; i < keys.length; i++) {
    if (checkedIds[keys[i]]) {
      ids.push(keys[i]);
    }
  }
  return ids;
}

function updateCompareButton() {
  var ids = checkedRunIds();
  btnCompare.disabled = ids.length < 2;
}

function buildRemaskFrameSet(remaskEdits) {
  var set = {};
  if (!remaskEdits) { return set; }
  for (var i = 0; i < remaskEdits.length; i++) {
    var fi = remaskEdits[i].frame_index;
    set[fi] = remaskEdits[i].token_positions;
  }
  return set;
}

// Build cumulative elapsed values so the timing
// line never drops to 0 after a resume. Returns
// {values, resumeStartSet} where resumeStartSet
// maps frame indices of each resume's first frame
// to true.
function buildCumulativeTiming(raw, remaskSet) {
  var values = [];
  var resumeStartSet = {};
  var offset = 0;

  for (var i = 0; i < raw.length; i++) {
    if (i > 0 && raw[i] < raw[i - 1]) {
      offset = values[i - 1];
      resumeStartSet[i] = true;
    }
    values.push(
      +(raw[i] + offset).toFixed(3)
    );
  }
  return {
    values: values,
    resumeStartSet: resumeStartSet,
  };
}

// Shared zoom plugin options for scroll + pinch.
function zoomPluginOptions() {
  return {
    zoom: {
      wheel: { enabled: true },
      pinch: { enabled: true },
      mode: "x",
    },
    pan: {
      enabled: true,
      mode: "x",
    },
  };
}

// Shared tooltip title callback that prefixes
// the frame number so it reads "Frame 112"
// on its own line rather than just "112".
function tooltipTitle(items) {
  if (items.length === 0) { return ""; }
  return "Frame " + items[0].label;
}

// ---- Sorting ----

function sortRuns(runs) {
  var key = sortKey;
  var asc = sortAsc;

  var sorted = runs.slice();
  sorted.sort(function (a, b) {
    var va = paramVal(a, key);
    var vb = paramVal(b, key);
    if (va === undefined || va === null) {
      va = "";
    }
    if (vb === undefined || vb === null) {
      vb = "";
    }
    if (typeof va === "number"
      && typeof vb === "number") {
      return asc ? va - vb : vb - va;
    }
    var sa = String(va).toLowerCase();
    var sb = String(vb).toLowerCase();
    if (sa < sb) { return asc ? -1 : 1; }
    if (sa > sb) { return asc ? 1 : -1; }
    return 0;
  });
  return sorted;
}

function updateSortHeaders() {
  var ths = document.querySelectorAll(
    "#runs-table thead th.sortable"
  );
  for (var i = 0; i < ths.length; i++) {
    ths[i].classList.remove(
      "sort-asc", "sort-desc"
    );
    if (ths[i].getAttribute("data-key") === sortKey) {
      ths[i].classList.add(
        sortAsc ? "sort-asc" : "sort-desc"
      );
    }
  }
}

// ---- Grouping ----

function groupRuns(runs, key) {
  if (key === "none") {
    return [{ label: null, runs: runs }];
  }

  var map = {};
  var order = [];
  for (var i = 0; i < runs.length; i++) {
    var v = String(paramVal(runs[i], key));
    if (!map[v]) {
      map[v] = [];
      order.push(v);
    }
    map[v].push(runs[i]);
  }

  var groups = [];
  for (var j = 0; j < order.length; j++) {
    groups.push({
      label: order[j],
      runs: map[order[j]],
    });
  }
  return groups;
}

// ---- Render table ----

// LLaDA-only hyperparameter columns were dropped because
// DiffusionGemma rows leave them blank; those values still appear in
// the per-run detail panel.
var TABLE_KEYS = [
  "prompt", "model", "elapsed_seconds", "created_at",
];

function renderTable() {
  var sorted = sortRuns(allRuns);
  var groupKey = groupBySelect.value;
  var groups = groupRuns(sorted, groupKey);

  runsTbody.innerHTML = "";

  if (allRuns.length === 0) {
    runsEmpty.hidden = false;
    return;
  }
  runsEmpty.hidden = true;

  for (var g = 0; g < groups.length; g++) {
    var group = groups[g];

    if (group.label !== null) {
      var gtr = document.createElement("tr");
      gtr.className = "group-header-row";
      var gtd = document.createElement("td");
      // check column + TABLE_KEYS + has-diff column + actions column.
      gtd.colSpan = TABLE_KEYS.length + 3;
      gtd.textContent = groupKey.toUpperCase()
        .replace("_", " ") + ": " + group.label;
      gtr.appendChild(gtd);
      runsTbody.appendChild(gtr);
    }

    for (var r = 0; r < group.runs.length; r++) {
      var run = group.runs[r];
      var tr = document.createElement("tr");
      tr.setAttribute("data-run-id", run.run_id);

      if (run.run_id === activeRunId) {
        tr.classList.add("row-selected");
      }

      var tdCheck = document.createElement("td");
      tdCheck.className = "col-check";
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "app-checkbox";
      cb.checked = !!checkedIds[run.run_id];
      cb.setAttribute(
        "data-run-id", run.run_id
      );
      tdCheck.appendChild(cb);
      tr.appendChild(tdCheck);

      for (var k = 0; k < TABLE_KEYS.length; k++) {
        var td = document.createElement("td");
        td.textContent = displayVal(
          run, TABLE_KEYS[k]
        );
        if (TABLE_KEYS[k] === "prompt") {
          td.className = "col-prompt";
          td.title = run.prompt || "";
        }
        tr.appendChild(td);
      }

      var tdDiff = document.createElement("td");
      tdDiff.className = "col-hasdiff";
      if (run.has_diff) {
        tdDiff.innerHTML =
          '<span class="hasdiff-yes" title="Diff vs'
          + ' Original available">\u2713</span>';
      } else {
        tdDiff.innerHTML =
          '<span class="hasdiff-no" title="No original'
          + ' snapshot for this run">\u2717</span>';
      }
      tr.appendChild(tdDiff);

      var tdActions = document.createElement("td");
      tdActions.className = "col-actions";
      var delBtn = document.createElement("button");
      delBtn.className = "row-delete-btn";
      delBtn.setAttribute("data-run-id", run.run_id);
      delBtn.title = "Delete run";
      delBtn.setAttribute("aria-label", "Delete run");
      delBtn.innerHTML =
        '<svg viewBox="0 0 24 24" width="11" height="11"'
        + ' fill="none" stroke="currentColor" stroke-width="2"'
        + ' stroke-linecap="round" stroke-linejoin="round"'
        + ' aria-hidden="true"><path d="M3 6h18"/>'
        + '<path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>'
        + '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0'
        + ' 1-2-2L5 6"/><line x1="10" y1="11" x2="10"'
        + ' y2="17"/><line x1="14" y1="11" x2="14"'
        + ' y2="17"/></svg>';
      tdActions.appendChild(delBtn);
      tr.appendChild(tdActions);

      runsTbody.appendChild(tr);
    }
  }

  updateSortHeaders();
}

// ---- Detail panel ----

function showDetail(runId) {
  activeRunId = runId;
  comparePanel.hidden = true;
  detailPanel.classList.remove("hidden");

  var run = null;
  for (var i = 0; i < allRuns.length; i++) {
    if (allRuns[i].run_id === runId) {
      run = allRuns[i];
      break;
    }
  }
  if (!run) { return; }

  detailTitle.textContent =
    "Run: " + run.run_id;

  var html = "";
  html += '<div class="meta-row">'
    + '<span class="meta-label">Prompt:</span>'
    + '</div>';
  html += '<div class="meta-prompt">'
    + escHtml(run.prompt || "N/A") + '</div>';

  var modelName = run.backend || run.model;
  if (modelName) {
    html += '<div class="meta-row">'
      + '<span class="meta-label">Model:</span> '
      + '<span class="meta-value">'
      + escHtml(String(modelName))
      + '</span></div>';
  }

  // Render whatever params this run recorded (model-agnostic).
  var params = run.params || {};
  var paramKeys = Object.keys(params);
  for (var j = 0; j < paramKeys.length; j++) {
    var pk = paramKeys[j];
    html += '<div class="meta-row">'
      + '<span class="meta-label">'
      + pk.replace(/_/g, " ")
      + ':</span> '
      + '<span class="meta-value">'
      + escHtml(String(params[pk]))
      + '</span></div>';
  }

  if (run.elapsed_seconds !== undefined
    && run.elapsed_seconds !== null) {
    html += '<div class="meta-row">'
      + '<span class="meta-label">Elapsed:</span> '
      + '<span class="meta-value">'
      + Number(run.elapsed_seconds).toFixed(2)
      + 's</span></div>';
  }

  detailMeta.innerHTML = html;

  renderTable();
  loadRunCharts(runId, run);
  loadRunOverlays(runId);
}

function hideDetail() {
  activeRunId = null;
  detailPanel.classList.add("hidden");
  clearOverlay();
  renderTable();
}

function escHtml(s) {
  var d = document.createElement("div");
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

// ---- Chart rendering ----

function destroyChart(chart) {
  if (chart) {
    chart.destroy();
  }
  return null;
}

// Inline Chart.js plugin: dashed vertical markers at the frame
// indices where a new canvas (block) begins. Empty list is a
// no-op, so single-canvas (LLaDA) runs draw nothing.
function canvasBoundaryPlugin(boundaries) {
  return {
    id: "canvasBoundaries",
    afterDatasetsDraw: function (chart) {
      if (!boundaries || boundaries.length === 0) { return; }
      var xScale = chart.scales.x;
      var yScale = chart.scales.y;
      var ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = "rgba(255,180,0,0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (var i = 0; i < boundaries.length; i++) {
        var x = xScale.getPixelForValue(boundaries[i]);
        ctx.beginPath();
        ctx.moveTo(x, yScale.top);
        ctx.lineTo(x, yScale.bottom);
        ctx.stroke();
      }
      ctx.restore();
    },
  };
}

// Show/hide the eye's diagonal slash. Driven via inline style (not
// only CSS) so it is robust to any stale-stylesheet caching.
function setEyeSlash(btn, show) {
  var slash = btn.querySelector(".eye-slash");
  if (slash) {
    slash.style.display = show ? "inline" : "none";
  }
}

// Each newly-opened run starts with all tooltip boxes visible (eye
// open, no slash).
function resetTooltipToggles() {
  tooltipEnabled.convergence = true;
  tooltipEnabled.timing = true;
  tooltipEnabled.confidence = true;
  var btns = document.querySelectorAll(
    ".tooltip-toggle-btn"
  );
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove("is-off");
    setEyeSlash(btns[i], false);
  }
}

function loadRunCharts(runId, run) {
  fetchMetrics(runId).then(function (data) {
    if (data.error) { return; }

    resetTooltipToggles();

    chartConvergence = destroyChart(
      chartConvergence
    );
    chartTiming = destroyChart(chartTiming);
    chartConfidence = destroyChart(chartConfidence);

    var remaskEdits = data.remask_edits || [];
    var remaskSet = buildRemaskFrameSet(
      remaskEdits
    );

    renderConvergenceChart(data, remaskSet);
    renderTimingChart(data, remaskSet);
    renderConfidenceChart(data);
  });
}

// ---- Token overlay viewer (durable commit-order / diff) ----

// Diff needs the pre-edit snapshot and at least one remask edit.
function overlayDiffAvailable(data) {
  return !!(
    data.records_available
    && data.original_frames
    && data.remask_edits
    && data.remask_edits.length > 0
  );
}

// Last frame that actually carries token records.
function overlayFinalFrame(frames) {
  if (!frames) {
    return null;
  }
  for (var i = frames.length - 1; i >= 0; i--) {
    if (frames[i] && frames[i].length > 0) {
      return frames[i];
    }
  }
  return null;
}

function overlayConfText(c) {
  if (typeof c !== "number") {
    return "0";
  }
  return String(+c.toFixed(3));
}

function loadRunOverlays(runId) {
  overlayData = null;
  fetchFrames(runId).then(function (data) {
    if (!data || data.error) {
      showOverlayUnavailable();
      return;
    }
    var hasCommit = !!data.records_available;
    var hasDiff = overlayDiffAvailable(data);
    if (!hasCommit && !hasDiff) {
      showOverlayUnavailable();
      return;
    }
    overlayData = data;
    overlayViewer.hidden = false;
    overlayEmpty.hidden = true;
    overlayOutput.hidden = false;
    overlaySelectGroup.hidden = false;
    setOverlayDrawerOpen(false);
    // Mirror the generator: default to None; the drawer offers the
    // durable overlays (Commit Order for record runs, Diff vs
    // Original when a pre-edit snapshot was saved).
    buildOverlaySelect(data);
    setOverlayMode("none");
  });
}

function showOverlayUnavailable() {
  overlayViewer.hidden = false;
  overlaySelectGroup.hidden = true;
  overlayOutput.textContent = "";
  overlayOutput.classList.remove("diff-overlay-mode");
  overlayOutput.hidden = true;
  overlayReadout.textContent = "";
  overlayReadout.hidden = true;
  overlayLegend.hidden = true;
  if (overlayDiffControls) {
    overlayDiffControls.hidden = true;
  }
  overlayEmpty.hidden = false;
}

function clearOverlay() {
  overlayData = null;
  overlayViewer.hidden = true;
  overlaySelectGroup.hidden = true;
  overlayOutput.textContent = "";
  overlayOutput.classList.remove("diff-overlay-mode");
  overlayOutput.hidden = false;
  overlayReadout.textContent = "";
  overlayReadout.hidden = true;
  overlayLegend.hidden = true;
  if (overlayDiffControls) {
    overlayDiffControls.hidden = true;
  }
  overlayEmpty.hidden = true;
}

// Slide the corner drawer open/closed and flip its handle glyph
// (matches the generator's overlay drawer behavior).
function setOverlayDrawerOpen(open) {
  if (!overlaySelectGroup) {
    return;
  }
  overlaySelectGroup.classList.toggle("open", open);
  if (overlayDrawerHandle) {
    overlayDrawerHandle.innerHTML =
      open ? "\u203A" : "\u2039";
  }
}

// Build the overlay custom-select mirroring the generator: None /
// Commit Order / Diff vs Original, each gated on data availability.
function buildOverlaySelect(data) {
  var canDiff = overlayDiffAvailable(data);
  var options = [
    { value: "none", label: "None" },
    {
      value: "commit",
      label: "Commit Order",
      disabled: !data.records_available,
    },
    {
      value: "diff",
      label: "Diff vs Original",
      disabled: !canDiff,
      title: canDiff
        ? undefined
        : "Only available for an edited run saved with"
          + " its original snapshot.",
    },
  ];
  overlaySelectMount.innerHTML = "";
  overlaySelect = createCustomSelect(options, "none");
  overlaySelectMount.appendChild(overlaySelect);
  sizeCustomSelect(overlaySelect);
  overlaySelect.addEventListener("change", function () {
    setOverlayMode(overlaySelect.value);
  });
}

function setOverlayMode(mode) {
  overlayMode = mode;
  if (overlaySelect && overlaySelect.value !== mode) {
    overlaySelect.value = mode;
  }
  overlayLegend.hidden = mode !== "commit";
  if (overlayDiffControls) {
    overlayDiffControls.hidden = mode !== "diff";
  }
  // The layered diff needs the stacking container; every other mode
  // renders plain token spans, so drop the class when leaving diff.
  if (mode !== "diff") {
    overlayOutput.classList.remove("diff-overlay-mode");
  }
  if (mode === "diff") {
    renderDiffOverlay();
  } else if (mode === "commit") {
    renderCommitOverlay();
  } else {
    renderNoneOverlay();
  }
}

// Plain final-frame tokens with no coloring (drawer set to None).
function renderNoneOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  renderOverlayTokens(
    overlayFinalFrame(overlayData.frames),
    function () { return null; },
    function () { return ""; }
  );
}

// Render the final frame as token spans, coloring each resolved
// token via colorFn(index) and appending titleFn(index) to its
// hover tooltip. Masked positions render as the mask glyph.
function renderOverlayTokens(frame, colorFn, titleFn) {
  overlayOutput.textContent = "";
  if (!frame) {
    return;
  }
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < frame.length; i++) {
    var tok = frame[i];
    var span = document.createElement("span");
    span.setAttribute("data-pos", String(i));
    if (!tok || tok.m) {
      span.className = "token-span token-mask";
      span.textContent = OVERLAYS_MASK_CHAR;
      span.title =
        "Token: " + (i + 1) + "\nConfidence: 0";
      fragment.appendChild(span);
      continue;
    }
    span.className = "token-span token-resolved";
    span.textContent = tok.t;
    var color = colorFn(i);
    if (color) {
      span.style.color = color;
    }
    span.title = "Token: " + (i + 1)
      + "\nConfidence: " + overlayConfText(tok.c)
      + titleFn(i);
    fragment.appendChild(span);
  }
  overlayOutput.appendChild(fragment);
}

function renderCommitOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  var frames = overlayData.frames;
  var frame = overlayFinalFrame(frames);
  var steps = overlaysComputeCommitSteps(frames);
  var maxStep = frames.length - 1;
  renderOverlayTokens(
    frame,
    function (i) {
      var step = steps[i];
      if (typeof step === "number" && step >= 0) {
        return commitColor(step, maxStep);
      }
      return null;
    },
    function (i) {
      var step = steps[i];
      if (typeof step === "number" && step >= 0) {
        return "\nResolved at step: " + step;
      }
      return "";
    }
  );
}

// Layered diff (mirrors the generator): the original and edited final
// frames are stacked with independent opacity and an optional
// difference blend, driven by the control row. The shared builder in
// overlays.js owns the layer construction.
function renderDiffOverlay() {
  var curFinal = overlayFinalFrame(overlayData.frames);
  var origFinal = overlayFinalFrame(
    overlayData.original_frames
  );
  var diff = overlaysComputeDiff(
    curFinal, origFinal, overlayData.remask_edits
  );
  overlayReadout.hidden = false;
  overlayReadout.textContent =
    "Diverged " + diff.changedCount
    + "/" + diff.totalCount;
  overlayOutput.textContent = "";
  overlayOutput.classList.add("diff-overlay-mode");
  overlayOutput.appendChild(
    overlaysBuildDiffLayers(
      origFinal || [],
      curFinal || [],
      diff,
      {
        originalOpacity: overlayDiffOrigOpacity,
        editedOpacity: overlayDiffEditOpacity,
        blend: overlayDiffBlendOn,
      }
    )
  );
}

// Wire the diff control row once: sliders and the blend toggle update
// the retained state and re-render only while the diff overlay is the
// active mode.
function wireOverlayDiffControls() {
  if (overlayDiffOrigInput) {
    overlayDiffOrigInput.addEventListener("input", function () {
      overlayDiffOrigOpacity = Number(overlayDiffOrigInput.value);
      if (overlayMode === "diff") {
        renderDiffOverlay();
      }
    });
  }
  if (overlayDiffEditInput) {
    overlayDiffEditInput.addEventListener("input", function () {
      overlayDiffEditOpacity = Number(overlayDiffEditInput.value);
      if (overlayMode === "diff") {
        renderDiffOverlay();
      }
    });
  }
  if (overlayDiffBlendInput) {
    overlayDiffBlendInput.addEventListener("change", function () {
      overlayDiffBlendOn = !!overlayDiffBlendInput.checked;
      if (overlayMode === "diff") {
        renderDiffOverlay();
      }
    });
  }
}

function renderConvergenceChart(data, remaskSet) {
  var canvas = document.getElementById(
    "chart-convergence"
  );

  var labels = [];
  var values = [];
  for (
    var i = 0;
    i < data.convergence.length;
    i++
  ) {
    labels.push(data.convergence[i].frame);
    values.push(
      +(data.convergence[i].resolved_ratio
        * 100).toFixed(2)
    );
  }

  chartConvergence = new Chart(
    canvas.getContext("2d"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "% Resolved",
          data: values,
          borderColor: "#00ff41",
          backgroundColor: "rgba(0,255,65,0.1)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 1.5,
          segment: {
            borderColor: function (ctx) {
              if (remaskSet[ctx.p1DataIndex]) {
                return "#00aaff";
              }
              return undefined;
            },
            borderWidth: function (ctx) {
              if (remaskSet[ctx.p1DataIndex]) {
                return 2.5;
              }
              return undefined;
            },
          },
        }],
      },
      options: convergenceOptions(remaskSet),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
      ],
    }
  );
  chartInstances.convergence = chartConvergence;
}

function convergenceOptions(remaskSet) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        position: "smart",
        caretSize: 0,
        xAlign: "left",
        yAlign: "top",
        callbacks: {
          title: tooltipTitle,
          label: function (ctx) {
            return ctx.dataset.label + ": "
              + ctx.formattedValue;
          },
          afterLabel: function (ctx) {
            var pos = remaskSet[ctx.dataIndex];
            if (!pos) { return ""; }
            return "User remasked "
              + pos.length + " token"
              + (pos.length !== 1 ? "s" : "")
              + ": ["
              + pos.join(", ") + "]";
          },
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame",
        },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        title: {
          display: true,
          text: "% Resolved",
        },
        beginAtZero: true,
      },
    },
  };
}

function renderTimingChart(data, remaskSet) {
  if (
    !data.per_frame_elapsed
    || data.per_frame_elapsed.length === 0
  ) {
    timingSection.hidden = true;
    return;
  }
  timingSection.hidden = false;

  if (gpuName) {
    gpuLabel.textContent = "(" + gpuName + ")";
  }

  var canvas = document.getElementById(
    "chart-timing"
  );

  var cumResult = buildCumulativeTiming(
    data.per_frame_elapsed, remaskSet
  );
  var values = cumResult.values;
  var resumeSet = cumResult.resumeStartSet;

  var labels = [];
  for (var t = 0; t < values.length; t++) {
    labels.push(t);
  }

  chartTiming = new Chart(
    canvas.getContext("2d"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "Elapsed (s)",
          data: values,
          borderColor: TIMING_COLOR,
          backgroundColor:
            "rgba(0,170,255,0.08)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 1.5,
          segment: {
            borderColor: function (ctx) {
              var fi = ctx.p1DataIndex;
              if (remaskSet[fi]) {
                return "#00ff41";
              }
              if (isInResumedRange(
                fi, resumeSet
              )) {
                return TIMING_RESUMED;
              }
              return undefined;
            },
            borderWidth: function (ctx) {
              if (remaskSet[ctx.p1DataIndex]) {
                return 2.5;
              }
              return undefined;
            },
          },
        }],
      },
      options: timingOptions(remaskSet),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
      ],
    }
  );
  chartInstances.timing = chartTiming;
}

// Check whether a frame index falls within a
// resumed range (after a resume boundary but
// not the remask point itself).
function isInResumedRange(fi, resumeSet) {
  var keys = Object.keys(resumeSet);
  for (var k = 0; k < keys.length; k++) {
    if (fi >= parseInt(keys[k], 10)) {
      return true;
    }
  }
  return false;
}

function timingOptions(remaskSet) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        position: "smart",
        caretSize: 0,
        xAlign: "left",
        yAlign: "top",
        callbacks: {
          title: tooltipTitle,
          label: function (ctx) {
            return ctx.dataset.label + ": "
              + ctx.formattedValue;
          },
          afterLabel: function (ctx) {
            var pos = remaskSet[ctx.dataIndex];
            if (!pos) { return ""; }
            return "Resume point ("
              + pos.length
              + " tokens remasked)";
          },
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame",
        },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        title: {
          display: true,
          text: "Seconds",
        },
        beginAtZero: true,
      },
    },
  };
}

// Mean per-frame confidence. Rises toward 100% as the canvas
// converges; canvas boundaries mark each adaptive stop. Hidden
// for legacy runs saved before confidence was recorded.
function renderConfidenceChart(data) {
  var section = document.getElementById(
    "confidence-section"
  );
  var meanConf = data.mean_conf;
  if (!meanConf || meanConf.length === 0) {
    if (section) { section.hidden = true; }
    return;
  }
  if (section) { section.hidden = false; }

  var canvas = document.getElementById(
    "chart-confidence"
  );

  var labels = [];
  var values = [];
  for (var i = 0; i < meanConf.length; i++) {
    labels.push(i);
    var v = meanConf[i];
    values.push(
      v === null || v === undefined
        ? null
        : +(v * 100).toFixed(2)
    );
  }

  chartConfidence = new Chart(
    canvas.getContext("2d"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "Mean confidence",
          data: values,
          borderColor: "#ffb400",
          backgroundColor: "rgba(255,180,0,0.08)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 1.5,
          spanGaps: true,
        }],
      },
      options: confidenceOptions(),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
      ],
    }
  );
  chartInstances.confidence = chartConfidence;
}

function confidenceOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        position: "smart",
        caretSize: 0,
        xAlign: "left",
        yAlign: "top",
        callbacks: {
          title: tooltipTitle,
          label: function (ctx) {
            return ctx.dataset.label + ": "
              + ctx.formattedValue + "%";
          },
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame",
        },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        title: {
          display: true,
          text: "Mean confidence (%)",
        },
        beginAtZero: true,
        max: 100,
      },
    },
  };
}

// ---- Comparison mode ----

function showComparison(ids) {
  detailPanel.classList.add("hidden");
  comparePanel.hidden = false;
  activeRunId = null;
  renderTable();

  fetchCompare(ids).then(function (results) {
    chartCompareConv = destroyChart(
      chartCompareConv
    );

    var convCanvas = document.getElementById(
      "chart-compare-conv"
    );

    var convDatasets = [];
    var maxConvLen = 0;

    for (var i = 0; i < results.length; i++) {
      if (results[i].error) { continue; }
      if (results[i].convergence.length
        > maxConvLen) {
        maxConvLen =
          results[i].convergence.length;
      }
    }

    var convLabels = [];
    for (var cl = 0; cl < maxConvLen; cl++) {
      convLabels.push(cl);
    }

    for (var j = 0; j < results.length; j++) {
      var res = results[j];
      if (res.error) { continue; }
      var color = COMPARE_COLORS[
        j % COMPARE_COLORS.length
      ];
      var label = buildCompareLabel(res.run_id);

      var cData = [];
      for (
        var ci = 0;
        ci < res.convergence.length;
        ci++
      ) {
        cData.push(
          +(res.convergence[ci].resolved_ratio
            * 100).toFixed(2)
        );
      }
      convDatasets.push({
        label: label,
        data: cData,
        borderColor: color,
        backgroundColor: "transparent",
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 1.5,
      });
    }

    chartCompareConv = new Chart(
      convCanvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: convLabels,
          datasets: convDatasets,
        },
        options: compareChartOptions(
          "Frame", "% Resolved"
        ),
      }
    );
  });
}

function buildCompareLabel(runId) {
  var run = null;
  for (var i = 0; i < allRuns.length; i++) {
    if (allRuns[i].run_id === runId) {
      run = allRuns[i];
      break;
    }
  }
  if (!run || !run.params) { return runId; }

  return "s=" + run.params.steps
    + " g=" + run.params.gen_length
    + " b=" + run.params.block_length
    + " t=" + run.params.temperature;
}

function compareChartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        display: true,
        position: "bottom",
        labels: { boxWidth: 12, padding: 8 },
      },
      tooltip: {
        position: "smart",
        caretSize: 0,
        xAlign: "left",
        yAlign: "top",
        callbacks: {
          title: tooltipTitle,
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: {
      x: {
        title: {
          display: true,
          text: xLabel,
        },
        ticks: { maxTicksLimit: 14 },
      },
      y: {
        title: {
          display: true,
          text: yLabel,
        },
        beginAtZero: true,
      },
    },
  };
}

function hideComparison() {
  comparePanel.hidden = true;
}

// ---- Zoom button handlers ----

function handleZoomClick(e) {
  var btn = e.target.closest(".zoom-btn");
  if (!btn) { return; }
  var chartName = btn.getAttribute("data-chart");
  var action = btn.getAttribute("data-action");
  var chart = chartInstances[chartName];
  if (!chart) { return; }

  if (action === "in") {
    chart.zoom(1.4);
  } else if (action === "out") {
    chart.zoom(0.7);
  } else if (action === "reset") {
    chart.resetZoom();
  }
}

document.addEventListener(
  "click", handleZoomClick
);

// ---- Event handlers ----

function onSortClick(e) {
  var th = e.target.closest("th.sortable");
  if (!th) { return; }

  var key = th.getAttribute("data-key");
  if (key === sortKey) {
    sortAsc = !sortAsc;
  } else {
    sortKey = key;
    sortAsc = true;
  }
  renderTable();
}

function onRowClick(e) {
  var delBtn = e.target.closest(".row-delete-btn");
  if (delBtn) {
    openDeleteModal(delBtn.getAttribute("data-run-id"));
    return;
  }

  var cb = e.target.closest(
    'input[type="checkbox"]'
  );
  if (cb) {
    var rid = cb.getAttribute("data-run-id");
    checkedIds[rid] = cb.checked;
    updateCompareButton();
    return;
  }

  var tr = e.target.closest("tr[data-run-id]");
  if (!tr) { return; }
  var runId = tr.getAttribute("data-run-id");
  showDetail(runId);
}

function onSelectAll() {
  var checked = selectAllCb.checked;
  checkedIds = {};
  if (checked) {
    for (var i = 0; i < allRuns.length; i++) {
      checkedIds[allRuns[i].run_id] = true;
    }
  }
  renderTable();
  updateCompareButton();
}

function onGroupChange() {
  renderTable();
}

function loadAndRender() {
  fetchRuns().then(function (runs) {
    allRuns = runs;
    checkedIds = {};
    selectAllCb.checked = false;
    updateCompareButton();
    renderTable();
  });
}

// ---- Delete a run ----

function runPath(runId) {
  return "Results/" + runId;
}

function openDeleteModal(runId) {
  pendingDeleteId = runId;
  deleteRunLabel.textContent = runPath(runId);
  btnDeleteConfirm.disabled = false;
  modalDelete.classList.remove("hidden");
}

// Transient bottom-right confirmation toast. Styled inline (rather
// than relying only on the stylesheet) so it renders correctly even
// if a stale CSS copy is cached: fixed bottom-right, app surface
// background, accent-green text, fading out after 3s.
var toastEl = document.getElementById("toast");
var toastTimer = null;

function showToast(message) {
  if (!toastEl) { return; }
  toastEl.textContent = message;
  var s = toastEl.style;
  s.position = "fixed";
  s.bottom = "20px";
  s.right = "24px";
  s.zIndex = "200";
  s.maxWidth = "min(60vw, 520px)";
  s.padding = "10px 16px";
  s.background = "var(--bg-surface)";
  s.border = "1px solid var(--border)";
  s.borderRadius = "var(--radius)";
  s.color = "var(--accent)";
  s.fontFamily = "var(--font-mono)";
  s.fontSize = "12px";
  s.letterSpacing = "0.03em";
  s.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.5)";
  s.pointerEvents = "none";
  s.transition = "opacity 0.25s ease, transform 0.25s ease";
  s.opacity = "0";
  s.transform = "translateY(8px)";
  // Force a reflow so the fade-in transition actually runs.
  void toastEl.offsetWidth;
  s.opacity = "1";
  s.transform = "translateY(0)";
  if (toastTimer !== null) {
    clearTimeout(toastTimer);
  }
  toastTimer = setTimeout(function () {
    s.opacity = "0";
    s.transform = "translateY(8px)";
    toastTimer = null;
  }, 3000);
}

function closeDeleteModal() {
  pendingDeleteId = null;
  modalDelete.classList.add("hidden");
}

function confirmDelete() {
  if (!pendingDeleteId) { return; }
  var runId = pendingDeleteId;
  btnDeleteConfirm.disabled = true;
  fetch(
    "/api/analytics/runs/" + encodeURIComponent(runId),
    { method: "DELETE" }
  )
    .then(function (r) { return r.json(); })
    .then(function (result) {
      if (result && result.success) {
        allRuns = allRuns.filter(function (run) {
          return run.run_id !== runId;
        });
        delete checkedIds[runId];
        if (activeRunId === runId) {
          hideDetail();
        }
        updateCompareButton();
        renderTable();
        showToast(
          "Successfully deleted run \u201c"
          + runPath(runId) + "\u201d"
        );
      }
      closeDeleteModal();
    })
    .catch(function () {
      btnDeleteConfirm.disabled = false;
      closeDeleteModal();
    });
}

// ---- Per-chart tooltip toggle ----

function handleTooltipToggle(e) {
  var btn = e.target.closest(".tooltip-toggle-btn");
  if (!btn) { return; }
  var name = btn.getAttribute("data-chart");
  var enabled = !tooltipEnabled[name];
  tooltipEnabled[name] = enabled;
  btn.classList.toggle("is-off", !enabled);
  // Slash on when the box is hidden; off when it's shown.
  setEyeSlash(btn, !enabled);
  var chart = chartInstances[name];
  if (chart) {
    chart.options.plugins.tooltip.enabled = enabled;
    chart.update();
  }
}

// ---- Wire up events ----

document.querySelector("#runs-table thead")
  .addEventListener("click", onSortClick);

runsTbody.addEventListener("click", onRowClick);

selectAllCb.addEventListener(
  "change", onSelectAll
);

if (groupByMount) {
  groupBySelect = createCustomSelect(
    [
      { value: "none", label: "Date" },
      { value: "model", label: "Model" },
      { value: "prompt", label: "Prompt" },
      { value: "has_diff", label: "Diff vs Original?" },
    ],
    "none"
  );
  groupByMount.appendChild(groupBySelect);
  sizeCustomSelect(groupBySelect);
  groupBySelect.addEventListener("change", onGroupChange);
}

btnRefresh.addEventListener(
  "click", loadAndRender
);

btnCloseDetail.addEventListener(
  "click", hideDetail
);

if (overlayDrawerHandle) {
  overlayDrawerHandle.addEventListener("click", function () {
    setOverlayDrawerOpen(
      !overlaySelectGroup.classList.contains("open")
    );
  });
}

// Close the detail modal when clicking the backdrop (outside the box).
detailPanel.addEventListener("click", function (e) {
  if (e.target === detailPanel) {
    hideDetail();
  }
});

// Escape closes the detail modal (matches the generator's modals).
document.addEventListener("keydown", function (e) {
  if (
    e.key === "Escape"
    && !detailPanel.classList.contains("hidden")
  ) {
    hideDetail();
  }
});

btnCloseCompare.addEventListener(
  "click", hideComparison
);

btnCompare.addEventListener("click", function () {
  var ids = checkedRunIds();
  if (ids.length < 2) { return; }
  showComparison(ids);
});

document.addEventListener("click", handleTooltipToggle);

btnDeleteConfirm.addEventListener("click", confirmDelete);
btnDeleteCancel.addEventListener("click", closeDeleteModal);
btnDeleteClose.addEventListener("click", closeDeleteModal);
modalDelete.addEventListener("click", function (e) {
  if (e.target === modalDelete) {
    closeDeleteModal();
  }
});

// ---- Boot ----

// Eye toggles start "open" (no slash) before any run is opened.
(function () {
  var btns = document.querySelectorAll(".tooltip-toggle-btn");
  for (var i = 0; i < btns.length; i++) {
    setEyeSlash(btns[i], false);
  }
})();

wireOverlayDiffControls();

// Opening Analytics clears the generator's "new run saved" cue.
try {
  sessionStorage.removeItem("diffusion_analytics_new");
} catch (_e) {
  // Storage unavailable: nothing to clear.
}

// Reveal the "Generation" nav link only when a model is resident. The
// generator is gated on an active model (see server.py), so surfacing
// the link only when one is loaded keeps navigation honest: reached
// from the menu with no model, the user has nowhere to "generate" yet.
(function revealGenerationLink() {
  var link = document.getElementById("link-generation");
  if (!link) {
    return;
  }
  fetch("/api/models")
    .then(function (response) {
      return response.json();
    })
    .then(function (info) {
      if (info && info.active) {
        link.hidden = false;
      }
    })
    .catch(function () {
      // Leave it hidden on failure; the menu is always reachable.
    });
})();

fetchSystemInfo().then(function (info) {
  if (info.gpu_name) {
    gpuName = info.gpu_name;
  }
});

loadAndRender();
