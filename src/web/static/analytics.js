// Analytics Suite — client-side logic.

"use strict";

// ---- DOM refs ----

var groupBySelect =
  document.getElementById("group-by-select");
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
  document.getElementById("detail-panel");
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

var comparePanel =
  document.getElementById("compare-panel");
var btnCloseCompare =
  document.getElementById("btn-close-compare");

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

// ---- State ----

var allRuns = [];
var sortKey = "created_at";
var sortAsc = false;
var checkedIds = {};
var activeRunId = null;
var gpuName = null;

var chartConvergence = null;
var chartTiming = null;
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

function fetchSystemInfo() {
  return fetch("/api/analytics/system")
    .then(function (r) { return r.json(); });
}

// ---- Helpers ----

function paramVal(run, key) {
  if (key === "prompt") {
    return run.prompt || "";
  }
  if (key === "elapsed_seconds") {
    return run.elapsed_seconds;
  }
  if (key === "created_at") {
    return run.created_at || run.run_id || "";
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

var TABLE_KEYS = [
  "prompt", "steps", "gen_length",
  "block_length", "temperature", "cfg_scale",
  "remasking", "elapsed_seconds", "created_at",
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
      gtd.colSpan = TABLE_KEYS.length + 1;
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

      runsTbody.appendChild(tr);
    }
  }

  updateSortHeaders();
}

// ---- Detail panel ----

function showDetail(runId) {
  activeRunId = runId;
  comparePanel.hidden = true;
  detailPanel.hidden = false;

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

  var paramKeys = [
    "steps", "gen_length", "block_length",
    "temperature", "cfg_scale", "remasking",
  ];
  for (var j = 0; j < paramKeys.length; j++) {
    html += '<div class="meta-row">'
      + '<span class="meta-label">'
      + paramKeys[j].replace("_", " ")
      + ':</span> '
      + '<span class="meta-value">'
      + escHtml(displayVal(run, paramKeys[j]))
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
}

function hideDetail() {
  activeRunId = null;
  detailPanel.hidden = true;
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

function loadRunCharts(runId, run) {
  fetchMetrics(runId).then(function (data) {
    if (data.error) { return; }

    chartConvergence = destroyChart(
      chartConvergence
    );
    chartTiming = destroyChart(chartTiming);

    var remaskEdits = data.remask_edits || [];
    var remaskSet = buildRemaskFrameSet(
      remaskEdits
    );

    renderConvergenceChart(data, remaskSet);
    renderTimingChart(data, remaskSet);
  });
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
        position: "topLeft",
        caretSize: 0,
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
        position: "topLeft",
        caretSize: 0,
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

// ---- Comparison mode ----

function showComparison(ids) {
  detailPanel.hidden = true;
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
        position: "topLeft",
        caretSize: 0,
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

// ---- Wire up events ----

document.querySelector("#runs-table thead")
  .addEventListener("click", onSortClick);

runsTbody.addEventListener("click", onRowClick);

selectAllCb.addEventListener(
  "change", onSelectAll
);

groupBySelect.addEventListener(
  "change", onGroupChange
);

btnRefresh.addEventListener(
  "click", loadAndRender
);

btnCloseDetail.addEventListener(
  "click", hideDetail
);

btnCloseCompare.addEventListener(
  "click", hideComparison
);

btnCompare.addEventListener("click", function () {
  var ids = checkedRunIds();
  if (ids.length < 2) { return; }
  showComparison(ids);
});

// ---- Boot ----

fetchSystemInfo().then(function (info) {
  if (info.gpu_name) {
    gpuName = info.gpu_name;
  }
});

loadAndRender();
