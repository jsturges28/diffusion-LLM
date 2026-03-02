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

// ---- State ----

var allRuns = [];
var sortKey = "created_at";
var sortAsc = false;
var checkedIds = {};
var activeRunId = null;

var chartConvergence = null;
var chartChurn = null;
var chartTiming = null;
var chartCompareConv = null;
var chartCompareChurn = null;

var COMPARE_COLORS = [
  "#00ff41", "#00aaff", "#ff9f1c",
  "#ff4444", "#aa66ff", "#ffee00",
  "#ff66aa", "#66ffcc",
];

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
    chartChurn = destroyChart(chartChurn);
    chartTiming = destroyChart(chartTiming);

    var convCanvas = document.getElementById(
      "chart-convergence"
    );
    var churnCanvas = document.getElementById(
      "chart-churn"
    );
    var timingCanvas = document.getElementById(
      "chart-timing"
    );

    // Convergence chart.
    var convLabels = [];
    var convData = [];
    for (
      var i = 0;
      i < data.convergence.length;
      i++
    ) {
      convLabels.push(data.convergence[i].frame);
      convData.push(
        +(data.convergence[i].resolved_ratio
          * 100).toFixed(2)
      );
    }

    chartConvergence = new Chart(
      convCanvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: convLabels,
          datasets: [{
            label: "% Resolved",
            data: convData,
            borderColor: "#00ff41",
            backgroundColor: "rgba(0,255,65,0.1)",
            fill: true,
            tension: 0.2,
            pointRadius: 0,
            borderWidth: 1.5,
          }],
        },
        options: chartLineOptions(
          "Frame", "% Resolved"
        ),
      }
    );

    // Churn chart.
    var churnLabels = [];
    var churnData = [];
    for (
      var j = 0;
      j < data.churn.length;
      j++
    ) {
      churnLabels.push(data.churn[j].frame);
      churnData.push(data.churn[j].changed_count);
    }

    chartChurn = new Chart(
      churnCanvas.getContext("2d"),
      {
        type: "bar",
        data: {
          labels: churnLabels,
          datasets: [{
            label: "Tokens Changed",
            data: churnData,
            backgroundColor: "rgba(255,159,28,0.5)",
            borderColor: "#ff9f1c",
            borderWidth: 1,
          }],
        },
        options: chartBarOptions(
          "Frame", "Changed"
        ),
      }
    );

    // Timing chart (only if data exists).
    if (
      data.per_frame_elapsed
      && data.per_frame_elapsed.length > 0
    ) {
      timingSection.hidden = false;

      var timeLabels = [];
      var timeData = [];
      for (
        var t = 0;
        t < data.per_frame_elapsed.length;
        t++
      ) {
        timeLabels.push(t);
        timeData.push(
          +data.per_frame_elapsed[t].toFixed(3)
        );
      }

      chartTiming = new Chart(
        timingCanvas.getContext("2d"),
        {
          type: "line",
          data: {
            labels: timeLabels,
            datasets: [{
              label: "Elapsed (s)",
              data: timeData,
              borderColor: "#00aaff",
              backgroundColor:
                "rgba(0,170,255,0.08)",
              fill: true,
              tension: 0.2,
              pointRadius: 0,
              borderWidth: 1.5,
            }],
          },
          options: chartLineOptions(
            "Frame", "Seconds"
          ),
        }
      );
    } else {
      timingSection.hidden = true;
    }
  });
}

function chartLineOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: xLabel,
        },
        ticks: { maxTicksLimit: 12 },
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

function chartBarOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: xLabel,
        },
        ticks: { maxTicksLimit: 12 },
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
    chartCompareChurn = destroyChart(
      chartCompareChurn
    );

    var convCanvas = document.getElementById(
      "chart-compare-conv"
    );
    var churnCanvas = document.getElementById(
      "chart-compare-churn"
    );

    var convDatasets = [];
    var churnDatasets = [];

    var maxConvLen = 0;
    var maxChurnLen = 0;

    for (var i = 0; i < results.length; i++) {
      if (results[i].error) { continue; }
      if (results[i].convergence.length
        > maxConvLen) {
        maxConvLen =
          results[i].convergence.length;
      }
      if (results[i].churn.length
        > maxChurnLen) {
        maxChurnLen =
          results[i].churn.length;
      }
    }

    var convLabels = [];
    for (var cl = 0; cl < maxConvLen; cl++) {
      convLabels.push(cl);
    }
    var churnLabels = [];
    for (var chl = 1; chl <= maxChurnLen; chl++) {
      churnLabels.push(chl);
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

      var chData = [];
      for (
        var chi = 0;
        chi < res.churn.length;
        chi++
      ) {
        chData.push(
          res.churn[chi].changed_count
        );
      }
      churnDatasets.push({
        label: label,
        data: chData,
        borderColor: color,
        backgroundColor: color + "44",
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

    chartCompareChurn = new Chart(
      churnCanvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: churnLabels,
          datasets: churnDatasets,
        },
        options: compareChartOptions(
          "Frame", "Changed"
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

loadAndRender();
