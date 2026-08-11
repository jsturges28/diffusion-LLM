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
var runsEmptyCollection =
  document.getElementById("runs-empty-collection");
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

var overlayViewer =
  document.getElementById("overlay-viewer");
var overlaySelectGroup =
  document.getElementById("overlay-select-group");
var overlayDrawerHandle =
  document.getElementById("overlay-drawer-handle");
var overlaySelectMount =
  document.getElementById("overlay-select-mount");
var overlayHighlightCheckbox =
  document.getElementById("overlay-highlight-tokens");
var overlayOutput =
  document.getElementById("overlay-output");
var tokenMetricsStrip =
  document.getElementById("token-metrics");
var overlayReadout =
  document.getElementById("overlay-readout");
var overlayLegend =
  document.getElementById("overlay-legend");
var overlayEmpty =
  document.getElementById("overlay-empty");
var overlayScrubber =
  document.getElementById("overlay-scrubber");
var overlayScrubSlider =
  document.getElementById("overlay-scrubber-slider");
var overlayScrubPrev =
  document.getElementById("overlay-scrub-prev");
var overlayScrubNext =
  document.getElementById("overlay-scrub-next");
var overlayScrubLabel =
  document.getElementById("overlay-scrubber-label");
var overlaySelect = null;
// Cached frames payload and current overlay mode for the open run.
var overlayData = null;
var overlayMode = "none";
// Frame shown by the scrubber. Defaults to the final frame so the
// viewer opens exactly as before; scrubbing back replays earlier
// frames through the active overlay.
var overlayFrameIndex = 0;
// Per-run derived data, memoized and invalidated when a new run loads:
// the Commit Order gradient's per-position steps and the diff change
// set (neither depends on the scrubber frame). The pre-edit run needs
// steps of its own, since a commit step is a property of a frame
// stream rather than of a token.
var overlayCommitSteps = null;
var overlayOriginalCommitSteps = null;
var overlayDiffData = null;
// Whether the open run is autoregressive; gates Commit Order off
// (diffusion-only for now), keeping None + Heatmap + Entropy.
var overlayIsAutoregressive = false;
// Candidate popover for the token overlay (mirrors the generator).
// The page is "original", "edited", or null where only one run
// captured candidates and there is nothing to page between.
var altsPopover =
  document.getElementById("token-alts-popover");
var altsPopoverPos = null;
var altsPopoverPage = null;

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

// Run-level crossfade between the pre-edit run and the branch: 1 is
// the edited run alone, 0 the snapshot. One slider rather than the
// diff overlay's two, because superimposed bars and tokens at
// matching opacity just occlude each other, so the useful axis is the
// mix. It governs every comparison surface at once (the token layers
// and the entropy chart), which is why it lives in the modal header
// rather than under either of them.
var runBlendRow =
  document.getElementById("run-blend-row");
var runBlendInput =
  document.getElementById("run-blend");
var compareBlend = 1;

var comparePanel =
  document.getElementById("compare-panel");
var btnCloseCompare =
  document.getElementById("btn-close-compare");

var modalDelete =
  document.getElementById("modal-delete");
var deleteRunLabel =
  document.getElementById("delete-run-label");
var deleteModalTitle =
  document.getElementById("delete-modal-title");
var deleteModalNote =
  document.getElementById("delete-modal-note");
var btnDeleteConfirm =
  document.getElementById("btn-delete-confirm");
var btnDeleteCancel =
  document.getElementById("btn-delete-cancel");
var btnDeleteClose =
  document.getElementById("btn-delete-close");
var btnBulkDelete =
  document.getElementById("btn-bulk-delete");
var bulkDeleteCount =
  document.getElementById("bulk-delete-count");
// Runs staged for the delete confirmation modal (1 for a row's own
// trashcan, N for the bulk "delete selected" action).
var pendingDeleteIds = [];

var collectionTabs =
  document.getElementById("collection-tabs");
var modalCollections =
  document.getElementById("modal-collections");
var collectionChoices =
  document.getElementById("collection-choices");
var collectionsRunLabel =
  document.getElementById("collections-run-label");
var collectionsNote =
  document.getElementById("collections-note");
var newCollectionName =
  document.getElementById("new-collection-name");
var btnNewCollection =
  document.getElementById("btn-new-collection");
var btnCollectionsDone =
  document.getElementById("btn-collections-done");
var btnCollectionsClose =
  document.getElementById("btn-collections-close");
var modalCollectionDelete =
  document.getElementById("modal-collection-delete");
var colDeleteLabel =
  document.getElementById("col-delete-label");
var btnColDeleteConfirm =
  document.getElementById("btn-col-delete-confirm");
var btnColDeleteCancel =
  document.getElementById("btn-col-delete-cancel");
var btnColDeleteClose =
  document.getElementById("btn-col-delete-close");

// ---- Chart.js defaults ----

Chart.defaults.color = "#888888";
Chart.defaults.borderColor = "#1e1e1e";
Chart.defaults.font.family =
  "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;

// Chart.js paints every tooltip swatch as a white rect at the full
// box size, then strokes it, then fills a square inset one pixel per
// side. The stroke is centered, so it covers only half that inset
// and leaves a half-pixel of white showing between the border and
// the fill (a whole physical pixel at 2x). Dropping the white
// backing removes that edge; see lineLabelColor for the border half
// of the same swatch.
Chart.defaults.plugins.tooltip.multiKeyBackground =
  "transparent";

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

// Corner preference for the smart positioner, most wanted first.
var TOOLTIP_CORNERS = ["tl", "tr", "bl", "br"];

// Smart positioner: parks the tooltip in a corner of the plotting
// area that is free of both the pointer and the drawn data,
// preferring the top-left and falling back through the rest.
//
// The rule this replaced put the box in the corner diagonally
// opposite the hovered point, which knows where the cursor is but
// not where the line goes: on a rising trend "diagonally opposite"
// aims the box straight at it. Returns the box's top-left origin,
// which is what the forced xAlign:"left"/yAlign:"top" expects.
//
// When no corner is free the box has to sit on the data, and
// burnThroughPlugin redraws the line through it.
Chart.Tooltip.positioners.smart =
  function (elements, eventPosition) {
    var chart = this.chart;
    var pad = 10;
    // Box size from the previous frame (0 on the very first hover,
    // corrected on the next frame as it fades in).
    var w = this.width || 120;
    var h = this.height || 44;
    var corner = smartTooltipCorner(
      chart, eventPosition, w, h, pad
    );
    chart.$smartCorner = corner;
    var rect = tooltipCornerRect(
      chart.chartArea, corner, w, h, pad
    );
    return { x: rect.left, y: rect.top };
  };

// The first corner, in TOOLTIP_CORNERS order, that clears both the
// pointer and the data. Corners the pointer occupies are out
// entirely, since a box under the cursor is a box in the way.
function smartTooltipCorner(chart, cursor, w, h, pad) {
  var area = chart.chartArea;
  var clear = [];
  var fallback = null;
  for (var i = 0; i < TOOLTIP_CORNERS.length; i++) {
    var name = TOOLTIP_CORNERS[i];
    var rect = tooltipCornerRect(area, name, w, h, pad);
    if (!cursor || !rectHasPoint(rect, cursor.x, cursor.y, pad)) {
      if (fallback === null) {
        fallback = name;
      }
      if (!chartDataHitsRect(chart, rect)) {
        clear.push(name);
      }
    }
  }
  return pickTooltipCorner(clear, chart.$smartCorner, fallback);
}

// Hysteresis: the standing corner wins while it is still clear, so
// the box settles instead of hopping between two equally good
// corners every time the pointer twitches.
function pickTooltipCorner(clear, previous, fallback) {
  for (var i = 0; i < clear.length; i++) {
    if (clear[i] === previous) {
      return previous;
    }
  }
  if (clear.length > 0) {
    return clear[0];
  }
  if (fallback) {
    return fallback;
  }
  return TOOLTIP_CORNERS[0];
}

// Where a box of w by h sits in one chart-area corner, clamped so it
// stays fully inside the plotting area on all four sides and never
// spills onto the axes.
function tooltipCornerRect(area, corner, w, h, pad) {
  var left = (corner === "tl" || corner === "bl")
    ? area.left + pad
    : area.right - pad - w;
  var top = (corner === "tl" || corner === "tr")
    ? area.top + pad
    : area.bottom - pad - h;
  left = Math.max(
    area.left + pad, Math.min(left, area.right - pad - w)
  );
  top = Math.max(
    area.top + pad, Math.min(top, area.bottom - pad - h)
  );
  return {
    left: left,
    top: top,
    right: left + w,
    bottom: top + h,
  };
}

// Whether any drawn data would end up under a box at ``rect``.
function chartDataHitsRect(chart, rect) {
  for (var di = 0; di < chart.data.datasets.length; di++) {
    var meta = chart.getDatasetMeta(di);
    if (meta.hidden || !meta.data || meta.data.length === 0) {
      continue;
    }
    if (meta.type === "bar") {
      if (barsHitRect(meta.data, rect)) {
        return true;
      }
    } else if (lineHitsRect(meta.data, rect)) {
      return true;
    }
  }
  return false;
}

// Bars are tested as their whole body rather than their top edge:
// the bottom corners of a bar chart are solid even where no bar top
// reaches them.
function barsHitRect(elements, rect) {
  for (var i = 0; i < elements.length; i++) {
    var el = elements[i];
    if (el) {
      var p = el.getProps(["x", "y", "base", "width"], true);
      var half = Math.max(1, p.width) / 2;
      var body = {
        left: p.x - half,
        right: p.x + half,
        top: Math.min(p.y, p.base),
        bottom: Math.max(p.y, p.base),
      };
      if (rectsOverlap(body, rect)) {
        return true;
      }
    }
  }
  return false;
}

// Each span between consecutive points, so a gap (a skipped point)
// breaks the chain rather than drawing a phantom segment across it.
// The first point after a break is tested on its own, which is also
// what a single-point dataset needs.
function lineHitsRect(elements, rect) {
  var previous = null;
  for (var i = 0; i < elements.length; i++) {
    var el = elements[i];
    if (!el || el.skip) {
      previous = null;
      continue;
    }
    var from = previous || el;
    if (segmentHitsRect(from.x, from.y, el.x, el.y, rect)) {
      return true;
    }
    previous = el;
  }
  return false;
}

function rectHasPoint(rect, x, y, margin) {
  if (x < rect.left - margin) { return false; }
  if (x > rect.right + margin) { return false; }
  if (y < rect.top - margin) { return false; }
  if (y > rect.bottom + margin) { return false; }
  return true;
}

function rectsOverlap(a, b) {
  if (a.right < b.left) { return false; }
  if (a.left > b.right) { return false; }
  if (a.bottom < b.top) { return false; }
  if (a.top > b.bottom) { return false; }
  return true;
}

// Whether a segment touches a rect, by Liang-Barsky clipping: walk
// the four boundary slabs, narrowing the stretch of the segment that
// could still be inside, and report whether any stretch survives.
// Segments rather than their endpoints alone, because a sparse run's
// trendline can stride clean across a corner box without landing a
// single vertex inside it. A zero-length segment degenerates into a
// point-in-rect test, which is what a lone point needs.
function segmentHitsRect(x0, y0, x1, y1, rect) {
  var dx = x1 - x0;
  var dy = y1 - y0;
  var edge = [-dx, dx, -dy, dy];
  var slack = [
    x0 - rect.left,
    rect.right - x0,
    y0 - rect.top,
    rect.bottom - y0,
  ];
  var enter = 0;
  var exit = 1;
  for (var i = 0; i < 4; i++) {
    if (edge[i] === 0) {
      // Parallel to this slab: outside it is outside the rect.
      if (slack[i] < 0) {
        return false;
      }
    } else {
      var t = slack[i] / edge[i];
      if (edge[i] < 0) {
        if (t > exit) { return false; }
        if (t > enter) { enter = t; }
      } else {
        if (t < enter) { return false; }
        if (t < exit) { exit = t; }
      }
    }
  }
  return true;
}

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
        var alpha = chartSeriesAlpha(chart, di);
        if (alpha <= 0.02) { continue; }
        var ds = chart.data.datasets[di];
        var color = (typeof ds.borderColor === "string")
          ? ds.borderColor : "#ffffff";
        ctx.save();
        ctx.globalAlpha = alpha;
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
        var aalpha = chartSeriesAlpha(
          chart, active[i].datasetIndex
        );
        if (aalpha <= 0.02) { continue; }
        var ads = chart.data.datasets[active[i].datasetIndex];
        var acolor = (ads && typeof ads.borderColor === "string")
          ? ads.borderColor : "#ffffff";
        ctx.save();
        ctx.globalAlpha = aalpha;
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
  tps: true,
  confidence: true,
  entropy: true,
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
// Shares the Timing slot with chartTiming; timingPage says which of
// the two is on screen.
var chartTps = null;
var chartConfidence = null;
// Per-position, so it is built from the frames payload in
// loadRunOverlays rather than the metrics payload in loadRunCharts.
var chartEntropy = null;
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
var CONFIDENCE_COLOR = "#ffb400";
// Tokens per second shares Timing's slot and is derived from the
// same series, so it keeps the same hue rather than claiming a new
// one for what is the same measurement read a second way.
var TPS_COLOR = TIMING_COLOR;

// The pre-edit run's line on the timing and confidence charts.
// Neutral grey rather than a second hue: timing already spends blue
// on the branch, lighter blue on its resumed stretch, and amber on
// canvas boundaries, so another color would read as a fourth
// category instead of as the baseline both runs share.
var COMPARE_ORIGINAL_COLOR = "#8b93a1";

// Solid for the run that happened, dashed for the branch. The
// counterfactual is the one drawn provisionally.
var COMPARE_EDITED_DASH = [5, 3];

// Wash strength for the band between the two runs. The grey side
// runs stronger because it is desaturated and vanishes at the alpha
// the saturated hues sit comfortably at.
var BAND_ALPHA_EDITED = 0.16;
var BAND_ALPHA_ORIGINAL = 0.24;

// "#00aaff" -> "rgba(0, 170, 255, 0.16)". The band washes are
// derived from the line colors rather than written out beside them,
// so a wash cannot drift away from the line it belongs to, and
// because their alpha has to be computed per draw anyway.
function withAlpha(hex, alpha) {
  var r = parseInt(hex.slice(1, 3), 16);
  var g = parseInt(hex.slice(3, 5), 16);
  var b = parseInt(hex.slice(5, 7), 16);
  return "rgba(" + r + ", " + g + ", " + b + ", "
    + alpha + ")";
}

// The app's edit color, shared with .token-remasked in style.css, so
// an intervention reads the same in the chart as it does in the
// generator's token view.
var EDIT_COLOR = "#ff9f1c";
var EDIT_TINT = "rgba(255, 159, 28, 0.15)";

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
  if (key === "processor") {
    return run.processor || "Unknown";
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

// The checked runs that are currently on screen. Scoped to the rows
// on display because everything downstream acts on them: Compare and
// the bulk delete both take this list, and a stale tick left behind
// by a tab switch would put a run nobody can see into either.
function checkedRunIds() {
  var ids = [];
  var shown = visibleRuns();
  for (var i = 0; i < shown.length; i++) {
    if (checkedIds[shown[i].run_id]) {
      ids.push(shown[i].run_id);
    }
  }
  return ids;
}

function updateCompareButton() {
  var ids = checkedRunIds();
  btnCompare.disabled = ids.length < 2;
}

// Show a trashcan with the selected count in the actions-column header
// when one or more rows are checked; hide it when the selection is
// empty. Kept in sync with the compare button on every selection change.
function updateBulkDeleteButton() {
  if (!btnBulkDelete) { return; }
  var count = checkedRunIds().length;
  if (count < 1) {
    btnBulkDelete.hidden = true;
    return;
  }
  btnBulkDelete.hidden = false;
  if (bulkDeleteCount) {
    bulkDeleteCount.textContent = "(" + count + ")";
  }
  var noun = count === 1 ? " run" : " runs";
  btnBulkDelete.title = "Delete " + count + " selected" + noun;
  btnBulkDelete.setAttribute(
    "aria-label", "Delete " + count + " selected" + noun
  );
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
//
// Runs saved since the client began carrying the
// elapsed offset across a splice are already
// cumulative, so this is a pass-through for them
// and resumeStartSet comes back empty; older runs
// still drop at each branch and get stitched here.
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

// Where the resumed part of an edited run begins, as a set of frame
// indices. Two sources, deliberately not merged: an elapsed drop is
// the only trustworthy marker in an older run, whose timing array
// still holds the pre-edit run's frames in full and so does not line
// up with remask_edits at all. Once the array is aligned there is no
// drop left to find, and the edit's own frame index is exact.
function resumeBoundarySet(resumeStartSet, remaskSet) {
  if (Object.keys(resumeStartSet).length > 0) {
    return resumeStartSet;
  }
  var set = {};
  var keys = Object.keys(remaskSet);
  for (var i = 0; i < keys.length; i++) {
    set[keys[i]] = true;
  }
  return set;
}

// Room under the x axis for the zoom controls docked in the chart's
// bottom-left corner. The y-axis gutter alone is narrower than the
// three buttons, so without this the pill would overlap the first
// tick label. Not used by the compare panel, which has no dock.
function chartGutterLayout() {
  return { padding: { bottom: 16 } };
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

// Shared tooltip swatch color for the line charts. A line's
// backgroundColor is an area wash at around 0.1 alpha, so a swatch
// filled with it reads as almost nothing; the line's own color is
// what tells one series from another in a two-row tooltip.
//
// The border is deliberately invisible rather than absent. Chart.js
// resolves the swatch stroke as ``borderWidth || 1``, so asking for
// zero still strokes a pixel, and that ring plus the white backing
// underneath (see the multiKeyBackground default above) is what made
// the chip read as a colored frame around a lighter square. With
// both suppressed the swatch is exactly the inset fill.
//
// The entropy chart deliberately does not use this: its bars carry
// solid per-bar colors, so its swatches already read correctly and
// showing the hovered bar's own ramp color says more than the series
// color would.
function lineLabelColor(ctx) {
  var color = ctx.dataset.borderColor;
  if (typeof color !== "string") {
    color = "#ffffff";
  }
  return {
    borderColor: "transparent",
    backgroundColor: color,
  };
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

// ---- Collections ----
//
// A collection is a named set of run ids. Membership is a set rather
// than an assignment, so one run can sit in several collections and
// filing it somewhere new never takes it out of where it already was.
//
// Stored as JSON under one durable UI-state key (see ui_state.py),
// the same mechanism the settings use. Unlike those, this key is not
// a cache: nothing on disk records which runs a user cared about, so
// losing it loses work rather than a preference. The server prunes
// ids for deleted runs on every hydrate, which is what keeps a run
// deleted in another window from lingering as an unopenable row.

var COLLECTIONS_KEY = "diffusion_collections";

// Favorites is created on first use rather than shipped empty, so a
// user who never stars anything never sees a tab. Its id is fixed so
// the star always knows where a plain click files to.
var FAVORITES_ID = "favorites";
var FAVORITES_NAME = "Favorites";

// Bounds. Names are truncated in the strip anyway, and a collection
// list long enough to overflow the toolbar would make the tabs
// useless as navigation.
var COLLECTION_NAME_MAX = 40;
var COLLECTIONS_MAX = 24;

var collections = [];
// null means the All view, which is not a collection: it is every run
// on disk, and it has no membership to add to or remove from.
var activeCollectionId = null;
// The run whose chooser is open, and the collection staged for the
// delete confirmation. Both null when their dialog is closed.
var chooserRunId = null;
var pendingCollectionDelete = null;

// Read the stored collections, discarding anything malformed. A
// corrupt value degrades to no collections rather than throwing: this
// runs during boot, and the table is worth more than the tabs.
function loadCollections() {
  collections = [];
  var raw = null;
  try {
    raw = localStorage.getItem(COLLECTIONS_KEY);
  } catch (_e) {
    return;
  }
  if (!raw) {
    return;
  }
  var parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch (_e) {
    return;
  }
  if (!Array.isArray(parsed)) {
    return;
  }
  for (var i = 0; i < parsed.length; i++) {
    var entry = sanitizeCollection(parsed[i]);
    if (entry !== null) {
      collections.push(entry);
    }
  }
}

// One stored entry, or null when it is not one. Validated on read
// rather than trusted because this file is shared with a server that
// deliberately passes shapes it does not recognize straight through.
function sanitizeCollection(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  if (typeof entry.id !== "string" || entry.id === "") {
    return null;
  }
  var runs = [];
  if (Array.isArray(entry.runs)) {
    for (var i = 0; i < entry.runs.length; i++) {
      if (typeof entry.runs[i] === "string") {
        runs.push(entry.runs[i]);
      }
    }
  }
  return {
    id: entry.id,
    name: typeof entry.name === "string" && entry.name !== ""
      ? entry.name
      : entry.id,
    runs: runs,
  };
}

// Write through to the server (see persistSet) so collections survive
// a desktop restart, where the window origin can change and partition
// localStorage.
function saveCollections() {
  persistSet(COLLECTIONS_KEY, JSON.stringify(collections));
}

function findCollection(id) {
  for (var i = 0; i < collections.length; i++) {
    if (collections[i].id === id) {
      return collections[i];
    }
  }
  return null;
}

// Favorites, creating it if this is the first star. Returns null when
// the collection cap is already reached, which the caller reports
// rather than silently dropping the click.
function ensureFavorites() {
  var favorites = findCollection(FAVORITES_ID);
  if (favorites) {
    return favorites;
  }
  if (collections.length >= COLLECTIONS_MAX) {
    return null;
  }
  favorites = {
    id: FAVORITES_ID,
    name: FAVORITES_NAME,
    runs: [],
  };
  // First, so the tab it creates lands where a user expects the
  // default one to be rather than after their own collections.
  collections.unshift(favorites);
  return favorites;
}

// Whether a run is filed anywhere. What the filled star reports, so
// it answers "did I save this" rather than "is this a favorite": a
// run filed only under Papers is still saved.
function runIsCollected(runId) {
  for (var i = 0; i < collections.length; i++) {
    if (collections[i].runs.indexOf(runId) !== -1) {
      return true;
    }
  }
  return false;
}

function collectionHasRun(collection, runId) {
  return collection.runs.indexOf(runId) !== -1;
}

// Add or remove one run from one collection. Returns whether anything
// changed, so callers can skip a write and a re-render.
function setRunMembership(collectionId, runId, member) {
  var collection = findCollection(collectionId);
  if (!collection) {
    return false;
  }
  var at = collection.runs.indexOf(runId);
  if (member && at === -1) {
    collection.runs.push(runId);
    return true;
  }
  if (!member && at !== -1) {
    collection.runs.splice(at, 1);
    return true;
  }
  return false;
}

// The star's plain click: file to Favorites, or take it back out. One
// click, no dialog, because the common case is deciding a run is
// worth keeping and that decision should cost nothing.
function toggleFavorite(runId) {
  if (runIsCollected(runId)) {
    var changed = false;
    for (var i = 0; i < collections.length; i++) {
      if (setRunMembership(collections[i].id, runId, false)) {
        changed = true;
      }
    }
    if (changed) {
      afterCollectionsChanged();
    }
    return;
  }
  if (ensureFavorites() === null) {
    showToast(
      "Collection limit reached (" + COLLECTIONS_MAX + ")"
    );
    return;
  }
  setRunMembership(FAVORITES_ID, runId, true);
  afterCollectionsChanged();
}

// Persist, then repaint everything that reads membership: the tabs
// (their counts changed), and the table (its stars, and its rows if a
// collection is the active view).
function afterCollectionsChanged() {
  saveCollections();
  renderCollectionTabs();
  renderTable();
}

// Runs the table should show. The All view is every run; a collection
// is its members, in the table's own sort order rather than the order
// they were filed, so switching tabs does not also change the sort.
function visibleRuns() {
  if (activeCollectionId === null) {
    return allRuns;
  }
  var collection = findCollection(activeCollectionId);
  if (!collection) {
    // The collection was deleted while active. Fall back to All
    // rather than show an empty table with no way to tell why.
    activeCollectionId = null;
    return allRuns;
  }
  return allRuns.filter(function (run) {
    return collectionHasRun(collection, run.run_id);
  });
}

// How many of a collection's runs actually exist. Counted against
// allRuns rather than taken from runs.length so the tab cannot claim
// more than the table can show, which matters in the window between a
// delete and the next hydrate.
function collectionPresentCount(collection) {
  var present = 0;
  for (var i = 0; i < allRuns.length; i++) {
    if (collectionHasRun(collection, allRuns[i].run_id)) {
      present++;
    }
  }
  return present;
}

function renderCollectionTabs() {
  if (!collectionTabs) {
    return;
  }
  collectionTabs.innerHTML = "";
  collectionTabs.appendChild(
    buildCollectionTab(null, "All", allRuns.length)
  );
  for (var i = 0; i < collections.length; i++) {
    collectionTabs.appendChild(
      buildCollectionTab(
        collections[i],
        collections[i].name,
        collectionPresentCount(collections[i])
      )
    );
  }
  collectionTabs.appendChild(buildCollectionAddButton());
}

function buildCollectionTab(collection, name, count) {
  var id = collection ? collection.id : null;
  var tab = document.createElement("button");
  tab.type = "button";
  tab.className = "collection-tab";
  tab.setAttribute("role", "tab");
  if (id === activeCollectionId) {
    tab.classList.add("is-active");
  }
  tab.setAttribute(
    "aria-selected", id === activeCollectionId ? "true" : "false"
  );
  if (id !== null) {
    tab.setAttribute("data-collection-id", id);
  }
  tab.title = name;

  var label = document.createElement("span");
  label.className = "collection-tab-name";
  label.textContent = name;
  tab.appendChild(label);

  var countEl = document.createElement("span");
  countEl.className = "collection-tab-count";
  countEl.textContent = String(count);
  tab.appendChild(countEl);

  // All is a view, so it has no name to change and nothing to delete.
  if (id !== null) {
    tab.appendChild(
      buildTabIcon("rename", "Rename", COLLECTION_RENAME_SVG)
    );
    tab.appendChild(
      buildTabIcon("delete", "Delete", COLLECTION_DELETE_SVG)
    );
  }
  return tab;
}

var COLLECTION_RENAME_SVG =
  '<svg viewBox="0 0 24 24" width="10" height="10" fill="none"'
  + ' stroke="currentColor" stroke-width="2.2"'
  + ' stroke-linecap="round" stroke-linejoin="round"'
  + ' aria-hidden="true"><path d="M12 20h9"/>'
  + '<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';

var COLLECTION_DELETE_SVG =
  '<svg viewBox="0 0 24 24" width="10" height="10" fill="none"'
  + ' stroke="currentColor" stroke-width="2.2"'
  + ' stroke-linecap="round" stroke-linejoin="round"'
  + ' aria-hidden="true"><path d="M18 6L6 18"/>'
  + '<path d="M6 6l12 12"/></svg>';

// Nested buttons are invalid HTML, so the tab's own icons are spans
// with a role. They are reached through the tab's click handler,
// which reads the action off the target.
function buildTabIcon(action, label, svg) {
  var icon = document.createElement("span");
  icon.className = "collection-tab-icon";
  icon.setAttribute("data-tab-action", action);
  icon.setAttribute("role", "button");
  icon.setAttribute("tabindex", "-1");
  icon.setAttribute("aria-label", label);
  icon.title = label;
  icon.innerHTML = svg;
  return icon;
}

function buildCollectionAddButton() {
  var add = document.createElement("button");
  add.type = "button";
  add.className = "collection-tab collection-tab-add";
  add.id = "btn-collection-add";
  add.title = "New collection";
  add.setAttribute("aria-label", "New collection");
  add.textContent = "+";
  return add;
}

// Replace a tab's label with an input, in place. Inline rather than
// in a dialog because renaming is a one-field edit and the strip is
// where the name is read, so this is the shortest path between
// seeing a bad name and having a better one.
//
// ``collection`` is null when creating, in which case committing adds
// a new collection instead of renaming one.
function beginCollectionNameEdit(tab, collection) {
  var input = document.createElement("input");
  input.type = "text";
  input.className = "collection-name-input";
  input.maxLength = COLLECTION_NAME_MAX;
  input.value = collection ? collection.name : "";
  input.placeholder = "Collection name";
  input.setAttribute("aria-label", "Collection name");
  tab.innerHTML = "";
  tab.appendChild(input);
  input.focus();
  input.select();

  var settled = false;
  function commit() {
    if (settled) {
      return;
    }
    settled = true;
    applyCollectionName(collection, input.value);
  }
  function cancel() {
    if (settled) {
      return;
    }
    settled = true;
    renderCollectionTabs();
  }

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancel();
    }
  });
  // Clicking away commits, matching the rename affordance everywhere
  // else in this app; Escape is the way to back out.
  input.addEventListener("blur", commit);
  // The tab is a button, so a click inside the input would otherwise
  // switch the active collection out from under the edit.
  input.addEventListener("click", function (e) {
    e.stopPropagation();
  });
}

// Commit a typed name. An empty one is a decision not to change
// anything rather than a request for a nameless collection.
function applyCollectionName(collection, raw) {
  var name = raw.trim().slice(0, COLLECTION_NAME_MAX);
  if (name === "") {
    renderCollectionTabs();
    return;
  }
  if (collection) {
    collection.name = name;
    afterCollectionsChanged();
    return;
  }
  var created = createCollection(name);
  if (created === null) {
    showToast(
      "Collection limit reached (" + COLLECTIONS_MAX + ")"
    );
    renderCollectionTabs();
    return;
  }
  // Switch to what was just made: creating a collection is almost
  // always the first half of filing something into it.
  activeCollectionId = created.id;
  afterCollectionsChanged();
}

// Add a collection under a name, or return null at the cap. Ids are
// derived from the name and disambiguated with a counter, so the
// stored value stays readable when inspected by hand.
function createCollection(name) {
  if (collections.length >= COLLECTIONS_MAX) {
    return null;
  }
  var base = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (base === "") {
    base = "collection";
  }
  var id = base;
  var suffix = 2;
  while (findCollection(id) !== null) {
    id = base + "-" + suffix;
    suffix++;
    if (suffix > COLLECTIONS_MAX + 2) {
      // Unreachable while ids are unique per collection and the cap
      // holds, but the loop must be bounded regardless.
      return null;
    }
  }
  var collection = { id: id, name: name, runs: [] };
  collections.push(collection);
  return collection;
}

function openCollectionDeleteModal(collection) {
  pendingCollectionDelete = collection.id;
  if (colDeleteLabel) {
    colDeleteLabel.textContent =
      "\u201c" + collection.name + "\u201d ("
      + collectionPresentCount(collection)
      + " runs)";
  }
  modalCollectionDelete.classList.remove("hidden");
}

function closeCollectionDeleteModal() {
  pendingCollectionDelete = null;
  modalCollectionDelete.classList.add("hidden");
}

function confirmCollectionDelete() {
  var id = pendingCollectionDelete;
  closeCollectionDeleteModal();
  if (id === null) {
    return;
  }
  deleteCollection(id);
}

// Remove a collection. The runs in it are untouched: this deletes a
// label, not data, which is why it is a plain confirm rather than the
// same danger copy the run delete carries.
function deleteCollection(id) {
  var kept = [];
  for (var i = 0; i < collections.length; i++) {
    if (collections[i].id !== id) {
      kept.push(collections[i]);
    }
  }
  collections = kept;
  if (activeCollectionId === id) {
    activeCollectionId = null;
  }
  afterCollectionsChanged();
}

// The caret's dialog: every collection with a checkbox, plus a field
// to make another. Opened from the row rather than from the detail
// panel so filing a run never costs opening it.
function openCollectionChooser(runId) {
  chooserRunId = runId;
  if (collectionsRunLabel) {
    collectionsRunLabel.textContent = runPath(runId);
  }
  renderCollectionChoices();
  if (newCollectionName) {
    newCollectionName.value = "";
  }
  setCollectionsNote("");
  modalCollections.classList.remove("hidden");
}

function closeCollectionChooser() {
  chooserRunId = null;
  modalCollections.classList.add("hidden");
}

function renderCollectionChoices() {
  if (!collectionChoices) {
    return;
  }
  collectionChoices.innerHTML = "";
  if (collections.length === 0) {
    var empty = document.createElement("div");
    empty.className = "collection-empty";
    empty.textContent =
      "No collections yet. Name one below to start.";
    collectionChoices.appendChild(empty);
    return;
  }
  for (var i = 0; i < collections.length; i++) {
    collectionChoices.appendChild(
      buildCollectionChoice(collections[i])
    );
  }
}

function buildCollectionChoice(collection) {
  var row = document.createElement("label");
  row.className = "collection-choice";

  var box = document.createElement("input");
  box.type = "checkbox";
  box.className = "app-checkbox";
  box.checked = collectionHasRun(collection, chooserRunId);
  box.setAttribute("data-collection-id", collection.id);
  row.appendChild(box);

  var name = document.createElement("span");
  name.textContent = collection.name;
  row.appendChild(name);

  var count = document.createElement("span");
  count.className = "collection-choice-count";
  count.textContent =
    collectionPresentCount(collection) + " runs";
  row.appendChild(count);
  return row;
}

// Tick or untick one collection for the open run. Applied immediately
// rather than on Done: the checkbox is the switch, and a dialog whose
// footer button is the one that commits invites closing it and
// wondering whether anything happened.
function onCollectionChoiceToggle(e) {
  var box = e.target.closest('input[type="checkbox"]');
  if (!box || chooserRunId === null) {
    return;
  }
  var id = box.getAttribute("data-collection-id");
  if (!id) {
    return;
  }
  setRunMembership(id, chooserRunId, box.checked);
  afterCollectionsChanged();
  renderCollectionChoices();
}

function onCreateCollectionFromChooser() {
  if (!newCollectionName) {
    return;
  }
  var name = newCollectionName.value.trim();
  if (name === "") {
    setCollectionsNote("Give the collection a name.", true);
    return;
  }
  var created = createCollection(name);
  if (created === null) {
    setCollectionsNote(
      "Collection limit reached ("
      + COLLECTIONS_MAX + ").",
      true
    );
    return;
  }
  // Filed straight away: naming a new collection from a run's own
  // dialog is asking for that run to go in it.
  if (chooserRunId !== null) {
    setRunMembership(created.id, chooserRunId, true);
  }
  newCollectionName.value = "";
  setCollectionsNote("");
  afterCollectionsChanged();
  renderCollectionChoices();
}

function setCollectionsNote(text, warn) {
  if (!collectionsNote) {
    return;
  }
  collectionsNote.textContent = text;
  collectionsNote.classList.toggle("is-warning", !!warn);
}

function aCollectionDialogIsOpen() {
  if (!modalCollections.classList.contains("hidden")) {
    return true;
  }
  return !modalCollectionDelete.classList.contains("hidden");
}

// ---- Render table ----

// LLaDA-only hyperparameter columns were dropped because
// DiffusionGemma rows leave them blank; those values still appear in
// the per-run detail panel.
var TABLE_KEYS = [
  "created_at", "model", "processor", "prompt",
  "elapsed_seconds",
];

function renderTable() {
  // The active collection narrows the rows before anything else runs,
  // so sorting and grouping see only what is on screen.
  var shown = visibleRuns();
  var sorted = sortRuns(shown);
  var groupKey = groupBySelect.value;
  var groups = groupRuns(sorted, groupKey);

  runsTbody.innerHTML = "";

  // Which "nothing here" message applies: no runs at all, or a
  // collection that has none of them.
  var inCollection = activeCollectionId !== null;
  runsEmpty.hidden = shown.length > 0 || inCollection;
  if (runsEmptyCollection) {
    runsEmptyCollection.hidden =
      shown.length > 0 || !inCollection;
  }
  if (shown.length === 0) {
    return;
  }

  for (var g = 0; g < groups.length; g++) {
    var group = groups[g];

    if (group.label !== null) {
      var gtr = document.createElement("tr");
      gtr.className = "group-header-row";
      var gtd = document.createElement("td");
      // check + star + TABLE_KEYS + has-diff + actions.
      gtd.colSpan = TABLE_KEYS.length + 4;
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
      if (checkedIds[run.run_id]) {
        tr.classList.add("row-checked");
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

      // Collecting sits beside selecting rather than beside
      // deleting: both are things you do to a row you have picked
      // out, and a star is read down the column, which suits the
      // left edge.
      var tdStar = document.createElement("td");
      tdStar.className = "col-star";
      tdStar.appendChild(buildRowStar(run.run_id));
      tdStar.appendChild(buildRowCollectCaret(run.run_id));
      tr.appendChild(tdStar);

      for (var k = 0; k < TABLE_KEYS.length; k++) {
        var td = document.createElement("td");
        td.textContent = displayVal(
          run, TABLE_KEYS[k]
        );
        // The leading data column (after the checkbox) carries the
        // "new run" dot slot, so the pulse sits at the front of the
        // row regardless of which column leads. A fixed-width slot
        // keeps the text aligned whether or not a dot is present.
        if (k === 0) {
          var slot = document.createElement("span");
          slot.className = "run-new-slot";
          if (overlaysIsNewRun(run.run_id)) {
            var newDot = document.createElement("span");
            newDot.className = "run-new-dot";
            newDot.setAttribute("aria-hidden", "true");
            slot.appendChild(newDot);
          }
          td.insertBefore(slot, td.firstChild);
        }
        if (TABLE_KEYS[k] === "prompt") {
          td.className = "col-prompt";
          td.title = run.prompt || "";
        }
        tr.appendChild(td);
      }

      // Edited marker: a plain accent checkmark for runs with a saved
      // original; blank otherwise (no negative marker). It was once
      // filled with the diffusion dot pattern, but at 16px that
      // texture only muddied the shape, and the column reads as a
      // status flag rather than a piece of the diffusion metaphor.
      var tdDiff = document.createElement("td");
      tdDiff.className = "col-edited";
      if (run.has_diff) {
        tdDiff.innerHTML =
          '<svg class="edited-check" viewBox="0 0 24 24"'
          + ' width="16" height="16" role="img"'
          + ' aria-label="Edited">'
          + '<title>Edited: diff vs original available</title>'
          + '<path d="M4.5 12.5 L9.5 17.5 L19.5 6.5" fill="none"'
          + ' stroke="var(--accent)" stroke-width="3.2"'
          + ' stroke-linecap="round" stroke-linejoin="round" />'
          + '</svg>';
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

// One star per row, always present and always filled when the run is
// filed somewhere. A star that only appeared on hover would leave the
// table unscannable, which defeats the point of collecting.
function buildRowStar(runId) {
  var collected = runIsCollected(runId);
  var star = document.createElement("button");
  star.type = "button";
  star.className = "row-star-btn";
  if (collected) {
    star.classList.add("is-collected");
  }
  star.setAttribute("data-run-id", runId);
  star.title = collected
    ? "Remove from collections"
    : "Add to Favorites";
  star.setAttribute("aria-label", star.title);
  star.setAttribute(
    "aria-pressed", collected ? "true" : "false"
  );
  star.innerHTML =
    '<svg viewBox="0 0 24 24" width="12" height="12"'
    + ' fill="none" stroke="currentColor" stroke-width="2"'
    + ' stroke-linecap="round" stroke-linejoin="round"'
    + ' aria-hidden="true"><path d="M12 3l2.9 5.9 6.6.9-4.8 4.6'
    + ' 1.2 6.5L12 17.8 6.1 20.9l1.2-6.5L2.5 9.8l6.6-.9z"/>'
    + "</svg>";
  return star;
}

// The way to reach a collection other than Favorites. Hover-only, so
// the row stays quiet until there is a reason to act on it.
function buildRowCollectCaret(runId) {
  var caret = document.createElement("button");
  caret.type = "button";
  caret.className = "row-collect-caret";
  caret.setAttribute("data-run-id", runId);
  caret.title = "Choose collections";
  caret.setAttribute("aria-label", "Choose collections");
  caret.innerHTML =
    '<svg viewBox="0 0 24 24" width="10" height="10"'
    + ' fill="none" stroke="currentColor" stroke-width="2.4"'
    + ' stroke-linecap="round" stroke-linejoin="round"'
    + ' aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>';
  return caret;
}

// ---- Detail panel ----

function showDetail(runId) {
  activeRunId = runId;
  comparePanel.hidden = true;
  detailPanel.classList.remove("hidden");

  // Opening a run clears its "new" dot (and decrements the generator's
  // count on the next visit). Remove just this row's dot in place.
  if (overlaysIsNewRun(runId)) {
    overlaysClearNewRun(runId);
    var openedRow = runsTbody.querySelector(
      'tr[data-run-id="' + runId + '"] .run-new-slot'
    );
    if (openedRow) {
      openedRow.textContent = "";
    }
  }

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

  html += processorMetaRow(run);

  html += tokenizerMetaRow(run);

  html += tokenizerVocabMetaRow(run);

  html += modelVocabMetaRow(run);

  html += contextMetaRows(run);

  html += elapsedMetaRows(run);

  detailMeta.innerHTML = html;

  renderTable();
  loadRunCharts(runId, run);
  loadRunOverlays(runId, run);
}

// An edited run has two totals worth reading: how long the run it
// branched from took end to end, and how long this one took, meaning
// the prefix it inherited up to the edit plus everything generated
// after. Reporting only the combined figure left no way to see
// whether an intervention cost time or saved it, which is the whole
// question an edit raises.
function elapsedMetaRows(run) {
  var edited = run.elapsed_seconds;
  if (edited === undefined || edited === null) {
    return "";
  }
  var original = run.original_elapsed_seconds;
  if (original === undefined || original === null) {
    return elapsedMetaRow("Elapsed", edited);
  }
  return elapsedMetaRow("Elapsed (original)", original)
    + elapsedMetaRow("Elapsed (edited)", edited);
}

function elapsedMetaRow(label, seconds) {
  return '<div class="meta-row">'
    + '<span class="meta-label">' + label + ':</span> '
    + '<span class="meta-value">'
    + Number(seconds).toFixed(2)
    + 's</span></div>';
}

// Which processor produced the run, on its own summary line. The
// label comes from the run itself, already normalized to GPU or CPU
// at save time, so a CPU run is not mislabelled. Older runs recorded
// neither field and fall back to the machine's current GPU, which is
// the same guess the timing header used to make.
function processorMetaRow(run) {
  var name = run.processor_name || gpuName;
  if (!name) {
    return "";
  }
  var label = run.processor === "CPU" ? "CPU" : "GPU";
  return '<div class="meta-row">'
    + '<span class="meta-label">' + label + ':</span> '
    + '<span class="meta-value">'
    + escHtml(String(name))
    + '</span></div>';
}

// Which tokenizer produced this run's ids. Read from the run's own
// metadata rather than from the resident model, so an old run still
// answers the question after its checkpoint has been swapped out or
// moved on. Absent on every run saved before the field existed,
// which is why this degrades to nothing rather than guessing: a
// wrong tokenizer name is worse than no tokenizer name.
//
// Bracket access because "class" is the payload's field name; see
// describe_tokenizer in worker_base.py. name_or_path is recorded
// too but not shown, since the Model row above already carries it.
function tokenizerMetaRow(run) {
  var tok = runTokenizer(run);
  var name = tok["class"];
  if (!name) {
    return "";
  }
  return metaRowHtml("Tokenizer", String(name));
}

// A row of its own rather than a parenthetical on the name. The two
// answer different questions, and this one is the number the entropy
// scale refers to, since its natural log is the ceiling on what any
// single position can carry.
function tokenizerVocabMetaRow(run) {
  var tok = runTokenizer(run);
  if (!tok.vocab_size) {
    return "";
  }
  return metaRowHtml(
    "Tokenizer vocab",
    Number(tok.vocab_size).toLocaleString()
  );
}

// The checkpoint's output width, next to the tokenizer's own count
// because the pair is the point: they differ wherever a checkpoint
// pads its embedding for alignment, and this larger figure is the one
// a candidate's rank is measured against.
function modelVocabMetaRow(run) {
  var tok = runTokenizer(run);
  if (!tok.model_vocab_size) {
    return "";
  }
  return metaRowHtml(
    "Model vocab",
    Number(tok.model_vocab_size).toLocaleString()
  );
}

// What the prompt cost and what it had to fit inside. Two rows rather
// than one ratio, because the ratio is only interesting when both
// numbers are visible: a 400-token prompt means one thing in a 4k
// window and another in a 128k one.
//
// The whole block is absent on runs saved before it existed, and the
// window alone is absent for a checkpoint that reported none, so each
// row is guarded separately rather than as a pair.
function contextMetaRows(run) {
  var context = run.context || {};
  var html = "";
  if (typeof context.prompt_tokens === "number") {
    html += metaRowHtml(
      "Prompt tokens",
      Number(context.prompt_tokens).toLocaleString()
    );
  }
  if (typeof context.context_length === "number") {
    html += metaRowHtml(
      "Context window",
      Number(context.context_length).toLocaleString()
    );
  }
  return html;
}

function runTokenizer(run) {
  var repro = run.reproducibility || {};
  return repro.tokenizer || {};
}

// The tokenizer of whichever run's overlay is on screen, for the
// candidate popover's footer. Deliberately the run's own and never a
// resident worker's: this page is routinely looking at a run whose
// checkpoint is not loaded at all.
function activeRunTokenizer() {
  for (var i = 0; i < allRuns.length; i++) {
    if (allRuns[i].run_id === activeRunId) {
      return runTokenizer(allRuns[i]);
    }
  }
  return {};
}

function metaRowHtml(label, value) {
  return '<div class="meta-row">'
    + '<span class="meta-label">' + escHtml(label) + ':</span> '
    + '<span class="meta-value">'
    + escHtml(value)
    + '</span></div>';
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

// Inline Chart.js plugin for the position-indexed entropy chart:
// a dashed vertical marker at each edited position, so a What If
// branch shows where the intervention happened and therefore where
// the shared prefix ends. Empty list is a no-op, so unedited runs
// draw nothing.
//
// Two hooks: the tint goes behind the bars (an edited column reads as
// touched even where its bar is short), the dashed line goes over
// them (a one-pixel bar would otherwise hide it).
function substitutionMarkerPlugin(positions) {
  return {
    id: "substitutionMarkers",
    beforeDatasetsDraw: function (chart) {
      if (!positions || positions.length === 0) { return; }
      var yScale = chart.scales.y;
      var ctx = chart.ctx;
      ctx.save();
      clipToChartArea(ctx, chart.chartArea);
      ctx.fillStyle = EDIT_TINT;
      for (var i = 0; i < positions.length; i++) {
        var span = entropyColumnSpan(chart, positions[i]);
        if (span) {
          ctx.fillRect(
            span.left,
            yScale.top,
            span.width,
            yScale.bottom - yScale.top
          );
        }
      }
      ctx.restore();
    },
    afterDatasetsDraw: function (chart) {
      if (!positions || positions.length === 0) { return; }
      var xScale = chart.scales.x;
      var yScale = chart.scales.y;
      var ctx = chart.ctx;
      ctx.save();
      clipToChartArea(ctx, chart.chartArea);
      ctx.strokeStyle = EDIT_COLOR;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (var i = 0; i < positions.length; i++) {
        var x = xScale.getPixelForValue(positions[i]);
        ctx.beginPath();
        ctx.moveTo(x, yScale.top);
        ctx.lineTo(x, yScale.bottom);
        ctx.stroke();
      }
      ctx.restore();
    },
  };
}

// Confine drawing to the plotting area. Chart.js clips each dataset
// for us, but the dataset-level hooks run outside that clip, so a
// marker for a position zoom or pan has pushed off screen would
// otherwise paint over the axes.
function clipToChartArea(ctx, area) {
  ctx.beginPath();
  ctx.rect(
    area.left,
    area.top,
    area.right - area.left,
    area.bottom - area.top
  );
  ctx.clip();
}

// Pixel span of one position's column on the entropy chart. A long
// run puts about a pixel per bar, so a highlight drawn at the true
// bar width would be invisible; the floor mirrors the generator's
// profile. Reads the laid-out element rather than the scale so it
// stays correct under zoom and pan. Falls through the datasets
// because a shorter original run has no element at a high index.
function entropyColumnSpan(chart, index) {
  var bar = null;
  for (var di = 0; di < chart.data.datasets.length; di++) {
    var meta = chart.getDatasetMeta(di);
    if (meta && meta.data && meta.data[index]) {
      bar = meta.data[index];
      break;
    }
  }
  if (!bar) {
    return null;
  }
  var props = bar.getProps(["x", "width"], true);
  var width = Math.max(2, props.width);
  return { left: props.x - width / 2, width: width };
}

// Inline Chart.js plugin: a faint full-height guide behind the bar
// under the pointer, so a one-pixel column is findable at a glance.
// The bar itself brightens via the dataset's hoverBackgroundColor,
// which (unlike a hand-drawn bar) still honors the crossfade alpha.
var entropyHoverPlugin = {
  id: "entropyHover",
  beforeDatasetsDraw: function (chart) {
    var active = chart.getActiveElements();
    if (!active || active.length === 0) { return; }
    var span = entropyColumnSpan(chart, active[0].index);
    if (!span) { return; }
    var yScale = chart.scales.y;
    var ctx = chart.ctx;
    ctx.save();
    clipToChartArea(ctx, chart.chartArea);
    ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
    ctx.fillRect(
      span.left,
      yScale.top,
      span.width,
      yScale.bottom - yScale.top
    );
    ctx.restore();
  },
};

// Inline Chart.js plugin: the chart-to-token half of the
// cross-highlight, so a tall warm bar can be read back to the word
// the model was torn over.
//
// This has to be a plugin rather than the options.onHover callback,
// which Chart.js only fires while the pointer is inside chartArea.
// Leaving through the axis gutter or off the canvas therefore never
// delivered the empty-elements call that clears the token, and the
// last position stayed lit. afterEvent is notified for every event
// in options.events (mouseout included) and runs after the active
// set is recomputed, so getActiveElements is authoritative here.
var tokenLinkPlugin = {
  id: "tokenLink",
  afterEvent: function (chart) {
    var active = chart.getActiveElements();
    var pos = active.length > 0 ? active[0].index : null;
    setTokenHighlight(pos);
    // The bar has no span behind it, so the strip takes the layer
    // the crossfade favors: the one a token hover would land on.
    setTokenMetricsHover(pos, null);
  },
};

// Inline Chart.js plugin: crossfade the entropy chart's two layers
// from the run-level compareBlend. Applied as canvas alpha at draw
// time rather than by rewriting several hundred color strings per
// slider step, which also keeps the entropy ramp itself untouched.
// No-op on an unedited run, where there is only one layer to show.
var compareBlendPlugin = {
  id: "compareBlend",
  beforeDatasetDraw: function (chart, args) {
    if (chart.data.datasets.length < 2) { return; }
    var alpha = (args.index === 0)
      ? 1 - compareBlend
      : compareBlend;
    chart.ctx.save();
    chart.ctx.globalAlpha = alpha;
  },
  // Guarded identically to the save above, so the pair can never
  // come apart and leak canvas state into the next dataset.
  afterDatasetDraw: function (chart) {
    if (chart.data.datasets.length < 2) { return; }
    chart.ctx.restore();
  },
};

// How much the run crossfade is currently borrowing the line charts,
// from 0 (the pins decide) to 1 (the slider decides). Raised while
// the slider is being dragged and eased back on release, so the
// slider never permanently governs a chart it does not sit next to.
// Armed on press but engaged only once the thumb actually moves, so
// a press that never becomes a drag leaves the charts alone.
var scrubWeight = 0;
var scrubEaseHandle = null;
var scrubArmed = false;
var scrubEngaged = false;

// Long enough to read as the charts handing control back, short
// enough not to sit between the release and the answer.
var SCRUB_EASE_MS = 180;

// What a line chart's dataset draws at, blending the resting answer
// (its pins) with the drag answer (the crossfade). At rest this is
// exactly the pin state; mid-drag it is exactly the slider.
function seriesBlendAlpha(name, index) {
  var state = linePinState[name];
  var pinned = state
    ? !!state[index === 0 ? "original" : "edited"]
    : true;
  var pinAlpha = pinned ? 1 : 0;
  if (scrubWeight === 0) {
    return pinAlpha;
  }
  var blendAlpha = (index === 0)
    ? 1 - compareBlend
    : compareBlend;
  return pinAlpha
    + (blendAlpha - pinAlpha) * scrubWeight;
}

// The line-chart counterpart to compareBlendPlugin. Alpha at draw
// time for the same reason: one number per dataset instead of
// rewriting colors, and the segment coloring underneath stays as it
// is. No-op on a run with only one series to show.
function seriesBlendPlugin(name) {
  return {
    id: "seriesBlend-" + name,
    beforeDatasetDraw: function (chart, args) {
      if (chart.data.datasets.length < 2) { return; }
      chart.ctx.save();
      chart.ctx.globalAlpha = seriesBlendAlpha(
        name, args.index
      );
    },
    // Guarded identically to the save above, so the pair can never
    // come apart and leak canvas state into the next dataset.
    afterDatasetDraw: function (chart) {
      if (chart.data.datasets.length < 2) { return; }
      chart.ctx.restore();
    },
  };
}

// Alpha for the difference band. It describes a relationship between
// the two runs rather than either one of them, so it follows
// whichever is closer to invisible: a band bounded by a line that is
// not there is a smear with no reading in it.
function bandAlpha(name) {
  return Math.min(
    seriesBlendAlpha(name, 0),
    seriesBlendAlpha(name, 1)
  );
}

// The area between the two runs, colored by whichever bounds it from
// above: the branch's own hue where the branch leads, the original's
// grey where it does not. That rule needs no legend and calls
// neither direction good nor bad, which matters because "higher"
// means slower on the timing chart and better on confidence. The
// runs share their prefix exactly, so the band is empty until the
// edit and opens up only where the intervention actually reached.
//
// Scriptable because its alpha tracks the pins and the crossfade,
// and every path that moves either already calls chart.update, which
// re-resolves this. Note the alpha lives in the color and not in
// canvas state: the Filler plugin is registered globally, so it
// draws on beforeDatasetDraw ahead of seriesBlendPlugin's inline
// hook and would never see a globalAlpha set there.
function compareBandFill(name, hue) {
  return function () {
    var alpha = bandAlpha(name);
    return {
      target: 0,
      above: withAlpha(hue, BAND_ALPHA_EDITED * alpha),
      below: withAlpha(
        COMPARE_ORIGINAL_COLOR, BAND_ALPHA_ORIGINAL * alpha
      ),
    };
  };
}

// The alpha a dataset is actually drawn at, resolved from the canvas
// so the shared burn-through plugin can honor it without knowing
// which chart it is decorating. Charts with a single series, and the
// ones outside the run comparison, resolve to fully opaque.
function chartSeriesAlpha(chart, index) {
  if (chart.data.datasets.length < 2) { return 1; }
  var id = chart.canvas ? chart.canvas.id : "";
  if (id === "chart-timing") {
    return seriesBlendAlpha("timing", index);
  }
  if (id === "chart-tps") {
    return seriesBlendAlpha("tps", index);
  }
  if (id === "chart-confidence") {
    return seriesBlendAlpha("confidence", index);
  }
  return 1;
}

// A run faded out to nothing still reports values to the tooltip, so
// a row is dropped once its series is effectively invisible. The
// floor is above zero to also catch a crossfade parked at an end.
function seriesRowVisible(name, item) {
  return seriesBlendAlpha(
    name, item.datasetIndex
  ) > 0.02;
}

// The branch is always the last dataset: an original series, when
// there is one, is inserted ahead of it.
function isEditedDataset(ctx) {
  var last = ctx.chart.data.datasets.length - 1;
  return ctx.datasetIndex === last;
}

// Redraw both line charts without animating: only alpha changed, and
// the drag needs every frame to land immediately.
function updateLineCharts() {
  if (chartTiming) {
    chartTiming.update("none");
  }
  if (chartTps) {
    chartTps.update("none");
  }
  if (chartConfidence) {
    chartConfidence.update("none");
  }
}

// Dim both charts' pins while the slider is driving them, so the
// override is visible without changing what the pins hold.
function setPinsPreviewing(previewing) {
  var buttons = document.querySelectorAll(
    ".compare-pin-btn"
  );
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].classList.toggle(
      "is-previewing", previewing
    );
  }
}

// Ease rather than snap, in both directions. An instant handover
// reads as a glitch where a short settle reads as the charts lending
// themselves out and taking themselves back.
function easeScrubWeight(target) {
  cancelScrubEase();
  var from = scrubWeight;
  if (from === target) { return; }
  var start = performance.now();
  var step = function (now) {
    var t = (now - start) / SCRUB_EASE_MS;
    if (t < 1) {
      var eased = 1 - Math.pow(1 - t, 3);
      scrubWeight = from + (target - from) * eased;
      scrubEaseHandle = requestAnimationFrame(step);
    } else {
      scrubWeight = target;
      scrubEaseHandle = null;
    }
    updateLineCharts();
  };
  scrubEaseHandle = requestAnimationFrame(step);
}

function cancelScrubEase() {
  if (scrubEaseHandle !== null) {
    cancelAnimationFrame(scrubEaseHandle);
    scrubEaseHandle = null;
  }
}

// Pressing the slider only arms the preview. Engaging here instead
// would fade the charts the instant the thumb is touched, before the
// user has asked for anything.
function armBlendScrub() {
  scrubArmed = true;
}

// Called from the slider's input handler, so the charts are borrowed
// on the first actual movement of a press. Pointer drags only: arrow
// keys on a focused slider produce input events with no press to
// arm them, so keyboard adjustments move the tokens and the entropy
// bars while the line charts stay on their pins.
function engageBlendScrub() {
  if (!scrubArmed || scrubEngaged) { return; }
  scrubEngaged = true;
  setPinsPreviewing(true);
  easeScrubWeight(1);
}

function endBlendScrub() {
  scrubArmed = false;
  if (!scrubEngaged) { return; }
  scrubEngaged = false;
  setPinsPreviewing(false);
  easeScrubWeight(0);
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
  tooltipEnabled.tps = true;
  tooltipEnabled.confidence = true;
  tooltipEnabled.entropy = true;
  var btns = document.querySelectorAll(
    ".tooltip-toggle-btn"
  );
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove("is-off");
    setEyeSlash(btns[i], false);
  }
}

// Which of the two runs each line chart draws. Both on by default,
// so an edited run opens with the comparison already visible.
// Exactly one of three states (original, edited, both) holds at any
// time: a chart drawing neither has nothing to read, so the last lit
// pin cannot be turned off.
var linePinState = {
  timing: { original: true, edited: true },
  tps: { original: true, edited: true },
  confidence: { original: true, edited: true },
};

// Each newly-opened run starts with both runs pinned on, and with
// the pins hidden until a chart actually renders a pre-edit series.
// Hiding here rather than only in the render functions covers the
// runs where a chart bails early for want of data, which would
// otherwise leave the previous run's pins standing.
function resetComparePins() {
  var names = Object.keys(linePinState);
  for (var i = 0; i < names.length; i++) {
    linePinState[names[i]].original = true;
    linePinState[names[i]].edited = true;
    updateComparePins(names[i], false);
  }
}

function comparePinsBothOn(state) {
  return state.original && state.edited;
}

// Reflect a chart's pin state onto its two buttons. The pin that is
// the only one lit is marked locked, so the dead click reads as
// unavailable before it is made rather than being swallowed.
function refreshComparePins(name) {
  var state = linePinState[name];
  if (!state) { return; }
  var buttons = document.querySelectorAll(
    '.compare-pin-btn[data-chart="' + name + '"]'
  );
  for (var i = 0; i < buttons.length; i++) {
    var btn = buttons[i];
    var on = !!state[btn.getAttribute("data-series")];
    btn.classList.toggle("is-on", on);
    btn.classList.toggle(
      "is-locked", on && !comparePinsBothOn(state)
    );
    btn.setAttribute(
      "aria-pressed", on ? "true" : "false"
    );
  }
}

// With only one run there is nothing to pin, so the pair is hidden
// for unedited runs and for those saved without the pre-edit signal.
function updateComparePins(name, hasOriginal) {
  var group = document.querySelector(
    '.compare-pins[data-chart="' + name + '"]'
  );
  if (group) {
    group.hidden = !hasOriginal;
  }
  refreshComparePins(name);
}

// ---- The Timing slot's two pages ----
//
// Elapsed time and tokens per second are the same measurement read
// two ways, so they share one section's worth of vertical space and a
// pager rather than each claiming a chart slot of its own.
var timingPage = "elapsed";
// Which pages the open run can actually draw. A run saved before a
// signal existed may have one and not the other, and flipping to a
// blank panel would read as a bug rather than as an absence.
var timingPageReady = { elapsed: false, tps: false };

function setTimingPage(page) {
  if (page !== "elapsed" && page !== "tps") {
    return;
  }
  timingPage = page;
  applyTimingPage();
}

function timingPageActive() {
  if (timingPageReady[timingPage]) {
    return timingPage;
  }
  if (timingPageReady.elapsed) {
    return "elapsed";
  }
  if (timingPageReady.tps) {
    return "tps";
  }
  return null;
}

function applyTimingPage() {
  var active = timingPageActive();
  var tpsSection = document.getElementById("tps-section");
  timingSection.hidden = active !== "elapsed";
  if (tpsSection) {
    tpsSection.hidden = active !== "tps";
  }
  refreshTimingPagers(active);
  // Both charts were built while both sections were visible; hiding
  // one changes the height the survivor has to fill.
  var chart = active === "tps" ? chartTps : chartTiming;
  if (chart) {
    chart.resize();
  }
}

function refreshTimingPagers(active) {
  var both =
    timingPageReady.elapsed && timingPageReady.tps;
  var pagers = document.querySelectorAll(
    ".chart-title-group .alt-pager"
  );
  for (var i = 0; i < pagers.length; i++) {
    pagers[i].hidden = !both;
  }
  var buttons = document.querySelectorAll(
    "[data-timing-page]"
  );
  for (var j = 0; j < buttons.length; j++) {
    buttons[j].disabled =
      buttons[j].getAttribute("data-timing-page") === active;
  }
}

function wireTimingPager() {
  var buttons = document.querySelectorAll(
    "[data-timing-page]"
  );
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener(
      "click",
      function (event) {
        setTimingPage(
          event.currentTarget.getAttribute("data-timing-page")
        );
      }
    );
  }
}

// Autoregressive runs have no masked canvas, so the convergence chart
// (percent resolved per frame) would flatline at 100%; it is hidden
// for them while timing and confidence stay, and Entropy by Position
// takes its slot.
function runIsAutoregressive(run) {
  return !!(run && run.model_type === "autoregressive");
}

// The compare metrics payload carries no model fields, so resolve a
// run's type from the already-loaded run list by id.
function runIdIsAutoregressive(runId) {
  for (var i = 0; i < allRuns.length; i++) {
    if (allRuns[i].run_id === runId) {
      return runIsAutoregressive(allRuns[i]);
    }
  }
  return false;
}

function loadRunCharts(runId, run) {
  fetchMetrics(runId).then(function (data) {
    if (data.error) { return; }

    resetTooltipToggles();
    resetComparePins();

    chartConvergence = destroyChart(
      chartConvergence
    );
    chartTiming = destroyChart(chartTiming);
    chartTps = destroyChart(chartTps);
    chartConfidence = destroyChart(chartConfidence);
    timingPageReady.elapsed = false;
    timingPageReady.tps = false;

    var remaskEdits = data.remask_edits || [];
    var remaskSet = buildRemaskFrameSet(
      remaskEdits
    );

    var convergenceSection = document.getElementById(
      "convergence-section"
    );
    if (runIsAutoregressive(run)) {
      if (convergenceSection) {
        convergenceSection.hidden = true;
      }
    } else {
      if (convergenceSection) {
        convergenceSection.hidden = false;
      }
      renderConvergenceChart(data, remaskSet);
    }
    renderTimingChart(data, remaskSet);
    renderTpsChart(data, remaskSet, runIsAutoregressive(run));
    // Last, so both charts have been sized while visible and the
    // slot settles on one page in the same paint.
    applyTimingPage();
    renderConfidenceChart(data);
    // The fourth chart, Entropy by Position, is built in
    // loadRunOverlays instead: it needs per-token records from the
    // frames payload, which that function already fetches.
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

// Index of the last frame carrying token records (mirrors
// overlayFinalFrame). Used as the scrubber's default position so the
// viewer opens on the resolved output. Returns 0 when none qualify.
function overlayFinalFrameIndex(frames) {
  if (!frames) {
    return 0;
  }
  for (var i = frames.length - 1; i >= 0; i--) {
    if (frames[i] && frames[i].length > 0) {
      return i;
    }
  }
  return 0;
}

// The token array at scrubber frame ``index`` (guarded). An empty or
// out-of-range frame yields null, which renders as a blank canvas
// (e.g. an all-masked early frame with no records).
function overlayFrameAt(index) {
  if (!overlayData || !overlayData.frames) {
    return null;
  }
  var frames = overlayData.frames;
  if (index < 0 || index >= frames.length) {
    return null;
  }
  return frames[index];
}

// Whether a frame series carries per-token entropy. Checked on the
// final frame, which is the series' ground truth. Split out from
// overlayEntropyAvailable so the pre-edit snapshot can be tested the
// same way: it was saved by the same code path but predates the
// signal on older runs.
function framesHaveEntropy(frames) {
  var list = frames || [];
  var final = list[overlayFinalFrameIndex(list)];
  if (!final) {
    return false;
  }
  for (var i = 0; i < final.length; i++) {
    if (final[i] && typeof final[i].e === "number") {
      return true;
    }
  }
  return false;
}

// Whether the saved run carries per-token entropy.
function overlayEntropyAvailable(data) {
  return framesHaveEntropy(data && data.frames);
}

// Per-position candidate sets for the open run, or an empty list.
function overlayAlternatives() {
  if (!overlayData || !overlayData.alternatives) {
    return [];
  }
  return overlayData.alternatives;
}

// The same for the pre-edit run. Present only for branches saved
// since the snapshot began carrying its candidates, so an older
// edited run pages through nothing.
function overlayOriginalAlternatives() {
  if (!overlayData || !overlayData.original_alternatives) {
    return [];
  }
  return overlayData.original_alternatives;
}

// Whether this position has a candidate set from each run to page
// between. Both runs record the same set left of the divergence
// point, where the branch copies its prefix verbatim, so a pager
// there would flip between two identical lists.
function altsPageable(pos) {
  var divergence = overlayData
    ? divergencePosition(overlayData) : null;
  if (divergence === null || pos < divergence) {
    return false;
  }
  var original = overlayOriginalAlternatives()[pos];
  var edited = overlayAlternatives()[pos];
  return !!(
    original && original.length > 0
    && edited && edited.length > 0
  );
}

// ---- Candidate popover (read-only mirror of the generator's) ----

function hideAltsPopover() {
  if (!altsPopover) {
    return;
  }
  altsPopover.hidden = true;
  altsPopover.textContent = "";
  altsPopoverPos = null;
  altsPopoverPage = null;
  // Same reason as in renderAltsPopover: the rows go without firing
  // the mouseleave that would have cleared their readout. Needed here
  // too, because scroll and resize close the popover on their own
  // rather than through a pointer leaving it.
  setCandidateMetricsHover(null);
}

// Which run's candidates a pageable position opens on: the one the
// crossfade is currently favoring, so the popover agrees with what
// the tokens and bars are showing. Both pages stay reachable through
// the arrows either way, so the midpoint chooses a default rather
// than gating access.
function defaultAltsPage() {
  return compareBlend < 0.5 ? "original" : "edited";
}

function showAltsPopover(pos, span) {
  altsPopoverPage = altsPageable(pos)
    ? defaultAltsPage() : null;
  renderAltsPopover(pos, span);
}

// Flip pages in place. Rendered without an anchor deliberately: the
// two pages can differ in height, and re-placing the box under the
// pointer that just clicked an arrow can slide it out from under that
// pointer, firing the mouseleave that closes it.
function setAltsPage(page) {
  if (altsPopoverPos === null) {
    return;
  }
  altsPopoverPage = page;
  renderAltsPopover(altsPopoverPos, null);
}

// With an anchor span, placed above the token (or below when that
// would overflow). Without one, left where it already sits.
function renderAltsPopover(pos, span) {
  if (!altsPopover) {
    return;
  }
  var original = altsPopoverPage === "original";
  var alts = original
    ? overlayOriginalAlternatives()[pos]
    : overlayAlternatives()[pos];
  if (!alts || alts.length === 0) {
    hideAltsPopover();
    return;
  }
  // Each page marks the token its own run drew, so the Original page
  // does not mark the branch's substitution as chosen.
  var frames = original
    ? (overlayData.original_frames || [])
    : (overlayData.frames || []);
  var frame = overlayClampedFrame(frames);
  var chosen = frame && frame[pos] ? frame[pos].id : null;

  // Discarding the rows discards their pending mouseleave: a removed
  // node never fires one, so a readout for a row that no longer
  // exists would sit in the strip until the next hover.
  setCandidateMetricsHover(null);
  altsPopover.textContent = "";
  altsPopover.appendChild(
    overlaysBuildAltHeading(pos, altsPopoverPage, setAltsPage)
  );
  for (var i = 0; i < alts.length; i++) {
    altsPopover.appendChild(
      overlaysBuildAltRow(
        alts[i], chosen, setCandidateMetricsHover, i
      )
    );
  }
  var tokenizer = overlaysBuildAltTokenizer(activeRunTokenizer());
  if (tokenizer) {
    altsPopover.appendChild(tokenizer);
  }
  altsPopover.classList.remove("alt-pickable");

  // Unhide before measuring: the height is unknown while hidden.
  altsPopover.hidden = false;
  if (span) {
    var rect = span.getBoundingClientRect();
    var box = altsPopover.getBoundingClientRect();
    altsPopover.style.left =
      overlaysPopoverLeft(rect, box) + "px";
    altsPopover.style.top =
      overlaysPopoverTop(
        rect, box, overlayOutput.getBoundingClientRect().top
      ) + "px";
  }
  altsPopoverPos = pos;
}

// A frame series at the scrubber's index, clamped to its end. The two
// runs can differ in length, so the snapshot may stop short.
function overlayClampedFrame(frames) {
  if (!frames || frames.length === 0) {
    return null;
  }
  var index = Math.min(overlayFrameIndex, frames.length - 1);
  return frames[index] || null;
}

// Candidate rows are built by overlaysBuildAltRow in overlays.js;
// this page's own copy was identical to the generator's and both
// needed the same hover wiring, so they share one.

function loadRunOverlays(runId, run) {
  overlayData = null;
  overlayCommitSteps = null;
  overlayOriginalCommitSteps = null;
  overlayDiffData = null;
  overlayIsAutoregressive = runIsAutoregressive(run);
  hideAltsPopover();
  // Torn down before the fetch, not inside it, so switching runs can
  // never leave the previous run's chart or crossfade on screen while
  // the new payload is in flight.
  clearEntropyChart();
  resetRunBlend(false);
  fetchFrames(runId).then(function (data) {
    if (!data || data.error) {
      showOverlayUnavailable();
      return;
    }
    var hasRecords = !!data.records_available;
    var hasDiff = overlayDiffAvailable(data);
    if (!hasRecords && !hasDiff) {
      showOverlayUnavailable();
      return;
    }
    overlayData = data;
    overlayViewer.hidden = false;
    overlayEmpty.hidden = true;
    overlayOutput.hidden = false;
    overlaySelectGroup.hidden = false;
    setOverlayDrawerOpen(false);
    resetRunBlend(hasDiff);
    // Mirror the generator: default to None; the drawer offers the
    // durable overlays (Heatmap for record runs, plus Commit Order and
    // Diff vs Original for diffusion runs with the required data).
    buildOverlaySelect(data);
    setupOverlayScrubber(data);
    setOverlayMode("none");
    renderEntropyChart(data);
  });
}

// Configure the per-frame scrubber for the loaded run. Opens on the
// final frame carrying records (the viewer's prior behavior); a run
// with a single usable frame keeps the scrubber hidden and disabled.
function setupOverlayScrubber(data) {
  var frames = data.frames || [];
  var maxIndex = frames.length > 0 ? frames.length - 1 : 0;
  overlayFrameIndex = overlayFinalFrameIndex(frames);
  if (!overlayScrubber) {
    return;
  }
  var hasMultiple = frames.length > 1;
  overlayScrubber.hidden = !hasMultiple;
  overlayScrubSlider.min = "0";
  overlayScrubSlider.max = String(maxIndex);
  overlayScrubSlider.value = String(overlayFrameIndex);
  overlayScrubSlider.disabled = !hasMultiple;
  updateOverlayScrubLabel();
}

// Clamp to range, sync the slider + label, and re-render the active
// overlay at the new frame.
function setOverlayFrame(index) {
  if (!overlayData || !overlayData.frames) {
    return;
  }
  var maxIndex = overlayData.frames.length > 0
    ? overlayData.frames.length - 1
    : 0;
  var clamped = Math.max(0, Math.min(index, maxIndex));
  overlayFrameIndex = clamped;
  if (overlayScrubSlider) {
    overlayScrubSlider.value = String(clamped);
  }
  updateOverlayScrubLabel();
  refreshEntropyFills();
  // The spans are about to be replaced, so an open popover would be
  // anchored to a detached element.
  hideAltsPopover();
  renderCurrentOverlay();
}

// Recolor the entropy bars for the frame the scrubber now sits on.
// Only the fills change, so the datasets are edited in place and the
// chart is updated with animation off: the slider fires continuously
// while dragged, and an animated color transition per tick would lag
// behind the pointer.
function refreshEntropyFills() {
  if (!chartEntropy) {
    return;
  }
  var sets = chartEntropy.data.datasets;
  for (var i = 0; i < sets.length; i++) {
    sets[i].backgroundColor = entropyFillColors(sets[i].data);
  }
  chartEntropy.update("none");
}

function updateOverlayScrubLabel() {
  if (!overlayScrubLabel) {
    return;
  }
  var maxIndex = overlayData && overlayData.frames
    && overlayData.frames.length > 0
    ? overlayData.frames.length - 1
    : 0;
  overlayScrubLabel.textContent =
    "Frame " + overlayFrameIndex + " / " + maxIndex;
}

function showOverlayUnavailable() {
  overlayViewer.hidden = false;
  overlaySelectGroup.hidden = true;
  overlayOutput.textContent = "";
  overlayOutput.classList.remove("token-layers");
  overlayOutput.hidden = true;
  overlayReadout.textContent = "";
  overlayReadout.hidden = true;
  overlayLegend.hidden = true;
  if (overlayDiffControls) {
    overlayDiffControls.hidden = true;
  }
  if (overlayScrubber) {
    overlayScrubber.hidden = true;
  }
  resetRunBlend(false);
  clearTokenMetrics();
  overlayEmpty.hidden = false;
}

function clearOverlay() {
  overlayData = null;
  overlayCommitSteps = null;
  overlayOriginalCommitSteps = null;
  overlayDiffData = null;
  overlayViewer.hidden = true;
  overlaySelectGroup.hidden = true;
  overlayOutput.textContent = "";
  overlayOutput.classList.remove("token-layers");
  overlayOutput.hidden = false;
  overlayReadout.textContent = "";
  overlayReadout.hidden = true;
  overlayLegend.hidden = true;
  if (overlayDiffControls) {
    overlayDiffControls.hidden = true;
  }
  if (overlayScrubber) {
    overlayScrubber.hidden = true;
  }
  resetRunBlend(false);
  clearTokenMetrics();
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
// Heatmap for every record run, Entropy for runs that saved it, plus
// Commit Order (diffusion only) and Diff vs Original (any model with
// an edited run and its original snapshot), each gated on data
// availability.
function buildOverlaySelect(data) {
  var canDiff = overlayDiffAvailable(data);
  var options = [
    { value: "none", label: "None" },
    {
      value: "heatmap",
      label: "Heatmap",
      disabled: !data.records_available,
    },
  ];
  // Entropy is gated on the saved data, not the model type: it shows
  // how undecided the model was over the whole vocabulary, which is a
  // different question than the confidence Heatmap answers.
  if (overlayEntropyAvailable(data)) {
    options.push({ value: "entropy", label: "Entropy" });
  }
  // Commit Order tints by resolution step, which a left-to-right run
  // does not have (its commit order is just position order).
  if (!overlayIsAutoregressive) {
    options.push({
      value: "commit",
      label: "Commit Order",
      disabled: !data.records_available,
    });
  }
  // A What If substitution gives autoregressive runs a real branch to
  // diff, so list it for them too once the data is there.
  if (!overlayIsAutoregressive || canDiff) {
    options.push({
      value: "diff",
      label: "Diff vs Original",
      disabled: !canDiff,
      title: canDiff
        ? undefined
        : "Only available for an edited run saved with"
          + " its original snapshot.",
    });
  }
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
  hideAltsPopover();
  renderCurrentOverlay();
}

// Re-render the active overlay mode at the current scrubber frame.
// Called on a mode change and on every scrubber move. Both are
// reasons a stationary pointer now reads something else, so the
// metrics strip is refreshed here rather than at each call site.
function renderCurrentOverlay() {
  if (overlayMode === "diff") {
    renderDiffOverlay();
  } else if (overlayMode === "commit") {
    renderCommitOverlay();
  } else if (overlayMode === "heatmap") {
    renderHeatmapOverlay();
  } else if (overlayMode === "entropy") {
    renderEntropyOverlay();
  } else {
    renderNoneOverlay();
  }
  refreshTokenMetrics();
}

// Plain tokens at the current scrubber frame, no coloring (None).
function renderNoneOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  renderOverlayTokens({
    frame: overlayFrameAt(overlayFrameIndex),
    colorFor: function () { return null; },
  });
}

// Confidence heatmap: recolor resolved tokens at the current frame by
// their persisted per-token confidence, using the shared heatColor
// scale. Kept for autoregressive runs too (the natural per-token
// confidence view). Masked positions render as the mask glyph.
function renderHeatmapOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  renderOverlayTokens({
    frame: overlayFrameAt(overlayFrameIndex),
    colorFor: function (index, tok) {
      if (typeof tok.c === "number") {
        return heatColor(tok.c);
      }
      return null;
    },
  });
}

// Entropy profile: recolor resolved tokens at the current frame by
// the sampling-time entropy persisted with each token, on a
// decisive (cool) to torn (hot) ramp. Autoregressive runs sample each
// position once, so a position's entropy never changes across frames.
function renderEntropyOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  renderOverlayTokens({
    frame: overlayFrameAt(overlayFrameIndex),
    colorFor: function (index, tok) {
      if (typeof tok.e === "number") {
        return entropyColor(tok.e);
      }
      return null;
    },
  });
}

// Commit Order. Unlike the per-token modes above, its colors come
// from the frame stream rather than from fields on the token, so the
// pre-edit layer cannot be described by the same callback: it needs
// its own steps, computed from the snapshot's frames.
function renderCommitOverlay() {
  overlayReadout.hidden = true;
  overlayReadout.textContent = "";
  var frames = overlayData.frames;
  // Commit steps come from the full frame stream (final frame is
  // ground truth), so they are memoized per run and applied to
  // whichever frame the scrubber shows (mirrors the generator).
  if (overlayCommitSteps === null) {
    overlayCommitSteps = overlaysComputeCommitSteps(frames);
  }
  var original = overlayData.original_frames || [];
  if (overlayOriginalCommitSteps === null
    && original.length > 0) {
    overlayOriginalCommitSteps =
      overlaysComputeCommitSteps(original);
  }
  renderOverlayTokens({
    frame: overlayFrameAt(overlayFrameIndex),
    colorFor: commitColorFor(
      overlayCommitSteps, frames.length - 1
    ),
    originalColorFor: commitColorFor(
      overlayOriginalCommitSteps || [], original.length - 1
    ),
  });
}

function commitColorFor(steps, maxStep) {
  return function (index) {
    var step = steps[index];
    if (typeof step === "number" && step >= 0) {
      return commitColor(step, maxStep);
    }
    return null;
  };
}

// Render the active mode's tokens: one layer normally, two stacked
// and crossfaded when the run carries a pre-edit snapshot. ``opts``
// carries the edited ``frame``, a colorFor(index, token) that never
// sees a masked position, and an optional originalColorFor for modes
// whose colors are not a function of the token alone.
//
// The default is to reuse the same callback for both layers, which
// is what makes the comparison mean anything: the pre-edit layer is
// colored by its own confidence or entropy, not the branch's.
function renderOverlayTokens(opts) {
  overlayOutput.textContent = "";
  tokenHighlightPos = null;
  var edited = {
    colorFor: overlayColorFn(opts.colorFor),
    classFor: editedClassFn,
  };
  var original = overlayComparisonFrame();
  if (original !== null) {
    renderOverlayLayers(original, opts.frame || [], edited, {
      colorFor: overlayColorFn(
        opts.originalColorFor || opts.colorFor
      ),
    });
    return;
  }
  overlayOutput.classList.remove("token-layers");
  if (!opts.frame) {
    return;
  }
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < opts.frame.length; i++) {
    fragment.appendChild(
      overlaysBuildTokenSpan(
        i, opts.frame[i], OVERLAYS_MASK_CHAR, edited
      )
    );
  }
  overlayOutput.appendChild(fragment);
}

// Mark the positions a saved edit touched, so a run's interventions
// can be found by looking rather than by remembering. Only ever given
// to the branch layer: the pre-edit run below it is what the model
// did on its own, and marking it would claim otherwise.
function editedClassFn(index) {
  if (editedPositionMarks(overlayData)[index] === true) {
    return "token-edited";
  }
  return "";
}

// Stack the pre-edit run under the branch at the crossfade's mix.
function renderOverlayLayers(
  origTokens, editedTokens, edited, original
) {
  overlayOutput.classList.add("token-layers");
  var editedTakes = overlaysEditedOwnsPointer(
    1 - compareBlend, compareBlend
  );
  overlayOutput.appendChild(
    overlaysBuildTokenLayer(origTokens, {
      layerClass: "token-layer-original",
      opacity: 1 - compareBlend,
      interactive: !editedTakes,
      colorFor: original.colorFor,
    })
  );
  overlayOutput.appendChild(
    overlaysBuildTokenLayer(editedTokens, {
      layerClass: "token-layer-edited",
      opacity: compareBlend,
      interactive: editedTakes,
      colorFor: edited.colorFor,
      classFor: edited.classFor,
    })
  );
}

// ---- Token hover highlight ----

// The pointer-driven half of the highlight, which is pure CSS once
// the class is on the container (see .token-hover-highlight in
// style.css). Analytics never applied it at all before, which left
// the chart-to-token direction lighting tokens that a direct hover
// could not.
//
// The preference is shared with the generator through the settings
// blob, so the drawer checkbox here and the one there mean the same
// thing rather than each page keeping its own idea.
function updateOverlayHoverHighlight() {
  var on = overlaysReadHighlightTokens();
  if (overlayHighlightCheckbox) {
    overlayHighlightCheckbox.checked = on;
  }
  if (!overlayOutput) {
    return;
  }
  overlayOutput.classList.toggle("token-hover-highlight", on);
}

function onOverlayHighlightToggle() {
  overlaysWriteHighlightTokens(overlayHighlightCheckbox.checked);
  updateOverlayHoverHighlight();
}

// ---- Cross-highlighting: token overlay <-> entropy chart ----

// Position currently lit from the chart side, so a pointer sweeping
// the bars does not re-query the DOM on every mousemove. Reset
// whenever the overlay re-renders, since that drops the class.
var tokenHighlightPos = null;

// Light the token(s) at a position. There are two when the run is
// layered, and lighting both keeps the mark visible wherever the
// crossfade happens to sit.
function setTokenHighlight(pos) {
  if (tokenHighlightPos === pos) {
    return;
  }
  clearTokenHighlight();
  tokenHighlightPos = pos;
  if (pos === null || !overlayOutput) {
    return;
  }
  var spans = overlayOutput.querySelectorAll(
    "[data-pos=\"" + pos + "\"]"
  );
  for (var i = 0; i < spans.length; i++) {
    spans[i].classList.add("token-cross-highlight");
  }
}

function clearTokenHighlight() {
  tokenHighlightPos = null;
  if (!overlayOutput) {
    return;
  }
  var lit = overlayOutput.querySelectorAll(
    ".token-cross-highlight"
  );
  for (var i = 0; i < lit.length; i++) {
    lit[i].classList.remove("token-cross-highlight");
  }
}

// The reverse: light the entropy bar at a token position. Chart.js
// already owns this highlight (the hover plugin's column guide and
// the bars' own hoverBackgroundColor both key off active elements),
// so driving it from a token hover is a matter of setting those.
// A no-op for runs without the chart, which is every diffusion run.
function setEntropyBarHighlight(pos) {
  if (!chartEntropy) {
    return;
  }
  var elements = [];
  var datasets = chartEntropy.data.datasets;
  for (var i = 0; pos !== null && i < datasets.length; i++) {
    if (pos < datasets[i].data.length) {
      elements.push({ datasetIndex: i, index: pos });
    }
  }
  chartEntropy.setActiveElements(elements);
  chartEntropy.update("none");
}

// The pre-edit run's tokens at the scrubber's frame, or null when
// there is nothing to compare against. Clamped to the snapshot's
// final frame once it ends, since a branch can outlive or fall short
// of the run it forked from (mirrors renderDiffOverlay).
function overlayComparisonFrame() {
  if (!overlayData || !overlayDiffAvailable(overlayData)) {
    return null;
  }
  return overlayClampedFrame(overlayData.original_frames);
}

// Spare every mode from repeating the masked-position check: a mask
// takes its color from .token-mask, not from the overlay.
function overlayColorFn(colorFor) {
  return function (index, tok) {
    if (!tok || tok.m) {
      return null;
    }
    return colorFor(index, tok);
  };
}

// ---- Token metrics strip ----
//
// The readout above the canvas, rendered by the same shared function
// the generator uses so a token reads identically on both pages. Two
// sources feed it here: a direct token hover, and the entropy chart
// (through tokenLinkPlugin), which is what makes a tall bar readable
// as a word without moving the pointer to the text.
var metricsHoverPos = null;

// Which stacked run the reading came from, taken from the hovered
// span's own layer so the strip reports what is on screen.
var metricsHoverOriginal = false;

// The candidate under the pointer in the popover, or null. A second,
// independent hover source: the left group answers "what is at this
// position" and this answers "what about the one I am reading".
var metricsCandidate = null;

function setTokenMetricsHover(pos, target) {
  metricsHoverPos = pos;
  metricsHoverOriginal =
    pos === null ? false : metricsLayerIsOriginal(target);
  refreshTokenMetrics();
}

// Fed by every candidate row. The reading carries the rank; the width
// it is measured against comes from the page, since a row cannot know
// it.
function setCandidateMetricsHover(reading) {
  metricsCandidate = reading === null ? null : {
    text: reading.t,
    probability: reading.p,
    rank: reading.rank || null,
    vocabSize: metricsVocabSize(),
  };
  refreshTokenMetrics();
}

// The output width of the model that produced the run on screen,
// read from the run itself and never from a resident worker: this
// page is routinely looking at a run whose checkpoint is not loaded.
// Runs saved before this was recorded report no width, and the rank
// then shows without a denominator rather than with a wrong one.
function metricsVocabSize() {
  return activeRunTokenizer().model_vocab_size || null;
}

// Re-read the held position, for anything that changes what a
// stationary pointer is pointing at: a new frame, a new overlay, a
// different run.
function refreshTokenMetrics() {
  overlaysRenderTokenMetrics(
    tokenMetricsStrip, buildTokenMetricsReading()
  );
}

function clearTokenMetrics() {
  metricsHoverPos = null;
  metricsHoverOriginal = false;
  metricsCandidate = null;
  overlaysRenderTokenMetrics(tokenMetricsStrip, null);
}

// A crossfade hands the pointer to the other layer at the midpoint.
// A stationary reading has no new span to ask, so it re-derives from
// ownership, which is what the next hover would report anyway.
function refreshTokenMetricsLayer() {
  if (metricsHoverPos !== null) {
    metricsHoverOriginal = metricsLayerIsOriginal(null);
  }
  refreshTokenMetrics();
}

// Chart hover has no span, so it falls back to whichever layer takes
// the pointer, which is the one the user could have hovered instead.
function metricsLayerIsOriginal(target) {
  if (target && target.closest) {
    var layer = target.closest(".token-layer");
    if (layer) {
      return layer.classList.contains("token-layer-original");
    }
  }
  if (!metricsLayered()) {
    return false;
  }
  if (overlayMode === "diff") {
    return !overlaysEditedOwnsPointer(
      overlayDiffOrigOpacity, overlayDiffEditOpacity
    );
  }
  return !overlaysEditedOwnsPointer(
    1 - compareBlend, compareBlend
  );
}

// Whether both runs are on the canvas together. Every layered mode
// gates on the same thing the crossfade does.
function metricsLayered() {
  return !!(overlayData && overlayDiffAvailable(overlayData));
}

// The tokens the canvas is drawing for the hovered layer, both
// clamped to their own final frame the way the render paths clamp
// them.
function metricsFrameTokens() {
  if (metricsHoverOriginal) {
    return overlayClampedFrame(
      overlayData ? overlayData.original_frames : null
    );
  }
  return overlayFrameAt(overlayFrameIndex);
}

// Assemble one reading, or null when the held position no longer
// names a token in the frame now on screen.
function buildTokenMetricsReading() {
  if (metricsHoverPos === null || !overlayData) {
    return null;
  }
  var tokens = metricsFrameTokens();
  if (!tokens || metricsHoverPos >= tokens.length) {
    return null;
  }
  var index = metricsHoverPos;
  var tok = tokens[index];
  var masked = !tok || !!tok.m;
  return {
    position: index,
    total: tokens.length,
    tokenText: tok ? tok.t : "",
    masked: masked,
    maskChar: OVERLAYS_MASK_CHAR,
    confidence: metricsConfidence(tok, masked),
    entropy:
      tok && typeof tok.e === "number" ? tok.e : null,
    extra: metricsExtra(index),
    candidate: metricsCandidate,
    runLabel: metricsRunLabel(),
  };
}

// A resolved token from a run saved before confidence was recorded
// reads as a dash rather than as zero, which would have claimed the
// model was certain of nothing. A mask keeps the zero it reported.
function metricsConfidence(tok, masked) {
  if (!tok) {
    return 0;
  }
  if (typeof tok.c === "number") {
    return tok.c;
  }
  return masked ? 0 : null;
}

// The overlay-specific line. Computed at hover time from the same
// memoized state the coloring uses, so no per-token callback has to
// be threaded through the render paths to carry it.
function metricsExtra(index) {
  if (overlayMode === "commit") {
    var steps = metricsHoverOriginal
      ? overlayOriginalCommitSteps
      : overlayCommitSteps;
    var step = steps ? steps[index] : null;
    if (typeof step !== "number" || step < 0) {
      return "";
    }
    return "Resolved at step: " + step;
  }
  if (overlayMode === "diff" && overlayDiffData) {
    if (overlayDiffData.origins[index]) {
      return "(remasked here)";
    }
    if (overlayDiffData.changed[index]) {
      return "was: " + overlayDiffData.origText[index];
    }
  }
  return "";
}

// Named only while both runs are on the canvas together. With one run
// drawn there is nothing to disambiguate.
function metricsRunLabel() {
  if (!metricsLayered()) {
    return "";
  }
  return metricsHoverOriginal ? "Original" : "Edited";
}

// Layered diff (mirrors the generator): the original and edited final
// frames are stacked with independent opacity and an optional
// difference blend, driven by the control row. The shared builder in
// overlays.js owns the layer construction.
function renderDiffOverlay() {
  // The change set is computed from the two runs' final frames (so it
  // is stable across the scrub) and memoized; only the rendered layers
  // vary per frame.
  if (overlayDiffData === null) {
    var curFinal = overlayFinalFrame(overlayData.frames);
    var origFinal = overlayFinalFrame(
      overlayData.original_frames
    );
    overlayDiffData = overlaysComputeDiff(
      curFinal, origFinal, overlayData.remask_edits
    );
  }
  var diff = overlayDiffData;
  overlayReadout.hidden = false;
  overlayReadout.textContent =
    "Diverged " + diff.changedCount
    + "/" + diff.totalCount;

  // Edited layer at the current frame; original layer clamped to its
  // final frame once it ends (the runs can differ in length / resume
  // boundaries), matching the generator (app.js renderDiffOverlay).
  var editedTokens = overlayFrameAt(overlayFrameIndex) || [];
  var origFrames = overlayData.original_frames || [];
  var oIdx = Math.min(
    overlayFrameIndex, origFrames.length - 1
  );
  var origTokens = (oIdx >= 0 ? origFrames[oIdx] : null) || [];

  overlayOutput.textContent = "";
  tokenHighlightPos = null;
  overlayOutput.classList.add("token-layers");
  overlayOutput.appendChild(
    overlaysBuildDiffLayers(
      origTokens,
      editedTokens,
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
      rerenderDiffOverlay();
    });
  }
  if (overlayDiffEditInput) {
    overlayDiffEditInput.addEventListener("input", function () {
      overlayDiffEditOpacity = Number(overlayDiffEditInput.value);
      rerenderDiffOverlay();
    });
  }
  if (overlayDiffBlendInput) {
    overlayDiffBlendInput.addEventListener("change", function () {
      overlayDiffBlendOn = !!overlayDiffBlendInput.checked;
      rerenderDiffOverlay();
    });
  }
}

// The three controls above share one response: redraw the layers and
// re-read the strip, which the opacity sliders can flip between runs.
function rerenderDiffOverlay() {
  if (overlayMode !== "diff") {
    return;
  }
  renderDiffOverlay();
  refreshTokenMetricsLayer();
}

// Wire the per-frame scrubber once: the slider and the prev/next
// arrows all route through setOverlayFrame, which clamps, syncs the
// controls, and re-renders the active overlay at the chosen frame.
function wireOverlayScrubber() {
  if (overlayScrubSlider) {
    overlayScrubSlider.addEventListener("input", function () {
      setOverlayFrame(Number(overlayScrubSlider.value));
    });
  }
  if (overlayScrubPrev) {
    overlayScrubPrev.addEventListener("click", function () {
      setOverlayFrame(overlayFrameIndex - 1);
    });
  }
  if (overlayScrubNext) {
    overlayScrubNext.addEventListener("click", function () {
      setOverlayFrame(overlayFrameIndex + 1);
    });
  }

  // Candidate popover on token hover, for runs saved with the
  // Alternatives capture. Read-only here (substitution lives on the
  // generator, which still holds the worker's run state). The same
  // hover lights the matching entropy bar.
  if (overlayOutput) {
    overlayOutput.addEventListener(
      "mouseover",
      function (e) {
        var target = e.target;
        if (!target.classList.contains("token-span")) {
          return;
        }
        var raw = target.getAttribute("data-pos");
        if (raw === null) {
          return;
        }
        var pos = parseInt(raw, 10);
        setEntropyBarHighlight(pos);
        setTokenMetricsHover(pos, target);
        if (pos === altsPopoverPos) {
          return;
        }
        showAltsPopover(pos, target);
      }
    );
    overlayOutput.addEventListener(
      "mouseleave",
      function () {
        setEntropyBarHighlight(null);
        // Reaching into the popover keeps it open, so its pagination
        // arrows are clickable (mirrors the generator). The strip
        // holds its reading for the same reason: it describes the
        // position whose candidates are being read.
        if (altsPopover && altsPopover.matches(":hover")) {
          return;
        }
        clearTokenMetrics();
        hideAltsPopover();
      }
    );
  }
  if (altsPopover) {
    altsPopover.addEventListener("mouseleave", function () {
      clearTokenMetrics();
      hideAltsPopover();
    });
  }
  window.addEventListener(
    "scroll",
    function () {
      if (altsPopoverPos !== null) {
        hideAltsPopover();
      }
    },
    true
  );
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
    layout: chartGutterLayout(),
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
          labelColor: lineLabelColor,
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
  // Shown before the chart is constructed, and possibly hidden again
  // by applyTimingPage once its sibling has been built too: Chart.js
  // sizes itself off the canvas it is handed, and a canvas in a
  // hidden section measures zero.
  timingSection.hidden = false;
  timingPageReady.elapsed = true;

  var canvas = document.getElementById(
    "chart-timing"
  );

  var cumResult = buildCumulativeTiming(
    data.per_frame_elapsed, remaskSet
  );
  var values = cumResult.values;
  var resumeSet = resumeBoundarySet(
    cumResult.resumeStartSet, remaskSet
  );

  var original = timingOriginalValues(data);
  var labels = compareFrameLabels(values, original);

  // Original first, so it draws beneath the branch it produced and
  // so dataset index 0 is the one the pins and crossfade fade out.
  var datasets = [];
  if (original) {
    datasets.push(compareOriginalDataset(original));
  }
  datasets.push(timingEditedDataset(
    values, remaskSet, resumeSet, !!original
  ));

  chartTiming = new Chart(
    canvas.getContext("2d"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: timingOptions(remaskSet),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
        seriesBlendPlugin("timing"),
      ],
    }
  );
  chartInstances.timing = chartTiming;
  updateComparePins("timing", !!original);
}

// ---- Tokens per second ----
//
// The running average, not the instantaneous rate: tokens produced so
// far over seconds spent so far. It shares the Timing slot because it
// is the same two numbers read as a ratio, and reading it as a
// running total keeps it level with the elapsed line beside it. A
// per-step rate on a diffusion run is mostly the sampler's reveal
// schedule sawtoothing, which says more about the schedule than about
// throughput.
//
// Nothing new is stored for this. Every run already has its frame
// timings, and mask counts fall out of the convergence series the
// endpoint computes from history.txt, so it works on runs saved long
// before the metric existed.
function renderTpsChart(data, remaskSet, isAutoregressive) {
  var section = document.getElementById("tps-section");
  var elapsed = tpsElapsedValues(data, remaskSet);
  var produced = tokensProducedSeries(
    data, elapsed.length, isAutoregressive
  );
  if (!elapsed.length || !produced) {
    if (section) { section.hidden = true; }
    return;
  }
  // See renderTimingChart on why this is shown before building.
  if (section) { section.hidden = false; }
  timingPageReady.tps = true;

  var values = tokenRateSeries(produced, elapsed);
  var original = tpsOriginalValues(data, isAutoregressive);
  var labels = compareFrameLabels(values, original);

  // Original first, so it draws beneath the branch it produced and
  // so dataset index 0 is the one the pins and crossfade fade out.
  var datasets = [];
  if (original) {
    datasets.push(compareOriginalDataset(original));
  }
  datasets.push(tpsEditedDataset(values, !!original));

  chartTps = new Chart(
    document.getElementById("chart-tps").getContext("2d"),
    {
      type: "line",
      data: { labels: labels, datasets: datasets },
      options: tpsOptions(remaskSet),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
        seriesBlendPlugin("tps"),
      ],
    }
  );
  chartInstances.tps = chartTps;
  updateComparePins("tps", !!original);
}

// The same stitched cumulative series the elapsed chart draws, so the
// two charts in this slot cannot disagree about when a frame landed.
function tpsElapsedValues(data, remaskSet) {
  var raw = data.per_frame_elapsed;
  if (!raw || raw.length === 0) {
    return [];
  }
  return buildCumulativeTiming(raw, remaskSet).values;
}

// Tokens resolved by frame i, counted from the start of the run.
//
// Autoregressive runs emit exactly one token per frame, so the frame
// index is the count. Diffusion runs get it from the convergence
// series: a masked token renders as exactly one mask glyph, so
// mask_count is a token count, and frame 0 (all masked) gives the
// canvas length to subtract from.
//
// Returns null when the run carries nothing to count from, which
// hides the chart rather than drawing a flat zero.
function tokensProducedSeries(data, frames, isAutoregressive) {
  if (frames === 0) {
    return null;
  }
  var produced = [];
  var i;
  if (isAutoregressive) {
    for (i = 0; i < frames; i++) {
      produced.push(i + 1);
    }
    return produced;
  }
  var convergence = data.convergence;
  if (!convergence || convergence.length === 0) {
    return null;
  }
  var start = convergence[0].mask_count;
  for (i = 0; i < frames; i++) {
    var point = convergence[i];
    var masked = point ? point.mask_count : 0;
    // Clamped because DiffusionGemma's mask count can rise between
    // drafts, which would otherwise read as negative production.
    produced.push(Math.max(0, start - masked));
  }
  return produced;
}

function tokenRateSeries(produced, elapsed) {
  var values = [];
  for (var i = 0; i < produced.length; i++) {
    var seconds = elapsed[i];
    // A frame that shares a timestamp with the run's start has no
    // window to average over. null rather than zero, so the line
    // skips the point instead of diving to the axis.
    if (!(seconds > 0)) {
      values.push(null);
    } else {
      values.push(+(produced[i] / seconds).toFixed(2));
    }
  }
  return values;
}

// Only autoregressive runs can show the pre-edit run here. A saved
// run keeps the original's frame timings but not its canvas history,
// and a rate needs both; an autoregressive run needs no history,
// because one token per frame is structural. So a diffusion
// comparison would have to invent the numerator.
function tpsOriginalValues(data, isAutoregressive) {
  if (!isAutoregressive) {
    return null;
  }
  var raw = data.original_per_frame_elapsed;
  if (!raw || raw.length === 0) {
    return null;
  }
  var elapsed = buildCumulativeTiming(raw, {}).values;
  var produced = [];
  for (var i = 0; i < elapsed.length; i++) {
    produced.push(i + 1);
  }
  return tokenRateSeries(produced, elapsed);
}

// See timingEditedDataset for what ``paired`` switches and why.
function tpsEditedDataset(values, paired) {
  return {
    label: paired ? "Edited" : "Tokens/s",
    data: values,
    borderColor: TPS_COLOR,
    backgroundColor: withAlpha(TPS_COLOR, 0.08),
    fill: paired
      ? compareBandFill("tps", TPS_COLOR)
      : true,
    borderDash: paired ? COMPARE_EDITED_DASH : [],
    tension: 0.2,
    pointRadius: 0,
    borderWidth: 1.5,
    spanGaps: true,
  };
}

function tpsOptions(remaskSet) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    layout: chartGutterLayout(),
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
        filter: function (item) {
          return seriesRowVisible("tps", item);
        },
        callbacks: {
          title: tooltipTitle,
          labelColor: lineLabelColor,
          label: function (ctx) {
            return ctx.dataset.label + ": "
              + ctx.formattedValue + " T/s";
          },
          afterLabel: function (ctx) {
            if (!isEditedDataset(ctx)) { return ""; }
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
        title: { display: true, text: "Frame" },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        title: { display: true, text: "Tokens/second" },
        beginAtZero: true,
      },
    },
  };
}

// Cumulative elapsed for the pre-edit run, or null when this run
// carries none: an unedited run, or one saved before the signal
// existed. That array is a single unbranched segment and so needs no
// stitching; it goes through buildCumulativeTiming only so both
// series are produced the same way.
function timingOriginalValues(data) {
  var raw = data.original_per_frame_elapsed;
  if (!raw || raw.length === 0) {
    return null;
  }
  return buildCumulativeTiming(raw, {}).values;
}

// Frame labels spanning the longer run: a branch can outlive or fall
// short of the run it forked from.
function compareFrameLabels(values, original) {
  var count = values.length;
  if (original && original.length > count) {
    count = original.length;
  }
  var labels = [];
  for (var i = 0; i < count; i++) {
    labels.push(i);
  }
  return labels;
}

// The pre-edit run's line, shared by the timing and confidence
// charts. Solid, neutral, and deliberately without the branch's
// segment coloring: no remask or resume happened in this run.
function compareOriginalDataset(values) {
  return {
    label: "Original",
    data: values,
    borderColor: COMPARE_ORIGINAL_COLOR,
    fill: false,
    tension: 0.2,
    pointRadius: 0,
    borderWidth: 1.5,
    spanGaps: true,
  };
}

// ``paired`` is true once there is an original series to compare
// against, which switches the label to name its run and turns the
// area fill into a band between the two runs. Filling both to the
// axis instead would stack two translucent washes over the prefix
// the runs share and read as a third color rather than as two runs.
function timingEditedDataset(
  values, remaskSet, resumeSet, paired
) {
  return {
    label: paired ? "Edited" : "Elapsed",
    data: values,
    borderColor: TIMING_COLOR,
    backgroundColor: withAlpha(TIMING_COLOR, 0.08),
    fill: paired
      ? compareBandFill("timing", TIMING_COLOR)
      : true,
    borderDash: paired ? COMPARE_EDITED_DASH : [],
    tension: 0.2,
    pointRadius: 0,
    borderWidth: 1.5,
    spanGaps: true,
    segment: {
      borderColor: function (ctx) {
        var fi = ctx.p1DataIndex;
        if (remaskSet[fi]) {
          return "#00ff41";
        }
        if (isInResumedRange(fi, resumeSet)) {
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
  };
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
    layout: chartGutterLayout(),
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
        filter: function (item) {
          return seriesRowVisible("timing", item);
        },
        callbacks: {
          title: tooltipTitle,
          labelColor: lineLabelColor,
          label: function (ctx) {
            return ctx.dataset.label + ": "
              + ctx.formattedValue + "s";
          },
          afterLabel: function (ctx) {
            // The branch is the last dataset, and the resume is an
            // event in it alone, so the note is not repeated under
            // the original run's row.
            if (!isEditedDataset(ctx)) { return ""; }
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

  var values = confidencePercentValues(meanConf);
  var original = confidenceOriginalValues(data);
  var labels = compareFrameLabels(values, original);

  var datasets = [];
  if (original) {
    datasets.push(compareOriginalDataset(original));
  }
  datasets.push(
    confidenceEditedDataset(values, !!original)
  );

  chartConfidence = new Chart(
    canvas.getContext("2d"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: confidenceOptions(),
      plugins: [
        canvasBoundaryPlugin(data.canvas_boundaries || []),
        burnThroughPlugin,
        seriesBlendPlugin("confidence"),
      ],
    }
  );
  chartInstances.confidence = chartConfidence;
  updateComparePins("confidence", !!original);
}

// Fractions to whole percents, preserving nulls so a frame that
// recorded no confidence stays a gap rather than reading as zero.
function confidencePercentValues(raw) {
  var out = [];
  for (var i = 0; i < raw.length; i++) {
    var v = raw[i];
    out.push(
      v === null || v === undefined
        ? null
        : +(v * 100).toFixed(2)
    );
  }
  return out;
}

// The pre-edit run's confidence, or null when this run carries none.
function confidenceOriginalValues(data) {
  var raw = data.original_mean_conf;
  if (!raw || raw.length === 0) {
    return null;
  }
  return confidencePercentValues(raw);
}

// See timingEditedDataset for what ``paired`` switches and why.
function confidenceEditedDataset(values, paired) {
  return {
    label: paired ? "Edited" : "Mean confidence",
    data: values,
    borderColor: CONFIDENCE_COLOR,
    backgroundColor: withAlpha(CONFIDENCE_COLOR, 0.08),
    fill: paired
      ? compareBandFill("confidence", CONFIDENCE_COLOR)
      : true,
    borderDash: paired ? COMPARE_EDITED_DASH : [],
    tension: 0.2,
    pointRadius: 0,
    borderWidth: 1.5,
    spanGaps: true,
  };
}

function confidenceOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    layout: chartGutterLayout(),
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
        filter: function (item) {
          return seriesRowVisible("confidence", item);
        },
        callbacks: {
          title: tooltipTitle,
          labelColor: lineLabelColor,
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

// ---- Entropy by position ----

// Tear the chart down and hide its section. Called before a new run's
// frames are fetched, and on the paths where a run turns out to carry
// no usable records at all.
function clearEntropyChart() {
  chartEntropy = destroyChart(chartEntropy);
  chartInstances.entropy = null;
  // The chart owns one direction of the cross-highlight, so tearing
  // it down while a bar is hovered would otherwise strand the class
  // on whichever token was last lit.
  clearTokenHighlight();
  var section = document.getElementById("entropy-section");
  if (section) {
    section.hidden = true;
  }
}

// Per-position entropy for one frame series, read off its final
// frame: every position is sampled once in an autoregressive run, so
// its entropy never changes after the frame that introduced it.
// Mirrors the generator's entropyProfileValues. Runs over both the
// open run and its pre-edit snapshot, which is why it takes frames
// rather than the payload.
function entropySeriesFrom(frames) {
  var list = frames || [];
  var final = list[overlayFinalFrameIndex(list)] || [];
  var values = [];
  var texts = [];
  for (var i = 0; i < final.length; i++) {
    var tok = final[i] || {};
    values.push(
      typeof tok.e === "number" ? +tok.e.toFixed(3) : null
    );
    texts.push(
      typeof tok.t === "string" ? overlaysAltDisplay(tok.t) : ""
    );
  }
  return { values: values, texts: texts };
}

// Every position touched by a saved edit. For an autoregressive What
// If branch that is the single substituted position; for a diffusion
// run it is the remasked set, so the marker generalizes.
function editedPositions(data) {
  var marks = editedPositionMarks(data);
  var positions = [];
  for (var key in marks) {
    if (marks[key] === true) {
      positions.push(Number(key));
    }
  }
  return positions;
}

// The same set as a lookup, for the token layer, which asks about
// every position it draws. Keyed on the run payload itself, which is
// replaced wholesale when a run is selected and never mutated in
// place, so switching runs rebuilds this and staying on one does not.
var editedMarksCache = { data: null, marks: {} };

function editedPositionMarks(data) {
  if (editedMarksCache.data === data) {
    return editedMarksCache.marks;
  }
  var edits = (data && data.remask_edits) || [];
  var marks = {};
  for (var i = 0; i < edits.length; i++) {
    var group = edits[i].token_positions || [];
    for (var j = 0; j < group.length; j++) {
      marks[group[j]] = true;
    }
  }
  editedMarksCache = { data: data, marks: marks };
  return marks;
}

// The position where the two runs part ways, or null when the run
// was never edited. A What If branch copies the original trace's
// prefix verbatim, so everything left of this index is identical in
// both series and only the right side is worth comparing.
//
// This single-boundary reading is autoregressive-shaped. Diffusion
// remasks are scattered rather than a prefix cut, so once those runs
// carry entropy the comparison will want per-position divergence
// instead of one index.
function divergencePosition(data) {
  var positions = editedPositions(data);
  if (positions.length === 0) {
    return null;
  }
  var earliest = positions[0];
  for (var i = 1; i < positions.length; i++) {
    if (positions[i] < earliest) {
      earliest = positions[i];
    }
  }
  return earliest;
}

// One entropy layer. grouped:false is load-bearing: left grouped,
// Chart.js sits the two runs side by side and halves every bar,
// where the whole point is to superimpose them and crossfade.
function entropyDataset(label, series) {
  var glowColors = [];
  for (var i = 0; i < series.values.length; i++) {
    glowColors.push(entropyGlowColor(series.values[i]));
  }
  return {
    label: label,
    data: series.values,
    backgroundColor: entropyFillColors(series.values),
    hoverBackgroundColor: glowColors,
    borderWidth: 0,
    barPercentage: 1,
    categoryPercentage: 1,
    grouped: false,
  };
}

// Per-bar fills, faded past the position the scrubbed frame reached,
// so the chart and the canvas above it agree about which tokens exist
// at this frame. Position and frame index are the same number here:
// this chart is autoregressive-only, and frame k is the frame that
// introduced position k.
//
// Baked into the color because Chart.js has no per-bar opacity. It
// multiplies with the whole-dataset globalAlpha the crossfade sets in
// compareBlendPlugin, which is the wanted composition: a dim bar in
// the receding run is dimmer still.
function entropyFillColors(values) {
  var colors = [];
  for (var i = 0; i < values.length; i++) {
    colors.push(
      i <= overlayFrameIndex
        ? entropyColor(values[i])
        : entropyDimColor(values[i])
    );
  }
  return colors;
}

// The pre-edit run's entropy, or null when there is nothing to
// compare against. Needs three things at once: a divergence point, a
// saved snapshot, and entropy inside it. The snapshot exists for any
// edited run but predates the entropy signal on older ones, so an
// older branch falls back to the single layer.
function entropyOriginalSeries(data, divergence) {
  if (divergence === null) {
    return null;
  }
  if (!data.original_frames) {
    return null;
  }
  if (!framesHaveEntropy(data.original_frames)) {
    return null;
  }
  return entropySeriesFrom(data.original_frames);
}

// One bar per generated position, tall and hot where the model was
// torn. Unlike the three charts above it is indexed by position
// rather than frame, which is also why it is drawn as bars: an
// autoregressive position is an independent decision, not a point in
// a time series. Hidden for runs saved without the entropy signal.
function renderEntropyChart(data) {
  var section = document.getElementById("entropy-section");
  if (!overlayEntropyAvailable(data)) {
    clearEntropyChart();
    return;
  }
  if (section) {
    section.hidden = false;
  }

  var divergence = divergencePosition(data);
  var edited = entropySeriesFrom(data.frames);
  var original = entropyOriginalSeries(data, divergence);

  // Labels span the longer run: a branch can outlive or fall short
  // of the run it forked from.
  var count = edited.values.length;
  if (original && original.values.length > count) {
    count = original.values.length;
  }
  var labels = [];
  for (var i = 0; i < count; i++) {
    labels.push(i);
  }

  // Original first, so it draws beneath the branch it produced and
  // so dataset index 0 is the one the crossfade fades out.
  var datasets = [];
  var texts = [];
  if (original) {
    datasets.push(entropyDataset("Original", original));
    texts.push(original.texts);
  }
  datasets.push(entropyDataset("Edited", edited));
  texts.push(edited.texts);

  var canvas = document.getElementById("chart-entropy");
  chartEntropy = new Chart(
    canvas.getContext("2d"),
    {
      type: "bar",
      data: {
        labels: labels,
        datasets: datasets,
      },
      options: entropyChartOptions(
        texts, original ? divergence : null
      ),
      // Deliberately without burnThroughPlugin: it redraws a
      // dataset's *line* through the tooltip box, which a bar chart
      // has none of, and would stroke a stray polyline across the bar
      // tops instead. The eye toggle covers hiding the box.
      //
      // Marker before hover so the pointer's white guide lays over
      // the edit tint rather than under it.
      plugins: [
        substitutionMarkerPlugin(editedPositions(data)),
        entropyHoverPlugin,
        compareBlendPlugin,
        tokenLinkPlugin,
      ],
    }
  );
  chartInstances.entropy = chartEntropy;
}

// Back to the edited run at full opacity, so each run opens on the
// branch it is a record of rather than on the previous run's mix.
// The control shows for any edited run saved with its snapshot, which
// is a wider gate than the entropy chart's: the token layers only
// need the snapshot, while a second bar series also needs it to carry
// per-token entropy.
function resetRunBlend(visible) {
  compareBlend = 1;
  // A run can be opened while a previous drag is still easing out.
  cancelScrubEase();
  scrubWeight = 0;
  scrubArmed = false;
  scrubEngaged = false;
  setPinsPreviewing(false);
  if (runBlendInput) {
    runBlendInput.value = "100";
  }
  if (runBlendRow) {
    runBlendRow.hidden = !visible;
  }
}

// Only layer alpha changes, so nothing is reparsed; "none" skips the
// animation that would otherwise lag the drag. The line charts are
// only touched once a press has become a drag, and are left entirely
// alone for a keyboard adjustment.
function onRunBlendInput() {
  compareBlend = Number(runBlendInput.value) / 100;
  if (chartEntropy) {
    chartEntropy.update("none");
  }
  applyTokenLayerBlend();
  refreshTokenMetricsLayer();
  engageBlendScrub();
  if (scrubEngaged) {
    updateLineCharts();
  }
}

// Restyle the stacked token layers in place. Rebuilding them would
// mean several hundred spans per slider step, and would also drop the
// popover mid-drag. Diff mode is left alone: its two sliders own the
// layers there.
function applyTokenLayerBlend() {
  if (overlayMode === "diff") {
    return;
  }
  var original =
    overlayOutput.querySelector(".token-layer-original");
  var edited =
    overlayOutput.querySelector(".token-layer-edited");
  if (!original || !edited) {
    return;
  }
  original.style.opacity = String(1 - compareBlend);
  edited.style.opacity = String(compareBlend);
  overlaysApplyLayerPointers(
    overlayOutput, 1 - compareBlend, compareBlend
  );
}

// ``texts`` holds one token-text array per dataset, so the tooltip
// can name the token each layer chose. ``divergence`` is null on a
// run with nothing to compare against, which collapses the tooltip
// back to the single unlabeled row.
function entropyChartOptions(texts, divergence) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    layout: chartGutterLayout(),
    interaction: {
      mode: "index",
      intersect: false,
    },
    // The bar-to-token half of the cross-highlight lives in
    // tokenLinkPlugin rather than onHover; see its comment.
    plugins: {
      legend: { display: false },
      tooltip: {
        position: "smart",
        caretSize: 0,
        xAlign: "left",
        yAlign: "top",
        filter: function (item) {
          return entropyTooltipFilter(item, divergence);
        },
        callbacks: {
          title: positionTooltipTitle,
          label: function (ctx) {
            return entropyTooltipLabel(ctx, texts, divergence);
          },
        },
      },
      zoom: zoomPluginOptions(),
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Position",
        },
        ticks: { maxTicksLimit: 12 },
        grid: { display: false },
      },
      y: {
        title: {
          display: true,
          text: "Entropy (nats)",
        },
        beginAtZero: true,
        // Suggested, not fixed: keeps the scale comparable across
        // runs at the overlay's reference maximum while still letting
        // an unusually torn position through instead of clipping it.
        suggestedMax: OVERLAYS_ENTROPY_REF_NATS,
      },
    },
  };
}

// The shared tooltipTitle prefixes "Frame", which would misread this
// chart's x axis.
function positionTooltipTitle(items) {
  if (items.length === 0) {
    return "";
  }
  return "Position " + items[0].label;
}

// Which rows the hovered position is worth showing. A null value is
// the tail of a run that stopped short of its counterpart. The
// original layer is dropped left of the divergence point because the
// branch copies its prefix verbatim there, so a second row would
// only ever restate the first.
function entropyTooltipFilter(item, divergence) {
  if (item.parsed.y === null) {
    return false;
  }
  if (divergence === null) {
    return true;
  }
  if (item.datasetIndex > 0) {
    return true;
  }
  return item.dataIndex >= divergence;
}

// Naming the token is the thing the generator's compact profile
// cannot do, so the tooltip carries it alongside the value.
//
// From the divergence point on, each row is named for its run. Note
// that at the marked position itself the two rows carry the same
// nats and different tokens, which is the intervention in one line:
// entropy describes the distribution the prefix produced, and
// forcing a token changes which one was drawn, not the distribution
// it was drawn from.
function entropyTooltipLabel(ctx, texts, divergence) {
  var value = ctx.formattedValue + " nats";
  var series = texts[ctx.datasetIndex] || [];
  var text = series[ctx.dataIndex];
  var row = text ? value + "  \u2022  " + text : value;
  if (divergence === null || ctx.dataIndex < divergence) {
    return row;
  }
  return ctx.dataset.label + ": " + row;
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
      // AR runs have no meaningful convergence curve; omit them.
      if (runIdIsAutoregressive(results[i].run_id)) {
        continue;
      }
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
      if (runIdIsAutoregressive(res.run_id)) { continue; }
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
          labelColor: lineLabelColor,
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

  // Both before the row handler below, so acting on a row's controls
  // does not also open the run's detail panel.
  var star = e.target.closest(".row-star-btn");
  if (star) {
    toggleFavorite(star.getAttribute("data-run-id"));
    return;
  }

  var caret = e.target.closest(".row-collect-caret");
  if (caret) {
    openCollectionChooser(caret.getAttribute("data-run-id"));
    return;
  }

  var cb = e.target.closest(
    'input[type="checkbox"]'
  );
  if (cb) {
    var rid = cb.getAttribute("data-run-id");
    checkedIds[rid] = cb.checked;
    // Shade the row immediately; renderTable applies row-checked on its
    // next pass, but ticking a box does not re-render on its own.
    var checkedRow = cb.closest("tr");
    if (checkedRow) {
      checkedRow.classList.toggle("row-checked", cb.checked);
    }
    updateCompareButton();
    updateBulkDeleteButton();
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
    // The rows on screen, not every run on disk. Under a collection
    // tab, selecting all and then bulk-deleting would otherwise
    // remove runs the user could not see.
    var shown = visibleRuns();
    for (var i = 0; i < shown.length; i++) {
      checkedIds[shown[i].run_id] = true;
    }
  }
  renderTable();
  updateCompareButton();
  updateBulkDeleteButton();
}

// Selecting a tab. Clears the selection: a checkbox ticked under one
// tab refers to a row that may not exist under the next, and carrying
// it across would put invisible runs in a bulk delete.
function selectCollection(id) {
  if (activeCollectionId === id) {
    return;
  }
  activeCollectionId = id;
  checkedIds = {};
  selectAllCb.checked = false;
  updateCompareButton();
  updateBulkDeleteButton();
  renderCollectionTabs();
  renderTable();
}

function onCollectionTabClick(e) {
  if (e.target.closest("#btn-collection-add")) {
    beginCollectionNameEdit(
      e.target.closest("#btn-collection-add"), null
    );
    return;
  }
  var tab = e.target.closest(".collection-tab");
  if (!tab) {
    return;
  }
  var id = tab.getAttribute("data-collection-id");
  var collection = id ? findCollection(id) : null;
  var icon = e.target.closest("[data-tab-action]");
  if (icon && collection) {
    onCollectionTabAction(
      icon.getAttribute("data-tab-action"), tab, collection
    );
    return;
  }
  selectCollection(id || null);
}

function onCollectionTabAction(action, tab, collection) {
  if (action === "rename") {
    beginCollectionNameEdit(tab, collection);
    return;
  }
  if (action === "delete") {
    openCollectionDeleteModal(collection);
    return;
  }
}

function onGroupChange() {
  renderTable();
}

function loadAndRender() {
  fetchRuns().then(function (runs) {
    allRuns = runs;
    checkedIds = {};
    selectAllCb.checked = false;
    // Re-read on every refresh, not just at boot: the server prunes
    // ids for deleted runs on hydrate, and another window may have
    // filed something since.
    loadCollections();
    updateCompareButton();
    updateBulkDeleteButton();
    renderCollectionTabs();
    renderTable();
  });
}

// ---- Delete a run ----

function runPath(runId) {
  return "results/" + runId;
}

// Update the confirmation modal copy for the staged deletion, then
// reveal it. Single deletes show the run's path; bulk deletes show the
// count. `pendingDeleteIds` must be set before calling.
function showDeleteModal() {
  var count = pendingDeleteIds.length;
  if (count === 1) {
    deleteModalTitle.textContent = "Delete this run?";
    deleteRunLabel.textContent = runPath(pendingDeleteIds[0]);
    deleteModalNote.innerHTML =
      "This permanently removes the saved run from "
      + "<code>results/</code>. This cannot be undone.";
  } else {
    deleteModalTitle.textContent =
      "Delete " + count + " runs?";
    deleteRunLabel.textContent =
      count + " selected runs will be removed.";
    deleteModalNote.innerHTML =
      "This permanently removes the saved runs from "
      + "<code>results/</code>. This cannot be undone.";
  }
  btnDeleteConfirm.disabled = false;
  modalDelete.classList.remove("hidden");
}

function openDeleteModal(runId) {
  pendingDeleteIds = [runId];
  showDeleteModal();
}

function openBulkDeleteModal() {
  var ids = checkedRunIds();
  if (ids.length < 1) { return; }
  pendingDeleteIds = ids;
  showDeleteModal();
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
  pendingDeleteIds = [];
  btnDeleteConfirm.disabled = false;
  modalDelete.classList.add("hidden");
}

// Delete a single run, resolving to a {runId, success} record so a
// batch can report partial failures without one rejection aborting the
// rest. Never rejects.
function deleteOneRun(runId) {
  return fetch(
    "/api/analytics/runs/" + encodeURIComponent(runId),
    { method: "DELETE" }
  )
    .then(function (r) { return r.json(); })
    .then(function (result) {
      return {
        runId: runId,
        success: !!(result && result.success),
      };
    })
    .catch(function () {
      return { runId: runId, success: false };
    });
}

// Drop the successfully deleted runs from local state and refresh the
// selection-dependent UI in one pass.
function applyDeletions(deletedIds) {
  if (deletedIds.length < 1) { return; }
  var removed = {};
  for (var i = 0; i < deletedIds.length; i++) {
    removed[deletedIds[i]] = true;
    delete checkedIds[deletedIds[i]];
    // Clear any "new run" cue for the deleted run so the generator's
    // and menu's counts decrement (write-through to the server).
    overlaysClearNewRun(deletedIds[i]);
    if (activeRunId === deletedIds[i]) {
      hideDetail();
    }
  }
  allRuns = allRuns.filter(function (run) {
    return !removed[run.run_id];
  });
  // A collection holding an id whose folder is gone would show a row
  // that cannot be opened. The server prunes on the next hydrate, but
  // that is a page load away, and this table is looking at it now.
  dropDeletedFromCollections(removed);
  selectAllCb.checked = false;
  updateCompareButton();
  updateBulkDeleteButton();
  renderCollectionTabs();
  renderTable();
}

function dropDeletedFromCollections(removed) {
  var changed = false;
  for (var i = 0; i < collections.length; i++) {
    var runs = collections[i].runs;
    var kept = runs.filter(function (runId) {
      return !removed[runId];
    });
    if (kept.length !== runs.length) {
      collections[i].runs = kept;
      changed = true;
    }
  }
  if (changed) {
    saveCollections();
  }
}

function reportDeletion(deleted, failed) {
  if (deleted.length === 1 && failed.length === 0) {
    showToast(
      "Successfully deleted run \u201c"
      + runPath(deleted[0]) + "\u201d"
    );
    return;
  }
  if (deleted.length > 0 && failed.length === 0) {
    showToast(
      "Successfully deleted " + deleted.length + " runs"
    );
    return;
  }
  if (deleted.length > 0 && failed.length > 0) {
    showToast(
      "Deleted " + deleted.length + " of "
      + (deleted.length + failed.length)
      + " runs; the rest failed"
    );
    return;
  }
  showToast("Failed to delete the selected runs");
}

function confirmDelete() {
  var ids = pendingDeleteIds.slice();
  if (ids.length < 1) { return; }
  btnDeleteConfirm.disabled = true;

  var requests = [];
  for (var i = 0; i < ids.length; i++) {
    requests.push(deleteOneRun(ids[i]));
  }
  Promise.all(requests).then(function (results) {
    var deleted = [];
    var failed = [];
    for (var j = 0; j < results.length; j++) {
      if (results[j].success) {
        deleted.push(results[j].runId);
      } else {
        failed.push(results[j].runId);
      }
    }
    applyDeletions(deleted);
    closeDeleteModal();
    reportDeletion(deleted, failed);
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

// ---- Per-chart run pins ----

// The pins own which runs a line chart draws at rest. They are
// deliberately independent of the run crossfade, which only borrows
// the charts for the duration of a drag (see engageBlendScrub).
function handleComparePinClick(e) {
  var btn = e.target.closest(".compare-pin-btn");
  if (!btn) { return; }
  var name = btn.getAttribute("data-chart");
  var state = linePinState[name];
  if (!state) { return; }
  var series = btn.getAttribute("data-series");
  // Turning off the only lit pin would blank the chart, which is the
  // one state with nothing in it to read.
  if (state[series] && !comparePinsBothOn(state)) {
    return;
  }
  state[series] = !state[series];
  refreshComparePins(name);
  var chart = chartInstances[name];
  if (chart) {
    chart.update("none");
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
      { value: "processor", label: "Processor" },
      { value: "prompt", label: "Prompt" },
      { value: "has_diff", label: "Edited" },
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

// The shared helper owns the handle click as well as the drag, so
// this binds none of its own (see overlaysMakeDrawerDraggable).
overlaysMakeDrawerDraggable({
  group: overlaySelectGroup,
  handle: overlayDrawerHandle,
  container: document.getElementById("overlay-output-wrap"),
  storageKey: "diffusion_overlay_drawer_top_analytics",
  onToggle: setOverlayDrawerOpen,
});

// Close the detail modal when clicking the backdrop (outside the box).
detailPanel.addEventListener("click", function (e) {
  if (e.target === detailPanel) {
    hideDetail();
  }
});

// Escape closes the detail modal (matches the generator's modals),
// unless a shallower dialog is open over it and gets the key first.
document.addEventListener("keydown", function (e) {
  if (e.key !== "Escape") {
    return;
  }
  if (aCollectionDialogIsOpen()) {
    return;
  }
  if (!detailPanel.classList.contains("hidden")) {
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
document.addEventListener("click", handleComparePinClick);

btnDeleteConfirm.addEventListener("click", confirmDelete);
btnDeleteCancel.addEventListener("click", closeDeleteModal);
btnDeleteClose.addEventListener("click", closeDeleteModal);
if (btnBulkDelete) {
  btnBulkDelete.addEventListener("click", openBulkDeleteModal);
}
modalDelete.addEventListener("click", function (e) {
  if (e.target === modalDelete) {
    closeDeleteModal();
  }
});

// Collections: the tab strip, the chooser, and the delete confirm.
if (collectionTabs) {
  collectionTabs.addEventListener(
    "click", onCollectionTabClick
  );
}
if (collectionChoices) {
  collectionChoices.addEventListener(
    "change", onCollectionChoiceToggle
  );
}
if (btnNewCollection) {
  btnNewCollection.addEventListener(
    "click", onCreateCollectionFromChooser
  );
}
if (newCollectionName) {
  newCollectionName.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      onCreateCollectionFromChooser();
    }
  });
}
if (btnCollectionsDone) {
  btnCollectionsDone.addEventListener(
    "click", closeCollectionChooser
  );
}
if (btnCollectionsClose) {
  btnCollectionsClose.addEventListener(
    "click", closeCollectionChooser
  );
}
if (modalCollections) {
  modalCollections.addEventListener("click", function (e) {
    if (e.target === modalCollections) {
      closeCollectionChooser();
    }
  });
}
btnColDeleteConfirm.addEventListener(
  "click", confirmCollectionDelete
);
btnColDeleteCancel.addEventListener(
  "click", closeCollectionDeleteModal
);
btnColDeleteClose.addEventListener(
  "click", closeCollectionDeleteModal
);
modalCollectionDelete.addEventListener("click", function (e) {
  if (e.target === modalCollectionDelete) {
    closeCollectionDeleteModal();
  }
});

// Escape closes whichever collection dialog is open. Separate from
// the detail modal's handler above because these two are shallow
// dialogs that can sit over it, so the innermost closes first.
document.addEventListener("keydown", function (e) {
  if (e.key !== "Escape") {
    return;
  }
  if (!modalCollectionDelete.classList.contains("hidden")) {
    closeCollectionDeleteModal();
    return;
  }
  if (!modalCollections.classList.contains("hidden")) {
    closeCollectionChooser();
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
wireOverlayScrubber();
wireTimingPager();

if (overlayHighlightCheckbox) {
  overlayHighlightCheckbox.addEventListener(
    "change", onOverlayHighlightToggle
  );
}

if (runBlendInput) {
  runBlendInput.addEventListener("input", onRunBlendInput);
  runBlendInput.addEventListener(
    "pointerdown", armBlendScrub
  );
  // On window rather than the slider: a drag frequently releases
  // with the pointer well outside the track.
  window.addEventListener("pointerup", endBlendScrub);
  window.addEventListener("pointercancel", endBlendScrub);
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

// Hydrate durable UI state (the "new run" cue and the shared settings
// blob) from the server before the first render, so per-row dots
// reflect saved runs across restarts and the drawer's highlight
// checkbox opens on the value the generator last wrote.
// persistHydrate always runs its callback, even on failure.
overlaysBuildTokenMetrics(tokenMetricsStrip);

persistHydrate(function () {
  updateOverlayHoverHighlight();
  loadAndRender();
});
