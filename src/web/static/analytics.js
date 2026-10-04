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
var chartsErrorNote =
  document.getElementById("charts-error");
var btnCloseDetail =
  document.getElementById("btn-close-detail");

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
var btnBulkStar =
  document.getElementById("btn-bulk-star");
var bulkStarCount =
  document.getElementById("bulk-star-count");
var btnBulkCollect =
  document.getElementById("btn-bulk-collect");
var btnShowAll =
  document.getElementById("btn-show-all");
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

if (chartSupportAvailable) {
  Chart.defaults.color = "#888888";
  Chart.defaults.borderColor = "#1e1e1e";
  Chart.defaults.font.family =
    "'JetBrains Mono', monospace";
  Chart.defaults.font.size = 10;
}

// Chart.js paints every tooltip swatch as a white rect at the full
// box size, then strokes it, then fills a square inset one pixel per
// side. The stroke is centered, so it covers only half that inset
// and leaves a half-pixel of white showing between the border and
// the fill (a whole physical pixel at 2x). Dropping the white
// backing removes that edge; see chartSupportLineLabelColor for the
// border half of the same swatch.
if (chartSupportAvailable) {
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
}

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
// When no corner is free the box has to sit on the data, and the
// line charts' burn-through plugin redraws the line through it.
if (chartSupportAvailable) {
  Chart.Tooltip.positioners.smart =
    function (elements, eventPosition) {
      var chart = this.chart;
      var pad = 10;
      // Box size from the previous frame (0 on the very first
      // hover, corrected on the next frame as it fades in).
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
}

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

// Per-chart tooltip-box visibility (the eye toggle in each header).
var tooltipEnabled = {
  convergence: true,
  timing: true,
  tps: true,
  confidence: true,
  stopping: true,
  entropy: true,
};

// ---- State ----

var allRuns = [];
var sortKey = "created_at";
var sortAsc = false;
var checkedIds = {};
var activeRunId = null;
var gpuName = null;

// Fences the detail panel's two fetches against each other and
// against the panel closing (see detail_requests.js). activeRunId
// above says what is on screen; this says which attempt is allowed
// to paint it, which is the part run id alone cannot answer.
var detailRequests = detailRequestsCreate();
// Compare gets its own counter rather than sharing the detail
// panel's, which is why detailRequestsCreate is a factory. The two
// surfaces open and close independently, and one epoch between them
// would have each cancelling the other's work.
var compareRequests = detailRequestsCreate();
// The fence keys its token by run id, and a comparison is not one
// run. A fixed key gives every comparison the same identity, so only
// the epoch distinguishes them, which is exactly what is wanted:
// each new comparison supersedes the last.
var COMPARE_REQUEST_KEY = "compare";

// The panel's line charts (see line_charts.js). The run crossfade is
// the token viewer's, so the charts are handed a way to read it.
var lineCharts = lineChartsCreate({
  readBlend: function () {
    return tokenViewer.blend();
  },
});

// The panel's token viewer and entropy chart (see token_viewer.js).
// The page passes on what its crossfade does to the line charts, and
// that a run's frames have landed, which builds the Stopping chart.
var tokenViewer = tokenViewerCreate({
  readTokenizer: activeRunTokenizer,
  onShown: lineCharts.renderStopping,
  onBlendReset: lineCharts.resetScrub,
  onBlendInput: lineCharts.followBlend,
  onBlendPress: lineCharts.armScrub,
  onBlendRelease: lineCharts.endScrub,
});

var chartCompareConv = null;
var linkedRunOpened = false;

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

// One error shape for every detail fetch, so a caller has a single
// thing to test. An HTTP status used to go unread entirely: a 404 or
// a 500 was parsed as if it were data, and only failed later at
// whichever property was missing first. A rejected promise, likewise,
// went nowhere at all.
function fetchDetailJson(url, signal) {
  return fetch(url, { signal: signal })
    .then(function (r) {
      if (!r.ok) {
        return {
          error: "Request failed (" + r.status + ")",
        };
      }
      return r.json();
    })
    .catch(function (err) {
      if (detailRequestsIsAbort(err)) {
        // Superseded or closed. The caller's epoch check will
        // discard this anyway; saying so keeps the log quiet.
        return { aborted: true };
      }
      return { error: "Could not load this run" };
    });
}

function fetchMetrics(runId, signal) {
  var url = "/api/analytics/runs/"
    + encodeURIComponent(runId) + "/metrics";
  return fetchDetailJson(url, signal);
}

function fetchCompare(ids, signal) {
  var url = "/api/analytics/compare?ids="
    + ids.map(encodeURIComponent).join(",");
  return fetch(url, { signal: signal })
    .then(function (r) {
      // A refused selection set (too many, or none) answers with a
      // status rather than an array, and reading it as one would
      // draw an empty chart instead of saying what was wrong.
      if (!r.ok) {
        return r.json().then(function (body) {
          throw new Error(
            (body && body.error) || "Comparison failed."
          );
        });
      }
      return r.json();
    });
}

function fetchRunMeta(runId, signal) {
  var url = "/api/analytics/runs/"
    + encodeURIComponent(runId) + "/metadata";
  return fetch(url, { signal: signal })
    .then(function (r) {
      if (!r.ok) {
        throw new Error("metadata " + r.status);
      }
      return r.json();
    });
}

function fetchFrames(runId, signal) {
  var url = "/api/analytics/runs/"
    + encodeURIComponent(runId) + "/frames";
  return fetchDetailJson(url, signal);
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
    // A stopped run says so on the duration rather than in a
    // column of its own, because duration is the field it would
    // otherwise mislead about: forty seconds of a run the user
    // cut short reads exactly like forty seconds of a finished
    // one, and the text ending early looks like the model's
    // choice.
    if (run.partial) {
      return Number(v).toFixed(1) + "s (stopped)";
    }
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

// Everything in the actions-column header that answers to the
// selection. One entry point because they all appear and disappear
// together, and five call sites each remembering to update three
// buttons is five chances for two of them to disagree about whether
// anything is selected.
function updateBulkActions() {
  updateBulkDeleteButton();
  updateBulkCollectButtons();
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

// The filing pair, shown and hidden with the delete beside them. The
// star goes straight to Favorites the way a row's own star does; the
// caret asks where. Kept in one function because a selection that
// can be deleted can always be filed, so the two must never disagree
// about whether there is one.
function updateBulkCollectButtons() {
  if (!btnBulkStar || !btnBulkCollect) { return; }
  var count = checkedRunIds().length;
  if (count < 1) {
    btnBulkStar.hidden = true;
    btnBulkCollect.hidden = true;
    return;
  }
  btnBulkStar.hidden = false;
  btnBulkCollect.hidden = false;
  if (bulkStarCount) {
    bulkStarCount.textContent = "(" + count + ")";
  }
  var noun = count === 1 ? " run" : " runs";
  var label = "Add " + count + " selected" + noun;
  // Named rather than assumed, because inside a collection the star
  // files there instead of into Favorites and the only way to know
  // that before clicking is to be told.
  var into = " to " + bulkFileTargetName();
  btnBulkStar.title = label + into;
  btnBulkStar.setAttribute("aria-label", label + into);
  btnBulkCollect.title = label + " to a collection";
  btnBulkCollect.setAttribute(
    "aria-label", label + " to a collection"
  );
}

function bulkFileTargetName() {
  var target = bulkFileTarget();
  var collection = findCollection(target);
  if (collection) {
    return collection.name;
  }
  return target === FAVORITES_ID ? "Favorites" : "the collection";
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

// Favorites is created on first use rather than shipped empty, so a
// user who never stars anything never sees a tab. Its id is fixed so
// the star always knows where a plain click files to. The creating
// is the server's now; this is here because the tab strip sorts it
// first and the chooser labels it.
var FAVORITES_ID = "favorites";

// The name length the input enforces, matching the server's bound so
// a name that fits the field is never refused after typing it. The
// collection cap is deliberately *not* mirrored here: the server
// refuses past it and says so, and a second copy of a limit is a
// second thing to get out of step.
var COLLECTION_NAME_MAX = 40;

var collections = [];
// null means the All view, which is not a collection: it is every run
// on disk, and it has no membership to add to or remove from.
var activeCollectionId = null;
// Inside a collection, whether the membership filter is relaxed so
// runs can be filed into it from where you are standing. Reset on
// every tab change, so it never outlives the visit that turned it on.
var showAllInCollection = false;
// The run whose chooser is open, and the collection staged for the
// delete confirmation. Both null when their dialog is closed.
var chooserRunId = null;
// The selection the chooser is filing, or null when it was opened
// from one row. Which of the two is set decides whether the dialog
// shows toggles or targets.
var chooserRunIds = null;
var pendingCollectionDelete = null;

// Collections are no longer read from localStorage. They are fetched
// from the server, which owns them, so there is no window-local copy
// to fall out of step with another window's. See refreshCollections.

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

// The collections API. This page no longer writes the list: it sends
// the gesture and the server applies it under the file lock, which
// is what stopped one window's filing from erasing another's.
var collectionsApi = collectionsClientCreate({});

// Take the server's answer as the truth. Every operation returns the
// whole list as it stands afterwards, so this replaces rather than
// merges, and a window that had fallen behind another is level again
// the moment it acts.
//
// Sanitized on the way in for the same reason the stored value was:
// this page should not be the thing that breaks if the shape it is
// handed is not the shape it expects.
function adoptCollections(list) {
  collections = [];
  for (var i = 0; i < list.length; i++) {
    var clean = sanitizeCollection(list[i]);
    if (clean !== null) {
      collections.push(clean);
    }
  }
  rebuildMembershipIndex();
  renderCollectionTabs();
  renderTable();
}

// Send one gesture and adopt what comes back, or report why not.
//
// A refusal now changes nothing here, which is the substantive
// difference from what this replaced. The old path wrote
// localStorage first and told the user afterwards that their change
// existed in this window only and would vanish on reload. There is
// no such state any more: either the server applied it or it did not
// happen, and the message says which.
function runCollectionOp(pending, onDone) {
  return pending
    .then(function (list) {
      adoptCollections(list);
      if (onDone) {
        onDone();
      }
    })
    .catch(function (error) {
      showToast(collectionRefusalText(error));
    });
}

// What a refusal says to a person.
//
// The server's own messages are deliberately terse, because they are
// API text read in a response body or a test: "at most 24
// collections" is exactly right there and reads like a log line in a
// toast. That is what the reason exists for, so the page can own its
// own wording without having to parse a sentence.
//
// A reason with no entry here falls back to the server's message
// rather than to something generic, because a specific sentence in
// the wrong register still beats "something went wrong". The test
// that every reason has an entry is what keeps that a safety net
// rather than the normal path.
// The cap is named without its number on purpose. The page stopped
// holding that limit when the server took ownership of it, and
// quoting it here would put a second copy somewhere it could drift.
// The name bound is different: the input's maxlength needs it
// anyway, so the page already knows it honestly.
var COLLECTION_REFUSALS = {
  collection_limit:
    "Collection limit reached. Delete one to make another.",
  collection_runs_limit:
    "That collection is full.",
  invalid_name:
    "Give the collection a name of "
    + COLLECTION_NAME_MAX + " characters or fewer.",
  unknown_collection:
    "That collection no longer exists. Refresh to catch up.",
  unknown_run:
    "That run no longer exists. Refresh to catch up.",
  collections_full:
    "There is no room to store more collections.",
  use_collection_operations:
    "Collections could not be changed. Reload the page.",
};

function collectionRefusalText(error) {
  if (!error) {
    return "The collection could not be changed.";
  }
  var known = COLLECTION_REFUSALS[error.reason];
  if (known) {
    return known;
  }
  return error.message
    ? error.message
    : "The collection could not be changed.";
}

function findCollection(id) {
  for (var i = 0; i < collections.length; i++) {
    if (collections[i].id === id) {
      return collections[i];
    }
  }
  return null;
}

// Creating Favorites on the first star, applying the collection cap,
// and generating an id from a name all moved to the server with the
// operations, because a limit this page enforces is a limit that
// holds only for pages that choose to.

// Membership, indexed by run rather than scanned per collection.
//
// The stored shape is a list of runs per collection, which is the
// wrong way round for every question this page asks: a table render
// asks "which collections is this run in" once per row, and did it
// by scanning up to 24 arrays each time. This is the same data keyed
// the way it is read.
//
// It is also the shape the server-authoritative collections decided
// under DATA-02 will want. Once add and remove are operations rather
// than a whole-array write, the client stops owning that array, and
// an index keyed by run survives that change where a local copy of
// the list does not.
var membershipIndex = Object.create(null);

function rebuildMembershipIndex() {
  membershipIndex = Object.create(null);
  for (var i = 0; i < collections.length; i++) {
    var collection = collections[i];
    for (var j = 0; j < collection.runs.length; j++) {
      var runId = collection.runs[j];
      if (!membershipIndex[runId]) {
        membershipIndex[runId] = Object.create(null);
      }
      membershipIndex[runId][collection.id] = true;
    }
  }
}

// Whether a run is filed anywhere. What the filled star reports, so
// it answers "did I save this" rather than "is this a favorite": a
// run filed only under Papers is still saved.
function runIsCollected(runId) {
  var entry = membershipIndex[runId];
  if (!entry) { return false; }
  for (var id in entry) {
    if (Object.prototype.hasOwnProperty.call(entry, id)) {
      return true;
    }
  }
  return false;
}

function collectionHasRun(collection, runId) {
  var entry = membershipIndex[runId];
  return !!(entry && entry[collection.id]);
}

// Add or remove one run from one collection.
function setRunMembership(collectionId, runId, member) {
  if (member) {
    return runCollectionOp(
      collectionsApi.addRun(collectionId, runId)
    );
  }
  return runCollectionOp(
    collectionsApi.removeRun(collectionId, runId)
  );
}

// The star's plain click: file to Favorites, or take it back out. One
// click, no dialog, because the common case is deciding a run is
// worth keeping and that decision should cost nothing.
//
// One request, not several, even though a filled star clears the run
// from every collection it is in. Composed here it would be several
// writes that can stop half way; sent as one gesture the server
// applies all of it or none.
function toggleFavorite(runId) {
  return runCollectionOp(collectionsApi.toggleFavorite(runId));
}

// Runs the table should show. The All view is every run; a collection
// is its members, in the table's own sort order rather than the order
// they were filed, so switching tabs does not also change the sort.
//
// Show all relaxes exactly that filter and nothing else. A collection
// is defined as its members, so a collection view has nothing to add
// by construction, and this is the smallest way out of that: the tab
// stays selected so you can still see where you are standing, and
// the runs it does not hold become visible to file.
function visibleRuns() {
  if (activeCollectionId === null) {
    return allRuns;
  }
  var collection = findCollection(activeCollectionId);
  if (!collection) {
    // The collection was deleted while active. Fall back to All
    // rather than show an empty table with no way to tell why.
    activeCollectionId = null;
    showAllInCollection = false;
    return allRuns;
  }
  // Checked after the fallback above, so relaxing the filter cannot
  // leave a tab selected that no longer exists.
  if (showAllInCollection) {
    return allRuns;
  }
  return allRuns.filter(function (run) {
    return collectionHasRun(collection, run.run_id);
  });
}

// Whether a run is already in the collection whose tab is selected.
// Only asked while Show all is on, where it is the difference
// between a row to file and one already filed.
function runIsInActiveCollection(runId) {
  if (activeCollectionId === null) {
    return false;
  }
  var collection = findCollection(activeCollectionId);
  if (!collection) {
    return false;
  }
  return collectionHasRun(collection, runId);
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
  updateShowAllToggle();
}

// The Show all control, which only means anything inside a
// collection. Updated with the tabs, because the active tab is the
// one thing that decides whether it applies.
function updateShowAllToggle() {
  if (!btnShowAll) {
    return;
  }
  if (activeCollectionId === null) {
    btnShowAll.hidden = true;
    return;
  }
  btnShowAll.hidden = false;
  btnShowAll.setAttribute(
    "aria-pressed", showAllInCollection ? "true" : "false"
  );
  btnShowAll.classList.toggle("is-on", showAllInCollection);
  btnShowAll.textContent = showAllInCollection
    ? "Showing all runs"
    : "Show all runs";
}

function onShowAllToggle() {
  if (activeCollectionId === null) {
    return;
  }
  showAllInCollection = !showAllInCollection;
  // Cleared for the same reason a tab change clears it: the rows it
  // referred to may not be on screen any more, and a bulk gesture
  // must never reach a run the user cannot see.
  checkedIds = {};
  selectAllCb.checked = false;
  updateCompareButton();
  updateBulkActions();
  updateShowAllToggle();
  renderTable();
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
    runCollectionOp(
      collectionsApi.rename(collection.id, name)
    );
    return;
  }
  runCollectionOp(collectionsApi.create(name), function () {
    // Switch to what was just made: creating a collection is almost
    // always the first half of filing something into it. The id is
    // the server's, so it is read off the answer rather than
    // predicted from the name.
    var created = collections[collections.length - 1];
    if (created) {
      activeCollectionId = created.id;
      renderCollectionTabs();
      renderTable();
    }
  });
}


// ---- Modal open/close ----
//
// These are native <dialog> elements. showModal is what traps focus,
// makes the rest of the page inert and answers Escape; the previous
// class toggle did none of that, so Tab kept walking the table behind
// an open modal. Guarded because opening an open dialog throws.
function openModal(modal) {
  if (modal && !modal.open) {
    modal.showModal();
  }
}

function closeModal(modal) {
  if (modal && modal.open) {
    modal.close();
  }
}

// Ask before deleting a collection, unless there is nothing to ask
// about.
//
// The confirmation exists because deleting a populated collection
// throws away filing done by hand, which nothing on disk can rebuild.
// An empty one throws away a name. Confirming that too made clearing
// up twice the clicks and, worse, taught the dialog to be dismissed
// without reading, which is exactly the habit it needs the user not
// to have when the collection does hold something.
function openCollectionDeleteModal(collection) {
  if (collectionPresentCount(collection) === 0) {
    deleteCollection(collection.id);
    return;
  }
  pendingCollectionDelete = collection.id;
  if (colDeleteLabel) {
    colDeleteLabel.textContent =
      "\u201c" + collection.name + "\u201d ("
      + collectionPresentCount(collection)
      + " runs)";
  }
  openModal(modalCollectionDelete);
}

function closeCollectionDeleteModal() {
  closeModal(modalCollectionDelete);
}

modalCollectionDelete.addEventListener("close", function () {
  pendingCollectionDelete = null;
});

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
  runCollectionOp(collectionsApi.destroy(id), function () {
    if (activeCollectionId === id) {
      activeCollectionId = null;
      renderCollectionTabs();
      renderTable();
    }
  });
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
  openModal(modalCollections);
}

// The same dialog, opened for a selection rather than one row. The
// runs are captured now rather than read at click time, so a
// selection cleared behind the dialog cannot turn a target click
// into a no-op.
function openCollectionBulkChooser(runIds) {
  chooserRunId = null;
  chooserRunIds = runIds.slice();
  if (collectionsRunLabel) {
    var noun = runIds.length === 1 ? " run" : " runs";
    collectionsRunLabel.textContent =
      runIds.length + " selected" + noun;
  }
  renderCollectionChoices();
  if (newCollectionName) {
    newCollectionName.value = "";
  }
  setCollectionsNote("");
  openModal(modalCollections);
}

function closeCollectionChooser() {
  closeModal(modalCollections);
}

modalCollections.addEventListener("close", function () {
  chooserRunId = null;
  chooserRunIds = null;
});

// Whether the dialog is filing a selection rather than one row.
function chooserIsBulk() {
  return chooserRunIds !== null;
}

// The runs the dialog is acting on, either way.
function chooserRuns() {
  if (chooserRunIds !== null) {
    return chooserRunIds;
  }
  return chooserRunId === null ? [] : [chooserRunId];
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
      chooserIsBulk()
        ? buildCollectionTarget(collections[i])
        : buildCollectionChoice(collections[i])
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

// A target rather than a toggle, which is the whole difference
// between filing one run and filing several.
//
// A checkbox answers "is this run in here", and for a selection the
// answer can be "some of them", which a checkbox cannot say without
// becoming tri-state. Rather than build that, filing a selection is
// add-only: the row reports how many are already in, and clicking it
// files the rest. Taking runs back out stays a per-row gesture,
// where the question has an answer.
function buildCollectionTarget(collection) {
  var row = document.createElement("button");
  row.type = "button";
  row.className = "collection-choice collection-target";
  row.setAttribute("data-collection-id", collection.id);

  var name = document.createElement("span");
  name.textContent = collection.name;
  row.appendChild(name);

  var runs = chooserRuns();
  var already = 0;
  for (var i = 0; i < runs.length; i++) {
    if (collectionHasRun(collection, runs[i])) {
      already++;
    }
  }
  var note = document.createElement("span");
  note.className = "collection-choice-count";
  if (already === 0) {
    note.textContent = "add " + runs.length;
  } else if (already === runs.length) {
    note.textContent = "all in";
    row.disabled = true;
  } else {
    note.textContent =
      already + " in, add " + (runs.length - already);
  }
  row.appendChild(note);
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
  setRunMembership(id, chooserRunId, box.checked).then(
    renderCollectionChoices
  );
}

// The bulk star: straight into Favorites, no dialog. Deliberately
// not the row star's toggle. That star means "in any collection" and
// clears every one of them when it is already lit, which across a
// mixed selection would be a gesture nobody could predict the result
// of. Filing is the half that stays honest in bulk.
function onBulkStar() {
  var runs = checkedRunIds();
  if (runs.length === 0) {
    return;
  }
  fileRunsInto(bulkFileTarget(), runs);
}

// Where the star files. Favorites by default, but the collection you
// are standing in when you are standing in one: having just turned
// on Show all to file into this collection, being sent to Favorites
// instead would be the wrong answer to an unambiguous gesture.
function bulkFileTarget() {
  if (activeCollectionId !== null) {
    return activeCollectionId;
  }
  return FAVORITES_ID;
}

function onBulkCollect() {
  var runs = checkedRunIds();
  if (runs.length === 0) {
    return;
  }
  openCollectionBulkChooser(runs);
}

// Click a target to file the selection into it. Add-only, so this
// closes: there is no second click that would take them out again,
// and leaving the dialog open would invite one.
//
// Reached by a click rather than by the change the checkbox rows
// report, because a target row is a button and a button never fires
// change. Routing this through the change handler is what made the
// whole rendering inert once: the rows drew correctly, counted
// correctly, and could not be clicked.
function onCollectionTargetClick(e) {
  if (!chooserIsBulk()) {
    return;
  }
  var row = e.target.closest("[data-collection-id]");
  if (!row || row.disabled) {
    return;
  }
  var id = row.getAttribute("data-collection-id");
  var runs = chooserRuns();
  if (!id || runs.length === 0) {
    return;
  }
  closeCollectionChooser();
  fileRunsInto(id, runs);
}

// File a selection into one collection, then say so. The toast is
// the whole feedback here: unlike the row star there is no glyph
// that changes, and the rows may not even be on screen.
function fileRunsInto(collectionId, runIds) {
  return runCollectionOp(
    collectionsApi.addRuns(collectionId, runIds),
    function () {
      var collection = findCollection(collectionId);
      var noun = runIds.length === 1 ? " run" : " runs";
      showToast(
        "Added " + runIds.length + noun + " to "
        + (collection ? collection.name : "the collection") + "."
      );
    }
  );
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
  // Created and filed in one request: naming a new collection from
  // the filing dialog is asking for those runs to go in it, and two
  // requests could leave the collection made and empty.
  if (chooserIsBulk()) {
    createCollectionForSelection(name, chooserRuns());
    return;
  }
  collectionsApi
    .create(name, chooserRunId)
    .then(function (list) {
      adoptCollections(list);
      newCollectionName.value = "";
      setCollectionsNote("");
      renderCollectionChoices();
    })
    .catch(function (error) {
      setCollectionsNote(collectionRefusalText(error), true);
    });
}

// Naming a collection while filing a selection closes the dialog,
// for the same reason clicking a target does: the runs are in the
// new collection and there is nothing left to choose.
function createCollectionForSelection(name, runIds) {
  collectionsApi
    .createWithRuns(name, runIds)
    .then(function (list) {
      adoptCollections(list);
      newCollectionName.value = "";
      closeCollectionChooser();
      var noun = runIds.length === 1 ? " run" : " runs";
      showToast(
        "Added " + runIds.length + noun + " to " + name + "."
      );
    })
    .catch(function (error) {
      setCollectionsNote(collectionRefusalText(error), true);
    });
}

function setCollectionsNote(text, warn) {
  if (!collectionsNote) {
    return;
  }
  collectionsNote.textContent = text;
  collectionsNote.classList.toggle("is-warning", !!warn);
}

function aCollectionDialogIsOpen() {
  return !!(modalCollections.open || modalCollectionDelete.open);
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
  // collection that has none of them. Show all makes it the former
  // even inside a collection, since nothing is being filtered out
  // and the message would otherwise tell the reader to turn on a
  // toggle that is already on.
  var inCollection =
    activeCollectionId !== null && !showAllInCollection;
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
      // Under Show all, mark what is already filed here. Without it
      // the table is a flat list with no way to see what the visit
      // was for. The row's own star cannot carry this: it means "in
      // any collection", and giving it a second meaning that depends
      // on the active tab would make one glyph answer two questions.
      if (showAllInCollection && runIsInActiveCollection(run.run_id)) {
        tr.classList.add("row-already-filed");
      }

      // A run the server could not read. It is listed rather than
      // hidden, because a run that quietly disappears looks deleted
      // and invites saving it again. Everything that would try to
      // plot it is withheld; deleting it is left available, since
      // that is the one useful thing to do with it.
      if (run.invalid) {
        tr.classList.add("row-invalid");
      }

      var tdCheck = document.createElement("td");
      tdCheck.className = "col-check";
      if (!run.invalid) {
        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "app-checkbox";
        cb.checked = !!checkedIds[run.run_id];
        cb.setAttribute(
          "data-run-id", run.run_id
        );
        tdCheck.appendChild(cb);
      }
      tr.appendChild(tdCheck);

      // Collecting sits beside selecting rather than beside
      // deleting: both are things you do to a row you have picked
      // out, and a star is read down the column, which suits the
      // left edge.
      var tdStar = document.createElement("td");
      tdStar.className = "col-star";
      if (!run.invalid) {
        tdStar.appendChild(buildRowStar(run.run_id));
        tdStar.appendChild(buildRowCollectCaret(run.run_id));
      }
      tr.appendChild(tdStar);

      // One spanning cell rather than per-column values, because a
      // run that could not be read has no values to put in them, and
      // because the reason has to appear whichever columns are
      // configured. Only the folder name and the reason are shown,
      // and both come from the server as text.
      if (run.invalid) {
        var tdWhy = document.createElement("td");
        tdWhy.className = "cell-invalid";
        tdWhy.colSpan = TABLE_KEYS.length + 1;
        tdWhy.textContent = run.run_id + ": "
          + (run.error || "Could not be read");
        tdWhy.title = tdWhy.textContent;
        tr.appendChild(tdWhy);
      }

      for (var k = 0; !run.invalid && k < TABLE_KEYS.length; k++) {
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
          if (persistIsNewRun(run.run_id)) {
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
      // The invalid row's spanning cell already covers this column,
      // so appending it too would push the row a cell wide and
      // misalign the delete buttons down the table.
      if (!run.invalid) {
        tr.appendChild(tdDiff);
      }

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

function findRun(runId) {
  for (var i = 0; i < allRuns.length; i++) {
    if (allRuns[i].run_id === runId) {
      return allRuns[i];
    }
  }
  return null;
}

function showInvalidDetail(run) {
  // Any request already in flight for another run is dropped, so the
  // panel cannot be repainted by a fetch the user has moved on from.
  detailRequests.begin(run.run_id);
  comparePanel.hidden = true;
  openModal(detailPanel);
  detailTitle.textContent = "Run: " + run.run_id;

  var reason = run.error || "This run could not be read.";
  detailMeta.innerHTML =
    '<div class="run-unreadable">'
    + '<div class="run-unreadable-title">'
    + 'This run could not be opened</div>'
    + '<div class="run-unreadable-reason">'
    + escHtml(reason) + '</div>'
    + '<div class="run-unreadable-hint">'
    + 'Its folder is still on disk. Delete it from the row if you '
    + 'no longer want it.</div>'
    + '</div>';

  // Torn down for the same reason the loaders tear down before their
  // fetch: otherwise the previous run's charts and tokens sit under
  // this run's title, which reads as this run's data.
  clearRunCharts();
  tokenViewer.clear();
  // A class on the panel rather than hiding each section, because
  // every section owns its own `hidden` flag for its own reasons
  // (model type, the timing/rate pager, whether entropy was
  // captured). Hiding them here would mean restoring them there,
  // and getting that wrong loses a chart on the next valid run.
  detailPanel.classList.add("detail-unreadable");
  renderTable();
}

function showDetail(runId) {
  // Hiding the compare panel stops nothing; its late answer
  // would paint behind the dialog.
  compareRequests.cancel();
  activeRunId = runId;
  // A run the catalog could not read has nothing to fetch. Say why
  // and stop, rather than firing two requests that can only fail and
  // leaving the panel on a spinner.
  var listed = findRun(runId);
  if (listed && listed.invalid) {
    showInvalidDetail(listed);
    return;
  }
  // One token for both fetches, taken before either starts, so the
  // pair either paints together or not at all.
  var token = detailRequests.begin(runId);
  comparePanel.hidden = true;
  openModal(detailPanel);
  detailPanel.classList.remove("detail-unreadable");

  // Opening a run clears its "new" dot (and decrements the generator's
  // count on the next visit). Remove just this row's dot in place.
  if (persistIsNewRun(runId)) {
    persistClearNewRun(runId);
    var openedRow = runsTbody.querySelector(
      'tr[data-run-id="' + runId + '"] .run-new-slot'
    );
    if (openedRow) {
      openedRow.textContent = "";
    }
  }

  var run = findRun(runId);
  if (!run) { return; }

  detailTitle.textContent =
    "Run: " + run.run_id;
  // The catalog row is a summary now, so the panel's rows arrive
  // separately. Shown from the summary first so the panel is never
  // blank, then replaced when the full record lands.
  detailMeta.innerHTML = renderRunMeta(run);

  renderTable();
  loadRunMeta(runId, run, token);
  loadRunCharts(runId, run, token);
  loadRunOverlays(runId, run, token);
}

// The catalog carries only what the table draws, so everything else
// the panel shows (the whole prompt, the hyperparameters, the
// tokenizer and context blocks) is fetched for the one run opened.
// Behind the same epoch as the charts and overlays, so a slow answer
// cannot land on a run the user has already moved on from.
function loadRunMeta(runId, summary, token) {
  fetchRunMeta(runId, token && token.signal).then(
    function (meta) {
      if (!detailRequests.accepts(token)) { return; }
      detailMeta.innerHTML = renderRunMeta(meta);
    }
  ).catch(function (error) {
    if (error && error.name === "AbortError") { return; }
    if (!detailRequests.accepts(token)) { return; }
    // The summary is still on screen from above, so a failure here
    // costs the extra rows rather than the panel.
    detailMeta.innerHTML = renderRunMeta(summary);
  });
}

function renderRunMeta(run) {
  var html = "";
  html += '<div class="meta-row">'
    + '<span class="meta-label">Prompt:</span>'
    + '</div>';
  html += '<div class="meta-prompt">'
    + escHtml(promptWithEllipsis(run)) + '</div>';

  var modelName = run.backend || run.model;
  if (modelName) {
    html += '<div class="meta-row">'
      + '<span class="meta-label">Model:</span> '
      + '<span class="meta-value">'
      + escHtml(String(modelName))
      + '</span></div>';
  }
  if (run.conversation_id) {
    html += metaRowHtml(
      "Conversation", String(run.conversation_id)
    );
  }
  if (Number.isInteger(run.turn_index)) {
    html += metaRowHtml(
      "Assistant turn", String(run.turn_index)
    );
  }

  // Stated only for a run that was stopped, so its absence keeps
  // meaning "finished" for every run saved before this existed.
  if (run.partial) {
    html += metaRowHtml(
      "Completion",
      "Stopped before the model finished"
    );
  }

  // Render whatever params this run recorded (model-agnostic).
  // Through metaRowHtml, which escapes the label as well as the
  // value. This loop used to interpolate the key raw, so a saved
  // run whose params carried markup in a *key* could execute it on
  // this origin, next to the model and deletion APIs. The keys come
  // off disk, and a run folder is not a trusted input just because
  // this app usually writes it.
  var params = run.params || {};
  var paramKeys = Object.keys(params);
  for (var j = 0; j < paramKeys.length; j++) {
    var pk = paramKeys[j];
    html += metaRowHtml(
      pk.replace(/_/g, " "), String(params[pk])
    );
  }

  html += modelRevisionMetaRow(run);

  html += processorMetaRow(run);

  html += tokenizerMetaRow(run);

  html += tokenizerVocabMetaRow(run);

  html += modelVocabMetaRow(run);

  html += contextMetaRows(run);

  html += elapsedMetaRows(run);

  html += peakVramMetaRow(run);

  return html;
}

// A gibibyte, past which a figure reads better in the larger unit.
var BYTES_PER_GIB = 1024 * 1024 * 1024;
var BYTES_PER_MIB = 1024 * 1024;

// Bytes at a size a reader can hold in their head. Adaptive because
// the two figures in the row below are three orders of magnitude
// apart: a peak is tens of gibibytes and what a run added on top of
// its weights is tens of mebibytes, and forcing either into the
// other's unit gives "0.01 GiB" or "17408 MiB".
function formatVramBytes(bytes) {
  if (bytes >= BYTES_PER_GIB) {
    return (bytes / BYTES_PER_GIB).toFixed(2) + " GiB";
  }
  return (bytes / BYTES_PER_MIB).toFixed(1) + " MiB";
}

// What the run cost the card. One row carrying two figures, because
// neither says much alone: the peak is mostly the weights the model
// had already loaded, and the distance above the baseline is the part
// a change to the sampler moves. Reporting only the peak is how an
// improvement of 80 MiB hides inside 17 GiB.
//
// Absent for a CPU run, for a run whose worker could not read the
// device, and for every run saved before this existed. All three are
// honestly unmeasured, and no row says that without claiming a zero.
function peakVramMetaRow(run) {
  var cost = run.resources || {};
  var peak = cost.vram_allocated_peak_bytes;
  var start = cost.vram_allocated_start_bytes;
  if (typeof peak !== "number" || typeof start !== "number") {
    return "";
  }
  return metaRowHtml(
    "Peak VRAM",
    formatVramBytes(peak)
      + " (" + formatVramBytes(peak - start)
      + " above baseline)"
  );
}

// A summary's prompt is cut to a fixed length, and saying so beats
// showing a sentence that stops mid-word as though the user typed it
// that way. The full record that follows carries no such flag.
function promptWithEllipsis(run) {
  var prompt = run.prompt || "";
  if (!prompt) { return "N/A"; }
  if (run.prompt_truncated) {
    return prompt + "...";
  }
  return prompt;
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

// How many characters of a commit to show. A full sha is 40 and
// carries no more meaning here than its prefix does: nobody is
// verifying it by eye, and a 40-character value would crowd every
// other row in the panel. Twelve is long enough to identify a commit
// and is what git itself grows to for large repositories.
var REVISION_DISPLAY_CHARS = 12;

// Which commit of the model produced this run, beside the model name
// rather than folded into it: the name says which model, the commit
// says which version of it, and two runs of "SmolLM3-3B" a month
// apart are not necessarily the same model at all.
//
// Absent for a run saved before the field existed and for a local
// checkpoint, which has no commit. Absence is honest in both cases,
// so this degrades to no row rather than to "unknown".
function modelRevisionMetaRow(run) {
  var repro = run.reproducibility || {};
  var revision = repro.model_revision;
  if (!revision) {
    return "";
  }
  return metaRowHtml(
    "Model commit",
    String(revision).slice(0, REVISION_DISPLAY_CHARS)
  );
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
  closeModal(detailPanel);
}

// The tidying that used to live in hideDetail now rides the dialog's
// own close event, so it runs however the panel was dismissed:
// the close button, a backdrop click, or the native Escape, which
// does not pass through any of this file's code at all.
//
// Nulling activeRunId used to be the whole of it, which stopped
// nothing: both fetches stayed in flight and repopulated a panel the
// user had already dismissed.
//
// Worth knowing that `close()` queues this rather than running it
// inline, so anything needing the panel retired before its next
// statement has to do that itself. `showComparison` is the one such
// caller and already does.
detailPanel.addEventListener("close", function () {
  activeRunId = null;
  detailRequests.cancel();
  hideChartsError();
  tokenViewer.clear();
  renderTable();
});

function escHtml(s) {
  var d = document.createElement("div");
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

// ---- The tooltip eyes ----

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
  tooltipEnabled.stopping = true;
  tooltipEnabled.entropy = true;
  var btns = document.querySelectorAll(
    ".tooltip-toggle-btn"
  );
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove("is-off");
    setEyeSlash(btns[i], false);
  }
}

// ---- Loading a run's charts and tokens ----
//
// The line charts are line_charts.js's, and the tokens and the
// entropy chart token_viewer.js's. The page keeps the fetches and
// their fence and the panel's error note, and resets the tooltip eyes
// beside the charts, since the eyes span every chart the panel draws.

// Autoregressive runs have no masked canvas, so the convergence chart
// (percent resolved per frame) would flatline at 100%; it is hidden
// for them while timing and confidence stay, and Entropy by Position
// takes its slot.
function runIsAutoregressive(run) {
  return !!(run && run.model_type === "autoregressive");
}

function loadRunCharts(runId, run, token) {
  // Torn down before the fetch, not inside it, the way
  // loadRunOverlays already did. Destroying on success only meant a
  // slow or failed response left the previous run's charts sitting
  // under the new run's title, which is the reading a user has no
  // way to catch.
  clearRunCharts();

  fetchMetrics(runId, token && token.signal).then(
    function (data) {
      if (!detailRequests.accepts(token)) { return; }
      if (data.aborted) { return; }
      if (data.error) {
        showChartsUnavailable(data.error);
        return;
      }
      renderRunCharts(data, run);
    }
  );
}

// Every chart surface back to empty. Split out because it is now
// called from two places: before a load, and when one fails.
function clearRunCharts() {
  resetTooltipToggles();
  lineCharts.clearMetrics();
}

// Shown when the charting library itself is missing, as opposed to
// one run's metrics failing to load. Names the cause, because the
// two look identical from the user's side and only one of them is
// worth reporting as a bug.
var CHARTS_MISSING_MESSAGE =
  "Charts are unavailable: the charting library did not load."
  + " Everything else on this page still works.";

function showChartsUnavailable(message) {
  clearRunCharts();
  if (chartsErrorNote) {
    chartsErrorNote.textContent = message;
    chartsErrorNote.hidden = false;
  }
}

function hideChartsError() {
  if (chartsErrorNote) {
    chartsErrorNote.hidden = true;
    chartsErrorNote.textContent = "";
  }
}

function renderRunCharts(data, run) {
  if (!chartSupportAvailable) {
    showChartsUnavailable(CHARTS_MISSING_MESSAGE);
    return;
  }
  hideChartsError();
  lineCharts.renderMetrics(data, runIsAutoregressive(run));
}

function loadRunOverlays(runId, run, token) {
  // Torn down before the fetch, not inside it, so switching runs can
  // never leave the previous run's chart, tokens, or crossfade on
  // screen while the new payload is in flight. The viewer's reset
  // covers the rendered tokens, which used to survive the switch
  // because only the globals behind them were reset here.
  tokenViewer.beginRun(runIsAutoregressive(run));
  lineCharts.clearStopping();
  fetchFrames(runId, token && token.signal).then(
    function (data) {
      if (!detailRequests.accepts(token)) { return; }
      if (!data || data.aborted) { return; }
      if (data.error) {
        tokenViewer.showUnavailable();
        return;
      }
      tokenViewer.show(data);
    }
  );
}

// ---- Comparison mode ----

function showComparison(ids) {
  closeModal(detailPanel);
  comparePanel.hidden = false;
  activeRunId = null;
  // Leaving the detail view by any route has to retire its
  // requests, and this route does not go through hideDetail.
  detailRequests.cancel();
  renderTable();

  if (!chartSupportAvailable) {
    // The compare view is nothing but a chart, so there is no
    // reduced version of it to show.
    return;
  }

  // Compare gets its own fence rather than sharing the detail
  // panel's. Two comparisons in flight used to race, and whichever
  // answered last painted, so reopening with a different selection
  // could leave the first one's lines on the chart. Closing
  // cancelled nothing either, which let a dismissed panel repopulate
  // itself.
  var token = compareRequests.begin(COMPARE_REQUEST_KEY);
  fetchCompare(ids, token.signal).then(function (results) {
    if (!compareRequests.accepts(token)) { return; }
    renderComparison(results);
  }).catch(function (error) {
    if (error && error.name === "AbortError") { return; }
    if (!compareRequests.accepts(token)) { return; }
    renderCompareOmissions([{
      run_id: "",
      status: "error",
      label: "Comparison failed",
      message: String(error && error.message ? error.message : error),
    }]);
  });
}

function renderComparison(results) {
  chartCompareConv = chartSupportDestroy(chartCompareConv);

  var convCanvas = document.getElementById(
    "chart-compare-conv"
  );

  var drawable = [];
  var omitted = [];
  var maxConvLen = 0;
  var i;
  for (i = 0; i < results.length; i++) {
    var entry = results[i];
    // The server accounts for every selection now, so a run without
    // data arrives saying why rather than simply not arriving.
    if (entry.status === "data" && entry.convergence) {
      drawable.push(entry);
      if (entry.convergence.length > maxConvLen) {
        maxConvLen = entry.convergence.length;
      }
    } else {
      omitted.push(entry);
    }
  }
  renderCompareOmissions(omitted);

  var convLabels = [];
  for (i = 0; i < maxConvLen; i++) {
    convLabels.push(i);
  }

  var convDatasets = [];
  for (i = 0; i < drawable.length; i++) {
    var res = drawable[i];
    var color = COMPARE_COLORS[i % COMPARE_COLORS.length];
    var cData = [];
    for (var ci = 0; ci < res.convergence.length; ci++) {
      cData.push(
        +(res.convergence[ci].resolved_ratio
          * 100).toFixed(2)
      );
    }
    convDatasets.push({
      // Built by the server, which is the only side that can read
      // the registry and so the only side that knows what a given
      // model's parameters are called.
      label: res.label || res.run_id,
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
}

// Name every selection that produced no line. Silence here was the
// defect: pick three runs, see one line, and nothing on screen says
// whether the other two were autoregressive, deleted, or unreadable.
function renderCompareOmissions(omitted) {
  var box = document.getElementById("compare-omitted");
  if (!box) { return; }
  box.innerHTML = "";
  box.hidden = omitted.length === 0;
  if (omitted.length === 0) { return; }

  for (var i = 0; i < omitted.length; i++) {
    var entry = omitted[i];
    var row = document.createElement("div");
    row.className = "compare-omitted-row";
    var name = entry.label || entry.run_id || "A selected run";
    row.textContent =
      name + ": " + (entry.message || "No data.");
    box.appendChild(row);
  }
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
          title: chartSupportTooltipTitle,
          labelColor: chartSupportLineLabelColor,
        },
      },
      zoom: chartSupportZoomOptions(),
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
  // Closing used to leave the fetch running, so a slow comparison
  // could paint itself into a panel the user had already dismissed
  // and then reappear on the next open.
  compareRequests.cancel();
}

// ---- Zoom button handlers ----

// A chart by the name its header buttons carry. The entropy chart is
// the token viewer's, and the line charts are their controller's.
function chartNamed(name) {
  if (name === "entropy") {
    return tokenViewer.entropyChart();
  }
  return lineCharts.chart(name);
}

function handleZoomClick(e) {
  var btn = e.target.closest(".zoom-btn");
  if (!btn) { return; }
  var chartName = btn.getAttribute("data-chart");
  var action = btn.getAttribute("data-action");
  var chart = chartNamed(chartName);
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
    updateBulkActions();
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
  updateBulkActions();
}

// Selecting a tab. Clears the selection: a checkbox ticked under one
// tab refers to a row that may not exist under the next, and carrying
// it across would put invisible runs in a bulk delete.
function selectCollection(id) {
  if (activeCollectionId === id) {
    return;
  }
  activeCollectionId = id;
  // Show all is per-visit, not a preference. A collection view that
  // quietly showed non-members the next time you opened it would no
  // longer be a collection view, and the tab would be lying about
  // what it holds.
  showAllInCollection = false;
  checkedIds = {};
  selectAllCb.checked = false;
  updateCompareButton();
  updateBulkActions();
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

// The catalog and collections the server inlined, or null. Shaped
// like what `fetchRuns` and the collections client return, so both
// paths below hand them to the same renderer.
function bootAnalyticsState() {
  var boot = window.__BOOT__;
  if (!boot) {
    return null;
  }
  if (!Array.isArray(boot.runs)) {
    return null;
  }
  if (!Array.isArray(boot.collections)) {
    return null;
  }
  return boot;
}

// First render from state we already have. Refresh still refetches:
// two windows can disagree about what is filed where, and this page
// is the one that shows it.
function renderFromState(runs, collections) {
  allRuns = runs;
  checkedIds = {};
  selectAllCb.checked = false;
  adoptCollections(collections);
  updateCompareButton();
  updateBulkActions();
  renderCollectionTabs();
  renderTable();
  openLinkedRun();
}

function loadAndRender() {
  fetchRuns().then(function (runs) {
    allRuns = runs;
    checkedIds = {};
    selectAllCb.checked = false;
    // Ask the server, rather than re-reading this window's copy.
    // Another window may have filed something since, and until this
    // fetched it that only showed up on a full page load, which made
    // Refresh a button that refreshed everything except the one
    // thing two windows can disagree about.
    refreshCollections();
    updateCompareButton();
    updateBulkActions();
    renderCollectionTabs();
    renderTable();
    openLinkedRun();
  });
}

function openLinkedRun() {
  if (linkedRunOpened) {
    return;
  }
  var runId = new URLSearchParams(location.search).get("run");
  if (!runId || !findRun(runId)) {
    return;
  }
  linkedRunOpened = true;
  showDetail(runId);
}

// ---- Delete a run ----

// Where saved runs actually live, as the server resolved it. Filled
// by fetchSystemInfo on load; the default stands in until that
// answers, and is what the server reports anyway unless the user
// passed --results-dir.
var resultsDirLabel = "results";

function runPath(runId) {
  return resultsDirLabel + "/" + runId;
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
      + "<code>" + escHtml(resultsDirLabel)
      + "/</code>. This cannot be undone.";
  } else {
    deleteModalTitle.textContent =
      "Delete " + count + " runs?";
    deleteRunLabel.textContent =
      count + " selected runs will be removed.";
    deleteModalNote.innerHTML =
      "This permanently removes the saved runs from "
      + "<code>" + escHtml(resultsDirLabel)
      + "/</code>. This cannot be undone.";
  }
  btnDeleteConfirm.disabled = false;
  openModal(modalDelete);
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
  closeModal(modalDelete);
}

// Safe to defer: `confirmDelete` copies the ids before it closes.
// This one gains Escape by the migration, having had no handler for
// it at all, so the reset has to happen on close rather than only on
// the routes that used to exist.
modalDelete.addEventListener("close", function () {
  pendingDeleteIds = [];
  btnDeleteConfirm.disabled = false;
});

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
    persistClearNewRun(deletedIds[i]);
    if (activeRunId === deletedIds[i]) {
      hideDetail();
    }
  }
  allRuns = allRuns.filter(function (run) {
    return !removed[run.run_id];
  });
  // A collection holding an id whose folder is gone would show a row
  // that cannot be opened. This page used to prune its own copy and
  // write the result back; now it asks, because the server prunes
  // against what is actually on disk and is the one that decides.
  refreshCollections();
  selectAllCb.checked = false;
  updateCompareButton();
  updateBulkActions();
  renderCollectionTabs();
  renderTable();
}

// Take the server's current list. Used after deleting runs, and by
// Refresh, so a window that has been sitting open while another
// filed something has a way back without a reload.
//
// Quiet on failure: this is a read that nobody asked for by name,
// and a toast for it would fire on every refresh of a page whose
// server has gone away, which the rest of the page already says.
function refreshCollections() {
  return collectionsApi
    .list()
    .then(adoptCollections)
    .catch(function () {});
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
  var chart = chartNamed(name);
  if (chart) {
    chart.options.plugins.tooltip.enabled = enabled;
    chart.update();
  }
}

// ---- Wire up events ----

document.querySelector("#runs-table thead")
  .addEventListener("click", onSortClick);

runsTbody.addEventListener("click", onRowClick);

// Enter opens the focused row's detail, which is what clicking the
// row does and what the keyboard had no way to reach: the row itself
// is not focusable, only its checkbox, star and caret are, and none
// of them opens a run.
//
// Enter is free to take. A checkbox toggles with Space, so Enter on
// one does nothing otherwise, which leaves Space for selecting and
// Enter for opening. Buttons keep it, since Enter is how a button is
// pressed and the star and caret have their own jobs.
runsTbody.addEventListener("keydown", function (e) {
  if (e.key !== "Enter") {
    return;
  }
  if (e.target.closest && e.target.closest("button")) {
    return;
  }
  var tr = e.target.closest("tr[data-run-id]");
  if (!tr) {
    return;
  }
  e.preventDefault();
  showDetail(tr.getAttribute("data-run-id"));
});

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

// Close the detail modal when clicking the backdrop (outside the box).
detailPanel.addEventListener("click", function (e) {
  if (e.target === detailPanel) {
    hideDetail();
  }
});

// Escape is the dialog's own now, and the top layer already orders
// the stack: the most recently opened dialog gets the key, so a
// collection dialog sitting over the detail panel closes first
// without anyone arbitrating it. The hand-rolled version of this had
// to check for that case explicitly, and still left `modal-delete`
// with no Escape at all.

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
if (btnBulkDelete) {
  btnBulkDelete.addEventListener("click", openBulkDeleteModal);
}
if (btnBulkStar) {
  btnBulkStar.addEventListener("click", onBulkStar);
}
if (btnBulkCollect) {
  btnBulkCollect.addEventListener("click", onBulkCollect);
}
if (btnShowAll) {
  btnShowAll.addEventListener("click", onShowAllToggle);
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
// Two listeners, one per rendering, rather than one that branches:
// the two emit different events. A checkbox row reports "change";
// a target row is a button, which only ever reports "click". Each
// handler returns early in the mode it does not own, so the label
// click that accompanies every checkbox change reaches nothing.
if (collectionChoices) {
  collectionChoices.addEventListener(
    "change", onCollectionChoiceToggle
  );
  collectionChoices.addEventListener(
    "click", onCollectionTargetClick
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

// ---- Boot ----

// Eye toggles start "open" (no slash) before any run is opened.
(function () {
  var btns = document.querySelectorAll(".tooltip-toggle-btn");
  for (var i = 0; i < btns.length; i++) {
    setEyeSlash(btns[i], false);
  }
})();

tokenViewer.wire();
lineCharts.wire();

// The "Generation" nav link is revealed by the server, which unhides it
// in the markup when a worker is resident (_reveal_generation_link in
// server.py). It used to be done here, from /api/models, which cost two
// nvidia-smi subprocesses to learn one boolean and shifted every link
// beside it when the answer landed.

// The data root, which the delete confirmation names. Inlined with
// the rest when the server served this page; otherwise fetched, and
// `resultsDirLabel` holds a sensible default until it lands.
//
// The GPU name is not inlined even though it is cheap now. It is only
// a fallback for a run that did not record its own processor, read in
// a detail view, so it has no business on the serve path.
(function adoptResultsDir() {
  var boot = window.__BOOT__;
  if (boot && typeof boot.results_dir === "string") {
    resultsDirLabel = boot.results_dir;
  }
  fetchSystemInfo().then(function (info) {
    if (info.gpu_name) {
      gpuName = info.gpu_name;
    }
    if (info.results_dir) {
      resultsDirLabel = info.results_dir;
    }
  });
})();

// Hydrate durable UI state (the "new run" cue and the shared settings
// blob) from the server before the first render, so per-row dots
// reflect saved runs across restarts and the drawer's highlight
// checkbox opens on the value the generator last wrote.
// persistHydrate is synchronous when that state was inlined, and
// always runs its callback either way.
persistHydrate(function () {
  tokenViewer.refreshHoverHighlight();
  var inlined = bootAnalyticsState();
  if (inlined !== null) {
    renderFromState(inlined.runs, inlined.collections);
    return;
  }
  loadAndRender();
});
