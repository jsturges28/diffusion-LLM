// Analytics' Revisions overlay, for a saved run.
//
// Strategy: load the Analytics page into the DOM stub beside this
// file and hand it saved runs the way loading one does, with frames
// shaped like DiffusionGemma's: a changed token reads as masked for a
// frame before it settles again. One run revises, one never does, and
// an edited run carries a pre-edit snapshot. The colour callbacks are
// read off what the overlay hands the renderer, and once through the
// real renderer to show it paints.
//
// Passing proves the option is offered exactly for a diffusion run
// that revised, that each token is tinted by its count at the
// scrubbed frame, that each canvas counts on its own, that the
// pre-edit layer counts its own run while the edited one starts a
// remasked position over, that the strip reads the hovered layer's
// count, that the legend follows the selection, and that loading
// another run forgets the last one's revisions.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "run_candidates.js",
  "candidate_flicker.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

function settled(id) {
  return { t: " w" + id, m: false, id: id, c: 0.5 };
}

function changing(id) {
  return { t: " w" + id, m: true, id: id, c: 0.2 };
}

// Position 0 settles on 10, changes to 20, then to 30; position 1
// settles on 11 and changes to 21. Positions 2 and 3 settle once.
// Revised: position 0 at frames 3 and 6, position 1 at frame 5.
const REVISING = [
  [changing(1), changing(2), changing(3), changing(4)],
  [settled(10), settled(11), changing(3), changing(4)],
  [changing(20), settled(11), settled(12), changing(4)],
  [settled(20), settled(11), settled(12), settled(13)],
  [settled(20), changing(21), settled(12), settled(13)],
  [changing(30), settled(21), settled(12), settled(13)],
  [settled(30), settled(21), settled(12), settled(13)],
];

// Every position settles once and never moves, as on LLaDA.
const SETTLING = [
  [changing(1), changing(2), changing(3), changing(4)],
  [settled(10), settled(11), changing(3), changing(4)],
  [settled(10), settled(11), settled(12), settled(13)],
];

// Resumed at frame 4 with position 2 remasked: its new token is the
// edit's, while position 3 changes on its own. One frame longer than
// the run it replaced.
const BRANCH = [
  [settled(20), settled(11), changing(40), settled(13)],
  [settled(20), settled(11), settled(40), changing(41)],
  [settled(20), settled(11), settled(40), settled(41)],
  [settled(20), settled(11), settled(40), settled(41)],
];

function bootFetch() {
  return function (url) {
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : { success: true, collections: [] };
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    });
  };
}

function copy(frames) {
  return frames.map(
    (frame) => frame.map((token) => Object.assign({}, token))
  );
}

// A page holding `data` the way renderRunOverlays leaves it, scrubbed
// to `frame`.
function pageWith(data, frame) {
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  const { context } = page;
  data.records_available = true;
  data.series = context.overlaySeriesOf(data, false);
  data.baseline = context.overlaySeriesOf(data, true);
  context.overlayData = data;
  context.overlayIsAutoregressive = false;
  context.overlayFrameIndex = frame;
  return page;
}

function editedRun(frame) {
  return pageWith({
    frames: copy(REVISING.slice(0, 4).concat(BRANCH)),
    original_frames: copy(REVISING),
    remask_edits: [{ frame_index: 4, token_positions: [2] }],
  }, frame);
}

// What the overlay hands the renderer, with the renderer replaced by
// a recorder, so the colour callbacks can be asked directly.
function capture(context) {
  let seen = null;
  context.renderOverlayTokens = function (opts) {
    seen = opts;
  };
  context.renderRevisionsOverlay();
  assert.notEqual(seen, null, "the overlay rendered nothing");
  return seen;
}

function colorsOf(colorFor) {
  return [0, 1, 2, 3].map((index) => colorFor(index, settled(99)));
}

function pickerValues(context, data) {
  context.buildOverlaySelect(data);
  const list = context.overlaySelect.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map((item) => item.getAttribute("data-value"));
}

// -- when it is offered --

test("a saved run that revised offers it, after Commit Order", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);

  const values = pickerValues(context, context.overlayData);

  assert.equal(
    values.indexOf("revisions"), values.indexOf("commit") + 1
  );
});

test("a saved run that never revised is not offered it", () => {
  const { context } = pageWith({ frames: copy(SETTLING) }, 2);

  assert.ok(
    !pickerValues(context, context.overlayData).includes("revisions")
  );
});

test("an autoregressive run is never offered it", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);
  context.overlayIsAutoregressive = true;

  assert.equal(context.overlayRevisionsAvailable(), false);
});

// -- what it paints --

test("tokens are tinted by their count at the scrubbed frame", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);

  const colors = colorsOf(capture(context).colorFor);

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), null, null,
  ]);
});

test("an earlier frame counts only what had happened by then", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 4);

  const colors = colorsOf(capture(context).colorFor);

  assert.deepEqual(
    colors, [context.revisionColor(1), null, null, null]
  );
});

test("each canvas counts on its own", () => {
  // Position 0 changes on canvas 0, then the next canvas puts an
  // unrelated token there, which is not a change of mind.
  const frames = [
    [settled(10)], [changing(20)], [settled(20)],
    [settled(30)], [settled(30)],
  ];
  const { context } = pageWith({
    frames: copy(frames), canvas_index: [0, 0, 0, 1, 1],
  }, 4);

  assert.deepEqual(
    [...context.overlayRevisionCountsFor(false)], []
  );
  context.overlayFrameIndex = 2;
  assert.deepEqual(
    [...context.overlayRevisionCountsFor(false)], [1]
  );
});

test("the edited layer starts a remasked position over", () => {
  const { context } = editedRun(6);

  const colors = colorsOf(capture(context).colorFor);

  // Position 0's change at frame 3 is shared history; position 2's
  // new token is the edit's, position 3's the model's own.
  assert.deepEqual(colors, [
    context.revisionColor(1), null, null, context.revisionColor(1),
  ]);
});

test("the pre-edit layer counts the run it came from", () => {
  const { context } = editedRun(6);

  const colors = colorsOf(capture(context).originalColorFor);

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), null, null,
  ]);
});

test("the pre-edit layer holds its last frame past its end", () => {
  const { context } = editedRun(7);

  const colors = colorsOf(capture(context).originalColorFor);

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), null, null,
  ]);
});

test("the strip reads the hovered layer's count", () => {
  const { context } = editedRun(6);
  context.overlayMode = "revisions";

  context.metricsHoverOriginal = false;
  assert.equal(context.metricsExtra(3, settled(41)), "Revisions: 1");
  assert.equal(context.metricsExtra(1, settled(11)), "");
  context.metricsHoverOriginal = true;
  assert.equal(context.metricsExtra(1, settled(21)), "Revisions: 1");
  assert.equal(context.metricsExtra(0, settled(30)), "Revisions: 2");
});

// Every token span drawn into `element`, through the fragments the
// stub keeps as children rather than flattening.
function drawnSpans(element) {
  const spans = [];
  for (const child of element.children || []) {
    if (child.tag === "span") {
      spans.push(child);
    } else {
      spans.push(...drawnSpans(child));
    }
  }
  return spans;
}

test("it paints through the real renderer", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);
  context.overlayMode = "revisions";

  context.renderCurrentOverlay();

  const colors = drawnSpans(context.overlayOutput).map(
    (span) => span.style.color
  );
  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("the legend shows only while Revisions is selected", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);

  context.setOverlayMode("revisions");
  assert.equal(context.overlayRevisionLegend.hidden, false);
  assert.equal(context.overlayLegend.hidden, true);

  context.setOverlayMode("commit");
  assert.equal(context.overlayRevisionLegend.hidden, true);
  assert.equal(context.overlayLegend.hidden, false);
});

test("loading another run forgets the last one's revisions", () => {
  const { context } = pageWith({ frames: copy(REVISING) }, 6);
  assert.equal(context.overlayRevisionsAvailable(), true);

  context.clearOverlay();

  assert.equal(context.overlayRevisions, null);
  assert.equal(context.overlayOriginalRevisions, null);
  assert.equal(context.overlayRevisionLegend.hidden, true);
});
