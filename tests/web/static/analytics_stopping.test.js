// The detail modal's stopping readout, and the page's side of the
// Stopping chart.
//
// Strategy: load the Analytics page into the DOM stub and hand it
// saved runs the way loading one does, with frames shaped like
// DiffusionGemma's (every position of a draft carrying its entropy
// and whether it changed, a commit carrying no entropy) and the
// `stop_rule` the frames endpoint reports. The readout's words are
// read off the element itself, and the chart's configuration off a
// recording Chart constructor. A run with no rule is the control.
//
// Passing proves the modal's readout follows the scrubber and the
// crossfade the way the generator's does, and that a run's frames
// landing builds its Stopping chart. What the chart draws is the
// line charts' own, and is tested without the page in
// line_charts.test.js.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, ANALYTICS_SCRIPTS } = require("./dom_stub.js");

const RULE = {
  confidence_threshold: 0.005,
  stability_threshold: 1,
  max_denoising_steps: 48,
};

function draft(entropy, changed) {
  const tokens = [];
  for (let i = 0; i < 4; i++) {
    tokens.push({
      t: " w" + i, m: i < changed, id: 10 + i, c: 0.5, e: entropy,
    });
  }
  return tokens;
}

function commit() {
  const tokens = [];
  for (let i = 0; i < 4; i++) {
    tokens.push({ t: " w" + i, m: false, id: 10 + i, c: 1 });
  }
  return tokens;
}

// Two canvases: the first stops after three drafts, the second is
// still going.
function twoCanvases() {
  return {
    frames: [
      draft(4, 4), draft(0.03, 1), draft(0.002, 0), commit(),
      draft(3, 4), draft(0.2, 0),
    ],
    canvas_index: [0, 0, 0, 0, 1, 1],
    stop_rule: Object.assign({}, RULE),
  };
}

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

// A page holding `data` the way renderRunOverlays leaves it, scrubbed
// to `frame`, with Chart replaced by a recorder of what it was given.
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
  page.charts = [];
  context.Chart = function (ctx, config) {
    page.charts.push(config);
    return { destroy() {}, update() {}, resize() {} };
  };
  return page;
}

// The readout's words, or null while it is hidden.
function words(page) {
  const readout = page.registry.get("stop-readout");
  if (readout.hidden) {
    return null;
  }
  const textOf = (el) => el.children.length === 0
    ? el.textContent
    : el.children.map(textOf).join("");
  return textOf(readout.overlaysStopNodes.text);
}

// -- the chart --

test("a run's frames landing build its Stopping chart", () => {
  const page = pageWith(twoCanvases(), 5);

  page.context.renderRunOverlays(page.context.overlayData);

  const stopping = page.charts.filter((config) =>
    config.plugins.some((plugin) => plugin.id === "stopThreshold")
  );
  assert.equal(stopping.length, 1);
  assert.equal(page.registry.get("stopping-section").hidden, false);
});

// -- the modal's readout --

test("the readout reads the scrubbed frame", () => {
  const page = pageWith(twoCanvases(), 2);
  page.context.refreshStopReadout();
  assert.equal(words(page), "entropy 0.0020 of 0.005, steady");

  page.context.setOverlayFrame(3);
  assert.equal(words(page), "Canvas 1 stopped after 3 steps");

  page.context.setOverlayFrame(4);
  assert.equal(words(page), "entropy 3.00 of 0.005, 4 changing");
});

test("the readout stays hidden for a run with no rule", () => {
  const data = twoCanvases();
  data.stop_rule = null;
  const page = pageWith(data, 2);

  page.context.refreshStopReadout();

  assert.equal(words(page), null);
});

test("the crossfade moves the readout to the original run", () => {
  const data = {
    frames: [draft(4, 4), draft(1.5, 2), draft(0.3, 1)],
    original_frames: [
      draft(4, 4), draft(0.5, 1), draft(0.001, 0), commit(),
    ],
    remask_edits: [{ frame_index: 1, token_positions: [2] }],
    canvas_index: [0, 0, 0],
    stop_rule: Object.assign({}, RULE),
  };
  const page = pageWith(data, 2);
  page.context.refreshStopReadout();
  assert.equal(words(page), "entropy 0.30 of 0.005, 1 changing");

  page.context.compareBlend = 0;
  page.context.refreshTokenMetricsLayer();

  assert.equal(words(page), "entropy 0.0010 of 0.005, steady");
});

test("an edit's first resumed frame is never steady", () => {
  // Frame 2 changed nothing against the frame it resumed from, but
  // it begins the resume's fresh history.
  const data = {
    frames: [
      draft(4, 4), draft(0.5, 1), draft(0.2, 0), draft(0.1, 0),
    ],
    original_frames: [draft(4, 4), draft(0.5, 1), draft(0.3, 0)],
    remask_edits: [{ frame_index: 2, token_positions: [1] }],
    canvas_index: [0, 0, 0, 0],
    stop_rule: Object.assign({}, RULE),
  };
  const page = pageWith(data, 2);

  page.context.refreshStopReadout();
  assert.equal(words(page), "entropy 0.20 of 0.005, steady 0 of 1");

  page.context.setOverlayFrame(3);
  assert.equal(words(page), "entropy 0.10 of 0.005, steady");
});
