// Analytics' Stopping chart and the detail modal's stopping readout.
//
// Strategy: load the Analytics page into the DOM stub and hand it
// saved runs the way loading one does, with frames shaped like
// DiffusionGemma's (every position of a draft carrying its entropy
// and whether it changed, a commit carrying no entropy) and the
// `stop_rule` the frames endpoint reports. The chart's configuration
// is read off a recording Chart constructor; the readout's words off
// the element itself. A run with no rule is the control.
//
// Passing proves the chart plots each draft's mean entropy on a log
// axis with gaps at commits, marks where each canvas stopped and
// where nothing changed, draws the threshold, shares the Confidence
// slot without disturbing Timing's, stays out of a run with no rule,
// and that the modal's readout follows the scrubber and the
// crossfade the way the generator's does.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, makeElement } = require("./dom_stub.js");

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

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function stoppingConfig(page) {
  page.context.renderStoppingChart(page.context.overlayData);
  assert.equal(page.charts.length, 1, "no chart was built");
  return page.charts[0];
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

test("each draft's mean entropy is plotted, commits as gaps", () => {
  const page = pageWith(twoCanvases(), 5);

  const config = stoppingConfig(page);
  const line = config.data.datasets[0];

  assert.deepEqual(
    host(line.data), [4, 0.03, 0.002, null, 3, 0.2]
  );
  assert.equal(line.spanGaps, false);
});

test("the axis is logarithmic and the threshold is drawn", () => {
  const page = pageWith(twoCanvases(), 5);

  const config = stoppingConfig(page);
  const ids = config.plugins.map((plugin) => plugin.id);

  assert.equal(config.options.scales.y.type, "logarithmic");
  assert.ok(ids.includes("stopThreshold"));
  assert.ok(ids.includes("canvasBoundaries"));
});

test("a stopped canvas and a still frame wear their marks", () => {
  // Frame 2 is the first canvas's last draft, which stopped; frame 5
  // changed nothing but its canvas is still going.
  const page = pageWith(twoCanvases(), 5);

  const line = stoppingConfig(page).data.datasets[0];

  assert.deepEqual(host(line.pointRadius), [0, 0, 4, 0, 0, 2.5]);
  assert.equal(line.pointBackgroundColor[2], "#00ff41");
  assert.equal(line.pointBackgroundColor[5], "transparent");
});

test("a canvas that ran out of steps wears no stop mark", () => {
  const data = twoCanvases();
  data.stop_rule.max_denoising_steps = 3;
  const page = pageWith(data, 5);

  const line = stoppingConfig(page).data.datasets[0];

  assert.equal(line.pointRadius[2], 2.5);
});

test("an edited run draws its original beneath it", () => {
  const data = twoCanvases();
  data.original_frames = [
    draft(4, 4), draft(0.5, 2), draft(0.001, 0), commit(),
  ];
  data.remask_edits = [{ frame_index: 1, token_positions: [2] }];
  const page = pageWith(data, 5);

  const config = stoppingConfig(page);

  assert.equal(config.data.datasets.length, 2);
  assert.equal(config.data.datasets[0].label, "Original");
  assert.equal(config.data.datasets[1].label, "Edited");
  assert.deepEqual(
    host(config.data.datasets[0].data), [4, 0.5, 0.001, null]
  );
});

test("a run with no rule gets no chart and no page", () => {
  const data = twoCanvases();
  data.stop_rule = null;
  const page = pageWith(data, 5);

  page.context.renderStoppingChart(page.context.overlayData);

  assert.equal(page.charts.length, 0);
  assert.equal(page.context.slotReady.confidence.stopping, false);
  assert.equal(page.registry.get("stopping-section").hidden, true);
});

test("a run that never measured entropy gets no chart", () => {
  const data = twoCanvases();
  data.frames = [commit(), commit()];
  data.canvas_index = [0, 0];
  const page = pageWith(data, 1);

  page.context.renderStoppingChart(page.context.overlayData);

  assert.equal(page.charts.length, 0);
});

test("the axis labels only its powers of ten", () => {
  const { context } = pageWith(twoCanvases(), 5);

  assert.equal(context.stoppingTickLabel(0.001), "0.001");
  assert.equal(context.stoppingTickLabel(10), "10");
  assert.equal(context.stoppingTickLabel(0.002), "");
});

test("boundaries fall where the canvas index changes", () => {
  const { context } = pageWith(twoCanvases(), 5);

  assert.deepEqual(
    host(context.stoppingBoundaries([0, 0, 1, 1, 2])), [2, 4]
  );
  assert.deepEqual(host(context.stoppingBoundaries(null)), []);
});

// -- the slot --

test("with only Stopping drawable, the slot shows it", () => {
  const page = pageWith(twoCanvases(), 5);

  page.context.renderStoppingChart(page.context.overlayData);

  assert.equal(page.registry.get("stopping-section").hidden, false);
  assert.equal(page.registry.get("confidence-section").hidden, true);
});

test("with both drawable, the chosen page shows", () => {
  const page = pageWith(twoCanvases(), 5);
  const { context, registry } = page;
  context.slotReady.confidence.confidence = true;
  context.renderStoppingChart(context.overlayData);

  assert.equal(registry.get("confidence-section").hidden, false);
  assert.equal(registry.get("stopping-section").hidden, true);

  context.setSlotPage("confidence", "stopping");

  assert.equal(registry.get("confidence-section").hidden, true);
  assert.equal(registry.get("stopping-section").hidden, false);
});

// Fake pager buttons, so the scoping can be seen: the stub's
// document finds no elements by selector of its own.
function pagerButtons(attribute, pages) {
  const pager = makeElement(null);
  pager.className = "alt-pager";
  return pages.map((page) => {
    const button = makeElement(null);
    button.setAttribute(attribute, page);
    pager.appendChild(button);
    return button;
  });
}

test("a slot's pagers never answer to the other's readiness", () => {
  const page = pageWith(twoCanvases(), 5);
  const { context } = page;
  const timing = pagerButtons(
    "data-timing-page", ["elapsed", "tps"]
  );
  const confidence = pagerButtons(
    "data-confidence-page", ["confidence", "stopping"]
  );
  context.document.querySelectorAll = (selector) =>
    selector === "[data-timing-page]" ? timing
      : selector === "[data-confidence-page]" ? confidence : [];
  context.slotReady.timing.elapsed = true;
  context.slotReady.timing.tps = true;

  context.applySlotPage("timing");
  context.applySlotPage("confidence");

  assert.equal(timing[0].parent.hidden, false);
  assert.equal(confidence[0].parent.hidden, true);
  assert.equal(timing[0].disabled, true);
  assert.equal(timing[1].disabled, false);
});

test("Timing still falls back to the page it can draw", () => {
  const page = pageWith(twoCanvases(), 5);
  const { context, registry } = page;
  context.slotPage.timing = "tps";
  context.slotReady.timing.elapsed = true;

  context.applySlotPage("timing");

  assert.equal(registry.get("timing-section").hidden, false);
  assert.equal(registry.get("tps-section").hidden, true);
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
