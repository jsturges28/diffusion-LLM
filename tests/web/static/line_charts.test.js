// Analytics' line charts, driven without the page.
//
// Strategy: load the controller with only what it reads, overlays.js,
// overlay_series.js and chart_support.js, into the DOM stub, never
// analytics.js, and create it the way the page does. Chart is a
// recorder of what each chart was given, and the stub's document
// finds any element by id. The Stopping chart is handed frames shaped
// like DiffusionGemma's (every position of a draft carrying its
// entropy and whether it changed, a commit carrying none), the
// metrics charts a payload shaped like the metrics endpoint's.
// Plugins and callbacks are read off the recorded configs and called
// the way Chart.js calls them, and pagers and pins are pressed rather
// than set.
//
// Passing proves the controller works with no page around it: the
// Stopping chart plots each draft's mean entropy on a log axis with
// gaps at commits, its threshold, its stop marks and its canvas
// boundaries; the tooltip's glow breaks where the line breaks; each
// slot pages only between what a run can draw; the pins decide what
// a line draws at rest, and the crossfade borrows the charts only for
// a drag. The page's side of the Stopping chart, and the readout, are
// in analytics_stopping.test.js.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, makeElement } = require("./dom_stub.js");

// What the controller reads, in the order Analytics loads it.
const SCRIPTS = [
  "overlays.js",
  "overlay_series.js",
  "chart_support.js",
  "line_charts.js",
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

// Three canvases, each a draft and then its commit but the last.
function threeCanvases() {
  return {
    frames: [
      draft(4, 4), commit(), draft(3, 4), commit(), draft(2, 4),
    ],
    canvas_index: [0, 0, 1, 1, 2],
    stop_rule: Object.assign({}, RULE),
  };
}

// A run of three frames the way the metrics endpoint sends it.
function metrics() {
  return {
    run_id: "run-a",
    convergence: [
      { frame: 0, resolved_ratio: 0 },
      { frame: 1, resolved_ratio: 0.5 },
      { frame: 2, resolved_ratio: 1 },
    ],
    convergence_basis: "tokens",
    total_frames: 3,
    model_type: "diffusion",
    model_label: "LLaDA-8B-Instruct",
    per_frame_elapsed: [0.5, 1, 1.5],
    remask_edits: [],
    mean_conf: [0.2, 0.6, 0.9],
    tokens_produced: [0, 2, 4],
  };
}

// The same run edited at frame 1, carrying the run it branched from,
// so each line chart draws two series for the pins to choose between.
function editedMetrics() {
  const data = metrics();
  data.remask_edits = [{ frame_index: 1, token_positions: [1] }];
  data.original_per_frame_elapsed = [0.5, 1, 1.4];
  data.original_mean_conf = [0.2, 0.5, 0.8];
  return data;
}

// The controller as the page creates it, with Chart recording every
// chart it is asked for and the crossfade parked at `page.blend`.
function load() {
  const page = loadPage({ scripts: SCRIPTS });
  page.blend = 1;
  page.built = [];
  page.context.performance = { now: () => 0 };
  page.context.Chart = function (ctx, config) {
    const chart = {
      config: config,
      destroyed: false,
      destroy() {
        chart.destroyed = true;
      },
      update() {},
      resize() {},
    };
    page.built.push(chart);
    return chart;
  };
  page.lineCharts = page.context.lineChartsCreate({
    readBlend: () => page.blend,
  });
  return page;
}

// Frames the way renderRunOverlays hands them over: the payload with
// its two series read off it.
function frames(page, data) {
  data.records_available = true;
  data.series = page.context.overlaySeriesOf(data, false);
  data.baseline = page.context.overlaySeriesOf(data, true);
  return data;
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function stoppingConfig(page, data) {
  page.lineCharts.renderStopping(frames(page, data));
  assert.equal(page.built.length, 1, "no chart was built");
  return page.built[0].config;
}

function pluginOf(config, id) {
  const plugin = config.plugins.find((each) => each.id === id);
  assert.ok(plugin, "no plugin " + id);
  return plugin;
}

// -- the Stopping chart --

test("each draft's mean entropy is plotted, commits as gaps", () => {
  const page = load();

  const config = stoppingConfig(page, twoCanvases());
  const line = config.data.datasets[0];

  assert.deepEqual(
    host(line.data), [4, 0.03, 0.002, null, 3, 0.2]
  );
  assert.equal(line.spanGaps, false);
});

test("the axis is logarithmic and the threshold is drawn", () => {
  const page = load();

  const config = stoppingConfig(page, twoCanvases());
  const ids = config.plugins.map((plugin) => plugin.id);

  assert.equal(config.options.scales.y.type, "logarithmic");
  assert.ok(ids.includes("stopThreshold"));
  assert.ok(ids.includes("canvasBoundaries"));
});

test("a stopped canvas and a still frame wear their marks", () => {
  // Frame 2 is the first canvas's last draft, which stopped; frame 5
  // changed nothing but its canvas is still going.
  const page = load();

  const line = stoppingConfig(page, twoCanvases()).data.datasets[0];

  assert.deepEqual(host(line.pointRadius), [0, 0, 4, 0, 0, 2.5]);
  assert.equal(line.pointBackgroundColor[2], "#00ff41");
  assert.equal(line.pointBackgroundColor[5], "transparent");
});

test("a canvas that ran out of steps wears no stop mark", () => {
  const data = twoCanvases();
  data.stop_rule.max_denoising_steps = 3;
  const page = load();

  const line = stoppingConfig(page, data).data.datasets[0];

  assert.equal(line.pointRadius[2], 2.5);
});

test("an edited run draws its original beneath it", () => {
  const data = twoCanvases();
  data.original_frames = [
    draft(4, 4), draft(0.5, 2), draft(0.001, 0), commit(),
  ];
  data.remask_edits = [{ frame_index: 1, token_positions: [2] }];
  const page = load();

  const config = stoppingConfig(page, data);

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
  const page = load();
  const { confidence } = withPagers(page);
  page.lineCharts.renderMetrics(metrics(), false);

  page.lineCharts.renderStopping(frames(page, data));

  assert.equal(page.lineCharts.chart("stopping"), null);
  assert.equal(page.registry.get("stopping-section").hidden, true);
  assert.equal(page.registry.get("confidence-section").hidden, false);
  // With no Stopping page on offer, the slot has one page to show
  // and no pager to choose with.
  assert.equal(confidence[0].parent.hidden, true);
});

test("a run that never measured entropy gets no chart", () => {
  const data = twoCanvases();
  data.frames = [commit(), commit()];
  data.canvas_index = [0, 0];
  const page = load();

  page.lineCharts.renderStopping(frames(page, data));

  assert.equal(page.built.length, 0);
});

test("the axis labels only its powers of ten", () => {
  const page = load();

  const config = stoppingConfig(page, twoCanvases());
  const label = config.options.scales.y.ticks.callback;

  assert.equal(label(0.001), "0.001");
  assert.equal(label(10), "10");
  assert.equal(label(0.002), "");
});

// Where a boundary plugin draws its markers, read in frames: the fake
// x scale puts frame N at pixel N.
function boundaryFrames(plugin) {
  const frames = [];
  plugin.afterDatasetsDraw({
    scales: {
      x: { getPixelForValue: (value) => value },
      y: { top: 0, bottom: 10 },
    },
    ctx: {
      save() {}, restore() {}, setLineDash() {}, beginPath() {},
      stroke() {}, lineTo() {},
      moveTo(x) {
        frames.push(x);
      },
    },
  });
  return frames;
}

test("boundaries fall where the canvas index changes", () => {
  const page = load();

  const config = stoppingConfig(page, threeCanvases());

  assert.deepEqual(
    boundaryFrames(pluginOf(config, "canvasBoundaries")), [2, 4]
  );
});

test("a run without canvas indices draws no boundary", () => {
  const data = twoCanvases();
  delete data.canvas_index;
  const page = load();

  const config = stoppingConfig(page, data);

  assert.deepEqual(
    boundaryFrames(pluginOf(config, "canvasBoundaries")), []
  );
});

// -- the tooltip's glow --

// A canvas context that keeps the path it is asked to draw.
function pathRecorder() {
  const calls = [];
  const ctx = {
    save() {}, restore() {}, beginPath() {}, rect() {}, clip() {},
    stroke() {}, fill() {}, arc() {},
    moveTo(x, y) { calls.push(["move", x, y]); },
    lineTo(x, y) { calls.push(["line", x, y]); },
  };
  return { calls, ctx };
}

// Two points, a gap where a commit carries no entropy, and a third.
const GAPPED = [
  { x: 0, y: 10 }, { x: 10, y: 20 }, { skip: true, x: 20, y: NaN },
  { x: 30, y: 5 },
];

// The burn-through plugin over one dataset with its tooltip box up,
// called the way Chart.js calls it after drawing.
function glowOver(plugin, dataset) {
  const { calls, ctx } = pathRecorder();
  plugin.afterDraw({
    tooltip: { opacity: 1, x: 0, y: 0, width: 100, height: 50 },
    ctx: ctx,
    canvas: { id: "chart-stopping" },
    data: { datasets: [dataset] },
    getDatasetMeta: () => ({ hidden: false, data: GAPPED }),
    getActiveElements: () => [],
  });
  return calls;
}

test("the glow breaks where a line that spans no gaps breaks", () => {
  const page = load();
  const config = stoppingConfig(page, twoCanvases());
  const line = { spanGaps: false, borderColor: "#a98bff" };

  assert.deepEqual(glowOver(pluginOf(config, "burnThrough"), line), [
    ["move", 0, 10], ["line", 10, 20], ["move", 30, 5],
  ]);
});

test("the glow bridges a gap the line itself bridges", () => {
  const page = load();
  const config = stoppingConfig(page, twoCanvases());
  const line = { spanGaps: true, borderColor: "#a98bff" };

  assert.deepEqual(glowOver(pluginOf(config, "burnThrough"), line), [
    ["move", 0, 10], ["line", 10, 20], ["line", 30, 5],
  ]);
});

test("a tooltip over the Stopping line adds no segment", () => {
  // The Stopping line's own dataset, as the chart was handed it: its
  // spanGaps decides, as it does for the line.
  const page = load();
  const config = stoppingConfig(page, twoCanvases());
  const line = config.data.datasets[0];

  assert.deepEqual(glowOver(pluginOf(config, "burnThrough"), line), [
    ["move", 0, 10], ["line", 10, 20], ["move", 30, 5],
  ]);
});

// -- the slots --

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

// Both slots' pagers, found the way the controller asks for them.
function withPagers(page) {
  const timing = pagerButtons(
    "data-timing-page", ["elapsed", "tps"]
  );
  const confidence = pagerButtons(
    "data-confidence-page", ["confidence", "stopping"]
  );
  page.context.document.querySelectorAll = (selector) =>
    selector === "[data-timing-page]" ? timing
      : selector === "[data-confidence-page]" ? confidence : [];
  return { timing, confidence };
}

// A pager button pressed, as its listener reads it.
function press(button) {
  button.dispatch("click", { currentTarget: button });
}

test("with only Stopping drawable, the slot shows it", () => {
  const page = load();

  page.lineCharts.renderStopping(frames(page, twoCanvases()));

  assert.equal(page.registry.get("stopping-section").hidden, false);
  assert.equal(page.registry.get("confidence-section").hidden, true);
});

test("with both drawable, the chosen page shows", () => {
  const page = load();
  const { confidence } = withPagers(page);
  page.lineCharts.wire();
  page.lineCharts.renderMetrics(metrics(), false);
  page.lineCharts.renderStopping(frames(page, twoCanvases()));
  const { registry } = page;

  assert.equal(registry.get("confidence-section").hidden, false);
  assert.equal(registry.get("stopping-section").hidden, true);

  press(confidence[1]);

  assert.equal(registry.get("confidence-section").hidden, true);
  assert.equal(registry.get("stopping-section").hidden, false);
});

test("a slot's pagers never answer to the other's readiness", () => {
  // Timing can draw both its pages and Confidence neither of its own.
  const page = load();
  const { timing, confidence } = withPagers(page);
  const data = metrics();
  delete data.mean_conf;

  page.lineCharts.renderMetrics(data, false);

  assert.equal(timing[0].parent.hidden, false);
  assert.equal(confidence[0].parent.hidden, true);
  assert.equal(timing[0].disabled, true);
  assert.equal(timing[1].disabled, false);
});

test("Timing still falls back to the page it can draw", () => {
  // The page a slot last chose is kept from one run to the next, and
  // the next may not be able to draw it: this one has no count of
  // the tokens it produced.
  const page = load();
  const { timing } = withPagers(page);
  page.lineCharts.wire();
  page.lineCharts.renderMetrics(metrics(), false);
  press(timing[1]);
  assert.equal(page.registry.get("tps-section").hidden, false);
  page.lineCharts.clearMetrics();
  const next = metrics();
  delete next.tokens_produced;

  page.lineCharts.renderMetrics(next, false);

  assert.equal(page.registry.get("timing-section").hidden, false);
  assert.equal(page.registry.get("tps-section").hidden, true);
});

// -- the object the page holds --

test("the charts need a way to read the crossfade", () => {
  const page = load();
  const refused = (error) =>
    error.name === "TypeError" && /readBlend/.test(error.message);

  assert.throws(() => page.context.lineChartsCreate(), refused);
  assert.throws(() => page.context.lineChartsCreate({}), refused);
});

test("the metrics payload draws four charts", () => {
  const page = load();

  page.lineCharts.renderMetrics(metrics(), false);

  for (const name of ["convergence", "timing", "tps", "confidence"]) {
    assert.ok(page.lineCharts.chart(name), name);
  }
  assert.equal(page.built.length, 4);
  assert.equal(
    page.registry.get("convergence-section").hidden, false
  );
});

test("an autoregressive run draws no convergence chart", () => {
  const page = load();

  page.lineCharts.renderMetrics(metrics(), true);

  assert.equal(page.lineCharts.chart("convergence"), null);
  assert.equal(page.built.length, 3);
  assert.equal(
    page.registry.get("convergence-section").hidden, true
  );
});

test("clearing the metrics charts destroys every one", () => {
  const page = load();
  page.lineCharts.renderMetrics(metrics(), false);

  page.lineCharts.clearMetrics();

  assert.equal(page.built.length, 4);
  assert.ok(page.built.every((chart) => chart.destroyed));
});

test("a name the controller does not draw answers null", () => {
  const page = load();
  page.lineCharts.renderMetrics(metrics(), false);

  assert.equal(page.lineCharts.chart("entropy"), null);
  assert.equal(page.lineCharts.chart("constructor"), null);
});

// -- the pins and the crossfade --

// One chart's two pins, found the way the controller asks for them.
function pinButtons(page, name) {
  const buttons = ["original", "edited"].map((series) => {
    const button = makeElement(null);
    button.className = "compare-pin-btn";
    button.setAttribute("data-chart", name);
    button.setAttribute("data-series", series);
    return button;
  });
  const own = '.compare-pin-btn[data-chart="' + name + '"]';
  page.context.document.querySelectorAll = (selector) =>
    selector === ".compare-pin-btn" || selector === own
      ? buttons : [];
  return buttons;
}

// The alpha a line chart's blend plugin draws each dataset at, asked
// the way Chart.js asks it before and after drawing each one.
function drawnAlphas(page, name) {
  const chart = page.lineCharts.chart(name);
  const plugin = pluginOf(chart.config, "seriesBlend-" + name);
  const canvas = {
    data: chart.config.data,
    ctx: { save() {}, restore() {}, globalAlpha: 1 },
  };
  const alphas = [];
  for (let i = 0; i < canvas.data.datasets.length; i++) {
    plugin.beforeDatasetDraw(canvas, { index: i });
    alphas.push(canvas.ctx.globalAlpha);
    plugin.afterDatasetDraw(canvas);
  }
  return alphas;
}

// One turn of the stub's animation clock, which settles an ease.
function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("a pin turns its run off, and the last lit pin refuses", () => {
  const page = load();
  const [original, edited] = pinButtons(page, "timing");
  page.lineCharts.wire();
  page.lineCharts.renderMetrics(editedMetrics(), false);
  assert.deepEqual(drawnAlphas(page, "timing"), [1, 1]);

  page.context.document.dispatch("click", { target: original });

  assert.deepEqual(drawnAlphas(page, "timing"), [0, 1]);
  assert.equal(original.classList.contains("is-on"), false);
  assert.equal(edited.classList.contains("is-locked"), true);

  page.context.document.dispatch("click", { target: edited });

  assert.deepEqual(drawnAlphas(page, "timing"), [0, 1]);
});

test("a drag borrows the line charts and a release returns them",
  async () => {
    const page = load();
    const pins = pinButtons(page, "timing");
    page.lineCharts.renderMetrics(editedMetrics(), false);
    page.blend = 0.25;

    page.lineCharts.armScrub();
    page.lineCharts.followBlend();
    await tick();

    assert.deepEqual(drawnAlphas(page, "timing"), [0.75, 0.25]);
    assert.ok(pins.every((pin) => pin.classList.contains(
      "is-previewing"
    )));

    page.lineCharts.endScrub();
    await tick();

    assert.deepEqual(drawnAlphas(page, "timing"), [1, 1]);
    assert.ok(pins.every((pin) => !pin.classList.contains(
      "is-previewing"
    )));
  });

test("a keyboard adjustment leaves the line charts on their pins",
  async () => {
    // An input with no press before it, which is what an arrow key on
    // a focused slider produces.
    const page = load();
    page.lineCharts.renderMetrics(editedMetrics(), false);
    page.blend = 0.25;

    page.lineCharts.followBlend();
    await tick();

    assert.deepEqual(drawnAlphas(page, "timing"), [1, 1]);
  });

test("a run opened mid-drag gets its charts back at once",
  async () => {
    const page = load();
    const pins = pinButtons(page, "timing");
    page.lineCharts.renderMetrics(editedMetrics(), false);
    page.blend = 0.25;
    page.lineCharts.armScrub();
    page.lineCharts.followBlend();
    await tick();

    page.lineCharts.resetScrub();

    assert.deepEqual(drawnAlphas(page, "timing"), [1, 1]);
    assert.ok(pins.every((pin) => !pin.classList.contains(
      "is-previewing"
    )));
  });
