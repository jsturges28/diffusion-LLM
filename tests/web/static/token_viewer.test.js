// Analytics' token viewer, driven without the page.
//
// Strategy: load the controller with only what it reads,
// custom_select.js, overlays.js, overlay_series.js, chart_support.js,
// run_candidates.js and candidate_flicker.js, into the DOM stub,
// never analytics.js, and create and wire it the way the page's boot
// does, with callbacks that record what they are told. Chart is a
// recorder of what each chart was given, and the stub's document
// finds any element by id. Runs are shaped like the frames
// endpoint's, with entropy declared per frame and position, so the
// bars have a frame to follow. Tokens are hovered, the scrubber and
// the crossfade's slider moved and its pointer pressed and released
// as events rather than set.
//
// Passing proves the viewer works with no page around it: it refuses
// to be created without its callbacks; a run's frames draw its
// tokens, offer its overlays, build its entropy chart and are
// reported once, and a run with nothing to show says so and is not
// reported; the scrubber moves the tokens and the bars together; the
// crossfade is the viewer's, reported as it moves and around a drag,
// and reset for the next run; and a token or a bar under the pointer
// reads its position in the strip and lights its token. The page's
// side of all this is in the analytics_* suites.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// What the controller reads, in the order Analytics loads it.
const SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "overlay_series.js",
  "chart_support.js",
  "run_candidates.js",
  "candidate_flicker.js",
  "token_viewer.js",
];

const WORDS = [" Yeast", " eats", " sugar", "."];

// The callbacks the factory needs, in the order it names them.
const CALLBACKS = [
  "readTokenizer",
  "onShown",
  "onBlendReset",
  "onBlendInput",
  "onBlendPress",
  "onBlendRelease",
];

const ENTROPY_BY_FRAME = {
  name: "entropy",
  unit: "nats",
  axes: ["frame", "position"],
  location: "token_record",
  key: "e",
  capture: "always",
};

const WATERMARK_MEMBERSHIP = {
  name: "watermark_membership",
  unit: "categorical",
  axes: ["position"],
  location: "token_record",
  key: "g",
  capture: "opt_in",
};

const WATERMARK_EVIDENCE = {
  name: "watermark_evidence",
  unit: "categorical",
  axes: ["position"],
  location: "token_record",
  key: "we",
  capture: "opt_in",
};

// Frame `at` of a saved diffusion run: the first `at` + 1 positions
// decided and the rest still masked. Entropy is the frame plus a
// tenth of the position, offset by `base`, so a bar read at the
// wrong frame or position cannot agree by luck.
function frameTokens(at, base) {
  return WORDS.map((word, position) => ({
    t: word,
    m: position > at,
    id: 100 + position,
    c: 0.5,
    e: (base || 0) + at + position / 10,
  }));
}

function frames(base) {
  return [0, 1, 2, 3].map((at) => frameTokens(at, base));
}

// A saved four-frame run, as the frames endpoint serves it, with
// `overrides` replacing keys.
function payload(overrides) {
  return Object.assign({
    run_id: "run",
    frames: frames(0),
    positions: null,
    original_frames: null,
    original_positions: null,
    records_available: true,
    alternatives: null,
    alternatives_available: false,
    original_alternatives: null,
    candidates: null,
    original_candidates: null,
    remask_edits: [],
    canvas_index: [0, 0, 0, 0],
    stop_rule: null,
    signals: [ENTROPY_BY_FRAME],
  }, overrides || {});
}

// The same run edited at frame 1 and saved with the run it branched
// from, so its tokens stack in two layers and the crossfade shows.
function edited() {
  return payload({
    original_frames: frames(10),
    remask_edits: [{ frame_index: 1, token_positions: [2] }],
  });
}

function watermarkTokens(layer, invert) {
  for (let frame = 0; frame < layer.length; frame++) {
    for (let index = 0; index < layer[frame].length; index++) {
      layer[frame][index].g = invert
        ? index % 2 !== 0
        : index % 2 === 0;
      layer[frame][index].we = index !== 0;
    }
  }
}

function watermarked(overrides) {
  const data = payload({
    signals: [
      ENTROPY_BY_FRAME,
      WATERMARK_MEMBERSHIP,
      WATERMARK_EVIDENCE,
    ],
    watermark: {
      attested: {
        p0: 0.25,
        green_count: 1,
        scored_count: 3,
        z_score: 0.3333333333333333,
      },
      recomputed: {
        p0: 0.25,
        green_count: 1,
        scored_count: 3,
        green_rate: 1 / 3,
        z_score: 0.3333333333333333,
      },
      record_consistency: "consistent",
    },
    watermark_display_threshold: 4,
  });
  watermarkTokens(data.frames, false);
  return Object.assign(data, overrides || {});
}

// The viewer as the page creates and boots it. Chart keeps what each
// chart was built with, and each callback adds its name to
// `page.told`, so a test can read what the page was told and when.
function load(storage) {
  const page = loadPage({ scripts: SCRIPTS, storage: storage });
  page.told = [];
  page.shown = null;
  page.entropy = null;
  page.context.Chart = function (ctx, config) {
    const chart = {
      config: config,
      data: config.data,
      options: config.options,
      setActiveElements() {},
      update() {},
      destroy() {},
      resize() {},
    };
    if (ctx.canvas.id === "chart-entropy") {
      page.entropy = chart;
    }
    return chart;
  };
  const tell = (name) => () => {
    page.told.push(name);
  };
  page.viewer = page.context.tokenViewerCreate({
    readTokenizer: () => ({ model_vocab_size: 50 }),
    onShown: (data) => {
      page.told.push("shown");
      page.shown = data;
    },
    onBlendReset: tell("reset"),
    onBlendInput: tell("input"),
    onBlendPress: tell("press"),
    onBlendRelease: tell("release"),
  });
  page.viewer.wire();
  return page;
}

// Every callback, each doing nothing.
function inertCallbacks() {
  const options = {};
  for (const name of CALLBACKS) {
    options[name] = () => {};
  }
  return options;
}

// The spans of the newest render, in the edited layer of a stacked
// run. The stub keeps every render's nodes, since clearing text
// leaves children, so the newest is the last.
function newestSpans(page) {
  const output = page.registry.get("overlay-output");
  const layers = output.children.filter(
    (child) => child.classList.contains("token-layer-edited")
  );
  const holder = layers.length > 0
    ? layers[layers.length - 1]
    : output;
  const last = holder.children[holder.children.length - 1];
  return last && last.tag === null ? last.children : holder.children;
}

function spansWith(page, name) {
  return newestSpans(page).map(
    (span) => span.classList.contains(name)
  );
}

function pickerValues(page) {
  const mount = page.registry.get("overlay-select-mount");
  const select = mount.children[mount.children.length - 1];
  const list = select.children.find((child) => child.tag === "ul");
  return list.children.map((item) => item.getAttribute("data-value"));
}

// The edited layer's bars, copied out of the context.
function bars(page) {
  const sets = page.entropy.data.datasets;
  return JSON.parse(JSON.stringify(sets[sets.length - 1].data));
}

function plugin(chart, id) {
  return chart.config.plugins.find((entry) => entry.id === id);
}

function strip(page) {
  return page.registry.get("token-metrics").overlaysMetricNodes;
}

// -- creating it --

test("the viewer refuses to be created without its callbacks", () => {
  const { context } = loadPage({ scripts: SCRIPTS });
  const refused = (name) => (error) =>
    error.name === "TypeError" && error.message.includes(name);

  assert.throws(
    () => context.tokenViewerCreate(), refused("readTokenizer")
  );
  for (const name of CALLBACKS) {
    const options = inertCallbacks();
    delete options[name];
    assert.throws(
      () => context.tokenViewerCreate(options), refused(name), name
    );
  }
  assert.doesNotThrow(
    () => context.tokenViewerCreate(inertCallbacks())
  );
});

// -- showing a run --

test("a run's frames draw its tokens, overlays and chart", () => {
  const page = load();
  const data = payload();

  page.viewer.show(data);

  assert.deepEqual(
    spansWith(page, "token-mask"), [false, false, false, false]
  );
  assert.deepEqual(
    pickerValues(page),
    ["none", "heatmap", "entropy", "commit", "diff"]
  );
  assert.equal(page.viewer.entropyChart(), page.entropy);
  assert.deepEqual(bars(page), [3, 3.1, 3.2, 3.3]);
  assert.deepEqual(page.told, ["reset", "shown"]);
  assert.equal(page.shown, data);
});

test("a run with nothing to show says so and is not reported", () => {
  const page = load();

  page.viewer.show(payload({ records_available: false }));

  assert.equal(page.registry.get("overlay-empty").hidden, false);
  assert.equal(page.registry.get("overlay-output").hidden, true);
  assert.equal(page.entropy, null);
  assert.equal(page.viewer.entropyChart(), null);
  assert.deepEqual(page.told, ["reset"]);
});

test("a run begun as autoregressive offers no Commit Order", () => {
  const page = load();

  page.viewer.beginRun(true);
  page.viewer.show(payload());

  assert.deepEqual(
    pickerValues(page), ["none", "heatmap", "entropy"]
  );
});

test("saved watermark records offer the overlay and readout", () => {
  const page = load();
  page.viewer.show(watermarked());
  page.viewer.setMode("watermark");

  assert.ok(pickerValues(page).includes("watermark"));
  const spans = newestSpans(page);
  assert.equal(
    spans[0].style.color,
    page.context.OVERLAYS_WATERMARK_FAVORED
  );
  assert.equal(
    spans[0].classList.contains("token-watermark-excluded"),
    true
  );
  const readout = page.registry.get("watermark-readout");
  assert.equal(readout.hidden, false);
  assert.match(
    readout.overlaysWatermarkNodes.status.textContent,
    /insufficient evidence/
  );
  assert.match(
    readout.overlaysWatermarkNodes.status.textContent,
    /record counts consistent/
  );
});

test(
  "legacy membership draws without inventing detector metadata",
  () => {
  const page = load();
  const data = watermarked({
    signals: null,
    watermark: null,
    watermark_display_threshold: null,
  });

  page.viewer.show(data);
  page.viewer.setMode("watermark");

  assert.ok(pickerValues(page).includes("watermark"));
  assert.equal(page.registry.get("watermark-readout").hidden, true);
  }
);

test("watermark crossfade keeps each layer's membership", () => {
  const page = load();
  const data = watermarked({
    original_frames: frames(10),
    remask_edits: [{ frame_index: 1, token_positions: [2] }],
  });
  watermarkTokens(data.original_frames, true);

  page.viewer.show(data);
  page.viewer.setMode("watermark");

  const output = page.registry.get("overlay-output");
  const originals = output.querySelectorAll(
    ".token-layer-original"
  );
  const editedLayers = output.querySelectorAll(
    ".token-layer-edited"
  );
  const original = originals[originals.length - 1];
  const editedLayer = editedLayers[editedLayers.length - 1];
  assert.equal(
    original.children[0].style.color,
    page.context.OVERLAYS_WATERMARK_COMPLEMENT
  );
  assert.equal(
    editedLayer.children[0].style.color,
    page.context.OVERLAYS_WATERMARK_FAVORED
  );
});

// -- scrubbing --

test("the scrubber moves the tokens and the bars together", () => {
  const page = load();
  page.viewer.show(payload());

  page.viewer.setFrame(1);

  assert.deepEqual(
    spansWith(page, "token-mask"), [false, false, true, true]
  );
  assert.deepEqual(bars(page), [1, 1.1, 1.2, 1.3]);

  const slider = page.registry.get("overlay-scrubber-slider");
  slider.value = "2";
  slider.dispatch("input");

  assert.deepEqual(
    spansWith(page, "token-mask"), [false, false, false, true]
  );
  assert.deepEqual(bars(page), [2, 2.1, 2.2, 2.3]);
});

// -- the crossfade --

test("the crossfade is the viewer's and it reports a drag", () => {
  const page = load();
  page.viewer.show(edited());
  page.told = [];
  const slider = page.registry.get("run-blend");

  slider.dispatch("pointerdown");
  slider.value = "40";
  slider.dispatch("input");
  page.fireWindow("pointerup");

  assert.equal(page.viewer.blend(), 0.4);
  assert.deepEqual(page.told, ["press", "input", "release"]);

  const fade = plugin(page.entropy, "compareBlend");
  const ctx = { globalAlpha: 1, save() {}, restore() {} };
  fade.beforeDatasetDraw(
    { data: page.entropy.data, ctx: ctx }, { index: 0 }
  );
  assert.equal(ctx.globalAlpha, 0.6);
});

test("beginning the next run resets the crossfade", () => {
  const page = load();
  page.viewer.show(edited());
  const slider = page.registry.get("run-blend");
  slider.value = "25";
  slider.dispatch("input");
  page.told = [];

  page.viewer.beginRun(false);

  assert.equal(page.viewer.blend(), 1);
  assert.equal(slider.value, "100");
  assert.equal(page.registry.get("run-blend-row").hidden, true);
  assert.equal(page.viewer.entropyChart(), null);
  assert.deepEqual(page.told, ["reset"]);
});

// -- pointing at a position --

test("a token under the pointer reads in the strip", () => {
  const page = load();
  page.viewer.show(payload());
  const output = page.registry.get("overlay-output");

  output.dispatch("mouseover", { target: newestSpans(page)[1] });

  assert.equal(strip(page).position.value.textContent, "2 / 4");
  assert.equal(strip(page).entropy.value.textContent, "3.100");
});

test("a bar under the pointer lights its token", () => {
  const page = load();
  page.viewer.show(payload());

  plugin(page.entropy, "tokenLink").afterEvent({
    getActiveElements: () => [{ index: 2 }],
  });

  assert.deepEqual(
    spansWith(page, "token-cross-highlight"),
    [false, false, true, false]
  );
  assert.equal(strip(page).position.value.textContent, "3 / 4");
});

test("the hover highlight follows the saved setting", () => {
  const page = load({
    diffusion_settings: JSON.stringify({ highlightTokens: true }),
  });
  const output = page.registry.get("overlay-output");

  page.viewer.refreshHoverHighlight();

  assert.equal(
    output.classList.contains("token-hover-highlight"), true
  );
  assert.equal(
    page.registry.get("overlay-highlight-tokens").checked, true
  );
});
