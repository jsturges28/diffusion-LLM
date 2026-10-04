// The Forgetting overlay: offered where it was declared, drawn from
// each token's own value, and carried into the save.
//
// Strategy: load the generator and the Analytics page into the DOM
// stub beside this file, and overlays.js on its own for the ramp,
// then drive them with runs shaped like Mamba-3's: append frames whose
// tokens carry `f`, from a model whose capabilities declare one
// forgetting value per position. The same run from a model that
// declares nothing is the control, since the option must follow the
// declaration and not merely the presence of a key.
//
// Passing proves the option appears only where a model declared the
// channel and the run carries it, that a token is coloured by its own
// value, that the metrics strip reads that value while the overlay is
// on, that a save keeps `f` on every record, and that Analytics offers
// the same overlay for a saved run.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { loadPage, ANALYTICS_SCRIPTS } = require("./dom_stub.js");

const OVERLAYS = path.join(
  __dirname, "..", "..", "..", "src", "web", "static", "overlays.js"
);

const FORGETTING = {
  name: "forgetting",
  unit: "fraction",
  axes: ["position"],
  location: "token_record",
  key: "f",
  capture: "always",
};

const WORDS = ["The", " river", " ran", " dry"];
const VALUES = [0.12, 0.31, 0.18, 0.25];

// Arrays built inside the vm context are not reference-equal to host
// ones, so deepEqual rejects them on realm rather than on content.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function stateSpaceModel(signals) {
  return {
    id: "mamba3",
    capabilities: {
      family: "state_space",
      generation_shape: "append_only",
      input_mode: "completion",
      signals: signals,
    },
  };
}

// One append frame as `_build_append_frame` in ar_sampler.py emits it
// for a model that reports a value on reading each token.
function appendFrame(index, withForgetting) {
  const position = index - 1;
  const token = {
    t: WORDS[position],
    m: false,
    id: 1000 + position,
    c: 0.5,
    e: 1.2,
  };
  if (withForgetting) {
    token.f = VALUES[position];
  }
  return {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: WORDS.length,
    canvas_index: 0,
    mean_conf: 0.5,
    token: token,
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// The generator after a whole run from a model declaring `signals`.
function generator(signals, withForgetting) {
  const { context } = loadPage({});
  context.activeModel = stateSpaceModel(signals);
  for (let index = 1; index <= WORDS.length; index++) {
    context.handleFrame(appendFrame(index, withForgetting));
  }
  return context;
}

// The values the real overlay picker lists, built the way a finished
// run builds it.
function pickerValues(context) {
  context.buildOverlaySelect();
  const list = context.overlaySelect.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map((item) => item.getAttribute("data-value"));
}

// -- the generator: when the option appears --

test("a model that declares forgetting offers it", () => {
  const context = generator([FORGETTING], true);

  assert.equal(context.forgettingAvailable(), true);
  assert.ok(pickerValues(context).includes("forgetting"));
});

test("a model that declares nothing never offers it", () => {
  // The control. The same tokens carry `f`, but no model promised
  // the key, so nothing says what the number means.
  const context = generator([], true);

  assert.equal(context.forgettingAvailable(), false);
  assert.ok(!pickerValues(context).includes("forgetting"));
});

test("a declaration with nothing recorded is not offered", () => {
  const context = generator([FORGETTING], false);

  assert.equal(context.forgettingAvailable(), false);
});

test("a declaration of another shape is refused", () => {
  // One value per position is the only shape the ramp can colour; a
  // per-frame forgetting would need a view this build does not have.
  const shaped = Object.assign({}, FORGETTING, {
    axes: ["frame", "position"],
  });
  const context = generator([shaped], true);

  assert.equal(context.forgettingAvailable(), false);
});

// -- the generator: what it draws, reads and saves --

test("each token is coloured by its own value", () => {
  const context = generator([FORGETTING], true);
  context.overlayMode = "forgetting";
  const token = context.runFrames.positions[1];

  assert.equal(
    context.tokenColorAt(1, token, false),
    context.forgettingColor(VALUES[1])
  );
  assert.equal(
    context.tokenColorAt(0, { t: "x", m: false, id: 1 }, false),
    null
  );
});

test("the strip reads the value while the overlay is on", () => {
  const context = generator([FORGETTING], true);
  const token = context.runFrames.positions[1];

  context.overlayMode = "forgetting";
  assert.equal(context.metricsExtra(1, token), "Forgetting: 0.310");
  context.overlayMode = "entropy";
  assert.equal(context.metricsExtra(1, token), "");
});

test("a save keeps every token's value", () => {
  const context = generator([FORGETTING], true);

  const records = host(
    context.positionRecordsFrom(context.runFrames.positions)
  );

  assert.deepEqual(records.map((record) => record.f), VALUES);
});

test("a run without the value saves no key for it", () => {
  const context = generator([], false);

  const records = host(
    context.positionRecordsFrom(context.runFrames.positions)
  );

  for (const record of records) {
    assert.ok(!("f" in record));
  }
});

test("per-frame records keep the value too", () => {
  const { context } = loadPage({});
  const frames = [
    [{ t: "a", m: false, id: 1, c: 0.5, f: 0.2 }],
    null,
  ];

  const records = host(context.tokenRecordsFrom(frames));

  assert.equal(records[0][0].f, 0.2);
  assert.equal(records[1], null);
});

// -- the ramp and the reading --

function overlays() {
  const sandbox = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(OVERLAYS, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return sandbox;
}

function lightness(color) {
  return Number(/hsl\(\d+, \d+%, (\d+)%\)/.exec(color)[1]);
}

function hue(color) {
  return Number(/hsl\((\d+),/.exec(color)[1]);
}

// WCAG contrast of an hsl() colour against the output area's #111.
function contrastOnCanvas(color) {
  const [h, s, l] = /hsl\((\d+), (\d+)%, (\d+)%\)/
    .exec(color).slice(1).map(Number);
  const chroma = (1 - Math.abs(2 * (l / 100) - 1)) * (s / 100);
  const second = chroma * (1 - Math.abs(((h / 60) % 2) - 1));
  const base = l / 100 - chroma / 2;
  // Every hue this ramp produces lies between 240 and 300.
  assert.ok(h >= 240 && h < 300, `hue ${h} outside the violet band`);
  const channels = [second + base, base, chroma + base];
  const linear = channels.map((v) =>
    v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  );
  const luminance =
    0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
  const canvas = ((0x11 / 255 + 0.055) / 1.055) ** 2.4;
  return (luminance + 0.05) / (canvas + 0.05);
}

test("the window sits inside a fraction's range, in order", () => {
  const ramp = overlays();

  assert.ok(ramp.OVERLAYS_FORGETTING_FLOOR > 0);
  assert.ok(
    ramp.OVERLAYS_FORGETTING_FLOOR < ramp.OVERLAYS_FORGETTING_CEILING
  );
  assert.ok(ramp.OVERLAYS_FORGETTING_CEILING < 1);
});

test("more erased reads brighter across the window", () => {
  const ramp = overlays();
  const floor = ramp.OVERLAYS_FORGETTING_FLOOR;
  const ceiling = ramp.OVERLAYS_FORGETTING_CEILING;
  const steps = [0, 0.25, 0.5, 0.75, 1].map(
    (at) => floor + (ceiling - floor) * at
  );

  const lights = steps.map((f) => lightness(ramp.forgettingColor(f)));

  for (let i = 1; i < lights.length; i++) {
    assert.ok(lights[i] > lights[i - 1], `step ${i}: ${lights}`);
  }
});

test("the tails clamp to the ends of the window", () => {
  // Digits fall below the window and line breaks above it; each reads
  // at the nearer end rather than off the ramp.
  const ramp = overlays();
  const floor = ramp.OVERLAYS_FORGETTING_FLOOR;
  const ceiling = ramp.OVERLAYS_FORGETTING_CEILING;

  assert.equal(ramp.forgettingColor(0.02), ramp.forgettingColor(floor));
  assert.equal(ramp.forgettingColor(0.9), ramp.forgettingColor(ceiling));
});

test("the middle of real text is told apart", () => {
  // The defect this pins. On two saved runs, one per device, the
  // middle half of tokens sat between about 0.14 and 0.25. The first
  // ramp ran from 0 to 0.4 and put those six points of lightness
  // apart, which read as a single colour. Twenty points is the least
  // that separates them at a glance.
  const ramp = overlays();

  const low = lightness(ramp.forgettingColor(0.14));
  const high = lightness(ramp.forgettingColor(0.25));

  assert.ok(high - low >= 20, `${low}% to ${high}%`);
});

test("the dimmest token stays legible on the canvas", () => {
  // The other side of that trade. Contrast bought by darkening the
  // dim end would cost the words themselves; 3:1 is where the
  // heatmap's dimmest green already sits.
  const ramp = overlays();

  assert.ok(contrastOnCanvas(ramp.forgettingColor(0)) >= 3);
});

test("a missing value reads as nothing erased", () => {
  const ramp = overlays();

  assert.equal(ramp.forgettingColor(undefined), ramp.forgettingColor(0));
  assert.equal(ramp.forgettingColor(NaN), ramp.forgettingColor(0));
});

test("the ramp keeps clear of the other overlays' hues", () => {
  // Entropy runs 45 to 205, the heatmap sits at 135 and the diff at
  // 320. The ramp turns within the violet band between them, so no
  // point on it can be read as any of the three.
  const ramp = overlays();

  for (const f of [0, 0.12, 0.18, 0.24, 0.3, 1]) {
    const at = hue(ramp.forgettingColor(f));
    assert.ok(at >= 205 + 40, `${f}: ${at}`);
    assert.ok(at <= 320 - 30, `${f}: ${at}`);
  }
});

test("the reading has three places, and is blank without one", () => {
  const ramp = overlays();

  assert.equal(
    ramp.overlaysForgettingReading({ f: 0.21449 }),
    "Forgetting: 0.214"
  );
  assert.equal(ramp.overlaysForgettingReading({ t: "a" }), "");
  assert.equal(ramp.overlaysForgettingReading({ f: NaN }), "");
  assert.equal(ramp.overlaysForgettingReading(null), "");
});

// -- Analytics, for a saved run --

function bootFetch() {
  return function (url) {
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : {};
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(""),
    });
  };
}

function analytics() {
  return loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  }).context;
}

// An Analytics page that has opened `run` the way a run's frames
// landing does.
function analyticsOpened(run) {
  const context = analytics();
  run.records_available = true;
  context.renderRunOverlays(run);
  return context;
}

// The spans the newest render drew. The stub keeps each document
// fragment as a node, and setting textContent leaves a node's
// children in place where a browser removes them, so every render's
// fragment stays behind and the newest is the last.
function newestSpans(context) {
  const output = context.document.getElementById("overlay-output");
  const last = output.children[output.children.length - 1];
  return last && last.tag === null ? last.children : output.children;
}

// A saved append run, as the flat positions Analytics reads it in.
function savedRun(signals, withForgetting) {
  const positions = WORDS.map((word, at) => {
    const token = { t: word, m: false, id: 1000 + at, c: 0.5 };
    if (withForgetting) {
      token.f = VALUES[at];
    }
    return token;
  });
  const record = { run_id: "r1", positions: positions };
  if (signals !== undefined) {
    record.signals = signals;
  }
  return record;
}

test("Analytics offers it for a run that declared it", () => {
  const context = analytics();

  assert.equal(
    context.overlaySeriesCarriesForgetting(
      savedRun([FORGETTING], true)
    ),
    true
  );
});

test("Analytics offers nothing for a run without the value", () => {
  const context = analytics();

  assert.equal(
    context.overlaySeriesCarriesForgetting(
      savedRun([FORGETTING], false)
    ),
    false
  );
});

test("Analytics refuses a declaration it cannot draw", () => {
  const context = analytics();
  const canvas = Object.assign({}, FORGETTING, { axes: ["canvas"] });

  assert.equal(
    context.overlaySeriesCarriesForgetting(savedRun([canvas], true)),
    false
  );
});

test("Analytics reads a run saved without a manifest by its data", () => {
  // A run whose save carried no provenance still has its values, and
  // they mean the same thing; hiding them would lose a real reading.
  const context = analytics();

  assert.equal(
    context.overlaySeriesCarriesForgetting(savedRun(undefined, true)),
    true
  );
});

test("Analytics reads the value in the strip", () => {
  const context = analyticsOpened(savedRun([FORGETTING], true));
  context.setOverlayMode("forgetting");
  const output = context.document.getElementById("overlay-output");

  output.dispatch("mouseover", { target: newestSpans(context)[0] });

  const strip = context.document.getElementById("token-metrics");
  assert.equal(
    strip.overlaysMetricNodes.extra.textContent, "Forgetting: 0.120"
  );
});

test("Analytics lists the option in its picker", () => {
  const context = analyticsOpened(savedRun([FORGETTING], true));

  const mount = context.document.getElementById(
    "overlay-select-mount"
  );
  const select = mount.children[mount.children.length - 1];
  const list = select.children.find((child) => child.tag === "ul");
  const values = list.children.map(
    (item) => item.getAttribute("data-value")
  );
  assert.ok(values.includes("forgetting"));
});

test("Analytics colours each token by its own value", () => {
  const context = analyticsOpened(savedRun([FORGETTING], true));

  context.setOverlayMode("forgetting");

  const colors = newestSpans(context).map((span) => span.style.color);
  assert.deepEqual(
    colors, VALUES.map((value) => context.forgettingColor(value))
  );
});
