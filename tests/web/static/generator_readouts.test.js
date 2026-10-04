// The generator readout controller, driven without app.js.
//
// Strategy: compose the factory around narrow fakes for generatorRun
// and generatorCanvas, then drive the real profile canvas listeners
// and overlay renderers in the shared DOM stub.
//
// Passing proves DOM and hover state stay private, metric labels keep
// their shipped wording, profile and token hovers cross-highlight,
// Original/Edited ownership follows the blend, and adaptive stopping
// reads only the supplied run and page snapshots.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "overlays.js",
  "generator_readouts.js",
];

function token(text, entropy, masked) {
  return {
    t: text,
    m: masked === true,
    id: text.length,
    c: 0.7,
    e: entropy,
  };
}

function draft(entropy, changed) {
  return [
    token(" one", entropy, changed > 0),
    token(" two", entropy, changed > 1),
  ];
}

function defaultState() {
  const edited = [
    token(" edited", 1.2, false),
    token("\n", 0.4, false),
  ];
  const original = [
    token(" original", 2.2, false),
    token("\n", 0.9, false),
  ];
  return {
    frames: [edited],
    originalFrames: [original],
    canvases: [0],
    parameters: {
      max_denoising_steps: 4,
      confidence_threshold: 0.005,
      stability_threshold: 1,
    },
    model: {
      capabilities: { adaptive_stopping: true },
      parameterDefaults: {},
      vocabSize: 128,
    },
    settings: {
      remaskedPositions: {},
      segmentStarts: [],
    },
    scrubber: {
      active: true,
      frame: 0,
      selectingTarget: false,
    },
    profile: {
      values: [1.2, 0.4],
      original: [2.2, 0.9],
      current: -1,
      filled: -1,
      asOfStep: 3,
      originalAsOfStep: 1,
    },
    entropyDeclared: true,
    entropyAvailable: true,
    blend: 1,
    favorsOriginal: false,
    layered: true,
    edited,
    original,
  };
}

function fakeRun(state) {
  return {
    frameCount() {
      return state.frames.length;
    },
    frameTokens(index) {
      return state.frames[index] || null;
    },
    frameCanvas(index) {
      return state.canvases[index] || 0;
    },
    originalTokenFrames() {
      return state.originalFrames.length;
    },
    originalTokens(index) {
      return state.originalFrames[index] || null;
    },
    parameters() {
      return state.parameters;
    },
  };
}

function fakeCanvas(state) {
  return {
    entropyProfile() {
      return state.profile;
    },
    editedPositionMarks() {
      return { 1: 0 };
    },
    blend() {
      return state.blend;
    },
    blendFavorsOriginal() {
      return state.favorsOriginal;
    },
    entropyDeclared() {
      return state.entropyDeclared;
    },
    entropyAvailable() {
      return state.entropyAvailable;
    },
    layerIsOriginal(target) {
      const layer = target && target.closest
        ? target.closest(".token-layer")
        : null;
      if (layer) {
        return layer.classList.contains(
          "token-layer-original"
        );
      }
      return state.layered && state.favorsOriginal;
    },
    layersActive() {
      return state.layered;
    },
    drawnTokens(original) {
      return original ? state.original : state.edited;
    },
    entropyReading(index, item, original) {
      return {
        value: item.e,
        asOfStep: original ? 1 : 3,
      };
    },
    maskChar() {
      return "?";
    },
    tokenExtra() {
      return "Resolved at step: 2";
    },
  };
}

function completeOptions(state) {
  return {
    run: fakeRun(state),
    canvas: fakeCanvas(state),
    readModel() {
      return state.model;
    },
    readSettings() {
      return state.settings;
    },
    readScrubber() {
      return state.scrubber;
    },
  };
}

function harness(configure) {
  const page = loadPage({ scripts: SCRIPTS });
  const state = defaultState();
  if (configure) {
    configure(state);
  }
  const readouts = page.context.generatorReadoutsCreate(
    completeOptions(state)
  );
  readouts.boot();
  readouts.wire();
  readouts.applyModel();
  return { page, state, readouts };
}

function descendants(node) {
  const found = [];
  const pending = (node.children || []).slice();
  while (pending.length > 0) {
    const next = pending.shift();
    found.push(next);
    pending.push(...(next.children || []));
  }
  return found;
}

function appendLayer(page, className, values) {
  const layer = page.document.createElement("span");
  layer.className = "token-layer " + className;
  values.forEach((value, index) => {
    const span = page.document.createElement("span");
    span.className = "token-span";
    span.textContent = value.t;
    span.setAttribute("data-pos", index);
    layer.appendChild(span);
  });
  page.registry.get("output-area").appendChild(layer);
  return layer;
}

function textOf(element) {
  if (element.children.length === 0) {
    return element.textContent;
  }
  return element.children.map(textOf).join("");
}

test("controllers and page snapshots are required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const state = defaultState();

  assert.throws(
    () => page.context.generatorReadoutsCreate({}),
    /options\.run/
  );
  const withoutCanvas = completeOptions(state);
  delete withoutCanvas.canvas;
  assert.throws(
    () => page.context.generatorReadoutsCreate(withoutCanvas),
    /options\.canvas/
  );
  for (const name of [
    "readModel",
    "readSettings",
    "readScrubber",
  ]) {
    const options = completeOptions(state);
    delete options[name];
    assert.throws(
      () => page.context.generatorReadoutsCreate(options),
      new RegExp(name)
    );
  }
});

test("DOM, hover and profile layout state stay private", () => {
  const { page, readouts } = harness();

  for (const name of [
    "entropyProfileRow",
    "entropyProfileCanvas",
    "entropyProfileReadout",
    "tokenMetricsStrip",
    "stopReadout",
    "entropyHoverPosition",
    "tokenHighlightPosition",
    "metricsHoverPosition",
    "metricsHoverOriginal",
    "metricsCandidate",
    "profileLayers",
    "profileLayout",
  ]) {
    assert.equal(page.context[name], undefined, name);
  }
  assert.equal(readouts.profileShowing(), false);
});

test("boot keeps the metrics titles", () => {
  const { page } = harness();
  const labels = descendants(
    page.registry.get("token-metrics")
  )
    .filter((node) =>
      node.classList.contains("token-metrics-label")
    )
    .map((node) => node.textContent);

  assert.deepEqual(labels, [
    "Position",
    "Confidence",
    "Entropy",
  ]);
});

test("profile hover highlights tokens and follows the blend", () => {
  const { page, state, readouts } = harness();
  const original = appendLayer(
    page, "token-layer-original", state.original
  );
  const edited = appendLayer(
    page, "token-layer-edited", state.edited
  );
  const profile = page.registry.get("entropy-profile");
  profile.clientWidth = 200;
  profile.clientHeight = 34;

  readouts.updateProfile();
  profile.dispatch("mousemove", { clientX: 150 });

  const originalToken = original.children[1];
  const editedToken = edited.children[1];
  assert.equal(
    originalToken.classList.contains(
      "token-cross-highlight"
    ),
    true
  );
  assert.equal(
    editedToken.classList.contains("token-zero-width"),
    true
  );
  assert.equal(
    page.registry.get("entropy-profile-readout").textContent,
    "0.4 nats, as of step 3"
  );
  const nodes = page.registry
    .get("token-metrics")
    .overlaysMetricNodes;
  assert.equal(nodes.position.value.textContent, "2 / 2");
  assert.equal(nodes.entropy.value.textContent, "0.400");
  assert.equal(nodes.run.textContent, "Edited");

  state.blend = 0;
  state.favorsOriginal = true;
  readouts.layerChanged({ profile: true });

  assert.equal(
    page.registry.get("entropy-profile-readout").textContent,
    "0.9 nats, as of step 1"
  );
  assert.equal(nodes.entropy.value.textContent, "0.900");
  assert.equal(nodes.run.textContent, "Original");

  profile.dispatch("mouseleave");
  assert.equal(
    originalToken.classList.contains(
      "token-cross-highlight"
    ),
    false
  );
  assert.equal(
    page.registry
      .get("token-metrics")
      .classList.contains("is-idle"),
    true
  );
});

test("candidate hover renders rank against model width", () => {
  const { page, readouts } = harness();
  readouts.setTokenHover(0, null);

  readouts.setCandidateHover({
    t: " candidate",
    p: 0.125,
    rank: 3,
  });

  const candidate = page.registry
    .get("token-metrics")
    .overlaysMetricNodes.candidate;
  assert.equal(candidate.group.hidden, false);
  assert.equal(candidate.chip.textContent, "\u00b7candidate");
  assert.equal(candidate.value.textContent, "0.125");
  assert.equal(candidate.rank.textContent, "#3 of 128");

  readouts.setCandidateHover(null);
  assert.equal(candidate.group.hidden, true);
});

test("adaptive stopping uses scrubber and segment snapshots", () => {
  const { page, state, readouts } = harness((draftState) => {
    draftState.layered = false;
    draftState.frames = [
      draft(2, 2),
      draft(0.001, 0),
    ];
    draftState.canvases = [0, 0];
    draftState.scrubber.frame = 1;
    draftState.settings.segmentStarts = [1];
  });

  readouts.refreshStop();
  const stop = page.registry.get("stop-readout");
  assert.equal(
    textOf(stop.overlaysStopNodes.text),
    "entropy 0.0010 of 0.005, steady 0 of 1"
  );

  state.settings.segmentStarts = [];
  readouts.refreshStop();
  assert.equal(
    textOf(stop.overlaysStopNodes.text),
    "entropy 0.0010 of 0.005, steady"
  );

  state.model.capabilities.adaptive_stopping = false;
  readouts.refreshStop();
  assert.equal(stop.hidden, true);
});
