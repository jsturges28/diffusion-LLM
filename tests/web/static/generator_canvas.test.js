// The generator canvas controller, driven without app.js.
//
// Strategy: compose the factory around a mutable fake of
// generatorRun's read API, then render real token spans through the
// shipped overlay builders. The fake counts expensive reads so cache
// invalidation can be observed without exposing a memo.
//
// Passing proves rendering and overlay state live in the controller's
// closure, invalidation refreshes derived run data, the picker
// follows run/model availability, both frame shapes render, and
// comparison controls preserve their two-layer semantics.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "custom_select.js",
  "reduced_motion.js",
  "persist.js",
  "overlays.js",
  "activation_progress.js",
  "generator_chrome.js",
  "generator_canvas.js",
];

function token(id, options) {
  return Object.assign({
    t: " w" + id,
    m: false,
    id: id,
    c: 0.5,
  }, options || {});
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function frameTokens(state, index) {
  if (state.append) {
    return state.positions.slice(0, index + 1);
  }
  return state.frames[index] || null;
}

function originalTokens(state, index) {
  if (state.originalAppend) {
    return state.originalPositions.slice(0, index + 1);
  }
  return state.originalFrames[index] || null;
}

function fakeRun(state) {
  return {
    frameCount() {
      return state.append
        ? state.positions.length
        : state.frames.length;
    },
    frameIsAppend() {
      return state.append;
    },
    frameTokens(index) {
      state.frameReads += 1;
      return frameTokens(state, index);
    },
    frameTokensLast() {
      state.lastReads += 1;
      return frameTokens(
        state,
        this.frameCount() - 1
      );
    },
    frameText(index) {
      const tokens = frameTokens(state, index) || [];
      return tokens.map((item) => item.t).join("");
    },
    frameTokenSeries() {
      state.seriesReads += 1;
      return state.append ? [] : state.frames.slice();
    },
    framePositions() {
      return state.positions.slice();
    },
    frameCanvas() {
      return 0;
    },
    originalCaptured() {
      return state.originalCaptured;
    },
    originalTokenFrames() {
      return state.originalAppend
        ? state.originalPositions.length
        : state.originalFrames.length;
    },
    originalIsAppend() {
      return state.originalAppend;
    },
    originalTokens(index) {
      state.originalReads += 1;
      return originalTokens(state, index);
    },
    originalTokensLast() {
      state.originalLastReads += 1;
      return originalTokens(
        state,
        this.originalTokenFrames() - 1
      );
    },
    originalText(index) {
      const tokens = originalTokens(state, index) || [];
      return tokens.map((item) => item.t).join("");
    },
    originalTokenSeries() {
      return state.originalAppend
        ? []
        : state.originalFrames.slice();
    },
    originalPositions() {
      return state.originalPositions.slice();
    },
    positionAlternatives() {
      return null;
    },
    candidateSets() {
      return null;
    },
  };
}

function defaultState() {
  return {
    frames: [
      [
        token(1, { m: true, e: 2.0 }),
        token(2, { m: true, e: 3.0 }),
      ],
      [
        token(10, { e: 1.0 }),
        token(2, { m: true, e: 2.5 }),
      ],
      [
        token(20, { e: 0.8 }),
        token(11, { e: 1.2 }),
      ],
    ],
    positions: [],
    append: false,
    originalFrames: [],
    originalPositions: [],
    originalAppend: false,
    originalCaptured: false,
    capabilities: {
      family: "diffusion",
      generation_shape: "iterative_canvas",
      signals: [{
        name: "entropy",
        axes: ["frame", "position"],
      }],
    },
    settings: {
      highlightTokens: true,
      unsettledShows: "guess",
      tokenBirthGlow: true,
      revisionGlow: true,
      glowBrightnessDiffusion: 1,
      glowFadeMsDiffusion: 500,
    },
    edit: {
      remaskEdits: [],
      remaskedPositions: {},
      mode: null,
      substituting: false,
      generating: false,
    },
    frameReads: 0,
    lastReads: 0,
    originalReads: 0,
    originalLastReads: 0,
    seriesReads: 0,
    renders: 0,
    resets: 0,
    overlays: [],
    layers: [],
  };
}

function completeOptions(state) {
  return {
    run: fakeRun(state),
    readModel() {
      return {
        capabilities: state.capabilities,
        maskChar: "?",
      };
    },
    readSettings() {
      return state.settings;
    },
    readEdit() {
      return state.edit;
    },
    readReducedMotion() {
      return false;
    },
    writeHighlight(value) {
      state.settings.highlightTokens = value;
    },
    startCandidates() {},
    stopCandidates() {},
    onOutputReset() {
      state.resets += 1;
    },
    onRender() {
      state.renders += 1;
    },
    onOverlayChanged(mode) {
      state.overlays.push(mode);
    },
    onLayerChanged(change) {
      state.layers.push(host(change));
    },
  };
}

function harness(configure) {
  const page = loadPage({ scripts: SCRIPTS });
  const state = defaultState();
  if (configure) {
    configure(state);
  }
  const canvas = page.context.generatorCanvasCreate(
    completeOptions(state)
  );
  canvas.wire();
  return { page, state, canvas };
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

function withClass(node, name) {
  return descendants(node).filter(
    (child) => child.classList.contains(name)
  );
}

function pickerValues(page) {
  const mount = page.registry.get("overlay-select-mount");
  const select = mount.children[mount.children.length - 1];
  const list = select.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map(
    (item) => item.getAttribute("data-value")
  );
}

test("every page callback and run reader are required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const state = defaultState();
  const names = Object.keys(completeOptions(state)).filter(
    (name) => name !== "run"
  );

  assert.throws(
    () => page.context.generatorCanvasCreate({}),
    /options\.run/
  );
  for (const name of names) {
    const options = completeOptions(state);
    delete options[name];
    assert.throws(
      () => page.context.generatorCanvasCreate(options),
      new RegExp(name)
    );
  }
});

test("DOM references and rendering state stay private", () => {
  const { page, canvas } = harness();

  for (const name of [
    "outputArea",
    "overlayMode",
    "commitSteps",
    "diffData",
    "runRevisions",
    "entropyBorrowSlots",
    "editedMarksCache",
    "diffOriginalOpacity",
    "runBlend",
    "liveTokenOptions",
    "liveTokenSpans",
    "tokenGlowQueue",
    "liveRevisionFold",
  ]) {
    assert.equal(page.context[name], undefined, name);
  }
  assert.equal(canvas.overlayMode(), "none");
});

test("invalidation refreshes each derived run reading", () => {
  const { state, canvas } = harness((draft) => {
    draft.originalCaptured = true;
    draft.originalFrames = [[token(1)], [token(10)]];
    draft.frames[2] = [token(20), token(11)];
    draft.edit.remaskEdits = [{
      frame_index: 1,
      token_positions: [0],
    }];
  });

  canvas.setOverlayMode("commit");
  canvas.renderFrame(2);
  const seriesReads = state.seriesReads;
  canvas.tokenExtra(0, state.frames[2][0], false);
  assert.equal(state.seriesReads, seriesReads);

  canvas.revisionsAvailable();
  const revisionReads = state.seriesReads;
  canvas.revisionsAvailable();
  assert.equal(state.seriesReads, revisionReads);

  canvas.diff();
  const finalReads = state.lastReads;
  const originalReads = state.originalLastReads;
  canvas.diff();
  assert.equal(state.lastReads, finalReads);
  assert.equal(state.originalLastReads, originalReads);

  const entropy = canvas.entropyReading(
    0, token(20), false
  );
  assert.equal(entropy.value, 1);
  assert.equal(entropy.asOfStep, 1);
  const frameReads = state.frameReads;
  canvas.entropyReading(0, token(20), false);
  assert.equal(state.frameReads, frameReads);

  canvas.invalidate();
  canvas.tokenExtra(0, state.frames[2][0], false);
  const commitReadsAfter = state.seriesReads;
  canvas.revisionsAvailable();
  canvas.diff();
  canvas.entropyReading(0, token(20), false);
  assert.ok(state.seriesReads > seriesReads);
  assert.ok(state.seriesReads > commitReadsAfter);
  assert.ok(state.lastReads > finalReads);
  assert.ok(state.originalLastReads > originalReads);
  assert.ok(state.frameReads > frameReads);
});

test("the picker follows availability and owns its mode", () => {
  const { page, state, canvas } = harness();

  canvas.activate();
  assert.deepEqual(pickerValues(page), [
    "none",
    "conf",
    "entropy",
    "commit",
    "revisions",
    "diff",
  ]);

  canvas.setOverlayMode("revisions");
  assert.equal(canvas.overlayMode(), "revisions");
  assert.equal(
    page.registry.get("revision-legend").hidden, false
  );
  assert.equal(page.registry.get("commit-legend").hidden, true);

  state.append = true;
  state.positions = state.frames[state.frames.length - 1];
  state.capabilities.generation_shape = "append_only";
  canvas.rebuildOverlaySelect();
  assert.equal(canvas.overlayMode(), "none");
  assert.ok(!pickerValues(page).includes("commit"));
  assert.ok(!pickerValues(page).includes("revisions"));
});

test("snapshot and append frames render through one API", () => {
  const snapshot = harness();
  const output = snapshot.page.registry.get("output-area");
  snapshot.state.frames.length = 2;
  snapshot.canvas.renderLiveFrame(
    snapshot.state.frames[1], [0], null
  );
  let spans = withClass(output, "token-span");
  assert.equal(spans.length, 2);
  assert.equal(spans[0].textContent, " w10");
  assert.equal(spans[0].hasAttribute("data-born"), true);
  assert.equal(output.classList.contains("live-tokens"), true);

  const append = harness((state) => {
    state.append = true;
    state.frames = [];
    state.positions = [token(1), token(2), token(3)];
    state.capabilities.generation_shape = "append_only";
  });
  const appendOutput =
    append.page.registry.get("output-area");
  append.canvas.renderFrame(1);
  spans = withClass(appendOutput, "token-span");
  assert.deepEqual(
    spans.map((span) => span.textContent),
    [" w1", " w2"]
  );
});

test("chrome placeholders can take and return the output area", () => {
  const { page, state, canvas } = harness();
  const chrome = page.context.generatorChromeCreate({
    onTpsToggle() {},
    readReducedMotion() {
      return false;
    },
    readDiffusionEffect() {
      return false;
    },
    readDiffusionTextMode() {
      return "once";
    },
    revealText(element, text, onDone) {
      element.textContent = text;
      if (onDone) {
        onDone();
      }
    },
    cancelReveal() {},
  });
  const output = page.registry.get("output-area");

  canvas.renderLiveFrame(state.frames[2], [], null);
  canvas.reset();
  chrome.showOutputPlaceholder("Test Model");
  const placeholder =
    output.children[output.children.length - 1];
  assert.equal(placeholder.id, "output-placeholder");
  assert.equal(
    placeholder.textContent,
    "Test Model output will appear here..."
  );

  state.frames.length = 1;
  state.settings.unsettledShows = "glyph";
  canvas.renderLiveFrame(state.frames[0], [], null);
  const spans = withClass(output, "token-span");
  assert.deepEqual(
    spans.slice(-2).map((span) => span.textContent),
    ["?", "?"]
  );
});

test("crossfade and Diff keep separate layer controls", () => {
  const { page, state, canvas } = harness((draft) => {
    draft.originalCaptured = true;
    draft.originalFrames = [
      [token(1), token(2)],
      [token(10), token(11)],
    ];
    draft.frames = [
      [token(1), token(2)],
      [token(20), token(11)],
    ];
    draft.edit.remaskEdits = [{
      frame_index: 1,
      token_positions: [0],
    }];
  });
  const output = page.registry.get("output-area");

  canvas.activate();
  canvas.renderFrame(1);
  let original = withClass(
    output, "token-layer-original"
  ).slice(-1)[0];
  let edited = withClass(
    output, "token-layer-edited"
  ).slice(-1)[0];
  assert.equal(original.style.opacity, "0");
  assert.equal(edited.style.opacity, "1");

  canvas.setBlend(0.25);
  assert.equal(original.style.opacity, "0.75");
  assert.equal(edited.style.opacity, "0.25");
  assert.equal(original.style.pointerEvents, "auto");
  assert.equal(edited.style.pointerEvents, "none");

  output.children = [];
  canvas.setOverlayMode("diff");
  original = withClass(
    output, "token-layer-original"
  ).slice(-1)[0];
  edited = withClass(
    output, "token-layer-edited"
  ).slice(-1)[0];
  assert.equal(
    page.registry.get("diff-overlay-controls").hidden,
    false
  );
  assert.equal(page.registry.get("run-blend-row").hidden, true);
  assert.equal(original.style.opacity, "0.5");
  assert.equal(edited.style.opacity, "1");
  assert.equal(state.overlays[state.overlays.length - 1], "diff");
});
