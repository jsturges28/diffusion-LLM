// The generator's entropy readings on a diffusion run, driven for
// real.
//
// Strategy: load the generator page into the DOM stub with a model
// entry that declares its signals, start a real run through
// startGeneration, stream frames shaped the way that model streams
// them, finish it, and read what the page would draw: the entropy
// profile's layers and readout, the Entropy overlay's token colors,
// and the metrics strip's reading. DiffusionGemma ends each canvas on
// a commit that carries no entropy, LLaDA carries entropy on every
// frame, and an autoregressive run is the control.
//
// The profile used to be built for autoregressive runs alone: it read
// the final frame and took the scrubbed frame's number for a
// position, so a diffusion run drew one frame's values whatever was
// on screen and faded positions that existed. A finished
// DiffusionGemma run showed none of it, since its final frame is a
// commit. Passing proves a finished DiffusionGemma run is offered its
// entropy views; a diffusion profile reads the frame under the
// scrubber, a commit through its canvas's last draft, with nothing
// faded and no column singled out; the overlay, the strip and the
// readout read a commit the same way and say so; and an
// autoregressive run's profile is what it was.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

function entropyChannel(axes) {
  return {
    name: "entropy",
    unit: "nats",
    axes: axes,
    location: "token_record",
    key: "e",
    capture: "always",
  };
}

function diffusionModel(id) {
  return {
    id: id,
    display_name: id,
    min_vram_gib: 18,
    capabilities: {
      family: "diffusion",
      generation_shape: "iterative_canvas",
      input_mode: "chat",
      supports_resume: true,
      unresolved_char: "\u2591",
      supported_devices: ["cuda"],
      signals: [entropyChannel(["frame", "position"])],
    },
    param_specs: [],
    status: "active",
  };
}

const AUTOREGRESSIVE = {
  id: "smollm3",
  display_name: "SmolLM3-3B",
  min_vram_gib: 6,
  capabilities: {
    family: "autoregressive",
    generation_shape: "append_only",
    input_mode: "chat",
    supports_resume: false,
    supports_substitution: true,
    supported_devices: ["cuda", "cpu"],
    signals: [entropyChannel(["position"])],
  },
  param_specs: [],
  status: "active",
};

function modelsFor(model) {
  return {
    models: [model],
    active: model.id,
    active_device: "cuda",
    active_tokenizer: { name: model.id },
    active_context_length: 4096,
    default: model.id,
    gpu_name: "NVIDIA GeForce RTX 4090",
  };
}

function quietFetch(models) {
  return function (url) {
    const path = String(url).split("?")[0];
    const body = path.startsWith("/api/models") ? models : {};
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  };
}

// A diffusion frame over four positions on `canvas`. A draft carries
// entropy at every position, frame N's being N + 1 plus a tenth of
// the position; a commit carries none.
function diffusionFrame(index, canvas, commit) {
  const tokens = ["a", "b", "c", "d"].map((letter, position) => {
    const token = {
      t: " " + letter + index,
      m: false,
      id: 100 + position,
      c: commit ? 1 : 0.5,
    };
    if (!commit) {
      token.e = index + 1 + position / 10;
    }
    return token;
  });
  return {
    type: "frame",
    index: index,
    total_steps: null,
    canvas_index: canvas,
    mean_conf: 0.5,
    text: tokens.map((token) => token.t).join(""),
    tokens: tokens,
    revealed: [],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// What draft `frame`'s positions carry.
function draftValues(frame) {
  return [0, 1, 2, 3].map((position) => frame + 1 + position / 10);
}

// An autoregressive frame adding position `index - 1`, the way the
// append stream sends one.
function appendFrame(index) {
  const position = index - 1;
  return {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: 4,
    canvas_index: 0,
    mean_conf: 0.5,
    token: {
      t: " w" + position,
      m: false,
      id: 200 + position,
      c: 0.5,
      e: position / 10,
    },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// A finished run of `model` over `frames`, the page as handleDone
// leaves it: the scrubber up, on the final frame.
function finishedRun(model, frames) {
  const models = modelsFor(model);
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch(models),
    bootState: { ui_state: {}, models: models },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  frames.forEach((frame) => {
    context.handleFrame(frame);
  });
  context.handleDone({ type: "done", final_text: "done" });
  return page;
}

// Two drafts and a commit on canvas 0, then a draft and a commit on
// canvas 1: how DiffusionGemma streams a run of two canvases.
function dgemmaRun() {
  return finishedRun(diffusionModel("diffusiongemma"), [
    diffusionFrame(0, 0, false),
    diffusionFrame(1, 0, false),
    diffusionFrame(2, 0, true),
    diffusionFrame(3, 1, false),
    diffusionFrame(4, 1, true),
  ]);
}

// Arrays built inside the vm context are not reference-equal to host
// ones, so deepEqual rejects them on realm rather than on content.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

// The values the overlay picker lists, as a finished run built it.
function pickerValues(context) {
  const list = context.overlaySelect.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map((item) => item.getAttribute("data-value"));
}

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

// -- a finished DiffusionGemma run --

test("a finished DiffusionGemma run is offered its entropy", () => {
  // Its final frame is a commit, which carries none, and the final
  // frame was the only place the page looked.
  const { context } = dgemmaRun();

  assert.equal(context.entropyAvailable(), true);
  assert.ok(pickerValues(context).includes("entropy"));
  assert.equal(context.entropyProfileShowing(), true);
});

test("its profile reads the final commit through its draft", () => {
  const { context } = dgemmaRun();

  const layers = context.entropyProfileLayers();

  assert.deepEqual(host(layers.values), draftValues(3));
  assert.equal(layers.asOfStep, 3);
  assert.equal(layers.current, -1);
  assert.equal(layers.filled, -1);
});

test("a draft reads its own values, unlabeled", () => {
  const { context } = dgemmaRun();
  context.navigateToFrame(1);

  const layers = context.entropyProfileLayers();

  assert.deepEqual(host(layers.values), draftValues(1));
  assert.equal(layers.asOfStep, null);
});

test("the readout says when the profile borrowed", () => {
  const { context } = dgemmaRun();

  context.setEntropyHoverPosition(1);

  assert.equal(
    context.entropyProfileReadout.textContent,
    "4.1 nats, as of step 3"
  );
});

test("the Entropy overlay colors a commit from its draft", () => {
  const { context, registry } = dgemmaRun();
  context.overlayMode = "entropy";
  // The stub keeps children when text is cleared, so the spans the
  // finishing render drew are dropped by hand.
  const output = registry.get("output-area");
  output.children = [];

  context.navigateToFrame(4);

  const colors = drawnSpans(output).map((span) => span.style.color);
  assert.deepEqual(
    colors,
    draftValues(3).map((value) => context.entropyColor(value))
  );
});

test("the metrics strip reads a commit through its draft", () => {
  const { context } = dgemmaRun();
  context.metricsHoverPos = 1;
  context.metricsHoverOriginal = false;

  const reading = context.buildTokenMetricsReading();

  assert.equal(reading.entropy, draftValues(3)[1]);
  assert.match(reading.extra, /entropy as of step 3/);
});

test("an edited run's two layers each borrow their own draft", () => {
  // Edited at frame 1, the way the page takes a resume: the branch
  // ends on a commit of its own, as the run it replaced did, and each
  // layer's commit reads that layer's last draft.
  const { context } = finishedRun(diffusionModel("diffusiongemma"), [
    diffusionFrame(0, 0, false),
    diffusionFrame(1, 0, false),
    diffusionFrame(2, 0, true),
  ]);
  context.remaskEdits = [{ frame_index: 1, token_positions: [2] }];
  context.truncateRunArraysAt(1);
  context.invalidateRunMemos();
  context.isResuming = true;
  context.handleFrame(diffusionFrame(7, 0, false));
  context.handleFrame(diffusionFrame(8, 0, true));
  context.handleDone({ type: "done", final_text: "done" });

  const layers = context.entropyProfileLayers();

  assert.equal(context.runBlendActive(), true);
  assert.deepEqual(host(layers.values), draftValues(7));
  assert.equal(layers.asOfStep, 1);
  assert.deepEqual(host(layers.original), draftValues(1));
  assert.equal(layers.originalAsOfStep, 1);
});

// -- LLaDA, which carries entropy on every frame --

test("a LLaDA profile follows the scrub, nothing faded", () => {
  const { context } = finishedRun(diffusionModel("llada"), [
    diffusionFrame(0, 0, false),
    diffusionFrame(1, 0, false),
    diffusionFrame(2, 0, false),
  ]);

  context.navigateToFrame(0);
  const early = context.entropyProfileLayers();
  context.navigateToFrame(2);
  const late = context.entropyProfileLayers();

  assert.deepEqual(host(early.values), draftValues(0));
  assert.deepEqual(host(late.values), draftValues(2));
  assert.equal(early.current, -1);
  assert.equal(early.filled, -1);
  assert.equal(early.asOfStep, null);
});

// -- the control --

test("an autoregressive run's profile is what it was", () => {
  // Frame N introduced position N, so the scrubbed frame marks its
  // own column and fades the positions still to come.
  const { context } = finishedRun(
    AUTOREGRESSIVE, [1, 2, 3, 4].map(appendFrame)
  );
  context.navigateToFrame(1);

  const layers = context.entropyProfileLayers();

  assert.deepEqual(host(layers.values), [0, 0.1, 0.2, 0.3]);
  assert.equal(layers.current, 1);
  assert.equal(layers.filled, 1);
  assert.equal(layers.asOfStep, null);
});

// -- the row's space --
//
// Held empty until a run fills it, so the canvas above does not
// shrink when a profile first appears. It was held for autoregressive
// models alone, the only ones that recorded entropy when the rule was
// written, so a diffusion run's profile pushed the canvas up as the
// run finished.

function bootedOn(model) {
  const models = modelsFor(model);
  return loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch(models),
    bootState: { ui_state: {}, models: models },
  });
}

function entropyRow(registry) {
  return registry.get("entropy-profile-row");
}

test("a diffusion model holds the row from page load", () => {
  const { registry } = bootedOn(diffusionModel("llada"));
  const row = entropyRow(registry);

  assert.equal(row.hidden, false);
  assert.equal(row.classList.contains("is-empty"), true);
});

test("the row stays held through a diffusion run", () => {
  const { context, registry } = bootedOn(diffusionModel("llada"));
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  context.handleFrame(diffusionFrame(0, 0, false));
  context.handleFrame(diffusionFrame(1, 0, false));

  assert.equal(entropyRow(registry).hidden, false);

  context.handleDone({ type: "done", final_text: "done" });

  assert.equal(entropyRow(registry).hidden, false);
  assert.equal(context.entropyProfileShowing(), true);
});

test("a model declaring no entropy leaves the row out", () => {
  const silent = diffusionModel("silent");
  silent.capabilities.signals = [];
  const { registry } = bootedOn(silent);

  assert.equal(entropyRow(registry).hidden, true);
});

test("an entropy with no per-position bars leaves it out", () => {
  // A canvas-level entropy has nothing for the row to draw.
  const canvasLevel = diffusionModel("canvas-level");
  canvasLevel.capabilities.signals = [entropyChannel(["frame"])];
  const { registry } = bootedOn(canvasLevel);

  assert.equal(entropyRow(registry).hidden, true);
});

test("an autoregressive model holds the row, as before", () => {
  const { registry } = bootedOn(AUTOREGRESSIVE);

  assert.equal(entropyRow(registry).hidden, false);
});
