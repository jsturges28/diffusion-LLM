// The stopping readout on the generator, beside the metrics strip.
//
// Strategy: load the generator page into the DOM stub with a
// DiffusionGemma entry that declares adaptive stopping and the three
// parameters the rule is read from, start a real run through
// startGeneration, and stream frames shaped as its worker sends them:
// every position of a draft carries its entropy and whether it
// changed, and a committed canvas carries no entropy. Then read the
// readout's words after each step. A model that does not stop
// adaptively is the control.
//
// Passing proves the readout follows the newest frame while a run
// streams and the scrubbed frame once it ends, says how a committed
// canvas ended, reads the rule from the run's own parameters, never
// counts a resume's first frame as steady, follows the crossfade to
// the original run, and stays off a model with no such rule.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const RULE_SPECS = [
  {
    name: "max_denoising_steps", label: "Denoising Steps",
    type: "int", default: 48, step: 1,
    recommended: [4, 64], experimental: [1, 256],
  },
  {
    name: "confidence_threshold", label: "Stop Entropy",
    type: "float", default: 0.005, step: 0.001,
    recommended: [0.001, 0.05], experimental: [0.0001, 1.0],
  },
  {
    name: "stability_threshold", label: "Steady Steps",
    type: "int", default: 1, step: 1,
    recommended: [0, 4], experimental: [0, 16],
  },
];

function modelsWith(adaptive) {
  const entry = {
    id: "dgemma",
    display_name: "DiffusionGemma-26B-A4B",
    min_vram_gib: 18,
    capabilities: {
      family: "diffusion",
      generation_shape: "iterative_canvas",
      input_mode: "chat",
      supports_resume: true,
      adaptive_stopping: adaptive,
      unresolved_char: "\u2591",
      supported_devices: ["cuda"],
    },
    param_specs: RULE_SPECS,
    status: "active",
  };
  return {
    models: [entry],
    active: "dgemma",
    active_device: "cuda",
    active_tokenizer: { name: "dgemma" },
    active_context_length: 4096,
    default: "dgemma",
    gpu_name: "NVIDIA GeForce RTX 4090",
  };
}

// A four-position draft at mean entropy `entropy`, of which the
// first `changed` moved since the last draft.
function draft(entropy, changed) {
  const tokens = [];
  for (let i = 0; i < 4; i++) {
    tokens.push({
      t: " w" + i, m: i < changed, id: 10 + i, c: 0.5, e: entropy,
    });
  }
  return tokens;
}

// A committed canvas: no entropy, nothing changing.
function commit() {
  const tokens = [];
  for (let i = 0; i < 4; i++) {
    tokens.push({ t: " w" + i, m: false, id: 10 + i, c: 1 });
  }
  return tokens;
}

function frameOf(tokens, index, canvas) {
  return {
    type: "frame",
    index: index,
    total_steps: null,
    canvas_index: canvas || 0,
    mean_conf: 0.5,
    text: tokens.map((token) => token.t).join(""),
    tokens: tokens,
    revealed: [],
    elapsed: +(index * 0.9).toFixed(2),
  };
}

function fetchFor(models) {
  return (url) => {
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

// A page mid-run: the parameter inputs set from `params`, started for
// real, then `frames` streamed in order.
function streaming(frames, options) {
  const settings = options || {};
  const models = modelsWith(settings.adaptive !== false);
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: fetchFor(models),
    bootState: { ui_state: {}, models: models },
  });
  const { context, registry } = page;
  page.generatorSocketController().connect();
  registry.get("prompt-input").value = "explain yeast";
  for (const [name, value] of Object.entries(settings.params || {})) {
    registry
      .get("param-fields")
      .querySelector("#param-" + name)
      .value = String(value);
  }
  context.startGeneration();
  frames.forEach((frame, index) => {
    context.handleFrame(frame.type ? frame : frameOf(frame, index));
  });
  return page;
}

function textOf(element) {
  if (element.children.length === 0) {
    return element.textContent;
  }
  return element.children.map(textOf).join("");
}

// The readout's words, or null while it is hidden.
function words(page) {
  const readout = page.registry.get("stop-readout");
  if (readout.hidden) {
    return null;
  }
  return textOf(readout.overlaysStopNodes.text);
}

// Branch the run at `frameIndex` the way an edit does, as the live
// cycling tests do: the page cuts back to that frame, and the
// resume's first frame takes its place. The scrubber gives way to
// the live view, as it does when the real resume starts generating.
function resumeAt(context, frameIndex) {
  context.handleDone({ type: "done", final_text: "done" });
  context.remaskEdits = [
    { frame_index: frameIndex, token_positions: [1] },
  ];
  context.truncateRunArraysAt(frameIndex);
  context.generatorCanvas.invalidate();
  context.isResuming = true;
  context.setGenerating(true);
}

// -- while a run streams --

test("the readout follows a canvas as it streams", () => {
  const page = streaming([draft(2.31, 4)]);
  assert.equal(words(page), "entropy 2.31 of 0.005, 4 changing");

  page.context.handleFrame(frameOf(draft(0.041, 2), 1));
  assert.equal(words(page), "entropy 0.041 of 0.005, 2 changing");

  page.context.handleFrame(frameOf(draft(0.0047, 0), 2));
  assert.equal(words(page), "entropy 0.0047 of 0.005, steady");
});

test("a committed canvas says how it ended as it lands", () => {
  const page = streaming([
    draft(2, 4), draft(0.01, 1), draft(0.002, 0), commit(),
  ]);

  assert.equal(words(page), "Canvas 1 stopped after 3 steps");
});

test("the next canvas starts its own reading", () => {
  const page = streaming([draft(2, 4), draft(0.002, 0), commit()]);

  page.context.handleFrame(frameOf(draft(3.1, 4), 3, 1));

  assert.equal(words(page), "entropy 3.10 of 0.005, 4 changing");
});

// -- once it ends --

test("scrubbing reads the frame on screen", () => {
  const page = streaming([
    draft(2, 4), draft(0.01, 1), draft(0.002, 0), commit(),
  ]);
  page.context.handleFrame(frameOf(draft(3.1, 4), 4, 1));
  page.context.handleDone({ type: "done", final_text: "done" });

  page.context.navigateToFrame(3);
  assert.equal(words(page), "Canvas 1 stopped after 3 steps");

  page.context.navigateToFrame(1);
  assert.equal(words(page), "entropy 0.010 of 0.005, 1 changing");
});

test("a canvas that ran out of steps says so", () => {
  const page = streaming(
    [draft(2, 4), draft(1, 2), draft(0.5, 1), commit()],
    { params: { max_denoising_steps: 3 } }
  );

  assert.equal(words(page), "Canvas 1 used all 3 steps");
});

// -- the rule --

test("the rule is the run's own parameters", () => {
  const page = streaming([draft(2, 4), draft(0.02, 0)], {
    params: { confidence_threshold: 0.03, stability_threshold: 2 },
  });
  assert.equal(
    words(page), "entropy 0.020 of 0.03, steady 1 of 2"
  );

  page.context.handleFrame(frameOf(draft(0.02, 0), 2));

  assert.equal(words(page), "entropy 0.020 of 0.03, steady");
});

test("a run without the rule shows no readout", () => {
  const page = streaming([draft(2, 4), draft(0.002, 0)], {
    adaptive: false,
  });

  assert.equal(words(page), null);
});

test("a cleared run takes the readout with it", () => {
  const page = streaming([draft(2, 4)]);

  page.context.resetRunState();

  assert.equal(words(page), null);
});

// -- an edit --

test("a resume's first frame is never steady", () => {
  const page = streaming([draft(2, 4), draft(0.5, 1), draft(0.2, 0)]);
  resumeAt(page.context, 2);

  page.context.handleFrame(frameOf(draft(0.1, 0), 0));
  assert.equal(words(page), "entropy 0.10 of 0.005, steady 0 of 1");

  page.context.handleFrame(frameOf(draft(0.05, 0), 1));
  assert.equal(words(page), "entropy 0.050 of 0.005, steady");
});

test("the crossfade moves the readout to the original run", () => {
  const page = streaming([
    draft(2, 4), draft(0.5, 1), draft(0.001, 0), commit(),
  ]);
  resumeAt(page.context, 1);
  page.context.handleFrame(frameOf(draft(1.5, 2), 0));
  page.context.handleFrame(frameOf(draft(0.3, 1), 1));
  page.context.handleDone({ type: "done", final_text: "done" });
  assert.equal(words(page), "entropy 0.30 of 0.005, 1 changing");

  page.registry.get("run-blend").value = "0";
  page.registry.get("run-blend").dispatch("input");

  assert.equal(words(page), "entropy 0.0010 of 0.005, steady");
});
