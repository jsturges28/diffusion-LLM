// The Edit Frames gate on a DiffusionGemma run, driven for real.
//
// Strategy: load the generator page into the DOM stub with a
// DiffusionGemma entry, start a real run through startGeneration,
// stream frames shaped the way DiffusionGemma sends them, and finish
// the run with its terminal frame. The shipped markup starts the
// button hidden and the stub does not read markup, so each run hides
// it first, as the page loads it. A single-canvas run is the
// control; the cases under test have frames on a second canvas.
//
// DiffusionGemma resumes by re-entering one 256-token canvas, and its
// worker refuses to resume a run that spans more than one
// (tests/backends/test_dgemma_resume_state.py). Passing proves the
// page never offers that edit: Edit Frames appears once a
// single-canvas run finishes, and stays hidden when any frame of the
// run belongs to a later canvas.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const DGEMMA = {
  id: "dgemma",
  display_name: "DiffusionGemma-26B-A4B",
  min_vram_gib: 18,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    input_mode: "chat",
    supports_resume: true,
    unresolved_char: "\u2591",
    supported_devices: ["cuda"],
  },
  param_specs: [],
  status: "active",
};

const MODELS = {
  models: [DGEMMA],
  active: "dgemma",
  active_device: "cuda",
  active_tokenizer: { name: "dgemma" },
  active_context_length: 4096,
  default: "dgemma",
  gpu_name: "NVIDIA GeForce RTX 4090",
};

function quietFetch(url) {
  const path = String(url).split("?")[0];
  const body = path.startsWith("/api/models") ? MODELS : {};
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

// One frame on canvas `canvasIndex`. DiffusionGemma reports no step
// total, because it stops each canvas when the canvas settles.
function frameOn(canvasIndex, index) {
  const token = { t: " w" + index, m: false, id: 10 + index, c: 0.5 };
  return {
    type: "frame",
    index: index,
    total_steps: null,
    canvas_index: canvasIndex,
    mean_conf: 0.5,
    text: token.t,
    tokens: [token],
    revealed: [],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// A finished run whose frames sit on the canvases listed, in order.
function finishedRun(canvases) {
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch,
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  registry.get("btn-edit-frames").hidden = true;
  page.generatorSocketController().connect();
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  canvases.forEach((canvasIndex, index) => {
    context.handleFrame(frameOn(canvasIndex, index));
  });
  context.handleDone({ type: "done", final_text: "done" });
  return context;
}

test("a single-canvas run offers Edit Frames", () => {
  const context = finishedRun([0, 0, 0, 0]);

  assert.equal(context.generatorEdit.runIsMultiCanvas(), false);
  assert.equal(
    context.document.getElementById("btn-edit-frames").hidden,
    false
  );
});

test("a run that reaches a second canvas does not", () => {
  const context = finishedRun([0, 0, 1, 1]);

  assert.equal(context.generatorEdit.runIsMultiCanvas(), true);
  assert.equal(
    context.document.getElementById("btn-edit-frames").hidden,
    true
  );
});

test("one frame on a later canvas is enough to withhold it", () => {
  const context = finishedRun([0, 0, 0, 1]);

  assert.equal(context.generatorEdit.runIsMultiCanvas(), true);
  assert.equal(
    context.document.getElementById("btn-edit-frames").hidden,
    true
  );
});
