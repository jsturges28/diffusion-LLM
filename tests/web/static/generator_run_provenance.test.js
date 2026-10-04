// A run's provenance on the generator page, from its first frame on.
//
// Strategy: load the generator page into the DOM stub, start a real
// run through startGeneration, and feed it the frames a worker sends:
// an opening frame carrying the worker's half of the envelope, later
// frames carrying none, and a terminal frame carrying all of it. What
// the page holds as the run's provenance is read back after each, for
// both shapes a frame arrives in.
//
// The gap being closed: the worker used to attest only on the
// terminal frame, so a run whose connection dropped before it had no
// provenance at all, and a save made from it described whatever model
// was resident when the save arrived. Passing proves the opening
// frame's envelope is recorded for a snapshot run and an append run
// alike, that later frames leave it alone, that the terminal frame's
// envelope replaces it, and that the next Generate starts without
// one.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const LLADA = {
  id: "llada",
  display_name: "LLaDA-8B-Instruct",
  min_vram_gib: 17,
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

const SMOL = {
  id: "smollm3",
  display_name: "SmolLM3-3B",
  min_vram_gib: 6,
  capabilities: {
    family: "autoregressive",
    generation_shape: "append_only",
    input_mode: "chat",
    supports_resume: false,
    supported_devices: ["cuda", "cpu"],
  },
  param_specs: [],
  status: "active",
};

// The worker's half, as an opening frame carries it.
function opening(model) {
  return { model_id: model.id, device: "cuda", checkpoint: "org/x" };
}

// The whole envelope, as a terminal frame carries it.
function terminal(model) {
  const envelope = opening(model);
  envelope.resources = { vram_allocated_peak_bytes: 1024 };
  return envelope;
}

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

// A page with a run of `model` under way and no frame yet.
function runOn(model) {
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
  return context;
}

// A snapshot frame over one position, carrying `provenance` if given.
function snapshotFrame(index, provenance) {
  const frame = {
    type: "frame",
    index: index,
    total_steps: 4,
    canvas_index: 0,
    mean_conf: 0.5,
    text: " a" + index,
    tokens: [{ t: " a" + index, m: false, id: 100, c: 0.5 }],
    revealed: [],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (provenance) {
    frame.provenance = provenance;
  }
  return frame;
}

// An append frame adding position `index - 1`, likewise.
function appendFrame(index, provenance) {
  const position = index - 1;
  const frame = {
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
    },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (provenance) {
    frame.provenance = provenance;
  }
  return frame;
}

// The envelope the page holds, copied out of the vm's realm so a deep
// comparison judges its content rather than which realm built it.
function held(context) {
  const value = context.generatorRun.provenance();
  return value === null ? null : JSON.parse(JSON.stringify(value));
}

test("a snapshot run's opening frame names its worker", () => {
  const context = runOn(LLADA);

  context.handleFrame(snapshotFrame(0, opening(LLADA)));

  assert.deepEqual(held(context), opening(LLADA));
});

test("an append run's opening frame names its worker", () => {
  const context = runOn(SMOL);

  context.handleFrame(appendFrame(1, opening(SMOL)));

  assert.deepEqual(held(context), opening(SMOL));
});

test("a later frame leaves the envelope alone", () => {
  const context = runOn(LLADA);

  context.handleFrame(snapshotFrame(0, opening(LLADA)));
  context.handleFrame(snapshotFrame(1));
  context.handleFrame(snapshotFrame(2));

  assert.deepEqual(held(context), opening(LLADA));
});

test("the terminal frame's envelope replaces it", () => {
  // It carries what the run cost, which the opening frame could not.
  const context = runOn(LLADA);
  context.handleFrame(snapshotFrame(0, opening(LLADA)));

  context.handleDone({
    type: "done",
    final_text: " a0",
    provenance: terminal(LLADA),
  });

  assert.deepEqual(held(context), terminal(LLADA));
});

test("the next Generate starts without one", () => {
  const context = runOn(LLADA);
  context.handleFrame(snapshotFrame(0, opening(LLADA)));
  context.handleDone({
    type: "done",
    final_text: " a0",
    provenance: terminal(LLADA),
  });

  context.startGeneration();

  assert.equal(context.generatorRun.provenance(), null);
});

test("an envelope that is not an object is ignored", () => {
  const context = runOn(LLADA);
  context.handleFrame(snapshotFrame(0, opening(LLADA)));

  context.handleFrame(snapshotFrame(1, "llada"));

  assert.deepEqual(held(context), opening(LLADA));
});
