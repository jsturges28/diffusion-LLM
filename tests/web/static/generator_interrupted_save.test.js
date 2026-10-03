// A run whose connection dropped can still be saved.
//
// Strategy: load the generator page into the DOM stub with the socket
// the page opens for itself, start a real run through
// startGeneration, feed it frames shaped the way each model streams
// them, then close that socket before any terminal frame, which is
// how a dropped connection or a model switch in another window ends a
// run. The page's own onclose handler runs, so this exercises the
// wiring rather than one function, and the save's request body is
// caught by the page's own fetch.
//
// The bug being pinned: the interrupted state kept the frames and
// offered Save, but only a terminal frame ever set the run's text, so
// Save, the session snapshot and the model-switch rescue all returned
// without a word. Passing proves an interrupted run saves the text
// its frames had reached, marked partial and naming the worker that
// drew it; that a DiffusionGemma run keeps the canvases before the
// one it was cut in; that the run survives a trip to Analytics and a
// model switch elsewhere; that a finished run is left as it was; and
// that a run with nothing to save says so.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const MASK = "\u2591";

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
      unresolved_char: MASK,
      supported_devices: ["cuda"],
    },
    param_specs: [],
    status: "active",
  };
}

const LLADA = diffusionModel("llada");
const DGEMMA = diffusionModel("diffusiongemma");

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

// The worker's half of the envelope, as a run's opening frame
// carries it.
function opening(model) {
  return { model_id: model.id, device: "cuda" };
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

// Answers every request, and keeps the body of each save. The save
// is refused so the page's success path, which is not under test,
// stays out of the way.
function savingFetch(models, saved) {
  return function (url, init) {
    const path = String(url).split("?")[0];
    let body = {};
    if (path === "/api/save") {
      saved.push(JSON.parse(init.body));
      body = { success: false, error: "held by the test" };
    } else if (path.startsWith("/api/models")) {
      body = models;
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  };
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// A page running `model`, with the socket it opened for itself.
//
// Two ticks, as in resource_meter.test.js: the first drains connects
// still pending from pages built earlier, since FakeSocket.opened is
// static and shared; the second waits for this page's own.
async function runOn(model) {
  await tick();
  const mark = FakeSocket.opened.length;
  const saved = [];
  const models = modelsFor(model);
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: savingFetch(models, saved),
    bootState: { ui_state: {}, models: models },
  });
  await tick();
  assert.equal(
    FakeSocket.opened.length,
    mark + 1,
    "expected exactly this page's socket"
  );
  const { context, registry } = page;
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  assert.equal(context.isGenerating, true, "the run did not start");
  return {
    context,
    registry,
    socket: FakeSocket.opened[mark],
    saved,
  };
}

// A snapshot frame on `canvas` showing `text`, one position per
// character, a mask character standing for an unresolved position.
function snapshotFrame(index, canvas, text, provenance) {
  const tokens = Array.from(text).map((character, position) => ({
    t: character,
    m: character === MASK,
    id: 100 + position,
    c: 0.5,
  }));
  const frame = {
    type: "frame",
    index: index,
    total_steps: null,
    canvas_index: canvas,
    mean_conf: 0.5,
    text: text,
    tokens: tokens,
    revealed: [],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (provenance) {
    frame.provenance = provenance;
  }
  return frame;
}

// An append frame adding `word` at position `index - 1`.
function appendFrame(index, word, provenance) {
  const position = index - 1;
  const frame = {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: 4,
    canvas_index: 0,
    mean_conf: 0.5,
    token: { t: word, m: false, id: 200 + position, c: 0.5 },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (provenance) {
    frame.provenance = provenance;
  }
  return frame;
}

// A LLaDA run cut off mid-denoise: two frames, one position settled.
async function interruptedLlada() {
  const run = await runOn(LLADA);
  run.context.handleFrame(
    snapshotFrame(0, 0, MASK + MASK, opening(LLADA))
  );
  run.context.handleFrame(snapshotFrame(1, 0, "a" + MASK));
  run.socket.close();
  return run;
}

test("an interrupted LLaDA run saves what it reached", async () => {
  const { context, saved } = await interruptedLlada();

  await context.saveRun();

  assert.equal(saved.length, 1);
  assert.equal(saved[0].final_text, "a" + MASK);
  assert.equal(saved[0].partial, true);
});

test("its save names the worker that drew it", async () => {
  // Read off the opening frame. A dropped connection sends no
  // terminal frame, and without this the save would describe
  // whichever model is resident by the time it lands.
  const { context, saved } = await interruptedLlada();

  await context.saveRun();

  assert.deepEqual(saved[0].provenance, opening(LLADA));
});

test("DiffusionGemma keeps every canvas it reached", async () => {
  // One canvas per frame: a draft and a commit on the first, then a
  // draft of the second, which the connection dropped in.
  const { context, socket, saved } = await runOn(DGEMMA);
  context.handleFrame(
    snapshotFrame(0, 0, "a" + MASK, opening(DGEMMA))
  );
  context.handleFrame(snapshotFrame(1, 0, "ab"));
  context.handleFrame(snapshotFrame(2, 1, "c" + MASK));

  socket.close();
  await context.saveRun();

  assert.equal(saved[0].final_text, "abc" + MASK);
});

test("an autoregressive run saves its tokens joined", async () => {
  const { context, socket, saved } = await runOn(SMOL);
  context.handleFrame(appendFrame(1, " Yeast", opening(SMOL)));
  context.handleFrame(appendFrame(2, " eats"));

  socket.close();
  await context.saveRun();

  assert.equal(saved[0].final_text, " Yeast eats");
  assert.equal(saved[0].partial, true);
});

test("it survives a trip to Analytics and back", async () => {
  // The snapshot is written as the run stops, so leaving for
  // Analytics before saving does not lose it or its provenance.
  const { context, saved } = await interruptedLlada();
  context.lastFinalText = null;
  context.runInterrupted = false;
  context.lastRunProvenance = null;

  assert.equal(context.restoreSessionState(), true);
  await context.saveRun();

  assert.equal(saved[0].final_text, "a" + MASK);
  assert.equal(saved[0].partial, true);
  assert.deepEqual(saved[0].provenance, opening(LLADA));
});

test("a switch elsewhere saves it before the reload", async () => {
  // Another window changed the model, so this page reloads onto it,
  // and the interrupted run is written down first.
  const { context, saved } = await interruptedLlada();

  context.handleResident({
    type: "resident",
    model: "smollm3",
    device: "cuda",
  });
  await tick();

  assert.equal(saved.length, 1);
  assert.equal(saved[0].model, "llada");
  assert.equal(saved[0].final_text, "a" + MASK);
  assert.equal(saved[0].partial, true);
});

test("a run that finished is left as it was", async () => {
  // Only a run still in flight is interrupted. A socket closing after
  // the terminal frame must not trade the run's own text for its
  // frames'.
  const { context, socket, saved } = await runOn(LLADA);
  context.handleFrame(
    snapshotFrame(0, 0, MASK + MASK, opening(LLADA))
  );
  context.handleFrame(snapshotFrame(1, 0, "ab"));
  context.handleDone({ type: "done", final_text: "ab, finished" });

  socket.close();
  await context.saveRun();

  assert.equal(saved[0].final_text, "ab, finished");
  assert.equal(saved[0].partial, undefined);
});

test("a run with nothing to save says so", async () => {
  // Save is offered whenever the run has frames, so a click that did
  // nothing at all read as a broken button.
  const { context, registry, socket, saved } = await runOn(LLADA);
  context.handleFrame(snapshotFrame(0, 0, "", opening(LLADA)));
  socket.close();

  await context.saveRun();

  assert.equal(saved.length, 0);
  assert.match(
    registry.get("status-message").textContent, /nothing to save/
  );
});
