// A run the worker can no longer answer for is locked, and says why.
//
// Strategy: load the generator page into the DOM stub with the socket
// the page opens for itself, run a real generation through
// startGeneration with frames shaped the way each model streams them,
// then end it the way a run ends: a terminal frame, Stop, or a socket
// that closes before either. The page's own handlers run, so this
// exercises the wiring rather than one function, and every request
// the page tries to send is caught on the socket.
//
// The bug being pinned: a run whose connection dropped mid-run kept
// Edit Frames and What If? on offer. No terminal frame ever named the
// run the worker holds, so every edit of it was refused, and refused
// as though another run had replaced it, which is not what happened.
// Passing proves such a run is locked up front with the real reason;
// that a run stopped with Stop, which the worker still holds, stays
// editable; that the lock survives a trip to Analytics; and that no
// stateful request reaches the worker for a locked run, even from a
// session already open, and before anything is cut from the run.
//
// The second half is `A2-LIFE-03`: the same model on the same device
// can be a different worker, reloaded from another window or after a
// restart, and the page only compared model and device. Passing proves
// a run made by a worker that is gone locks in place and says why,
// that reconnecting to the same worker leaves it alone, that a run
// restored after a trip to Analytics is judged the same way, that an
// open session closes unless it holds a branch the page can still
// save, and that an older supervisor or snapshot behaves as before.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const MASK = "\u2591";

const LOST = /lost its connection mid-run/;
const REPLACED = /reloaded since this run was made/;

// What a page sends that only the worker holding its run can answer.
const STATEFUL = ["resume", "substitute", "probe", "rewind"];

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

const SMOL = {
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
  },
  param_specs: [],
  status: "active",
};

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

// Answers the model listing and nothing else of interest.
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

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// The frame a socket opens with, naming the worker it reaches.
function resident(model, worker) {
  const frame = {
    type: "resident",
    model: model.id,
    device: "cuda",
    operation: 1,
  };
  if (worker !== undefined) {
    frame.worker = worker;
  }
  return frame;
}

// A page running `model` with a generation under way, and the socket
// it opened for itself. Two ticks, as in the interrupted-save tests:
// the first drains connects still pending from pages built earlier.
// With `worker`, the socket first says which worker it reaches, as a
// supervisor that names its workers does.
async function runOn(model, worker) {
  await tick();
  const mark = FakeSocket.opened.length;
  const models = modelsFor(model);
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch(models),
    bootState: { ui_state: {}, models: models },
  });
  await tick();
  assert.equal(FakeSocket.opened.length, mark + 1);
  const { context, registry } = page;
  if (worker !== undefined) {
    context.handleResident(resident(model, worker));
  }
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  assert.equal(context.isGenerating, true, "the run did not start");
  return {
    page,
    context,
    registry,
    socket: FakeSocket.opened[mark],
  };
}

function snapshotFrame(index, text, provenance) {
  const frame = {
    type: "frame",
    index: index,
    total_steps: null,
    canvas_index: 0,
    mean_conf: 0.5,
    text: text,
    tokens: Array.from(text).map((character, position) => ({
      t: character,
      m: character === MASK,
      id: 100 + position,
      c: 0.5,
    })),
    revealed: [],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (provenance) {
    frame.provenance = provenance;
  }
  return frame;
}

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

// A LLaDA run whose connection dropped mid-denoise.
async function cutOffLlada() {
  const run = await runOn(LLADA);
  run.context.handleFrame(
    snapshotFrame(0, MASK + MASK, opening(LLADA))
  );
  run.context.handleFrame(snapshotFrame(1, "a" + MASK));
  run.socket.close();
  return run;
}

// A SmolLM3 run whose connection dropped two tokens in.
async function cutOffSmol() {
  const run = await runOn(SMOL);
  run.context.handleFrame(appendFrame(1, " Yeast", opening(SMOL)));
  run.context.handleFrame(appendFrame(2, " eats"));
  run.socket.close();
  return run;
}

function stateful(socket) {
  return socket.sent
    .map((raw) => JSON.parse(raw).type)
    .filter((type) => STATEFUL.includes(type));
}

function isLocked(button) {
  return button.classList.contains("is-locked");
}

test("a run cut off by a dropped connection is locked, saying why", async () => {
  const { registry } = await cutOffLlada();
  const button = registry.get("btn-edit-frames");

  assert.equal(isLocked(button), true);
  assert.match(button.title, LOST);
});

test("Edit Frames then opens nothing", async () => {
  const { context, socket } = await cutOffLlada();

  context.generatorEdit.enterFrames();

  assert.equal(context.generatorEdit.phaseState().mode, null);
  assert.deepEqual(stateful(socket), []);
});

test("What If? is locked the same way", async () => {
  const { registry } = await cutOffSmol();
  const button = registry.get("btn-what-if");

  assert.equal(isLocked(button), true);
  assert.match(button.title, LOST);
});

test("a run stopped with Stop stays editable", async () => {
  // A cancelled run still ends in a terminal frame naming the run,
  // and the worker still holds it.
  const { context, registry } = await runOn(LLADA);
  context.handleFrame(snapshotFrame(0, MASK + MASK, opening(LLADA)));
  context.handleFrame(snapshotFrame(1, "a" + MASK));
  context.handleDone({
    type: "done",
    final_text: "a" + MASK,
    cancelled: true,
    run_token: "a3f9c1:1",
  });

  assert.equal(isLocked(registry.get("btn-edit-frames")), false);
});

test("the lock comes back from a trip to Analytics", async () => {
  const { context, registry } = await cutOffLlada();
  const button = registry.get("btn-edit-frames");
  context.generatorRun.reset();
  button.classList.remove("is-locked");
  button.removeAttribute("aria-disabled");

  assert.equal(context.restoreSessionState(), true);

  assert.equal(isLocked(button), true);
  assert.match(button.title, LOST);
});

test("a composed resume is refused before a stale run is cut", async () => {
  // A session open as the run stopped being editable keeps its own
  // buttons. The request must not go, and the frames must not be cut
  // back for a branch that will never arrive.
  const { context, registry, socket } =
    await finishedLlada("b0a7:1");
  context.generatorEdit.enterFrames();
  context.generatorEdit.navigate(1);
  context.generatorEdit.selectFrame();
  context.generatorEdit.togglePosition(0);
  context.generatorEdit.lockSelection();
  const frames = context.generatorRun.frameCount();
  const before = stateful(socket);
  context.generatorRun.adoptResidentWorker("b0a7:2");

  context.generatorEdit.resumeToEnd();

  assert.equal(context.generatorRun.frameCount(), frames);
  assert.deepEqual(stateful(socket), before);
  assert.match(registry.get("status-message").textContent, REPLACED);
});

test("a substitution already chosen is refused before the run is cut", async () => {
  const run = await runOn(SMOL, "b0a7:1");
  const { context, socket } = run;
  context.handleFrame(appendFrame(1, " Yeast", opening(SMOL)));
  context.handleFrame(appendFrame(2, " eats"));
  context.handleDone({
    type: "done",
    final_text: " Yeast eats",
    run_token: "a3f9c1:1",
  });
  context.generatorEdit.enterWhatIf();
  const frames = context.generatorRun.frameCount();
  const before = stateful(socket);
  context.generatorRun.adoptResidentWorker("b0a7:2");

  context.generatorEdit.substitute({
    position: 0,
    tokenId: 7,
    typedText: null,
  });

  assert.equal(context.generatorRun.frameCount(), frames);
  assert.deepEqual(stateful(socket), before);
});

test("nor is a probe or a rewind sent once the page reconnects", async () => {
  // The two that check the socket first: after a reconnect the socket
  // is open again, and only the lock stands between them and a worker
  // that would refuse them.
  const { page, context } = await cutOffSmol();
  const mark = FakeSocket.opened.length;
  page.generatorSocketController().connect();
  const reconnected = FakeSocket.opened[mark];
  context.generatorRun.finish({
    final_text: " Yeast eats",
    run_token: "a3f9c1:1",
  });

  context.generatorCandidatesRequestProbe({
    position: 0, tokenId: 7, requestId: 1,
  });
  context.generatorEdit.enterWhatIf();

  assert.deepEqual(stateful(reconnected), []);
});

// -- a run whose worker is gone (`A2-LIFE-03`) --

// A LLaDA run that finished on `worker`, which held it.
async function finishedLlada(worker) {
  const run = await runOn(LLADA, worker);
  run.context.handleFrame(
    snapshotFrame(0, MASK + MASK, opening(LLADA))
  );
  run.context.handleFrame(snapshotFrame(1, "ab"));
  run.context.handleDone({
    type: "done",
    final_text: "ab",
    run_token: "a3f9c1:1",
  });
  return run;
}

test("a run whose worker was replaced is locked, saying why", async () => {
  // Reloaded from another window: same model, same device, a worker
  // that holds none of this page's runs.
  const { context, registry } = await finishedLlada("b0a7:1");
  const button = registry.get("btn-edit-frames");

  context.handleResident(resident(LLADA, "b0a7:2"));

  assert.equal(isLocked(button), true);
  assert.match(button.title, REPLACED);
  assert.match(registry.get("status-message").textContent, REPLACED);
});

test("a restart under the run locks it the same way", async () => {
  // The activation number starts again at one after a restart, so
  // only the supervisor's own name tells the two workers apart.
  const { context, registry } = await finishedLlada("b0a7:1");

  context.handleResident(resident(LLADA, "c1d2:1"));

  assert.equal(isLocked(registry.get("btn-edit-frames")), true);
});

test("reconnecting to the same worker leaves the run editable", async () => {
  // Every socket open sends the frame, so a page that locked on all
  // of them would lock every run on the first blip.
  const { context, registry, socket } = await finishedLlada("b0a7:1");

  context.handleResident(resident(LLADA, "b0a7:1"));
  context.generatorEdit.enterFrames();

  assert.equal(isLocked(registry.get("btn-edit-frames")), false);
  assert.equal(
    context.generatorEdit.phaseState().mode,
    "select"
  );
  assert.deepEqual(stateful(socket), ["rewind"]);
});

test("a locked run sends nothing to the worker that replaced it", async () => {
  // The socket is open here, unlike a dropped connection, so the lock
  // is all that keeps the request off the wire.
  const { context, socket } = await finishedLlada("b0a7:1");
  context.handleResident(resident(LLADA, "b0a7:2"));

  context.generatorEdit.enterFrames();
  context.generatorCandidatesRequestProbe({
    position: 0, tokenId: 7, requestId: 1,
  });

  assert.equal(context.generatorEdit.phaseState().mode, null);
  assert.deepEqual(stateful(socket), []);
});

// The trip to Analytics and back, as far as this page can tell: the
// run is forgotten, restored from the snapshot, and the socket opens
// again to say which worker it now reaches.
function returnFromAnalytics(run) {
  const button = run.registry.get("btn-edit-frames");
  run.context.generatorRun.reset();
  button.classList.remove("is-locked");
  button.removeAttribute("aria-disabled");
  assert.equal(run.context.restoreSessionState(), true);
  return button;
}

test("a restored run locks once the socket names another worker", async () => {
  const run = await finishedLlada("b0a7:1");
  const button = returnFromAnalytics(run);

  run.context.handleResident(resident(LLADA, "b0a7:2"));

  assert.equal(isLocked(button), true);
  assert.match(button.title, REPLACED);
});

test("and stays editable when the socket names the same one", async () => {
  const run = await finishedLlada("b0a7:1");
  const button = returnFromAnalytics(run);

  run.context.handleResident(resident(LLADA, "b0a7:1"));

  assert.equal(isLocked(button), false);
});

test("an open frame selection closes when its worker goes", async () => {
  // Nothing in it can be saved yet, and nothing in it can run now.
  const { context, registry } = await finishedLlada("b0a7:1");
  context.generatorEdit.enterFrames();
  assert.equal(
    context.generatorEdit.phaseState().mode,
    "select"
  );

  context.handleResident(resident(LLADA, "b0a7:2"));

  assert.equal(context.generatorEdit.phaseState().mode, null);
  assert.equal(isLocked(registry.get("btn-edit-frames")), true);
});

test("a branch awaiting Confirm is kept, and Retry locks", async () => {
  // Confirm is a save, which needs no worker, so the finished branch
  // stays saveable. Retry would start the edit again on a worker
  // that does not hold the run.
  const { context, registry, socket } = await finishedLlada("b0a7:1");
  context.generatorEdit.enterFrames();
  context.generatorEdit.navigate(1);
  context.generatorEdit.selectFrame();
  context.generatorEdit.togglePosition(0);
  context.generatorEdit.lockSelection();
  context.generatorEdit.resumeToEnd();
  context.handleFrame(snapshotFrame(0, "xb"));
  context.handleDone({
    type: "done",
    final_text: "xb",
    run_token: "a3f9c1:1",
  });
  assert.equal(
    context.generatorEdit.phaseState().mode,
    "review"
  );
  const retry = registry.get("btn-retry-edit");
  const before = stateful(socket);

  context.handleResident(resident(LLADA, "b0a7:2"));
  context.generatorEdit.retry();

  assert.equal(
    context.generatorEdit.phaseState().mode,
    "review"
  );
  assert.equal(isLocked(retry), true);
  assert.match(retry.title, REPLACED);
  assert.equal(isLocked(registry.get("btn-confirm-edit")), false);
  assert.deepEqual(stateful(socket), before);
});

test("a supervisor that names no worker changes nothing", async () => {
  const { context, registry } = await finishedLlada("b0a7:1");

  context.handleResident(resident(LLADA));

  assert.equal(isLocked(registry.get("btn-edit-frames")), false);
});

test("a run made before workers were named stays editable", async () => {
  // Its worker is unknown, so the worker's own refusal is still what
  // answers an edit it cannot serve, as it always was.
  const { context, registry } = await finishedLlada(undefined);

  context.handleResident(resident(LLADA, "b0a7:2"));

  assert.equal(isLocked(registry.get("btn-edit-frames")), false);
});
