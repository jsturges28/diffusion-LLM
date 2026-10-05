// The edit session's Back and Continue controls.
//
// Strategy: load the generator page into the DOM stub with the socket
// the page opens for itself, finish a DiffusionGemma-shaped run of
// four frames, and drive an edit session through the page's own
// functions and buttons. Worker replies are frames and terminal
// frames handed to the page's handlers, shaped the way the worker
// sends them.
//
// Back sits beside Lock In, Clear and Exit once a frame is selected.
// Before it, the only way to choose a different frame was Exit, which
// throws away every earlier step of the session with it. Passing
// proves Back returns to choosing a frame with this frame's unlocked
// selection dropped and the scrubber free, that the next frame chosen
// is the one locked in, and that after a Run to Here it keeps the
// edit before it and the forward-only floor that edit set.
//
// Continue sits in review beside Confirm and Retry when the branch
// was stopped, which used to leave only saving it short or throwing
// it away. Passing proves it is offered for a stopped diffusion
// branch and nowhere else, that review says the branch stopped, that
// it asks the worker to carry the branch on from its last frame
// without recording an edit, that the carried-on branch lands in
// review, that one stopped before its first frame changes nothing,
// and that it locks with Retry once the run's worker is gone.
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
const SCHEMA_ID = "a".repeat(64);

const DGEMMA = {
  id: "dgemma",
  display_name: "DiffusionGemma",
  min_vram_gib: 18,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    input_mode: "chat",
    supports_resume: true,
    unresolved_char: MASK,
    supported_devices: ["cuda"],
  },
  generation_schema_ids: { cuda: SCHEMA_ID },
  param_specs: [],
  status: "active",
};

const MODELS = {
  models: [DGEMMA],
  active: DGEMMA.id,
  active_device: "cuda",
  active_tokenizer: { name: DGEMMA.id },
  active_context_length: 4096,
  default: DGEMMA.id,
  gpu_name: "NVIDIA GeForce RTX 4090",
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
    supports_substitution: true,
    supported_devices: ["cuda", "cpu"],
  },
  generation_schema_ids: { cuda: SCHEMA_ID, cpu: SCHEMA_ID },
  param_specs: [],
  status: "active",
};

const TOKEN = "a3f9c1:1";

const UNCHANGED =
  "Stopped before the edit produced a frame. The run is unchanged.";

// The run's four frames, settling one position at a time.
const TEXTS = [
  MASK + MASK + MASK,
  "a" + MASK + MASK,
  "ab" + MASK,
  "abc",
];

function modelsFor(model) {
  return Object.assign({}, MODELS, {
    models: [model],
    active: model.id,
    active_tokenizer: { name: model.id },
    default: model.id,
  });
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

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function snapshotFrame(index, text) {
  return {
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
}

function appendFrame(index, word) {
  return {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: 4,
    canvas_index: 0,
    mean_conf: 0.5,
    token: { t: word, m: false, id: 200 + index, c: 0.5 },
    revealed: [index - 1],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// The frame a socket opens with, naming the worker it reaches.
function resident(model, worker) {
  return {
    type: "resident",
    model: model.id,
    device: "cuda",
    generation_schema_id: SCHEMA_ID,
    operation: 1,
    worker: worker,
  };
}

// A page running `model` with a generation under way, and the socket
// it opened. Two ticks, as in the other socket tests: the first
// drains connects still pending from pages built earlier. With
// `worker`, the socket first says which worker it reaches.
async function generating(model, worker) {
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
  return { context, registry, socket: FakeSocket.opened[mark] };
}

// A page holding a finished four-frame run.
async function finishedRun(worker) {
  const run = await generating(DGEMMA, worker);
  const { context } = run;
  for (let index = 0; index < TEXTS.length; index++) {
    context.handleFrame(snapshotFrame(index, TEXTS[index]));
  }
  context.handleDone({
    type: "done",
    final_text: "abc",
    thinking: "",
    prompt_len: 12,
    run_token: TOKEN,
  });
  assert.equal(context.generatorRun.frameCount(), 4);
  return run;
}

// Edit Frames on `frame`, with one token selected and not locked in.
function selectingAt(run, frame) {
  const { context } = run;
  context.generatorEdit.enterFrames();
  context.generatorEdit.navigate(frame);
  context.generatorEdit.selectFrame();
  context.generatorEdit.togglePosition(1);
  assert.equal(context.generatorEdit.phaseState().mode, "edit");
}

function press(run, id) {
  run.registry.get(id).dispatch("click");
}

// A copy out of the page's realm, so it compares by value here.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function resumesSent(socket) {
  return socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) => message.type === "resume");
}

function guidedStatus(run) {
  return run.registry.get("guided-edit-status").textContent;
}

function continueHidden(run) {
  return run.registry.get("btn-continue-edit").hidden;
}

// Resume to End from an edit on frame 2, stopped once the branch's
// first frame is on screen: review, with three frames.
function stoppedBranch(run) {
  selectingAt(run, 2);
  run.context.generatorEdit.lockSelection();
  press(run, "btn-resume-end");
  run.context.handleFrame(snapshotFrame(0, "a" + MASK + "c"));
  run.context.handleDone({
    type: "done",
    final_text: "a" + MASK + "c",
    thinking: "",
    prompt_len: 12,
    cancelled: true,
    run_token: TOKEN,
  });
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
}

function finishedDone(text) {
  return {
    type: "done",
    final_text: text,
    thinking: "",
    prompt_len: 12,
    run_token: TOKEN,
  };
}

// -- Back --

test("Back returns to choosing a frame", async () => {
  const run = await finishedRun();
  selectingAt(run, 2);

  press(run, "btn-back-frame");

  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "select"
  );
  assert.deepEqual(
    host(
      run.context.generatorEdit
        .readoutsSettings().remaskedPositions
    ),
    {}
  );
  assert.equal(run.registry.get("btn-scrub-prev").disabled, false);
  assert.equal(
    run.registry.get("guided-edit-status").textContent,
    "Navigate to a frame, then select it for editing."
  );
});

test("and the frame chosen next is the one locked in", async () => {
  const run = await finishedRun();
  selectingAt(run, 2);
  press(run, "btn-back-frame");

  run.context.generatorEdit.navigate(3);
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();

  assert.deepEqual(
    host(run.context.generatorEdit.phaseState().lockedEdits),
    [
    { frame_index: 3, token_positions: [0] },
    ]
  );
});

test("Back after Run to Here keeps the edit before it", async () => {
  const run = await finishedRun();
  selectingAt(run, 1);
  run.context.generatorEdit.lockSelection();
  press(run, "btn-edit-another");
  run.context.generatorEdit.navigate(3);
  press(run, "btn-run-to-here");
  // The branch, from the edited frame up to the target.
  run.context.handleFrame(snapshotFrame(0, "a" + MASK + MASK));
  run.context.handleFrame(snapshotFrame(1, "ax" + MASK));
  run.context.handleFrame(snapshotFrame(2, "axc"));
  run.context.handleDone({
    type: "done",
    final_text: "axc",
    thinking: "",
    prompt_len: 12,
    run_token: TOKEN,
  });
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "edit"
  );
  assert.equal(run.context.generatorEdit.currentFrame(), 3);

  press(run, "btn-back-frame");
  run.context.generatorEdit.navigate(0);

  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "select"
  );
  // The floor the earlier edit set: the frame after it.
  assert.equal(run.context.generatorEdit.currentFrame(), 2);
  assert.equal(
    run.context.generatorEdit.readArtifacts().remaskEdits.length,
    1
  );
});

// -- Continue --

test("a stopped Resume to End offers Continue", async () => {
  const run = await finishedRun();

  stoppedBranch(run);

  assert.equal(continueHidden(run), false);
  assert.equal(
    guidedStatus(run),
    "Stopped at frame 2. Continue, confirm to save it as it is,"
      + " or retry from the start."
  );
});

test("a finished one offers only Confirm and Retry", async () => {
  const run = await finishedRun();
  selectingAt(run, 2);
  run.context.generatorEdit.lockSelection();
  press(run, "btn-resume-end");
  run.context.handleFrame(snapshotFrame(0, "a" + MASK + "c"));
  run.context.handleFrame(snapshotFrame(1, "abc"));

  run.context.handleDone(finishedDone("abc"));

  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
  assert.equal(continueHidden(run), true);
  assert.equal(
    guidedStatus(run),
    "Edit complete. Confirm to save, or retry from the start."
  );
});

test("Continue carries the branch on without an edit", async () => {
  const run = await finishedRun();
  stoppedBranch(run);

  press(run, "btn-continue-edit");

  const sent = resumesSent(run.socket);
  assert.equal(sent.length, 2);
  assert.deepEqual(sent[1], {
    type: "resume",
    frame_index: 2,
    remask_positions: [],
    continue: true,
    run_token: TOKEN,
  });
  assert.equal(
    run.context.generatorEdit.readArtifacts().remaskEdits.length,
    1
  );
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "generating"
  );
  assert.equal(run.context.generatorRun.frameCount(), 2);
});

test("and the branch it carries on lands in review", async () => {
  const run = await finishedRun();
  stoppedBranch(run);
  press(run, "btn-continue-edit");
  // The frame it continued from, as it was, then the rest.
  run.context.handleFrame(snapshotFrame(0, "a" + MASK + "c"));
  run.context.handleFrame(snapshotFrame(1, "abc"));

  run.context.handleDone(finishedDone("abc"));

  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
  assert.equal(run.context.generatorRun.frameCount(), 4);
  assert.equal(
    run.context.generatorEdit.readArtifacts().remaskEdits.length,
    1
  );
  assert.equal(continueHidden(run), true);
});

test("a Continue stopped at once changes nothing", async () => {
  const run = await finishedRun();
  stoppedBranch(run);
  press(run, "btn-continue-edit");

  run.context.handleDone({
    type: "done",
    final_text: "",
    thinking: "",
    prompt_len: 12,
    cancelled: true,
    run_token: TOKEN,
  });

  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
  assert.equal(run.context.generatorRun.frameCount(), 3);
  assert.equal(continueHidden(run), false);
  assert.equal(
    run.registry.get("status-message").textContent, UNCHANGED
  );
});

test("Continue locks once the run's worker is replaced", async () => {
  const run = await finishedRun("b0a7:1");
  stoppedBranch(run);
  const button = run.registry.get("btn-continue-edit");

  run.context.handleResident(resident(DGEMMA, "b0a7:2"));
  press(run, "btn-continue-edit");

  assert.equal(button.classList.contains("is-locked"), true);
  assert.equal(resumesSent(run.socket).length, 1);
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
});

test("a stopped What If branch offers no Continue", async () => {
  // Its worker keeps no branch, so there is nothing to carry on.
  const run = await generating(SMOL);
  const { context } = run;
  context.handleFrame(appendFrame(1, " Yeast"));
  context.handleFrame(appendFrame(2, " eats"));
  context.handleFrame(appendFrame(3, " sugar"));
  context.handleDone(finishedDone(" Yeast eats sugar"));
  context.generatorEdit.enterWhatIf();
  context.generatorEdit.substitute({
    position: 1,
    tokenId: 7,
    typedText: null,
  });
  context.handleFrame(appendFrame(2, " ale"));

  context.handleDone({
    type: "done",
    final_text: " Yeast ale",
    thinking: "",
    prompt_len: 12,
    cancelled: true,
    run_token: TOKEN,
  });

  assert.equal(
    context.generatorEdit.phaseState().mode,
    "review"
  );
  assert.equal(continueHidden(run), true);
  assert.equal(
    guidedStatus(run),
    "Stopped at frame 1. Confirm to save it as it is, or retry"
      + " from the start."
  );
});
