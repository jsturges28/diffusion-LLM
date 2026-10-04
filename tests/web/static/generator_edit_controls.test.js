// The edit session's Back control.
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

const TOKEN = "a3f9c1:1";

// The run's four frames, settling one position at a time.
const TEXTS = [
  MASK + MASK + MASK,
  "a" + MASK + MASK,
  "ab" + MASK,
  "abc",
];

function quietFetch() {
  return function (url) {
    const path = String(url).split("?")[0];
    const body = path.startsWith("/api/models") ? MODELS : {};
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

// A page holding a finished four-frame run, and the socket it opened.
// Two ticks, as in the other socket tests: the first drains connects
// still pending from pages built earlier.
async function finishedRun() {
  await tick();
  const mark = FakeSocket.opened.length;
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch(),
    bootState: { ui_state: {}, models: MODELS },
  });
  await tick();
  assert.equal(FakeSocket.opened.length, mark + 1);
  const { context, registry } = page;
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
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
  assert.equal(context.runFramesLength(context.runFrames), 4);
  return { context, registry, socket: FakeSocket.opened[mark] };
}

// Edit Frames on `frame`, with one token selected and not locked in.
function selectingAt(run, frame) {
  const { context } = run;
  context.enterRemaskMode();
  context.navigateToFrame(frame);
  context.selectCurrentFrame();
  context.toggleRemaskPosition(1);
  assert.equal(context.runPhase.mode, "edit");
}

function press(run, id) {
  run.registry.get(id).dispatch("click");
}

// A copy out of the page's realm, so it compares by value here.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

// -- Back --

test("Back returns to choosing a frame", async () => {
  const run = await finishedRun();
  selectingAt(run, 2);

  press(run, "btn-back-frame");

  assert.equal(run.context.runPhase.mode, "select");
  assert.deepEqual(host(run.context.remaskedPositions), {});
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

  run.context.navigateToFrame(3);
  run.context.selectCurrentFrame();
  run.context.toggleRemaskPosition(0);
  run.context.lockInEdits();

  assert.deepEqual(host(run.context.runPhase.lockedEdits), [
    { frame_index: 3, token_positions: [0] },
  ]);
});

test("Back after Run to Here keeps the edit before it", async () => {
  const run = await finishedRun();
  selectingAt(run, 1);
  run.context.lockInEdits();
  press(run, "btn-edit-another");
  run.context.navigateToFrame(3);
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
  assert.equal(run.context.runPhase.mode, "edit");
  assert.equal(run.context.currentScrubFrame, 3);

  press(run, "btn-back-frame");
  run.context.navigateToFrame(0);

  assert.equal(run.context.runPhase.mode, "select");
  // The floor the earlier edit set: the frame after it.
  assert.equal(run.context.currentScrubFrame, 2);
  assert.equal(run.context.remaskEdits.length, 1);
});
