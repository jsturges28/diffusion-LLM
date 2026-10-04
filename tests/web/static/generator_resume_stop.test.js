// A resume stopped before its first frame changes nothing.
//
// Strategy: load the generator page into the DOM stub with the socket
// the page opens for itself, finish a DiffusionGemma-shaped run of
// four frames, and open an edit session through the page's own
// functions and buttons: Edit Frames, a frame, a token, Lock In, then
// Resume to End, or Edit Another Frame and Run to Here. The worker's
// answer is a terminal frame handed to the page's handler, cancelled
// and with no frame before it, which is what DiffusionGemma sends
// when Stop lands before its first denoising step.
//
// The page cuts its run back before it sends a resume, and the worker
// keeps the run it had when a resume stops before sending a frame.
// The page used to stay cut back, in review with the run ending
// before the edit, where Confirm would have saved it that way.
// Passing proves it now goes back to where the resume was sent from,
// with the whole run, the locked edit and its selection, the run's
// thinking, and a status line saying nothing changed; that the resume
// can then be sent again; and that a resume stopped after a frame
// still keeps its branch and opens review, as before.
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

const UNCHANGED =
  "Stopped before the edit produced a frame. The run is unchanged.";

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
    thinking: "pondering",
    prompt_len: 12,
    run_token: TOKEN,
  });
  assert.equal(context.generatorRun.frameCount(), 4);
  return { context, registry, socket: FakeSocket.opened[mark] };
}

// Edit Frames on `frame`, one token remasked and locked in.
function lockedEditAt(run, frame) {
  const { context } = run;
  context.enterRemaskMode();
  context.navigateToFrame(frame);
  context.selectCurrentFrame();
  context.toggleRemaskPosition(1);
  context.lockInEdits();
  assert.equal(context.runPhase.mode, "choice");
}

function press(run, id) {
  run.registry.get(id).dispatch("click");
}

function resumesSent(socket) {
  return socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) => message.type === "resume");
}

// What DiffusionGemma ends a resume with when Stop lands before its
// first denoising step: cancelled, and no frame before it.
function stoppedBeforeAFrame(run) {
  run.context.handleDone({
    type: "done",
    final_text: "",
    thinking: "",
    prompt_len: 12,
    cancelled: true,
    run_token: TOKEN,
  });
}

function statusLine(run) {
  return run.registry.get("status-message").textContent;
}

// -- Resume to End --

test("a stopped Resume to End puts the run back", async () => {
  const run = await finishedRun();
  lockedEditAt(run, 2);
  press(run, "btn-resume-end");
  assert.equal(run.context.generatorRun.frameCount(), 2);

  stoppedBeforeAFrame(run);

  assert.equal(run.context.generatorRun.frameCount(), 4);
  assert.equal(run.context.generatorRun.finalText(), "abc");
  assert.equal(run.context.remaskEdits.length, 0);
  assert.equal(run.context.generatorRun.interrupted(), false);
  assert.equal(statusLine(run), UNCHANGED);
});

test("and goes back to the choice it was sent from", async () => {
  const run = await finishedRun();
  lockedEditAt(run, 2);
  press(run, "btn-resume-end");

  stoppedBeforeAFrame(run);

  assert.equal(run.context.runPhase.mode, "choice");
  assert.equal(run.context.runPhase.lockedEdits.length, 1);
  assert.equal(run.context.currentScrubFrame, 2);
  assert.equal(
    run.registry.get("guided-edit-status").textContent,
    "1 token locked on Frame 2."
  );
});

test("from where the same resume can be sent again", async () => {
  const run = await finishedRun();
  lockedEditAt(run, 2);
  press(run, "btn-resume-end");
  stoppedBeforeAFrame(run);

  press(run, "btn-resume-end");

  const sent = resumesSent(run.socket);
  assert.equal(sent.length, 2);
  assert.deepEqual(sent[1], sent[0]);
  assert.equal(run.context.runPhase.mode, "generating");
});

test("the run keeps its thinking", async () => {
  // The panel is what Save reads, so the empty thinking on the
  // stopped resume's terminal frame must not clear it.
  const run = await finishedRun();
  lockedEditAt(run, 2);
  press(run, "btn-resume-end");

  stoppedBeforeAFrame(run);

  const panel = run.registry.get("thinking-panel");
  const content = run.registry.get("thinking-content");
  assert.equal(content.textContent, "pondering");
  assert.equal(panel.hidden, false);
});

// -- Run to Here --

test("a stopped Run to Here goes back to its target", async () => {
  const run = await finishedRun();
  lockedEditAt(run, 1);
  press(run, "btn-edit-another");
  run.context.navigateToFrame(3);
  press(run, "btn-run-to-here");
  assert.equal(resumesSent(run.socket)[0].max_frames, 3);

  stoppedBeforeAFrame(run);

  assert.equal(run.context.generatorRun.frameCount(), 4);
  assert.equal(run.context.runPhase.mode, "select_target");
  assert.equal(run.context.runPhase.targetFrame, null);
  assert.equal(run.context.runPhase.guidedAction, null);
  assert.equal(run.context.currentScrubFrame, 3);
  assert.equal(statusLine(run), UNCHANGED);

  press(run, "btn-run-to-here");
  assert.equal(resumesSent(run.socket).length, 2);
});

// -- and a resume that did send a frame --

test("a resume stopped after a frame keeps its branch", async () => {
  // Negative space: the worker committed that frame, so the page
  // keeps it and opens review, as it always has.
  const run = await finishedRun();
  lockedEditAt(run, 2);
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

  assert.equal(run.context.generatorRun.frameCount(), 3);
  assert.equal(run.context.runPhase.mode, "review");
  assert.equal(run.context.remaskEdits.length, 1);
  assert.equal(statusLine(run), "Stopped.");
});
