// What a save records is the run, not the form.
//
// Strategy: load the generator page into the DOM stub with SmolLM3's
// parameter panel and a socket that reports itself open, start a real
// run through startGeneration, finish it with the frames and the
// terminal frame a worker sends, then change the form the way a user
// can once a run is over: browse the prompt history to another
// prompt, or edit a parameter. The save's request body, caught by the
// page's own fetch, is what is checked.
//
// The bug being pinned: Save read the prompt box at save time, so a
// prompt browsed to after the run was recorded as the run's. The
// parameters had the same flaw one step removed: they were read back
// from the form on every terminal frame, including a resumed edit's,
// which sends none.
//
// Passing proves a save records the prompt and the parameters the run
// was generated from, whatever the form shows at save time, and that
// the snapshot carrying a run through a trip to Analytics keeps the
// run's prompt apart from the box's.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

const RAN = "explain how yeast makes bread rise";
const BROWSED = "an older prompt from the history";
const WORDS = [" Yeast", " eats", " sugar", "."];

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const SMOL = {
  id: "smollm3",
  display_name: "SmolLM3-3B",
  min_vram_gib: 6,
  capabilities: {
    family: "autoregressive",
    generation_shape: "append_only",
    supported_devices: ["cuda", "cpu"],
  },
  param_specs: [
    {
      name: "max_new_tokens",
      label: "Max new tokens",
      type: "int",
      default: 256,
      min: 1,
      max: 2048,
    },
    {
      name: "temperature",
      label: "Temperature",
      type: "float",
      default: 0.7,
      min: 0,
      max: 2,
      step: 0.1,
    },
  ],
  status: "active",
};

const MODELS = {
  models: [SMOL],
  active: "smollm3",
  active_device: "cuda",
  active_tokenizer: { name: "smollm3" },
  active_context_length: 65536,
  default: "smollm3",
  gpu_name: "NVIDIA GeForce RTX 4090",
};

// Answers every request, and keeps the body of each save. The save
// is refused so the page's success path, which is not under test,
// stays out of the way.
function savingFetch(saved) {
  return function (url, init) {
    const path = String(url).split("?")[0];
    let body = {};
    if (path === "/api/save") {
      saved.push(JSON.parse(init.body));
      body = { success: false, error: "held by the test" };
    } else if (path.startsWith("/api/models")) {
      body = MODELS;
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  };
}

// One append frame as `_build_append_frame` in ar_sampler.py emits it.
function appendFrame(index) {
  const position = index - 1;
  return {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: WORDS.length,
    canvas_index: 0,
    mean_conf: 0.5,
    token: {
      t: WORDS[position],
      m: false,
      id: 1000 + position,
      c: 0.5,
      e: 1.2,
    },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// A page holding a finished run of RAN, with BROWSED already in the
// history behind it.
function finishedRun() {
  const saved = [];
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: savingFetch(saved),
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  context.promptHistory = [BROWSED];
  context.updatePromptHistoryUI();
  registry.get("prompt-input").value = RAN;

  context.startGeneration();
  for (let index = 1; index <= WORDS.length; index++) {
    context.handleFrame(appendFrame(index));
  }
  context.handleDone({
    type: "done",
    final_text: WORDS.join(""),
    prompt_len: 7,
  });
  return { page, context, registry, saved };
}

function browseToOlder(context) {
  context.enterPromptHistory();
  context.cyclePromptHistory(1);
}

test("a save records the prompt the run was generated from", async () => {
  const { context, registry, saved } = finishedRun();
  browseToOlder(context);
  assert.equal(registry.get("prompt-input").value, BROWSED);

  await context.saveRun();

  assert.equal(saved.length, 1);
  assert.equal(saved[0].prompt, RAN);
});

test("a prompt kept from the history is still not the run's", async () => {
  // Confirming the browsed prompt makes it the box's text for real,
  // ready to be edited and run next. It still did not produce this
  // run.
  const { context, saved } = finishedRun();
  browseToOlder(context);
  context.confirmPromptHistory();

  await context.saveRun();

  assert.equal(saved[0].prompt, RAN);
});

test("a save records the parameters the run was generated from", async () => {
  const { context, saved } = finishedRun();
  context.paramInputs.temperature.value = "1.5";

  await context.saveRun();

  assert.equal(saved[0].params.temperature, 0.7);
});

test("a resumed edit keeps the run's parameters", async () => {
  // A resume sends no parameters; the worker reuses the run's. Its
  // terminal frame used to read the form back anyway, so a value
  // changed between the run and the edit was saved as if it had
  // produced both.
  const { context, saved } = finishedRun();
  context.paramInputs.temperature.value = "1.5";

  context.handleDone({ type: "done", final_text: "Yeast eats." });
  await context.saveRun();

  assert.equal(saved[0].params.temperature, 0.7);
});

test("the Analytics snapshot keeps the run's prompt apart", () => {
  // The box's text comes back into the box; the run's prompt comes
  // back as the run's. Folding the two is how the wrong prompt used
  // to survive the round trip and be saved later.
  const { page, context } = finishedRun();
  browseToOlder(context);
  context.confirmPromptHistory();

  context.saveSessionState();

  const stored = page.sandbox.sessionStorage.getItem(
    context.SESSION_KEY
  );
  const snapshot = JSON.parse(stored);
  assert.equal(snapshot.runPrompt, RAN);
  assert.equal(snapshot.prompt, BROWSED);
});

test("an older snapshot falls back to the text it kept", () => {
  // Written before the run carried its prompt: the box text at that
  // time is the best record left of what ran.
  const { context } = loadPage({});
  const runPrompt = context.runSnapshotRunPrompt;

  assert.equal(runPrompt({ runPrompt: RAN, prompt: BROWSED }), RAN);
  assert.equal(runPrompt({ prompt: ` ${RAN} ` }), RAN);
  assert.equal(runPrompt({ prompt: "" }), null);
  assert.equal(runPrompt({}), null);
});
