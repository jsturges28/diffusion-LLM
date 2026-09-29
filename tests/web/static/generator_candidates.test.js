// The generator's candidate popover on a diffusion run, driven for real.
//
// Strategy: load the generator page into the DOM stub with a LLaDA
// entry, start a real run through startGeneration, and finish it with
// what a worker sends: snapshot frames, one candidates message, and
// the terminal frame. Then scrub, render the popover at a position
// and read what it drew; save and read the request body; and take a
// resume, a Retry and a trip through the session snapshot the way
// the page does.
//
// Passing proves the popover follows the scrubber: a captured frame
// shows its own candidates under "Step N", a frame the stride skipped
// shows the latest captured before it under "As of step N", and the
// opening frame, a resumed edit's first frame and the pre-edit layer
// show nothing. It also proves the candidates reach the save, land
// after the point a resume branched from, come back on Retry, and
// survive the snapshot a trip to Analytics depends on.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const WORDS = [" Yeast", " eats", " sugar", "."];
const STEPS = 4;
const MASK_ID = 126336;

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

const MODELS = {
  models: [LLADA],
  active: "llada",
  active_device: "cuda",
  active_tokenizer: { name: "llada" },
  active_context_length: 4096,
  default: "llada",
  gpu_name: "NVIDIA GeForce RTX 4090",
};

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

// Frame `index` of a canvas that settles one word per step.
function canvasFrame(index) {
  const tokens = WORDS.map((word, position) => {
    const settled = position < index;
    return {
      t: word,
      m: !settled,
      id: settled ? 1000 + position : MASK_ID,
      c: 0.5,
    };
  });
  return {
    type: "frame",
    index: index,
    total_steps: STEPS,
    canvas_index: 0,
    mean_conf: 0.5,
    text: WORDS.join(""),
    tokens: tokens,
    revealed: index > 0 ? [index - 1] : [],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

// A set per position whose likeliest candidate names the frame and
// the position, so a popover says which frame it was read at.
function setsAt(frame) {
  return WORDS.map((_, position) => {
    const lead = frame * 100 + position;
    return {
      h: lead,
      c: [
        { id: lead, t: " lead", p: 0.6 },
        { id: 7, t: " seven", p: 0.2 },
      ],
    };
  });
}

// What the worker sends when the capture thinned to a stride of 2:
// every other step, and the final one whatever the stride.
function candidatesMessage(frames) {
  return {
    type: "candidates",
    k: 5,
    stride: 2,
    frames: frames,
    sets: frames.map(setsAt),
  };
}

// A finished run whose capture kept `captured`. `canvases` gives each
// frame's canvas, for a run that chains them.
function finishedRun(options) {
  const settings = options || {};
  const captured = settings.captured || [1, 3, 4];
  const saved = [];
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: savingFetch(saved),
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  for (let index = 0; index <= STEPS; index++) {
    const frame = canvasFrame(index);
    if (settings.canvases) {
      frame.canvas_index = settings.canvases[index];
    }
    context.handleFrame(frame);
  }
  context.handleMessage(candidatesMessage(captured));
  context.handleDone({ type: "done", final_text: WORDS.join("") });
  return { context, registry, saved };
}

function descendants(node) {
  const found = [];
  const stack = node.children.slice();
  while (stack.length > 0) {
    const next = stack.shift();
    found.push(next);
    stack.push(...next.children);
  }
  return found;
}

function withClass(node, name) {
  return descendants(node).filter((n) => n.classes.has(name));
}

// The popover as drawn for `position` with the scrubber at `frame`.
// The stub keeps children when text is cleared, so it starts empty.
function popoverAt(context, registry, frame, position) {
  const popover = registry.get("token-alts-popover");
  context.navigateToFrame(frame);
  popover.children = [];
  context.renderAltsPopover(position, null);
  return popover;
}

function stepLabel(popover) {
  return withClass(popover, "alt-step")[0].textContent;
}

function rowIds(popover) {
  return withClass(popover, "alt-row").map(
    (row) => Number(row.getAttribute("data-alt-id"))
  );
}

// -- the popover follows the scrubber --

test("a captured frame shows its own candidates", () => {
  const { context, registry } = finishedRun();

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(popover.hidden, false);
  assert.equal(stepLabel(popover), "Step 3");
  assert.deepEqual(rowIds(popover), [301, 7]);
});

test("a skipped frame shows the latest captured, and says so", () => {
  const { context, registry } = finishedRun();

  const popover = popoverAt(context, registry, 2, 1);

  assert.equal(stepLabel(popover), "As of step 1");
  assert.deepEqual(rowIds(popover), [101, 7]);
});

test("the row for the token on screen is marked", () => {
  const { context, registry } = finishedRun();

  const popover = popoverAt(context, registry, 4, 2);
  const chosen = withClass(popover, "alt-row-chosen");

  assert.equal(chosen.length, 1);
  assert.equal(Number(chosen[0].getAttribute("data-alt-id")), 402);
});

test("the opening frame has no popover", () => {
  // Frame 0 is the canvas before the first forward pass.
  const { context, registry } = finishedRun();

  const popover = popoverAt(context, registry, 0, 1);

  assert.equal(popover.hidden, true);
});

test("a frame on a new canvas never borrows from the last", () => {
  // DiffusionGemma chains canvases. Frame 3 starts canvas 1 and was
  // skipped; frame 2 before it describes unrelated positions.
  const { context, registry } = finishedRun({
    captured: [1, 2, 4],
    canvases: [0, 0, 0, 1, 1],
  });

  assert.equal(popoverAt(context, registry, 3, 1).hidden, true);
  const next = popoverAt(context, registry, 4, 1);
  assert.equal(stepLabel(next), "Step 4");
});

test("the pre-edit layer has no popover", () => {
  // Its candidates are not kept, and the edited run's would describe
  // tokens that are not the ones under the pointer.
  const { context, registry } = finishedRun();
  context.remaskEdits = [{ frame_index: 2, token_positions: [1] }];
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(popover.hidden, true);
});

// -- the run's candidates travel with it --

test("the save carries the candidates", async () => {
  const { context, saved } = finishedRun();

  await context.saveRun();

  const candidates = saved[0].candidates;
  assert.deepEqual(candidates.frames, [1, 3, 4]);
  assert.deepEqual(candidates.segments, [0]);
  assert.equal(candidates.k, 5);
  assert.equal(candidates.sets[1][2].h, 302);
});

test("a run without candidates saves none", async () => {
  const saved = [];
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: savingFetch(saved),
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  for (let index = 0; index <= STEPS; index++) {
    context.handleFrame(canvasFrame(index));
  }
  context.handleDone({ type: "done", final_text: WORDS.join("") });

  await context.saveRun();

  assert.equal("candidates" in saved[0], false);
});

// -- an edit, Retry, and the snapshot --

function resumeFrom(context, offset, frames) {
  context.truncateRunArraysAt(offset);
  context.isResuming = true;
  for (let local = 0; local < frames; local++) {
    context.handleFrame(canvasFrame(offset + local));
  }
}

test("a resume's candidates land after the point it branched from", () => {
  const { context, registry } = finishedRun();
  resumeFrom(context, 2, 3);

  context.handleMessage(candidatesMessage([1, 2]));

  const popover = popoverAt(context, registry, 3, 1);
  assert.equal(stepLabel(popover), "Step 3");
  assert.deepEqual(rowIds(popover), [101, 7]);
  assert.deepEqual(
    Array.from(context.runCandidates.segments), [0, 2]
  );
});

test("a resumed frame never shows the replaced run's candidates", () => {
  // The resume's own first frame is the remasked canvas, before any
  // step. Frame 1 before it has candidates, from the replaced run.
  const { context, registry } = finishedRun();
  resumeFrom(context, 2, 3);
  context.handleMessage(candidatesMessage([1, 2]));

  const popover = popoverAt(context, registry, 2, 1);

  assert.equal(popover.hidden, true);
});

test("an edit that brings no candidates shows none", () => {
  // A guided Run to Here sends none. Its frames must not show the
  // candidates of the frames they replaced.
  const { context, registry } = finishedRun();
  resumeFrom(context, 2, 3);

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(popover.hidden, true);
});

test("a new run forgets the last run's candidates", () => {
  const { context, registry } = finishedRun();
  context.startGeneration();
  for (let index = 0; index <= STEPS; index++) {
    context.handleFrame(canvasFrame(index));
  }
  context.handleDone({ type: "done", final_text: WORDS.join("") });

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(popover.hidden, true);
});

test("Retry brings the run's candidates back", () => {
  const { context } = finishedRun();
  context.captureEditSnapshot();
  resumeFrom(context, 2, 3);
  context.handleMessage(candidatesMessage([1]));

  context.restoreEditSnapshot();

  assert.deepEqual(
    Array.from(context.runCandidates.frames), [1, 3, 4]
  );
  assert.deepEqual(Array.from(context.runCandidates.segments), [0]);
});

test("the session snapshot carries the candidates", () => {
  const { context } = finishedRun();
  context.saveSessionState();
  context.runCandidates = context.runCandidatesCreate();

  assert.equal(context.restoreSessionState(), true);

  assert.deepEqual(
    Array.from(context.runCandidates.frames), [1, 3, 4]
  );
});

test("candidates over the quota give way to per-token detail", () => {
  // A default run's candidates are about 4 MiB. Where they tip the
  // snapshot over the quota, dropping them keeps the run savable
  // after a trip to Analytics; dropping the tokens would not.
  const { context } = finishedRun();
  const storage = context.sessionStorage;
  const write = storage.setItem.bind(storage);
  storage.setItem = (key, value) => {
    if (value.indexOf("\"candidates\"") !== -1) {
      throw new Error("QuotaExceededError");
    }
    write(key, value);
  };
  context.saveSessionState();
  context.runCandidates = context.runCandidatesCreate();

  assert.equal(context.restoreSessionState(), true);

  assert.ok(context.runCandidatesIsEmpty(context.runCandidates));
  assert.equal(context.runFrames.tokens.length, STEPS + 1);
});
