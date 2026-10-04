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
// opening frame and a resumed edit's first frame show nothing. On an
// edited run it opens, from the frame the edit branched at, on the
// run the crossfade favours, turns to the other, reads the original
// at the frame its layer clamps to, and stays closed where the
// favoured run has nothing. It also proves the candidates, the
// pre-edit run's among them, reach the save, land after the point a
// resume branched from, come back on Retry, and survive the snapshot
// a trip to Analytics depends on. With the candidates chosen, a
// finished run's unsettled positions cycle through them, each crossfade
// layer its own run's, and nothing cycles while a run streams,
// mid-edit, with motion reduced, or for the other two choices.

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
  page.generatorSocketController().connect();
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

// The popover as a hover opens it for `position` with the scrubber at
// `frame`. The stub keeps children when text is cleared, so it starts
// empty.
function popoverAt(context, registry, frame, position) {
  const popover = registry.get("token-alts-popover");
  context.navigateToFrame(frame);
  popover.children = [];
  context.showAltsPopover(position, null);
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

// -- an edited run pages between the two runs --

const EDIT_AT_2 = { frame_index: 2, token_positions: [1] };

// A finished run edited at frame 2 and resumed for `frames` frames,
// whose capture kept `captured` of its own frames, or none when null
// (a guided Run to Here), then finished again. The original captured
// frames 1, 3 and 4, so its sets lead with 301 at frame 3; the
// resume's frame 3 is its own frame 1, whose sets lead with 101.
function editedRun(options) {
  const settings = Object.assign(
    { frames: 3, captured: [1, 2] }, options || {}
  );
  const run = finishedRun();
  run.context.remaskEdits = [EDIT_AT_2];
  resumeFrom(run.context, 2, settings.frames);
  if (settings.captured !== null) {
    run.context.handleMessage(candidatesMessage(settings.captured));
  }
  run.context.handleDone({ type: "done", final_text: WORDS.join("") });
  return run;
}

function titleOf(popover) {
  return withClass(popover, "alt-heading")[0].children[0].textContent;
}

function pagerTo(popover, label) {
  return withClass(popover, "alt-pager-btn").find(
    (button) => button.getAttribute("aria-label") === label + " run"
  );
}

function chosenId(popover) {
  const chosen = withClass(popover, "alt-row-chosen");
  assert.equal(chosen.length, 1);
  return Number(chosen[0].getAttribute("data-alt-id"));
}

test("an edited run opens on the run the crossfade favours", () => {
  const { context, registry } = editedRun();

  context.runBlend = 0.2;
  const original = popoverAt(context, registry, 3, 1);
  assert.equal(titleOf(original), "Position 2: Original");
  assert.deepEqual(rowIds(original), [301, 7]);

  context.runBlend = 0.8;
  const edited = popoverAt(context, registry, 3, 1);
  assert.equal(titleOf(edited), "Position 2: Edited");
  assert.deepEqual(rowIds(edited), [101, 7]);
});

test("the pager turns to the other run, marking its own token", () => {
  const { context, registry } = editedRun();
  context.runBlend = 0.2;
  const popover = popoverAt(context, registry, 3, 1);
  assert.equal(pagerTo(popover, "Original").disabled, true);
  assert.equal(chosenId(popover), 301);
  const toEdited = pagerTo(popover, "Edited");

  popover.children = [];
  toEdited.dispatch("click", { stopPropagation() {} });

  assert.equal(titleOf(popover), "Position 2: Edited");
  assert.equal(chosenId(popover), 101);
});

test("before the edit there is one run, so no pager", () => {
  const { context, registry } = editedRun();
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 1, 1);

  assert.equal(titleOf(popover), "Position 2: candidates");
  assert.equal(withClass(popover, "alt-pager").length, 0);
});

test("the frame the edit branched at already has two runs", () => {
  // The edited run's frame 2 is its remasked canvas, with nothing
  // captured yet, so there is no pager to it, but the original's
  // frame 2 is its own and reads as of its step 1.
  const { context, registry } = editedRun();
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 2, 1);

  assert.equal(titleOf(popover), "Position 2: Original");
  assert.equal(stepLabel(popover), "As of step 1");
  assert.equal(withClass(popover, "alt-pager").length, 0);
});

test("the runs part at the earliest edit, whatever the order", () => {
  const { context, registry } = editedRun();
  context.remaskEdits = [
    { frame_index: 3, token_positions: [2] }, EDIT_AT_2,
  ];
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 2, 1);

  assert.equal(titleOf(popover), "Position 2: Original");
});

test("past its end, the Original page reads the original's last frame", () => {
  // The edit ran two frames longer than the run it replaced. The
  // crossfade's pre-edit layer holds the original's final frame
  // there, and the page names that frame as its own step.
  const { context, registry } = editedRun({ frames: 5 });
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 6, 1);

  assert.equal(stepLabel(popover), "Step 4");
  assert.deepEqual(rowIds(popover), [401, 7]);
});

test("where the favoured run has nothing, the popover stays closed", () => {
  // The original's candidates would describe tokens that are not the
  // ones under the pointer.
  const { context, registry } = editedRun({ captured: null });
  context.runBlend = 0.8;

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(popover.hidden, true);
});

test("an edited run without its baseline has one page", () => {
  // A quota-light restore carries no baseline token detail, so there
  // is only one run on screen even though the edit log survives.
  const { context, registry } = editedRun();
  context.saveSessionState();
  const storage = context.sessionStorage;
  const stored = JSON.parse(
    storage.getItem(context.SESSION_KEY)
  );
  delete stored.originalFrameTokens;
  delete stored.originalFrameHistory;
  storage.setItem(context.SESSION_KEY, JSON.stringify(stored));
  context.generatorRun.reset();
  assert.equal(context.restoreSessionState(), true);
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(titleOf(popover), "Position 2: candidates");
  assert.deepEqual(rowIds(popover), [101, 7]);
});

test("a page with nothing to turn to has no pager", () => {
  const { context, registry } = editedRun({ captured: null });
  context.runBlend = 0.2;

  const popover = popoverAt(context, registry, 3, 1);

  assert.equal(titleOf(popover), "Position 2: Original");
  assert.equal(withClass(popover, "alt-pager").length, 0);
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
  page.generatorSocketController().connect();
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  for (let index = 0; index <= STEPS; index++) {
    context.handleFrame(canvasFrame(index));
  }
  context.handleDone({ type: "done", final_text: WORDS.join("") });

  await context.saveRun();

  assert.equal("candidates" in saved[0], false);
});

test("an edited run's save carries the pre-edit candidates", async () => {
  const { context, saved } = editedRun();

  await context.saveRun();

  const original = saved[0].original_candidates;
  assert.deepEqual(original.frames, [1, 3, 4]);
  assert.deepEqual(original.segments, [0]);
  assert.equal(original.sets[1][1].h, 301);
  assert.deepEqual(saved[0].candidates.segments, [0, 2]);
});

test("an unedited run saves its candidates once", async () => {
  const { context, saved } = finishedRun();

  await context.saveRun();

  assert.equal("original_candidates" in saved[0], false);
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
    Array.from(
      context.generatorRun.candidateSegments(false)
    ),
    [0, 2]
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

test("a new run's Original page never shows the last run's", () => {
  // The pre-edit candidates are frozen when a run first finishes, so
  // a new run has to let go of the last one's to freeze its own.
  const { context, registry } = finishedRun();
  context.startGeneration();
  for (let index = 0; index <= STEPS; index++) {
    context.handleFrame(canvasFrame(index));
  }
  context.handleDone({ type: "done", final_text: WORDS.join("") });
  context.remaskEdits = [EDIT_AT_2];
  context.runBlend = 0.2;

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
    Array.from(context.generatorRun.candidateFrames(false)),
    [1, 3, 4]
  );
  assert.deepEqual(
    Array.from(context.generatorRun.candidateSegments(false)),
    [0]
  );
});

test("the session snapshot carries the candidates", () => {
  const { context } = finishedRun();
  context.saveSessionState();
  context.generatorRun.reset();

  assert.equal(context.restoreSessionState(), true);

  assert.deepEqual(
    Array.from(context.generatorRun.candidateFrames(false)),
    [1, 3, 4]
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
  context.generatorRun.reset();

  assert.equal(context.restoreSessionState(), true);

  assert.equal(context.generatorRun.candidatesEmpty(false), true);
  assert.equal(
    context.generatorRun.frameTokenSeries().length,
    STEPS + 1
  );
});

test("the snapshot keeps both runs' candidates", () => {
  const { context } = editedRun();
  context.saveSessionState();
  context.generatorRun.reset();

  assert.equal(context.restoreSessionState(), true);

  assert.deepEqual(
    Array.from(context.generatorRun.candidateFrames(true)),
    [1, 3, 4]
  );
  assert.deepEqual(
    Array.from(context.generatorRun.candidateSegments(true)),
    [0]
  );
  assert.deepEqual(
    Array.from(
      context.generatorRun.candidateSegments(false)
    ),
    [0, 2]
  );
});

test("an unedited run's snapshot writes its candidates once", () => {
  // Until an edit the pre-edit run's candidates are the live run's,
  // and writing them twice could cost the quota the tokens.
  const { context } = finishedRun();
  context.saveSessionState();
  const written = JSON.parse(
    context.sessionStorage.getItem(context.SESSION_KEY)
  );
  context.generatorRun.reset();

  assert.equal("originalCandidates" in written, false);
  assert.equal(context.restoreSessionState(), true);
  assert.deepEqual(
    context.generatorRun.candidateRecord(true),
    context.generatorRun.candidateRecord(false)
  );
});

test("over the quota, the pre-edit candidates give way first", () => {
  // The Original page then shows nothing, rather than the edited
  // run's candidates under the original's tokens.
  const { context, registry } = editedRun();
  const storage = context.sessionStorage;
  const write = storage.setItem.bind(storage);
  storage.setItem = (key, value) => {
    if (value.indexOf("\"originalCandidates\"") !== -1) {
      throw new Error("QuotaExceededError");
    }
    write(key, value);
  };
  context.saveSessionState();

  assert.equal(context.restoreSessionState(), true);

  assert.deepEqual(
    Array.from(
      context.generatorRun.candidateSegments(false)
    ),
    [0, 2]
  );
  context.runBlend = 0.2;
  assert.equal(popoverAt(context, registry, 3, 1).hidden, true);
});

// -- the candidates cycle --

// What is cycling, as a plain array: one {span, position, texts,
// width} per position the flicker is stepping. The width is read
// before it stops, since stopping hands each span back as drawn.
function cyclingAt(context, frame) {
  context.navigateToFrame(frame);
  const entries = [...context.flickerEntries].map((entry) => ({
    span: entry.span,
    position: entry.position,
    texts: entry.texts,
    width: entry.span.style.width,
  }));
  context.flickerStop();
  return entries;
}

function chooseCandidates(context) {
  context.appSettings.unsettledShows = "candidates";
}

test("a finished run's unsettled positions cycle", () => {
  // At frame 3 only position 3 is unsettled. Its set is " lead" at
  // 0.6 and " seven" at 0.2, so a fifth of the cycle is the glyph.
  const { context } = finishedRun();
  chooseCandidates(context);

  const cycling = cyclingAt(context, 3);

  assert.deepEqual(cycling.map((entry) => entry.position), [3]);
  const texts = [...cycling[0].texts];
  assert.equal(texts.filter((t) => t === " lead").length, 12);
  assert.equal(texts.filter((t) => t === " seven").length, 4);
  assert.equal(texts.filter((t) => t === "\u2591").length, 4);
  assert.equal(cycling[0].width, "6ch");
});

test("a tick shows the slot the clock is in", () => {
  const { context } = finishedRun();
  chooseCandidates(context);
  context.navigateToFrame(3);
  const entry = context.flickerEntries[0];

  context.flickerNow = () => 0;
  context.flickerTick();

  assert.equal(
    entry.span.textContent,
    entry.texts[context.flickerSlotAt(0, 3)]
  );
  context.flickerStop();
});

test("the glyph and the guess never cycle", () => {
  const { context } = finishedRun();

  for (const choice of ["glyph", "guess"]) {
    context.appSettings.unsettledShows = choice;
    assert.equal(cyclingAt(context, 3).length, 0, choice);
  }
});

test("nothing cycles while a run streams", () => {
  // The candidates arrive as a run ends, so a streaming canvas shows
  // its guesses even with them chosen.
  const saved = [];
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: savingFetch(saved),
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  chooseCandidates(context);
  page.generatorSocketController().connect();
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();

  for (let index = 0; index <= 2; index++) {
    context.handleFrame(canvasFrame(index));
  }

  assert.equal(context.flickerEntries.length, 0);
});

test("nothing cycles mid-edit", () => {
  // The canvas is a click target there, not something to read.
  const { context } = finishedRun();
  chooseCandidates(context);
  context.beginEditSession();

  assert.equal(cyclingAt(context, 3).length, 0);
});

test("cycling needs a finished run the user is not editing", () => {
  // The streaming canvas never reaches the scrubbed path, so these
  // guards are what hold if some later path does.
  const { context } = finishedRun();
  chooseCandidates(context);
  assert.equal(context.candidatesCycle(), true);

  context.scrubberActive = false;
  assert.equal(context.candidatesCycle(), false);
  context.scrubberActive = true;
  context.isGenerating = true;
  assert.equal(context.candidatesCycle(), false);
  context.isGenerating = false;
  context.beginEditSession();
  assert.equal(context.candidatesCycle(), false);
});

test("nothing cycles with motion reduced", () => {
  const { context } = finishedRun();
  chooseCandidates(context);
  context.prefersReducedMotion = () => true;

  assert.equal(cyclingAt(context, 3).length, 0);
});

// The resume's candidates, with the lead renamed so each crossfade
// layer's source can be told apart.
function branchMessage(frames) {
  const message = candidatesMessage(frames);
  message.sets = message.sets.map((frame) => frame.map((set) => ({
    h: set.h,
    c: [Object.assign({}, set.c[0], { t: " branch" }), set.c[1]],
  })));
  return message;
}

test("each crossfade layer cycles its own run's candidates", () => {
  const { context } = finishedRun();
  context.remaskEdits = [EDIT_AT_2];
  resumeFrom(context, 2, 3);
  context.handleMessage(branchMessage([1, 2]));
  context.handleDone({ type: "done", final_text: WORDS.join("") });
  chooseCandidates(context);
  context.runBlend = 0.5;

  const cycling = cyclingAt(context, 3);

  assert.equal(cycling.length, 2);
  const original = [...cycling[0].texts];
  const edited = [...cycling[1].texts];
  assert.ok(original.includes(" lead"));
  assert.equal(original.includes(" branch"), false);
  assert.ok(edited.includes(" branch"));
  assert.equal(edited.includes(" lead"), false);
});
