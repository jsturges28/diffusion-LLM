// The session snapshot fits the desktop app's storage.
//
// Strategy: load the generator page into the DOM stub with a LLaDA
// entry, and give it session storage that refuses what the desktop
// app's engine refused when measured on 2026-09-29, counted across
// every key the page writes. Then stream it a default-sized run, 160
// positions over 128 steps with five candidates at every step, built
// to the proportions of a real run saved that day; finish it, and
// restore the snapshot it wrote. The same again after an edit at
// frame 56, which gives the snapshot a second store to carry.
//
// Passing proves a default LLaDA run's candidates, and both runs'
// after an edit, survive a trip to Analytics in the desktop app. Its
// session storage holds about 5.2 million characters, and before the
// candidates were packed the snapshot came to 6.2 million unedited
// and 10 million edited, so the popover came back empty every time.
// The fixture is held to the size of the run it models, so this
// cannot pass by being small.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

// The longest value QtWebEngine 6.11 (Chromium 140), the desktop
// app's engine, accepted in session storage on 2026-09-29: 10 MiB at
// two bytes a character, whatever the text.
const DESKTOP_SESSION_STORAGE_CHARS = 5236815;

// The run the fixture models, a default LLaDA run saved that day, as
// the page's snapshot wrote it: 2.40 million characters without its
// candidates, and 1.04 million for each packed store.
const MODELLED_FRAMES_CHARS = 2400000;
const MODELLED_STORE_CHARS = 1000000;

const POSITIONS = 160;
const STEPS = 128;
const EDIT_FRAME = 56;
const MASK_ID = 126336;
// That run's candidates drew on about 620 distinct ids, mostly of
// three and four digits, and one set in about 40 appended its held
// token from outside the five.
const ID_POOL = 640;
const RANKED_EVERY = 40;

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

function fetchModels(url) {
  const path = String(url).split("?")[0];
  const body = path.startsWith("/api/models") ? MODELS : {};
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

// A pool entry's id and text, one text per id as the worker decodes.
// 1571 and 9900 share no factor, so the 640 ids are distinct.
function poolId(entry) {
  return 100 + ((entry * 1571) % 9900);
}

function poolText(entry) {
  return " w" + (entry % 997);
}

// Frame `index` of a canvas settling left to right, `variant` picking
// each position's token so the edited run differs from the original.
// Masked positions show the model's guess and carry its confidence,
// as with Reveal the mask candidate on; values sit on the four-place
// grid the worker rounds to.
function frameAt(index, variant) {
  const settled = Math.round((index * POSITIONS) / STEPS);
  const tokens = [];
  for (let position = 0; position < POSITIONS; position += 1) {
    const entry = (position * 7 + variant) % ID_POOL;
    tokens.push({
      t: poolText(entry),
      m: position >= settled,
      id: position >= settled ? MASK_ID : poolId(entry),
      c: ((position * 37 + index * 11) % 9999 + 1) / 10000,
      e: ((position * 53 + index * 13) % 30000 + 1) / 10000,
    });
  }
  return {
    type: "frame",
    index: index,
    total_steps: STEPS,
    canvas_index: 0,
    mean_conf: 0.5,
    text: tokens.map((token) => token.t).join(""),
    tokens: tokens,
    revealed: [],
    elapsed: index * 0.036,
  };
}

// One step's sets: five candidates from the pool, the lead above a
// half and the rest small, on the four-place grid; and in one set in
// RANKED_EVERY the held token appended from outside the five, with
// its rank and the unrounded probability the worker sends for it.
function setsAt(frame, variant) {
  const sets = [];
  for (let position = 0; position < POSITIONS; position += 1) {
    const base = position + frame + variant;
    const c = [];
    for (let rank = 0; rank < 5; rank += 1) {
      const entry = (base + rank * 97) % ID_POOL;
      c.push({
        id: poolId(entry),
        t: poolText(entry),
        p: rank === 0
          ? (5000 + ((base * 7) % 5000)) / 10000
          : (((base * 3 + rank * 17) % 900) + 1) / 10000,
      });
    }
    let held = c[0].id;
    if (base % RANKED_EVERY === 0) {
      const entry = (base + 300) % ID_POOL;
      held = poolId(entry);
      c.push({
        id: held, t: poolText(entry), p: 1 / (3000 + base),
        rank: 6 + (base % 400),
      });
    }
    sets.push({ h: held, c: c });
  }
  return sets;
}

// The candidates message for frames 1 to `count` of a stream that
// began at `offset`, as the worker sends them: local to the stream.
function candidatesMessage(count, offset, variant) {
  const frames = [];
  for (let frame = 1; frame <= count; frame += 1) {
    frames.push(frame);
  }
  return {
    type: "candidates",
    k: 5,
    stride: 1,
    frames: frames,
    sets: frames.map((frame) => setsAt(offset + frame, variant)),
  };
}

// The generator page, its session storage held to the desktop app's
// quota across every key, as the engine counts it.
function desktopPage() {
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: fetchModels,
    bootState: { ui_state: {}, models: MODELS },
  });
  const storage = page.context.sessionStorage;
  const write = storage.setItem.bind(storage);
  const remove = storage.removeItem.bind(storage);
  const sizes = new Map();
  storage.setItem = (key, value) => {
    let others = 0;
    for (const [name, size] of sizes) {
      others += name === key ? 0 : size;
    }
    const size = key.length + String(value).length;
    if (others + size > DESKTOP_SESSION_STORAGE_CHARS) {
      throw new Error("QuotaExceededError");
    }
    sizes.set(key, size);
    write(key, value);
  };
  storage.removeItem = (key) => {
    sizes.delete(key);
    remove(key);
  };
  page.context.ws = new OpenSocket("ws://test");
  page.registry.get("prompt-input").value = "explain yeast";
  return page;
}

function generate(context) {
  context.startGeneration();
  for (let index = 0; index <= STEPS; index += 1) {
    context.handleFrame(frameAt(index, 0));
  }
  context.handleMessage(candidatesMessage(STEPS, 0, 0));
  context.handleDone({ type: "done", final_text: "generated" });
}

// Remask at EDIT_FRAME and resume to the end, as Edit Frames does.
function edit(context) {
  context.remaskEdits = [
    { frame_index: EDIT_FRAME, token_positions: [1, 2, 3] },
  ];
  context.truncateRunArraysAt(EDIT_FRAME);
  context.isResuming = true;
  for (let index = EDIT_FRAME; index <= STEPS; index += 1) {
    context.handleFrame(frameAt(index, 1));
  }
  context.handleMessage(
    candidatesMessage(STEPS - EDIT_FRAME, EDIT_FRAME, 1)
  );
  context.handleDone({ type: "done", final_text: "edited" });
}

// What a trip to Analytics and back leaves: the stores as the
// snapshot restores them.
function restore(context) {
  context.runCandidates = context.runCandidatesCreate();
  context.originalCandidates = null;
  assert.equal(context.restoreSessionState(), true);
}

// A store in full, as a save would send it, so "kept" means every
// set came back as it was rather than merely that frames did.
function storeText(context, store) {
  return JSON.stringify(context.runCandidatesToJson(store));
}

test("a default-sized run keeps its candidates", () => {
  const { context } = desktopPage();
  generate(context);
  const before = storeText(context, context.runCandidates);

  restore(context);

  assert.equal(context.runCandidates.frames.length, STEPS);
  assert.equal(storeText(context, context.runCandidates), before);
});

test("an edited one keeps both runs' candidates", () => {
  const { context } = desktopPage();
  generate(context);
  edit(context);
  const live = storeText(context, context.runCandidates);
  const original = storeText(context, context.originalCandidates);

  restore(context);

  assert.deepEqual(
    Array.from(context.runCandidates.segments), [0, EDIT_FRAME]
  );
  assert.equal(context.originalCandidates.frames.length, STEPS);
  assert.equal(storeText(context, context.runCandidates), live);
  assert.equal(
    storeText(context, context.originalCandidates), original
  );
});

test("the fixture is the size of the run it models", () => {
  const { context } = desktopPage();
  generate(context);
  edit(context);

  const written = context.sessionStorage.getItem(context.SESSION_KEY);
  const snapshot = JSON.parse(written);
  const live = JSON.stringify(snapshot.candidates).length;
  const original = JSON.stringify(snapshot.originalCandidates).length;

  const frames = written.length - live - original;
  assert.ok(frames >= MODELLED_FRAMES_CHARS);
  assert.ok(live >= MODELLED_STORE_CHARS);
  assert.ok(original >= MODELLED_STORE_CHARS);
});
