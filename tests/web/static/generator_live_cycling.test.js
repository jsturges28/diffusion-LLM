// DiffusionGemma's candidates cycling while the run streams.
//
// Strategy: load the generator page into the DOM stub with a
// DiffusionGemma entry, start a real run through startGeneration, and
// stream frames shaped as its worker sends them: each draft carries,
// under `live_candidates`, the sets of the positions that changed on
// it. Then read what the flicker is stepping after each frame. Frames
// that carry no sets, as LLaDA's never do, are the control.
//
// Passing proves that while a run streams, exactly the positions a
// frame changed and sent sets for cycle, that the next frame replaces
// them and hands the previous spans back, that the other two
// choices, reduced motion and frames without sets cycle nothing,
// that a payload which does not fit the canvas is ignored rather
// than cycled at the wrong place, and that a resume's frames cycle
// the same way.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

// The real WebSocket carries its states as statics and the page
// compares against them; the shared stub leaves them off.
class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const DGEMMA = {
  id: "dgemma",
  display_name: "DiffusionGemma-26B-A4B",
  min_vram_gib: 18,
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
  models: [DGEMMA],
  active: "dgemma",
  active_device: "cuda",
  active_tokenizer: { name: "dgemma" },
  active_context_length: 4096,
  default: "dgemma",
  gpu_name: "NVIDIA GeForce RTX 4090",
};

function settled(id, t) {
  return { t: t, m: false, id: id, c: 0.6 };
}

function changing(id, t) {
  return { t: t, m: true, id: id, c: 0.2 };
}

// A set as the worker sends it: the held token first, as the
// draft's argmax always is, then the rest of the candidates.
function set(rows) {
  return {
    h: rows[0][0],
    c: rows.map(([id, t, p]) => ({ id: id, t: t, p: p })),
  };
}

const THE = set([[1, " the", 0.5], [2, " a", 0.3]]);
const CAT = set([[3, " cat", 0.4], [4, " dog", 0.4]]);
const SAT = set([[5, " sat", 0.6], [6, " ran", 0.2]]);
const DOWN = set([[7, " down", 0.5], [8, " off", 0.25]]);
const RAN = set([[6, " ran", 0.55], [5, " sat", 0.3]]);
const OFF = set([[8, " off", 0.45], [9, " away", 0.35]]);
const AWAY = set([[9, " away", 0.5], [8, " off", 0.4]]);

// The opening draft changes every position, the next two settle
// them in turn; each sends the sets of the positions it changed.
const DRAFTS = [
  {
    tokens: [
      changing(1, " the"), changing(3, " cat"),
      changing(5, " sat"), changing(7, " down"),
    ],
    live: { positions: [0, 1, 2, 3], sets: [THE, CAT, SAT, DOWN] },
  },
  {
    tokens: [
      settled(1, " the"), settled(3, " cat"),
      changing(6, " ran"), changing(8, " off"),
    ],
    live: { positions: [2, 3], sets: [RAN, OFF] },
  },
  {
    tokens: [
      settled(1, " the"), settled(3, " cat"),
      settled(6, " ran"), changing(9, " away"),
    ],
    live: { positions: [3], sets: [AWAY] },
  },
];

function frameOf(spec, index, total) {
  const tokens = spec.tokens.map((token) => Object.assign({}, token));
  const frame = {
    type: "frame",
    index: index,
    total_steps: total,
    canvas_index: 0,
    mean_conf: 0.5,
    text: tokens.map((token) => token.t).join(""),
    tokens: tokens,
    revealed: [],
    elapsed: +(index * 0.9).toFixed(2),
  };
  if (spec.live !== undefined) {
    frame.live_candidates = JSON.parse(JSON.stringify(spec.live));
  }
  return frame;
}

function quietFetch(url) {
  const path = String(url).split("?")[0];
  const body = path.startsWith("/api/models") ? MODELS : {};
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

// What loadSettings would have written for a profile that chose
// `choice`, set once the run has started.
function choose(choice) {
  return (page) => {
    page.appSettings.unsettledShows = choice;
    page.LIVE_TOKEN_OPTIONS.revealMask = choice !== "glyph";
  };
}

// What matchMedia answers for a system that prefers reduced motion.
function prefersStill() {
  return { matches: true, addEventListener() {} };
}

// A page mid-run: started for real, `prepare` applied, then `specs`
// streamed.
function streaming(specs, prepare) {
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch,
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  (prepare || choose("candidates"))(context);
  specs.forEach((spec, index) => {
    context.handleFrame(frameOf(spec, index, specs.length));
  });
  return page;
}

// The positions the flicker is stepping now, in its order.
function cycling(context) {
  return [...context.flickerEntries].map((entry) => entry.position);
}

// -- what cycles --

test("a streaming frame cycles the positions it changed", () => {
  // In the flicker's order, least decided first: " off" holds 0.45
  // at position 3, " ran" 0.55 at position 2.
  const { context } = streaming(DRAFTS.slice(0, 2));

  assert.deepEqual(cycling(context), [3, 2]);
  assert.notEqual(context.flickerTimer, null);
  const span = context.liveTokenSpans[2];
  assert.equal("data-cycling" in span.attributes, true);
  context.flickerStop();
});

test("the opening frame cycles every position it sent", () => {
  // Ordered by contest: " cat" and " dog" tie at 0.4, the least
  // decided, so position 1 comes first.
  const { context } = streaming(DRAFTS.slice(0, 1));

  assert.deepEqual(cycling(context), [1, 0, 3, 2]);
  context.flickerStop();
});

test("the next frame replaces what cycled, and hands it back", () => {
  const { context } = streaming(DRAFTS.slice(0, 2));
  const before = [...context.flickerEntries].map((e) => e.span);

  context.handleFrame(frameOf(DRAFTS[2], 2, DRAFTS.length));

  assert.deepEqual(cycling(context), [3]);
  for (const span of before) {
    assert.equal("data-cycling" in span.attributes, false);
    assert.equal(span.style.width, "");
  }
  context.flickerStop();
});

test("a changed position without a set does not cycle", () => {
  const partial = {
    tokens: DRAFTS[1].tokens,
    live: { positions: [3], sets: [OFF] },
  };
  const { context } = streaming([DRAFTS[0], partial]);

  assert.deepEqual(cycling(context), [3]);
  context.flickerStop();
});

// -- what does not --

test("frames that carry no sets cycle nothing", () => {
  // LLaDA's frames never carry them.
  const bare = DRAFTS.map((spec) => ({ tokens: spec.tokens }));
  const { context } = streaming(bare);

  assert.deepEqual(cycling(context), []);
  assert.equal(context.flickerTimer, null);
});

test("the glyph and the guess never cycle live", () => {
  for (const choice of ["glyph", "guess"]) {
    const { context } = streaming(DRAFTS.slice(0, 2), choose(choice));

    assert.deepEqual(cycling(context), [], choice);
  }
});

test("reduced motion holds every position still", () => {
  const { context } = streaming(DRAFTS.slice(0, 2), (page) => {
    choose("candidates")(page);
    page.matchMedia = prefersStill;
  });

  assert.deepEqual(cycling(context), []);
});

test("a payload that does not fit the canvas is ignored", () => {
  const broken = [
    { positions: [2, 3], sets: [RAN] },
    { positions: [2, 9], sets: [RAN, OFF] },
    { positions: ["2"], sets: [RAN] },
    { positions: [2] },
    "not a payload",
  ];
  for (const live of broken) {
    const spec = { tokens: DRAFTS[1].tokens, live: live };
    const { context } = streaming([DRAFTS[0], spec]);

    assert.deepEqual(cycling(context), [], JSON.stringify(live));
  }
});

// -- a resume --

test("a resume's frames cycle the same way", () => {
  const page = streaming(DRAFTS);
  const { context } = page;
  context.handleDone({ type: "done", final_text: "done" });
  context.remaskEdits = [{ frame_index: 1, token_positions: [2, 3] }];
  context.truncateRunArraysAt(1);
  context.invalidateRunMemos();
  context.isResuming = true;

  context.handleFrame(frameOf(DRAFTS[1], 0, 2));

  assert.deepEqual(cycling(context), [3, 2]);
  context.flickerStop();
});
