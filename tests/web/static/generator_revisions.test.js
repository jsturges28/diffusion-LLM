// The generator's revisions: the Revisions overlay and the live
// revision glow, driven for real.
//
// Strategy: load the generator page into the DOM stub with a
// DiffusionGemma entry, start a real run through startGeneration and
// finish it with the frames a worker sends. The fixture is four
// positions over seven frames, shaped the way DiffusionGemma streams:
// a changed token reads as masked for a frame before it settles, and
// two positions settle, change and settle again. A LLaDA-shaped run,
// where every position settles once, is the control. Then scrub, pick
// the overlay, and read the picker, the colours, the strip and the
// legend; read the flashes each streamed frame marks; take an edit
// and a resume the way the page does.
//
// Passing proves the overlay is offered exactly when a run revised
// something, that each token is tinted by how often it had changed by
// the scrubbed frame, that the strip reads the same count, that the
// legend follows the selection, that each crossfade layer counts its
// own run, with a remasked position starting over, and that nothing
// is counted from a run still streaming. For the glow it proves a
// revision flashes cyan where a birth flashes white, that the setting
// and reduced motion each turn it off, that the two flashes share one
// capped queue in which revisions survive, that a newer flash on a
// span replaces an older one, and that the live check is rebuilt from
// the run's own frames, edits included, after any cut.

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

function settled(id) {
  return { t: " w" + id, m: false, id: id, c: 0.5 };
}

function changing(id) {
  return { t: " w" + id, m: true, id: id, c: 0.2 };
}

// Position 0 settles on 10, changes to 20, then to 30; position 1
// settles on 11 and changes to 21. Positions 2 and 3 settle once.
// Revised: position 0 at frames 3 and 6, position 1 at frame 5.
const REVISING = [
  { tokens: [changing(1), changing(2), changing(3), changing(4)],
    revealed: [] },
  { tokens: [settled(10), settled(11), changing(3), changing(4)],
    revealed: [0, 1] },
  { tokens: [changing(20), settled(11), settled(12), changing(4)],
    revealed: [2] },
  { tokens: [settled(20), settled(11), settled(12), settled(13)],
    revealed: [3] },
  { tokens: [settled(20), changing(21), settled(12), settled(13)],
    revealed: [] },
  { tokens: [changing(30), settled(21), settled(12), settled(13)],
    revealed: [] },
  { tokens: [settled(30), settled(21), settled(12), settled(13)],
    revealed: [] },
];

// The control: every position settles once and never moves, which
// is what LLaDA does.
const SETTLING = [
  { tokens: [changing(1), changing(2), changing(3), changing(4)],
    revealed: [] },
  { tokens: [settled(10), changing(2), changing(3), changing(4)],
    revealed: [0] },
  { tokens: [settled(10), settled(11), changing(3), changing(4)],
    revealed: [1] },
  { tokens: [settled(10), settled(11), settled(12), settled(13)],
    revealed: [2, 3] },
];

function frameOf(spec, index, total) {
  const tokens = spec.tokens.map((token) => Object.assign({}, token));
  return {
    type: "frame",
    index: index,
    total_steps: total,
    canvas_index: 0,
    mean_conf: 0.5,
    text: tokens.map((token) => token.t).join(""),
    tokens: tokens,
    revealed: spec.revealed.slice(),
    elapsed: +(index * 0.1).toFixed(2),
  };
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

// A page mid-run: started for real, with `specs` streamed. `prepare`
// adjusts the page once the run has started, before any frame lands.
function streaming(specs, prepare) {
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch,
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  page.generatorSocketController().connect();
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
  if (prepare) {
    prepare(context);
  }
  specs.forEach((spec, index) => {
    context.handleFrame(frameOf(spec, index, specs.length - 1));
  });
  return page;
}

function finishedRun(specs) {
  const page = streaming(specs);
  page.context.handleDone({ type: "done", final_text: "done" });
  return page;
}

// The values the overlay picker lists, as a finished run built it.
function pickerValues(context) {
  const mount = context.document.getElementById(
    "overlay-select-mount"
  );
  const select = mount.children[mount.children.length - 1];
  const list = select.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map((item) => item.getAttribute("data-value"));
}

// Every token span drawn into `element`, through the fragments the
// stub keeps as children rather than flattening.
function drawnSpans(element) {
  const spans = [];
  for (const child of element.children || []) {
    if (child.tag === "span") {
      spans.push(child);
    } else {
      spans.push(...drawnSpans(child));
    }
  }
  return spans;
}

function liveSpans(context) {
  const output = context.document.getElementById("output-area");
  const spans = drawnSpans(output);
  const count = context.generatorRun.frameTokensLast().length;
  return spans.slice(spans.length - count);
}

// -- when it is offered --

test("a revising run offers Revisions, after Commit Order", () => {
  const { context } = finishedRun(REVISING);

  const values = pickerValues(context);

  assert.equal(context.generatorCanvas.revisionsAvailable(), true);
  assert.equal(
    values.indexOf("revisions"), values.indexOf("commit") + 1
  );
});

test("a run that never revised is not offered it", () => {
  const { context } = finishedRun(SETTLING);

  assert.equal(context.generatorCanvas.revisionsAvailable(), false);
  assert.ok(!pickerValues(context).includes("revisions"));
});

test("a run that only appends never paints it", () => {
  // A stale selection from a diffusion run must not tint a run that
  // appends: there is no canvas for a position to change on.
  const { context } = finishedRun(REVISING);
  context.generatorCanvas.setOverlayMode("revisions");
  context.generatorModelPanel.configure({
    models: [{
      id: "append",
      display_name: "Append-only",
      capabilities: {
        generation_shape: "append_only",
        supported_devices: ["cpu"],
      },
      param_specs: [],
    }],
    active: "append",
    active_device: "cpu",
  });

  context.generatorCanvas.rebuildOverlaySelect();
  assert.equal(context.generatorCanvas.overlayMode(), "none");
  assert.equal(context.generatorCanvas.revisionsAvailable(), false);
});

// -- what it paints --

// The colours of the spans one render draws. The stub keeps children
// when text is cleared, so the output area starts empty.
function colorsDrawnBy(registry, render) {
  const output = registry.get("output-area");
  output.children = [];
  render();
  return drawnSpans(output).map((span) => span.style.color);
}

test("tokens are tinted by their count at the scrubbed frame", () => {
  const { context, registry } = finishedRun(REVISING);
  context.navigateToFrame(6);

  const colors = colorsDrawnBy(registry, () => {
    context.generatorCanvas.setOverlayMode("revisions");
  });

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("scrubbing back counts only what had happened by then", () => {
  const { context, registry } = finishedRun(REVISING);
  context.generatorCanvas.setOverlayMode("revisions");

  const colors = colorsDrawnBy(registry, () => {
    context.navigateToFrame(4);
  });

  assert.deepEqual(colors, [context.revisionColor(1), "", "", ""]);
});

test("the strip reads the count while the overlay is on", () => {
  const { context } = finishedRun(REVISING);
  context.navigateToFrame(6);
  const token = context.generatorRun.frameTokens(6)[0];

  context.generatorCanvas.setOverlayMode("revisions");
  assert.equal(
    context.generatorCanvas.tokenExtra(0, token, false),
    "Revisions: 2"
  );
  assert.equal(
    context.generatorCanvas.tokenExtra(2, token, false),
    ""
  );
  context.generatorCanvas.setOverlayMode("conf");
  assert.equal(
    context.generatorCanvas.tokenExtra(0, token, false),
    ""
  );
});

test("the legend shows only while Revisions is selected", () => {
  const { context, registry } = finishedRun(REVISING);
  const revision = registry.get("revision-legend");
  const commit = registry.get("commit-legend");

  context.generatorCanvas.setOverlayMode("revisions");
  assert.equal(revision.hidden, false);
  assert.equal(commit.hidden, true);

  context.generatorCanvas.setOverlayMode("commit");
  assert.equal(revision.hidden, true);
  assert.equal(commit.hidden, false);
});

test("a stale selection is dropped when the run has none", () => {
  const { context } = finishedRun(SETTLING);
  context.generatorCanvas.setOverlayMode("revisions");

  context.generatorCanvas.rebuildOverlaySelect();

  assert.equal(context.generatorCanvas.overlayMode(), "none");
});

test("nothing is counted while a run streams", () => {
  // A resume streams with the scrubber still on the frame it left,
  // which belongs to the run being replaced, so a hover there must
  // not read a count from it.
  const { context } = finishedRun(REVISING);
  context.navigateToFrame(6);
  context.generatorCanvas.setOverlayMode("revisions");
  assert.equal(context.generatorCanvas.revisionCount(0, false), 2);

  context.isGenerating = true;

  assert.equal(context.generatorCanvas.revisionCount(0, false), 0);
});

test("a memo taken while the run grew is not kept", () => {
  const { context } = streaming(REVISING.slice(0, 4));
  context.isGenerating = false;
  assert.equal(context.generatorCanvas.revisionCount(0, false), 1);

  context.isGenerating = true;
  REVISING.slice(4).forEach((spec, at) => {
    context.handleFrame(frameOf(spec, 4 + at, REVISING.length - 1));
  });

  context.isGenerating = false;
  assert.equal(context.generatorCanvas.revisionCount(0, false), 2);
});

// -- an edited run: each layer counts its own run --

// The branch resumed at frame 4 with position 2 remasked. Position 2
// settles on a new token, which is a birth because the user asked for
// it; position 3 then changes on its own, which is a revision. One
// frame longer than the run it replaced, so the original layer has to
// hold its last frame.
const BRANCH = [
  { tokens: [settled(20), settled(11), changing(40), settled(13)],
    revealed: [] },
  { tokens: [settled(20), settled(11), settled(40), changing(41)],
    revealed: [2] },
  { tokens: [settled(20), settled(11), settled(40), settled(41)],
    revealed: [] },
  { tokens: [settled(20), settled(11), settled(40), settled(41)],
    revealed: [] },
];

function editedRun() {
  const page = finishedRun(REVISING);
  const { context } = page;
  context.remaskEdits = [{ frame_index: 4, token_positions: [2] }];
  context.truncateRunArraysAt(4);
  context.generatorCanvas.invalidate();
  context.isResuming = true;
  BRANCH.forEach((spec, at) => {
    context.handleFrame(frameOf(spec, at, BRANCH.length));
  });
  context.handleDone({ type: "done", final_text: "done" });
  return page;
}

function countsAt(context, isOriginal) {
  return [0, 1, 2, 3].map(
    (index) => context.generatorCanvas.revisionCount(
      index, isOriginal
    )
  );
}

test("the branch counts its own run, the edit starting over", () => {
  const { context } = editedRun();
  context.navigateToFrame(6);

  // Position 0's change at frame 3 is shared history; position 2's
  // new token is the edit's, and position 3's is the model's own.
  assert.deepEqual(countsAt(context, false), [1, 0, 0, 1]);
});

test("the original layer counts the run it came from", () => {
  const { context } = editedRun();
  context.navigateToFrame(6);

  assert.deepEqual(countsAt(context, true), [2, 1, 0, 0]);
});

test("the original layer holds its last frame past its end", () => {
  // The branch outran the original by a frame. The crossfade draws
  // the original's last frame there, so that is what it counts.
  const { context } = editedRun();
  context.navigateToFrame(7);

  assert.deepEqual(countsAt(context, true), [2, 1, 0, 0]);
  assert.deepEqual(countsAt(context, false), [1, 0, 0, 1]);
});

test("each crossfade layer is painted from its own counts", () => {
  const { context, registry } = editedRun();
  context.navigateToFrame(6);
  context.generatorCanvas.setOverlayMode("revisions");
  registry.get("output-area").children = [];
  context.generatorCanvas.renderFrame(6);
  const colors = drawnSpans(
    registry.get("output-area")
  ).map((span) => span.style.color);

  assert.deepEqual(colors, [
    context.revisionColor(2),
    context.revisionColor(1),
    "",
    "",
    context.revisionColor(1),
    "",
    "",
    context.revisionColor(1),
  ]);
});

// -- the live glow --
//
// The stub links children through `parent` rather than `parentNode`,
// so the live view rebuilds its spans on every frame and an attribute
// set on one frame is never seen on the next. What a single frame
// marks is read off the stream; what happens to a span across frames
// is driven on a span directly.

function flashes(span) {
  return {
    born: span.hasAttribute("data-born"),
    revised: span.hasAttribute("data-revised"),
  };
}

const DARK = { born: false, revised: false };
const WHITE = { born: true, revised: false };
const CYAN = { born: false, revised: true };

// What matchMedia answers for a system that prefers reduced motion.
function prefersStill() {
  return { matches: true, addEventListener() {} };
}

test("a streamed revision flashes cyan, and a birth white", () => {
  const { context } = streaming(REVISING.slice(0, 4));
  const spans = liveSpans(context);

  assert.deepEqual(flashes(spans[0]), CYAN);
  assert.deepEqual(flashes(spans[3]), WHITE);
  assert.deepEqual(flashes(spans[1]), DARK);
});

test("a return to the token it held does not flash", () => {
  const returning = [
    { tokens: [changing(1)], revealed: [] },
    { tokens: [settled(10)], revealed: [0] },
    { tokens: [changing(20)], revealed: [] },
    { tokens: [settled(10)], revealed: [] },
  ];
  const { context } = streaming(returning);

  assert.deepEqual(flashes(liveSpans(context)[0]), DARK);
});

test("the setting off leaves revisions dark, births lit", () => {
  const { context } = streaming(REVISING.slice(0, 4), (page) => {
    page.appSettings.revisionGlow = false;
  });
  const spans = liveSpans(context);

  assert.equal(spans[0].hasAttribute("data-revised"), false);
  assert.equal(spans[3].hasAttribute("data-born"), true);
});

test("reduced motion flashes nothing", () => {
  const { context } = streaming(REVISING.slice(0, 4), (page) => {
    page.matchMedia = prefersStill;
  });
  const spans = liveSpans(context);

  assert.deepEqual(flashes(spans[0]), DARK);
  assert.deepEqual(flashes(spans[3]), DARK);
});

test("a full queue keeps the frame's revisions", () => {
  // The default cap is 48. This frame births 49 positions and then
  // marks a revision, so the oldest births are the ones discarded.
  const unsettled = Array.from(
    { length: 50 }, (_, index) => changing(index + 1)
  );
  const first = unsettled.slice();
  first[0] = settled(10);
  const changingAgain = first.slice();
  changingAgain[0] = changing(20);
  const final = Array.from(
    { length: 50 }, (_, index) => settled(index + 30)
  );
  const revealed = Array.from(
    { length: 49 }, (_, index) => index + 1
  );
  const { context } = streaming([
    { tokens: unsettled, revealed: [] },
    { tokens: first, revealed: [0] },
    { tokens: changingAgain, revealed: [] },
    { tokens: final, revealed: revealed },
  ]);
  const spans = liveSpans(context);
  const glowing = spans.filter((span) => (
    span.hasAttribute("data-born")
    || span.hasAttribute("data-revised")
  ));
  assert.deepEqual(flashes(spans[0]), CYAN);
  assert.deepEqual(flashes(spans[1]), DARK);
  assert.deepEqual(flashes(spans[49]), WHITE);
  assert.equal(glowing.length, 48);
});

test("a newer flash on a span replaces the older one", () => {
  const page = streaming([
    { tokens: [changing(1)], revealed: [] },
    { tokens: [settled(10)], revealed: [0] },
  ]);
  const span = liveSpans(page.context)[0];
  span.parent = page.registry.get("output-area");
  page.context.handleFrame(frameOf(
    { tokens: [changing(20)], revealed: [] }, 2, 3
  ));
  page.context.handleFrame(frameOf(
    { tokens: [settled(20)], revealed: [] }, 3, 3
  ));

  assert.deepEqual(flashes(span), CYAN);
});

test("a flash that ends leaves the queue", () => {
  const { context, registry } = streaming([
    { tokens: [changing(1)], revealed: [] },
    { tokens: [settled(10)], revealed: [0] },
  ]);
  const span = liveSpans(context)[0];
  const output = registry.get("output-area");

  output.dispatch("animationend", {
    animationName: "other",
    target: span,
  });
  assert.equal(span.hasAttribute("data-born"), true);

  output.dispatch("animationend", {
    animationName: "token-birth",
    target: span,
  });
  assert.equal(span.hasAttribute("data-born"), false);
});

test("a resume's remasked position is born live, not revised", () => {
  const page = finishedRun(REVISING);
  const { context } = page;
  context.remaskEdits = [{ frame_index: 4, token_positions: [2] }];
  context.truncateRunArraysAt(4);
  context.generatorCanvas.invalidate();
  context.isResuming = true;

  context.handleFrame(frameOf(BRANCH[0], 0, BRANCH.length));
  context.handleFrame(frameOf(BRANCH[1], 1, BRANCH.length));
  assert.deepEqual(flashes(liveSpans(context)[2]), WHITE);

  context.handleFrame(frameOf(BRANCH[2], 2, BRANCH.length));
  assert.deepEqual(flashes(liveSpans(context)[3]), CYAN);
});

test("the live revision fold stays private", () => {
  const { context } = streaming(REVISING);
  assert.equal(context.liveRevisionFold, undefined);
});

test("a controller cut rebuilds the fold", () => {
  // The fold read all seven frames, so position 0 last held 30.
  // Cutting through the run owner invalidates it; the next frame
  // settling on 20 matches what frame 3 held: not a revision.
  const { context } = finishedRun(REVISING);
  context.generatorRun.truncate(4);

  context.handleFrame(frameOf(REVISING[3], 4, REVISING.length - 1));

  assert.equal(
    liveSpans(context)[0].hasAttribute("data-revised"), false
  );
});

test("a rebuilt fold still starts a remasked position over", () => {
  // Position 2 is remasked at frame 4 and has not settled again when
  // the fold is dropped, so the rebuild has to apply the edit itself.
  const { context } = finishedRun(REVISING);
  context.remaskEdits = [{ frame_index: 4, token_positions: [2] }];
  context.truncateRunArraysAt(4);
  context.generatorCanvas.invalidate();
  context.handleFrame(frameOf(BRANCH[0], 0, BRANCH.length));

  context.generatorCanvas.invalidate();
  context.handleFrame(frameOf(BRANCH[1], 1, BRANCH.length));

  assert.equal(
    liveSpans(context)[2].hasAttribute("data-revised"), false
  );
});

test("a new run is checked against its own frames only", () => {
  // Position 0 ended the last run on 30. The new run settling it on
  // 10 is a birth, not a change from the old run's token.
  const page = finishedRun(REVISING);
  const { context, registry } = page;
  registry.get("prompt-input").value = "again";
  context.startGeneration();
  SETTLING.forEach((spec, index) => {
    context.handleFrame(frameOf(spec, index, SETTLING.length - 1));
  });

  assert.equal(
    liveSpans(context)[0].hasAttribute("data-revised"), false
  );
});
