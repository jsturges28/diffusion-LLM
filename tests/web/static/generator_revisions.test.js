// The generator's Revisions overlay, driven for real.
//
// Strategy: load the generator page into the DOM stub with a
// DiffusionGemma entry, start a real run through startGeneration and
// finish it with the frames a worker sends. The fixture is four
// positions over seven frames, shaped the way DiffusionGemma streams:
// a changed token reads as masked for a frame before it settles, and
// two positions settle, change and settle again. A LLaDA-shaped run,
// where every position settles once, is the control. Then scrub, pick
// the overlay, and read the picker, the colours, the strip and the
// legend; take an edit and a resume the way the page does.
//
// Passing proves the overlay is offered exactly when a run revised
// something, that each token is tinted by how often it had changed by
// the scrubbed frame, that the strip reads the same count, that the
// legend follows the selection, that each crossfade layer counts its
// own run, with a remasked position starting over, and that nothing
// is counted from a run still streaming.

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

// A page mid-run: started for real, with `specs` streamed.
function streaming(specs) {
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: quietFetch,
    bootState: { ui_state: {}, models: MODELS },
  });
  const { context, registry } = page;
  context.ws = new OpenSocket("ws://test");
  registry.get("prompt-input").value = "explain yeast";
  context.startGeneration();
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
  const list = context.overlaySelect.children.find(
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

// -- when it is offered --

test("a revising run offers Revisions, after Commit Order", () => {
  const { context } = finishedRun(REVISING);

  const values = pickerValues(context);

  assert.equal(context.revisionsAvailable(), true);
  assert.equal(
    values.indexOf("revisions"), values.indexOf("commit") + 1
  );
});

test("a run that never revised is not offered it", () => {
  const { context } = finishedRun(SETTLING);

  assert.equal(context.revisionsAvailable(), false);
  assert.ok(!pickerValues(context).includes("revisions"));
});

test("a run that only appends never paints it", () => {
  // A stale selection from a diffusion run must not tint a run that
  // appends: there is no canvas for a position to change on.
  const { context } = finishedRun(REVISING);
  context.overlayMode = "revisions";
  context.activeModel = {
    capabilities: { generation_shape: "append_only" },
  };

  assert.equal(context.effectiveColorMode(), "none");
  assert.equal(context.revisionsAvailable(), false);
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
    context.setOverlayMode("revisions");
  });

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("scrubbing back counts only what had happened by then", () => {
  const { context, registry } = finishedRun(REVISING);
  context.setOverlayMode("revisions");

  const colors = colorsDrawnBy(registry, () => {
    context.navigateToFrame(4);
  });

  assert.deepEqual(colors, [context.revisionColor(1), "", "", ""]);
});

test("the strip reads the count while the overlay is on", () => {
  const { context } = finishedRun(REVISING);
  context.navigateToFrame(6);
  const token = context.runFrames.tokens[6][0];

  context.overlayMode = "revisions";
  assert.equal(context.metricsExtra(0, token), "Revisions: 2");
  assert.equal(context.metricsExtra(2, token), "");
  context.overlayMode = "conf";
  assert.equal(context.metricsExtra(0, token), "");
});

test("the legend shows only while Revisions is selected", () => {
  const { context, registry } = finishedRun(REVISING);
  const revision = registry.get("revision-legend");
  const commit = registry.get("commit-legend");

  context.setOverlayMode("revisions");
  assert.equal(revision.hidden, false);
  assert.equal(commit.hidden, true);

  context.setOverlayMode("commit");
  assert.equal(revision.hidden, true);
  assert.equal(commit.hidden, false);
});

test("a stale selection is dropped when the run has none", () => {
  const { context } = finishedRun(SETTLING);
  context.overlayMode = "revisions";

  context.buildOverlaySelect();

  assert.equal(context.overlayMode, "none");
});

test("nothing is counted while a run streams", () => {
  // A resume streams with the scrubber still on the frame it left,
  // which belongs to the run being replaced, so a hover there must
  // not read a count from it.
  const { context } = finishedRun(REVISING);
  context.navigateToFrame(6);
  context.overlayMode = "revisions";
  assert.equal(context.tokenRevisionCount(0, false), 2);

  context.isGenerating = true;

  assert.equal(context.tokenRevisionCount(0, false), 0);
});

test("a memo taken while the run grew is not kept", () => {
  const { context } = streaming(REVISING.slice(0, 4));
  assert.equal(context.revisionsFor(false).length, 4);

  REVISING.slice(4).forEach((spec, at) => {
    context.handleFrame(frameOf(spec, 4 + at, REVISING.length - 1));
  });

  const revisions = context.revisionsFor(false);
  assert.equal(revisions.length, REVISING.length);
  assert.deepEqual([...revisions[6]], [0]);
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
  context.invalidateRunMemos();
  context.isResuming = true;
  BRANCH.forEach((spec, at) => {
    context.handleFrame(frameOf(spec, at, BRANCH.length));
  });
  context.handleDone({ type: "done", final_text: "done" });
  return page;
}

function countsAt(context, isOriginal) {
  return [0, 1, 2, 3].map(
    (index) => context.tokenRevisionCount(index, isOriginal)
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
  const { context } = editedRun();
  context.navigateToFrame(6);
  context.overlayMode = "revisions";
  const token = settled(99);

  assert.equal(
    context.tokenColorAt(1, token, true), context.revisionColor(1)
  );
  assert.equal(context.tokenColorAt(1, token, false), null);
  assert.equal(
    context.tokenColorAt(3, token, false), context.revisionColor(1)
  );
  assert.equal(context.tokenColorAt(3, token, true), null);
});
