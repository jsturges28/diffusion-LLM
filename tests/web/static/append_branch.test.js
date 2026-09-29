// An autoregressive branch keeps its comparison views.
//
// Strategy: load the generator page into the DOM stub, stream an
// append run the way SmolLM3 and Mamba-3 send one, finish it, then
// make a What If branch the way a substitution does: record the edit,
// cut the run back to the edited position, and stream the branch in
// its place. Then ask the page what a reader would see.
//
// Why: an append run's pre-edit baseline keeps positions and no
// per-frame token arrays, and the comparison views measured it by
// those arrays. From the append-only change on, an autoregressive
// branch had no crossfade, no diff and no original layer, and nothing
// failed. Passing proves the baseline reads as present, the crossfade
// is offered, both layers are drawn, the diff finds the substituted
// position, and the metrics strip reads a streaming append run.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const WORDS = ["The", " cat", " sat", " on", " the", " mat"];
const BRANCH = ["The", " dog", " ran", " off", " the", " path"];
const EDITED_POSITION = 1;

function appendFrame(index, words) {
  const position = index - 1;
  return {
    type: "frame",
    shape: "append",
    index: index,
    total_steps: words.length,
    canvas_index: 0,
    mean_conf: 0.5,
    token: {
      t: words[position],
      m: false,
      id: 1000 + position + (words === BRANCH ? 500 : 0),
      c: 0.5,
      e: 1.2,
    },
    revealed: [position],
    elapsed: +(index * 0.1).toFixed(2),
  };
}

function branchedRun() {
  const page = loadPage({});
  const { context } = page;
  for (let index = 1; index <= WORDS.length; index++) {
    context.handleFrame(appendFrame(index, WORDS));
  }
  context.handleDone({ type: "done", final_text: WORDS.join("") });
  // What a substitution does before its stream arrives.
  context.remaskEdits.push({
    frame_index: EDITED_POSITION,
    token_positions: [EDITED_POSITION],
  });
  context.truncateRunArraysAt(EDITED_POSITION);
  context.isResuming = true;
  const first = EDITED_POSITION + 1;
  for (let index = first; index <= BRANCH.length; index++) {
    context.handleFrame(appendFrame(index, BRANCH));
  }
  context.handleDone({ type: "done", final_text: BRANCH.join("") });
  context.activateScrubber();
  return page;
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

test("an append branch has a baseline to compare against", () => {
  const { context } = branchedRun();

  assert.equal(context.originalRun.tokens.length, 0);
  assert.equal(context.diffAvailable(), true);
});

test("the crossfade is offered once the branch exists", () => {
  const { context, registry } = branchedRun();

  assert.equal(context.runBlendActive(), true);
  assert.equal(registry.get("run-blend-row").hidden, false);
});

test("both runs are drawn as layers, the original in full", () => {
  const { context } = branchedRun();
  const last = WORDS.length - 1;
  context.outputArea.children = [];

  context.renderFrameWithTokens(last);

  const layers = descendants(context.outputArea).filter(
    (node) => node.classes.has("token-layer")
  );
  const original = layers.find(
    (node) => node.classes.has("token-layer-original")
  );
  assert.equal(layers.length, 2);
  assert.ok(original, "no original layer was drawn");
  const spans = descendants(original).filter(
    (node) => node.classes.has("token-span")
  );
  assert.equal(spans.length, WORDS.length);
});

test("the diff overlay draws the original layer too", () => {
  const { context } = branchedRun();
  context.overlayMode = "diff";
  context.outputArea.children = [];

  context.renderFrameWithTokens(WORDS.length - 1);

  const spans = descendants(context.outputArea).filter(
    (node) => node.classes.has("token-span")
  );
  assert.equal(spans.length, WORDS.length + BRANCH.length);
});

test("hovering the original layer reads the baseline", () => {
  const { context } = branchedRun();
  context.navigateToFrame(WORDS.length - 1);
  context.metricsHoverOriginal = true;

  const tokens = context.metricsFrameTokens();

  assert.ok(tokens, "the strip read nothing from the baseline");
  assert.equal(tokens[EDITED_POSITION].t, WORDS[EDITED_POSITION]);
});

test("the diff finds the substituted position", () => {
  const { context } = branchedRun();

  const diff = context.computeDiff();

  assert.ok(diff.changedCount >= 1, "the diff saw no change");
});

test("the metrics strip reads a streaming append run", () => {
  // While a run streams the scrubber is off, and the strip reads the
  // newest frame, which an append run keeps as positions.
  const { context } = loadPage({});
  for (let index = 1; index <= 3; index++) {
    context.handleFrame(appendFrame(index, WORDS));
  }

  const tokens = context.metricsFrameTokens();

  assert.ok(tokens, "the strip read nothing");
  assert.equal(tokens.length, 3);
});
