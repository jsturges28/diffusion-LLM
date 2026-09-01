// What the generator page does with a run's frames, driven for real.
//
// Strategy: load the whole generator page into the DOM stub beside
// this file, feed it the frames a worker would send, and read back
// what it retained, what it renders at each scrub position, and what
// it would post on save. Nothing is mocked between the frame arriving
// and the answer coming out.
//
// This is a **characterization** test. It was written against the
// full-snapshot protocol before any of it changed, so the numbers
// below are not a specification anyone reasoned out: they are what
// the page did on the day, recorded so that replacing per-token
// snapshots with an append stream can be shown to change nothing a
// user sees. A test written afterwards would only prove the new code
// agrees with itself.
//
// Passing proves the retained frames grow the way the page expects,
// that scrubbing to frame N yields exactly the tokens of frame N,
// that the pre-edit baseline freezes at first completion, and that
// the save payload carries one full record set per frame. Those are
// the four things the append work has to preserve.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// One AR frame as `_build_frame` in ar_sampler.py emits it: a full
// snapshot of every position decoded so far.
function snapshotFrame(index, words, options) {
  const settings = options || {};
  const tokens = words.slice(0, index).map((word, position) => ({
    t: word,
    m: false,
    id: 1000 + position,
    c: +(0.5 + position / 100).toFixed(4),
    e: +(1.5 - position / 100).toFixed(4),
  }));
  const frame = {
    type: "frame",
    index: index,
    total_steps: settings.totalSteps || words.length,
    canvas_index: 0,
    mean_conf: tokens.length
      ? +(tokens.reduce((sum, t) => sum + t.c, 0)
        / tokens.length).toFixed(4)
      : 0,
    text: tokens.map((t) => t.t).join(""),
    tokens: tokens,
    revealed: index > 0 ? [index - 1] : [],
    elapsed: +(index * 0.1).toFixed(2),
  };
  if (settings.alts) {
    frame.alts = [{ id: 42, t: " other", p: 0.2 }];
  }
  return frame;
}



const WORDS = ["The", " cat", " sat", " on", " the", " mat"];

// Drive a whole run through the page and hand back the context.
function runThrough(words, options) {
  const page = loadPage({});
  const context = page.context;
  const settings = options || {};
  for (let index = 1; index <= words.length; index++) {
    context.handleFrame(
      snapshotFrame(index, words, {
        totalSteps: words.length,
        alts: !!settings.alts,
      })
    );
  }
  return page;
}

function frames(context) {
  return context.runFrames;
}

// -- what the page retains while a run streams --

test("a run retains one frame per token", () => {
  const { context } = runThrough(WORDS);

  assert.equal(frames(context).history.length, WORDS.length);
  assert.equal(frames(context).tokens.length, WORDS.length);
});

test("frame N holds N tokens, which is the quadratic", () => {
  // The shape of the defect, written down. Six frames hold 21 token
  // records between them, and at 2,048 tokens the same rule gives
  // 2,098,176. An append stream has to make this row read 1,2,3...
  // in what arrives while this assertion still passes for what the
  // page can hand out.
  const { context } = runThrough(WORDS);

  const counts = frames(context).tokens.map((t) => t.length);

  assert.deepEqual(
    Array.from(counts), [1, 2, 3, 4, 5, 6]
  );
});

test("each retained frame's text is the run so far", () => {
  const { context } = runThrough(WORDS);

  const texts = frames(context).history;

  assert.equal(texts[0], "The");
  assert.equal(texts[2], "The cat sat");
  assert.equal(texts[5], "The cat sat on the mat");
});

test("the per-frame scalars stay aligned with the frames", () => {
  const { context } = runThrough(WORDS);
  const held = frames(context);

  assert.equal(context.runFramesAligned(held), true);
  assert.equal(held.elapsed.length, WORDS.length);
  assert.equal(held.revealed.length, WORDS.length);
  assert.equal(held.canvasIndex.length, WORDS.length);
});

test("elapsed carries the worker's own reading", () => {
  const { context } = runThrough(WORDS);

  assert.deepEqual(
    Array.from(frames(context).elapsed),
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  );
});

test("candidates are kept by position, not by frame", () => {
  // Already linear, and it stays that way. Worth pinning because a
  // reader could reasonably assume alts ride the frames.
  const { context } = runThrough(WORDS, { alts: true });

  assert.equal(context.positionAlts.length, WORDS.length);
  assert.equal(context.positionAlts[3][0].t, " other");
});

// -- what scrubbing reads back --

test("scrubbing to frame N reads exactly N tokens", () => {
  // The property the append store must preserve. Every consumer of a
  // scrub position asks for a full token array at that frame, so
  // whatever backs it has to answer the same way.
  const { context } = runThrough(WORDS);

  for (let index = 0; index < WORDS.length; index++) {
    const at = frames(context).tokens[index];
    assert.equal(at.length, index + 1, `frame ${index}`);
    assert.equal(at[index].t, WORDS[index], `frame ${index} newest`);
  }
});

test("a scrubbed frame's tokens are the run's own prefix", () => {
  // Why an append stream is sound here: frame N is not a rewrite of
  // frame N-1, it is frame N-1 plus one. Nothing before the newest
  // position ever changes, which is exactly what diffusion does not
  // guarantee and why it keeps snapshots.
  const { context } = runThrough(WORDS);
  const held = frames(context).tokens;

  for (let index = 1; index < WORDS.length; index++) {
    const earlier = held[index - 1];
    const later = held[index].slice(0, index);
    assert.deepEqual(
      earlier.map((t) => t.t), later.map((t) => t.t),
      `frame ${index} changed a settled position`
    );
  }
});

test("navigating sets the scrub position the renderers read", () => {
  const { context } = runThrough(WORDS);
  context.activateScrubber();

  context.navigateToFrame(2);

  assert.equal(context.currentScrubFrame, 2);
});

// -- the pre-edit baseline --

test("the baseline freezes the run at first completion", () => {
  const { context } = runThrough(WORDS);

  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  assert.equal(context.originalRun.totalFrames, WORDS.length);
  assert.equal(context.originalRun.tokens.length, WORDS.length);
});

test("the baseline is a second copy of the same quadratic", () => {
  // Why an edited run pays twice: original_tokens.json is nearly as
  // large as tokens.json on disk.
  const { context } = runThrough(WORDS);
  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  const counts = context.originalRun.tokens.map((t) => t.length);

  assert.deepEqual(Array.from(counts), [1, 2, 3, 4, 5, 6]);
});

test("the baseline does not move when the live run is cut", () => {
  const { context } = runThrough(WORDS);
  context.originalRunCapture(
    context.originalRun, context.runFrames, context.positionAlts
  );

  context.runFramesTruncate(context.runFrames, 3);

  assert.equal(context.originalRun.totalFrames, WORDS.length);
  assert.equal(context.runFrames.tokens.length, 3);
});

// -- what a save would carry --

test("the save payload carries one record set per frame", () => {
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);

  assert.equal(records.length, WORDS.length);
  assert.deepEqual(
    Array.from(records.map((r) => r.length)), [1, 2, 3, 4, 5, 6]
  );
});

test("a saved record keeps text, id, confidence and entropy", () => {
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);
  const first = records[0][0];

  assert.equal(first.t, "The");
  assert.equal(first.id, 1000);
  assert.equal(first.m, false);
  assert.equal(typeof first.c, "number");
  assert.equal(typeof first.e, "number");
});

test("the payload's totals match the retained run", () => {
  // The invariant the server-side expansion has to reproduce: as
  // many text frames as token frames, each the right length.
  const { context } = runThrough(WORDS);

  const records = context.tokenRecordsFrom(context.runFrames.tokens);
  const texts = context.runFrames.history;

  assert.equal(records.length, texts.length);
  for (let index = 0; index < texts.length; index++) {
    assert.equal(
      records[index].map((r) => r.t).join(""), texts[index],
      `frame ${index} text disagrees with its tokens`
    );
  }
});

// -- truncation, which is how an edit rewinds --

test("cutting the run shortens every array together", () => {
  const { context } = runThrough(WORDS);

  context.runFramesTruncate(context.runFrames, 3);
  const held = frames(context);

  assert.equal(context.runFramesAligned(held), true);
  assert.equal(held.history.length, 3);
  assert.equal(held.tokens.length, 3);
});

test("a cut run continues from where it was cut", () => {
  // Substitution's shape: truncate to the edit point, then take new
  // frames from the worker. The append path has to behave the same,
  // since the seed frame becomes one appended token.
  const { context } = runThrough(WORDS);
  context.runFramesTruncate(context.runFrames, 3);

  context.handleFrame(snapshotFrame(4, [
    "The", " cat", " sat", " down",
  ]));

  const held = frames(context);
  assert.equal(held.tokens.length, 4);
  assert.equal(held.tokens[3][3].t, " down");
  assert.equal(held.history[3], "The cat sat down");
});
