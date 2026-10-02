// Tests for overlaysEntropyFrame, the rule every entropy view uses to
// decide which frame's entropy describes the frame on screen.
//
// Strategy: load the shipped overlays.js into a fresh vm context and
// call the helper over hand-built frames shaped like DiffusionGemma
// streams them: drafts whose positions carry `e`, and commits that
// carry none.
//
// A DiffusionGemma run ends on a committed canvas, and a commit has
// no entropy of its own, so a view that read only the frame on screen
// found nothing at the frame a finished run opens on. Passing proves
// a frame with entropy answers for itself, a commit borrows its
// canvas's last draft, the borrow never reaches into another canvas,
// an append stream is never searched, and a run with no entropy
// anywhere says so.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "src",
  "web",
  "static",
  "overlays.js"
);

function load() {
  const sandbox = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return sandbox;
}

function draft(entropy) {
  return [
    { t: "a", m: false, id: 1, c: 0.5, e: entropy },
    { t: "b", m: false, id: 2, c: 0.5, e: entropy },
  ];
}

function commit() {
  return [
    { t: "a", m: false, id: 1, c: 1 },
    { t: "b", m: false, id: 2, c: 1 },
  ];
}

// Two canvases, the way DiffusionGemma streams them: two drafts and a
// commit on canvas 0, then a draft and a commit on canvas 1.
const FRAMES = [draft(4), draft(1), commit(), draft(3), commit()];
const CANVASES = [0, 0, 0, 1, 1];

function entropyFrame(sandbox, frames, canvases, index, isAppend) {
  return sandbox.overlaysEntropyFrame(
    (frame) => frames[frame] || null,
    (frame) => canvases[frame],
    index,
    isAppend
  );
}

test("a frame with entropy answers for itself", () => {
  const sandbox = load();

  assert.equal(entropyFrame(sandbox, FRAMES, CANVASES, 1, false), 1);
  assert.equal(entropyFrame(sandbox, FRAMES, CANVASES, 3, false), 3);
});

test("a commit borrows its canvas's last draft", () => {
  const sandbox = load();

  assert.equal(entropyFrame(sandbox, FRAMES, CANVASES, 2, false), 1);
  assert.equal(entropyFrame(sandbox, FRAMES, CANVASES, 4, false), 3);
});

test("the borrow never reaches into another canvas", () => {
  // Canvas 1 holds only a commit, so the draft before it belongs to
  // canvas 0, whose positions are unrelated to canvas 1's.
  const sandbox = load();
  const frames = [draft(4), commit(), commit()];
  const canvases = [0, 0, 1];

  assert.equal(entropyFrame(sandbox, frames, canvases, 2, false), -1);
});

test("an append stream is never searched", () => {
  // Every earlier frame of an append stream is a prefix of this one,
  // so one read is the whole answer.
  const sandbox = load();
  const frames = [draft(4), commit()];
  let reads = 0;

  const found = sandbox.overlaysEntropyFrame(
    (frame) => {
      reads += 1;
      return frames[frame];
    },
    () => 0,
    1,
    true
  );

  assert.equal(found, -1);
  assert.equal(reads, 1);
});

test("a run with no entropy anywhere says so", () => {
  const sandbox = load();

  assert.equal(
    entropyFrame(sandbox, [commit(), commit()], [0, 0], 1, false), -1
  );
  assert.equal(entropyFrame(sandbox, [], [], -1, false), -1);
});
