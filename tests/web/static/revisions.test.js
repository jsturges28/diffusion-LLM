// Revisions: when a diffusion position changes its mind, and how
// the Revisions overlay colours it.
//
// Strategy: load overlays.js on its own into a vm context and drive
// its revision primitives with small hand-built frame streams shaped
// like DiffusionGemma's, where a changed token reads as masked for a
// frame before it settles again. Each case pairs a stream that should
// count with the nearest one that should not, so a test cannot pass
// on a rule that never counts anything.
//
// Passing proves that a first settle and a return to the same token
// are not revisions, that a different token is, that masked guesses
// are ignored, that each canvas and each remasked position starts
// over, that the whole-run walk is the per-frame fold, that counts
// run from the start of the shown frame's canvas through that frame,
// and that the colour steps deepen, stay legible, and keep their own
// hue.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const OVERLAYS = path.join(
  __dirname, "..", "..", "..", "src", "web", "static", "overlays.js"
);

function overlays() {
  const sandbox = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(OVERLAYS, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return sandbox;
}

// Arrays built inside the vm context are not reference-equal to host
// ones, so deepEqual rejects them on realm rather than on content.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function settled(id) {
  return { t: "w" + id, m: false, id: id };
}

// A changed position, holding the model's current guess.
function changing(id) {
  return { t: "w" + id, m: true, id: id };
}

// Every frame's revised positions for a one-canvas stream.
function revisionsOf(frames, edits) {
  const page = overlays();
  return host(page.overlaysComputeRevisions(
    page.overlaysFrameReader(frames),
    frames.length,
    () => 0,
    edits || []
  ));
}

// -- the rule --

test("a first settle is a birth, not a revision", () => {
  const frames = [[changing(7)], [settled(7)]];

  assert.deepEqual(revisionsOf(frames), [[], []]);
});

test("settling on a different token is a revision", () => {
  const frames = [[settled(7)], [changing(9)], [settled(9)]];

  assert.deepEqual(revisionsOf(frames), [[], [], [0]]);
});

test("returning to the token it held is not a revision", () => {
  const frames = [[settled(7)], [changing(9)], [settled(7)]];

  assert.deepEqual(revisionsOf(frames), [[], [], []]);
});

test("a masked guess never counts, whatever it holds", () => {
  // The guess changes twice while masked; only the settle matters,
  // and it lands back on the token the position held.
  const frames = [
    [settled(7)], [changing(9)], [changing(11)], [settled(7)],
  ];

  assert.deepEqual(revisionsOf(frames), [[], [], [], []]);
});

test("a swap from one settled token to another counts", () => {
  // The shape an edit leaves: the branch's first frame replaces the
  // edited one, so the changed frame between them is gone.
  const frames = [[settled(7), settled(3)], [settled(9), settled(3)]];

  assert.deepEqual(revisionsOf(frames), [[], [0]]);
});

test("each position is judged against its own last token", () => {
  const frames = [
    [settled(1), settled(2), settled(3)],
    [changing(5), settled(2), changing(6)],
    [settled(5), settled(2), settled(3)],
  ];

  assert.deepEqual(revisionsOf(frames), [[], [], [0]]);
});

test("a token record without an id never counts", () => {
  // An older record shape carries no id to compare, which has to
  // read as nothing known rather than as a change on every frame.
  const frames = [
    [{ t: "a", m: false }], [{ t: "b", m: false }],
  ];

  assert.deepEqual(revisionsOf(frames), [[], []]);
});

test("each canvas starts fresh", () => {
  const page = overlays();
  const frames = [[settled(7)], [settled(9)]];
  const canvases = [0, 1];

  const revisions = host(page.overlaysComputeRevisions(
    page.overlaysFrameReader(frames), 2, (f) => canvases[f], []
  ));

  assert.deepEqual(revisions, [[], []]);
  // The same two frames on one canvas are a revision.
  assert.deepEqual(revisionsOf(frames), [[], [0]]);
});

test("a remasked position starts over at its edit's frame", () => {
  // The user sent position 0 back at frame 1, so its new token is a
  // birth. Position 1 was not remasked, so its change still counts.
  const frames = [
    [settled(7), settled(3)],
    [settled(9), settled(4)],
  ];
  const edits = [{ frame_index: 1, token_positions: [0] }];

  assert.deepEqual(revisionsOf(frames, edits), [[], [1]]);
  assert.deepEqual(revisionsOf(frames), [[], [0, 1]]);
});

test("an edit applies from its own frame on", () => {
  const frames = [[settled(7)], [changing(9)], [settled(9)]];
  const before = [{ frame_index: 1, token_positions: [0] }];
  const after = [{ frame_index: 3, token_positions: [0] }];

  // Remasked before it settled again: the settle is a birth.
  assert.deepEqual(revisionsOf(frames, before), [[], [], []]);
  // Remasked only afterwards: the settle was the model's own change.
  assert.deepEqual(revisionsOf(frames, after), [[], [], [0]]);
});

test("a remasked position's later changes still count", () => {
  // Reborn at the edit, then the model changes its mind about the
  // new token: that second change is a revision like any other.
  const frames = [
    [settled(7)], [settled(9)], [changing(11)], [settled(11)],
  ];
  const edits = [{ frame_index: 1, token_positions: [0] }];

  assert.deepEqual(revisionsOf(frames, edits), [[], [], [], [0]]);
});

test("a run without an edit log is read as unedited", () => {
  const page = overlays();
  const frames = [[settled(7)], [settled(9)]];

  const revisions = host(page.overlaysComputeRevisions(
    page.overlaysFrameReader(frames), 2, () => 0, undefined
  ));

  assert.deepEqual(revisions, [[], [0]]);
});

test("a reader that is not a function is refused", () => {
  const page = overlays();

  assert.throws(
    () => page.overlaysComputeRevisions([], 0, () => 0, []),
    /readFrame/
  );
  assert.throws(
    () => page.overlaysComputeRevisions(() => null, 0, 0, []),
    /canvasOf/
  );
});

// -- the fold --

test("the run walk is the fold, frame by frame", () => {
  const page = overlays();
  const frames = [
    [settled(1), changing(2)],
    [changing(4), settled(2)],
    [settled(4), changing(8)],
    [settled(4), settled(9)],
  ];

  let fold = page.overlaysRevisionFold();
  const stepped = [];
  for (const frame of frames) {
    const step = page.overlaysRevisionStep(fold, frame, 0, []);
    stepped.push(step.revised);
    fold = step.fold;
  }

  assert.deepEqual(host(stepped), revisionsOf(frames));
  assert.deepEqual(host(stepped), [[], [], [0], [1]]);
});

test("a step leaves the fold it was given untouched", () => {
  // The live page keeps the fold between frames and swaps in the one
  // a step returns, so a step that wrote into its input would let a
  // rebuilt fold and a kept one disagree.
  const page = overlays();
  const first = page.overlaysRevisionStep(
    page.overlaysRevisionFold(), [settled(7)], 0, []
  ).fold;
  const before = host(first);

  page.overlaysRevisionStep(first, [settled(9)], 0, [0]);

  assert.deepEqual(host(first), before);
});

// -- the counts --

test("counts run from the canvas's start to the shown frame", () => {
  const page = overlays();
  // Position 0 revises at frames 2 and 4 on canvas 0, then the next
  // canvas begins at frame 5 and revises position 0 at frame 7.
  const revisions = [[], [], [0], [], [0], [], [], [0]];
  const canvases = [0, 0, 0, 0, 0, 1, 1, 1];
  const canvasOf = (f) => canvases[f];

  const at = (frame) => host(
    page.overlaysRevisionCounts(revisions, frame, canvasOf)
  );

  assert.deepEqual(at(1), []);
  assert.deepEqual(at(2), [1]);
  assert.deepEqual(at(3), [1]);
  assert.deepEqual(at(4), [2]);
  assert.deepEqual(at(6), []);
  assert.deepEqual(at(7), [1]);
});

test("a frame outside the run counts nothing", () => {
  const page = overlays();
  const revisions = [[0], [0]];

  const before = page.overlaysRevisionCounts(revisions, -1, () => 0);
  const after = page.overlaysRevisionCounts(revisions, 2, () => 0);

  assert.equal(before.length, 0);
  assert.equal(after.length, 0);
});

test("a run that revised nothing says so", () => {
  const page = overlays();

  assert.equal(page.overlaysHasRevisions([[], [], []]), false);
  assert.equal(page.overlaysHasRevisions([]), false);
  assert.equal(page.overlaysHasRevisions([[], [3], []]), true);
});

// -- the colour --

// Linear-light luminance of a #rrggbb colour.
function luminance(hex) {
  const channels = [1, 3, 5].map(
    (at) => parseInt(hex.slice(at, at + 2), 16) / 255
  );
  const linear = channels.map((v) =>
    v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  );
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

// WCAG contrast against the output area's #111.
function contrastOnCanvas(hex) {
  return (luminance(hex) + 0.05) / (luminance("#111111") + 0.05);
}

function hue(hex) {
  const [r, g, b] = [1, 3, 5].map(
    (at) => parseInt(hex.slice(at, at + 2), 16) / 255
  );
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  assert.equal(max, b, `${hex} is not blue-dominant`);
  return 60 * ((r - g) / (max - min)) + 240;
}

test("an unrevised position keeps its own colour", () => {
  const page = overlays();

  assert.equal(page.revisionColor(0), null);
  assert.equal(page.revisionColor(undefined), null);
  assert.equal(page.revisionColor(-1), null);
});

test("each further revision reads deeper, up to three", () => {
  const page = overlays();
  const steps = [1, 2, 3].map((n) => page.revisionColor(n));

  assert.equal(new Set(steps).size, 3, `${steps}`);
  for (let i = 1; i < steps.length; i++) {
    assert.ok(
      luminance(steps[i]) < luminance(steps[i - 1]),
      `${steps[i - 1]} then ${steps[i]}`
    );
  }
  assert.equal(page.revisionColor(4), page.revisionColor(3));
  assert.equal(page.revisionColor(9), page.revisionColor(3));
});

test("the deepest step stays legible on the canvas", () => {
  const page = overlays();

  assert.ok(contrastOnCanvas(page.revisionColor(3)) >= 4.5);
});

test("every step keeps a cyan hue of its own", () => {
  // Clear of the heatmap's and the mask's green near 135 and of the
  // violet Forgetting ramp from 250, so a tinted word cannot be read
  // as either.
  const page = overlays();

  for (const n of [1, 2, 3]) {
    const at = hue(page.revisionColor(n));
    assert.ok(at >= 180 && at <= 200, `${n}: ${at}`);
  }
});

test("the reading names the count, and is blank without one", () => {
  const page = overlays();

  assert.equal(page.overlaysRevisionReading(2), "Revisions: 2");
  assert.equal(page.overlaysRevisionReading(1), "Revisions: 1");
  assert.equal(page.overlaysRevisionReading(0), "");
  assert.equal(page.overlaysRevisionReading(undefined), "");
});
