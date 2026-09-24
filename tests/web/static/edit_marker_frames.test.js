// An edit marker names the frame its edit was made at.
//
// Strategy: drive the two surfaces that draw these markers, the
// generator's entropy strip and the Analytics entropy chart, through
// the DOM stub beside this file. Feed each an edit log whose entries
// sit at different frames and require the colour resolved for a
// position to be the shared Commit Order ramp evaluated at that
// position's own frame. Then the three cases that make the mapping
// more than a rename: an edit at frame 0, a position edited twice,
// and a log with no frame at all.
//
// Why frame 0 gets its own test: the map used to hold `true` and its
// readers compared against `true`. Storing a frame there makes 0 a
// legitimate value that is also falsy, so a reader written the
// obvious way would drop the edit made at the very first frame and
// nothing else would look wrong. positionWasEdited exists to stop
// that, and this pins it.
//
// Passing proves both surfaces colour a marker by its own edit's
// frame, normalize against the run's frame count, agree with each
// other, and fall back to the flat colour for an edit that cannot be
// placed in the run.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

const WORDS = ["The", " cat", " sat", " on", " the", " mat"];

function positions(words) {
  return words.map((word, at) => ({
    t: word,
    m: false,
    id: 1000 + at,
    c: +(0.5 + at / 100).toFixed(4),
    e: +(1.5 - at / 100).toFixed(4),
  }));
}

function snapshotFrame(index, words) {
  const tokens = positions(words.slice(0, index));
  return {
    type: "frame",
    index: index,
    total_steps: words.length,
    canvas_index: 0,
    mean_conf: 0.5,
    text: tokens.map((t) => t.t).join(""),
    tokens: tokens,
    revealed: index > 0 ? [index - 1] : [],
  };
}

// A generator page holding a run of WORDS.length frames plus an edit
// log. The run matters because it sets the ramp's denominator.
function generatorPage(edits) {
  const { context } = loadPage({});
  for (let index = 1; index <= WORDS.length; index++) {
    context.handleFrame(snapshotFrame(index, WORDS));
  }
  context.remaskEdits = edits;
  return context;
}

function bootFetch() {
  return function (url) {
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : { success: true, collections: [] };
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    });
  };
}

// An Analytics page and a saved run payload carrying the same log.
// Returned together because the page's resolver takes the payload
// rather than reading module state.
function analyticsRun(edits) {
  const { context } = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  const canvas = positions(WORDS);
  const frames = WORDS.map((_, at) => canvas.slice(0, at + 1));
  const data = { frames: frames, remask_edits: edits };
  return { context, data };
}

// -- the mapping itself --

test("a touched position remembers its edit's frame", () => {
  const context = generatorPage([
    { frame_index: 1, token_positions: [0, 1] },
    { frame_index: 4, token_positions: [3] },
  ]);

  const marks = context.editedPositionMarks();

  assert.equal(marks[0], 1);
  assert.equal(marks[1], 1);
  assert.equal(marks[3], 4);
});

test("an untouched position has no frame", () => {
  // The negative space. A map that answered for every position would
  // mark the whole canvas.
  const context = generatorPage([
    { frame_index: 1, token_positions: [0] },
  ]);

  const marks = context.editedPositionMarks();

  assert.equal(context.positionWasEdited(marks, 2), false);
  assert.equal(context.positionWasEdited(marks, 0), true);
});

test("a position edited twice reports the later frame", () => {
  // Remasking the same position in a second round should describe the
  // most recent intervention, not the one it replaced. This relies on
  // the log being chronological, which is why it is asserted rather
  // than left to the loop.
  const context = generatorPage([
    { frame_index: 2, token_positions: [5] },
    { frame_index: 5, token_positions: [5] },
  ]);

  assert.equal(context.editedPositionMarks()[5], 5);
});

test("an edit at frame 0 is still an edit", () => {
  // The falsy-value trap. Frame 0 is reachable by scrubbing to the
  // start before remasking, and a truthiness check would lose it.
  const context = generatorPage([
    { frame_index: 0, token_positions: [2] },
  ]);

  const marks = context.editedPositionMarks();

  assert.equal(context.positionWasEdited(marks, 2), true);
  // Array.from because the page builds its arrays in the vm's realm,
  // so a strict deep comparison would fail on the prototype alone.
  assert.deepEqual(
    Array.from(context.editedProfilePositions()), [2]
  );
});

// -- the colours --

test("a marker takes the ramp at its own frame", () => {
  const context = generatorPage([
    { frame_index: 1, token_positions: [0] },
    { frame_index: 4, token_positions: [3] },
  ]);
  const last = context.runFramesLength(context.runFrames) - 1;

  const colors = context.editMarkerColors([0, 3]);

  assert.equal(colors[0], context.commitColor(1, last));
  assert.equal(colors[1], context.commitColor(4, last));
});

test("an early and a late edit are different colours", () => {
  // The point of the change, stated as the thing a reader sees. Two
  // rounds of remasking used to be indistinguishable.
  const context = generatorPage([
    { frame_index: 0, token_positions: [0] },
    { frame_index: 5, token_positions: [3] },
  ]);

  const colors = context.editMarkerColors([0, 3]);

  assert.notEqual(colors[0], colors[1]);
});

test("an edit with no frame falls back to the flat colour", () => {
  // What a log saved before frame_index existed looks like. It should
  // still mark its positions, just without claiming an order.
  const context = generatorPage([
    { token_positions: [1] },
  ]);

  const colors = context.editMarkerColors([1]);

  assert.equal(colors[0], context.OVERLAYS_EDIT_COLOR);
});

// -- the same answers in Analytics --

test("Analytics colours a marker by the same rule", () => {
  const { context, data } = analyticsRun([
    { frame_index: 1, token_positions: [0] },
    { frame_index: 4, token_positions: [3] },
  ]);
  const last = data.frames.length - 1;

  const colors = context.editedPositionColors(data, [0, 3]);

  assert.equal(colors[0], context.commitColor(1, last));
  assert.equal(colors[1], context.commitColor(4, last));
});

test("Analytics keeps an edit made at frame 0", () => {
  const { context, data } = analyticsRun([
    { frame_index: 0, token_positions: [2] },
  ]);

  assert.deepEqual(Array.from(context.editedPositions(data)), [2]);
  assert.equal(
    context.editedPositionColors(data, [2])[0],
    context.commitColor(0, data.frames.length - 1)
  );
});

test("the two surfaces agree on one run", () => {
  // Stated directly rather than inferred from the two tests above,
  // because the whole reason the colour helper moved into overlays.js
  // is that a marker must not mean different things on two pages.
  const edits = [{ frame_index: 3, token_positions: [2] }];
  const generator = generatorPage(edits);
  const { context: analytics, data } = analyticsRun(edits);

  assert.equal(
    generator.editMarkerColors([2])[0],
    analytics.editedPositionColors(data, [2])[0]
  );
});
