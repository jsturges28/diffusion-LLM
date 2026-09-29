// Tests for a diffusion run's candidates, held per captured frame.
//
// Strategy: load the shipped file into a fresh vm context and drive
// the operations directly, as run_frames.test.js does. The store is
// a plain object and every operation returns a new one, so each case
// is a few calls and a look at what came back.
//
// What passing proves is the lookup the popover depends on. A frame
// the stride skipped borrows the latest captured frame before it, and
// never across the two lines where borrowing would show the wrong
// thing: the start of a resumed edit, whose frames replaced the ones
// the edit branched from, and the start of a new DiffusionGemma
// canvas, whose positions are unrelated to the last one's. It also
// proves a save fits the budget the server enforces, that thinning
// keeps each segment's last frame, and that the store survives the
// JSON round trip a session snapshot and a saved run both take.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname, "..", "..", "..", "src", "web", "static",
  "run_candidates.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "run_candidates.js",
  });
  return sandbox;
}

// A set naming the frame it came from, so a lookup can say which
// frame answered.
function setsFor(frame, positions) {
  const sets = [];
  for (let p = 0; p < positions; p += 1) {
    sets.push({ h: frame * 100 + p, c: [{ id: 1, t: "a", p: 1 }] });
  }
  return sets;
}

function message(frames, positions, stride) {
  return {
    type: "candidates",
    k: 5,
    stride: stride || 1,
    frames: frames,
    sets: frames.map((frame) => setsFor(frame, positions || 2)),
  };
}

function oneCanvas() {
  return 0;
}

// -- the lookup --

test("a captured frame answers for itself", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3])
  );

  const found = api.runCandidatesAt(store, 2, oneCanvas);

  assert.equal(found.frame, 2);
  assert.equal(found.sets[1].h, 201);
});

test("a skipped frame borrows the latest captured before it", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 5, 9], 2, 4)
  );

  assert.equal(api.runCandidatesAt(store, 7, oneCanvas).frame, 5);
  assert.equal(api.runCandidatesAt(store, 12, oneCanvas).frame, 9);
});

test("one position's set comes with the frame it was read at", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 5], 3)
  );

  const found = api.runCandidatesSetAt(store, 6, 2, oneCanvas);

  assert.equal(found.frame, 5);
  assert.equal(found.set.h, 502);
  assert.equal(api.runCandidatesSetAt(store, 6, 3, oneCanvas), null);
  assert.equal(api.runCandidatesSetAt(store, 6, -1, oneCanvas), null);
});

test("the opening frame has nothing to borrow", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2])
  );

  assert.equal(api.runCandidatesAt(store, 0, oneCanvas), null);
});

test("a frame never borrows from another canvas", () => {
  // Frames 0 to 3 are canvas 0, 4 onward canvas 1. Frame 4 was
  // skipped, and the latest captured before it is canvas 0's.
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([2, 3, 6])
  );
  const canvasOf = (frame) => (frame < 4 ? 0 : 1);

  assert.equal(api.runCandidatesAt(store, 5, canvasOf), null);
  assert.equal(api.runCandidatesAt(store, 7, canvasOf).frame, 6);
});

// -- a resumed edit --

test("a resumed stream lands at its offset", () => {
  const api = load();
  const generated = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3, 4])
  );
  const truncated = api.runCandidatesTruncate(generated, 2);

  const edited = api.runCandidatesAddStream(
    truncated, 2, message([1, 2])
  );

  assert.deepEqual(Array.from(edited.frames), [1, 3, 4]);
  assert.deepEqual(Array.from(edited.segments), [0, 2]);
  assert.equal(api.runCandidatesAt(edited, 4, oneCanvas).frame, 4);
});

test("a resumed frame never borrows from the replaced run", () => {
  // The resume's own frame 0 is the remasked canvas, before any step,
  // so it has no candidates; the frame before the edit does, and is
  // exactly what must not be lent to it.
  const api = load();
  const generated = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3, 4])
  );
  const edited = api.runCandidatesAddStream(
    api.runCandidatesTruncate(generated, 3), 3, message([1])
  );

  assert.equal(api.runCandidatesAt(edited, 3, oneCanvas), null);
  assert.equal(api.runCandidatesAt(edited, 2, oneCanvas).frame, 2);
});

test("an edit that brings no candidates still draws the line", () => {
  // A guided Run to Here sends none. Its frames must show nothing
  // rather than the candidates of the frames they replaced.
  const api = load();
  const generated = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3, 4])
  );

  const truncated = api.runCandidatesTruncate(generated, 2);

  assert.equal(api.runCandidatesAt(truncated, 3, oneCanvas), null);
  assert.equal(api.runCandidatesAt(truncated, 1, oneCanvas).frame, 1);
});

test("truncating twice at one point draws one line", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3])
  );

  const twice = api.runCandidatesTruncate(
    api.runCandidatesTruncate(store, 2), 2
  );

  assert.deepEqual(Array.from(twice.segments), [0, 2]);
});

test("a fresh run's stream replaces everything", () => {
  const api = load();
  const first = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3])
  );

  const second = api.runCandidatesAddStream(first, 0, message([1]));

  assert.deepEqual(Array.from(second.frames), [1]);
  assert.deepEqual(Array.from(second.segments), [0]);
});

test("operations leave the store they were given alone", () => {
  // What lets the edit snapshot hold a reference for Retry.
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3])
  );

  api.runCandidatesTruncate(store, 1);
  api.runCandidatesAddStream(store, 2, message([1]));
  api.runCandidatesThin(store, 5);

  assert.deepEqual(Array.from(store.frames), [1, 2, 3]);
  assert.deepEqual(Array.from(store.segments), [0]);
});

test("a message without a set per frame is refused", () => {
  const api = load();
  const broken = message([1, 2]);
  broken.sets.pop();

  assert.throws(() => api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, broken
  ));
});

// -- thinning for a save --

test("a store within the budget is saved whole", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3, 4])
  );

  const thinned = api.runCandidatesThin(store, 40);

  assert.equal(thinned, store);
});

test("thinning halves each segment and keeps its last", () => {
  // Two segments, of four frames and three, at 10 records a frame.
  const api = load();
  const generated = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1, 2, 3, 4])
  );
  const edited = api.runCandidatesAddStream(
    api.runCandidatesTruncate(generated, 5), 5, message([1, 2, 3])
  );

  const thinned = api.runCandidatesThin(edited, 50);

  assert.deepEqual(Array.from(thinned.frames), [1, 3, 4, 6, 8]);
  assert.equal(thinned.stride, 2);
  assert.ok(api.runCandidatesRecords(thinned) <= 50);
});

test("a store that cannot fit is saved without candidates", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0, message([1], 30)
  );

  assert.equal(api.runCandidatesThin(store, 100), null);
});

test("the budget matches the server's", () => {
  const api = load();

  assert.equal(api.RUN_CANDIDATES_BUDGET, 160 * 128 * 5);
});

// -- the JSON round trip --

test("a store survives the round trip", () => {
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesTruncate(
      api.runCandidatesAddStream(
        api.runCandidatesCreate(), 0, message([1, 2, 3])
      ),
      2
    ),
    2,
    message([1], 2, 2)
  );

  const back = api.runCandidatesFromJson(
    JSON.parse(JSON.stringify(api.runCandidatesToJson(store)))
  );

  assert.deepEqual(Array.from(back.frames), [1, 3]);
  assert.deepEqual(Array.from(back.segments), [0, 2]);
  assert.equal(back.stride, 2);
  assert.equal(api.runCandidatesAt(back, 3, oneCanvas).sets[0].h, 100);
});

test("anything that is not a store reads back empty", () => {
  const api = load();
  const cases = [
    undefined,
    null,
    [],
    { k: 5, stride: 1, frames: [1], segments: [0], sets: [] },
    { k: 5, stride: 1, frames: [1], segments: [2], sets: [[]] },
    { k: 0, stride: 1, frames: [1], segments: [0], sets: [[]] },
  ];

  for (const source of cases) {
    const back = api.runCandidatesFromJson(source);
    assert.ok(api.runCandidatesIsEmpty(back));
  }
});
