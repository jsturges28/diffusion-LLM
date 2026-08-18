// Tests for the run's per-frame arrays, held as one family.
//
// Strategy: load the shipped file into a fresh vm context and drive
// the operations directly, the same pattern the other browser tests
// use. There is no DOM here and no state outside the object under
// test, so every case is a few calls and a length check.
//
// What passing proves is the arithmetic ORG-02 was written about. Six
// arrays are indexed by frame and every one must answer for the same
// frames. They used to be six separate variables enumerated by hand
// at nine sites, and the comment on the old truncateRunArraysAt
// records what happened when a site missed one: the timing chart's x
// axis drifted against the frames it was plotting. That bug is a test
// here now, and it fails against any operation that touches five of
// the six.
//
// The invariant is deliberately not "all six are always equal". A
// session snapshot is allowed to drop three of them when localStorage
// refuses the full payload, so what holds is that each array is
// either empty or exactly as long as `history`. The pair of tests at
// the bottom pins both halves of that, including the part where a
// truncate repairs a degraded restore rather than preserving it.
//
// Run with: node --test tests/web/static/

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
  "run_frames.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "run_frames.js",
  });
  return sandbox;
}

// One frame's worth of every field, distinguishable by index.
function entry(index) {
  return {
    history: "frame " + index,
    tokens: [{ text: "t" + index }],
    canvasIndex: index,
    meanConf: index / 10,
    elapsed: index * 1.5,
    revealed: index,
  };
}

function loadWith(count) {
  const api = load();
  const frames = api.runFramesCreate();
  for (let i = 0; i < count; i += 1) {
    api.runFramesAppend(frames, entry(i));
  }
  return { api, frames };
}

// Built in this realm on purpose. An array produced inside the vm
// has that context's prototype, and a strict deepEqual against a
// host literal fails on identity even when the contents match.
function lengths(api, frames) {
  const out = [];
  for (const name of api.RUN_FRAME_FIELDS) {
    out.push(frames[name].length);
  }
  return out;
}

// -- the family moves together --

test("a new run holds six empty arrays", () => {
  const api = load();
  const frames = api.runFramesCreate();

  assert.equal(api.RUN_FRAME_FIELDS.length, 6);
  assert.deepEqual(lengths(api, frames), [0, 0, 0, 0, 0, 0]);
});

test("appending grows every array by one", () => {
  const { api, frames } = loadWith(3);

  assert.deepEqual(lengths(api, frames), [3, 3, 3, 3, 3, 3]);
  assert.equal(api.runFramesLength(frames), 3);
});

test("appending keeps each field's own value", () => {
  const { frames } = loadWith(2);

  assert.equal(frames.history[1], "frame 1");
  assert.equal(frames.canvasIndex[1], 1);
  assert.equal(frames.elapsed[1], 1.5);
});

test("truncating shortens every array", () => {
  const { api, frames } = loadWith(5);

  api.runFramesTruncate(frames, 2);

  assert.deepEqual(lengths(api, frames), [2, 2, 2, 2, 2, 2]);
});

test("clearing empties every array", () => {
  const { api, frames } = loadWith(4);

  api.runFramesClear(frames);

  assert.deepEqual(lengths(api, frames), [0, 0, 0, 0, 0, 0]);
});

test("clearing keeps the same object", () => {
  // Held in one place and mutated in place: a reassignment would
  // strand every reference taken before it.
  const { api, frames } = loadWith(2);

  api.runFramesClear(frames);

  assert.ok(api.runFramesAligned(frames));
});

// -- an incomplete frame is refused --

test("a frame missing a field is refused", () => {
  // The bug the family exists to prevent, at the append end: five of
  // the six growing while the sixth stands still.
  const api = load();
  const frames = api.runFramesCreate();
  const partial = entry(0);
  delete partial.elapsed;

  assert.throws(
    () => api.runFramesAppend(frames, partial),
    /missing elapsed/
  );
});

test("the refusal names every field that is missing", () => {
  const api = load();
  const frames = api.runFramesCreate();
  const partial = entry(0);
  delete partial.meanConf;
  delete partial.revealed;

  assert.throws(
    () => api.runFramesAppend(frames, partial),
    /meanConf, revealed/
  );
});

test("a null value is a value, not a missing field", () => {
  // A model that sends no per-token detail still produces frames, and
  // null is what the run records for them.
  const api = load();
  const frames = api.runFramesCreate();
  const sparse = entry(0);
  sparse.tokens = null;
  sparse.meanConf = null;

  api.runFramesAppend(frames, sparse);

  assert.equal(api.runFramesLength(frames), 1);
  assert.equal(frames.tokens[0], null);
});

// -- snapshot and restore --

test("a snapshot survives later appends", () => {
  const { api, frames } = loadWith(2);
  const snapshot = api.runFramesSnapshot(frames);

  api.runFramesAppend(frames, entry(9));

  assert.equal(snapshot.history.length, 2);
  assert.equal(api.runFramesLength(frames), 3);
});

test("restoring puts every array back", () => {
  const { api, frames } = loadWith(4);
  const snapshot = api.runFramesSnapshot(frames);
  api.runFramesTruncate(frames, 1);

  api.runFramesRestore(frames, snapshot);

  assert.deepEqual(lengths(api, frames), [4, 4, 4, 4, 4, 4]);
  assert.equal(frames.history[3], "frame 3");
});

test("restoring twice from one snapshot is safe", () => {
  // The snapshot is copied on the way out as well as in, so a second
  // restore cannot be corrupted by edits made after the first.
  const { api, frames } = loadWith(3);
  const snapshot = api.runFramesSnapshot(frames);

  api.runFramesRestore(frames, snapshot);
  api.runFramesAppend(frames, entry(7));
  api.runFramesRestore(frames, snapshot);

  assert.equal(api.runFramesLength(frames), 3);
});

// -- the invariant --

test("a hand-desynchronised family is caught", () => {
  // Reaching past the operations is the only way to get here, which
  // is the point: this is what every call site could do before.
  const { api, frames } = loadWith(3);

  frames.elapsed.pop();

  assert.equal(api.runFramesAligned(frames), false);
});

test("the historical timing bug fails the invariant", () => {
  // Truncating five of the six is exactly what left the saved timing
  // array longer than the frames it indexed.
  const { api, frames } = loadWith(5);

  frames.history.length = 2;
  frames.tokens.length = 2;
  frames.canvasIndex.length = 2;
  frames.meanConf.length = 2;
  frames.revealed.length = 2;

  assert.equal(api.runFramesAligned(frames), false);
});

test("an operation on a broken family throws", () => {
  // Caught on the way in. Truncate writes all six, so a check
  // afterwards would find them agreeing again and the corruption
  // would survive as frames whose detail belongs to other frames.
  const { api, frames } = loadWith(3);
  frames.revealed.pop();

  assert.throws(
    () => api.runFramesTruncate(frames, 3),
    /fell out of step at truncate entry/
  );
});

test("appending to a broken family throws too", () => {
  const { api, frames } = loadWith(3);
  frames.canvasIndex.pop();

  assert.throws(
    () => api.runFramesAppend(frames, entry(4)),
    /fell out of step at append entry/
  );
});

test("snapshotting a broken family throws", () => {
  // Otherwise the damage is copied into the rollback point and comes
  // back on every restore.
  const { api, frames } = loadWith(3);
  frames.meanConf.pop();

  assert.throws(
    () => api.runFramesSnapshot(frames),
    /fell out of step at snapshot/
  );
});

test("the error says which lengths disagreed", () => {
  const { api, frames } = loadWith(2);
  frames.tokens.pop();

  assert.throws(
    () => api.runFramesTruncate(frames, 2),
    /history=2 tokens=1/
  );
});

test("a light family is not mistaken for a broken one", () => {
  // Empty is the legitimate shape after a quota-limited restore, and
  // refusing to truncate it would strand the run uneditable.
  const { api, frames } = loadWith(3);
  const light = api.runFramesFromJson(
    api.runFramesToJson(frames, api.RUN_FRAME_LIGHT_FIELDS)
  );

  api.runFramesTruncate(light, 2);

  assert.deepEqual(lengths(api, light), [2, 2, 2, 2, 2, 2]);
});

// -- serialisation, including the degraded case --

test("a full snapshot round-trips under the old key names", () => {
  // The wire names predate the family and are kept, so a snapshot
  // written by the previous build still restores.
  const { api, frames } = loadWith(3);

  const json = api.runFramesToJson(frames);

  assert.equal(json.frameHistory.length, 3);
  assert.equal(json.perFrameElapsed.length, 3);
  const round = api.runFramesFromJson(json);
  assert.deepEqual(lengths(api, round), [3, 3, 3, 3, 3, 3]);
});

test("a light snapshot keeps only what redraws the run", () => {
  const { api, frames } = loadWith(3);

  const json = api.runFramesToJson(
    frames,
    api.RUN_FRAME_LIGHT_FIELDS
  );

  assert.equal(json.frameHistory.length, 3);
  assert.equal(json.perFrameElapsed.length, 3);
  assert.equal(json.frameRevealed.length, 3);
  assert.equal(json.frameTokens, undefined);
});

test("a light snapshot restores as aligned", () => {
  // Empty is allowed, which is what lets a quota-limited snapshot
  // come back at all instead of being discarded.
  const { api, frames } = loadWith(3);
  const json = api.runFramesToJson(
    frames,
    api.RUN_FRAME_LIGHT_FIELDS
  );

  const restored = api.runFramesFromJson(json);

  assert.ok(api.runFramesAligned(restored));
  assert.equal(restored.tokens.length, 0);
});

test("a truncate repairs a light restore", () => {
  // The behaviour the old code relied on and documented: the three
  // missing arrays come back to the same length as the rest rather
  // than staying empty and failing the save's own length check.
  const { api, frames } = loadWith(4);
  const restored = api.runFramesFromJson(
    api.runFramesToJson(frames, api.RUN_FRAME_LIGHT_FIELDS)
  );

  api.runFramesTruncate(restored, 2);

  assert.deepEqual(lengths(api, restored), [2, 2, 2, 2, 2, 2]);
});

test("a snapshot with nothing in it reads as an empty run", () => {
  const api = load();

  const frames = api.runFramesFromJson({});

  assert.equal(api.runFramesLength(frames), 0);
  assert.ok(api.runFramesAligned(frames));
});

test("a non-array in a snapshot is ignored", () => {
  // sessionStorage is user-writable and survives upgrades; a value of
  // the wrong shape must not become the run.
  const api = load();

  const frames = api.runFramesFromJson({
    frameHistory: ["a", "b"],
    frameTokens: "not an array",
  });

  assert.equal(api.runFramesLength(frames), 2);
  assert.equal(frames.tokens.length, 0);
});

// -- the baseline: the run as it was before the first edit --
//
// A second family, frozen once and never appended to, read by every
// view that compares an edited run against what it branched from.
// Deliberately four of the six live arrays plus the candidate sets:
// nothing compares canvas index or reveal counts, and the session
// snapshot is already large enough to be refused by the quota.

function baselineFrom(api, count) {
  const frames = api.runFramesCreate();
  for (let i = 0; i < count; i += 1) {
    api.runFramesAppend(frames, entry(i));
  }
  return frames;
}

test("a fresh baseline reports nothing captured", () => {
  const api = load();

  const original = api.originalRunCreate();

  assert.equal(api.originalRunCaptured(original), false);
  assert.equal(original.totalFrames, 0);
});

test("capturing takes the live run's length and arrays", () => {
  const api = load();
  const original = api.originalRunCreate();

  api.originalRunCapture(original, baselineFrom(api, 4), ["alts"]);

  assert.equal(api.originalRunCaptured(original), true);
  assert.equal(original.totalFrames, 4);
  assert.equal(original.history.length, 4);
  assert.equal(original.positionAlts.length, 1);
});

test("capturing twice keeps the first run", () => {
  // Every later terminal frame belongs to a branch, and overwriting
  // the baseline with one would leave the comparison views diffing a
  // run against itself.
  const api = load();
  const original = api.originalRunCreate();
  api.originalRunCapture(original, baselineFrom(api, 4), []);

  api.originalRunCapture(original, baselineFrom(api, 2), []);

  assert.equal(original.totalFrames, 4);
});

test("the baseline does not move when the live run does", () => {
  const api = load();
  const frames = baselineFrom(api, 3);
  const original = api.originalRunCreate();
  api.originalRunCapture(original, frames, []);

  api.runFramesTruncate(frames, 1);

  assert.equal(original.history.length, 3);
  assert.equal(api.runFramesLength(frames), 1);
});

test("clearing forgets the baseline", () => {
  const api = load();
  const original = api.originalRunCreate();
  api.originalRunCapture(original, baselineFrom(api, 3), ["a"]);

  api.originalRunClear(original);

  assert.equal(api.originalRunCaptured(original), false);
  assert.equal(original.positionAlts.length, 0);
});

test("the baseline round-trips under its old key names", () => {
  const api = load();
  const original = api.originalRunCreate();
  api.originalRunCapture(original, baselineFrom(api, 3), ["a"]);

  const json = api.originalRunToJson(original);

  assert.equal(json.originalTotalFrames, 3);
  assert.equal(json.originalFrameHistory.length, 3);
  assert.equal(json.originalPerFrameElapsed.length, 3);
  assert.equal(api.originalRunFromJson(json, 0).totalFrames, 3);
});

test("a snapshot predating the count falls back to the run", () => {
  // Without the fallback a restored run reads as having no baseline
  // and silently loses its comparison views.
  const api = load();

  const original = api.originalRunFromJson({}, 7);

  assert.equal(original.totalFrames, 7);
  assert.equal(api.originalRunCaptured(original), true);
});

test("restoring the baseline keeps the same object", () => {
  const api = load();
  const original = api.originalRunCreate();
  const held = original;

  api.originalRunRestore(original, { originalTotalFrames: 5 }, 0);

  assert.equal(held.totalFrames, 5);
});

test("a non-array in a stored baseline is ignored", () => {
  const api = load();

  const original = api.originalRunFromJson(
    { originalTotalFrames: 2, originalFrameTokens: "nope" },
    0
  );

  assert.equal(original.tokens.length, 0);
});

// -- a run that came back without its detail --
//
// sessionStorage refuses the full payload for a long run, so the
// fallback keeps the frame text and the timings and drops the
// per-token detail. The run still reads and scrubs, which is why the
// fallback is worth having, but nothing hovers and there is no
// entropy to profile. Saving one would write that hollowed-out
// version permanently, so the page asks here before it offers to.

test("a live run has its detail", () => {
  const { api, frames } = loadWith(3);

  assert.equal(api.runFramesLackDetail(frames), false);
});

test("a light restore is missing it", () => {
  const { api, frames } = loadWith(4);
  const light = api.runFramesFromJson(
    api.runFramesToJson(frames, api.RUN_FRAME_LIGHT_FIELDS)
  );

  assert.equal(api.runFramesLackDetail(light), true);
});

test("a full restore is not", () => {
  const { api, frames } = loadWith(4);
  const full = api.runFramesFromJson(api.runFramesToJson(frames));

  assert.equal(api.runFramesLackDetail(full), false);
});

test("a model that reports no tokens still counts as detailed", () => {
  // The distinction that makes this predicate worth having. A model
  // with no per-token data appends a null per frame, so its array is
  // as long as the rest; only a restore leaves it empty.
  const api = load();
  const frames = api.runFramesCreate();
  const sparse = entry(0);
  sparse.tokens = null;
  api.runFramesAppend(frames, sparse);

  assert.equal(api.runFramesLackDetail(frames), false);
});

test("an empty run is not missing anything", () => {
  // Nothing to save either way, and reporting a fresh page as
  // damaged would put an error in front of every new run.
  const api = load();

  assert.equal(
    api.runFramesLackDetail(api.runFramesCreate()),
    false
  );
});

test("a truncate restores the detail flag with the arrays", () => {
  // The repair path: once the missing arrays are squared up, the run
  // is savable again rather than permanently refused.
  const { api, frames } = loadWith(4);
  const light = api.runFramesFromJson(
    api.runFramesToJson(frames, api.RUN_FRAME_LIGHT_FIELDS)
  );

  api.runFramesTruncate(light, 2);

  assert.equal(api.runFramesLackDetail(light), false);
});
