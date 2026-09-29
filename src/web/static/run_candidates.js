// A diffusion run's candidates, per captured frame, for both pages.
//
// Loaded as a classic global script before the page that uses it, as
// run_frames.js is, and like it touches no DOM, which is what lets a
// test drive it in a `vm` away from a browser.
//
// A diffusion position is re-decided at every denoising step, so the
// popover has to answer "what was this position weighing at the frame
// on screen" rather than "what did it weigh once". The worker sends
// that as one message when a run ends: the frames it captured, a set
// per position for each, and the stride it thinned to when the run
// would otherwise have passed its budget. A frame the stride skipped
// borrows the latest captured frame before it, and the popover says
// so ("As of step N").
//
// There are two lines a borrowed frame may not cross. A resumed edit
// replaces every frame from its first on, so `segments` records where
// each stream of candidates began, and a frame never borrows from
// before its own segment: the run the edit replaced was weighing a
// different canvas. And a DiffusionGemma run chains canvases whose
// positions are unrelated, so a frame never borrows from another
// canvas either.
//
// Every operation returns a new store and leaves the one it was given
// alone, so a snapshot can hold a reference rather than a copy.
//
// A save sends the plain form the server checks. The session snapshot
// a trip to Analytics depends on writes a packed one instead, because
// the plain form did not fit the desktop app's session storage; see
// runCandidatesToSnapshot.

"use strict";

// Mirrors CANDIDATE_BUDGET_RECORDS in src/backends/protocol.py, which
// the server enforces on save: records counted as positions times k
// per frame.
var RUN_CANDIDATES_BUDGET = 102400;
// Halving passes a thinning may take. A pass halves every segment of
// three or more frames, so this covers any store a page could build.
var RUN_CANDIDATES_THIN_PASSES = 32;
// Marks the packed form a session snapshot writes, so a reader can
// tell it from the plain form an older snapshot holds.
var RUN_CANDIDATES_SNAPSHOT_FORM = 1;
// The worker rounds a candidate's probability to four places
// (PROBABILITY_PLACES in src/inference/candidate_capture.py), so the
// packed form writes it as a whole number of ten-thousandths. One off
// that grid is written as it is, which keeps the form exact whatever
// the worker sends; only the size depends on the two agreeing.
var RUN_CANDIDATES_PROBABILITY_SCALE = 10000;

function runCandidatesCreate() {
  return { k: 0, stride: 1, frames: [], segments: [0], sets: [] };
}

function runCandidatesIsEmpty(store) {
  return store.frames.length === 0;
}

// Records the store holds, in the budget's unit.
function runCandidatesRecords(store) {
  var positions = 0;
  for (var i = 0; i < store.sets.length; i++) {
    positions += store.sets[i].length;
  }
  return positions * store.k;
}

// Add a worker's message at `offset`: 0 for a fresh run, the resume
// point for an edit. Its frames are local to the stream that produced
// it, and everything at or past the offset belongs to that stream.
function runCandidatesAddStream(store, offset, message) {
  runCandidatesAssertMessage(message);
  if (store.k !== 0 && store.k !== message.k) {
    throw new Error("candidates: one k per run");
  }
  var kept = runCandidatesTruncate(store, offset);
  var frames = kept.frames.slice();
  var sets = kept.sets.slice();
  for (var i = 0; i < message.frames.length; i++) {
    frames.push(offset + message.frames[i]);
    sets.push(message.sets[i]);
  }
  return {
    k: message.k,
    stride: Math.max(kept.stride, message.stride),
    frames: frames,
    segments: kept.segments,
    sets: sets,
  };
}

// Drop every frame at or past `offset` and start a segment there, so
// the frames that replace them never borrow from before it, whether
// or not the stream that replaces them brings candidates of its own.
function runCandidatesTruncate(store, offset) {
  if (offset < 0) {
    throw new Error("candidates: a frame offset is never negative");
  }
  var frames = [];
  var sets = [];
  for (var i = 0; i < store.frames.length; i++) {
    if (store.frames[i] < offset) {
      frames.push(store.frames[i]);
      sets.push(store.sets[i]);
    }
  }
  var segments = [0];
  for (var s = 1; s < store.segments.length; s++) {
    if (store.segments[s] < offset) {
      segments.push(store.segments[s]);
    }
  }
  if (offset > 0) {
    segments.push(offset);
  }
  return {
    k: store.k,
    stride: store.stride,
    frames: frames,
    segments: segments,
    sets: sets,
  };
}

// The candidates for `frame`: the latest captured frame at or before
// it, in the same segment and on the same canvas, as
// {frame, sets}; or null when there is none. The captured frame comes
// back so the popover can say when it is an earlier one.
function runCandidatesAt(store, frame, canvasOf) {
  var index = runCandidatesFloor(store.frames, frame);
  if (index < 0) {
    return null;
  }
  var found = store.frames[index];
  if (found < runCandidatesFloorValue(store.segments, frame)) {
    return null;
  }
  if (canvasOf(found) !== canvasOf(frame)) {
    return null;
  }
  return { frame: found, sets: store.sets[index] };
}

// One position's set at `frame`, as {frame, set}, or null: what both
// popovers ask for.
function runCandidatesSetAt(store, frame, pos, canvasOf) {
  var found = runCandidatesAt(store, frame, canvasOf);
  if (found === null || pos < 0 || pos >= found.sets.length) {
    return null;
  }
  return { frame: found.frame, set: found.sets[pos] };
}

// Thin to `budget` for saving. Each stream arrived within the budget,
// but an edited run carries several and their sum may not be. Halves
// every segment's frames, keeping each segment's last since that is
// the frame a reader lands on, until the store fits. Null when it
// cannot, which only a run with a great many edits could reach; the
// run is then saved without its candidates rather than not at all.
function runCandidatesThin(store, budget) {
  var thinned = store;
  for (var pass = 0; pass < RUN_CANDIDATES_THIN_PASSES; pass++) {
    if (runCandidatesRecords(thinned) <= budget) {
      return thinned;
    }
    var halved = runCandidatesHalve(thinned);
    if (halved.frames.length === thinned.frames.length) {
      return null;
    }
    thinned = halved;
  }
  return runCandidatesRecords(thinned) <= budget ? thinned : null;
}

function runCandidatesHalve(store) {
  var frames = [];
  var sets = [];
  var ordinal = 0;
  for (var i = 0; i < store.frames.length; i++) {
    var last = runCandidatesEndsSegment(store, i);
    if (ordinal % 2 === 0 || last) {
      frames.push(store.frames[i]);
      sets.push(store.sets[i]);
    }
    ordinal = last ? 0 : ordinal + 1;
  }
  return {
    k: store.k,
    stride: store.stride * 2,
    frames: frames,
    segments: store.segments,
    sets: sets,
  };
}

// Whether frame `index` is the last its segment captured.
function runCandidatesEndsSegment(store, index) {
  if (index === store.frames.length - 1) {
    return true;
  }
  var here = runCandidatesFloorValue(
    store.segments, store.frames[index]
  );
  var next = runCandidatesFloorValue(
    store.segments, store.frames[index + 1]
  );
  return here !== next;
}

function runCandidatesToJson(store) {
  return {
    k: store.k,
    stride: store.stride,
    frames: store.frames.slice(),
    segments: store.segments.slice(),
    sets: store.sets,
  };
}

// A saved store, or a snapshot written plainly, or an empty one for
// anything that is not a well-formed store: a snapshot or a run from
// before candidates existed, or one that captured none.
function runCandidatesFromJson(source) {
  if (!runCandidatesWellFormed(source)) {
    return runCandidatesCreate();
  }
  return {
    k: source.k,
    stride: source.stride,
    frames: source.frames.slice(),
    segments: source.segments.slice(),
    sets: source.sets,
  };
}

function runCandidatesWellFormed(source) {
  if (!source || typeof source !== "object") {
    return false;
  }
  if (typeof source.k !== "number" || source.k < 1) {
    return false;
  }
  if (typeof source.stride !== "number") {
    return false;
  }
  if (!Array.isArray(source.frames) || !Array.isArray(source.sets)) {
    return false;
  }
  if (!Array.isArray(source.segments) || source.segments[0] !== 0) {
    return false;
  }
  return source.frames.length === source.sets.length;
}

// A store as the session snapshot writes it. The plain form repeats
// every candidate's text and spells out every field name, and at 3.8
// million characters for a default LLaDA run it never fit the desktop
// app's session storage, about 5.2 million for the whole snapshot.
// This writes each token's text once and each set as one row of
// numbers, about a quarter of the size. It is exact for every field
// a save carries: a store holding something the rows cannot say, one
// id with two texts or a ranked candidate that is not last, is
// written plainly instead, and runCandidatesFromSnapshot reads both.
function runCandidatesToSnapshot(store) {
  var texts = Object.create(null);
  var sets = [];
  for (var f = 0; f < store.sets.length; f++) {
    var rows = runCandidatesPackFrame(store.sets[f], texts);
    if (rows === null) {
      return runCandidatesToJson(store);
    }
    sets.push(rows);
  }
  return {
    form: RUN_CANDIDATES_SNAPSHOT_FORM,
    k: store.k,
    stride: store.stride,
    frames: store.frames.slice(),
    segments: store.segments.slice(),
    texts: texts,
    sets: sets,
  };
}

// A store as the session snapshot left it, in either form, or an
// empty one for anything that is not a well-formed store, a form this
// build does not know among them.
function runCandidatesFromSnapshot(source) {
  if (!source || source.form === undefined) {
    return runCandidatesFromJson(source);
  }
  if (source.form !== RUN_CANDIDATES_SNAPSHOT_FORM) {
    return runCandidatesCreate();
  }
  var unpacked = runCandidatesUnpack(source);
  return unpacked === null ? runCandidatesCreate() : unpacked;
}

function runCandidatesPackFrame(frameSets, texts) {
  if (!Array.isArray(frameSets)) {
    return null;
  }
  var rows = [];
  for (var p = 0; p < frameSets.length; p++) {
    var row = runCandidatesPackSet(frameSets[p], texts);
    if (row === null) {
      return null;
    }
    rows.push(row);
  }
  return rows;
}

// One set as [h, n, id, q, ...]: the held token, how many plain
// candidates follow, and each as its id and packed probability, then
// the id, probability and rank of a held token appended from outside
// them, which comes last. Records each text in `texts`. Null for a
// set the row cannot say exactly.
function runCandidatesPackSet(set, texts) {
  if (!set || !Array.isArray(set.c)) {
    return null;
  }
  var count = set.c.length;
  var last = set.c[count - 1];
  var ranked = !!last && last.rank !== undefined;
  var plain = ranked ? count - 1 : count;
  var row = [set.h, plain];
  for (var i = 0; i < count; i++) {
    var alt = set.c[i];
    if (!runCandidatesPackable(alt, i < plain, texts)) {
      return null;
    }
    row.push(alt.id, runCandidatesPackProbability(alt.p));
  }
  if (ranked) {
    row.push(last.rank);
  }
  return row;
}

// Whether a candidate fits its row: a plain one carries no rank, and
// its text agrees with any text its id already recorded.
function runCandidatesPackable(alt, plain, texts) {
  if (!alt || typeof alt.t !== "string") {
    return false;
  }
  if (plain && alt.rank !== undefined) {
    return false;
  }
  var known = texts[alt.id];
  if (known === undefined) {
    texts[alt.id] = alt.t;
    return true;
  }
  return known === alt.t;
}

// Whole ten-thousandths where that reads back as the same number,
// and the number itself otherwise. The two cannot be confused: any
// integer is on the grid, so what is written as itself is never one.
function runCandidatesPackProbability(p) {
  var scaled = Math.round(p * RUN_CANDIDATES_PROBABILITY_SCALE);
  if (scaled / RUN_CANDIDATES_PROBABILITY_SCALE === p) {
    return scaled;
  }
  return p;
}

function runCandidatesUnpackProbability(q) {
  if (Number.isInteger(q)) {
    return q / RUN_CANDIDATES_PROBABILITY_SCALE;
  }
  return q;
}

// The packed form's inverse, or null when anything in it is not what
// runCandidatesToSnapshot writes.
function runCandidatesUnpack(source) {
  if (!runCandidatesWellFormed(source)) {
    return null;
  }
  var texts = source.texts;
  if (!texts || typeof texts !== "object") {
    return null;
  }
  var sets = [];
  for (var f = 0; f < source.sets.length; f++) {
    var frameSets = runCandidatesUnpackFrame(source.sets[f], texts);
    if (frameSets === null) {
      return null;
    }
    sets.push(frameSets);
  }
  return {
    k: source.k,
    stride: source.stride,
    frames: source.frames.slice(),
    segments: source.segments.slice(),
    sets: sets,
  };
}

function runCandidatesUnpackFrame(rows, texts) {
  if (!Array.isArray(rows)) {
    return null;
  }
  var sets = [];
  for (var p = 0; p < rows.length; p++) {
    var set = runCandidatesUnpackSet(rows[p], texts);
    if (set === null) {
      return null;
    }
    sets.push(set);
  }
  return sets;
}

// One row back as a set, or null for a row runCandidatesPackSet could
// not have written.
function runCandidatesUnpackSet(row, texts) {
  var plain = runCandidatesPackedCount(row);
  if (plain < 0) {
    return null;
  }
  var ranked = row.length > 2 + 2 * plain;
  var total = ranked ? plain + 1 : plain;
  var candidates = [];
  for (var i = 0; i < total; i++) {
    var alt = runCandidatesUnpackCandidate(row, 2 + 2 * i, texts);
    if (alt === null) {
      return null;
    }
    candidates.push(alt);
  }
  if (ranked) {
    candidates[plain].rank = row[row.length - 1];
  }
  return { h: row[0], c: candidates };
}

// How many plain candidates a packed row holds, or -1 when it is not
// one: after the held token and the count come exactly that many
// pairs, then nothing, or a ranked candidate's three values.
function runCandidatesPackedCount(row) {
  if (!Array.isArray(row)) {
    return -1;
  }
  var plain = row[1];
  if (!Number.isInteger(plain) || plain < 0) {
    return -1;
  }
  var over = row.length - 2 - 2 * plain;
  return over === 0 || over === 3 ? plain : -1;
}

function runCandidatesUnpackCandidate(row, at, texts) {
  var text = texts[row[at]];
  if (typeof text !== "string") {
    return null;
  }
  return {
    id: row[at],
    t: text,
    p: runCandidatesUnpackProbability(row[at + 1]),
  };
}

function runCandidatesAssertMessage(message) {
  if (!message || !Array.isArray(message.frames)) {
    throw new Error("candidates: a message names its frames");
  }
  if (!Array.isArray(message.sets)) {
    throw new Error("candidates: a message carries its sets");
  }
  if (message.frames.length !== message.sets.length) {
    throw new Error("candidates: one list of sets per frame");
  }
}

// The index of the last value at or below `target` in an ascending
// list, or -1 when every value is above it.
function runCandidatesFloor(values, target) {
  var low = 0;
  var high = values.length - 1;
  var found = -1;
  while (low <= high) {
    var middle = (low + high) >> 1;
    if (values[middle] <= target) {
      found = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return found;
}

// The last value at or below `target`, or -1. Segments always start
// at 0, so for a frame this is the start of its segment.
function runCandidatesFloorValue(values, target) {
  var index = runCandidatesFloor(values, target);
  return index < 0 ? -1 : values[index];
}
