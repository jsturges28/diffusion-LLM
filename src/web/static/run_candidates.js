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

"use strict";

// Mirrors CANDIDATE_BUDGET_RECORDS in src/backends/protocol.py, which
// the server enforces on save: records counted as positions times k
// per frame.
var RUN_CANDIDATES_BUDGET = 102400;
// Halving passes a thinning may take. A pass halves every segment of
// three or more frames, so this covers any store a page could build.
var RUN_CANDIDATES_THIN_PASSES = 32;

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

// A saved or snapshotted store, or an empty one for anything that is
// not a well-formed store: a snapshot or a run from before candidates
// existed, or one that captured none.
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
