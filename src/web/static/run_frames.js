// The per-frame arrays a run is made of, as one thing.
//
// Loaded as a classic global script before app.js, the same way
// activation_client.js and wire_errors.js are, and like them it
// touches no DOM, which is what makes it testable away from a
// browser. It will become a native module when the page entrypoints
// do; that conversion is deliberately not this change.
//
// The problem it solves is arithmetic, not architecture. A run keeps
// six arrays indexed by frame, and every one of them has to answer
// for the same frames. They were declared separately and enumerated
// by hand at nine sites: appended in handleFrame, frozen into the
// original-run copy, snapshotted, restored, truncated, cleared,
// projected into the save payload, serialised into sessionStorage and
// read back out. Adding a seventh array meant getting all nine right,
// and the comment on the old truncateRunArraysAt records what
// happened when one was missed: the timing chart's x-axis drifted
// against the frames it was plotting.
//
// So the family is one object with six operations, and the operations
// are the only way to change its shape.
//
// The invariant is not quite "all six are the same length", because
// the session snapshot is allowed to drop three of them when
// localStorage refuses the full payload. What holds everywhere is
// that each array is either empty or exactly as long as `history`,
// and a degraded restore is repaired by the first truncate, which
// gives every array the same length whether or not it had one.

"use strict";

// Order matters only for readability; nothing indexes by position.
var RUN_FRAME_FIELDS = [
  "history",
  "tokens",
  "canvasIndex",
  "meanConf",
  "elapsed",
  "revealed",
];

// The names these arrays have in a sessionStorage snapshot. Kept as
// they were before the family had a home, so a snapshot written by
// the previous build still restores rather than dropping the run.
var RUN_FRAME_JSON_KEYS = {
  history: "frameHistory",
  tokens: "frameTokens",
  canvasIndex: "frameCanvasIndex",
  meanConf: "frameMeanConf",
  elapsed: "perFrameElapsed",
  revealed: "frameRevealed",
};

// The three a snapshot keeps when the full one will not fit. They are
// enough to redraw the run with the character renderer and to report
// its timings; the other three are per-token detail.
var RUN_FRAME_LIGHT_FIELDS = ["history", "elapsed", "revealed"];

function runFramesCreate() {
  return {
    history: [],
    tokens: [],
    canvasIndex: [],
    meanConf: [],
    elapsed: [],
    revealed: [],
  };
}

function runFramesLength(frames) {
  return frames.history.length;
}

// Every array is either empty or as long as `history`. Empty is
// allowed because a snapshot that hit the storage quota carries only
// three of the six, and the page still renders from those.
function runFramesAligned(frames) {
  var count = frames.history.length;
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var arr = frames[RUN_FRAME_FIELDS[i]];
    if (arr.length !== 0 && arr.length !== count) {
      return false;
    }
  }
  return true;
}

// Thrown rather than logged. A run whose arrays disagree produces a
// saved record whose charts contradict each other, and that is worth
// stopping at the point of corruption instead of discovering later
// in Analytics.
//
// Checked on the way in as well as on the way out, and the way in is
// the one that earns its keep. Every operation here writes all six,
// so a check afterwards can only catch a bug in this file; a check
// beforehand catches a caller that reached past these functions and
// changed one array on its own, which is what every call site used
// to do. Without it the next truncate would quietly paper over the
// damage by setting all six to the same length, and the run would
// carry frames whose token detail belongs to different frames.
function runFramesAssertAligned(frames, where) {
  if (runFramesAligned(frames)) {
    return;
  }
  var lengths = RUN_FRAME_FIELDS.map(function (name) {
    return name + "=" + frames[name].length;
  }).join(" ");
  throw new Error(
    "run frames fell out of step at " + where + ": " + lengths
  );
}

// One frame, into all six. Every field is required, including the
// ones a worker might have omitted: the caller decides what a missing
// value means, because only it knows what the previous frame said.
function runFramesAppend(frames, entry) {
  runFramesAssertAligned(frames, "append entry");
  var missing = [];
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    if (!(RUN_FRAME_FIELDS[i] in entry)) {
      missing.push(RUN_FRAME_FIELDS[i]);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      "run frame entry is missing " + missing.join(", ")
    );
  }
  frames.history.push(entry.history);
  frames.tokens.push(entry.tokens);
  frames.canvasIndex.push(entry.canvasIndex);
  frames.meanConf.push(entry.meanConf);
  frames.elapsed.push(entry.elapsed);
  frames.revealed.push(entry.revealed);
  runFramesAssertAligned(frames, "append");
}

// Cut back to `count` frames so a branch appends cleanly at that
// index. Setting `length` on a shorter array extends it with holes,
// which is deliberate and is what repairs a degraded restore: the
// three arrays a quota-limited snapshot dropped come back the same
// length as the rest instead of staying empty and failing the save's
// own length check.
function runFramesTruncate(frames, count) {
  runFramesAssertAligned(frames, "truncate entry");
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    frames[RUN_FRAME_FIELDS[i]].length = count;
  }
  runFramesAssertAligned(frames, "truncate");
}

function runFramesClear(frames) {
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    frames[RUN_FRAME_FIELDS[i]] = [];
  }
}

function runFramesSnapshot(frames) {
  runFramesAssertAligned(frames, "snapshot");
  var copy = {};
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var name = RUN_FRAME_FIELDS[i];
    copy[name] = frames[name].slice();
  }
  return copy;
}

// In place, so whoever holds the family keeps the same object. A
// reassignment here would strand every reference taken before it,
// which is the failure mode that made these six separate variables
// awkward in the first place.
function runFramesRestore(frames, snapshot) {
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var name = RUN_FRAME_FIELDS[i];
    frames[name] = (snapshot[name] || []).slice();
  }
  runFramesAssertAligned(frames, "restore");
}

// Serialised under the old key names. `fields` narrows it to the
// three a snapshot keeps when the full payload will not fit.
function runFramesToJson(frames, fields) {
  var names = fields || RUN_FRAME_FIELDS;
  var out = {};
  for (var i = 0; i < names.length; i++) {
    out[RUN_FRAME_JSON_KEYS[names[i]]] = frames[names[i]];
  }
  return out;
}

// Read back from a snapshot, tolerating the three that a degraded
// write dropped. Not asserted afterwards: a light snapshot is
// legitimately unaligned until the first truncate squares it up.
function runFramesFromJson(source) {
  var frames = runFramesCreate();
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var name = RUN_FRAME_FIELDS[i];
    var stored = source[RUN_FRAME_JSON_KEYS[name]];
    frames[name] = Array.isArray(stored) ? stored : [];
  }
  return frames;
}
