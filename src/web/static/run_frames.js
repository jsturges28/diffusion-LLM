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

// One value per frame whichever way the run is stored. The other two,
// `history` and `tokens`, describe a frame's whole contents and so
// exist only for a snapshot run.
var RUN_FRAME_SCALAR_FIELDS = [
  "canvasIndex",
  "meanConf",
  "elapsed",
  "revealed",
];

// How a run's frames are kept.
//
// SNAPSHOT is the original: `history` and `tokens` carry one entry
// per frame, each holding the whole sequence as it stood. Diffusion
// needs it, because a denoising step revises positions behind the
// newest one, so there is no smaller truthful description of a
// frame than the frame.
//
// APPEND is for a run that only ever grows. Decoding left to right
// never disturbs a settled position, so the run is one flat list of
// positions and frame N is its first N+1 entries. `history` and
// `tokens` stay empty and the readers assemble what they need. The
// difference is N records against N(N+1)/2: at 2,048 tokens, 2,048
// against 2,098,176.
var RUN_FRAME_SHAPE_SNAPSHOT = "snapshot";
var RUN_FRAME_SHAPE_APPEND = "append";

function runFramesCreate() {
  return {
    shape: RUN_FRAME_SHAPE_SNAPSHOT,
    // How many frames the run has. Its own field rather than a
    // length read off `history`, which stopped being the frame count
    // the moment a shape existed that does not store one text per
    // frame.
    count: 0,
    // APPEND only: one entry per token, in decode order.
    positions: [],
    history: [],
    tokens: [],
    canvasIndex: [],
    meanConf: [],
    elapsed: [],
    revealed: [],
  };
}

function runFramesLength(frames) {
  return frames.count;
}

function runFramesIsAppend(frames) {
  return frames.shape === RUN_FRAME_SHAPE_APPEND;
}

// Every array is either empty or as long as `history`. Empty is
// allowed because a snapshot that hit the storage quota carries only
// three of the six, and the page still renders from those.
function runFramesAligned(frames) {
  var count = frames.count;
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var arr = frames[RUN_FRAME_FIELDS[i]];
    if (arr.length !== 0 && arr.length !== count) {
      return false;
    }
  }
  // An append run's positions are the frames: one is added per
  // frame, so a run holding a different number of them has lost or
  // gained one and every frame after that point would assemble
  // wrong.
  if (runFramesIsAppend(frames)) {
    if (frames.positions.length !== 0
      && frames.positions.length !== count) {
      return false;
    }
  }
  return true;
}

// A run that came back from a snapshot too large to store whole.
// The light payload keeps the frame text and the timings and drops
// the per-token detail, so the run reads and scrubs but has nothing
// to hover and no entropy to profile.
//
// Told apart from a model that simply reports no per-token data,
// which is a different thing: that one still appends a `null` per
// frame, so its array is as long as the rest. Only a restore leaves
// the array empty beside a populated history.
function runFramesLackDetail(frames) {
  if (runFramesIsAppend(frames)) {
    // An append run's detail is its positions, and they are the
    // whole run: there is no lighter form of it to fall back to,
    // which is the point. The full store is small enough to keep,
    // so a restore either has the run or has nothing.
    return frames.count > 0 && frames.positions.length === 0;
  }
  return frames.count > 0 && frames.tokens.length === 0;
}

// ---- Reading one frame ----
//
// Everything that draws a frame goes through these rather than
// indexing the arrays, so how a frame is *stored* stops being
// something the page knows.
//
// That distinction is the point. An autoregressive frame is a prefix
// of the next one, never a rewrite, so the same run can be kept as
// one flat list of positions instead of a snapshot per frame: the
// difference between N records and N(N+1)/2 of them, which at 2,048
// tokens is 2,048 against 2,098,176. Diffusion cannot be kept that
// way, because a denoising step really does change positions behind
// the newest one, so it keeps its snapshots. Two storage shapes, one
// way to read them.
//
// Callers ask for a frame and get an array; whether that array was
// stored or assembled is not their business.

function runFramesTokensAt(frames, index) {
  if (runFramesIsAppend(frames)) {
    if (index < 0 || index >= frames.positions.length) {
      return null;
    }
    // The reconstruction, and all of it. Frame N is the run's first
    // N+1 positions, so this is a slice rather than a replay, which
    // is why the audit's "periodic checkpoints if random scrubbing
    // needs bounded seek time" does not apply: seek is already one
    // copy of what the caller was going to read anyway.
    return frames.positions.slice(0, index + 1);
  }
  var stored = frames.tokens[index];
  return stored === undefined ? null : stored;
}

// The finished run, which several readers want without caring where
// it ended: the diff, the entropy profile, the candidate popover.
//
// Its own function rather than `runFramesTokensAt(frames, length-1)`
// because those two stop meaning the same thing the moment a restore
// arrives with fewer token arrays than frames. Asking for the last
// one there is answerable; asking for index `length - 1` is not.
function runFramesTokensLast(frames) {
  if (runFramesIsAppend(frames)) {
    return runFramesTokensAt(frames, frames.positions.length - 1);
  }
  return runFramesTokensAt(frames, frames.tokens.length - 1);
}

function runFramesTextAt(frames, index) {
  if (runFramesIsAppend(frames)) {
    if (index < 0 || index >= frames.positions.length) {
      return null;
    }
    var parts = [];
    for (var i = 0; i <= index; i++) {
      parts.push(frames.positions[i].t);
    }
    return parts.join("");
  }
  var stored = frames.history[index];
  return stored === undefined ? null : stored;
}

// The pre-edit baseline reads the same way, and needs to: the diff
// and crossfade overlays put a live frame and a baseline frame side
// by side, so a difference in how they are reached would show up as
// a difference in what they mean.
function originalRunIsAppend(original) {
  return original.shape === RUN_FRAME_SHAPE_APPEND;
}

function originalRunTokensAt(original, index) {
  if (originalRunIsAppend(original)) {
    if (index < 0 || index >= original.positions.length) {
      return null;
    }
    return original.positions.slice(0, index + 1);
  }
  var stored = original.tokens[index];
  return stored === undefined ? null : stored;
}

function originalRunTokensLast(original) {
  if (originalRunIsAppend(original)) {
    return originalRunTokensAt(
      original, original.positions.length - 1
    );
  }
  return originalRunTokensAt(original, original.tokens.length - 1);
}

function originalRunTextAt(original, index) {
  if (originalRunIsAppend(original)) {
    if (index < 0 || index >= original.positions.length) {
      return null;
    }
    var parts = [];
    for (var i = 0; i <= index; i++) {
      parts.push(original.positions[i].t);
    }
    return parts.join("");
  }
  var stored = original.history[index];
  return stored === undefined ? null : stored;
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
  frames.count += 1;
  runFramesAssertAligned(frames, "append");
}

// One position, which for an append run is also one frame.
//
// The frame's own arrival number is checked against what the run
// already holds, and a disagreement throws. A snapshot protocol
// repairs itself: a frame that went missing costs one bad render and
// the next one puts it right, because every frame restates
// everything. An append protocol has no such property. A gap here
// would shift every later position by one and the run would read as
// fluent text the model never produced, which is the worst way for
// this to fail: silently, and plausibly.
function runFramesAppendPosition(frames, entry) {
  runFramesAssertAligned(frames, "append position entry");
  if (frames.count === 0 && frames.positions.length === 0) {
    frames.shape = RUN_FRAME_SHAPE_APPEND;
  }
  if (!runFramesIsAppend(frames)) {
    throw new Error(
      "cannot append a position to a snapshot run"
    );
  }
  var expected = frames.positions.length + 1;
  if (entry.index !== expected) {
    throw new Error(
      "run frames arrived out of order: expected position "
      + expected + ", got " + entry.index
    );
  }
  frames.positions.push(entry.token);
  frames.canvasIndex.push(entry.canvasIndex);
  frames.meanConf.push(entry.meanConf);
  frames.elapsed.push(entry.elapsed);
  frames.revealed.push(entry.revealed);
  frames.count += 1;
  runFramesAssertAligned(frames, "append position");
}

// Cut back to `count` frames so a branch appends cleanly at that
// index. Setting `length` on a shorter array extends it with holes,
// which is deliberate and is what repairs a degraded restore: the
// three arrays a quota-limited snapshot dropped come back the same
// length as the rest instead of staying empty and failing the save's
// own length check.
function runFramesTruncate(frames, count) {
  runFramesAssertAligned(frames, "truncate entry");
  if (runFramesIsAppend(frames)) {
    // `history` and `tokens` are empty for an append run and have to
    // stay that way. The hole extension below is a repair for a
    // light restore, where those arrays were dropped and the rest
    // were kept; applying it to a run that never had them
    // manufactures a frame count out of nothing, and the next append
    // then finds five arrays saying one thing and positions saying
    // another. That is what broke What If: the truncate before a
    // substitution left two holes beside two real positions, and the
    // seed frame landed on a family already out of step.
    frames.positions.length = count;
    for (var s = 0; s < RUN_FRAME_SCALAR_FIELDS.length; s++) {
      frames[RUN_FRAME_SCALAR_FIELDS[s]].length = count;
    }
  } else {
    for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
      frames[RUN_FRAME_FIELDS[i]].length = count;
    }
  }
  frames.count = count;
  runFramesAssertAligned(frames, "truncate");
}

function runFramesClear(frames) {
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    frames[RUN_FRAME_FIELDS[i]] = [];
  }
  frames.positions = [];
  frames.count = 0;
  // Back to snapshot, because the next run decides its own shape and
  // a family that stayed append would refuse a diffusion frame.
  frames.shape = RUN_FRAME_SHAPE_SNAPSHOT;
}

function runFramesSnapshot(frames) {
  runFramesAssertAligned(frames, "snapshot");
  var copy = { shape: frames.shape, count: frames.count };
  for (var i = 0; i < RUN_FRAME_FIELDS.length; i++) {
    var name = RUN_FRAME_FIELDS[i];
    copy[name] = frames[name].slice();
  }
  copy.positions = frames.positions.slice();
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
  frames.positions = (snapshot.positions || []).slice();
  frames.shape = snapshot.shape || RUN_FRAME_SHAPE_SNAPSHOT;
  frames.count = typeof snapshot.count === "number"
    ? snapshot.count
    // Older snapshots predate the count and are all snapshot-shaped,
    // where the number of texts was the number of frames.
    : (snapshot.history || []).length;
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
  out.frameShape = frames.shape;
  out.frameCount = frames.count;
  if (runFramesIsAppend(frames)) {
    // Written even for the light field set. On an append run the
    // positions *are* the light payload: the whole store is linear,
    // so there is nothing to drop and no reason to drop it. That is
    // what retires the degraded restore rather than merely making
    // it rarer.
    out.framePositions = frames.positions;
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
  frames.positions = Array.isArray(source.framePositions)
    ? source.framePositions
    : [];
  frames.shape = source.frameShape === RUN_FRAME_SHAPE_APPEND
    ? RUN_FRAME_SHAPE_APPEND
    : RUN_FRAME_SHAPE_SNAPSHOT;
  frames.count = typeof source.frameCount === "number"
    ? source.frameCount
    // A snapshot written before shapes existed, where one text per
    // frame was the only arrangement there was.
    : frames.history.length;
  return frames;
}

// -- the run as it was before the first edit --
//
// A second family, frozen once when a run first completes and never
// appended to. Everything that compares an edited run against what it
// branched from reads it: the Diff overlay, the Original/Edited
// crossfade, the commit-order chart, and the save, which carries the
// pre-edit series alongside the new one.
//
// Deliberately a subset. It keeps four of the six live arrays plus
// the candidate sets, and not `canvasIndex` or `revealed`, because
// nothing compares those and the session snapshot is already large
// enough to be refused by the storage quota. Its own family rather
// than a sixth field on the live one for the same reason: the shapes
// differ, and pretending otherwise would either copy two arrays
// nobody reads or hide which ones are actually there.

var ORIGINAL_RUN_FIELDS = [
  "history",
  "tokens",
  "elapsed",
  "meanConf",
  "positionAlts",
];

var ORIGINAL_RUN_JSON_KEYS = {
  history: "originalFrameHistory",
  tokens: "originalFrameTokens",
  elapsed: "originalPerFrameElapsed",
  meanConf: "originalMeanConf",
  positionAlts: "originalPositionAlts",
};

function originalRunCreate() {
  return {
    // How long the run was before any edit shortened it. Zero means
    // nothing has been captured, which is also the gate every
    // comparison checks before reading the arrays below.
    totalFrames: 0,
    // Carried from the live run it was frozen from. Without it the
    // baseline would keep snapshots for a run the live side keeps as
    // positions, and an edited run would pay the quadratic on the
    // copy after paying nothing on the original.
    shape: RUN_FRAME_SHAPE_SNAPSHOT,
    positions: [],
    history: [],
    tokens: [],
    elapsed: [],
    meanConf: [],
    positionAlts: [],
  };
}

function originalRunCaptured(original) {
  return original.totalFrames > 0;
}

// Freeze the live run as the baseline. Called once, on the first
// terminal frame, and a no-op afterwards: every later `done` belongs
// to a branch, and overwriting the baseline with one would leave
// the comparison views diffing a run against itself.
function originalRunCapture(original, frames, positionAlts) {
  if (originalRunCaptured(original)) {
    return;
  }
  original.totalFrames = frames.count;
  original.shape = frames.shape;
  original.positions = frames.positions.slice();
  original.history = frames.history.slice();
  original.tokens = frames.tokens.slice();
  original.elapsed = frames.elapsed.slice();
  original.meanConf = frames.meanConf.slice();
  original.positionAlts = positionAlts.slice();
}

function originalRunClear(original) {
  original.totalFrames = 0;
  original.shape = RUN_FRAME_SHAPE_SNAPSHOT;
  original.positions = [];
  for (var i = 0; i < ORIGINAL_RUN_FIELDS.length; i++) {
    original[ORIGINAL_RUN_FIELDS[i]] = [];
  }
}

function originalRunToJson(original) {
  var out = {
    originalTotalFrames: original.totalFrames,
    originalFrameShape: original.shape,
  };
  if (original.shape === RUN_FRAME_SHAPE_APPEND) {
    out.originalFramePositions = original.positions;
  }
  for (var i = 0; i < ORIGINAL_RUN_FIELDS.length; i++) {
    var name = ORIGINAL_RUN_FIELDS[i];
    out[ORIGINAL_RUN_JSON_KEYS[name]] = original[name];
  }
  return out;
}

// `fallbackTotal` is the live run's length, used when a snapshot
// predates the field. Without it a restored run would read as having
// no baseline and quietly lose its comparison views.
function originalRunFromJson(source, fallbackTotal) {
  var original = originalRunCreate();
  original.totalFrames = source.originalTotalFrames || fallbackTotal;
  original.shape =
    source.originalFrameShape === RUN_FRAME_SHAPE_APPEND
      ? RUN_FRAME_SHAPE_APPEND
      : RUN_FRAME_SHAPE_SNAPSHOT;
  original.positions =
    Array.isArray(source.originalFramePositions)
      ? source.originalFramePositions
      : [];
  for (var i = 0; i < ORIGINAL_RUN_FIELDS.length; i++) {
    var name = ORIGINAL_RUN_FIELDS[i];
    var stored = source[ORIGINAL_RUN_JSON_KEYS[name]];
    original[name] = Array.isArray(stored) ? stored : [];
  }
  return original;
}

// In place, for the same reason the live family is: whoever holds
// the baseline keeps the same object, so a reference taken anywhere
// stays valid.
function originalRunRestore(original, source, fallbackTotal) {
  var parsed = originalRunFromJson(source, fallbackTotal);
  original.totalFrames = parsed.totalFrames;
  original.shape = parsed.shape;
  original.positions = parsed.positions;
  for (var i = 0; i < ORIGINAL_RUN_FIELDS.length; i++) {
    var name = ORIGINAL_RUN_FIELDS[i];
    original[name] = parsed[name];
  }
}
