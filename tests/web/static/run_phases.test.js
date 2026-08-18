// Tests for which editing phase may follow which.
//
// Strategy: load the shipped file into a fresh vm context and walk
// the workflows the buttons actually produce, then try the moves they
// cannot. No DOM and no state beyond the object under test.
//
// What passing proves is that the eight phases have an account of
// themselves. Editing a run used to hold its phase in a string that
// ten call sites assigned directly, with nothing saying which of them
// were reachable from where; the workflow existed only as the union
// of whichever buttons happened to be enabled. A phase set from the
// wrong place is not a crash. It is a run offering Confirm on an edit
// it never made, or a resume appending to frames it did not branch
// from, which is the kind of wrong answer that looks right.
//
// The four full workflows at the top are the point of the file. They
// are transcribed from the call sites rather than invented, so if the
// table is wrong they fail here rather than on a GPU.
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
  "run_phases.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "run_phases.js",
  });
  return sandbox;
}

// Walk a sequence of moves, returning the phase it ends in.
function walk(api, phase, moves) {
  for (const move of moves) {
    api.runPhasesEnter(phase, move);
  }
  return phase.mode;
}

// -- the workflows the buttons produce --

test("a diffusion edit runs to review", () => {
  // Edit Frames, pick a frame, lock its tokens, run to the end.
  const api = load();
  const phase = api.runPhasesCreate();

  const ended = walk(api, phase, [
    "select",
    "edit",
    "choice",
    "generating",
  ]);
  phase.guidedAction = null;
  api.runPhasesEnter(phase, "review");

  assert.equal(ended, "generating");
  assert.equal(phase.mode, "review");
});

test("editing another frame loops back to edit", () => {
  // Edit another, choose where the partial resume stops, run it, and
  // land back in edit with the next frame selected.
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["select", "edit", "choice", "select_target"]);
  phase.guidedAction = "another";
  phase.targetFrame = 12;
  api.runPhasesEnter(phase, "generating");

  // The resume is over, so what described it is cleared first.
  phase.guidedAction = null;
  phase.targetFrame = null;
  api.runPhasesEnter(phase, "edit");

  assert.equal(phase.mode, "edit");
});

test("what if runs straight from substitute to generating", () => {
  // Left to right, so there is no frame-selection phase.
  const api = load();
  const phase = api.runPhasesCreate();

  walk(api, phase, ["substitute", "generating"]);
  api.runPhasesEnter(phase, "review");

  assert.equal(phase.mode, "review");
});

test("retry leaves review and starts the session again", () => {
  // Retry resets before re-entering, which is why review needs no
  // outgoing move of its own.
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["substitute", "generating", "review"]);

  api.runPhasesReset(phase);
  api.runPhasesEnter(phase, "substitute");

  assert.equal(phase.mode, "substitute");
});

test("confirm leaves review for idle", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["select", "edit", "choice", "generating"]);
  phase.guidedAction = null;
  api.runPhasesEnter(phase, "review");

  api.runPhasesReset(phase);

  assert.equal(phase.mode, null);
  assert.equal(api.runPhasesEditing(phase), false);
});

// -- and the moves they cannot make --

test("a run cannot start generating from idle", () => {
  // Generation outside a session is an ordinary run, which does not
  // go through here at all.
  const api = load();
  const phase = api.runPhasesCreate();

  assert.throws(
    () => api.runPhasesEnter(phase, "generating"),
    /illegal run phase move: {2}-> generating/
  );
});

test("tokens cannot be locked before a frame is chosen", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  api.runPhasesEnter(phase, "select");

  assert.throws(
    () => api.runPhasesEnter(phase, "choice"),
    /select -> choice/
  );
});

test("review cannot be reached from a locked edit", () => {
  // Confirm on an edit that was never generated is the exact shape
  // of the bug this table exists to refuse.
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["select", "edit", "choice"]);

  assert.throws(
    () => api.runPhasesEnter(phase, "review"),
    /choice -> review/
  );
});

test("review is a dead end without a reset", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["substitute", "generating", "review"]);

  assert.throws(
    () => api.runPhasesEnter(phase, "select"),
    /review -> select/
  );
});

test("a diffusion session cannot become a substitution", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  api.runPhasesEnter(phase, "select");

  assert.throws(
    () => api.runPhasesEnter(phase, "substitute"),
    /select -> substitute/
  );
});

test("the same phase cannot be entered twice", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  api.runPhasesEnter(phase, "select");

  assert.throws(
    () => api.runPhasesEnter(phase, "select"),
    /select -> select/
  );
});

// -- a finished resume must not describe itself any more --

test("leaving generating with a resume still set is refused", () => {
  // The clearing stays at the call sites; this is what checks they
  // did it. A leftover action would make the next resume answer a
  // question the user did not ask again.
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["substitute"]);
  phase.guidedAction = "end";
  api.runPhasesEnter(phase, "generating");

  assert.throws(
    () => api.runPhasesEnter(phase, "review"),
    /with a resume still described/
  );
});

test("a leftover target frame is refused too", () => {
  // The near miss worth catching: the site that clears the action it
  // just finished and forgets the frame that action was aimed at.
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["select", "edit", "choice", "select_target"]);
  phase.guidedAction = "another";
  phase.targetFrame = 4;
  api.runPhasesEnter(phase, "generating");
  phase.guidedAction = null;

  assert.throws(
    () => api.runPhasesEnter(phase, "edit"),
    /with a resume still described/
  );
});

test("generating is allowed to describe its resume", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  walk(api, phase, ["select", "edit", "choice", "select_target"]);
  phase.guidedAction = "another";
  phase.targetFrame = 9;

  api.runPhasesEnter(phase, "generating");

  assert.equal(phase.mode, "generating");
});

// -- leaving, from anywhere --

test("reset works from every phase", () => {
  // An error can arrive at any point, and both Confirm and Retry
  // land here, so this is the one move with no precondition.
  const api = load();
  const reachable = [
    ["select"],
    ["select", "edit"],
    ["select", "edit", "choice"],
    ["select", "edit", "choice", "select_target"],
    ["substitute"],
    ["substitute", "generating"],
  ];

  for (const moves of reachable) {
    const phase = api.runPhasesCreate();
    walk(api, phase, moves);
    api.runPhasesReset(phase);
    assert.equal(phase.mode, null, moves.join(">"));
  }
});

test("reset clears everything a session was carrying", () => {
  const api = load();
  const phase = api.runPhasesCreate();
  api.runPhasesEnter(phase, "substitute");
  phase.substituting = true;
  phase.lockedEdits = [{ frame_index: 2 }];
  phase.guidedAction = "end";
  phase.targetFrame = 3;

  api.runPhasesReset(phase);

  assert.equal(phase.substituting, false);
  assert.equal(phase.guidedAction, null);
  assert.equal(phase.targetFrame, null);
  assert.equal(phase.lockedEdits.length, 0);
});

test("reset from idle is allowed and changes nothing", () => {
  // resetGuidedMode is called on paths that may not have opened a
  // session at all, including an error before anything started.
  const api = load();
  const phase = api.runPhasesCreate();

  api.runPhasesReset(phase);

  assert.equal(phase.mode, null);
});

// -- the table itself --

test("every phase named is a phase the table knows", () => {
  // A move to a phase with no outgoing entry would be a dead end
  // nobody meant to create.
  const api = load();
  const known = Object.keys(api.RUN_PHASE_TRANSITIONS);

  for (const targets of Object.values(api.RUN_PHASE_TRANSITIONS)) {
    for (const target of targets) {
      assert.ok(known.includes(target), target);
    }
  }
});

test("idle is spelled as the empty key", () => {
  // null cannot be an object key, and mixing the two spellings is
  // how a lookup silently returns undefined.
  const api = load();

  assert.equal(api.RUN_PHASE_IDLE, null);
  assert.ok("" in api.RUN_PHASE_TRANSITIONS);
});

test("a phase outside the table is refused", () => {
  const api = load();
  const phase = api.runPhasesCreate();

  assert.equal(api.runPhasesAllows(phase, "teleport"), false);
});
