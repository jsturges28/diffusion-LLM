// Which editing phase a run is in, and which phase may follow it.
//
// Loaded as a classic global script before app.js, like the other
// extracted modules, and like them it touches no DOM. It decides
// whether a move is legal; app.js decides what the screen looks like
// afterwards. That split is why the rules can be tested without a
// browser, and it is the one the report asks for.
//
// The problem it solves. Editing a run walks through eight phases
// held in a string, with four companion values that describe an
// edit in progress. Ten call sites assigned the string directly and
// nothing said which of them were reachable from where, so the only
// account of the workflow was the union of the buttons that happened
// to be enabled at the time. A phase set from the wrong place is not
// a crash; it is a run that offers Confirm on an edit it never made,
// or a resume that appends to frames it did not branch from.
//
// What is deliberately *not* moved here is the clearing. Each site
// still resets its own companions, exactly as it did, and this
// module checks afterwards that it did. Doing the clearing here
// would have been the tidier design and a worse change: the guided
// edit flow cannot be exercised without a GPU, so the safe move is
// to describe the existing behaviour and refuse anything that does
// not match it, rather than to reimplement it unobserved.

"use strict";

// No edit session. The run is either fresh or finished and being
// read; the scrubber is free and Edit Frames is on offer.
var RUN_PHASE_IDLE = null;
// Diffusion: choosing which frame to edit.
var RUN_PHASE_SELECT = "select";
// Diffusion: choosing which tokens in that frame to remask.
var RUN_PHASE_EDIT = "edit";
// Edits locked; choosing whether to edit another frame or run on.
var RUN_PHASE_CHOICE = "choice";
// Choosing the frame a partial resume should stop at.
var RUN_PHASE_SELECT_TARGET = "select_target";
// Autoregressive: choosing a token to substitute ("What If").
var RUN_PHASE_SUBSTITUTE = "substitute";
// A branch is being generated.
var RUN_PHASE_GENERATING = "generating";
// The branch finished; Confirm and Retry are on offer.
var RUN_PHASE_REVIEW = "review";

// Every move the buttons can actually make, read off the call sites
// rather than imagined. Leaving a session is not here: that is
// `runPhasesReset`, which is reachable from anywhere because an
// error or an abandoned edit can arrive in any phase.
var RUN_PHASE_TRANSITIONS = {
  "": [RUN_PHASE_SELECT, RUN_PHASE_SUBSTITUTE],
  select: [RUN_PHASE_EDIT],
  edit: [RUN_PHASE_CHOICE],
  choice: [RUN_PHASE_SELECT_TARGET, RUN_PHASE_GENERATING],
  select_target: [RUN_PHASE_GENERATING],
  substitute: [RUN_PHASE_GENERATING],
  generating: [RUN_PHASE_EDIT, RUN_PHASE_REVIEW],
  // Only by reset, which Confirm does after saving and Retry does
  // before starting the session again.
  review: [],
};

// `null` is the idle phase and cannot be an object key, so the table
// spells it as the empty string.
function runPhasesKey(mode) {
  return mode === null ? "" : mode;
}

function runPhasesCreate() {
  return {
    mode: RUN_PHASE_IDLE,
    // Autoregressive substitution is open: captured candidates are
    // clickable. Tracked apart from the phase because the popover
    // reads it on hover, where the phase alone would not say whether
    // a click is being invited.
    substituting: false,
    // What a resume in flight was asked to do, "another" or "end",
    // and where it should stop. Both belong to a generation that has
    // been started, which is why they are the only companions a
    // phase other than idle is allowed to carry.
    guidedAction: null,
    targetFrame: null,
    // Frames edited during this session and not yet committed to the
    // run's edit list.
    lockedEdits: [],
  };
}

function runPhasesAllows(phase, next) {
  var allowed = RUN_PHASE_TRANSITIONS[runPhasesKey(phase.mode)];
  if (!allowed) {
    return false;
  }
  return allowed.indexOf(next) !== -1;
}

// An edit in flight is described by `guidedAction` and
// `targetFrame`, so any phase that is not generating must have
// neither. A leftover would make the next resume answer a question
// the user did not ask again.
function runPhasesConsistent(phase) {
  if (phase.mode === RUN_PHASE_GENERATING) {
    return true;
  }
  return phase.guidedAction === null && phase.targetFrame === null;
}

// Move to `next`, refusing anything the table does not list.
//
// Thrown rather than ignored. A refused move means the workflow
// reached a state this file does not describe, and carrying on from
// there is how a run ends up offering Confirm on an edit it never
// made. Loud and at the point of the mistake beats plausible and
// three screens later.
function runPhasesEnter(phase, next) {
  if (!runPhasesAllows(phase, next)) {
    throw new Error(
      "illegal run phase move: "
        + runPhasesKey(phase.mode)
        + " -> "
        + runPhasesKey(next)
    );
  }
  phase.mode = next;
  if (!runPhasesConsistent(phase)) {
    throw new Error(
      "entered " + runPhasesKey(next) + " with a resume still"
        + " described: action=" + phase.guidedAction
        + " target=" + phase.targetFrame
    );
  }
}

// Leave whatever session is open. Reachable from every phase,
// including idle, because an error can arrive at any time and
// because Confirm and Retry both land here.
function runPhasesReset(phase) {
  phase.mode = RUN_PHASE_IDLE;
  phase.substituting = false;
  phase.guidedAction = null;
  phase.targetFrame = null;
  phase.lockedEdits = [];
}

function runPhasesEditing(phase) {
  return phase.mode !== RUN_PHASE_IDLE;
}
