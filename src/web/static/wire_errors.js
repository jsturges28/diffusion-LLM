// How far an error frame from the worker reaches. Loaded as a classic
// global script before app.js, the same way activation_client.js and
// overlays.js are, and like them it touches no DOM, which is what
// makes it testable away from a browser.
//
// The problem it solves. Every worker failure used to arrive as
// {"type": "error", "message": <a sentence>}, and app.js had one
// handler for all of them. That handler ends the run and, if an edit
// session is open, rolls back to the pre-edit snapshot and leaves
// guided mode. Correct for a generation that died. Wrong for a probe
// refused because a generation was already running, which is a
// measurement that could not be taken, and which used to close What
// If and discard the edit the user was composing.
//
// Since PROTOCOL-01 the worker says which it is, and this file is the
// one place that reads it. Deliberately a classifier and not a
// dispatcher: it decides what a frame means, the page decides what to
// do about it. That split is what lets the meaning be tested without
// a browser and the doing be read in one place in app.js.

"use strict";

// The connection or the model is gone. Nothing else can be attempted.
var WIRE_ERROR_SCOPE_FATAL = "fatal";
// One generation-class operation failed. The socket is fine.
var WIRE_ERROR_SCOPE_RUN = "run";
// One auxiliary request failed and nothing else may be disturbed.
var WIRE_ERROR_SCOPE_REQUEST = "request";

var WIRE_ERROR_SCOPES = [
  WIRE_ERROR_SCOPE_FATAL,
  WIRE_ERROR_SCOPE_RUN,
  WIRE_ERROR_SCOPE_REQUEST,
];

// A frame that names no scope, or names one this page does not know,
// is read as fatal. That is the behaviour every error had before this
// existed, so an unrecognised frame degrades to the old handling
// rather than to silence. It also means a newer worker inventing a
// fourth scope makes an older page over-react, which is the failure
// worth having: cleaning up too much is recoverable, and leaving a
// half-applied edit on screen because a frame was not understood is
// not.
function wireErrorsScope(frame) {
  if (!frame || typeof frame.scope !== "string") {
    return WIRE_ERROR_SCOPE_FATAL;
  }
  if (WIRE_ERROR_SCOPES.indexOf(frame.scope) === -1) {
    return WIRE_ERROR_SCOPE_FATAL;
  }
  return frame.scope;
}

// What the page has to undo, and what it can say. One object rather
// than three predicates, because the caller needs all of it at once
// and reading it as a unit is how the branches stay in agreement.
//
// `unwindsRun` is the whole behavioural change. It covers stopping
// the run indicators and, when an edit session is open, restoring the
// snapshot: the client truncates a run optimistically before the
// worker answers a resume or a substitution, so a failure there has
// to roll that back. An auxiliary request never truncated anything,
// so it has nothing to roll back and must not pretend otherwise.
function wireErrorsRoute(frame) {
  var scope = wireErrorsScope(frame);
  return {
    scope: scope,
    code:
      frame && typeof frame.code === "string" ? frame.code : "",
    message:
      frame && typeof frame.message === "string" && frame.message
        ? frame.message
        : "unknown",
    // Which request this answers, when it answers one. Carried so a
    // local control can recognise its own failure; nothing consumes
    // it yet, and the first thing that needs to should read it here
    // rather than re-parsing the frame.
    requestType:
      frame && typeof frame.request_type === "string"
        ? frame.request_type
        : null,
    requestId:
      frame && typeof frame.request_id === "number"
        ? frame.request_id
        : null,
    unwindsRun: scope !== WIRE_ERROR_SCOPE_REQUEST,
  };
}
