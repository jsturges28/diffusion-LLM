// Tests for how far a worker error reaches.
//
// Strategy: load the shipped file into a fresh vm context and call
// the classifier directly, the same pattern the other browser tests
// use. It touches no DOM and holds no state, so there is nothing to
// stub and every case is one call.
//
// What passing proves is the client half of PROTOCOL-01. app.js had
// one error handler, and it ended the run and, if an edit session was
// open, restored the pre-edit snapshot and left guided mode. Right
// for a generation that died; wrong for a probe refused because a
// generation was running, which is a measurement that could not be
// taken and which used to close What If and discard the edit being
// composed. The single decision this file exports is whether the page
// unwinds, and the tests below pin it for each scope, including for
// frames that name no scope at all.
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
  "wire_errors.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "wire_errors.js",
  });
  return sandbox;
}

function route(frame) {
  return load().wireErrorsRoute(frame);
}

// -- the decision the page acts on --

test("a fatal error unwinds the run", () => {
  const routed = route({
    type: "error",
    message: "No model is active.",
    code: "no_model_active",
    scope: "fatal",
  });

  assert.equal(routed.unwindsRun, true);
});

test("a failed generation unwinds the run", () => {
  const routed = route({
    type: "error",
    message: "CUDA out of memory",
    code: "generation_failed",
    scope: "run",
    request_type: "generate",
  });

  assert.equal(routed.unwindsRun, true);
});

test("a refused probe leaves the run alone", () => {
  // The finding, in one assertion. This frame used to close What If.
  const routed = route({
    type: "error",
    message: "A generation is already running. Please wait.",
    code: "busy",
    scope: "request",
    request_type: "probe",
    request_id: 4,
  });

  assert.equal(routed.unwindsRun, false);
});

test("the same refusal for a resume does unwind", () => {
  // Same code and same sentence as the probe above; only the scope
  // differs, which is the point of scoping by operation.
  const routed = route({
    type: "error",
    message: "A generation is already running. Please wait.",
    code: "busy",
    scope: "run",
    request_type: "resume",
  });

  assert.equal(routed.unwindsRun, true);
});

test("a failed tokenize leaves the run alone", () => {
  const routed = route({
    type: "error",
    message: "No tokenizer is loaded.",
    code: "no_tokenizer",
    scope: "request",
    request_type: "tokenize",
    request_id: 1,
  });

  assert.equal(routed.unwindsRun, false);
});

// -- frames this page does not understand --

test("a frame with no scope is treated as fatal", () => {
  // What every error was before this existed, so an unrecognised
  // frame degrades to the old handling rather than to silence.
  const routed = route({ type: "error", message: "something" });

  assert.equal(routed.scope, "fatal");
  assert.equal(routed.unwindsRun, true);
});

test("a scope from the future is treated as fatal", () => {
  const routed = route({
    type: "error",
    message: "something",
    scope: "galaxy",
  });

  assert.equal(routed.scope, "fatal");
  assert.equal(routed.unwindsRun, true);
});

test("a non-string scope is treated as fatal", () => {
  const routed = route({
    type: "error",
    message: "something",
    scope: 3,
  });

  assert.equal(routed.unwindsRun, true);
});

test("a missing frame does not throw", () => {
  // The handler is wired to a socket; a malformed frame must not
  // take the page down on the way to reporting itself.
  const routed = route(undefined);

  assert.equal(routed.unwindsRun, true);
});

// -- what it reads off the frame --

test("the message survives", () => {
  const routed = route({
    message: "disk on fire",
    scope: "run",
  });

  assert.equal(routed.message, "disk on fire");
});

test("a missing message becomes something printable", () => {
  const routed = route({ scope: "run" });

  assert.equal(routed.message, "unknown");
});

test("an empty message becomes something printable", () => {
  // "Error: " with nothing after it reads as a broken page.
  const routed = route({ message: "", scope: "run" });

  assert.equal(routed.message, "unknown");
});

test("the code is carried through", () => {
  const routed = route({
    message: "x",
    scope: "run",
    code: "stale_run",
  });

  assert.equal(routed.code, "stale_run");
});

test("a missing code reads as empty rather than undefined", () => {
  const routed = route({ message: "x", scope: "run" });

  assert.equal(routed.code, "");
});

test("the owning request is carried when there is one", () => {
  const routed = route({
    message: "x",
    scope: "request",
    request_type: "probe",
    request_id: 12,
  });

  assert.equal(routed.requestType, "probe");
  assert.equal(routed.requestId, 12);
});

test("an unowned frame reports no owner", () => {
  const routed = route({ message: "x", scope: "fatal" });

  assert.equal(routed.requestType, null);
  assert.equal(routed.requestId, null);
});

test("request id zero is kept", () => {
  // Zero is a legitimate id, and a truthiness check here would drop
  // it and orphan the first request of a session.
  const routed = route({
    message: "x",
    scope: "request",
    request_id: 0,
  });

  assert.equal(routed.requestId, 0);
});
