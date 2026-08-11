// Tests for the Analytics detail panel's request fence.
//
// Strategy: load the shipped browser file into a fresh vm context
// and drive the factory directly. Evaluating the real file rather
// than importing a copy is what makes this a test of the thing that
// runs: detail_requests.js is a classic global script with no module
// syntax, because analytics.js is one too, and a vm context gives it
// the same top-level-var-becomes-a-global semantics a browser does
// without adding an export tail that only tests would use.
//
// What passing proves is the property the panel could not previously
// hold: a response may paint only if it belongs to the attempt that
// is still open. Delayed responses used to render run A's charts
// under run B's title, and a closed panel could repopulate itself,
// both of which produce a plausible screen that is simply false.
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
  "detail_requests.js"
);

// A fresh global per test, so no test can see another's epoch.
function load() {
  const sandbox = { AbortController };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "detail_requests.js",
  });
  return sandbox;
}

test("a response from the open attempt is accepted", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const token = fence.begin("run-a");

  assert.equal(fence.accepts(token), true);
});

test("opening another run retires the first attempt", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const first = fence.begin("run-a");
  const second = fence.begin("run-b");

  assert.equal(
    fence.accepts(first),
    false,
    "run A's response would have painted under run B"
  );
  assert.equal(fence.accepts(second), true);
});

test("reopening the same run retires the first attempt", () => {
  // The case a run id check alone cannot catch, and the reason the
  // fence counts epochs instead of comparing ids: both attempts are
  // for the same run, but only the second one's answers are the
  // ones the user is waiting on.
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const first = fence.begin("run-a");
  fence.cancel();
  const second = fence.begin("run-a");

  assert.equal(fence.accepts(first), false);
  assert.equal(fence.accepts(second), true);
});

test("closing the panel retires everything in flight", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const token = fence.begin("run-a");
  fence.cancel();

  assert.equal(
    fence.accepts(token),
    false,
    "a dismissed panel would have repopulated itself"
  );
});

test("a second close changes nothing", () => {
  // Close, Escape, and the backdrop all reach cancel, and a delete
  // of the open run can follow any of them, so it has to be safe to
  // call with nothing outstanding.
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  fence.begin("run-a");
  fence.cancel();
  fence.cancel();

  assert.equal(fence.accepts(null), false);
});

test("beginning an attempt aborts the previous signal", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const first = fence.begin("run-a");
  assert.equal(first.signal.aborted, false);
  const second = fence.begin("run-b");

  assert.equal(
    first.signal.aborted,
    true,
    "the superseded fetches were left running"
  );
  assert.equal(second.signal.aborted, false);
});

test("cancelling aborts the outstanding signal", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const token = fence.begin("run-a");
  fence.cancel();

  assert.equal(token.signal.aborted, true);
});

test("a token carries the run it was issued for", () => {
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();

  const token = fence.begin("run-a");

  assert.equal(token.runId, "run-a");
  assert.ok(token.signal);
});

test("a missing or foreign token is refused", () => {
  // Negative space. accepts() is the last gate before a state
  // commit, so it must refuse anything it did not issue rather than
  // treating an unrecognized shape as permission.
  const { detailRequestsCreate } = load();
  const fence = detailRequestsCreate();
  fence.begin("run-a");

  assert.equal(fence.accepts(null), false);
  assert.equal(fence.accepts(undefined), false);
  assert.equal(fence.accepts({}), false);
  assert.equal(
    fence.accepts({ epoch: 1, runId: "run-z" }),
    false
  );
  assert.equal(
    fence.accepts({ epoch: 99, runId: "run-a" }),
    false
  );
});

test("two fences do not share an epoch", () => {
  const { detailRequestsCreate } = load();
  const first = detailRequestsCreate();
  const second = detailRequestsCreate();

  const token = first.begin("run-a");
  second.begin("run-b");
  second.cancel();

  assert.equal(
    first.accepts(token),
    true,
    "another surface's activity retired this one's token"
  );
});

test("an abort is told apart from a real failure", () => {
  const { detailRequestsIsAbort } = load();
  const aborted = new Error("aborted");
  aborted.name = "AbortError";

  assert.equal(detailRequestsIsAbort(aborted), true);
  assert.equal(
    detailRequestsIsAbort(new TypeError("network")),
    false
  );
  assert.equal(detailRequestsIsAbort(null), false);
  assert.equal(detailRequestsIsAbort(undefined), false);
});

test("a real AbortController rejection is recognized", () => {
  // Paired with the hand-built error above, because the shape that
  // matters is the one the platform actually throws.
  const { detailRequestsIsAbort } = load();
  const controller = new AbortController();
  controller.abort();

  assert.equal(
    detailRequestsIsAbort(controller.signal.reason),
    true
  );
});
