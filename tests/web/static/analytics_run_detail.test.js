// Reusable request-fenced Analytics run detail orchestration.
//
// Strategy: load the request fence and controller against promises
// controlled by the test. Passing proves a later run supersedes every
// payload of the first and closing restores the exchange trigger.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function panel() {
  const listeners = {};
  return {
    open: false,
    addEventListener(type, callback) {
      listeners[type] = callback;
    },
    close() {
      this.open = false;
      if (listeners.close) {
        listeners.close();
      }
    },
  };
}

function load() {
  const context = vm.createContext({
    Promise,
    Error,
    TypeError,
    DOMException,
    AbortController,
  });
  for (const name of [
    "detail_requests.js",
    "analytics_run_detail.js",
  ]) {
    const source = path.join(STATIC, name);
    vm.runInContext(
      fs.readFileSync(source, "utf8"),
      context,
      { filename: source }
    );
  }
  return context;
}

function harness() {
  const context = load();
  const modal = panel();
  const calls = {
    starts: [],
    metadata: [],
    metrics: [],
    frames: [],
    closes: 0,
  };
  const replies = {};
  function request(input, kind) {
    const key = input.key + ":" + kind;
    replies[key] = deferred();
    return replies[key].promise;
  }
  const controller = context.analyticsRunDetailCreate({
    panel: modal,
    openModal(target) {
      target.open = true;
    },
    closeModal(target) {
      target.close();
    },
    onStart: (input) => calls.starts.push(input.key),
    onInvalid() {},
    fetchMeta: (input) => request(input, "meta"),
    fetchMetrics: (input) => request(input, "metrics"),
    fetchFrames: (input) => request(input, "frames"),
    onMeta: (value) => calls.metadata.push(value),
    onMetaFailure() {},
    onMetrics: (value) => calls.metrics.push(value),
    onMetricsFailure() {},
    onFrames: (value) => calls.frames.push(value),
    onFramesFailure() {},
    onClose: () => {
      calls.closes += 1;
    },
  });
  controller.wire();
  return { controller, replies, calls, modal };
}

function input(key, returnFocus = null) {
  return {
    key,
    summary: { run_id: key },
    returnFocus,
  };
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
}

test("a newer detail drops every older payload", async () => {
  const page = harness();
  page.controller.show(input("first"));
  page.controller.show(input("second"));

  page.replies["first:meta"].resolve({ id: "old-meta" });
  page.replies["first:metrics"].resolve({ id: "old-metrics" });
  page.replies["first:frames"].resolve({ id: "old-frames" });
  page.replies["second:meta"].resolve({ id: "new-meta" });
  page.replies["second:metrics"].resolve({ id: "new-metrics" });
  page.replies["second:frames"].resolve({ id: "new-frames" });
  await settle();

  assert.deepEqual(page.calls.metadata, [{ id: "new-meta" }]);
  assert.deepEqual(page.calls.metrics, [{ id: "new-metrics" }]);
  assert.deepEqual(page.calls.frames, [{ id: "new-frames" }]);
});

test("closing cancels and restores the launching control", () => {
  const page = harness();
  let focused = 0;
  const trigger = {
    isConnected: true,
    focus() {
      focused += 1;
    },
  };
  page.controller.show(input("snapshot", trigger));

  page.controller.close();

  assert.equal(page.modal.open, false);
  assert.equal(page.calls.closes, 1);
  assert.equal(focused, 1);
  assert.equal(page.controller.activeKey(), null);
});
