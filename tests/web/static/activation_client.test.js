// Tests for the shared activation client.
//
// Strategy: load the shipped browser file into a fresh vm context and
// drive the factory with an injected fetch and an injected scheduler,
// the same approach detail_requests.test.js takes and for the same
// reason. Evaluating the real file rather than a copy is what makes
// this a test of the thing that runs: activation_client.js is a
// classic global script, because app.js and menu.js are too, and a vm
// context gives it browser semantics without an export tail that only
// tests would use.
//
// The scheduler is injected rather than faked with timers so the poll
// loop itself is under test, not just one turn of it. A test holds
// the pending callback and decides when the next tick happens, which
// makes "does this stop when it should" answerable rather than a
// matter of waiting and hoping.
//
// What passing proves is what the four separate pollers could not
// agree on: one place decides when an activation is finished, when to
// retry, and whose activation it was. That last part is LIFE-03's:
// a terminal state belonging to another window's activation must not
// fire this one's onReady, or one window navigates for another's
// model.
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
  "activation_client.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "activation_client.js",
  });
  return sandbox;
}

// A scheduler the test drives by hand. Callbacks queue up instead of
// firing on a clock, so a test can run exactly as many polls as it
// means to and then assert that no more were booked.
function manualScheduler() {
  let pending = [];
  let nextHandle = 1;
  const handles = new Map();

  function schedule(fn, ms) {
    const handle = nextHandle;
    nextHandle += 1;
    handles.set(handle, { fn, ms });
    pending.push(handle);
    return handle;
  }

  function unschedule(handle) {
    handles.delete(handle);
    pending = pending.filter((h) => h !== handle);
  }

  // Run every queued callback once, and return how many ran. Async
  // because the callbacks start promise chains the caller has to let
  // settle before asserting.
  async function flush() {
    const running = pending;
    pending = [];
    let ran = 0;
    for (const handle of running) {
      const entry = handles.get(handle);
      if (entry) {
        handles.delete(handle);
        entry.fn();
        ran += 1;
      }
    }
    await settle();
    return ran;
  }

  return {
    schedule,
    unschedule,
    flush,
    booked: () => pending.length,
    delays: () =>
      Array.from(handles.values()).map((e) => e.ms),
  };
}

// Let queued promise callbacks run. Several awaits because each poll
// chains a fetch, a json() and the handler.
async function settle() {
  for (let i = 0; i < 8; i += 1) {
    await Promise.resolve();
  }
}

// A fetch that answers from a script and records what it was asked.
function fakeFetch(script) {
  const calls = [];
  const queue = script.slice();
  function fetchImpl(url, init) {
    calls.push({ url, init });
    const next = queue.length > 1 ? queue.shift() : queue[0];
    if (next instanceof Error) {
      return Promise.reject(next);
    }
    return Promise.resolve({
      ok: next.ok !== false,
      json: () => Promise.resolve(next.body),
    });
  }
  fetchImpl.calls = calls;
  return fetchImpl;
}

const ACCEPTED = { body: { ok: true, operation: 7 } };

function statuses(...list) {
  return list.map((body) => ({ body }));
}

// -- watching an activation to its end --

test("a ready status ends the watch and reports it", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let ready = 0;
  const client = activationClientCreate({
    fetchImpl: fakeFetch(
      statuses({ state: "ready", operation: 7 })
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onReady: () => {
      ready += 1;
    },
  });

  client.observe();
  await clock.flush();

  assert.equal(ready, 1);
  assert.equal(clock.booked(), 0);
});

test("progress is reported for every non-terminal poll", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const seen = [];
  const client = activationClientCreate({
    fetchImpl: fakeFetch(
      statuses(
        { state: "starting" },
        { state: "loading", progress: { fraction: 0.5 } },
        { state: "ready" }
      )
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onProgress: (state) => seen.push(state),
  });

  client.observe();
  await clock.flush();
  await clock.flush();
  await clock.flush();

  assert.deepEqual(seen, ["starting", "loading", "ready"]);
});

test("an error status ends the watch and carries the reason", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let failure = null;
  const client = activationClientCreate({
    fetchImpl: fakeFetch(
      statuses({ state: "error", message: "CUDA out of memory" })
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onFailed: (message) => {
      failure = message;
    },
  });

  client.observe();
  await clock.flush();

  assert.equal(failure, "CUDA out of memory");
  assert.equal(clock.booked(), 0);
});

test("a failed poll retries instead of giving up", async () => {
  // The activation is server-side and outlives a dropped request, so
  // a read that fails says nothing about the load.
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let failure = null;
  const client = activationClientCreate({
    fetchImpl: fakeFetch([new Error("network down")]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    retryMs: 800,
    onFailed: (message) => {
      failure = message;
    },
  });

  client.observe();
  await clock.flush();

  assert.equal(failure, null);
  assert.equal(clock.booked(), 1);
  assert.deepEqual(clock.delays(), [800]);
});

test("a retry waits longer than an ordinary poll", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch(statuses({ state: "loading" })),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    pollMs: 250,
    retryMs: 800,
  });

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [250]);
});

test("stop ends the loop without cancelling", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch(statuses({ state: "loading" }));
  const client = activationClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  client.observe();
  await clock.flush();
  client.stop();
  const ran = await clock.flush();

  assert.equal(ran, 0);
  // Only the polls, no cancel request.
  for (const call of fetchImpl.calls) {
    assert.equal(call.url, "/api/models/activation");
  }
});

// -- starting one --

test("start posts the activation and then watches", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([
    ACCEPTED,
    { body: { state: "ready", operation: 7 } },
  ]);
  let ready = 0;
  const client = activationClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onReady: () => {
      ready += 1;
    },
  });

  await client.start("llada", {});
  await clock.flush();

  assert.equal(fetchImpl.calls[0].url, "/api/models/llada/activate");
  assert.equal(fetchImpl.calls[0].init.method, "POST");
  assert.equal(ready, 1);
});

test("start sends the device only when one was chosen", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const withDevice = fakeFetch([ACCEPTED]);
  const without = fakeFetch([ACCEPTED]);
  const options = {
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  };

  await activationClientCreate(
    Object.assign({ fetchImpl: withDevice }, options)
  ).start("smollm3", { device: "cpu" });
  await activationClientCreate(
    Object.assign({ fetchImpl: without }, options)
  ).start("llada", {});

  assert.equal(
    withDevice.calls[0].init.body,
    JSON.stringify({ device: "cpu" })
  );
  assert.equal(without.calls[0].init.body, undefined);
});

test("a refused activation rejects and starts no watch", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      {
        ok: false,
        body: { ok: false, message: "cannot run on CPU" },
      },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await assert.rejects(
    () => client.start("diffusiongemma", { device: "cpu" }),
    /cannot run on CPU/
  );
  assert.equal(clock.booked(), 0);
});

test("a model id with awkward characters is encoded", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([ACCEPTED]);
  const client = activationClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("a/b", {});

  assert.equal(fetchImpl.calls[0].url, "/api/models/a%2Fb/activate");
});

// -- whose activation is this (LIFE-03) --

test("a client remembers the operation it was given", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch([ACCEPTED]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada", {});

  assert.equal(client.operation(), 7);
});

test("another operation's ready does not fire our onReady", async () => {
  // The case the finding names: window B supersedes window A's
  // activation between two polls, and A must not navigate for it.
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let ready = 0;
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      ACCEPTED,
      { body: { state: "ready", operation: 8 } },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onReady: () => {
      ready += 1;
    },
  });

  await client.start("llada", {});
  await clock.flush();

  assert.equal(ready, 0);
  assert.equal(clock.booked(), 0);
});

test("another operation's error does not fire our onFailed", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let failure = null;
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      ACCEPTED,
      { body: { state: "error", message: "theirs", operation: 8 } },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onFailed: (message) => {
      failure = message;
    },
  });

  await client.start("llada", {});
  await clock.flush();

  assert.equal(failure, null);
});

test("an observer accepts whichever activation is running", async () => {
  // Observing is for a page that arrived mid-load and owns nothing,
  // so filtering by operation would leave it watching nothing.
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let ready = 0;
  const client = activationClientCreate({
    fetchImpl: fakeFetch(
      statuses({ state: "ready", operation: 99 })
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onReady: () => {
      ready += 1;
    },
  });

  client.observe();
  await clock.flush();

  assert.equal(ready, 1);
});

test("a server that numbers nothing is still followed", async () => {
  // Forward compatibility in reverse: a page from this build against
  // a supervisor that predates operation ids must still work.
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  let ready = 0;
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      { body: { ok: true } },
      { body: { state: "ready" } },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    onReady: () => {
      ready += 1;
    },
  });

  await client.start("llada", {});
  await clock.flush();

  assert.equal(client.operation(), null);
  assert.equal(ready, 1);
});

// -- cancelling --

test("cancel names the operation it is stopping", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([
    ACCEPTED,
    { body: { ok: true } },
  ]);
  const client = activationClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada", {});
  await client.cancel();

  const last = fetchImpl.calls[fetchImpl.calls.length - 1];
  assert.equal(last.url, "/api/models/activate/cancel");
  assert.equal(last.init.body, JSON.stringify({ operation: 7 }));
});

test("cancel stops the watch", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      ACCEPTED,
      { body: { ok: true } },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada", {});
  await client.cancel();
  const ran = await clock.flush();

  assert.equal(ran, 0);
});

test("a refused cancel rejects with the server's reason", async () => {
  // So the menu can say who owns the load instead of showing a
  // button that appears to do nothing.
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch([
      ACCEPTED,
      {
        ok: false,
        body: {
          ok: false,
          message: "smollm3 is loading, started elsewhere",
        },
      },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada", {});

  await assert.rejects(
    () => client.cancel(),
    /started elsewhere/
  );
});

// -- a single read --

test("readOnce reads the status and books nothing", async () => {
  const { activationClientCreate } = load();
  const clock = manualScheduler();
  const client = activationClientCreate({
    fetchImpl: fakeFetch(
      statuses({ state: "error", message: "earlier failure" })
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  const status = await client.readOnce();

  assert.equal(status.message, "earlier failure");
  assert.equal(clock.booked(), 0);
});
