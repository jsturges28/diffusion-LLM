// Tests for the shared download transport.
//
// Strategy: load the shipped file into a fresh vm context and drive
// it with a manual scheduler and a scripted fetch, the pattern
// activation_client.test.js established. No DOM, no network, no real
// clock, so a test runs exactly as many polls as it means to.
//
// What this replaced: three readers of one endpoint. menu.js polled
// a bound row every 500ms and re-read the status once at load;
// download_toast.js ran its own 1000ms loop from boot to unload. On
// the downloading row two of them ran at once, on different clocks,
// each deciding for itself what a terminal state meant. Nothing
// tested any of it, and nothing could cancel, because until
// TRUST-04 the server had no cancel to call.
//
// Passing proves one watcher feeds every listener, that a failed
// poll backs off instead of hammering, that a page which only
// observes cannot cancel a download it did not start, and that
// stopping a watch is not the same as stopping a fetch.
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
  "download_client.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "download_client.js",
  });
  return sandbox;
}

async function settle() {
  for (let i = 0; i < 8; i += 1) {
    await Promise.resolve();
  }
}

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
    delays: () => Array.from(handles.values()).map((e) => e.ms),
  };
}

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

const ACCEPTED = { body: { ok: true, operation: 4 } };

function statuses(...list) {
  return list.map((body) => ({ body }));
}

function downloading(fraction) {
  return {
    state: "downloading",
    target: "llada",
    progress: { fraction },
  };
}

// -- one watcher, many listeners --

test("every listener hears the same reading", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const heard = [[], []];
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses(downloading(0.25))),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe((s) => heard[0].push(s.progress.fraction));
  client.subscribe((s) => heard[1].push(s.progress.fraction));

  client.observe();
  await clock.flush();

  assert.deepEqual(heard[0], [0.25]);
  assert.deepEqual(heard[1], [0.25]);
});

test("one poll serves them, not one poll each", async () => {
  // The duplication this file exists to remove: the row and the
  // toast used to fetch the same URL on two clocks.
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch(statuses(downloading(0.5)));
  const client = downloadClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe(() => {});
  client.subscribe(() => {});
  client.subscribe(() => {});

  client.observe();
  await clock.flush();

  assert.equal(fetchImpl.calls.length, 1);
});

test("unsubscribing stops one listener, not the watch", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const kept = [];
  const dropped = [];
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses(downloading(0.1))),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe((s) => kept.push(s.state));
  const off = client.subscribe((s) => dropped.push(s.state));

  client.observe();
  await clock.flush();
  off();
  await clock.flush();

  assert.equal(kept.length, 2);
  assert.equal(dropped.length, 1);
});

test("a throwing listener does not rob the others", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const heard = [];
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses(downloading(0.3))),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe(() => {
    throw new Error("a render blew up");
  });
  client.subscribe((s) => heard.push(s.state));

  client.observe();
  await clock.flush();

  assert.deepEqual(heard, ["downloading"]);
  assert.equal(clock.booked(), 1, "the loop survived it");
});

// -- the loop --

test("a terminal state keeps being reported", async () => {
  // Unlike an activation watch. A finished download stays
  // reportable until it is acknowledged, because the page showing
  // the completion toast may not be the page that started it.
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses({ state: "done" })),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe(() => {});

  client.observe();
  await clock.flush();

  assert.equal(clock.booked(), 1);
});

// -- the idle cadence --

function cadenceClient(script) {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(script),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    pollMs: 500,
    idlePollMs: 5000,
    retryMs: 1000,
  });
  client.subscribe(() => {});
  return { client, clock };
}

test("nothing to report is asked about slowly", async () => {
  // Every page loads the toast and the toast observes from boot to
  // unload, so the fast rate used to run for the life of the window
  // whether or not a download existed.
  const { client, clock } = cadenceClient(statuses({ state: "idle" }));

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [5000]);
});

test("a running download is still asked about quickly", async () => {
  const { client, clock } = cadenceClient(statuses(downloading(0.25)));

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [500]);
});

test("a finished download keeps the fast rate", async () => {
  // Terminal is not idle. A finished download stays reportable until
  // it is acknowledged, and slowing here would delay the toast that
  // ends the story.
  const { client, clock } = cadenceClient(statuses({ state: "done" }));

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [500]);
});

test("an errored download keeps the fast rate", async () => {
  const { client, clock } = cadenceClient(statuses({ state: "error" }));

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [500]);
});

test("the rate follows the state in both directions", async () => {
  // The half worth pinning. Speeding up on a download that started
  // elsewhere is what makes a slow idle rate affordable, and slowing
  // down again is what stops one download costing the rest of the
  // session at two requests a second.
  const { client, clock } = cadenceClient(
    statuses({ state: "idle" }, downloading(0.5), { state: "idle" })
  );

  client.observe();
  await clock.flush();
  const first = clock.delays();
  await clock.flush();
  const during = clock.delays();
  await clock.flush();

  assert.deepEqual(first, [5000]);
  assert.deepEqual(during, [500]);
  assert.deepEqual(clock.delays(), [5000]);
});

test("a reading with no state at all stays quick", async () => {
  // Only "idle" earns the slow rate. A malformed or truncated
  // reading is not evidence that nothing is happening, and guessing
  // that it is would stall the bar on the one page that can see it.
  const { client, clock } = cadenceClient(statuses({}));

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [500]);
});

test("the shipped defaults are two different rates", async () => {
  // Every test above injects both intervals, which proves the
  // machinery and nothing about what the app runs: the page calls
  // downloadClientCreate() with no cadence options at all. With the
  // defaults equal, the backoff would be dead on arrival and every
  // other test here would still pass.
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(
      statuses({ state: "idle" }, downloading(0.5))
    ),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe(() => {});

  client.observe();
  await clock.flush();
  const idle = clock.delays()[0];
  await clock.flush();
  const active = clock.delays()[0];

  assert.ok(
    idle > active,
    `idle default ${idle} should exceed active default ${active}`
  );
});

// -- catching up without waiting out the interval --

test("checkNow reads at once instead of waiting", async () => {
  const { client, clock } = cadenceClient(statuses({ state: "idle" }));
  client.observe();
  await clock.flush();
  assert.deepEqual(clock.delays(), [5000]);

  client.checkNow();

  assert.deepEqual(clock.delays(), [0]);
});

test("checkNow replaces the pending poll rather than adding one",
  async () => {
    // Two loops against one endpoint is the defect this whole module
    // was written to remove, so a focus event must not start a
    // second one.
    const { client, clock } = cadenceClient(
      statuses({ state: "idle" })
    );
    client.observe();
    await clock.flush();

    client.checkNow();
    client.checkNow();

    assert.equal(clock.booked(), 1);
  });

test("checkNow does nothing when nothing is being watched", async () => {
  // A page that never observed has nothing to catch up on, and
  // starting a loop from a focus event would make it poll anyway.
  const { client, clock } = cadenceClient(statuses({ state: "idle" }));

  assert.equal(client.checkNow(), false);
  assert.equal(clock.booked(), 0);
});

test("checkNow does nothing after stopping", async () => {
  const { client, clock } = cadenceClient(statuses({ state: "idle" }));
  client.observe();
  await clock.flush();
  client.stop();

  assert.equal(client.checkNow(), false);
  assert.equal(clock.booked(), 0);
});

test("a failed poll backs off rather than hammering", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch([new Error("offline")]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
    pollMs: 500,
    retryMs: 1000,
  });
  client.subscribe(() => {});

  client.observe();
  await clock.flush();

  assert.deepEqual(clock.delays(), [1000]);
});

test("stopping books nothing further", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses(downloading(0.4))),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  client.subscribe(() => {});

  client.observe();
  await clock.flush();
  client.stop();

  assert.equal(clock.booked(), 0);
});

// -- starting, and what that claims --

test("a start posts to the model's download endpoint", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([ACCEPTED]);
  const client = downloadClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada");

  assert.equal(
    fetchImpl.calls[0].url,
    "/api/models/llada/download"
  );
  assert.equal(fetchImpl.calls[0].init.method, "POST");
});

test("a start adopts the operation it was given", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch([ACCEPTED]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada");

  assert.equal(client.operation(), 4);
});

test("a refused start rejects with the server's words", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch([
      { body: { ok: false, message: "already running" } },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await assert.rejects(
    () => client.start("llada"),
    /already running/
  );
});

test("observing claims no operation", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch(statuses(downloading(0.2))),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  client.observe();
  await clock.flush();

  assert.equal(client.operation(), null);
});

// -- cancelling --

test("a cancel names the download it started", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([
    ACCEPTED,
    { body: { ok: true } },
  ]);
  const client = downloadClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.start("llada");
  await client.cancel();

  const sent = fetchImpl.calls[1];
  assert.equal(sent.url, "/api/models/download/cancel");
  assert.deepEqual(JSON.parse(sent.init.body), { operation: 4 });
});

test("an observer cancels without naming one", async () => {
  // It has no number to name, so the server decides. A page that
  // can see a download can stop it; it just cannot claim it was
  // stopping a specific one.
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([{ body: { ok: true } }]);
  const client = downloadClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  client.observe();
  await client.cancel();

  assert.equal(fetchImpl.calls[0].init.body, undefined);
});

test("a refused cancel rejects and says which", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch([
      ACCEPTED,
      {
        ok: false,
        body: { ok: false, message: "another window" },
      },
    ]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  await client.start("llada");

  await assert.rejects(
    () => client.cancel(),
    (error) => {
      assert.match(error.message, /another window/);
      assert.equal(error.operation, 4);
      return true;
    }
  );
});

test("a cancel gives up the claim either way", async () => {
  // Otherwise a second press would name a download the server has
  // already forgotten.
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const client = downloadClientCreate({
    fetchImpl: fakeFetch([ACCEPTED, { body: { ok: true } }]),
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });
  await client.start("llada");

  await client.cancel();

  assert.equal(client.operation(), null);
});

// -- ack --

test("an ack posts and survives a failure", async () => {
  const { downloadClientCreate } = load();
  const clock = manualScheduler();
  const fetchImpl = fakeFetch([new Error("offline")]);
  const client = downloadClientCreate({
    fetchImpl,
    schedule: clock.schedule,
    unschedule: clock.unschedule,
  });

  await client.ack();

  assert.equal(
    fetchImpl.calls[0].url,
    "/api/models/download/ack"
  );
});

// -- the terminal predicate --

test("done and error end a fetch, idle does not", () => {
  const { downloadClientIsTerminal } = load();

  assert.equal(downloadClientIsTerminal("done"), true);
  assert.equal(downloadClientIsTerminal("error"), true);
  assert.equal(downloadClientIsTerminal("idle"), false);
  assert.equal(downloadClientIsTerminal("downloading"), false);
});
