// Tests for the durable UI-state write-through in overlays.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// way detail_requests.test.js does, and supply the handful of globals
// its persistence half touches: localStorage, fetch, the timer pair,
// and the two event targets. overlays.js does no DOM work at the top
// level, so evaluating it costs nothing but the definitions. The
// timers are fakes, which is what makes the debounce observable
// rather than something to wait out.
//
// What passing proves is the DATA-02 slice's client half. Three
// separate ways a write could be lost or lie about itself:
//
//   - Collections went through a 250 ms debounce. Navigating inside
//     that window discarded the timer with the document, and the next
//     page's hydrate then wrote the older server copy back over it.
//     One window, no race, and no way to recover the filing. They
//     have since left this mechanism entirely: the server owns them
//     and the key is no longer writable here at all, so what remains
//     below is the immediate-write branch they proved the need for.
//   - Every other key still debounces, so those needed a flush on the
//     way out rather than an exemption.
//   - A failed PUT resolved silently. The tab kept showing the change
//     as saved because localStorage already had it, so the value only
//     disappeared later, somewhere else.
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
  "overlays.js"
);

const SETTINGS = "diffusion_settings";

// A fresh global per test. Returns the sandbox plus the recorders,
// so a test can read what was sent and fire what is pending.
function load(options) {
  const settings = options || {};
  const puts = [];
  const timers = [];
  const listeners = { document: {}, window: {} };
  const stored = {};

  function respond() {
    if (settings.reject) {
      return Promise.reject(new Error("offline"));
    }
    return Promise.resolve({ ok: settings.ok !== false });
  }

  const sandbox = {
    localStorage: {
      getItem: (key) =>
        Object.prototype.hasOwnProperty.call(stored, key)
          ? stored[key]
          : null,
      setItem: (key, value) => {
        stored[key] = value;
      },
    },
    fetch: (url, init) => {
      puts.push({ url, init });
      return respond();
    },
    setTimeout: (fn) => {
      timers.push(fn);
      return timers.length; // Truthy id, which the code checks.
    },
    clearTimeout: (id) => {
      if (id) {
        timers[id - 1] = null;
      }
    },
    document: {
      visibilityState: "visible",
      addEventListener: (name, fn) => {
        listeners.document[name] = fn;
      },
    },
    window: {
      addEventListener: (name, fn) => {
        listeners.window[name] = fn;
      },
    },
  };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return { sandbox, puts, timers, listeners, stored };
}

// Loaded and armed, for the tests about leaving the page. Arming is
// what persistHydrate does at boot; a test that wants to prove that
// wiring exists calls persistHydrate instead.
function loadArmed(options) {
  const harness = load(options);
  harness.sandbox.persistArmFlush();
  return harness;
}

function fireTimers(timers) {
  for (let i = 0; i < timers.length; i += 1) {
    const fn = timers[i];
    timers[i] = null;
    if (fn) {
      fn();
    }
  }
}

function bodyOf(entry) {
  return JSON.parse(entry.init.body).value;
}

// -- what is written, and when --

test("a key marked immediate skips the debounce", () => {
  // The list is empty today: collections were its only member, and
  // they moved to their own endpoints when the server took ownership
  // of them. The branch stays because the reasoning will apply to
  // the next value that cannot be recomputed, so this marks a key
  // itself rather than relying on one being listed.
  const { sandbox, puts } = load();
  sandbox.PERSIST_IMMEDIATE_KEYS.push(SETTINGS);

  sandbox.persistSet(SETTINGS, "{}");

  assert.equal(puts.length, 1);
  assert.match(puts[0].url, /diffusion_settings$/);
  assert.equal(puts[0].init.method, "PUT");
});

test("a cache key still debounces", () => {
  const { sandbox, puts, timers } = load();

  sandbox.persistSet(SETTINGS, "{}");

  assert.equal(puts.length, 0, "should be waiting on a timer");
  fireTimers(timers);
  assert.equal(puts.length, 1);
});

test("rapid writes to one cache key coalesce", () => {
  const { sandbox, puts, timers } = load();

  sandbox.persistSet(SETTINGS, "{}");
  sandbox.persistSet(SETTINGS, '{"a":1}');
  sandbox.persistSet(SETTINGS, '{"a":2}');
  fireTimers(timers);

  assert.equal(puts.length, 1, "the point of the debounce");
  assert.equal(bodyOf(puts[0]), '{"a":2}');
});

test("localStorage is written before any of that", () => {
  const { sandbox, stored } = load();

  sandbox.persistSet(SETTINGS, '{"live":true}');

  assert.equal(stored[SETTINGS], '{"live":true}');
});

test("an unknown key stays local", () => {
  const { sandbox, puts, timers, stored } = load();

  sandbox.persistSet("not_a_persisted_key", "x");
  fireTimers(timers);

  assert.equal(stored.not_a_persisted_key, "x");
  assert.equal(puts.length, 0);
});

// -- the way out --

test("hiding the page sends what is still pending", () => {
  const { sandbox, puts, listeners } = loadArmed();
  sandbox.persistSet(SETTINGS, '{"unsaved":true}');
  assert.equal(puts.length, 0);

  sandbox.document.visibilityState = "hidden";
  listeners.document.visibilitychange();

  assert.equal(puts.length, 1);
  assert.equal(bodyOf(puts[0]), '{"unsaved":true}');
});

test("merely losing focus does not flush", () => {
  const { sandbox, puts, listeners } = loadArmed();
  sandbox.persistSet(SETTINGS, "{}");

  sandbox.document.visibilityState = "visible";
  listeners.document.visibilitychange();

  assert.equal(puts.length, 0);
});

test("pagehide flushes too", () => {
  const { sandbox, puts, listeners } = loadArmed();
  sandbox.persistSet(SETTINGS, "{}");

  listeners.window.pagehide();

  assert.equal(puts.length, 1);
});

test("a flushed write does not also fire on its timer", () => {
  const { sandbox, puts, timers, listeners } = loadArmed();
  sandbox.persistSet(SETTINGS, "{}");

  listeners.window.pagehide();
  fireTimers(timers);

  assert.equal(puts.length, 1, "sent once, not twice");
});

test("a flush marks the request to outlive the document", () => {
  const { sandbox, puts, listeners } = loadArmed();
  sandbox.persistSet(SETTINGS, "{}");

  listeners.window.pagehide();

  assert.equal(puts[0].init.keepalive, true);
});

test("an oversized body is sent without keepalive", () => {
  // The keepalive budget is small and shared, and a request over it
  // is refused outright, which would be worse than a slow one.
  const { sandbox, puts, listeners } = loadArmed();
  sandbox.persistSet("diffusion_prompt_history", "x".repeat(60000));

  listeners.window.pagehide();

  assert.equal(puts.length, 1);
  assert.notEqual(puts[0].init.keepalive, true);
});

test("an ordinary debounced write is not keepalive", () => {
  const { sandbox, puts, timers } = load();

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);

  assert.notEqual(puts[0].init.keepalive, true);
});

test("boot arms the flush", () => {
  // Registered from persistHydrate because that is the one call every
  // page already makes, and overlays.js does nothing at the top level.
  const { sandbox, listeners } = load();

  sandbox.persistHydrate(() => {});

  assert.equal(typeof listeners.document.visibilitychange, "function");
  assert.equal(typeof listeners.window.pagehide, "function");
});

// -- saying so when it fails --

test("a rejected write reaches the handler", async () => {
  const { sandbox, timers } = load({ reject: true });
  const seen = [];
  sandbox.persistOnFailure(SETTINGS, (key) => seen.push(key));

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(seen, [SETTINGS]);
});

test("a 4xx reaches the handler as well", async () => {
  // The bug this closes: a rejected write resolves rather than
  // throwing, so a status-blind caller reads it as success.
  const { sandbox, timers } = load({ ok: false });
  const seen = [];
  sandbox.persistOnFailure(SETTINGS, (key) => seen.push(key));

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(seen, [SETTINGS]);
});

test("a successful write says nothing", async () => {
  const { sandbox, timers } = load();
  const seen = [];
  sandbox.persistOnFailure(SETTINGS, (key) => seen.push(key));

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(seen, []);
});

test("a key with no handler fails quietly", async () => {
  // Caches are allowed to fail silently; only the key that cannot be
  // recomputed is worth interrupting the user over.
  const { sandbox, timers } = load({ reject: true });

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.ok(true, "no throw");
});

test("a throwing handler does not break the next write", async () => {
  const { sandbox, puts, timers } = load({ ok: false });
  sandbox.persistOnFailure(SETTINGS, () => {
    throw new Error("reporter is broken");
  });

  sandbox.persistSet(SETTINGS, "{}");
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));
  sandbox.persistSet(SETTINGS, '{"a":1}');
  fireTimers(timers);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(puts.length, 2);
});
