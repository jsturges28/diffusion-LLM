// Tests for the candidate flicker's schedule and its timer.
//
// Strategy: load the shipped file into a fresh vm context beside
// run_candidates.js, with the timer, the clock and the motion
// preference supplied by the test, so a tick happens exactly when a
// test calls it and at exactly the time the test sets. The schedule
// is pure and is read straight back; the timer is driven through fake
// spans that count their writes.
//
// What passing proves is that what a position shows over time is its
// distribution: each candidate for its probability's share of a
// cycle, in rank order, and the glyph for the probability outside
// them. It also proves the display rule matches the canvas, the plan
// keeps only positions that would move and caps them by contest, the
// phases are fixed and spread, and the timer writes only changes,
// stops when its spans leave the page and never starts with motion
// reduced, and that stopping hands each span back as it was drawn.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);
const MASK = "\u2591";

function load(options) {
  const settings = options || {};
  const timers = { next: 1, active: new Map(), cleared: [] };
  const sandbox = {
    prefersReducedMotion: () => !!settings.reducedMotion,
    setInterval: (fn, ms) => {
      const id = timers.next++;
      timers.active.set(id, { fn: fn, ms: ms });
      return id;
    },
    clearInterval: (id) => {
      timers.cleared.push(id);
      timers.active.delete(id);
    },
    document: { hidden: !!settings.hidden },
  };
  for (const name of ["run_candidates.js", "candidate_flicker.js"]) {
    vm.runInNewContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"), sandbox,
      { filename: name }
    );
  }
  sandbox.timers = timers;
  return sandbox;
}

function set(rows) {
  return {
    h: rows[0][0],
    c: rows.map(([id, t, p]) => ({ id: id, t: t, p: p })),
  };
}

function counts(texts) {
  const out = {};
  for (const text of texts) {
    out[text] = (out[text] || 0) + 1;
  }
  return out;
}

// A span that records every write, which is what "writes only what
// changes" is measured against.
function fakeSpan(text) {
  const span = {
    attributes: {},
    style: {},
    writes: 0,
    value: text,
    setAttribute(name, value) {
      span.attributes[name] = value;
    },
    removeAttribute(name) {
      delete span.attributes[name];
    },
  };
  Object.defineProperty(span, "textContent", {
    get: () => span.value,
    set: (next) => {
      span.writes += 1;
      span.value = next;
    },
  });
  return span;
}

const MASKED = { t: " the", m: true, id: 1, c: 0.4 };
const SETTLED = { t: " end", m: false, id: 2, c: 0.9 };

// -- the schedule --

test("a cycle is each candidate's share, in rank order", () => {
  const api = load();

  const texts = api.flickerSchedule(set([
    [1, " the", 0.5], [2, " a", 0.3], [3, " an", 0.2],
  ]), MASK);

  assert.equal(texts.length, api.FLICKER_SLOTS);
  assert.deepEqual(counts(texts), { " the": 10, " a": 6, " an": 4 });
  assert.deepEqual(
    [...new Set(texts)], [" the", " a", " an"]
  );
});

test("probability outside the candidates shows as the glyph", () => {
  // Whatever the model gave tokens beyond the five, so an undecided
  // position reads as mostly undecided.
  const api = load();

  const texts = api.flickerSchedule(set([
    [1, " the", 0.3], [2, " a", 0.2],
  ]), MASK);

  assert.deepEqual(counts(texts), { " the": 6, " a": 4, [MASK]: 10 });
  assert.equal(texts[texts.length - 1], MASK);
});

test("shares round by largest remainder to exactly twenty", () => {
  const api = load();

  const texts = api.flickerSchedule(set([
    [1, " the", 0.42], [2, " a", 0.18], [3, " an", 0.07],
  ]), MASK);

  assert.equal(texts.length, 20);
  assert.deepEqual(
    counts(texts), { " the": 8, " a": 4, " an": 1, [MASK]: 7 }
  );
});

test("a candidate worth under half a slot does not show", () => {
  const api = load();

  const texts = api.flickerSchedule(set([
    [1, " the", 0.9], [2, " a", 0.02],
  ]), MASK);

  assert.equal(texts.includes(" a"), false);
});

test("a candidate draws as the canvas draws a guess", () => {
  const api = load();

  assert.equal(api.flickerText("<|endoftext|>", MASK), MASK);
  assert.equal(api.flickerText("<eos>", MASK), MASK);
  assert.equal(api.flickerText("<|turn>", MASK), MASK);
  assert.equal(api.flickerText("", MASK), MASK);
  assert.equal(api.flickerText(null, MASK), MASK);
  assert.equal(api.flickerText("\n", MASK), "\u21B5");
  assert.equal(api.flickerText("a\tb", MASK), "a\u21E5b");
  assert.equal(api.flickerText(" <b", MASK), " <b");
});

test("a position reserves its longest scheduled text", () => {
  const api = load();

  const texts = api.flickerSchedule(set([
    [1, " the", 0.5], [2, " however", 0.3], [3, "<eos>", 0.2],
  ]), MASK);

  assert.equal(api.flickerWidth(texts), " however".length);
});

// -- the phase --

test("a position steps through its slots once a cycle", () => {
  const api = load();
  const slotMs = api.FLICKER_CYCLE_MS / api.FLICKER_SLOTS;

  const slots = [];
  for (let step = 0; step <= api.FLICKER_SLOTS; step += 1) {
    slots.push(api.flickerSlotAt(step * slotMs, 0));
  }

  assert.deepEqual(slots.slice(0, 20), [...Array(20).keys()]);
  assert.equal(slots[20], 0);
});

test("neighbours start apart, and always at the same place", () => {
  const api = load();

  const starts = [...Array(10).keys()].map(
    (position) => api.flickerSlotAt(0, position)
  );

  assert.deepEqual(
    starts, [...Array(10).keys()].map((p) => api.flickerSlotAt(0, p))
  );
  assert.ok(new Set(starts).size >= 8);
  for (let p = 1; p < starts.length; p += 1) {
    assert.notEqual(starts[p], starts[p - 1]);
  }
});

// -- the plan --

test("only unsettled positions whose set moves are planned", () => {
  const api = load();
  const moving = set([[1, " the", 0.5], [2, " a", 0.3]]);
  const still = set([[1, " the", 0.99]]);
  const tokens = [MASKED, SETTLED, MASKED, MASKED, null];
  const sets = [moving, moving, still, null, moving];

  const plan = api.flickerPlan(tokens, sets, MASK);

  assert.deepEqual([...plan].map((entry) => entry.position), [0]);
});

test("the cap keeps the most contested positions", () => {
  const api = load();
  const count = api.FLICKER_MAX_POSITIONS + 6;
  const tokens = Array(count).fill(MASKED);
  const sets = [...Array(count).keys()].map(
    (position) => set([[1, " the", 0.2 + position * 0.005],
      [2, " a", 0.1]])
  );

  const plan = api.flickerPlan(tokens, sets, MASK);

  assert.equal(plan.length, api.FLICKER_MAX_POSITIONS);
  const kept = new Set(plan.map((entry) => entry.position));
  for (let position = 0; position < api.FLICKER_MAX_POSITIONS;
    position += 1) {
    assert.ok(kept.has(position));
  }
});

test("a layer reads its store at the frame it shows", () => {
  // Borrowing the latest captured frame before a skipped one, and
  // nothing for a frame before any capture or for no store at all.
  const api = load();
  const store = api.runCandidatesAddStream(
    api.runCandidatesCreate(), 0,
    {
      type: "candidates", k: 5, stride: 2, frames: [1, 3],
      sets: [
        [set([[11, " one", 0.6]])], [set([[33, " three", 0.6]])],
      ],
    }
  );
  const oneCanvas = () => 0;

  const borrowed = api.flickerLayer([], [], store, 2, oneCanvas);
  const opening = api.flickerLayer([], [], store, 0, oneCanvas);
  const none = api.flickerLayer([], [], null, 3, oneCanvas);

  assert.equal(borrowed.sets[0].h, 11);
  assert.equal(opening.sets, null);
  assert.equal(none.sets, null);
});

// -- the timer --

function started(api, spanText) {
  const span = fakeSpan(spanText || " the");
  api.flickerStart([{
    spans: [span],
    tokens: [MASKED],
    sets: [set([[1, " the", 0.5], [2, " a", 0.5]])],
  }], MASK);
  return span;
}

test("starting marks the span and reserves its width", () => {
  const api = load();

  const span = started(api);

  assert.equal("data-cycling" in span.attributes, true);
  assert.equal(span.style.width, "4ch");
  assert.equal(api.timers.active.size, 1);
  assert.equal(
    [...api.timers.active.values()][0].ms,
    api.FLICKER_CYCLE_MS / api.FLICKER_SLOTS
  );
});

test("a tick writes a span only when its text changes", () => {
  const api = load();
  const span = started(api);
  const slotMs = api.FLICKER_CYCLE_MS / api.FLICKER_SLOTS;
  api.flickerNow = () => 0;
  api.flickerTick();
  const settled = span.writes;

  api.flickerTick();
  const same = span.writes;
  api.flickerNow = () => slotMs * 10;
  api.flickerTick();

  assert.equal(same, settled);
  assert.equal(span.writes, settled + 1);
  assert.equal(span.textContent, " a");
});

test("a tick while the page is hidden changes nothing", () => {
  const api = load({ hidden: true });
  const span = started(api, "guess");

  api.flickerTick();

  assert.equal(span.textContent, "guess");
});

test("the timer stops once its spans leave the page", () => {
  const api = load();
  const span = started(api);
  const id = [...api.timers.active.keys()][0];

  span.isConnected = false;
  api.flickerTick();

  assert.deepEqual(api.timers.cleared, [id]);
  assert.equal(api.flickerEntries.length, 0);
});

test("nothing starts with motion reduced", () => {
  const api = load({ reducedMotion: true });

  const span = started(api, "guess");

  assert.equal(api.timers.active.size, 0);
  assert.equal("data-cycling" in span.attributes, false);
  assert.equal(span.textContent, "guess");
});

test("stopping clears the timer; starting again replaces it", () => {
  const api = load();
  started(api);
  const first = [...api.timers.active.keys()][0];

  started(api);
  api.flickerStop();

  assert.deepEqual(api.timers.cleared.slice(0, 1), [first]);
  assert.equal(api.timers.active.size, 0);
  assert.equal(api.flickerEntries.length, 0);
});

test("stopping hands each span back as it was drawn", () => {
  // The live view keeps its spans between frames, so a position that
  // settles must not keep the cycling mark, the reserved width or a
  // candidate's text.
  const api = load();
  const span = started(api, "guess");
  api.flickerNow = () => 0;
  api.flickerTick();
  assert.notEqual(span.textContent, "guess");

  api.flickerStop();

  assert.equal("data-cycling" in span.attributes, false);
  assert.equal(span.style.width, "");
  assert.equal(span.textContent, "guess");
});

test("a layer with nothing to cycle starts no timer", () => {
  const api = load();

  api.flickerStart([{ spans: [fakeSpan(" end")], tokens: [SETTLED],
    sets: [set([[1, " end", 0.5], [2, " a", 0.5]])] }], MASK);

  assert.equal(api.timers.active.size, 0);
});
