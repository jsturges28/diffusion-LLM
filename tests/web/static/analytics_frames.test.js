// The Analytics viewer reads one run the same way in either shape.
//
// Strategy: load the whole Analytics page into the DOM stub beside
// this file and hand it the two payloads the server can now send for
// one run: per-frame arrays, and a flat list of positions whose
// frames are its prefixes. Then ask the page the questions its
// renderers ask, at every scrub position, and require the same
// answers.
//
// The flat payload is why stage two exists. On the maintainer's
// 2,048-token runs the per-frame form is a 130 MiB download and the
// flat one is 1.04 MiB, measured through the real endpoint. None of
// that is worth anything if the page draws a different run, and
// source inspection cannot see which run it draws.
//
// Diffusion is deliberately part of this. Its positions change
// behind the newest one, so it keeps the per-frame form, and a test
// that only exercised the flat path would not notice the day the
// page stopped handling the other.
//
// Passing proves scrubbing, the final frame, entropy, commit order
// and the diff baseline all read the same from either shape, and
// that a run with changing positions still reads per-frame.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

const WORDS = ["The", " cat", " sat", " on", " the", " mat"];

function positions(words) {
  return words.map((word, at) => ({
    t: word,
    m: false,
    id: 1000 + at,
    c: +(0.5 + at / 100).toFixed(4),
    e: +(1.5 - at / 100).toFixed(4),
  }));
}

// The same run written the way it used to be: every prefix, spelled
// out. This is what a diffusion run still sends, and what an old
// autoregressive run still has on disk.
function framed(flat) {
  return flat.map((_, at) => flat.slice(0, at + 1));
}

// The page fetches its catalog while booting. None of these tests
// are about that, so it gets an empty one: enough shape for boot to
// finish, and stated here rather than guessed at by the shared stub,
// which has no business knowing this page's endpoints.
function bootFetch() {
  return function (url) {
    // The catalog answers with a bare array; everything else this
    // page asks for during boot is happy with an empty object.
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : { success: true, collections: [] };
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    });
  };
}

function page() {
  return loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
}

// Ask a loaded page for everything a renderer would read.
function readAll(context, payload) {
  const series = context.overlaySeriesOf(payload, false);
  const count = context.overlaySeriesLength(series);
  const frames = [];
  for (let index = 0; index < count; index++) {
    frames.push(context.overlaySeriesAt(series, index));
  }
  return {
    count,
    frames,
    finalIndex: context.overlaySeriesFinalIndex(series),
    final: context.overlaySeriesFinal(series),
    hasEntropy: context.framesHaveEntropy(series),
    entropy: context.entropySeriesFrom(series),
    commitSteps: context.overlaySeriesCommitSteps(series),
  };
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

// -- the two shapes describe one run --

test("both payload shapes give the same frame count", () => {
  const { context } = page();
  const flat = positions(WORDS);

  assert.equal(
    readAll(context, { positions: flat }).count,
    readAll(context, { frames: framed(flat) }).count
  );
});

test("both shapes give the same tokens at every frame", () => {
  // The assertion the whole change rests on. Every scrub position,
  // not just the ends, because a prefix slice that was off by one
  // would still agree at the last frame.
  const { context } = page();
  const flat = positions(WORDS);

  const fromFlat = readAll(context, { positions: flat });
  const fromFrames = readAll(context, { frames: framed(flat) });

  assert.deepEqual(host(fromFlat.frames), host(fromFrames.frames));
});

test("both shapes agree on the final frame", () => {
  const { context } = page();
  const flat = positions(WORDS);

  const fromFlat = readAll(context, { positions: flat });
  const fromFrames = readAll(context, { frames: framed(flat) });

  assert.equal(fromFlat.finalIndex, fromFrames.finalIndex);
  assert.deepEqual(host(fromFlat.final), host(fromFrames.final));
});

test("both shapes give the same entropy series", () => {
  const { context } = page();
  const flat = positions(WORDS);

  const fromFlat = readAll(context, { positions: flat });
  const fromFrames = readAll(context, { frames: framed(flat) });

  assert.equal(fromFlat.hasEntropy, true);
  assert.deepEqual(
    host(fromFlat.entropy), host(fromFrames.entropy)
  );
});

test("both shapes give the same commit order", () => {
  // The one consumer that folds over every frame. The flat path
  // answers without assembling any, so this is where an optimisation
  // that disagreed with the general walk would show.
  const { context } = page();
  const flat = positions(WORDS);

  const fromFlat = readAll(context, { positions: flat });
  const fromFrames = readAll(context, { frames: framed(flat) });

  assert.deepEqual(
    Array.from(fromFlat.commitSteps),
    Array.from(fromFrames.commitSteps)
  );
});

test("a masked position is left uncoloured", () => {
  // Zero means "settled at step 0"; -1 means "never resolved" and
  // the overlay leaves it alone. An append run from the samplers
  // here has no masked position, but the shortcut must not be the
  // one place that forgets the distinction: it would paint an
  // unresolved position as though the model had committed to it.
  const { context } = page();
  const flat = positions(WORDS);
  flat[2] = { ...flat[2], m: true };

  const steps = context.overlaysAppendCommitSteps(flat);

  assert.deepEqual(Array.from(steps), [0, 0, -1, 0, 0, 0]);
});

test("the shortcut agrees with the walk on a masked run", () => {
  // The same run through both paths, so the shortcut cannot drift
  // from the general definition it is standing in for.
  const { context } = page();
  const flat = positions(WORDS);
  flat[2] = { ...flat[2], m: true };

  const shortcut = context.overlaysAppendCommitSteps(flat);
  const walked = context.overlaysComputeCommitSteps(
    context.overlaysFrameReader(framed(flat)), flat.length
  );

  assert.deepEqual(Array.from(shortcut), Array.from(walked));
});

test("an append run settles every position at once", () => {
  // Not merely equal to the frame walk, but equal to the right
  // thing: a position appears at its final value and never moves.
  const { context } = page();

  const steps = readAll(
    context, { positions: positions(WORDS) }
  ).commitSteps;

  assert.deepEqual(Array.from(steps), [0, 0, 0, 0, 0, 0]);
});

// -- a run whose positions change --

test("a per-frame run is still read per frame", () => {
  // Diffusion revises positions behind the newest one, so its
  // frames are not prefixes and there is no flat form of it.
  const { context } = page();
  const canvas = positions(WORDS);
  const revised = framed(canvas).map((frame) => frame.slice());
  revised[revised.length - 1] = revised[revised.length - 1]
    .map((token, at) => (at === 0 ? { ...token, id: 77 } : token));

  const series = context.overlaySeriesOf(
    { frames: revised }, false
  );

  assert.equal(series.positions, null);
  assert.equal(
    context.overlaySeriesLength(series), revised.length
  );
});

test("a changed position moves its commit step", () => {
  // The behaviour the append shortcut must never be applied to.
  const { context } = page();
  const canvas = positions(WORDS);
  const revised = framed(canvas).map((frame) => frame.slice());
  const last = revised.length - 1;
  revised[last] = revised[last].map(
    (token, at) => (at === 0 ? { ...token, id: 77 } : token)
  );

  const steps = readAll(context, { frames: revised }).commitSteps;

  assert.equal(steps[0], last);
  assert.equal(steps[1], 0);
});

// -- the pre-edit baseline --

test("the baseline reads from either shape too", () => {
  const { context } = page();
  const flat = positions(WORDS);

  const fromFlat = context.overlaySeriesOf(
    { original_positions: flat }, true
  );
  const fromFrames = context.overlaySeriesOf(
    { original_frames: framed(flat) }, true
  );

  assert.deepEqual(
    host(context.overlaySeriesFinal(fromFlat)),
    host(context.overlaySeriesFinal(fromFrames))
  );
});

test("an absent baseline is empty rather than broken", () => {
  const { context } = page();

  const series = context.overlaySeriesOf({}, true);

  assert.equal(context.overlaySeriesPresent(series), false);
  assert.equal(context.overlaySeriesAt(series, 0), null);
  assert.equal(context.overlaySeriesFinal(series), null);
});

// -- edges --

test("a frame past the end is null, not a wrap", () => {
  const { context } = page();
  const series = context.overlaySeriesOf(
    { positions: positions(WORDS) }, false
  );

  assert.equal(context.overlaySeriesAt(series, 99), null);
  assert.equal(context.overlaySeriesAt(series, -1), null);
});

test("a single-position run still has a final frame", () => {
  const { context } = page();
  const series = context.overlaySeriesOf(
    { positions: positions(["only"]) }, false
  );

  assert.equal(context.overlaySeriesFinalIndex(series), 0);
  assert.equal(context.overlaySeriesFinal(series).length, 1);
});
