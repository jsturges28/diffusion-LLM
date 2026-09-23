// The entropy view reads the frame a channel's axes tell it to.
//
// Strategy: load the Analytics page into the DOM stub beside this file
// and hand it runs whose frames disagree with each other, so reading
// the wrong frame produces visibly wrong numbers rather than the same
// answer by luck. Then the three availability states.
//
// The bug being pinned: every reader took entropy off the *final*
// frame. For an autoregressive run that is correct, because a position
// is sampled once and never revisited. For a diffusion run a position
// is re-decided at every denoising step, so the final frame is one
// arbitrary slice of a trajectory, and scrubbing to frame 3 showed the
// values from the last step regardless.
//
// Passing proves a per-position channel still reads the final frame, a
// frame-by-position channel follows the scrub, a run with no manifest
// behaves exactly as it did before, and a channel whose shape this
// build cannot draw says so instead of leaving an empty space that
// looks identical to a dropped signal.

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

function bootFetch() {
  return function (url) {
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : {};
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(""),
    });
  };
}

function page() {
  return loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
}

// Three frames whose entropy differs per frame, so a reader that took
// the wrong one cannot accidentally agree. Frame N holds entropy N.
function framesWithEntropy() {
  return [0, 1, 2].map((frame) =>
    ["a", "b"].map((text, position) => ({
      t: text,
      m: false,
      id: 100 + position,
      c: 0.5,
      e: frame,
    }))
  );
}

function channel(axes) {
  return {
    name: "entropy",
    unit: "nats",
    axes: axes,
    location: "token_record",
    key: "e",
    capture: "always",
  };
}

function run(axes) {
  const record = { run_id: "r1", frames: framesWithEntropy() };
  if (axes !== undefined) {
    record.signals = [channel(axes)];
  }
  return record;
}

// Arrays built inside the vm context are not reference-equal to host
// ones, so deepEqual rejects them on realm rather than on content. The
// other page tests round-trip through JSON for the same reason.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

// -- which frame a shape is read at --

test("a per-position channel reads the final frame", () => {
  // Correct for an autoregressive run: the value is the same in every
  // frame, so the last one is as good as any and is what the charts
  // have always used.
  const { context } = page();
  const data = run(["position"]);
  const series = context.overlaySeriesOf(data, false);

  const at = context.channelFrameIndex(
    context.signalChannel(data, "entropy"), series
  );

  assert.equal(at, 2);
});

test("a frame-by-position channel follows the scrub", () => {
  // The case that was silently wrong. Scrubbed to frame 1, the reader
  // must report frame 1 rather than the last step's values.
  const { context } = page();
  const data = run(["frame", "position"]);
  const series = context.overlaySeriesOf(data, false);
  context.overlayFrameIndex = 1;

  const at = context.channelFrameIndex(
    context.signalChannel(data, "entropy"), series
  );

  assert.equal(at, 1);
});

test("the values follow the frame, not just the index", () => {
  // The index is only useful if the series actually reads it. Frame N
  // holds entropy N, so this catches a reader that computed the right
  // frame and then went to the final one anyway.
  const { context } = page();
  const data = run(["frame", "position"]);
  const series = context.overlaySeriesOf(data, false);

  const early = context.entropySeriesFrom(series, 0);
  const late = context.entropySeriesFrom(series, 2);

  assert.deepEqual(host(early.values), [0, 0]);
  assert.deepEqual(host(late.values), [2, 2]);
});

test("a scrub past the last frame is clamped", () => {
  // Switching from a long run to a short one leaves the slider beyond
  // the new run's end, and reading past it would be undefined.
  const { context } = page();
  const data = run(["frame", "position"]);
  const series = context.overlaySeriesOf(data, false);
  context.overlayFrameIndex = 99;

  const at = context.channelFrameIndex(
    context.signalChannel(data, "entropy"), series
  );

  assert.equal(at, 2);
});

// -- the three availability states --

test("a drawable channel reports ok", () => {
  const { context } = page();

  assert.equal(
    context.entropyAvailability(run(["frame", "position"])), "ok"
  );
  assert.equal(
    context.entropyAvailability(run(["position"])), "ok"
  );
});

test("a shape this build cannot draw reports unsupported", () => {
  // A canvas-level entropy has no per-position bars to draw. Saying
  // so is the point: hiding the section is what a channel lost by
  // accident would also look like.
  const { context } = page();

  assert.equal(
    context.entropyAvailability(run(["canvas"])), "unsupported"
  );
});

test("a run with no entropy at all reports absent", () => {
  const { context } = page();
  const bare = {
    run_id: "r2",
    frames: [[{ t: "a", m: false, id: 1, c: 0.5 }]],
    signals: [channel(["position"])],
  };

  assert.equal(context.entropyAvailability(bare), "absent");
});

// -- runs from before the manifest existed --

test("a run with no manifest is read as it always was", () => {
  // 258 saved runs predate this. Absent has to mean "infer as
  // before", not "unsupported", or the corpus goes dark.
  const { context } = page();

  assert.equal(context.entropyAvailability(run()), "ok");
});

test("an unmanifested run still reads the final frame", () => {
  const { context } = page();
  const data = run();
  const series = context.overlaySeriesOf(data, false);

  const at = context.channelFrameIndex(
    context.signalChannel(data, "entropy"), series
  );

  assert.equal(at, 2);
});

// -- the shape string --

test("axes join in declaration order", () => {
  // Not sorted: ("frame","position") and ("position","frame") would
  // be a distinction nobody makes, and normalising by sorting would
  // hide a genuinely reversed declaration.
  const { context } = page();

  assert.equal(
    context.channelShape(channel(["frame", "position"])),
    "frame|position"
  );
  assert.equal(context.channelShape(channel(["position"])), "position");
  assert.equal(context.channelShape(null), "");
});
