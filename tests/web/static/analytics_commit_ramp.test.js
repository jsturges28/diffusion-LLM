// Analytics colours Commit Order against the run's own frame count.
//
// Strategy: load the Analytics page into the DOM stub beside this
// file, hand it a run whose positions settle at known, *different*
// steps, and require each position's colour to equal the shared ramp
// evaluated at that step over the run's last frame index.
//
// A wrong denominator does
// not shift the colours a little, it collapses them: commitColor
// clamps, so any non-positive maximum paints every position the same
// green and the run reads as though nothing was ordered at all.
//
// The bug being pinned: renderCommitOverlay asked for two identifiers
// this page never declares, `frames` and `original`. In a browser the
// first silently resolves to `window.frames`, whose length is the
// iframe count, so the maximum step became -1; the second has no
// browser equivalent at all, and there is no try/catch on the page,
// so picking Commit Order threw before a token was painted. The
// generator got this right by declaring a local first, which is why
// the same overlay worked there and not here.
//
// Passing proves the primary layer's ramp spans the run, the pre-edit
// layer's ramp spans the *baseline* rather than the branch, and the
// overlay paints through the real renderer without throwing.

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
  }));
}

// A diffusion-shaped canvas: fixed width, with position p holding a
// different token until frame p, so commit steps come out 0, 1, 2 and
// so on. Deliberately not the prefix shape used elsewhere in these
// tests: an append-only run settles every position at step 0, which
// would make all the colours equal for a legitimate reason and hide a
// wrong denominator behind a passing test. Commit Order is only
// offered for diffusion runs anyway.
function canvasFrames(width) {
  const settled = positions(WORDS.slice(0, width));
  const frames = [];
  for (let at = 0; at < width; at++) {
    frames.push(settled.map((token, p) => (
      p <= at ? token : { t: "?", m: false, id: 9000 + p }
    )));
  }
  return frames;
}

function bootFetch() {
  return function (url) {
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

// A page holding one run, scrubbed to its final frame, with the
// memoized commit data cleared the way loading a run clears it.
// ``baselineWidth`` gives the run a pre-edit series of its own,
// deliberately shorter so the two denominators cannot be confused.
function pageWithRun(baselineWidth) {
  const { context } = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  const data = { frames: canvasFrames(WORDS.length) };
  if (baselineWidth) {
    data.original_frames = canvasFrames(baselineWidth);
  }
  data.series = context.overlaySeriesOf(data, false);
  data.baseline = context.overlaySeriesOf(data, true);
  context.overlayData = data;
  context.overlayCommitSteps = null;
  context.overlayOriginalCommitSteps = null;
  context.overlayFrameIndex = data.frames.length - 1;
  return { context, data, last: data.frames.length - 1 };
}

// Run the overlay with the renderer replaced by a recorder, so the
// colour callbacks can be asked about any position without going
// through a layout. This captures exactly the seam that broke: what
// renderCommitOverlay hands the renderer.
function capture(context) {
  let seen = null;
  context.renderOverlayTokens = function (opts) {
    seen = opts;
  };
  context.renderCommitOverlay();
  assert.notEqual(seen, null, "the overlay rendered nothing");
  return seen;
}

// The stub keeps a document fragment as a node rather than splicing
// its children into the parent, so the spans sit one level down.
// Detected rather than assumed, so this still reads correctly if the
// stub ever grows real fragment semantics.
function renderedSpans(output) {
  const first = output.children[0];
  if (first && first.tag === null) {
    return first.children;
  }
  return output.children;
}

test("the fixture really does spread its commit steps", () => {
  // Guarding the guard. Every assertion below is only meaningful if
  // the positions settle at different steps, and the shape that does
  // that is not obvious, so state it before relying on it.
  const { context, data } = pageWithRun(null);

  const steps = context.overlaySeriesCommitSteps(data.series);

  assert.deepEqual(Array.from(steps), [0, 1, 2, 3, 4, 5]);
});

test("every position takes the ramp at its own commit step", () => {
  // The exact statement, position by position. An off-by-one or a
  // borrowed denominator changes at least one of these.
  const { context, data, last } = pageWithRun(null);

  const opts = capture(context);

  for (let at = 0; at <= last; at++) {
    assert.equal(
      opts.colorFor(at, data.frames[last][at]),
      context.commitColor(at, last),
      "position " + at + " is off the ramp"
    );
  }
});

test("the ramp's ends are different colours", () => {
  // Stated separately because it is the symptom a reader would see.
  // A clamped-away maximum keeps every call legal and returns the
  // same green for all of them, so the comparison above could in
  // principle agree with a recomputed ramp that was equally flat.
  const { context, data, last } = pageWithRun(null);

  const opts = capture(context);
  const first = opts.colorFor(0, data.frames[last][0]);
  const final = opts.colorFor(last, data.frames[last][last]);

  assert.notEqual(first, final);
});

test("the pre-edit layer spans the baseline, not the branch", () => {
  // The second undeclared identifier. The baseline is shorter here,
  // so reusing the branch's frame count would stretch its ramp and
  // the two layers would disagree about what "late" means.
  const { context } = pageWithRun(4);
  const token = { t: " on", m: false, id: 1003 };

  const opts = capture(context);

  assert.equal(
    opts.originalColorFor(3, token), context.commitColor(3, 3)
  );
  assert.notEqual(
    opts.originalColorFor(3, token), opts.colorFor(3, token)
  );
});

test("a run with no baseline still colours its own layer", () => {
  // The common case: most saved runs were never edited, so the
  // baseline is absent and its length is 0. That must leave the
  // primary layer alone rather than taking the whole overlay down
  // with a negative maximum.
  const { context, data, last } = pageWithRun(null);

  const opts = capture(context);

  assert.equal(context.overlaySeriesPresent(data.baseline), false);
  assert.equal(
    opts.colorFor(last, data.frames[last][last]),
    context.commitColor(last, last)
  );
});

test("the overlay paints through the real renderer", () => {
  // The recorder above cannot catch a failure further down, so run
  // the genuine path once and read the spans it produced. This is the
  // test that would have thrown outright on the old code.
  const { context, data, last } = pageWithRun(null);

  context.renderCommitOverlay();

  const spans = renderedSpans(context.overlayOutput);
  assert.equal(spans.length, data.frames[last].length);
  assert.equal(spans[0].style.color, context.commitColor(0, last));
  assert.equal(
    spans[last].style.color, context.commitColor(last, last)
  );
});
