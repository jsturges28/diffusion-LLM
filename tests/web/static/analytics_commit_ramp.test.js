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

const { loadPage, ANALYTICS_SCRIPTS } = require("./dom_stub.js");

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

// A page that has opened one run, which opens on its final frame.
// ``baselineWidth`` gives the run a pre-edit series of its own,
// deliberately shorter so the two denominators cannot be confused,
// and the edit that branched it, so the page stacks the two layers.
function pageWithRun(baselineWidth) {
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  const { context } = page;
  const data = {
    frames: canvasFrames(WORDS.length),
    records_available: true,
  };
  if (baselineWidth) {
    data.original_frames = canvasFrames(baselineWidth);
    data.remask_edits = [{ frame_index: 1, token_positions: [3] }];
  }
  context.renderRunOverlays(data);
  return { page, context, data, last: data.frames.length - 1 };
}

// The spans the newest render drew. The stub keeps each document
// fragment as a node, and setting textContent leaves a node's
// children in place where a browser removes them, so every render's
// fragment stays behind and the newest is the last.
function renderedSpans(output) {
  const last = output.children[output.children.length - 1];
  if (last && last.tag === null) {
    return last.children;
  }
  return output.children;
}

// The spans one layer draws: the only layer of a run that stands
// alone, or the named one of the two a pre-edit snapshot stacks.
function layerSpans(output, layerClass) {
  if (!layerClass) {
    return renderedSpans(output);
  }
  const layers = output.children.filter(
    (child) => child.classList.contains(layerClass)
  );
  assert.ok(layers.length > 0, "no " + layerClass + " layer");
  return renderedSpans(layers[layers.length - 1]);
}

// What Commit Order paints each position of a layer, read off the
// spans the real renderer drew for it.
function commitColors(page, layerClass) {
  page.context.setOverlayMode("commit");
  const output = page.registry.get("overlay-output");
  return layerSpans(output, layerClass).map(
    (span) => span.style.color
  );
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
  const { page, context, last } = pageWithRun(null);

  const colors = commitColors(page);

  for (let at = 0; at <= last; at++) {
    assert.equal(
      colors[at],
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
  const { page, last } = pageWithRun(null);

  const colors = commitColors(page);

  assert.notEqual(colors[0], colors[last]);
});

test("the pre-edit layer spans the baseline, not the branch", () => {
  // The second undeclared identifier. The baseline is shorter here,
  // so reusing the branch's frame count would stretch its ramp and
  // the two layers would disagree about what "late" means.
  const { page, context } = pageWithRun(4);

  const original = commitColors(page, "token-layer-original");
  const edited = commitColors(page, "token-layer-edited");

  assert.equal(original[3], context.commitColor(3, 3));
  assert.notEqual(original[3], edited[3]);
});

test("a run with no baseline still colours its own layer", () => {
  // The common case: most saved runs were never edited, so the
  // baseline is absent and its length is 0. That must leave the
  // primary layer alone rather than taking the whole overlay down
  // with a negative maximum.
  const { page, context, data, last } = pageWithRun(null);

  const colors = commitColors(page);

  assert.equal(context.overlaySeriesPresent(data.baseline), false);
  assert.equal(colors[last], context.commitColor(last, last));
});

test("the overlay paints a span for every position", () => {
  // The case that would have thrown outright on the old code, before
  // a single token was painted.
  const { page, context, data, last } = pageWithRun(null);

  const colors = commitColors(page);

  assert.equal(colors.length, data.frames[last].length);
  assert.equal(colors[0], context.commitColor(0, last));
  assert.equal(colors[last], context.commitColor(last, last));
});
