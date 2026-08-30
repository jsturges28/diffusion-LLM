// Tests for the shared mask-opacity curve in overlays.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// pattern the other browser tests use, and call the curve directly.
// It is pure over its argument, so no DOM and no storage are
// involved.
//
// The curve lived in app.js until it was needed on both pages, which
// is also why it had no test of its own: app.js cannot be loaded this
// way, so its behavior was only ever asserted by reading the source.
// Now it can be run.
//
// What the curve is for: a position that has not settled is drawn
// faded by how sure the model is of what it is holding there, so a
// canvas shows its own certainty forming. What passing proves is the
// shape of that claim: zero confidence is nearly invisible, full
// confidence is solid, the climb between them is monotonic and
// concave, and an unmeasured position is not drawn as a hopeless one.
//
// The concavity is the part chosen from data rather than taste. A
// masked position's confidence is skewed low (median 0.11 to 0.21
// across a measured LLaDA run), so a linear ramp crowds a whole
// canvas into the bottom of the range and the grading stops being
// visible. A test pins it because "make it linear, it is simpler" is
// an obvious and wrong simplification.
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

function load() {
  const sandbox = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return sandbox;
}

test("a hopeless position sits at the floor", () => {
  const sandbox = load();

  assert.equal(
    sandbox.overlaysMaskOpacity(0), sandbox.MASK_OPACITY_FLOOR
  );
});

test("a certain position is solid", () => {
  const sandbox = load();

  assert.equal(sandbox.overlaysMaskOpacity(1), 1);
});

test("an unmeasured position is solid, not hopeless", () => {
  // The distinction this floor makes load-bearing. LLaDA's opening
  // frame, every run saved before the capture, and DiffusionGemma
  // without the Entropy Signal carry no number at all. Treating that
  // as zero would render a whole canvas at 5 percent and call it a
  // measurement.
  const sandbox = load();

  assert.equal(sandbox.overlaysMaskOpacity(null), 1);
  assert.equal(sandbox.overlaysMaskOpacity(undefined), 1);
  assert.notEqual(
    sandbox.overlaysMaskOpacity(undefined),
    sandbox.overlaysMaskOpacity(0)
  );
});

test("the floor leaves a position findable, not gone", () => {
  // Deliberately near-invisible: a token the model has no opinion
  // about should take looking for. But it still renders, because
  // hovering it is how you read what it is.
  const sandbox = load();
  const floor = sandbox.MASK_OPACITY_FLOOR;

  assert.ok(floor > 0, "an opacity of zero is an unhoverable span");
  assert.ok(floor < 0.15, "the floor should read as barely there");
});

test("the climb is monotonic", () => {
  const sandbox = load();
  const steps = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1];

  let previous = -1;
  for (const c of steps) {
    const value = sandbox.overlaysMaskOpacity(c);
    assert.ok(
      value > previous,
      `opacity fell going from ${previous} to ${value} at c=${c}`
    );
    previous = value;
  }
});

test("the climb is concave, not linear", () => {
  // A linear ramp on a left-skewed distribution puts a typical frame
  // in a band too narrow to see. The midpoint sitting well above the
  // straight line between the endpoints is what spends the channel
  // where the confidences actually are.
  const sandbox = load();
  const floor = sandbox.MASK_OPACITY_FLOOR;
  const linearMidpoint = floor + (1 - floor) * 0.5;

  const actual = sandbox.overlaysMaskOpacity(0.5);

  assert.ok(
    actual > linearMidpoint + 0.1,
    `expected a concave lift, got ${actual} against ${linearMidpoint}`
  );
});

test("a typical frame spreads across the channel", () => {
  // The failure this replaced: the previous curve mapped a real
  // frame's confidences into 0.48 to 0.65, and a 1.35x spread on
  // 14px text is not a gradient anyone can see. These three are the
  // p10, p50 and p90 confidences measured on one frame of a real
  // LLaDA run.
  const sandbox = load();

  const low = sandbox.overlaysMaskOpacity(0.05);
  const mid = sandbox.overlaysMaskOpacity(0.115);
  const high = sandbox.overlaysMaskOpacity(0.21);

  assert.ok(high / low > 1.8, `spread was only ${high / low}`);
  assert.ok(mid > low && mid < high);
});

test("a confidence outside the unit range is clamped", () => {
  const sandbox = load();

  assert.equal(sandbox.overlaysMaskOpacity(-0.5), 0.05);
  assert.equal(sandbox.overlaysMaskOpacity(4), 1);
});
