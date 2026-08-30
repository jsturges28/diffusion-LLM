// Tests for the shared mask-opacity curve and the remask summary in
// overlays.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// pattern the other browser tests use, and call the two functions
// directly. Both are pure over their arguments, so no DOM and no
// storage are involved.
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

// ---- The curve ----

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
  // frame carries no number at all, and neither does any run saved
  // before its model measured every position, DiffusionGemma's
  // runs from before its Entropy Signal was removed included.
  // Treating that as zero would render a whole canvas at 5 percent
  // and call it a measurement.
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

// ---- The remask summary ----

// Chart.js renders one line per entry when a tooltip callback returns
// an array, and the box takes the width of the longest. Only that
// longest line can clip, so it is what every width test below reads.
function widest(lines) {
  return lines.reduce((a, b) => (a.length >= b.length ? a : b));
}

test("a small selection is listed in full", () => {
  const sandbox = load();

  assert.deepEqual(
    Array.from(sandbox.overlaysRemaskSummary([2, 3, 4])),
    ["User remasked 3 tokens:", "[2, 3, 4]"]
  );
});

test("one position stays singular", () => {
  const sandbox = load();

  assert.match(
    sandbox.overlaysRemaskSummary([7])[0], /1 token:/
  );
});

test("a large selection is counted rather than listed", () => {
  // The bug: a 44-token edit put 44 numbers on one tooltip line, and
  // a chart tooltip is sized by its longest line, so the box ran off
  // the side of the chart.
  const sandbox = load();
  const positions = [];
  for (let i = 0; i < 44; i++) {
    positions.push(i);
  }

  assert.deepEqual(
    Array.from(sandbox.overlaysRemaskSummary(positions)),
    ["User remasked 44 tokens:", "[0, 1, 2, 3, 4, ... and 39 others]"]
  );
});

test("the count and the list are separate lines", () => {
  // Truncating alone was not enough: five positions still came to 60
  // characters on one line against a budget of about 57, so the box
  // clipped by three. The split is what actually fixed the width.
  const sandbox = load();

  const lines = sandbox.overlaysRemaskSummary([1, 3, 4, 6, 10, 22]);

  assert.equal(lines.length, 2);
  assert.ok(
    widest(lines).length < 45,
    `longest line was ${widest(lines).length} characters`
  );
});

test("the summary is bounded whatever the selection", () => {
  // The property the width fix rests on, stated independently of the
  // exact wording above. Positions run to three digits on a 256-wide
  // canvas, so the worst case is five of those plus the tail.
  const sandbox = load();
  const huge = [];
  for (let i = 100; i < 4096; i++) {
    huge.push(i);
  }

  const longest = widest(sandbox.overlaysRemaskSummary(huge));

  assert.ok(
    longest.length < 50, `longest line was ${longest.length}`
  );
});

test("the cap is exactly the boundary it claims", () => {
  const sandbox = load();
  const max = sandbox.OVERLAYS_REMASK_LIST_MAX;
  const exact = [];
  for (let i = 0; i < max; i++) {
    exact.push(i);
  }

  assert.doesNotMatch(
    sandbox.overlaysRemaskSummary(exact)[1], /others/
  );
  assert.match(
    sandbox.overlaysRemaskSummary(exact.concat([max]))[1],
    /and 1 others/
  );
});
