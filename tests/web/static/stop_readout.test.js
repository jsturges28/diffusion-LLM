// The adaptive-stopping readout: how far a DiffusionGemma canvas is
// from stopping, read off the frames the page already holds.
//
// Strategy: load overlays.js on its own over the DOM stub and drive
// the stopping primitives with small hand-built frame streams shaped
// like DiffusionGemma's: every position of a draft carries its
// entropy `e` and whether it changed `m`, and a committed canvas
// carries no entropy at all. Each rule is pinned at its boundary
// with the nearest case that should read the other way.
//
// Passing proves the readout counts steadiness the way transformers
// does (restarting with every canvas and every resume), compares
// entropy strictly, judges a committed canvas by its step budget
// alone, draws the trace on a log scale across that budget, words
// each state as the mocks showed, and lets its words give way before
// the metrics strip loses anything of its own.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { loadPage, makeElement } = require("./dom_stub.js");

function page() {
  return loadPage({ scripts: ["overlays.js"] }).context;
}

// Arrays built inside the vm context are not reference-equal to host
// ones, so deepEqual rejects them on realm rather than on content.
function host(value) {
  return JSON.parse(JSON.stringify(value));
}

// A draft of `width` positions at mean entropy `entropy`, of which
// the first `changed` moved since the last draft.
function draft(entropy, changed, width) {
  const tokens = [];
  for (let i = 0; i < (width || 4); i++) {
    tokens.push({ t: "w", m: i < changed, id: i, e: entropy });
  }
  return tokens;
}

// A committed canvas: accepted, so confident at 1.0, and measured by
// nothing, so no entropy anywhere.
function commit(width) {
  const tokens = [];
  for (let i = 0; i < (width || 4); i++) {
    tokens.push({ t: "w", m: false, id: i, c: 1 });
  }
  return tokens;
}

function trackOf(context, frames, options) {
  const settings = options || {};
  const canvases = settings.canvases || frames.map(() => 0);
  return context.overlaysStopTrack({
    count: frames.length,
    readFrame: (f) => frames[f],
    canvasAt: (f) => canvases[f],
    segmentStarts: settings.segmentStarts || [],
  });
}

const RULE = { threshold: 0.005, steadySteps: 1, budget: 48 };

function readingAt(frames, index, rule, options) {
  const context = page();
  return context.overlaysStopReadingAt(
    trackOf(context, frames, options), index, rule || RULE
  );
}

function field(track, name) {
  return host(track).map((entry) => entry[name]);
}

// The words a reading renders to, as one string.
function wordsOf(reading) {
  const context = page();
  return host(context.overlaysStopClauses(reading))
    .map((clause) => clause.parts.map((part) => part.text).join(""))
    .join("");
}

// Every leaf's text under an element, in order.
function textOf(element) {
  if (element.children.length === 0) {
    return element.textContent;
  }
  return element.children.map(textOf).join("");
}

// -- one frame's measurements --

test("a draft's entropy is the mean over every position", () => {
  const context = page();
  const tokens = [
    { m: true, e: 0.1 }, { m: false, e: 0.3 },
    { m: true, e: 0.2 }, { m: false, e: 0.0 },
  ];

  const summary = host(context.overlaysStopSummary(tokens));

  assert.ok(Math.abs(summary.entropy - 0.15) < 1e-12);
  assert.equal(summary.changed, 2);
});

test("a frame measured nowhere has no entropy, not zero", () => {
  const context = page();

  const summary = host(context.overlaysStopSummary(commit()));

  assert.equal(summary.entropy, null);
  assert.equal(summary.changed, 0);
});

test("a frame measured at only some positions is nothing", () => {
  const context = page();
  const tokens = [{ m: false, e: 0.01 }, { m: false }];

  assert.equal(context.overlaysStopSummary(tokens), null);
  assert.equal(context.overlaysStopSummary([]), null);
});

// -- the track --

test("drafts count their steps and a commit keeps it", () => {
  const context = page();
  const frames = [
    draft(2, 4), draft(0.5, 2), draft(0.004, 0), commit(),
  ];

  const track = trackOf(context, frames);

  assert.deepEqual(
    field(track, "kind"), ["draft", "draft", "draft", "commit"]
  );
  assert.deepEqual(field(track, "step"), [1, 2, 3, 3]);
});

test("a run that never measured entropy has no commits", () => {
  // Without drafts carrying entropy, a frame without it is just an
  // old frame, not a committed canvas.
  const context = page();
  const frames = [commit(), commit(), commit()];

  const track = trackOf(context, frames);

  assert.deepEqual(field(track, "kind"), ["none", "none", "none"]);
});

test("steadiness counts still drafts, resetting on change", () => {
  const context = page();
  const frames = [
    draft(1, 4), draft(1, 0), draft(1, 0), draft(1, 1), draft(1, 0),
  ];

  const track = trackOf(context, frames);

  assert.deepEqual(field(track, "steady"), [0, 1, 2, 0, 1]);
});

test("each canvas restarts the steps and the steadiness", () => {
  const context = page();
  const frames = [
    draft(1, 4), draft(1, 0), commit(), draft(1, 0), draft(1, 0),
  ];

  const track = trackOf(context, frames, {
    canvases: [0, 0, 0, 1, 1],
  });

  assert.deepEqual(field(track, "step"), [1, 2, 2, 1, 2]);
  assert.deepEqual(field(track, "steady"), [0, 1, 0, 0, 1]);
});

test("a canvas's first draft is never steady, however still", () => {
  // Real first drafts mark every position changed, but the rule is
  // that a fresh history cannot be stable, not that `m` says so.
  const context = page();

  const track = trackOf(context, [draft(1, 0), draft(1, 0)]);

  assert.deepEqual(field(track, "steady"), [0, 1]);
});

test("a resume's first frame is never steady either", () => {
  // transformers starts a fresh history on every resume, so the
  // first resumed step cannot be stable even when nothing moved;
  // the frames alone would say it was.
  const context = page();
  const frames = [
    draft(1, 4), draft(1, 0), draft(1, 0), draft(1, 0),
  ];

  const resumed = trackOf(context, frames, { segmentStarts: [2] });
  const plain = trackOf(context, frames);

  assert.deepEqual(field(resumed, "steady"), [0, 1, 0, 1]);
  assert.deepEqual(field(plain, "steady"), [0, 1, 2, 3]);
  assert.deepEqual(field(resumed, "step"), [1, 2, 3, 4]);
});

// -- what the readout reads --

test("entropy is met strictly below the threshold", () => {
  const at = readingAt([draft(0.005, 0)], 0);
  const below = readingAt([draft(0.0049, 0)], 0);

  assert.equal(at.entropyMet, false);
  assert.equal(below.entropyMet, true);
});

test("zero steady steps is always met, even while moving", () => {
  const rule = { threshold: 0.005, steadySteps: 0, budget: 48 };

  const reading = readingAt([draft(1, 3)], 0, rule);

  assert.equal(reading.steadyMet, true);
});

test("two steady steps need two still drafts in a row", () => {
  const rule = { threshold: 0.005, steadySteps: 2, budget: 48 };
  const frames = [draft(1, 4), draft(1, 0), draft(1, 0)];

  assert.equal(readingAt(frames, 1, rule).steadyMet, false);
  assert.equal(readingAt(frames, 2, rule).steadyMet, true);
});

test("a canvas shorter than its budget stopped by the rule", () => {
  // Judged by length, not by re-reading its last draft: this one's
  // last draft looks unmet, and the canvas still ended early.
  const frames = [draft(1, 4), draft(0.2, 1), commit()];

  const reading = readingAt(frames, 2, {
    threshold: 0.005, steadySteps: 1, budget: 3,
  });

  assert.equal(reading.kind, "commit");
  assert.equal(reading.stopped, true);
});

test("a canvas that reached its budget used every step", () => {
  const frames = [
    draft(1, 4), draft(0.2, 1), draft(0.1, 1), commit(),
  ];

  const reading = readingAt(frames, 3, {
    threshold: 0.005, steadySteps: 1, budget: 3,
  });

  assert.equal(reading.stopped, false);
  assert.equal(reading.step, 3);
});

test("the trace holds this canvas's drafts up to the frame", () => {
  const frames = [
    draft(4, 4), draft(0.3, 1), commit(), draft(3, 4), draft(0.2, 2),
  ];
  const options = { canvases: [0, 0, 0, 1, 1] };

  const mid = host(readingAt(frames, 4, RULE, options).trace);
  const done = host(readingAt(frames, 2, RULE, options).trace);

  assert.deepEqual(mid, [
    { step: 1, entropy: 3 }, { step: 2, entropy: 0.2 },
  ]);
  assert.deepEqual(done, [
    { step: 1, entropy: 4 }, { step: 2, entropy: 0.3 },
  ]);
});

test("a frame with nothing to show reads as null", () => {
  const context = page();
  const track = trackOf(context, [draft(1, 4)]);

  assert.equal(readingAt([commit()], 0), null);
  assert.equal(context.overlaysStopReadingAt(track, 0, null), null);
  assert.equal(context.overlaysStopReadingAt(track, 1, RULE), null);
});

// -- the rule a run carries --

const DEFAULTS = {
  confidence_threshold: 0.005,
  stability_threshold: 1,
  max_denoising_steps: 48,
};

test("a rule is read from the run's own parameters", () => {
  const context = page();

  const rule = host(context.overlaysStopRuleFrom({
    confidence_threshold: 0.02,
    stability_threshold: 2,
    max_denoising_steps: 32,
  }, null));

  assert.deepEqual(
    rule, { threshold: 0.02, steadySteps: 2, budget: 32 }
  );
});

test("a value the run did not record is the fallback's", () => {
  const context = page();

  const rule = host(context.overlaysStopRuleFrom(
    { max_denoising_steps: 16 }, DEFAULTS
  ));

  assert.deepEqual(
    rule, { threshold: 0.005, steadySteps: 1, budget: 16 }
  );
});

test("parameters that make no rule give none", () => {
  const context = page();
  const broken = [
    { confidence_threshold: 0 },
    { stability_threshold: -1 },
    { stability_threshold: 1.5 },
    { max_denoising_steps: 0 },
  ];

  for (const change of broken) {
    const params = Object.assign({}, DEFAULTS, change);
    assert.equal(context.overlaysStopRuleFrom(params, null), null);
  }
  assert.equal(context.overlaysStopRuleFrom(null, null), null);
});

// -- the words --

test("a draft names its entropy, the threshold, what moved", () => {
  const frames = [draft(1, 4), draft(0.041, 13, 20)];

  assert.equal(
    wordsOf(readingAt(frames, 1)),
    "entropy 0.041 of 0.005, 13 changing"
  );
});

test("a still canvas reads steady once long enough", () => {
  const frames = [draft(1, 4), draft(0.0017, 0)];

  assert.equal(
    wordsOf(readingAt(frames, 1)),
    "entropy 0.0017 of 0.005, steady"
  );
});

test("the steady count shows while it is building", () => {
  const rule = { threshold: 0.005, steadySteps: 2, budget: 48 };
  const frames = [draft(1, 4), draft(0.024, 0)];

  assert.equal(
    wordsOf(readingAt(frames, 1, rule)),
    "entropy 0.024 of 0.005, steady 1 of 2"
  );
});

test("with no steadiness required, the words leave it out", () => {
  const rule = { threshold: 0.01, steadySteps: 0, budget: 48 };

  assert.equal(
    wordsOf(readingAt([draft(2.31, 4)], 0, rule)),
    "entropy 2.31 of 0.01"
  );
});

test("a committed canvas says how it ended", () => {
  const rule = { threshold: 0.005, steadySteps: 1, budget: 3 };
  const short = [draft(1, 4), draft(0.001, 0), commit()];
  const full = [draft(1, 4), draft(1, 4), draft(1, 4), commit()];
  const single = [draft(0.001, 0), commit()];

  assert.equal(
    wordsOf(readingAt(short, 2, rule)),
    "Canvas 1 stopped after 2 steps"
  );
  assert.equal(
    wordsOf(readingAt(full, 3, rule)),
    "Canvas 1 used all 3 steps"
  );
  assert.equal(
    wordsOf(readingAt(single, 1, rule)),
    "Canvas 1 stopped after 1 step"
  );
});

test("a canvas is numbered from one, for people", () => {
  const frames = [draft(1, 4), commit(), draft(0.001, 0), commit()];
  const options = { canvases: [0, 0, 1, 1] };

  assert.equal(
    wordsOf(readingAt(frames, 3, RULE, options)),
    "Canvas 2 stopped after 1 step"
  );
});

test("only the conditions that hold are marked met", () => {
  const context = page();
  const frames = [draft(1, 4), draft(0.004, 2)];

  const clauses = host(context.overlaysStopClauses(
    readingAt(frames, 1)
  ));

  assert.deepEqual(
    clauses.map((clause) => clause.met), [true, false, false]
  );
});

test("entropy keeps two figures at either end of its range", () => {
  const context = page();

  assert.equal(context.overlaysStopEntropyText(4.512), "4.51");
  assert.equal(context.overlaysStopEntropyText(0.04138), "0.041");
  assert.equal(context.overlaysStopEntropyText(0.00467), "0.0047");
  assert.equal(context.overlaysStopEntropyText(0), "0");
});

test("the tooltip states the run's own rule", () => {
  const context = page();

  assert.equal(
    context.overlaysStopRuleText(RULE),
    "A canvas stops once its mean entropy is below 0.005 nats"
      + " and no position has changed for 1 step, or once it"
      + " has used all 48 steps."
  );
  assert.equal(
    context.overlaysStopRuleText({
      threshold: 0.02, steadySteps: 0, budget: 1,
    }),
    "A canvas stops once its mean entropy is below 0.02 nats,"
      + " or once it has used all 1 step."
  );
});

// -- drawing --

test("the readout renders a reading and hides without one", () => {
  const context = page();
  const el = makeElement("stop-readout");
  context.overlaysBuildStopReadout(el);
  const frames = [draft(1, 4), draft(0.041, 13, 20)];

  assert.equal(el.hidden, true);
  context.overlaysRenderStopReadout(el, readingAt(frames, 1));

  assert.equal(el.hidden, false);
  assert.equal(
    textOf(el), "Stopentropy 0.041 of 0.005, 13 changing"
  );
  assert.match(el.getAttribute("title"), /below 0\.005 nats/);

  context.overlaysRenderStopReadout(el, null);

  assert.equal(el.hidden, true);
  assert.equal(el.getAttribute("title"), null);
});

test("a re-render replaces the words, not adds to them", () => {
  const context = page();
  const el = makeElement("stop-readout");
  context.overlaysBuildStopReadout(el);
  const frames = [draft(1, 4), draft(0.0017, 0)];

  context.overlaysRenderStopReadout(el, readingAt(frames, 0));
  context.overlaysRenderStopReadout(el, readingAt(frames, 1));

  const text = el.overlaysStopNodes.text;
  const met = text.children.filter(
    (clause) => clause.classList.contains("is-met")
  );
  assert.equal(textOf(text), "entropy 0.0017 of 0.005, steady");
  assert.equal(met.length, 2);
});

test("the trace falls on a log scale across the budget", () => {
  const context = page();
  const reading = {
    rule: { threshold: 0.005, steadySteps: 1, budget: 48 },
    trace: [
      { step: 1, entropy: 4.5 },
      { step: 24, entropy: 0.05 },
      { step: 48, entropy: 0.0005 },
    ],
  };

  const layout = host(
    context.overlaysStopTraceLayout(reading, 62, 13)
  );

  assert.equal(layout.points[0].x, 0);
  assert.equal(layout.points[2].x, 62);
  assert.ok(layout.points[0].y < layout.points[1].y);
  // A decade below the threshold is the floor, so the last point
  // sits on the bottom edge and the threshold a decade above it.
  assert.ok(Math.abs(layout.points[2].y - 13) < 1e-9);
  const decade = 13 / Math.log10(10 / 0.0005);
  assert.ok(Math.abs(13 - layout.threshold - decade) < 1e-9);
});

test("values past either end of the scale stay in the box", () => {
  const context = page();
  const reading = {
    rule: { threshold: 0.005, steadySteps: 1, budget: 1 },
    trace: [{ step: 1, entropy: 50 }, { step: 1, entropy: 1e-9 }],
  };

  const layout = host(
    context.overlaysStopTraceLayout(reading, 62, 13)
  );

  assert.equal(layout.points[0].y, 0);
  assert.ok(Math.abs(layout.points[1].y - 13) < 1e-9);
  assert.equal(layout.points[0].x, 31);
});

// -- giving way --

function stripWith(scrollWidth, clientWidth, extra) {
  const strip = makeElement("token-metrics");
  strip.scrollWidth = scrollWidth;
  strip.clientWidth = clientWidth;
  strip.overlaysMetricNodes = { extra: extra || null };
  return strip;
}

function shownReadout() {
  const readout = makeElement("stop-readout");
  readout.hidden = false;
  return readout;
}

test("the words give way when the strip runs out of room", () => {
  const context = page();
  const readout = shownReadout();

  context.overlaysFitStopReadout(stripWith(900, 780), readout);

  assert.equal(readout.hasAttribute("data-compact"), true);
});

test("the words stay when the strip has room", () => {
  const context = page();
  const readout = shownReadout();
  readout.setAttribute("data-compact", "");

  context.overlaysFitStopReadout(stripWith(780, 780), readout);

  assert.equal(readout.hasAttribute("data-compact"), false);
});

test("a cut overlay note counts as running out of room", () => {
  // The note shrinks before the strip overflows, so the strip alone
  // would report room while the note was being cut to nothing.
  const context = page();
  const readout = shownReadout();
  const note = makeElement(null);
  note.scrollWidth = 85;
  note.clientWidth = 0;

  context.overlaysFitStopReadout(
    stripWith(700, 700, note), readout
  );

  assert.equal(readout.hasAttribute("data-compact"), true);
});

test("a hidden readout is never left compact", () => {
  const context = page();
  const readout = makeElement("stop-readout");
  readout.hidden = true;
  readout.setAttribute("data-compact", "");

  context.overlaysFitStopReadout(stripWith(900, 780), readout);

  assert.equal(readout.hasAttribute("data-compact"), false);
});
