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
// looks identical to a dropped signal. The sections after that prove
// the same of the chart itself, opened the way Analytics opens a
// saved run and then scrubbed; that a bar fades only where its
// position does not exist yet at the scrubbed frame; and that a
// DiffusionGemma commit, which carries no entropy of its own, is read
// through its canvas's last draft by every view, and says so.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, ANALYTICS_SCRIPTS } = require("./dom_stub.js");

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

  const at = context.overlaySeriesChannelFrame(
    context.overlaySeriesChannel(data, "entropy"),
    series,
    context.overlayFrameIndex
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

  const at = context.overlaySeriesChannelFrame(
    context.overlaySeriesChannel(data, "entropy"),
    series,
    context.overlayFrameIndex
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

  const early = context.overlaySeriesEntropyValues(series, 0);
  const late = context.overlaySeriesEntropyValues(series, 2);

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

  const at = context.overlaySeriesChannelFrame(
    context.overlaySeriesChannel(data, "entropy"),
    series,
    context.overlayFrameIndex
  );

  assert.equal(at, 2);
});

// -- the three availability states --

test("a drawable channel reports ok", () => {
  const { context } = page();

  assert.equal(
    context.overlaySeriesEntropyAvailability(
      run(["frame", "position"])
    ),
    "ok"
  );
  assert.equal(
    context.overlaySeriesEntropyAvailability(run(["position"])), "ok"
  );
});

test("a shape this build cannot draw reports unsupported", () => {
  // A canvas-level entropy has no per-position bars to draw. Saying
  // so is the point: hiding the section is what a channel lost by
  // accident would also look like.
  const { context } = page();

  assert.equal(
    context.overlaySeriesEntropyAvailability(run(["canvas"])),
    "unsupported"
  );
});

test("a run with no entropy at all reports absent", () => {
  const { context } = page();
  const bare = {
    run_id: "r2",
    frames: [[{ t: "a", m: false, id: 1, c: 0.5 }]],
    signals: [channel(["position"])],
  };

  assert.equal(
    context.overlaySeriesEntropyAvailability(bare), "absent"
  );
});

// -- runs from before the manifest existed --

test("a run with no manifest is read as it always was", () => {
  // 258 saved runs predate this. Absent has to mean "infer as
  // before", not "unsupported", or the corpus goes dark.
  const { context } = page();

  assert.equal(context.overlaySeriesEntropyAvailability(run()), "ok");
});

test("an unmanifested run still reads the final frame", () => {
  const { context } = page();
  const data = run();
  const series = context.overlaySeriesOf(data, false);

  const at = context.overlaySeriesChannelFrame(
    context.overlaySeriesChannel(data, "entropy"),
    series,
    context.overlayFrameIndex
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
    context.overlaySeriesChannelShape(channel(["frame", "position"])),
    "frame|position"
  );
  assert.equal(
    context.overlaySeriesChannelShape(channel(["position"])),
    "position"
  );
  assert.equal(context.overlaySeriesChannelShape(null), "");
});

// -- the chart, as a saved run opens and is scrubbed --
//
// The tests above hand the readers a run with its manifest already
// attached. These open one the way Analytics does, through
// renderRunOverlays and the scrubber, from a payload carrying exactly
// the keys the frames endpoint returns. A manifest the endpoint drops
// or a chart that never re-reads its bars shows up here, where the
// readers alone would still pass.

// The frames endpoint's response for one run, as _compute_run_frames
// in src/web/server.py builds it, with `overrides` replacing keys.
function framesPayload(overrides) {
  return Object.assign({
    run_id: "r1",
    frames: null,
    positions: null,
    original_frames: null,
    original_positions: null,
    records_available: true,
    alternatives: null,
    alternatives_available: false,
    original_alternatives: null,
    candidates: null,
    original_candidates: null,
    remask_edits: [],
    canvas_index: null,
    stop_rule: null,
    signals: null,
  }, overrides || {});
}

// A diffusion run's frames over four positions. Entropy at frame N
// is N plus a tenth of the position, and each token names its frame,
// so a bar read from the wrong frame or position cannot agree by
// luck. `base` offsets both, so a pre-edit run reads differently.
function canvasFrames(count, base) {
  const offset = base || 0;
  const frames = [];
  for (let frame = 0; frame < count; frame++) {
    frames.push(["a", "b", "c", "d"].map((letter, position) => ({
      t: letter + (offset + frame),
      m: false,
      id: 100 + position,
      c: 0.5,
      e: offset + frame + position / 10,
    })));
  }
  return frames;
}

// An autoregressive run as the endpoint sends it: one record per
// position, frame N being the first N + 1 of them.
function appendPositions(count) {
  const positions = [];
  for (let position = 0; position < count; position++) {
    positions.push({
      t: "w" + position,
      m: false,
      id: 200 + position,
      c: 0.5,
      e: position / 10,
    });
  }
  return positions;
}

// The page with `payload` opened, Chart replaced by a recorder that
// keeps the configuration it was given, so the scrub's edits to the
// entropy chart can be read back.
function openedRun(payload) {
  const opened = page();
  const { context } = opened;
  context.Chart = function (ctx, config) {
    return {
      data: config.data,
      options: config.options,
      setActiveElements() {},
      update() {},
      destroy() {},
      resize() {},
    };
  };
  context.renderRunOverlays(payload);
  assert.ok(context.chartEntropy, "the entropy chart was not built");
  return context;
}

// One layer of the open entropy chart, by its label.
function layer(context, label) {
  const sets = context.chartEntropy.data.datasets;
  const found = sets.find((set) => set.label === label);
  assert.ok(found, "the chart has no " + label + " layer");
  return found;
}

const FRAME_BY_POSITION = channel(["frame", "position"]);

test("a saved diffusion run's bars follow the scrub", () => {
  const context = openedRun(framesPayload({
    frames: canvasFrames(3),
    canvas_index: [0, 0, 0],
    signals: [FRAME_BY_POSITION],
  }));

  context.setOverlayFrame(0);
  const early = layer(context, "Edited");
  assert.deepEqual(host(early.data), [0, 0.1, 0.2, 0.3]);
  assert.deepEqual(host(early.texts), ["a0", "b0", "c0", "d0"]);

  context.setOverlayFrame(2);
  const late = layer(context, "Edited");
  assert.deepEqual(host(late.data), [2, 2.1, 2.2, 2.3]);
  assert.deepEqual(host(late.texts), ["a2", "b2", "c2", "d2"]);
});

test("an edited run's original layer follows to its own end", () => {
  // The pre-edit run is one frame shorter than the branch, so the
  // last scrub reaches past it and has to stop at its final frame
  // rather than read a frame it does not have.
  const context = openedRun(framesPayload({
    frames: canvasFrames(3),
    original_frames: canvasFrames(2, 10),
    remask_edits: [{ frame_index: 1, token_positions: [2] }],
    canvas_index: [0, 0, 0],
    signals: [FRAME_BY_POSITION],
  }));

  context.setOverlayFrame(0);
  assert.deepEqual(
    host(layer(context, "Original").data), [10, 10.1, 10.2, 10.3]
  );

  context.setOverlayFrame(2);
  assert.deepEqual(
    host(layer(context, "Original").data), [11, 11.1, 11.2, 11.3]
  );
  assert.deepEqual(
    host(layer(context, "Edited").data), [2, 2.1, 2.2, 2.3]
  );
});

test("a run saved before manifests reads its final frame", () => {
  // Negative space for the first test: with no manifest the run is
  // read the way it always was, whatever the scrubber does.
  const context = openedRun(framesPayload({
    frames: canvasFrames(3),
    canvas_index: [0, 0, 0],
  }));

  context.setOverlayFrame(0);

  assert.deepEqual(
    host(layer(context, "Edited").data), [2, 2.1, 2.2, 2.3]
  );
});

test("an autoregressive run's bars stay put", () => {
  // Each position is decided once, so there is nothing to follow.
  const context = openedRun(framesPayload({
    positions: appendPositions(4),
    signals: [channel(["position"])],
  }));
  const opened = host(layer(context, "Edited").data);

  context.setOverlayFrame(1);

  assert.deepEqual(opened, [0, 0.1, 0.2, 0.3]);
  assert.deepEqual(host(layer(context, "Edited").data), opened);
});

// -- which bars fade --
//
// A bar past the scrubbed frame fades when the position it stands
// for does not exist yet at that frame, so the chart agrees with the
// canvas above it. That holds on an append stream, where frame N
// introduced position N, and never on a diffusion canvas, which
// holds every position at every frame.

// Each bar of a layer: true when drawn at full strength, false when
// faded. Built here rather than mapped over the page's own arrays,
// so it compares by value across the vm boundary.
function fullStrength(context, set) {
  const strengths = [];
  for (let i = 0; i < set.data.length; i++) {
    const fill = set.backgroundColor[i];
    if (fill === context.entropyColor(set.data[i])) {
      strengths.push(true);
    } else {
      assert.equal(
        fill,
        context.entropyDimColor(set.data[i]),
        "bar " + i + " is neither drawn nor faded"
      );
      strengths.push(false);
    }
  }
  return strengths;
}

test("a diffusion run draws every bar at full strength", () => {
  // Three frames over four positions: every bar past the frame
  // number is a position the canvas already holds.
  const context = openedRun(framesPayload({
    frames: canvasFrames(3),
    canvas_index: [0, 0, 0],
    signals: [FRAME_BY_POSITION],
  }));
  const opened = fullStrength(context, layer(context, "Edited"));

  context.setOverlayFrame(0);

  assert.deepEqual(opened, [true, true, true, true]);
  assert.deepEqual(
    fullStrength(context, layer(context, "Edited")),
    [true, true, true, true]
  );
});

test("so does a diffusion run saved before manifests", () => {
  // Decided by its stream, which is a canvas of snapshots.
  const context = openedRun(framesPayload({
    frames: canvasFrames(3),
    canvas_index: [0, 0, 0],
  }));

  context.setOverlayFrame(0);

  assert.deepEqual(
    fullStrength(context, layer(context, "Edited")),
    [true, true, true, true]
  );
});

test("an autoregressive run fades the positions to come", () => {
  const context = openedRun(framesPayload({
    positions: appendPositions(4),
    signals: [channel(["position"])],
  }));

  context.setOverlayFrame(1);

  assert.deepEqual(
    fullStrength(context, layer(context, "Edited")),
    [true, true, false, false]
  );
});

test("as does one saved before manifests", () => {
  // Decided by its stream too, which is an append.
  const context = openedRun(framesPayload({
    positions: appendPositions(4),
  }));

  context.setOverlayFrame(1);

  assert.deepEqual(
    fullStrength(context, layer(context, "Edited")),
    [true, true, false, false]
  );
});

// -- a DiffusionGemma run, whose canvases end on commits --
//
// A committed canvas carries no entropy of its own, and a finished
// DiffusionGemma run opens on one. Every entropy view at a commit
// reads its canvas's last draft instead and says "as of step N", as
// the candidate popover does when it borrows an earlier frame.

// Frames following `plan`, one entry per frame naming its canvas and
// whether it is a commit. A draft's every position carries entropy,
// draft N's being `base` + N + 1 plus a tenth of the position; a
// commit's carries none, since the model accepted its canvas rather
// than drew it.
function committedFrames(plan, base) {
  return plan.map((step, frame) =>
    ["a", "b", "c", "d"].map((letter, position) => {
      const token = {
        t: letter + frame,
        m: false,
        id: 100 + position,
        c: step.commit ? 1 : 0.5,
      };
      if (!step.commit) {
        token.e = base + frame + 1 + position / 10;
      }
      return token;
    })
  );
}

// What draft `frame`'s bars hold, rounded as the chart rounds them.
function draftValues(frame, base) {
  return [0, 1, 2, 3].map(
    (position) => +(base + frame + 1 + position / 10).toFixed(3)
  );
}

// Two drafts and a commit on canvas 0, then a draft and a commit on
// canvas 1: how DiffusionGemma streams a run of two canvases.
const TWO_CANVASES = [
  { canvas: 0, commit: false },
  { canvas: 0, commit: false },
  { canvas: 0, commit: true },
  { canvas: 1, commit: false },
  { canvas: 1, commit: true },
];

function committedPayload() {
  return framesPayload({
    frames: committedFrames(TWO_CANVASES, 0),
    canvas_index: TWO_CANVASES.map((step) => step.canvas),
    signals: [FRAME_BY_POSITION],
  });
}

function pickerValues(context, data) {
  context.buildOverlaySelect(data);
  const list = context.overlaySelect.children.find(
    (child) => child.tag === "ul"
  );
  return list.children.map((item) => item.getAttribute("data-value"));
}

// Every token span drawn into `element`, through the fragments the
// stub keeps as children rather than flattening.
function drawnSpans(element) {
  const spans = [];
  for (const child of element.children || []) {
    if (child.tag === "span") {
      spans.push(child);
    } else {
      spans.push(...drawnSpans(child));
    }
  }
  return spans;
}

test("a DiffusionGemma run's entropy is found past commits", () => {
  // Its final frame is a commit, and that frame is where every probe
  // used to look, so the chart and the overlay were never offered.
  const { context } = page();
  const payload = committedPayload();

  assert.equal(
    context.overlaySeriesEntropyAvailability(payload), "ok"
  );
  assert.equal(context.overlaySeriesCarriesEntropy(payload), true);
});

test("the picker offers its Entropy overlay", () => {
  const context = openedRun(committedPayload());

  assert.ok(
    pickerValues(context, context.overlayData).includes("entropy")
  );
});

test("opened on a commit, the bars borrow the last draft", () => {
  const context = openedRun(committedPayload());
  const edited = layer(context, "Edited");

  assert.deepEqual(host(edited.data), draftValues(3, 0));
  assert.equal(edited.asOfStep, 3);
  const row = context.entropyTooltipLabel({
    formattedValue: "4.1",
    dataIndex: 1,
    datasetIndex: 0,
    dataset: edited,
  }, null);
  assert.match(row, /as of step 3$/);
});

test("a draft reads its own entropy, unlabeled", () => {
  const context = openedRun(committedPayload());

  context.setOverlayFrame(1);

  const edited = layer(context, "Edited");
  assert.deepEqual(host(edited.data), draftValues(1, 0));
  assert.equal(edited.asOfStep, null);
});

test("a commit borrows from its own canvas's draft", () => {
  // Canvas 0's commit reads canvas 0's last draft, frame 1, rather
  // than anything of canvas 1's.
  const context = openedRun(committedPayload());

  context.setOverlayFrame(2);

  const edited = layer(context, "Edited");
  assert.deepEqual(host(edited.data), draftValues(1, 0));
  assert.equal(edited.asOfStep, 1);
});

test("a pre-edit layer borrows from its own draft", () => {
  // The pre-edit run is a frame shorter, so at the branch's commit it
  // is clamped to its own commit, which borrows its own draft.
  const context = openedRun(framesPayload({
    frames: committedFrames(TWO_CANVASES.slice(0, 3), 0),
    original_frames: committedFrames(
      [{ canvas: 0, commit: false }, { canvas: 0, commit: true }], 10
    ),
    remask_edits: [{ frame_index: 1, token_positions: [2] }],
    canvas_index: [0, 0, 0],
    signals: [FRAME_BY_POSITION],
  }));

  const pre = layer(context, "Original");
  assert.deepEqual(host(pre.data), draftValues(0, 10));
  assert.equal(pre.asOfStep, 0);
});

test("the metrics strip reads a commit through its draft", () => {
  const context = openedRun(committedPayload());
  context.metricsHoverPos = 1;
  context.metricsHoverOriginal = false;

  const reading = context.buildTokenMetricsReading();

  assert.equal(reading.entropy, draftValues(3, 0)[1]);
  assert.match(reading.extra, /entropy as of step 3/);
});

test("the Entropy overlay colors a commit from its draft", () => {
  const context = openedRun(committedPayload());
  context.overlayMode = "entropy";
  // The stub keeps children when text is cleared, so the spans the
  // opening render drew are dropped by hand.
  context.overlayOutput.children = [];

  context.renderCurrentOverlay();

  const colors = drawnSpans(context.overlayOutput).map(
    (span) => span.style.color
  );
  assert.deepEqual(
    colors,
    draftValues(3, 0).map((value) => context.entropyColor(value))
  );
});
