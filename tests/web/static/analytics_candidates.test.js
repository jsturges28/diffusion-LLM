// The Analytics candidate popover on a saved diffusion run.
//
// Strategy: load the Analytics page into the DOM stub, hand the
// overlay viewer a frames payload the way the server serves a saved
// run, move its scrubber, and render the popover at a position. The
// payload goes through renderRunOverlays, so the path from the
// server's `candidates` to what the popover draws is the real one.
//
// Passing proves the saved popover reads as the live one does: a
// captured frame shows its own candidates, a skipped frame the
// latest captured before it with "As of step N", and a frame on a
// new canvas, the pre-edit layer, and a run saved before candidates
// existed show nothing. An autoregressive run keeps its per-position
// popover, untouched by any of it.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "run_candidates.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

const WORDS = [" Yeast", " eats", " sugar", "."];

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

function frameTokens(index) {
  return WORDS.map((word, position) => ({
    t: word,
    m: position >= index,
    id: 1000 + position,
    c: 0.5,
    e: 1.0,
  }));
}

function setsAt(frame) {
  return WORDS.map((_, position) => {
    const lead = frame * 100 + position;
    return {
      h: lead,
      c: [
        { id: lead, t: " lead", p: 0.6 },
        { id: 7, t: " seven", p: 0.2 },
      ],
    };
  });
}

function candidatesAt(captured) {
  return {
    k: 5,
    stride: 2,
    frames: captured,
    segments: [0],
    sets: captured.map(setsAt),
  };
}

// A saved five-frame run, as `/api/analytics/runs/{id}/frames` serves
// it, with candidates kept at a stride of 2 and the final frame.
function payload(overrides) {
  const frames = [0, 1, 2, 3, 4].map(frameTokens);
  return Object.assign({
    run_id: "run",
    frames: frames,
    positions: null,
    original_frames: null,
    original_positions: null,
    records_available: true,
    alternatives: null,
    alternatives_available: false,
    original_alternatives: null,
    candidates: candidatesAt([1, 3, 4]),
    remask_edits: [],
    canvas_index: [0, 0, 0, 0, 0],
  }, overrides || {});
}

function opened(data) {
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  page.context.renderRunOverlays(data);
  // The stub's Chart keeps no datasets for scrubbing to recolour,
  // and the entropy chart is not what these tests are about.
  page.context.chartEntropy = null;
  return page;
}

function descendants(node) {
  const found = [];
  const stack = node.children.slice();
  while (stack.length > 0) {
    const next = stack.shift();
    found.push(next);
    stack.push(...next.children);
  }
  return found;
}

function withClass(node, name) {
  return descendants(node).filter((n) => n.classes.has(name));
}

// The stub keeps children when text is cleared, so it starts empty.
function popoverAt(page, frame, position) {
  const popover = page.registry.get("token-alts-popover");
  page.context.setOverlayFrame(frame);
  popover.children = [];
  page.context.renderAltsPopover(position, null);
  return popover;
}

function stepLabel(popover) {
  return withClass(popover, "alt-step")[0].textContent;
}

function rowIds(popover) {
  return withClass(popover, "alt-row").map(
    (row) => Number(row.getAttribute("data-alt-id"))
  );
}

test("a captured frame shows its own candidates", () => {
  const page = opened(payload());

  const popover = popoverAt(page, 3, 2);

  assert.equal(popover.hidden, false);
  assert.equal(stepLabel(popover), "Step 3");
  assert.deepEqual(rowIds(popover), [302, 7]);
});

test("a skipped frame shows the latest captured, and says so", () => {
  const page = opened(payload());

  const popover = popoverAt(page, 2, 2);

  assert.equal(stepLabel(popover), "As of step 1");
  assert.deepEqual(rowIds(popover), [102, 7]);
});

test("the row for the token on screen is marked", () => {
  const page = opened(payload());

  const popover = popoverAt(page, 4, 1);
  const chosen = withClass(popover, "alt-row-chosen");

  assert.equal(chosen.length, 1);
  assert.equal(Number(chosen[0].getAttribute("data-alt-id")), 401);
});

test("a frame on a new canvas never borrows from the last", () => {
  // Frame 3 starts canvas 1 and was skipped; the latest captured
  // before it belongs to canvas 0, whose positions are unrelated.
  const page = opened(payload({
    candidates: candidatesAt([1, 2, 4]),
    canvas_index: [0, 0, 0, 1, 1],
  }));

  assert.equal(popoverAt(page, 3, 2).hidden, true);
  assert.equal(stepLabel(popoverAt(page, 4, 2)), "Step 4");
});

test("the pre-edit layer has no popover", () => {
  const page = opened(payload({
    original_frames: [0, 1, 2, 3, 4].map(frameTokens),
    remask_edits: [{ frame_index: 2, token_positions: [1] }],
  }));
  page.context.compareBlend = 0.2;

  const popover = popoverAt(page, 3, 2);

  assert.equal(popover.hidden, true);
});

test("a run saved before candidates existed has no popover", () => {
  const page = opened(payload({ candidates: null }));

  const popover = popoverAt(page, 3, 2);

  assert.equal(popover.hidden, true);
});

test("an autoregressive run keeps its per-position popover", () => {
  const alternatives = WORDS.map((_, position) => [
    { id: 1000 + position, t: " own", p: 0.7 },
  ]);
  const page = opened(payload({
    candidates: null,
    alternatives: alternatives,
    alternatives_available: true,
  }));
  page.context.overlayIsAutoregressive = true;

  const popover = popoverAt(page, 3, 2);

  assert.equal(popover.hidden, false);
  assert.deepEqual(rowIds(popover), [1002]);
  assert.equal(withClass(popover, "alt-step").length, 0);
});
