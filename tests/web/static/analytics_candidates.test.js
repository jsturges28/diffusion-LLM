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
// new canvas and a run saved before candidates existed show nothing.
// An edited run pages between its two runs from the frame the edit
// branched at, opening on the one the crossfade favours, and a run
// saved before the pre-edit candidates were kept has only its edited
// page. An autoregressive run keeps its per-position popover,
// untouched by any of it. With the candidates chosen, a saved run's
// unsettled positions cycle at the scrubbed frame, each crossfade
// layer through its own run's candidates.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "run_candidates.js",
  "candidate_flicker.js",
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

// The popover as a hover opens it. The stub keeps children when text
// is cleared, so it starts empty.
function popoverAt(page, frame, position) {
  const popover = page.registry.get("token-alts-popover");
  page.context.setOverlayFrame(frame);
  popover.children = [];
  page.context.showAltsPopover(position, null);
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

// -- an edited run pages between the two runs --

const EDIT_AT_2 = { frame_index: 2, token_positions: [1] };

// A run edited at frame 2, as its save serves it. The original kept
// frames 1, 3 and 4, so its sets lead with 302 at frame 3 for
// position 2; the resume re-ran frames 2 to 4 and kept its own frames
// 1 and 2, landing on 3 and 4, whose sets lead with 102 at frame 3.
function editedPayload(overrides) {
  return payload(Object.assign({
    original_frames: [0, 1, 2, 3, 4].map(frameTokens),
    original_candidates: candidatesAt([1, 3, 4]),
    candidates: {
      k: 5,
      stride: 2,
      frames: [1, 3, 4],
      segments: [0, 2],
      sets: [setsAt(1), setsAt(1), setsAt(2)],
    },
    remask_edits: [EDIT_AT_2],
  }, overrides || {}));
}

function titleOf(popover) {
  return withClass(popover, "alt-heading")[0].children[0].textContent;
}

function pagerTo(popover, label) {
  return withClass(popover, "alt-pager-btn").find(
    (button) => button.getAttribute("aria-label") === label + " run"
  );
}

test("an edited run opens on the run the crossfade favours", () => {
  const page = opened(editedPayload());

  page.context.compareBlend = 0.2;
  const original = popoverAt(page, 3, 2);
  assert.equal(titleOf(original), "Position 3: Original");
  assert.deepEqual(rowIds(original), [302, 7]);

  page.context.compareBlend = 0.8;
  const edited = popoverAt(page, 3, 2);
  assert.equal(titleOf(edited), "Position 3: Edited");
  assert.deepEqual(rowIds(edited), [102, 7]);
});

test("the pager turns to the other run", () => {
  const page = opened(editedPayload());
  page.context.compareBlend = 0.2;
  const popover = popoverAt(page, 3, 2);
  const toEdited = pagerTo(popover, "Edited");

  popover.children = [];
  toEdited.dispatch("click", { stopPropagation() {} });

  assert.equal(titleOf(popover), "Position 3: Edited");
  assert.deepEqual(rowIds(popover), [102, 7]);
});

test("before the edit there is one run, so no pager", () => {
  const page = opened(editedPayload());
  page.context.compareBlend = 0.2;

  const popover = popoverAt(page, 1, 2);

  assert.equal(titleOf(popover), "Position 3: candidates");
  assert.equal(withClass(popover, "alt-pager").length, 0);
});

test("the runs part at the earliest edit, whatever the order", () => {
  const page = opened(editedPayload({
    remask_edits: [
      { frame_index: 3, token_positions: [2] }, EDIT_AT_2,
    ],
  }));
  page.context.compareBlend = 0.2;

  const popover = popoverAt(page, 2, 2);

  assert.equal(titleOf(popover), "Position 3: Original");
  assert.equal(stepLabel(popover), "As of step 1");
});

test("past its end, the Original page reads the original's last frame", () => {
  // The original ran four frames to the edit's five, so its layer
  // holds its frame 3 at frame 4, and the page names that its step.
  const page = opened(editedPayload({
    original_frames: [0, 1, 2, 3].map(frameTokens),
    original_candidates: candidatesAt([1, 3]),
  }));
  page.context.compareBlend = 0.2;

  const popover = popoverAt(page, 4, 2);

  assert.equal(stepLabel(popover), "Step 3");
  assert.deepEqual(rowIds(popover), [302, 7]);
});

test("an edited run saved without its baseline has one page", () => {
  // With no original to crossfade to, one run is on screen.
  const page = opened(editedPayload({ original_frames: null }));
  page.context.compareBlend = 0.2;

  const popover = popoverAt(page, 3, 2);

  assert.equal(titleOf(popover), "Position 3: candidates");
  assert.deepEqual(rowIds(popover), [102, 7]);
});

test("where the favoured run has nothing, the popover stays closed", () => {
  const page = opened(editedPayload({ candidates: null }));
  page.context.compareBlend = 0.8;

  const popover = popoverAt(page, 3, 2);

  assert.equal(popover.hidden, true);
});

test("a run saved without pre-edit candidates has only its edited page", () => {
  // Saved before they were kept: nothing under the original's
  // tokens, and nothing to turn to from the edited run's.
  const page = opened(editedPayload({ original_candidates: null }));

  page.context.compareBlend = 0.2;
  assert.equal(popoverAt(page, 3, 2).hidden, true);

  page.context.compareBlend = 0.8;
  const edited = popoverAt(page, 3, 2);
  assert.equal(titleOf(edited), "Position 3: Edited");
  assert.equal(withClass(edited, "alt-pager").length, 0);
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

// -- the candidates cycle --

function cyclingAt(page, frame) {
  page.context.setOverlayFrame(frame);
  const entries = [...page.context.flickerEntries];
  page.context.flickerStop();
  return entries;
}

// A store whose lead candidate is renamed, so each crossfade layer's
// source can be told apart.
function relabelled(store, text) {
  return Object.assign({}, store, {
    sets: store.sets.map((frame) => frame.map((set) => ({
      h: set.h,
      c: [Object.assign({}, set.c[0], { t: text }), set.c[1]],
    }))),
  });
}

test("a saved run's unsettled positions cycle", () => {
  // At frame 3 only position 3 is unsettled.
  const page = opened(payload());
  page.context.analyticsSettings.unsettledShows = "candidates";

  const cycling = cyclingAt(page, 3);

  assert.deepEqual(cycling.map((entry) => entry.position), [3]);
  assert.ok([...cycling[0].texts].includes(" lead"));
});

test("the guess keeps a saved run still", () => {
  const page = opened(payload());
  page.context.analyticsSettings.unsettledShows = "guess";

  assert.equal(cyclingAt(page, 3).length, 0);
});

test("each saved crossfade layer cycles its own run's candidates", () => {
  const edited = editedPayload();
  const page = opened(editedPayload({
    original_candidates: relabelled(candidatesAt([1, 3, 4]), " before"),
    candidates: relabelled(edited.candidates, " after"),
  }));
  page.context.analyticsSettings.unsettledShows = "candidates";
  page.context.compareBlend = 0.5;

  const cycling = cyclingAt(page, 3);

  assert.equal(cycling.length, 2);
  const original = [...cycling[0].texts];
  const branch = [...cycling[1].texts];
  assert.ok(original.includes(" before"));
  assert.equal(original.includes(" after"), false);
  assert.ok(branch.includes(" after"));
  assert.equal(branch.includes(" before"), false);
});
