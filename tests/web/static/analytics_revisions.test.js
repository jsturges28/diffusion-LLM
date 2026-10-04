// Analytics' Revisions overlay, for a saved run.
//
// Strategy: load the Analytics page into the DOM stub beside this
// file and hand it saved runs the way loading one does, with frames
// shaped like DiffusionGemma's: a changed token reads as masked for a
// frame before it settles again. One run revises, one never does, and
// an edited run carries a pre-edit snapshot. Each colour is read off
// the span the real renderer painted, one layer at a time for the
// stacked run, and the strip off its own element under a hover.
//
// Passing proves the option is offered exactly for a diffusion run
// that revised, that each token is tinted by its count at the
// scrubbed frame, that each canvas counts on its own, that the
// pre-edit layer counts its own run while the edited one starts a
// remasked position over, that the strip reads the hovered layer's
// count, that the legend follows the selection, and that loading
// another run forgets the last one's revisions.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, ANALYTICS_SCRIPTS } = require("./dom_stub.js");

function settled(id) {
  return { t: " w" + id, m: false, id: id, c: 0.5 };
}

function changing(id) {
  return { t: " w" + id, m: true, id: id, c: 0.2 };
}

// Position 0 settles on 10, changes to 20, then to 30; position 1
// settles on 11 and changes to 21. Positions 2 and 3 settle once.
// Revised: position 0 at frames 3 and 6, position 1 at frame 5.
const REVISING = [
  [changing(1), changing(2), changing(3), changing(4)],
  [settled(10), settled(11), changing(3), changing(4)],
  [changing(20), settled(11), settled(12), changing(4)],
  [settled(20), settled(11), settled(12), settled(13)],
  [settled(20), changing(21), settled(12), settled(13)],
  [changing(30), settled(21), settled(12), settled(13)],
  [settled(30), settled(21), settled(12), settled(13)],
];

// Every position settles once and never moves, as on LLaDA.
const SETTLING = [
  [changing(1), changing(2), changing(3), changing(4)],
  [settled(10), settled(11), changing(3), changing(4)],
  [settled(10), settled(11), settled(12), settled(13)],
];

// Resumed at frame 4 with position 2 remasked: its new token is the
// edit's, while position 3 changes on its own. One frame longer than
// the run it replaced.
const BRANCH = [
  [settled(20), settled(11), changing(40), settled(13)],
  [settled(20), settled(11), settled(40), changing(41)],
  [settled(20), settled(11), settled(40), settled(41)],
  [settled(20), settled(11), settled(40), settled(41)],
];

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

function copy(frames) {
  return frames.map(
    (frame) => frame.map((token) => Object.assign({}, token))
  );
}

// The same, with the frames endpoint answering `data` for any run.
function framesFetch(data) {
  const boot = bootFetch();
  return function (url) {
    if (!String(url).endsWith("/frames")) {
      return boot(url);
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(data),
    });
  };
}

// Long enough for a fetched answer to work through its promises.
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function withRecords(data) {
  data.records_available = true;
  return data;
}

// A page that has opened `data` the way a run's frames landing does,
// scrubbed to `frame`.
function pageWith(data, frame) {
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  page.context.renderRunOverlays(withRecords(data));
  page.context.setOverlayFrame(frame);
  return page;
}

function editedRun(frame) {
  return pageWith({
    frames: copy(REVISING.slice(0, 4).concat(BRANCH)),
    original_frames: copy(REVISING),
    remask_edits: [{ frame_index: 4, token_positions: [2] }],
  }, frame);
}

// The spans the newest render drew. The stub keeps each document
// fragment as a node, and setting textContent leaves a node's
// children in place where a browser removes them, so every render's
// fragment stays behind and the newest is the last.
function renderedSpans(element) {
  const last = element.children[element.children.length - 1];
  if (last && last.tag === null) {
    return last.children;
  }
  return element.children;
}

// The spans one layer draws: the only layer of a run that stands
// alone, or the named one of the two a pre-edit snapshot stacks.
function layerSpans(page, layerClass) {
  const output = page.registry.get("overlay-output");
  if (!layerClass) {
    return renderedSpans(output);
  }
  const layers = output.children.filter(
    (child) => child.classList.contains(layerClass)
  );
  assert.ok(layers.length > 0, "no " + layerClass + " layer");
  return renderedSpans(layers[layers.length - 1]);
}

// What Revisions paints each position of a layer. An untinted token
// carries no colour of its own.
function revisionColors(page, layerClass) {
  page.context.setOverlayMode("revisions");
  return layerSpans(page, layerClass).map((span) => span.style.color);
}

// The values the overlay picker offers, read off the select the page
// built under its mount when the run opened.
function pickerValues(page) {
  const mount = page.registry.get("overlay-select-mount");
  const select = mount.children[mount.children.length - 1];
  const list = select.children.find((child) => child.tag === "ul");
  return list.children.map((item) => item.getAttribute("data-value"));
}

// The pointer over a position's span in one layer, as the page's own
// mouseover listener receives it, and the strip's reading of it.
function hoverExtra(page, layerClass, position) {
  const output = page.registry.get("overlay-output");
  output.dispatch("mouseover", {
    target: layerSpans(page, layerClass)[position],
  });
  const strip = page.registry.get("token-metrics");
  return strip.overlaysMetricNodes.extra.textContent;
}

// -- when it is offered --

test("a saved run that revised offers it, after Commit Order", () => {
  const page = pageWith({ frames: copy(REVISING) }, 6);

  const values = pickerValues(page);

  assert.equal(
    values.indexOf("revisions"), values.indexOf("commit") + 1
  );
});

test("a saved run that never revised is not offered it", () => {
  const page = pageWith({ frames: copy(SETTLING) }, 2);

  assert.ok(!pickerValues(page).includes("revisions"));
});

test("an autoregressive run is never offered it", async () => {
  // Opened the way the panel opens any run, since whether a run is
  // autoregressive is read off its catalog entry, not its frames.
  const data = withRecords({ frames: copy(REVISING) });
  const page = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: framesFetch(data),
  });
  const { context } = page;
  const run = { run_id: "run-ar", model_type: "autoregressive" };

  context.loadRunOverlays(
    run.run_id, run, context.detailRequests.begin(run.run_id)
  );
  await settle();

  // Heatmap is offered for any run with records, so the run opened.
  assert.ok(pickerValues(page).includes("heatmap"));
  assert.ok(!pickerValues(page).includes("revisions"));
});

// -- what it paints --

test("tokens are tinted by their count at the scrubbed frame", () => {
  const page = pageWith({ frames: copy(REVISING) }, 6);
  const { context } = page;

  const colors = revisionColors(page);

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("an earlier frame counts only what had happened by then", () => {
  const page = pageWith({ frames: copy(REVISING) }, 4);
  const { context } = page;

  const colors = revisionColors(page);

  assert.deepEqual(colors, [context.revisionColor(1), "", "", ""]);
});

test("each canvas counts on its own", () => {
  // Position 0 changes on canvas 0, then the next canvas puts an
  // unrelated token there, which is not a change of mind.
  const frames = [
    [settled(10)], [changing(20)], [settled(20)],
    [settled(30)], [settled(30)],
  ];
  const page = pageWith({
    frames: copy(frames), canvas_index: [0, 0, 0, 1, 1],
  }, 4);

  assert.deepEqual(revisionColors(page), [""]);
  page.context.setOverlayFrame(2);
  assert.deepEqual(
    revisionColors(page), [page.context.revisionColor(1)]
  );
});

test("the edited layer starts a remasked position over", () => {
  const page = editedRun(6);
  const { context } = page;

  const colors = revisionColors(page, "token-layer-edited");

  // Position 0's change at frame 3 is shared history; position 2's
  // new token is the edit's, position 3's the model's own.
  assert.deepEqual(colors, [
    context.revisionColor(1), "", "", context.revisionColor(1),
  ]);
});

test("the pre-edit layer counts the run it came from", () => {
  const page = editedRun(6);
  const { context } = page;

  const colors = revisionColors(page, "token-layer-original");

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("the pre-edit layer holds its last frame past its end", () => {
  const page = editedRun(7);
  const { context } = page;

  const colors = revisionColors(page, "token-layer-original");

  assert.deepEqual(colors, [
    context.revisionColor(2), context.revisionColor(1), "", "",
  ]);
});

test("the strip reads the hovered layer's count", () => {
  const page = editedRun(6);
  page.context.setOverlayMode("revisions");

  const edited = "token-layer-edited";
  const original = "token-layer-original";
  assert.equal(hoverExtra(page, edited, 3), "Revisions: 1");
  assert.equal(hoverExtra(page, edited, 1), "");
  assert.equal(hoverExtra(page, original, 1), "Revisions: 1");
  assert.equal(hoverExtra(page, original, 0), "Revisions: 2");
});

test("every position gets a span, tinted or not", () => {
  const page = pageWith({ frames: copy(REVISING) }, 6);

  const colors = revisionColors(page);

  assert.equal(colors.length, REVISING[6].length);
  assert.deepEqual(colors.slice(2), ["", ""]);
});

test("the legend shows only while Revisions is selected", () => {
  const page = pageWith({ frames: copy(REVISING) }, 6);
  const revisions = page.registry.get("overlay-revision-legend");
  const commit = page.registry.get("overlay-legend");

  page.context.setOverlayMode("revisions");
  assert.equal(revisions.hidden, false);
  assert.equal(commit.hidden, true);

  page.context.setOverlayMode("commit");
  assert.equal(revisions.hidden, true);
  assert.equal(commit.hidden, false);
});

test("loading another run forgets the last one's revisions", () => {
  // Held revisions would tint a run that never revised with the
  // previous run's counts, and keep offering their overlay.
  const page = pageWith({ frames: copy(REVISING) }, 6);
  assert.ok(pickerValues(page).includes("revisions"));
  page.context.setOverlayMode("revisions");

  page.context.clearOverlay();
  assert.equal(
    page.registry.get("overlay-revision-legend").hidden, true
  );
  page.context.renderRunOverlays(withRecords({
    frames: copy(SETTLING),
  }));

  assert.ok(!pickerValues(page).includes("revisions"));
  assert.deepEqual(revisionColors(page), ["", "", "", ""]);
});
