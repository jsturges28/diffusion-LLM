// The Analytics detail reports what a run cost the card.
//
// Strategy: load the Analytics page into the DOM stub beside this file
// and ask it to format figures and build the row directly, with run
// payloads shaped like the three cases that reach it: a CUDA run, a
// CPU run whose worker measured nothing, and a run saved before any of
// this existed.
//
// The figures used are a resident LLaDA in bf16 plus a small per-step
// transient, because that ratio is the reason the row carries two
// numbers. A peak of 17 GiB moves by a fraction of a percent when the
// sampler stops building a canvas-wide softmax, so a row showing only
// the peak would report the improvement as no change at all.
//
// Passing proves the row states the peak and the distance above the
// baseline, picks a unit that suits each of those two very different
// magnitudes, and is absent rather than zeroed whenever there was no
// measurement to report.

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

const GIB = 1024 * 1024 * 1024;
const MIB = 1024 * 1024;

const WEIGHTS = 17 * GIB;
const TRANSIENT = 15 * MIB;

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

function page() {
  return loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  }).context;
}

// A saved run carrying the block the worker attests.
function withCost(start, peak) {
  return {
    resources: {
      vram_allocated_start_bytes: start,
      vram_allocated_peak_bytes: peak,
      vram_reserved_peak_bytes: peak + 256 * MIB,
    },
  };
}

// -- the units --

test("a peak reads in gibibytes", () => {
  const context = page();

  assert.equal(context.formatVramBytes(17 * GIB), "17.00 GiB");
});

test("a transient reads in mebibytes", () => {
  // The other end of the range. In GiB this would be "0.01 GiB",
  // which is the same string for anything between 5 and 15 MiB and so
  // cannot show the change the measurement exists to show.
  const context = page();

  assert.equal(context.formatVramBytes(15 * MIB), "15.0 MiB");
});

test("the boundary falls on the gibibyte", () => {
  // Stated because an off-by-one here would print "1024.0 MiB"
  // instead of "1.00 GiB", which is correct and still wrong.
  const context = page();

  assert.equal(context.formatVramBytes(GIB), "1.00 GiB");
  assert.equal(context.formatVramBytes(GIB - 1), "1024.0 MiB");
});

// -- the row --

test("the row states the peak and the distance above it", () => {
  const context = page();

  const html = context.peakVramMetaRow(
    withCost(WEIGHTS, WEIGHTS + TRANSIENT)
  );

  assert.match(html, /Peak VRAM/);
  assert.match(html, /17\.01 GiB/);
  assert.match(html, /15\.0 MiB above baseline/);
});

test("a larger transient reads as a larger figure", () => {
  // The comparison the maintainer actually makes: the same run before
  // and after the chunked reduction. If the row could not tell 96 MiB
  // from 15, it would not be worth the field.
  const context = page();

  const before = context.peakVramMetaRow(
    withCost(WEIGHTS, WEIGHTS + 96 * MIB)
  );
  const after = context.peakVramMetaRow(
    withCost(WEIGHTS, WEIGHTS + TRANSIENT)
  );

  assert.match(before, /96\.0 MiB above baseline/);
  assert.match(after, /15\.0 MiB above baseline/);
});

test("a run that added nothing says so rather than nothing", () => {
  // Zero above baseline is a real measurement, unlike an absent
  // block, so the row stays and reports it.
  const context = page();

  const html = context.peakVramMetaRow(withCost(WEIGHTS, WEIGHTS));

  assert.match(html, /0\.0 MiB above baseline/);
});

// -- absence --

test("a run with no cost block gets no row", () => {
  // A CPU run, and every run saved before this existed.
  const context = page();

  assert.equal(context.peakVramMetaRow({}), "");
});

test("a block missing the baseline gets no row", () => {
  // Defensive against a worker that reports half. The row's second
  // figure is a subtraction, and without the baseline there is
  // nothing honest to subtract.
  const context = page();

  const html = context.peakVramMetaRow({
    resources: { vram_allocated_peak_bytes: WEIGHTS },
  });

  assert.equal(html, "");
});

test("a block missing the peak gets no row", () => {
  const context = page();

  const html = context.peakVramMetaRow({
    resources: { vram_allocated_start_bytes: WEIGHTS },
  });

  assert.equal(html, "");
});
