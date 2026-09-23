// The Analytics panel says which commit of a model produced a run.
//
// Strategy: load the Analytics page into the DOM stub beside this
// file and call the row builder with run records shaped the way the
// server sends them, then read the HTML it returns. Executing the
// real function rather than inspecting its source, because the thing
// worth checking is what reaches the panel.
//
// Why the row exists: "SmolLM3-3B" is not an identifier. A
// repository moves, and two runs of the same displayed model, with
// the same seed and the same parameters, can be runs of different
// weights. The app commit was already recorded; the model's was not,
// so the reproducibility block named half its inputs.
//
// Passing proves the commit reaches the panel when there is one,
// that it is shortened rather than shown in full, and that the three
// cases with nothing to say render no row at all: a run saved before
// pinning, a local checkpoint, and a run that carried no
// reproducibility block.

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

const SHA = "08b83a6feb34df1a6011b80c3c00c7563e963b07";

// The page fetches its catalog while booting; none of these tests
// are about that, so it gets an empty one.
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

function run(reproducibility) {
  const record = { run_id: "r1", backend: "smollm3" };
  if (reproducibility !== undefined) {
    record.reproducibility = reproducibility;
  }
  return record;
}

// -- when there is a commit --

test("the commit reaches the panel", () => {
  const { context } = page();

  const html = context.modelRevisionMetaRow(
    run({ model_revision: SHA })
  );

  assert.ok(html.indexOf("Model commit") >= 0);
  assert.ok(html.indexOf(SHA.slice(0, 12)) >= 0);
});

test("the full forty characters are not shown", () => {
  // Nobody verifies a sha by eye, and the full value would crowd
  // every other row in a panel of short labelled values.
  const { context } = page();

  const html = context.modelRevisionMetaRow(
    run({ model_revision: SHA })
  );

  assert.equal(html.indexOf(SHA), -1);
});

// -- when there is nothing to say --

test("a run saved before pinning renders no row", () => {
  // The corpus on disk predates the field. An empty value must read
  // as "not recorded" and not as a commit named "".
  const { context } = page();

  assert.equal(
    context.modelRevisionMetaRow(run({ model_revision: "" })),
    ""
  );
});

test("a local checkpoint renders no row", () => {
  // DiffusionGemma's real case: a directory with no commit. Absence
  // is the honest answer, so there is nothing to draw.
  const { context } = page();

  assert.equal(context.modelRevisionMetaRow(run({})), "");
});

test("a run with no reproducibility block renders no row", () => {
  // The page reads runs off disk, so it cannot assume the block is
  // there at all; reaching into a missing object would throw and
  // take the whole panel with it.
  const { context } = page();

  assert.equal(context.modelRevisionMetaRow(run()), "");
});

// -- the row is actually wired in --

test("the model metadata includes the commit row", () => {
  // The pair to the tests above: a builder nothing calls would pass
  // every one of them and show the user nothing.
  const { context } = page();

  const html = context.renderRunMeta(
    run({ model_revision: SHA })
  );

  assert.ok(html.indexOf("Model commit") >= 0);
});
