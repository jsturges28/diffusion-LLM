// Comparison work stops when a run's detail takes over the page.
//
// Strategy: load the Analytics page into the DOM stub with a fetch
// that holds the comparison request until the test releases it, and a
// recording Chart that keeps what gets built. Open a comparison, open
// a run's detail before the comparison answers, then let the answer
// arrive: the order a quick click produces against a slow server.
//
// The defect being pinned: entering detail hid the comparison panel
// but did not cancel its request, so a late answer still built its
// chart and omission rows behind the dialog, for a view the user had
// left. Entering a comparison already retired the detail's requests,
// and closing one retired its own; this was the one route that did
// not. Passing proves the late answer builds nothing and the request
// is aborted, and the control proves the same answer does build its
// chart when nothing intervenes.

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

// Two runs' convergence, in the shape the comparison answers with.
const RESULTS = [
  {
    run_id: "a",
    status: "data",
    label: "A",
    convergence: [{ resolved_ratio: 0.5 }, { resolved_ratio: 1 }],
  },
  {
    run_id: "b",
    status: "data",
    label: "B",
    convergence: [{ resolved_ratio: 1 }],
  },
];

function answer(body) {
  return {
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

// Holds the comparison until `release` is called and keeps the signal
// it was sent with. The catalog is empty, and anything else never
// answers, since what a run's detail would paint is not under test.
function heldFetch() {
  const held = { release: null, signal: null };
  const fetchImpl = function (url, init) {
    const path = String(url).split("?")[0];
    if (path === "/api/analytics/compare") {
      held.signal = init ? init.signal : null;
      return new Promise((resolve) => {
        held.release = () => resolve(answer(RESULTS));
      });
    }
    if (path === "/api/analytics/runs") {
      return Promise.resolve(answer([]));
    }
    return new Promise(() => {});
  };
  return { held, fetchImpl };
}

// The page, with charts available and every one it builds recorded.
function analyticsPage(fetchImpl) {
  const { context } = loadPage({
    scripts: ANALYTICS_SCRIPTS,
    fetchImpl: fetchImpl,
  });
  const built = [];
  context.Chart = function (ctx, config) {
    built.push(config);
    return {
      data: config.data,
      options: config.options,
      setActiveElements() {},
      update() {},
      destroy() {},
      resize() {},
    };
  };
  context.chartsAvailable = true;
  return { context, built };
}

// Long enough for a released answer to work through its promises.
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("a comparison answering late builds nothing", async () => {
  const { held, fetchImpl } = heldFetch();
  const { context, built } = analyticsPage(fetchImpl);

  context.showComparison(["a", "b"]);
  context.showDetail("a");
  held.release();
  await settle();

  assert.equal(built.length, 0);
  assert.equal(context.chartCompareConv, null);
});

test("opening a run aborts the comparison in flight", () => {
  const { held, fetchImpl } = heldFetch();
  const { context } = analyticsPage(fetchImpl);

  context.showComparison(["a", "b"]);
  context.showDetail("a");

  assert.equal(held.signal.aborted, true);
});

test("left alone, the same answer builds its chart", async () => {
  // The control: without it, a recorder that never saw a chart would
  // pass the first test for the wrong reason.
  const { held, fetchImpl } = heldFetch();
  const { context, built } = analyticsPage(fetchImpl);

  context.showComparison(["a", "b"]);
  held.release();
  await settle();

  assert.equal(built.length, 1);
  assert.notEqual(context.chartCompareConv, null);
});
