// The generator opens on state it was handed, not state it asked for.
//
// Strategy: load the real page into the DOM stub twice, once with the
// boot state the server inlines and once without, and compare what it
// fetched before it had drawn anything. Nothing here inspects source
// text: the claim is about which requests the page makes and when, so
// it is checked by making the page boot.
//
// The page used to chain two round trips. `persistHydrate` fetched
// `/api/ui-state`, and only in its callback did `boot` fetch
// `/api/models`, because the second read what the first had written
// into localStorage. Until both answered, the hyperparameter column
// was an empty container and around twenty elements were sitting at
// their markup defaults, so the first paint was a skeleton and the
// second was the page.
//
// Passing proves the common path takes neither trip, that the same
// page still works when served without the state (which is how these
// very tests load it, and how a file opened directly loads), and that
// the one genuinely expensive field is fetched after first paint
// rather than in front of it.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SMOL = {
  id: "smollm3",
  display_name: "SmolLM3-3B",
  min_vram_gib: 6,
  capabilities: { model_type: "autoregressive" },
  param_specs: [
    {
      name: "max_new_tokens",
      label: "Max new tokens",
      type: "int",
      default: 256,
      min: 1,
      max: 2048,
    },
    {
      name: "temperature",
      label: "Temperature",
      type: "float",
      default: 0.7,
      min: 0,
      max: 2,
      step: 0.1,
    },
  ],
  status: "active",
};

// Shaped like the `/api/models` body, which is the point: the page
// feeds it to the same code either way.
function modelsPayload(extra) {
  return Object.assign({
    models: [SMOL],
    active: "smollm3",
    active_device: "cuda",
    active_tokenizer: { name: "smollm3" },
    active_context_length: 65536,
    default: "smollm3",
    gpu_name: "NVIDIA GeForce RTX 4090",
  }, extra || {});
}

function bootState() {
  return {
    ui_state: { diffusion_settings: JSON.stringify({ theme: "dark" }) },
    models: modelsPayload(),
  };
}

// Requests recorded by url, so a test can ask what the page wanted
// before it drew. Answers `/api/models` properly and everything else
// with an empty object.
function recordingFetch(urls) {
  return function (url) {
    urls.push(String(url).split("?")[0]);
    const body = String(url).startsWith("/api/models")
      ? modelsPayload({ models: [
        Object.assign({}, SMOL, { vram_headroom_gib: 5.2 }),
      ] })
      : {};
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  };
}

// -- with the state inlined --

test("the page boots without asking for what it was given", () => {
  const urls = [];

  loadPage({ bootState: bootState(), fetchImpl: recordingFetch(urls) });

  assert.equal(
    urls.includes("/api/ui-state"), false,
    "the durable state was inlined and still fetched"
  );
});

test("the parameter column is filled before anything is awaited", () => {
  const urls = [];
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch(urls),
  });

  // Synchronously after load: no microtask has been drained, so this
  // is what the browser would have painted.
  const fields = page.registry.get("param-fields");
  assert.ok(fields, "the page never looked for the parameter column");
  assert.ok(
    fields.children.length > 0,
    "the hyperparameter column was still empty at first paint"
  );
});

test("the resident model is known at first paint", () => {
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch([]),
  });

  assert.equal(page.sandbox.activeModelId, "smollm3");
  assert.equal(page.sandbox.activeDevice, "cuda");
  assert.equal(page.sandbox.activeContextLength, 65536);
  assert.equal(page.sandbox.gpuPresent, true);
});

test("the entropy row is settled for the resident model", () => {
  // The conditional reservation: held for a model that reports
  // entropy, absent for one that never will. It used to be answered
  // after the fetch, which is a strip appearing under the canvas.
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch([]),
  });

  const row = page.registry.get("entropy-profile-row");
  assert.equal(
    row.hidden, false,
    "SmolLM3 is autoregressive, so its row is reserved"
  );
});

// -- and the one field worth a probe --

test("VRAM headroom is fetched, but after the page is drawn", async () => {
  const urls = [];
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch(urls),
  });

  const drawnWith = page.registry.get("param-fields").children.length;
  assert.ok(drawnWith > 0);
  assert.equal(
    urls.includes("/api/models"), true,
    "headroom feeds the hover popover and still has to arrive"
  );

  await new Promise((resolve) => setTimeout(resolve, 0));

  const model = page.sandbox.models.smollm3;
  assert.equal(
    model.vram_headroom_gib, 5.2,
    "the refresh landed but its answer was dropped"
  );
});

// -- and without it, which is not a hypothetical --

test("a page served without the state still fetches both", async () => {
  // How these tests load the page, and how a file opened directly
  // loads it. If this breaks, the harness breaks with it.
  const urls = [];

  loadPage({ fetchImpl: recordingFetch(urls) });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(urls.includes("/api/ui-state"), true);
  assert.equal(urls.includes("/api/models"), true);
});

test("the fallback reaches the same page", async () => {
  const page = loadPage({ fetchImpl: recordingFetch([]) });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(page.sandbox.activeModelId, "smollm3");
  assert.ok(page.registry.get("param-fields").children.length > 0);
});

// Each of these would, if adopted, give a page with no models and no
// intention of asking for any. `null` is the one worth spelling out:
// `typeof null` is "object", so a guard written only as a typeof test
// lets it straight through.
const MALFORMED = [
  ["null", null],
  ["a string", "nonsense"],
  ["a number", 0],
  ["absent", undefined],
];

// -- measuring text before the font exists --

test("the field width is measured again once the font lands", async () => {
  // Canvas measureText answers with whatever font is available, so a
  // measurement taken before the webfont arrives is the fallback's.
  // `--param-width` decides where the hyperparameter row wraps, so a
  // few pixels out moves a field to a second line and drops the whole
  // column below it. This used to be safe by accident: the
  // measurement sat inside the /api/models callback, and a network
  // round trip outlasted the font every time. Synchronous boot put it
  // in front.
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch([]),
  });
  const root = page.document.documentElement;

  const measured = [];
  const inner = root.style.setProperty.bind(root.style);
  root.style.setProperty = function (name, value) {
    measured.push(name);
    return inner(name, value);
  };

  assert.equal(
    measured.length, 0,
    "nothing should have measured before the font is announced"
  );

  await page.announceFontsLoaded();
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.ok(
    measured.includes("--param-width"),
    "the width was never re-measured, so it keeps the fallback's"
  );
});

test("a page with no font API still sizes its fields", () => {
  // The guard matters: `document.fonts` is not universal, and a page
  // that only measured in the callback would ship an unsized column
  // to anything lacking it.
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: recordingFetch([]),
  });

  const width = page.document.documentElement.style["--param-width"];
  assert.ok(width, "no width was set on the synchronous path");
  assert.match(String(width), /^\d+px$/);
});

// -- and the same bargain on Analytics --

// The page's real script order, minus nothing: `model_client.js` came
// out of it when the nav link stopped being a fetch.
const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

const RUN_ROW = {
  run_id: "2026-09-01_02-29-09_smollm3",
  prompt: "Explain diffusion",
  model: "SmolLM3-3B",
  backend: "SmolLM3-3B",
  created_at: "2026-09-01T02:29:09",
  steps: 128,
};

function analyticsBoot() {
  return {
    ui_state: {},
    runs: [RUN_ROW],
    collections: [{ id: "favorites", name: "Favorites", runs: [] }],
    results_dir: "/tmp/isolated",
  };
}

function analyticsPage(options) {
  const settings = options || {};
  return loadPage({
    scripts: ANALYTICS_SCRIPTS,
    bootState: settings.bootState,
    fetchImpl: settings.fetchImpl,
  });
}

// Answers the catalog with a bare array, collections with the
// envelope the client unwraps, and everything else with an empty
// object, recording what was asked for.
function analyticsFetch(urls) {
  return function (url) {
    const path = String(url).split("?")[0];
    urls.push(path);
    let body = {};
    if (path === "/api/analytics/runs") {
      body = [RUN_ROW];
    } else if (path === "/api/collections") {
      body = { success: true, collections: [] };
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    });
  };
}

test("the table is populated before any catalog fetch", () => {
  const urls = [];
  const page = analyticsPage({
    bootState: analyticsBoot(),
    fetchImpl: analyticsFetch(urls),
  });

  assert.equal(
    page.sandbox.allRuns.length, 1,
    "the catalog was inlined and the page still started empty"
  );
  assert.equal(urls.includes("/api/analytics/runs"), false);
  assert.equal(urls.includes("/api/collections"), false);
});

test("the data root is known before a delete can be confirmed", () => {
  // The confirmation names the directory it is about to delete from,
  // and used to spell it from a default until a fetch corrected it.
  const page = analyticsPage({
    bootState: analyticsBoot(),
    fetchImpl: analyticsFetch([]),
  });

  assert.equal(page.sandbox.resultsDirLabel, "/tmp/isolated");
});

test("collections arrive with the table, not after it", () => {
  const page = analyticsPage({
    bootState: analyticsBoot(),
    fetchImpl: analyticsFetch([]),
  });

  assert.equal(page.sandbox.collections.length, 1);
  assert.equal(page.sandbox.collections[0].id, "favorites");
});

test("Analytics still fetches when served without the state", async () => {
  const urls = [];
  const page = analyticsPage({ fetchImpl: analyticsFetch(urls) });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(urls.includes("/api/analytics/runs"), true);
  assert.equal(page.sandbox.allRuns.length, 1);
});

test("a catalog that is not a list is refetched", async () => {
  // Adopting it would leave a page with no runs that had already
  // decided not to ask for any.
  const urls = [];
  const broken = analyticsBoot();
  broken.runs = "nonsense";

  const page = analyticsPage({
    bootState: broken,
    fetchImpl: analyticsFetch(urls),
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(urls.includes("/api/analytics/runs"), true);
  assert.equal(page.sandbox.allRuns.length, 1);
});

test("collections that are not a list are refetched too", async () => {
  const urls = [];
  const broken = analyticsBoot();
  broken.collections = null;

  analyticsPage({
    bootState: broken,
    fetchImpl: analyticsFetch(urls),
  });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(urls.includes("/api/analytics/runs"), true);
});

// -- the malformed cases, on the generator --

for (const [label, value] of MALFORMED) {
  test(`inlined state that is ${label} falls back`, async () => {
    // Asserted on the page rather than on the request log, which was
    // the weaker claim: the headroom refresh calls `/api/models` on
    // the inlined path too, so "it was requested" is true even when
    // the page adopted rubbish and rendered nothing.
    const urls = [];
    const page = loadPage({
      bootState: { ui_state: value, models: value },
      fetchImpl: recordingFetch(urls),
    });
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.equal(
      page.sandbox.activeModelId, "smollm3",
      `${label} models was adopted, leaving no resident model`
    );
    assert.ok(
      page.registry.get("param-fields").children.length > 0,
      `${label} models was adopted, leaving an empty panel`
    );
    assert.equal(
      urls.includes("/api/ui-state"), true,
      `${label} ui_state was adopted instead of refetched`
    );
  });
}
