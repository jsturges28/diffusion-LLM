// The Main Menu's model rows, driven by keys.
//
// Strategy: load the real menu page into the DOM stub with a fetch
// that answers `/api/models`, let it build its rows, then send events
// at them. `menu.js` is a closed IIFE that exports nothing, so
// everything here goes through the DOM it produces, which is the only
// surface it has and, conveniently, the same one a user has.
//
// Two things are covered. Enter on a Confirm or Cancel button used to
// be swallowed: the confirm popover is a child of the row, so its
// keys bubble to the row's handler, which called preventDefault and
// cancelled the click the browser was about to synthesise. The
// popover could be reached by Tab and then not operated.
//
// And the row's device buttons were tab stops, so Tab moved between
// GPU and CPU rather than between models, which is the opposite of
// what it should do. They are out of the tab order now, with Left and
// Right in their place.
//
// The stub does not synthesise a click from Enter, so the first group
// asserts the mechanism (preventDefault is not called, the row does
// not act) rather than the outcome.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const MENU_SCRIPTS = [
  "overlays.js",
  "model_client.js",
  "activation_client.js",
  "download_client.js",
  "download_toast.js",
  "menu.js",
];

const LLADA = {
  id: "llada",
  display_name: "LLaDA-8B-Instruct",
  description: "Masked discrete diffusion",
  min_vram_gib: 17,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    supported_devices: ["cuda"],
  },
  status: "idle",
  fits: true,
  downloaded: true,
  downloadable: true,
  vram_headroom_gib: 5.2,
};
// The only model declaring two placements, so the only row with a
// device choice. The others are GPU-only and carry a static tag.
const SMOL = {
  id: "smollm3",
  display_name: "SmolLM3-3B",
  description: "Autoregressive baseline",
  min_vram_gib: 6,
  capabilities: {
    family: "autoregressive",
    generation_shape: "append_only",
    supported_devices: ["cuda", "cpu"],
  },
  status: "idle",
  fits: true,
  downloaded: true,
  downloadable: true,
  vram_headroom_gib: 16.4,
};

// A diffusion model that will not fit. `buildRow` never wires it, so
// it gets no tabIndex and is not in the tab order; arrow traversal
// has to agree with that rather than inventing a way in.
const TOO_BIG = {
  id: "diffusiongemma",
  display_name: "DiffusionGemma-26B-A4B",
  description: "Block-autoregressive MoE",
  min_vram_gib: 18,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    supported_devices: ["cuda"],
  },
  status: "idle",
  fits: false,
  downloaded: true,
  downloadable: false,
  vram_headroom_gib: -4.1,
};

function modelsBody() {
  return {
    // Ordered so the unwired row sits between two wired ones, which
    // is the arrangement that catches a traversal stepping onto it.
    models: [LLADA, TOO_BIG, SMOL],
    active: null,
    active_device: null,
    active_tokenizer: {},
    active_context_length: null,
    default: "smollm3",
    gpu_name: "NVIDIA GeForce RTX 4090",
    free_vram_gib: 23.0,
    gpu_status: "ok",
  };
}

function menuFetch(calls) {
  return function (url, init) {
    const path = String(url).split("?")[0];
    calls.push({ path, init: init || {} });
    let body = {};
    if (path === "/api/models") {
      body = modelsBody();
    }
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  };
}

// Boot the page and wait for the model fetch to render its rows.
async function menu() {
  const calls = [];
  const page = loadPage({
    scripts: MENU_SCRIPTS, fetchImpl: menuFetch(calls),
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  const list = page.registry.get("menu-model-list");
  return { page, calls, list, rows: list.children };
}

function rowFor(harness, id) {
  const row = harness.rows.find(
    (li) => li.getAttribute("data-id") === id
  );
  assert.ok(row, `no row for ${id}`);
  return row;
}

function key(el, name, event) {
  el.dispatch(
    "keydown",
    Object.assign({ key: name, preventDefault() {} }, event || {})
  );
}

function devicePills(row) {
  return row.querySelectorAll(".menu-device-btn");
}

function activePill(row) {
  const on = devicePills(row).find((b) => b.classes.has("is-active"));
  return on ? on.textContent : null;
}

// -- the rows render at all --

test("the menu builds a row per model", async () => {
  const h = await menu();

  assert.equal(h.rows.length, 3);
  assert.ok(rowFor(h, "llada"));
  assert.ok(rowFor(h, "smollm3"));
});

test("a model that will not fit is not in the tab order", async () => {
  const h = await menu();

  assert.equal(rowFor(h, "diffusiongemma").tabIndex, undefined);
  assert.equal(rowFor(h, "llada").tabIndex, 0);
});

// -- Enter reaches the confirm buttons --

async function confirming(id) {
  const h = await menu();
  const row = rowFor(h, id);
  row.dispatch("click", { target: row });
  const box = row.querySelector(".menu-model-confirm");
  assert.ok(box, "no confirm popover appeared");
  return { h, row, box };
}

test("Enter inside the confirm is left for the button", async () => {
  const { row, box } = await confirming("llada");
  const yes = box.querySelector(".menu-confirm-yes");
  assert.ok(yes);

  let prevented = false;
  row.dispatch("keydown", {
    key: "Enter",
    target: yes,
    preventDefault() { prevented = true; },
  });

  assert.equal(
    prevented, false,
    "the row cancelled the click the button was about to get"
  );
});

test("and the row does not re-confirm on top of itself", async () => {
  const { row, box } = await confirming("llada");

  row.dispatch("keydown", {
    key: "Enter",
    target: box.querySelector(".menu-confirm-no"),
    preventDefault() {},
  });

  assert.equal(
    row.querySelectorAll(".menu-model-confirm").length, 1,
    "the row opened a second confirm behind the first"
  );
});

test("Enter on the row itself still opens the confirm", async () => {
  // The guard is scoped to buttons inside the row, not to the row.
  const h = await menu();
  const row = rowFor(h, "llada");

  let prevented = false;
  row.dispatch("keydown", {
    key: "Enter",
    target: row,
    preventDefault() { prevented = true; },
  });

  assert.equal(prevented, true);
  assert.ok(row.querySelector(".menu-model-confirm"));
});

// -- devices move with arrows, not with Tab --

test("the device buttons are out of the tab order", async () => {
  const h = await menu();
  const pills = devicePills(rowFor(h, "smollm3"));

  assert.ok(pills.length >= 2);
  for (const pill of pills) {
    assert.equal(pill.tabIndex, -1);
  }
});

test("Right and Left move the device", async () => {
  const h = await menu();
  const row = rowFor(h, "smollm3");
  assert.equal(activePill(row), "GPU");

  key(row, "ArrowRight", { target: row });
  assert.equal(activePill(row), "CPU");

  key(row, "ArrowLeft", { target: row });
  assert.equal(activePill(row), "GPU");
});

test("the arrow moves the value the request will use", async () => {
  // The row keeps its device in a closure behind `_getDevice`, which
  // is what `selectModel` posts. Moving only the class would show one
  // device and load the other.
  const h = await menu();
  const row = rowFor(h, "smollm3");
  assert.equal(row._getDevice(), "cuda");

  key(row, "ArrowRight", { target: row });

  assert.equal(row._getDevice(), "cpu");
});

test("a row with no choice ignores the arrows", async () => {
  const h = await menu();
  const row = rowFor(h, "llada");

  key(row, "ArrowRight", { target: row });

  assert.equal(activePill(row), "GPU");
});

// -- and up and down move between models --

test("Down steps over the row it cannot reach", async () => {
  // DiffusionGemma sits between the two wired rows and was never
  // wired, so Down from LLaDA must land past it rather than on it.
  const h = await menu();
  const first = rowFor(h, "llada");
  let focused = null;
  for (const row of h.rows) {
    row.focus = function () { focused = row; };
  }

  key(first, "ArrowDown", { target: first });

  assert.equal(focused, rowFor(h, "smollm3"));
  assert.notEqual(focused, rowFor(h, "diffusiongemma"));
});

test("Up from the first row wraps to the last", async () => {
  const h = await menu();
  const first = rowFor(h, "llada");
  let focused = null;
  for (const row of h.rows) {
    row.focus = function () { focused = row; };
  }

  key(first, "ArrowUp", { target: first });

  assert.equal(focused, rowFor(h, "smollm3"));
});
