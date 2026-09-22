// The modals, driven as dialogs.
//
// Strategy: load the generator into the DOM stub, open and close its
// modals through the same calls the page makes, and read back what
// happened. `test_keyboard_reach.py` covers the markup and the CSS;
// this covers the behaviour those cannot see.
//
// The migration's real prize is that `close` is a single funnel.
// Before it, each dismissal route did its own tidying: the close
// button, the backdrop and a hand-rolled Escape listener each had to
// remember, and any route that forgot leaked state into the next
// open. Native Escape does not pass through page code at all, so
// anything not hanging off `close` would simply not run for it.
//
// Passing proves a modal opens as modal rather than inline, that
// state is dropped however it was dismissed, and that raising the
// loading curtain clears the dialogs it can no longer cover.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

function bootState() {
  return {
    ui_state: {},
    models: {
      models: [{
        id: "smollm3",
        display_name: "SmolLM3-3B",
        min_vram_gib: 6,
        capabilities: {
          family: "autoregressive",
          generation_shape: "append_only",
          supported_devices: ["cuda", "cpu"],
        },
        param_specs: [],
        status: "active",
      }],
      active: "smollm3",
      active_device: "cuda",
      active_tokenizer: {},
      active_context_length: 65536,
      default: "smollm3",
      gpu_name: "NVIDIA GeForce RTX 4090",
    },
  };
}

function generator() {
  const page = loadPage({
    bootState: bootState(),
    fetchImpl: () => Promise.resolve({
      ok: true, status: 200, json: () => Promise.resolve({}),
    }),
  });
  return {
    page,
    about: page.registry.get("modal-about"),
    help: page.registry.get("modal-help"),
    imports: page.registry.get("modal-import"),
  };
}

// -- opening --

test("a modal opens as modal, not inline", () => {
  // `show()` would put it on screen and trap nothing, which is the
  // whole reason for the migration: the previous implementation also
  // put modals on screen and left Tab walking the page behind them.
  const h = generator();

  h.page.context.openModal(h.about);

  assert.equal(h.about.open, true);
  assert.equal(
    h.about.openedModally, true,
    "opened with show(), which traps no focus"
  );
});

test("opening an open modal does not throw", () => {
  // `showModal` on an already-open dialog is a DOM exception, and
  // two paths can reach the same open.
  const h = generator();
  h.page.context.openModal(h.about);

  assert.doesNotThrow(() => h.page.context.openModal(h.about));
});

test("closing a closed modal is quiet too", () => {
  const h = generator();

  assert.doesNotThrow(() => h.page.context.closeModal(h.about));
  assert.equal(h.about.open, false);
});

// -- closing, whichever way --

test("a click beside the box closes it", () => {
  // The dialog fills the viewport and centres the box inside, so a
  // click on the dialog itself landed outside the box. The backdrop
  // pseudo-element cannot be hit directly, which is why this is
  // tested against the element.
  const h = generator();
  h.page.context.openModal(h.about);

  h.about.dispatch("click", { target: h.about });

  assert.equal(h.about.open, false);
});

test("a click on the box does not", () => {
  const h = generator();
  h.page.context.openModal(h.about);
  const box = h.about.children[0];

  h.about.dispatch("click", { target: box });

  assert.equal(h.about.open, true);
});

test("the pending import is dropped on close, however it closed", () => {
  // The case the funnel exists for. Native Escape runs no page code,
  // so a cleanup hanging off the close button alone would leave the
  // file staged and a later import would act on it.
  const h = generator();
  h.page.context.pendingImportFile = { name: "notes.txt" };
  h.page.context.openModal(h.imports);

  h.imports.close();

  assert.equal(h.page.context.pendingImportFile, null);
});

test("closing one modal leaves another alone", () => {
  const h = generator();
  h.page.context.openModal(h.about);
  h.page.context.openModal(h.help);

  h.page.context.closeModal(h.about);

  assert.equal(h.help.open, true);
});

// -- and the curtain that can no longer cover them --

test("raising the loading overlay closes the dialogs", () => {
  // A dialog is in the top layer, above every z-index, so the
  // overlay at 100 would otherwise have About floating over it.
  const h = generator();
  h.page.context.openModal(h.about);

  h.page.context.raiseLoadingOverlay();

  assert.equal(h.about.open, false);
  assert.equal(
    h.page.registry.get("loading-overlay").classes.has("hidden"),
    false
  );
});

test("it still raises the overlay when nothing is open", () => {
  const h = generator();

  h.page.context.raiseLoadingOverlay();

  assert.equal(
    h.page.registry.get("loading-overlay").classes.has("hidden"),
    false
  );
});
