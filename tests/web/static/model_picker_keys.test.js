// Driving the model picker with nothing but keys.
//
// Strategy: load the generator into the DOM stub with an inlined boot
// state, then send keydown events at `#model-select` and read back
// what moved. `requestSwitch` is replaced so a test can see what
// would have been asked for without a worker anywhere.
//
// This picker is not `createCustomSelect`. It is a second, hand-built
// dropdown whose rows carry a name, a device control and a headroom
// popover, and `RUNTIME-03` left it alone. Before this it answered
// Enter, Space and Escape only, so the sole keyboard route to a model
// was Tab, which walked through the GPU/CPU buttons inside the rows
// rather than between models: two stops on the one autoregressive
// model, none on the others.
//
// The risky half is the swap that makes that tidy. Taking those
// buttons out of the tab order removes the only way a keyboard could
// reach a device, so Left and Right have to work or this is a
// regression wearing an improvement's clothes. Most of the file is
// about that.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const LLADA = {
  id: "llada",
  display_name: "LLaDA-8B-Instruct",
  min_vram_gib: 17,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    supported_devices: ["cuda"],
  },
  param_specs: [],
  status: "idle",
};
const DGEMMA = {
  id: "diffusiongemma",
  display_name: "DiffusionGemma-26B-A4B",
  min_vram_gib: 18,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    supported_devices: ["cuda"],
  },
  param_specs: [],
  status: "idle",
};
// The only one declaring two placements, so the only row with two
// device pills. The others are GPU-only and carry a static pill.
const SMOL = {
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
};

function bootState(options) {
  const settings = options || {};
  const hasGpu = settings.gpu !== false;
  return {
    ui_state: {},
    models: {
      models: [LLADA, DGEMMA, SMOL],
      active: "smollm3",
      active_device: hasGpu ? "cuda" : "cpu",
      active_tokenizer: {},
      active_context_length: 65536,
      default: "smollm3",
      // Absent means no GPU, which is what disables the CUDA pill.
      gpu_name: hasGpu ? "NVIDIA GeForce RTX 4090" : null,
    },
  };
}

function inertFetch() {
  return () => Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
  });
}

// Load the page and hand back the picker plus a log of switches it
// would have requested.
function picker(options) {
  const page = loadPage({
    bootState: bootState(options), fetchImpl: inertFetch(),
  });
  const switches = [];
  page.context.requestSwitch = (id, device) => {
    switches.push({ id, device });
  };
  return {
    page,
    switches,
    select: page.registry.get("model-select"),
    list: page.registry.get("model-select-list"),
  };
}

function key(el, name) {
  el.dispatch("keydown", { key: name, preventDefault() {} });
}

// Arrow down until the wanted row is the focused one. Spelled out
// rather than assuming a key count, because traversal starts from
// whichever model is resident and wraps, so "how many presses to
// reach SmolLM3" is a fact about the fixture, not about the widget.
function focusRow(harness, id) {
  for (let i = 0; i < 12; i += 1) {
    if (focusedName(harness) === id) {
      return;
    }
    key(harness.select, "ArrowDown");
  }
  throw new Error(`never reached the ${id} row`);
}

function rows(harness) {
  return harness.list.children;
}

function focusedRow(harness) {
  return rows(harness).find((li) => li.classes.has("is-focused"));
}

function focusedName(harness) {
  const row = focusedRow(harness);
  if (!row) {
    return null;
  }
  return row.getAttribute("data-id");
}

function pills(row) {
  return row.querySelectorAll(".device-pill-btn");
}

function activeDeviceOf(row) {
  const on = pills(row).find((b) => b.classes.has("is-active"));
  return on ? on.getAttribute("data-device") : null;
}

// -- moving between models --

test("Down opens the list and moves off the resident model", () => {
  const h = picker();

  key(h.select, "ArrowDown");

  assert.equal(h.list.hidden, false);
  assert.equal(
    focusedName(h), "llada",
    "traversal should start from the resident model and wrap"
  );
});

test("Up and Down walk the list and wrap at both ends", () => {
  const h = picker();

  key(h.select, "ArrowDown");
  assert.equal(focusedName(h), "llada");

  key(h.select, "ArrowDown");
  assert.equal(focusedName(h), "diffusiongemma");

  key(h.select, "ArrowUp");
  assert.equal(focusedName(h), "llada");

  key(h.select, "ArrowUp");
  assert.equal(focusedName(h), "smollm3");
});

test("the device buttons are out of the tab order", () => {
  // The swap that makes one tab stop out of three. Only safe
  // because the tests below reach them another way.
  const h = picker();
  key(h.select, "ArrowDown");

  const smol = rows(h).find(
    (li) => li.getAttribute("data-id") === "smollm3"
  );
  const buttons = pills(smol);

  assert.ok(buttons.length >= 2);
  for (const button of buttons) {
    assert.equal(button.tabIndex, -1);
  }
});

// -- and between a model's devices --

test("Right and Left move the targeted device", () => {
  const h = picker();

  // Onto SmolLM3, the one row with a choice.
  focusRow(h, "smollm3");
  const row = focusedRow(h);
  assert.equal(activeDeviceOf(row), "cuda");

  key(h.select, "ArrowRight");
  assert.equal(activeDeviceOf(row), "cpu");

  key(h.select, "ArrowLeft");
  assert.equal(activeDeviceOf(row), "cuda");
});

test("moving the device does not switch on its own", () => {
  // A pointer commits on click; a keyboard needs somewhere to stand
  // between pointing at a device and choosing it.
  const h = picker();
  focusRow(h, "smollm3");

  key(h.select, "ArrowRight");

  assert.deepEqual(h.switches, []);
});

test("a GPU that does not exist cannot be arrowed onto", () => {
  // On a host with no GPU the CUDA pill ships disabled, and the
  // pointer cannot click it. Traversal has to agree, or the keyboard
  // becomes the one way to ask for a device that is not there.
  const h = picker({ gpu: false });
  focusRow(h, "smollm3");
  const row = focusedRow(h);
  assert.equal(activeDeviceOf(row), "cpu");

  // One press, deliberately. Right then Left lands back on CPU
  // whether or not the disabled pill is skipped, so a round trip
  // proves nothing.
  key(h.select, "ArrowRight");

  assert.equal(
    activeDeviceOf(row), "cpu",
    "traversal reached a GPU pill that is disabled"
  );
});

test("a row with one device ignores Left and Right", () => {
  const h = picker();
  key(h.select, "ArrowDown");
  assert.equal(focusedName(h), "llada");

  key(h.select, "ArrowRight");

  assert.equal(focusedName(h), "llada");
  assert.deepEqual(h.switches, []);
});

// -- committing --

test("Enter asks for the focused model", () => {
  const h = picker();

  key(h.select, "ArrowDown");
  key(h.select, "Enter");

  assert.deepEqual(h.switches, [{ id: "llada", device: "cuda" }]);
});

test("a CPU switch is reachable with keys alone", () => {
  // The one that proves the tab-order change did not cost anything:
  // resident model, other device, no pointer involved anywhere.
  const h = picker();

  focusRow(h, "smollm3");
  key(h.select, "ArrowRight");
  key(h.select, "Enter");

  assert.deepEqual(h.switches, [{ id: "smollm3", device: "cpu" }]);
});

test("re-choosing exactly what is loaded does nothing", () => {
  // Mirrors the pointer, where the resident row's name area is inert
  // and only its other-device pill does anything.
  const h = picker();

  focusRow(h, "smollm3");
  key(h.select, "Enter");

  assert.deepEqual(h.switches, []);
  assert.equal(
    h.list.hidden, true, "it should close rather than sit open"
  );
});

test("Escape closes without switching", () => {
  const h = picker();
  key(h.select, "ArrowDown");

  key(h.select, "Escape");

  assert.equal(h.list.hidden, true);
  assert.deepEqual(h.switches, []);
});

test("Tab closes the list behind it", () => {
  const h = picker();
  key(h.select, "ArrowDown");

  key(h.select, "Tab");

  assert.equal(h.list.hidden, true);
});

test("reopening starts from the resident model again", () => {
  const h = picker();

  key(h.select, "ArrowDown");
  key(h.select, "ArrowDown");
  assert.equal(focusedName(h), "diffusiongemma");
  key(h.select, "Escape");
  key(h.select, "ArrowDown");

  assert.equal(focusedName(h), "llada");
});

// -- the confirm popover keeps its own keys --
//
// It is a child of `#model-select`, so its keys bubble to the picker's
// handler. That handler called preventDefault on Enter, which cancels
// the click a browser synthesises on the focused button, so Confirm
// and Cancel could be reached by Tab and then not pressed at all.
//
// The stub does not synthesise that click, so these assert the
// mechanism rather than the outcome: the picker must leave the event
// alone and must not act on it.

function openConfirm(h, id, device) {
  h.page.context.openSwitchConfirm(id, device);
  const box = h.page.context.switchConfirmEl;
  assert.ok(box, "the confirm popover was not built");
  return box;
}

test("Enter inside the confirm is left for the button", () => {
  const h = picker();
  const box = openConfirm(h, "llada", "cuda");
  const yes = box.querySelector(".switch-confirm-yes");
  assert.ok(yes);

  let prevented = false;
  h.select.dispatch("keydown", {
    key: "Enter",
    target: yes,
    preventDefault() { prevented = true; },
  });

  assert.equal(
    prevented, false,
    "the picker cancelled the click the button was about to get"
  );
});

test("and the picker does not act on it either", () => {
  // Doing both would be its own bug: pressing Confirm would also
  // toggle the list behind the popover.
  const h = picker();
  const box = openConfirm(h, "llada", "cuda");
  const before = h.list.hidden;

  h.select.dispatch("keydown", {
    key: "Enter",
    target: box.querySelector(".switch-confirm-no"),
    preventDefault() {},
  });

  assert.equal(h.list.hidden, before);
  assert.deepEqual(h.switches, []);
});

test("Escape inside the confirm closes the confirm", () => {
  // Not the list underneath, which is what the unguarded handler did.
  const h = picker();
  const box = openConfirm(h, "llada", "cuda");

  h.select.dispatch("keydown", {
    key: "Escape",
    target: box.querySelector(".switch-confirm-yes"),
    preventDefault() {},
  });

  assert.equal(
    h.page.context.switchConfirmEl, null,
    "the confirm popover survived Escape"
  );
});

test("keys on the picker itself still work with a confirm open", () => {
  // The guard is scoped to the popover's own subtree, not to "a
  // confirm exists somewhere".
  const h = picker();
  openConfirm(h, "llada", "cuda");

  key(h.select, "ArrowDown");

  assert.equal(h.list.hidden, false);
});

// -- and saying so --

test("the picker announces expansion and its active row", () => {
  // Only the transitions. The resting values are attributes on
  // index.html, which this stub never reads, so they are asserted in
  // tests/web/test_keyboard_reach.py instead.
  const h = picker();

  key(h.select, "ArrowDown");

  assert.equal(h.select.getAttribute("aria-expanded"), "true");
  assert.equal(
    h.select.getAttribute("aria-activedescendant"),
    focusedRow(h).id
  );

  key(h.select, "Escape");

  assert.equal(h.select.getAttribute("aria-expanded"), "false");
  assert.equal(
    h.select.getAttribute("aria-activedescendant"), null
  );
});

test("the travelling highlight follows the keyboard here too", () => {
  // One implementation shared with the dropdown, so the two lists
  // cannot drift into looking different.
  const h = picker();
  assert.equal(h.list.classes.has("has-cursor"), false);

  key(h.select, "ArrowDown");
  assert.equal(h.list.classes.has("has-cursor"), true);
  assert.notEqual(h.list.style["--cursor-top"], undefined);

  key(h.select, "Escape");
  assert.equal(h.list.classes.has("has-cursor"), false);
});

test("the headroom popover follows the keyboard", () => {
  // Otherwise it is mouse-only, and a keyboard user picks a model
  // with no idea whether it fits in the VRAM that is free.
  const h = picker();

  key(h.select, "ArrowDown");

  const row = focusedRow(h);
  const info = row.querySelector(".option-info");
  if (info) {
    assert.equal(info.classes.has("is-visible"), true);
  }
});

test("a disabled picker ignores the keyboard", () => {
  const h = picker();
  h.page.context.setModelSelectDisabled(true);

  key(h.select, "ArrowDown");

  assert.equal(h.list.hidden, true);
});
