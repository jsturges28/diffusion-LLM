// The generator model panel, driven without app.js.
//
// Strategy: compose the factory with recording callbacks, configure
// it from a registry payload, and interact only through its public
// reads and owned DOM controls. A second harness composes it with the
// prompt controller over their shared compatibility record.
//
// Passing proves registry and DOM state are closure-owned, schema
// values keep their wire shape, validation copy and defaults survive
// extraction, switching stops at the page callback, and each form
// controller preserves the other controller's draft members.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "custom_select.js",
  "model_client.js",
  "generator_model_panel.js",
];
const COMPOSED_FILES = [
  "custom_select.js",
  "persist.js",
  "model_client.js",
  "generator_composer.js",
  "generator_model_panel.js",
];
const DRAFT_KEY = "diffusion_param_state";

const MODEL = {
  id: "test-model",
  display_name: "Test Model",
  min_vram_gib: 6,
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    supported_devices: ["cuda", "cpu"],
    input_mode: "chat",
    unresolved_char: "?",
  },
  param_specs: [
    {
      name: "gen_length",
      label: "Gen Length",
      type: "int",
      default: 8,
      recommended: [4, 16],
      experimental: [1, 32],
      overrides: {
        cpu: {
          default: 4,
          recommended: [2, 8],
          experimental: [1, 16],
        },
      },
      help: "Output length.",
    },
    {
      name: "block_length",
      label: "Block Length",
      type: "int",
      default: 4,
      recommended: [2, 8],
      experimental: [1, 16],
    },
    {
      name: "steps",
      label: "Steps",
      type: "int",
      default: 8,
      recommended: [2, 32],
      experimental: [1, 64],
    },
    {
      name: "temperature",
      label: "Temperature",
      type: "float",
      default: 0.7,
      recommended: [0, 1],
      experimental: [0, 2],
      step: 0.1,
    },
    {
      name: "thinking",
      label: "Thinking",
      type: "bool",
      default: false,
    },
    {
      name: "strategy",
      label: "Strategy",
      type: "select",
      default: "low_confidence",
      options: ["low_confidence", "random"],
    },
  ],
  status: "active",
};

const OTHER_MODEL = {
  id: "other",
  display_name: "Other Model",
  min_vram_gib: 4,
  capabilities: {
    family: "autoregressive",
    generation_shape: "append_only",
    supported_devices: ["cuda"],
  },
  param_specs: [],
  status: "idle",
};

function modelInfo() {
  return {
    models: [MODEL, OTHER_MODEL],
    active: MODEL.id,
    active_device: "cpu",
    active_tokenizer: {
      name: "test-tokenizer",
      model_vocab_size: 32000,
    },
    active_context_length: 4096,
    default: MODEL.id,
    gpu_name: "Test GPU",
  };
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function completeOptions(state) {
  return {
    onSwitchRequested(id, device) {
      state.switches.push({ id: id, device: device });
    },
    onValidationChanged(validation) {
      state.validations.push(host(validation));
    },
    onParametersChanged() {
      state.parameterChanges += 1;
    },
    readReducedMotion() {
      return state.reducedMotion;
    },
    readGpuTicker() {
      return state.gpuTicker;
    },
  };
}

function loadPanel(settings) {
  const config = settings || {};
  const page = loadPage({ scripts: SCRIPTS });
  if (config.draftState) {
    page.sandbox.sessionStorage.setItem(
      DRAFT_KEY, JSON.stringify(config.draftState)
    );
  }
  const state = {
    switches: [],
    validations: [],
    parameterChanges: 0,
    reducedMotion: false,
    gpuTicker: false,
  };
  const panel = page.context.generatorModelPanelCreate(
    completeOptions(state)
  );
  panel.wire();
  panel.configure(modelInfo());
  return { page, panel, state };
}

function input(harness, name) {
  const selector = "#param-" + name;
  return harness.page.registry
    .get("param-fields")
    .querySelector(selector)
    || harness.page.registry
      .get("mode-extra")
      .querySelector(selector);
}

test("every page callback is required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const state = {
    switches: [],
    validations: [],
    parameterChanges: 0,
    reducedMotion: false,
    gpuTicker: false,
  };
  const names = Object.keys(completeOptions(state));

  for (const name of names) {
    const options = completeOptions(state);
    delete options[name];
    assert.throws(
      () => page.context.generatorModelPanelCreate(options),
      new RegExp(name)
    );
  }
});

test("registry, picker and parameter references stay private", () => {
  const h = loadPanel({});

  for (const name of [
    "models",
    "activeModelId",
    "activeDevice",
    "gpuPresent",
    "paramInputs",
    "switchConfirmEl",
  ]) {
    assert.equal(h.page.context[name], undefined, name);
  }
  assert.equal(h.panel.activeModelId(), MODEL.id);
  assert.equal(h.panel.activeDevice(), "cpu");
  assert.equal(h.panel.activeContext(), 4096);
  assert.equal(
    h.panel.activeTokenizer().model_vocab_size, 32000
  );
  assert.equal(h.panel.capabilities().family, "diffusion");
});

test("schema controls preserve parameter values and budget reads",
  () => {
    const h = loadPanel({});

    input(h, "gen_length").value = "8";
    input(h, "temperature").value = "0.9";
    input(h, "thinking").checked = true;
    input(h, "strategy").value = "random";

    assert.deepEqual(host(h.panel.parameterValues()), {
      gen_length: 8,
      block_length: 4,
      steps: 8,
      temperature: 0.9,
      thinking: true,
      strategy: "random",
    });
    assert.equal(h.panel.thinking(), true);
    assert.equal(h.panel.outputBudget(), 8);
    assert.equal(h.panel.experimental(), false);
  }
);

test("validation keeps the divisibility messages", () => {
  const h = loadPanel({});
  const hint = h.page.registry.get("validation-hint");

  input(h, "gen_length").value = "6";
  input(h, "gen_length").dispatch("input");
  assert.equal(
    hint.textContent,
    "Gen Length (6) must be divisible by Block Length (4)."
  );
  assert.equal(h.panel.validation().valid, false);

  input(h, "gen_length").value = "8";
  input(h, "steps").value = "7";
  input(h, "steps").dispatch("input");
  assert.equal(
    hint.textContent,
    "Steps (7) must be divisible by num_blocks (2)."
  );

  input(h, "steps").value = "8";
  input(h, "temperature").value = "";
  input(h, "temperature").dispatch("input");
  assert.equal(
    hint.textContent,
    "Temperature is empty or invalid."
  );
});

test("Experimental and Defaults use device-aware bounds", () => {
  const h = loadPanel({});
  const experimental =
    h.page.registry.get("toggle-experimental");
  const defaults = h.page.registry.get("btn-param-defaults");

  assert.equal(input(h, "gen_length").value, "4");
  input(h, "gen_length").value = "12";
  experimental.checked = true;
  experimental.dispatch("change");
  assert.equal(input(h, "gen_length").max, 16);
  assert.equal(defaults.disabled, false);

  defaults.click();

  assert.equal(experimental.checked, false);
  assert.equal(input(h, "gen_length").value, "4");
  assert.equal(defaults.disabled, true);
});

test("switch confirmation calls only the execution callback", () => {
  const h = loadPanel({});
  const list = h.page.registry.get("model-select-list");
  const select = h.page.registry.get("model-select");
  const other = list.children.find(
    (row) => row.getAttribute("data-id") === OTHER_MODEL.id
  );
  const name = other.querySelector(".model-select-name");

  list.dispatch("click", { target: name });

  const confirm = select.querySelector(".switch-confirm");
  assert.ok(confirm);
  assert.equal(h.state.switches.length, 0);
  assert.match(
    confirm.querySelector(".switch-confirm-msg").textContent,
    /Other Model on GPU/
  );
  confirm.querySelector(".switch-confirm-yes").dispatch("click", {
    stopPropagation() {},
  });
  assert.deepEqual(h.state.switches, [{
    id: OTHER_MODEL.id,
    device: "cuda",
  }]);
});

test("panel and composer preserve each other's draft members", () => {
  const page = loadPage({ scripts: COMPOSED_FILES });
  page.sandbox.sessionStorage.setItem(
    DRAFT_KEY,
    JSON.stringify({
      [MODEL.id]: {
        prompt: "Keep this prompt",
        experimental: true,
        params: {
          gen_length: "12",
          block_length: "4",
          steps: "12",
          temperature: "0.8",
          thinking: true,
          strategy: "random",
        },
      },
    })
  );
  let composer = null;
  const state = {
    switches: [],
    validations: [],
    parameterChanges: 0,
    reducedMotion: false,
    gpuTicker: false,
  };
  const panelOptions = completeOptions(state);
  panelOptions.onParametersChanged = function () {
    composer.saveDraft();
    composer.parametersChanged();
  };
  const panel =
    page.context.generatorModelPanelCreate(panelOptions);
  composer = page.context.generatorComposerCreate({
    onSubmit() {},
    onDraftChanged() {
      panel.saveDraft();
    },
    reportStatus() {},
    readThinking: panel.thinking,
    readOutputBudget: panel.outputBudget,
    isCountReady() { return false; },
    sendCountPrompt() {},
  });
  panel.wire();
  composer.wire();
  composer.boot();
  panel.configure(modelInfo());
  composer.configure({
    modelId: panel.activeModelId(),
    inputMode: panel.capabilities().input_mode,
    contextLength: panel.activeContext(),
  });

  assert.equal(composer.value(), "Keep this prompt");
  input({ page: page }, "temperature").value = "0.6";
  input({ page: page }, "temperature").dispatch("input");
  let stored = JSON.parse(
    page.sandbox.sessionStorage.getItem(DRAFT_KEY)
  );
  assert.equal(stored[MODEL.id].prompt, "Keep this prompt");
  assert.equal(stored[MODEL.id].params.temperature, "0.6");

  page.registry.get("prompt-input").value = "Changed prompt";
  page.registry.get("prompt-input").dispatch("input");
  stored = JSON.parse(
    page.sandbox.sessionStorage.getItem(DRAFT_KEY)
  );
  assert.equal(stored[MODEL.id].prompt, "Changed prompt");
  assert.equal(stored[MODEL.id].experimental, true);
  assert.equal(stored[MODEL.id].params.temperature, "0.6");
});
