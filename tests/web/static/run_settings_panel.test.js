// Reusable mounted Run settings disclosures.
//
// Strategy: mount real factory instances into the shared DOM stub,
// configure them from registry-shaped schemas, and drive only their
// controls and public reads. Passing proves instances keep unique
// DOM, state, persistence, validation, reset and cleanup boundaries.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "custom_select.js",
  "run_settings_core.js",
  "run_settings_panel.js",
];
const SCHEMA_ID = "4".repeat(64);

const MODEL = {
  id: "test-model",
  display_name: "Test Model",
  capabilities: {
    input_mode: "chat",
  },
  generation_schema_ids: {
    cpu: SCHEMA_ID,
    cuda: SCHEMA_ID,
  },
  param_specs: [
    {
      name: "gen_length",
      label: "Gen Length",
      type: "int",
      default: 8,
      group: "output",
      prominence: "primary",
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
      group: "output",
      recommended: [2, 8],
      experimental: [1, 16],
    },
    {
      name: "steps",
      label: "Steps",
      type: "int",
      default: 8,
      group: "output",
      prominence: "primary",
      recommended: [2, 32],
      experimental: [1, 64],
    },
    {
      name: "temperature",
      label: "Temperature",
      type: "float",
      default: 0.7,
      group: "sampling",
      prominence: "primary",
      recommended: [0, 1],
      experimental: [0, 2],
      step: 0.1,
    },
    {
      name: "strategy",
      label: "Strategy",
      type: "select",
      default: "low_confidence",
      group: "sampling",
      options: ["low_confidence", "random"],
    },
    {
      name: "thinking",
      label: "Thinking",
      type: "bool",
      default: false,
      group: "features",
    },
    {
      name: "watermark",
      label: "KGW Watermark",
      type: "bool",
      default: false,
      group: "signals",
      experimental_only: true,
    },
  ],
};

const OTHER_MODEL = {
  id: "other-model",
  display_name: "Other Model",
  capabilities: {
    input_mode: "completion",
  },
  generation_schema_ids: {
    cpu: SCHEMA_ID,
    cuda: SCHEMA_ID,
  },
  param_specs: [
    {
      name: "max_new_tokens",
      label: "Max Tokens",
      type: "int",
      default: 32,
      group: "output",
      prominence: "primary",
      recommended: [1, 64],
      experimental: [1, 128],
    },
  ],
};

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function configure(panel, options) {
  const settings = options || {};
  const model = settings.model || MODEL;
  panel.configure({
    model,
    models: [MODEL, OTHER_MODEL],
    modelId: model.id,
    modelDisplay: model.display_name,
    device: settings.device || "cpu",
    inputMode: model.capabilities.input_mode,
    seed: settings.seed,
  });
}

function mountPanel(page, prefix, options) {
  const mount = page.document.createElement("div");
  page.document.body.appendChild(mount);
  const panel = page.context.runSettingsPanelCreate(
    Object.assign({
      idPrefix: prefix,
      mount,
    }, options || {})
  );
  panel.wire();
  configure(panel);
  return { panel, mount, root: panel.root() };
}

function pageHarness() {
  return loadPage({ scripts: SCRIPTS });
}

function byId(harness, prefix, name) {
  return harness.root.querySelector("#" + prefix + name);
}

function input(harness, prefix, name) {
  return byId(harness, prefix, "param-" + name);
}

test("factory boundary requires one uniquely prefixed host", () => {
  const page = pageHarness();
  const mount = page.document.createElement("div");
  const root = page.document.createElement("details");

  assert.throws(
    () => page.context.runSettingsPanelCreate({}),
    /root or a mount/
  );
  assert.throws(
    () => page.context.runSettingsPanelCreate({
      root,
      mount,
    }),
    /not both/
  );
  assert.throws(
    () => page.context.runSettingsPanelCreate({ mount }),
    /unique idPrefix/
  );
  assert.throws(
    () => page.context.runSettingsPanelCreate({
      idPrefix: "bad-",
      mount,
      onParametersChanged: true,
    }),
    /must be a function/
  );
  assert.throws(
    () => page.context.runSettingsPanelCreate({
      idPrefix: "x".repeat(65) + "-",
      mount,
    }),
    /bounded and canonical/
  );
  assert.throws(
    () => page.context.runSettingsPanelCreate({
      idPrefix: "Bad prefix-",
      mount,
    }),
    /bounded and canonical/
  );
});

test("mounted disclosures use native details and prefixed labels",
  () => {
    const page = pageHarness();
    const first = mountPanel(page, "first-");
    const second = mountPanel(page, "second-");

    assert.equal(first.root.tag, "details");
    assert.equal(first.root.id, "first-run-settings");
    assert.equal(second.root.id, "second-run-settings");
    const summary = byId(
      first, "first-", "run-settings-summary"
    );
    assert.equal(summary.tag, "summary");
    assert.equal(
      summary.getAttribute("aria-controls"),
      "first-run-settings-body"
    );
    assert.equal(summary.getAttribute("role"), null);
    assert.equal(summary.getAttribute("tabindex"), null);
    const labels = first.root.querySelectorAll("label");
    assert.ok(labels.some(
      (label) =>
        label.getAttribute("for")
        === "first-param-gen_length"
    ));
    assert.notEqual(
      input(first, "first-", "gen_length").id,
      input(second, "second-", "gen_length").id
    );
  }
);

test("native disclosure mirrors user-expanded state", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "toggle-");
  const summary = byId(
    harness, "toggle-", "run-settings-summary"
  );

  assert.equal(summary.getAttribute("aria-expanded"), "false");
  harness.root.open = true;
  harness.root.dispatch("toggle");
  assert.equal(summary.getAttribute("aria-expanded"), "true");
  harness.root.open = false;
  harness.root.dispatch("toggle");
  assert.equal(summary.getAttribute("aria-expanded"), "false");
});

test("parameter help is reachable by keyboard and touch", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "help-");
  const info = input(
    harness, "help-", "gen_length"
  ).closest(".param-group").querySelector(".info-icon");
  const tooltip = info.querySelector(".tooltip");

  assert.equal(info.tabIndex, 0);
  assert.equal(info.getAttribute("role"), "button");
  assert.equal(info.getAttribute("aria-expanded"), "false");
  assert.equal(tooltip.getAttribute("role"), "tooltip");
  assert.equal(
    info.getAttribute("aria-describedby"), tooltip.id
  );

  info.dispatch("keydown", {
    key: "Enter",
    preventDefault() {},
  });
  assert.equal(info.classList.contains("is-open"), true);
  assert.equal(info.getAttribute("aria-expanded"), "true");
  info.dispatch("keydown", {
    key: "Escape",
    preventDefault() {},
  });
  assert.equal(info.classList.contains("is-open"), false);
});

test("two seeded panels keep independent values", () => {
  const page = pageHarness();
  const first = mountPanel(page, "first-");
  const second = mountPanel(page, "second-");

  first.panel.apply({
    modelId: MODEL.id,
    experimental: false,
    parameters: {
      gen_length: 8,
      block_length: 4,
      steps: 8,
      temperature: 0.9,
      strategy: "random",
      thinking: true,
    },
  });
  second.panel.apply({
    modelId: MODEL.id,
    experimental: false,
    parameters: {
      gen_length: 4,
      block_length: 4,
      steps: 8,
      temperature: 0.4,
      strategy: "low_confidence",
      thinking: false,
    },
  });

  input(first, "first-", "temperature").value = "0.8";
  input(first, "first-", "temperature").dispatch("input");

  assert.equal(
    first.panel.parameterValues().temperature,
    0.8
  );
  assert.equal(
    second.panel.parameterValues().temperature,
    0.4
  );
});

test("configure seed is copied into the initial snapshot", () => {
  const page = pageHarness();
  const mount = page.document.createElement("div");
  const panel = page.context.runSettingsPanelCreate({
    idPrefix: "seed-",
    mount,
  });
  const seed = {
    modelId: MODEL.id,
    experimental: true,
    parameters: {
      gen_length: 12,
      block_length: 4,
      steps: 12,
      temperature: 1.2,
      strategy: "random",
      thinking: true,
      watermark: true,
    },
  };
  panel.wire();
  configure(panel, { seed });
  seed.parameters.temperature = 0.1;

  const snapshot = panel.snapshot();
  assert.equal(snapshot.parameters.temperature, 1.2);
  assert.equal(snapshot.parameters.watermark, true);
  assert.equal(snapshot.experimental, true);
  assert.equal(
    mount.querySelector("#seed-param-temperature").max,
    2
  );
  assert.equal(Object.isFrozen(snapshot), true);
});

test("action seeds preserve invalid raw values for confirmation",
  () => {
  const page = pageHarness();
  const mount = page.document.createElement("div");
  const panel = page.context.runSettingsPanelCreate({
    idPrefix: "invalid-seed-",
    mount,
  });
  panel.wire();
  configure(panel, {
    seed: {
      modelId: MODEL.id,
      experimental: false,
      parameters: {
        gen_length: "6",
        block_length: "4",
        steps: "8",
        temperature: "",
      },
    },
  });

  assert.equal(
    mount.querySelector("#invalid-seed-param-gen_length").value,
    "6"
  );
  assert.equal(
    mount.querySelector("#invalid-seed-param-temperature").value,
    ""
  );
  assert.equal(panel.validation().valid, false);
  assert.equal(
    panel.formState().parameters.temperature,
    ""
  );
  assert.equal(Object.isFrozen(panel.formState()), true);
});

test("Reset restores device defaults and recommended mode", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "reset-");
  const experimental = byId(
    harness, "reset-", "toggle-experimental"
  );
  const reset = byId(
    harness, "reset-", "btn-param-defaults"
  );

  experimental.checked = true;
  experimental.dispatch("change");
  input(harness, "reset-", "gen_length").value = "12";
  input(harness, "reset-", "gen_length").dispatch("input");
  assert.equal(reset.disabled, false);

  reset.click();

  assert.equal(experimental.checked, false);
  assert.equal(
    input(harness, "reset-", "gen_length").value,
    "4"
  );
  assert.equal(reset.disabled, true);
});

test("invalid collapsed settings reveal and focus first control",
  () => {
    const page = pageHarness();
    const harness = mountPanel(page, "invalid-");
    const length = input(
      harness, "invalid-", "gen_length"
    );
    const summary = byId(
      harness, "invalid-", "run-settings-summary"
    );
    const status = byId(
      harness, "invalid-", "validation-hint"
    );

    harness.root.open = false;
    length.value = "6";
    length.dispatch("input");

    assert.equal(harness.root.open, true);
    assert.equal(summary.getAttribute("aria-expanded"), "true");
    assert.equal(length.focused, true);
    assert.equal(length.classList.contains("input-warn"), true);
    assert.equal(length.getAttribute("aria-invalid"), "true");
    assert.equal(
      length.getAttribute("aria-describedby"),
      "invalid-validation-hint"
    );
    assert.equal(status.hidden, false);
    assert.equal(status.getAttribute("role"), "status");
    assert.equal(
      harness.panel.validation().message,
      "Gen Length (6) must be divisible by Block Length (4)."
    );
  }
);

test("validation status describes every invalid control", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "errors-");
  const length = input(harness, "errors-", "gen_length");
  const temperature = input(
    harness, "errors-", "temperature"
  );
  const status = byId(
    harness, "errors-", "validation-hint"
  );

  length.value = "";
  length.dispatch("input");
  temperature.value = "";
  temperature.dispatch("input");

  assert.match(
    status.textContent,
    /Gen Length is empty or invalid\./
  );
  assert.match(
    status.textContent,
    /Temperature is empty or invalid\./
  );
  assert.equal(
    length.getAttribute("aria-describedby"), status.id
  );
  assert.equal(
    temperature.getAttribute("aria-describedby"), status.id
  );
});

test("Experimental reveals controls and updates range help", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "exp-");
  const experimental = byId(
    harness, "exp-", "toggle-experimental"
  );
  const watermark = input(
    harness, "exp-", "watermark"
  );
  const signalGroup = watermark.closest(
    ".run-settings-mode-group"
  );
  const length = input(harness, "exp-", "gen_length");
  const tooltip = length.closest(
    ".param-group"
  ).querySelector(".tooltip");

  assert.equal(watermark.closest(".mode-toggle").hidden, true);
  assert.equal(signalGroup.hidden, true);
  assert.equal(length.max, 8);
  assert.equal(
    tooltip.querySelector(".tooltip-desc").textContent,
    "Output length."
  );

  experimental.checked = true;
  experimental.dispatch("change");

  assert.equal(watermark.closest(".mode-toggle").hidden, false);
  assert.equal(signalGroup.hidden, false);
  assert.equal(length.max, 16);
  assert.equal(
    harness.panel.parameterValues().watermark,
    false
  );
});

test("primary chips follow live values and invalid status", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "chips-");
  const chips = byId(
    harness, "chips-", "run-settings-summary-chips"
  );

  input(harness, "chips-", "temperature").value = "0.9";
  input(harness, "chips-", "temperature").dispatch("input");
  const temperature = chips.children.find(
    (chip) =>
      chip.getAttribute("data-param-name") === "temperature"
  );
  assert.equal(
    temperature.querySelector(
      ".run-settings-chip-value"
    ).textContent,
    "0.9"
  );
  assert.equal(
    temperature.getAttribute("aria-label"),
    "Temperature 0.9"
  );
  assert.equal(temperature.getAttribute("role"), "group");

  input(harness, "chips-", "gen_length").value = "8";
  input(harness, "chips-", "gen_length").dispatch("input");
  input(harness, "chips-", "steps").value = "7";
  input(harness, "chips-", "steps").dispatch("input");
  const steps = chips.children.find(
    (chip) =>
      chip.getAttribute("data-param-name") === "steps"
  );
  assert.equal(steps.classList.contains("is-invalid"), true);
});

test("frozen state keeps disclosure and help readable", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "disabled-");
  const experimental = byId(
    harness, "disabled-", "toggle-experimental"
  );
  const reset = byId(
    harness, "disabled-", "btn-param-defaults"
  );
  const summary = byId(
    harness, "disabled-", "run-settings-summary"
  );
  const strategy = input(harness, "disabled-", "strategy");
  const info = input(
    harness, "disabled-", "gen_length"
  ).closest(".param-group").querySelector(".info-icon");

  harness.panel.setDisabled(true);

  assert.equal(harness.root.getAttribute("aria-disabled"), null);
  assert.equal(summary.getAttribute("aria-disabled"), null);
  assert.equal(summary.getAttribute("tabindex"), null);
  assert.equal(info.getAttribute("aria-disabled"), null);
  assert.equal(info.tabIndex, 0);
  assert.equal(experimental.disabled, true);
  assert.equal(reset.disabled, true);
  assert.equal(strategy.getAttribute("aria-disabled"), "true");
  assert.equal(strategy.tabIndex, -1);
  assert.equal(
    input(harness, "disabled-", "gen_length").disabled,
    true
  );
  harness.root.open = true;
  harness.root.dispatch("toggle");
  assert.equal(summary.getAttribute("aria-expanded"), "true");
  info.dispatch("click", {
    preventDefault() {},
    stopPropagation() {},
  });
  assert.equal(info.classList.contains("is-open"), true);

  harness.panel.setDisabled(false);
  assert.equal(summary.getAttribute("tabindex"), null);
  assert.equal(info.tabIndex, 0);
});

test("a panel without persistence never writes Draft storage", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "local-");

  input(harness, "local-", "temperature").value = "0.8";
  input(harness, "local-", "temperature").dispatch("input");
  harness.panel.reset();

  assert.equal(page.sandbox.sessionStorage.size, 0);
});

test("optional persistence reads and writes per model", () => {
  const page = pageHarness();
  const writes = [];
  const persisted = {
    [MODEL.id]: {
      experimental: false,
      params: {
        gen_length: "8",
        block_length: "4",
        steps: "8",
        temperature: "0.8",
        strategy: "random",
        thinking: true,
      },
    },
    [OTHER_MODEL.id]: {
      experimental: false,
      params: {
        max_new_tokens: "48",
      },
    },
  };
  const harness = mountPanel(page, "persist-", {
    readPersistedState(id) {
      return persisted[id] || null;
    },
    writePersistedState(id, state) {
      writes.push({ id, state: host(state) });
    },
  });

  assert.equal(
    input(harness, "persist-", "temperature").value,
    "0.8"
  );
  input(harness, "persist-", "temperature").value = "0.6";
  input(harness, "persist-", "temperature").dispatch("input");
  assert.equal(writes.length, 1);
  assert.equal(writes[0].id, MODEL.id);
  assert.equal(writes[0].state.params.temperature, "0.6");

  configure(harness.panel, { model: OTHER_MODEL });
  assert.equal(
    input(harness, "persist-", "max_new_tokens").value,
    "48"
  );
});

test("persisted Draft values clamp to active device bounds", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "clamped-", {
    readPersistedState() {
      return {
        experimental: false,
        params: {
          gen_length: "99",
          block_length: "4",
          steps: "8",
          temperature: "9",
        },
      };
    },
  });

  assert.equal(
    input(harness, "clamped-", "gen_length").value,
    "8"
  );
  assert.equal(
    input(harness, "clamped-", "temperature").value,
    "1"
  );
  assert.equal(harness.panel.validation().valid, true);
});

test("programmatic apply neither persists nor reports a user edit",
  () => {
    const page = pageHarness();
    let changes = 0;
    let writes = 0;
    const harness = mountPanel(page, "apply-", {
      onParametersChanged() {
        changes += 1;
      },
      writePersistedState() {
        writes += 1;
      },
    });

    harness.panel.apply({
      modelId: MODEL.id,
      experimental: false,
      parameters: {
        temperature: 0.9,
      },
    });

    assert.equal(changes, 0);
    assert.equal(writes, 0);
    assert.equal(
      harness.panel.parameterValues().temperature,
      0.9
    );
  }
);

test("a seed for another model is rejected", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "foreign-");

  assert.throws(
    () => harness.panel.apply({
      modelId: OTHER_MODEL.id,
      experimental: false,
      parameters: {},
    }),
    /another model/
  );
});

test("model reconfigure replaces controls and context", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "model-");
  const experimental = byId(
    harness, "model-", "toggle-experimental"
  );
  experimental.checked = true;
  experimental.dispatch("change");

  configure(harness.panel, { model: OTHER_MODEL });

  assert.equal(experimental.checked, false);
  assert.equal(
    input(harness, "model-", "gen_length"),
    null
  );
  assert.equal(
    input(harness, "model-", "max_new_tokens").value,
    "32"
  );
  assert.equal(harness.panel.snapshot().modelId, OTHER_MODEL.id);
  assert.equal(
    harness.panel.snapshot().inputMode,
    "completion"
  );
});

test("compact mode is only a safe styling seam", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "compact-", {
    compact: true,
  });

  assert.equal(
    harness.root.classList.contains("run-settings-compact"),
    true
  );
  assert.equal(harness.root.open, false);
});

test("destroy removes generated DOM and listeners", () => {
  const page = pageHarness();
  const harness = mountPanel(page, "gone-");
  const root = harness.root;
  const calls = [
    () => harness.panel.wire(),
    () => harness.panel.configure({}),
    () => harness.panel.apply({}),
    () => harness.panel.setDisabled(true),
    () => harness.panel.reset(),
    () => harness.panel.persist(),
    () => harness.panel.revealFirstInvalid(),
    () => harness.panel.parameterValues(),
    () => harness.panel.parameterDefaults(),
    () => harness.panel.experimental(),
    () => harness.panel.thinking(),
    () => harness.panel.outputBudget(),
    () => harness.panel.validation(),
    () => harness.panel.snapshot(),
    () => harness.panel.formState(),
    () => harness.panel.root(),
  ];

  harness.panel.destroy();
  harness.panel.destroy();

  assert.equal(root.parentNode, null);
  assert.deepEqual(root.listeners.toggle || [], []);
  for (const call of calls) {
    assert.throws(call, /destroyed/);
  }
});
