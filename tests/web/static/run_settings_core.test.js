// Pure Run settings schema semantics.
//
// Strategy: load only the classic-script factory, feed it one schema
// covering every ParamSpec shape, and compare host copies of results.
// Passing proves defaults, bounds, parsing, omission, validation,
// summaries and frozen launch snapshots do not depend on DOM state.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = ["run_settings_core.js"];

const SPECS = [
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
    overrides: {
      cpu: {
        default: 0,
      },
    },
  },
  {
    name: "strategy",
    label: "Strategy",
    type: "select",
    default: "low_confidence",
    group: "sampling",
    prominence: "primary",
    options: ["low_confidence", "random"],
  },
  {
    name: "capture",
    label: "Capture",
    type: "bool",
    default: true,
    group: "features",
    prominence: "primary",
  },
  {
    name: "watermark",
    label: "KGW Watermark",
    type: "bool",
    default: false,
    group: "signals",
    experimental_only: true,
  },
  {
    name: "watermark_delta",
    label: "Green Bias",
    type: "float",
    default: 2,
    group: "signals",
    recommended: [0, 5],
    experimental: [0, 10],
    experimental_only: true,
  },
];

function harness() {
  const page = loadPage({ scripts: SCRIPTS });
  return page.context.runSettingsCoreCreate();
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function rawValues(overrides) {
  return Object.assign({
    gen_length: "8",
    block_length: "4",
    steps: "8",
    temperature: "0.7",
    strategy: "low_confidence",
    capture: true,
    watermark: false,
    watermark_delta: "2",
  }, overrides || {});
}

test("device defaults honor overrides including zero", () => {
  const core = harness();

  assert.deepEqual(host(core.defaults(SPECS, "cpu")), {
    gen_length: 4,
    block_length: 4,
    steps: 8,
    temperature: 0,
    strategy: "low_confidence",
    capture: true,
    watermark: false,
    watermark_delta: 2,
  });
  assert.equal(core.defaultValue(SPECS[0], "cuda"), 8);
  assert.equal(Object.isFrozen(core.defaults(SPECS, "cpu")), true);
});

test("bounds select mode then device with base fallback", () => {
  const core = harness();

  assert.deepEqual(
    host(core.bounds(SPECS[0], "cpu", false)),
    { min: 2, max: 8 }
  );
  assert.deepEqual(
    host(core.bounds(SPECS[0], "cpu", true)),
    { min: 1, max: 16 }
  );
  assert.deepEqual(
    host(core.bounds(SPECS[3], "cpu", true)),
    { min: 0, max: 2 }
  );
  assert.equal(core.bounds(SPECS[4], "cpu", false), null);
});

test("wire values keep each declared parameter type", () => {
  const core = harness();
  const values = core.parameterValues(
    SPECS,
    rawValues({
      gen_length: "12.9",
      temperature: "0.95",
      strategy: "random",
      capture: false,
    }),
    false
  );

  assert.deepEqual(host(values), {
    gen_length: 12,
    block_length: 4,
    steps: 8,
    temperature: 0.95,
    strategy: "random",
    capture: false,
  });
});

test("experimental-only values and errors are omitted until enabled",
  () => {
    const core = harness();
    const raw = rawValues({
      watermark: true,
      watermark_delta: "99",
    });

    assert.equal(
      "watermark" in core.parameterValues(SPECS, raw, false),
      false
    );
    assert.equal(
      core.parameterValues(SPECS, raw, true).watermark,
      true
    );
    assert.equal(
      "watermark_delta"
        in core.parameterValues(SPECS, raw, false),
      false
    );
    assert.equal(core.validate({
      specs: SPECS,
      rawValues: raw,
      device: "cuda",
      experimental: false,
    }).valid, true);
    assert.equal(core.validate({
      specs: SPECS,
      rawValues: raw,
      device: "cuda",
      experimental: true,
    }).message, "Green Bias must be at most 10.");
    assert.equal(core.included(SPECS[6], false), false);
    assert.equal(core.included(SPECS[6], true), true);
  }
);

test("clamping uses active bounds and preserves invalid text",
  () => {
    const core = harness();

    assert.equal(
      core.clampRawValue(SPECS[0], "12", "cpu", false),
      "8"
    );
    assert.equal(
      core.clampRawValue(SPECS[0], "0", "cpu", true),
      "1"
    );
    assert.equal(
      core.clampRawValue(SPECS[0], "bad", "cpu", false),
      "bad"
    );
    assert.equal(
      core.clampRawValue(SPECS[4], "random", "cpu", false),
      "random"
    );
  }
);

test("numeric validation names empty and boundary failures", () => {
  const core = harness();
  const empty = core.validate({
    specs: SPECS,
    rawValues: rawValues({ temperature: "" }),
    device: "cuda",
    experimental: false,
  });
  const low = core.validate({
    specs: SPECS,
    rawValues: rawValues({ gen_length: "-2" }),
    device: "cuda",
    experimental: false,
  });
  const high = core.validate({
    specs: SPECS,
    rawValues: rawValues({ temperature: "1.1" }),
    device: "cuda",
    experimental: false,
  });

  assert.equal(empty.message, "Temperature is empty or invalid.");
  assert.deepEqual(host(empty.invalidNames), ["temperature"]);
  assert.equal(low.message, "Gen Length cannot be negative.");
  assert.equal(
    high.message,
    "Temperature must be at most 1."
  );
  assert.equal(Object.isFrozen(empty.invalidNames), true);
});

test("recommended and Experimental validation use distinct caps",
  () => {
    const core = harness();
    const raw = rawValues({
      gen_length: "20",
      steps: "10",
    });
    const recommended = core.validate({
      specs: SPECS,
      rawValues: raw,
      device: "cuda",
      experimental: false,
    });
    const experimental = core.validate({
      specs: SPECS,
      rawValues: raw,
      device: "cuda",
      experimental: true,
    });

    assert.equal(recommended.valid, false);
    assert.equal(
      recommended.message,
      "Gen Length must be at most 16."
    );
    assert.equal(experimental.valid, true);
  }
);

test("generation length must divide by block length", () => {
  const core = harness();
  const validation = core.validate({
    specs: SPECS,
    rawValues: rawValues({ gen_length: "6" }),
    device: "cuda",
    experimental: false,
  });

  assert.equal(
    validation.message,
    "Gen Length (6) must be divisible by Block Length (4)."
  );
  assert.deepEqual(
    host(validation.invalidNames),
    ["gen_length", "block_length"]
  );
});

test("steps must divide by the derived block count", () => {
  const core = harness();
  const validation = core.validate({
    specs: SPECS,
    rawValues: rawValues({ steps: "7" }),
    device: "cuda",
    experimental: false,
  });

  assert.equal(
    validation.message,
    "Steps (7) must be divisible by num_blocks (2)."
  );
  assert.deepEqual(host(validation.invalidNames), ["steps"]);
});

test("partial schemas do not acquire diffusion arithmetic", () => {
  const core = harness();
  const validation = core.validate({
    specs: [SPECS[0], SPECS[2]],
    rawValues: {
      gen_length: "6",
      steps: "7",
    },
    device: "cuda",
    experimental: false,
  });

  assert.equal(validation.valid, true);
  assert.deepEqual(host(validation.errors), []);
});

test("primary summary formats numbers, choices and booleans", () => {
  const core = harness();

  assert.equal(
    core.primarySummary(SPECS, rawValues()),
    "Gen Length 8, Steps 8, Temperature 0.7, "
      + "Strategy Low confidence, Capture On"
  );
  assert.equal(core.primarySummary([], {}), "Model defaults");
});

test("defaults comparison is device-aware and mode-aware", () => {
  const core = harness();
  const cpuDefaults = rawValues({
    gen_length: "4",
    temperature: "0",
  });

  assert.equal(core.valuesAtDefaults({
    specs: SPECS,
    rawValues: cpuDefaults,
    device: "cpu",
    experimental: false,
  }), true);
  assert.equal(core.valuesAtDefaults({
    specs: SPECS,
    rawValues: cpuDefaults,
    device: "cpu",
    experimental: true,
  }), false);
});

test("configuration snapshots are detached and frozen", () => {
  const core = harness();
  const raw = rawValues({ gen_length: "4" });
  const snapshot = core.configurationSnapshot({
    modelId: "test-model",
    modelDisplay: "Test Model",
    inputMode: "chat",
    specs: SPECS,
    rawValues: raw,
    device: "cpu",
    experimental: false,
  });

  raw.gen_length = "8";

  assert.equal(snapshot.parameters.gen_length, 4);
  assert.equal(Object.isFrozen(snapshot), true);
  assert.equal(Object.isFrozen(snapshot.parameters), true);
  assert.equal(snapshot.valid, true);
  assert.equal(snapshot.validationMessage, "");
  assert.match(snapshot.settingsSummary, /Gen Length 4/);
});

test("invalid snapshots retain exact launch-blocking status", () => {
  const core = harness();
  const snapshot = core.configurationSnapshot({
    modelId: "test-model",
    modelDisplay: "Test Model",
    inputMode: "completion",
    specs: SPECS,
    rawValues: rawValues({ steps: "7" }),
    device: "cuda",
    experimental: false,
  });

  assert.equal(snapshot.valid, false);
  assert.equal(
    snapshot.validationMessage,
    "Steps (7) must be divisible by num_blocks (2)."
  );
});

test("snapshot identity rejects missing model context", () => {
  const core = harness();
  const base = {
    modelId: "test-model",
    modelDisplay: "Test Model",
    inputMode: "chat",
    specs: SPECS,
    rawValues: rawValues(),
    device: "cuda",
    experimental: false,
  };

  assert.throws(
    () => core.configurationSnapshot(
      Object.assign({}, base, { modelId: null })
    ),
    /active model/
  );
  assert.throws(
    () => core.configurationSnapshot(
      Object.assign({}, base, { inputMode: "image" })
    ),
    /input mode/
  );
});

test("schema reads do not mutate registry objects", () => {
  const core = harness();
  const before = JSON.stringify(SPECS);

  core.defaults(SPECS, "cpu");
  core.validate({
    specs: SPECS,
    rawValues: rawValues(),
    device: "cpu",
    experimental: false,
  });
  core.configurationSnapshot({
    modelId: "test-model",
    modelDisplay: "Test Model",
    inputMode: "chat",
    specs: SPECS,
    rawValues: rawValues(),
    device: "cpu",
    experimental: false,
  });

  assert.equal(JSON.stringify(SPECS), before);
});
