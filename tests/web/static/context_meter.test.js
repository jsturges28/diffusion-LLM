// Exact next-inference context meter.
//
// Strategy: drive the pure bounded snapshot builder with legacy and
// structured count shapes, then mount the real controller against the
// shared DOM stub. Passing proves segment math, omission/error states,
// dialog details and focus return share one exact interpretation.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function harness() {
  const page = loadPage({ scripts: ["context_meter.js"] });
  const meter = page.context.contextMeterCreate();
  meter.wire();
  meter.configure({
    modelId: "llada",
    modelDisplay: "LLaDA",
    inputMode: "chat",
    device: "cuda",
    contextLength: 8192,
  });
  return { page, meter };
}

function structured(overrides) {
  return Object.assign({
    count: 1200,
    outputReserve: 256,
    contextLength: 8192,
    contextPack: {
      prompt_token_count: 1200,
      output_reserve: 256,
      effective_total_budget: 4096,
      requested_total_budget: 8192,
      omitted_turn_count: 4,
      first_included_index: 4,
      included_turn_ids: ["turn-5", "turn-6", "turn-7"],
    },
    truncated: false,
  }, overrides || {});
}

test("structured math separates prompt output and remaining", () => {
  const page = loadPage({ scripts: ["context_meter.js"] });
  const state = page.context.contextMeterReadyState(
    structured()
  );

  assert.deepEqual(host(state), {
    phase: "ready",
    structured: true,
    promptTokens: 1200,
    outputTokens: 256,
    remainingTokens: 2640,
    effectiveTokens: 4096,
    requestedTokens: 8192,
    includedTurns: 3,
    omittedTurns: 4,
    firstIncludedTurn: 5,
    promptEnd: 29.296875,
    outputEnd: 35.546875,
    usedPercent: 36,
    over: false,
    truncated: false,
    errorMessage: "",
  });
  assert.equal(Object.isFrozen(state), true);
});

test("legacy math uses the loaded checkpoint window", () => {
  const page = loadPage({ scripts: ["context_meter.js"] });
  const state = page.context.contextMeterReadyState({
    count: 90,
    outputReserve: 20,
    contextPack: null,
    contextLength: 100,
    truncated: false,
  });

  assert.equal(state.structured, false);
  assert.equal(state.remainingTokens, 0);
  assert.equal(state.over, true);
  assert.equal(state.usedPercent, 100);
});

test("malformed or disagreeing count packets fail loudly", () => {
  const page = loadPage({ scripts: ["context_meter.js"] });
  const api = page.context;

  assert.throws(
    () => api.contextMeterReadyState(
      structured({ count: 1199 })
    ),
    /disagrees/
  );
  assert.throws(
    () => api.contextMeterReadyState({
      count: -1,
      outputReserve: 1,
      contextLength: 10,
    }),
    /integer bound/
  );
  assert.throws(
    () => api.contextMeterReadyState({
      count: "10",
      outputReserve: 1,
      contextLength: 100,
      truncated: false,
    }),
    /integer bound/
  );
  assert.throws(
    () => api.contextMeterReadyState({
      count: 10,
      outputReserve: 1,
      contextLength: 100,
      truncated: "false",
    }),
    /boolean/
  );
  const tooMany = structured();
  tooMany.contextPack.included_turn_ids =
    Array.from({ length: 201 }, (_, index) => "turn-" + index);
  assert.throws(
    () => api.contextMeterReadyState(tooMany),
    /exceed/
  );
  const missing = structured();
  delete missing.contextPack.omitted_turn_count;
  assert.throws(
    () => api.contextMeterReadyState(missing),
    /missing/
  );
});

test("ready state paints the ring and exact dialog", () => {
  const h = harness();
  const button = h.page.registry.get("btn-context-meter");
  const omitted = h.page.registry.get("context-meter-omitted");

  h.meter.ready(structured());

  assert.equal(button.disabled, false);
  assert.equal(
    button.style.getPropertyValue("--context-prompt-end"),
    "29.30%"
  );
  assert.equal(
    button.style.getPropertyValue("--context-output-end"),
    "35.55%"
  );
  assert.equal(
    h.page.registry.get("context-meter-value").textContent,
    "36%"
  );
  assert.equal(omitted.hidden, false);
  assert.match(button.getAttribute("aria-label"), /4 earlier turns/);

  button.click();
  assert.equal(
    h.page.registry.get("modal-context-meter").open,
    true
  );
  assert.equal(
    h.page.registry.get("context-meter-model").textContent,
    "LLaDA (llada)"
  );
  assert.equal(
    h.page.registry.get("context-meter-remaining").textContent,
    "2,640"
  );
  assert.equal(
    h.page.registry.get("context-meter-first").textContent,
    "Turn 5"
  );
  assert.match(
    h.page.registry.get("context-meter-limit-note").textContent,
    /checkpoint window lowered/
  );
});

test("pending unavailable and error states never show old numbers",
  () => {
  const h = harness();
  const button = h.page.registry.get("btn-context-meter");
  const value = h.page.registry.get("context-meter-value");
  h.meter.ready(structured());

  h.meter.pending();
  assert.equal(button.disabled, false);
  assert.equal(button.getAttribute("aria-disabled"), "true");
  assert.equal(value.textContent, "\u2026");
  assert.equal(
    button.style.getPropertyValue("--context-output-end"),
    "0%"
  );

  h.meter.unavailable();
  assert.equal(value.textContent, "--");
  assert.equal(button.disabled, false);
  assert.equal(button.getAttribute("aria-disabled"), "true");

  h.meter.error("The pending turn exceeds its context budget.");
  assert.equal(value.textContent, "!");
  assert.equal(button.disabled, false);
  button.click();
  assert.equal(
    h.page.registry.get("context-meter-detail-status").textContent,
    "The pending turn exceeds its context budget."
  );
});

test("truncated legacy counts are labelled as lower bounds", () => {
  const h = harness();
  h.meter.ready({
    count: 100000,
    outputReserve: 1,
    contextPack: null,
    truncated: true,
  });
  const button = h.page.registry.get("btn-context-meter");

  assert.equal(
    button.getAttribute("aria-label"),
    "Context count exceeded its safety limit. Open details."
  );
  button.click();
  assert.equal(
    h.page.registry.get("context-meter-prompt").textContent,
    "At least 100,000"
  );
  assert.equal(
    h.page.registry.get("context-meter-remaining").textContent,
    "Not available"
  );
});

test("closing the native dialog returns focus to the meter", () => {
  const h = harness();
  const button = h.page.registry.get("btn-context-meter");
  const dialog = h.page.registry.get("modal-context-meter");
  h.meter.ready(structured());
  button.click();

  h.page.registry.get("btn-context-meter-close").click();

  assert.equal(dialog.open, false);
  assert.equal(button.getAttribute("aria-expanded"), "false");
  assert.equal(button.focused, true);

  button.click();
  dialog.dispatch("click", { target: dialog });
  assert.equal(dialog.open, false);
  assert.equal(button.focused, true);

  button.click();
  h.meter.unavailable();
  assert.equal(dialog.open, false);
  assert.equal(button.focused, true);
  assert.equal(button.disabled, false);
  assert.equal(button.getAttribute("aria-disabled"), "true");
});

test("model input and device labels cover every shipped family", () => {
  const page = loadPage({ scripts: ["context_meter.js"] });
  const meter = page.context.contextMeterCreate();
  meter.wire();
  const cases = [
    ["llada", "LLaDA", "chat", "cuda", "Chat", "GPU"],
    [
      "diffusiongemma",
      "DiffusionGemma",
      "chat",
      "cuda",
      "Chat",
      "GPU",
    ],
    ["smollm3", "SmolLM3", "chat", "cpu", "Chat", "CPU"],
    [
      "mamba3",
      "Mamba-3",
      "completion",
      "cpu",
      "Completion",
      "CPU",
    ],
  ];

  for (const item of cases) {
    meter.configure({
      modelId: item[0],
      modelDisplay: item[1],
      inputMode: item[2],
      device: item[3],
      contextLength: 4096,
    });
    meter.ready({
      count: 20,
      outputReserve: 10,
      contextPack: null,
      truncated: false,
    });
    assert.equal(
      page.registry.get("context-meter-mode").textContent,
      item[4]
    );
    assert.equal(
      page.registry.get("context-meter-device").textContent,
      item[5]
    );
  }
});
