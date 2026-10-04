// The conversation shell keeps the existing single-turn lifecycle.
//
// Strategy: load the complete generator page, seed an active run
// through its public controller, and press the shipped toolbar action.
// Passing proves New Conversation reaches the established fresh-run
// reset, Generate keeps only its Generate/Stop roles, and an in-flight
// run cannot be discarded from the toolbar.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const MODEL = {
  id: "test-model",
  display_name: "Test Model",
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    input_mode: "chat",
    supported_devices: ["cuda"],
    unresolved_char: "?",
  },
  param_specs: [],
  status: "active",
};

function pageWithModel() {
  const models = {
    models: [MODEL],
    active: MODEL.id,
    active_device: "cuda",
    active_tokenizer: { name: "test-tokenizer" },
    active_context_length: 4096,
    default: MODEL.id,
    gpu_name: "Test GPU",
  };
  return loadPage({
    bootState: { ui_state: {}, models: models },
  });
}

test("New Conversation clears the existing single turn", () => {
  const page = pageWithModel();
  const run = page.context.generatorRun;
  const prompt = page.registry.get("prompt-input");

  prompt.value = "A prompt to clear";
  run.begin(prompt.value, {});
  run.appendFrame({
    index: 0,
    text: "answer",
    tokens: [{ t: "answer", m: false, id: 1, c: 0.8 }],
    canvas_index: 0,
    elapsed: 0.1,
    revealed: [0],
  });
  assert.equal(run.frameCount(), 1);

  page.registry.get("btn-new-conversation").click();

  assert.equal(prompt.value, "");
  assert.equal(run.frameCount(), 0);
  assert.match(
    page.registry.get("output-area").lastChild.textContent,
    /Test Model output will appear here/
  );
});

test("Generate keeps only Generate and Stop labels", () => {
  const page = pageWithModel();
  const run = page.context.generatorRun;
  const readEditedSaved = run.editedSaved;

  run.editedSaved = () => true;
  page.context.updateGenerateButton();
  assert.equal(
    page.registry.get("btn-generate-label").textContent,
    "Generate"
  );
  assert.equal(page.registry.get("btn-generate").disabled, true);
  assert.equal(
    page.registry.get("btn-new-conversation").disabled,
    false
  );
  run.editedSaved = readEditedSaved;
});

test("New Conversation is unavailable during generation", () => {
  const page = pageWithModel();
  const prompt = page.registry.get("prompt-input");
  const button = page.registry.get("btn-new-conversation");

  prompt.value = "Keep this while running";
  page.context.setGenerating(true);
  assert.equal(button.disabled, true);
  button.click();
  assert.equal(prompt.value, "Keep this while running");

  page.context.setGenerating(false);
  assert.equal(button.disabled, false);
});
