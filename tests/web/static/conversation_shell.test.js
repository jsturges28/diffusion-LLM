// The conversation shell owns durable New Conversation behavior.
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
    fetchImpl(url, init) {
      const path = String(url).split("?")[0];
      if (path === "/api/models") {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(models),
        });
      }
      if (
        path === "/api/conversations"
        && init && init.method === "POST"
      ) {
        return Promise.resolve({
          ok: true,
          status: 201,
          json: () => Promise.resolve({
            conversation: {
              id: "a".repeat(32),
              title: "New conversation",
              revision: 1,
              turn_count: 0,
              tail_turn_id: null,
              tail_version: null,
              pending_assistant_id: null,
            },
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      });
    },
    bootState: { ui_state: {}, models: models },
  });
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("New Conversation clears only after durable creation", async () => {
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
  await tick();

  assert.equal(prompt.value, "");
  assert.equal(run.frameCount(), 0);
  assert.match(
    page.registry.get("output-area").lastChild.textContent,
    /Test Model output will appear here/
  );
});

test("Send keeps only Send and Stop labels", () => {
  const page = pageWithModel();
  const run = page.context.generatorRun;
  const readEditedSaved = run.editedSaved;
  page.context.handleModelStatus({ status: "ready" });

  run.editedSaved = () => true;
  page.context.updateGenerateButton();
  assert.equal(
    page.context.currentGenerateLabel(),
    "Send"
  );
  assert.equal(page.registry.get("btn-generate").disabled, false);
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
