// The conversation shell owns geometry, scrolling, and durable New
// Conversation behavior.
//
// Strategy: drive the focused classic controller in a VM, then load
// the complete generator page and press its shipped toolbar action.
// Passing proves only append/fork follow the tail, older pages keep
// their anchor, workspace ownership is exact, and New Conversation
// still reaches the established fresh-run reset.

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

function identity(overrides) {
  return Object.assign({
    conversation_id: "a".repeat(32),
    branch_id: "b_" + "b".repeat(32),
    branch_revision: 3,
    assistant_turn_id:
      "t_" + "b".repeat(32) + "_00000002_0000000000000002",
    assistant_turn_index: 2,
    assistant_turn_version: 2,
    assistant_text: "answer",
  }, overrides || {});
}

function shellHarness() {
  const page = loadPage({
    scripts: ["conversation_shell.js"],
  });
  return {
    page,
    shell: page.context.conversationShellCreate(),
    transcript: page.registry.get("conversation-transcript"),
    active: page.registry.get("active-assistant-card"),
  };
}

function renderShell(harness, overrides) {
  const current = identity();
  harness.shell.render(Object.assign({
    action: null,
    runIdentity: current,
    conversationIdentity: current,
    runFrameCount: 0,
    renderTranscript() {},
  }, overrides || {}));
}

test("only append and fork events follow the tail", async () => {
  const h = shellHarness();
  for (const type of ["appended", "forked"]) {
    h.transcript.scrollTop = 31;
    h.transcript.scrollHeight = 400;
    renderShell(h, {
      action: { type },
      renderTranscript() {
        h.transcript.scrollHeight = 900;
      },
    });
    await tick();
    assert.equal(h.transcript.scrollTop, 900);
  }
});

test("assistant updates preserve transcript position", async () => {
  const h = shellHarness();
  h.transcript.scrollTop = 73;
  h.transcript.scrollHeight = 400;

  renderShell(h, {
    action: { type: "assistant_updated" },
    renderTranscript() {
      h.transcript.scrollHeight = 900;
    },
  });
  await tick();

  assert.equal(h.transcript.scrollTop, 73);
});

test("loading older turns preserves the visible anchor", () => {
  const h = shellHarness();
  h.transcript.scrollTop = 120;
  h.transcript.scrollHeight = 500;

  renderShell(h, {
    action: { type: "older_loaded" },
    renderTranscript() {
      h.transcript.scrollHeight = 760;
    },
  });

  assert.equal(h.transcript.scrollTop, 380);
});

test("the workspace requires the exact active run", () => {
  const h = shellHarness();

  renderShell(h);
  assert.equal(h.active.hidden, false);

  renderShell(h, {
    conversationIdentity: identity({
      assistant_turn_id: "00000004",
    }),
  });
  assert.equal(h.active.hidden, true);

  renderShell(h, {
    conversationIdentity: identity({
      branch_id: "b_" + "c".repeat(32),
    }),
  });
  assert.equal(h.active.hidden, true);
});

test("legacy framed runs retain their workspace", () => {
  const h = shellHarness();

  renderShell(h, {
    runIdentity: null,
    conversationIdentity: null,
    runFrameCount: 2,
  });

  assert.equal(h.active.hidden, false);
});

test("an idle shell hides the rich assistant card", () => {
  const h = shellHarness();

  renderShell(h, {
    runIdentity: null,
    conversationIdentity: null,
    runFrameCount: 0,
  });

  assert.equal(h.active.hidden, true);
});

test("New Conversation clears after durable creation", async () => {
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
