// The generator composer, driven without app.js.
//
// Strategy: load only its persistence dependency and the factory,
// then compose it with recording callbacks. Interactions go through
// its public methods and the DOM controls it owns. Storage is read
// back under the shipped keys. Context-count replies are delivered in
// and out of order the way a worker can deliver them.
//
// Passing proves the extraction is real: prompt DOM and mutable state
// are private to the closure, draft and history formats stay
// unchanged, imports still confirm replacement, Enter submits, and
// count_prompt keeps its payload, fencing and warning behavior.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "persist.js",
  "generator_composer.js",
];
const HISTORY_KEY = "diffusion_prompt_history";
const DRAFT_KEY = "diffusion_param_state";

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function waitForCount() {
  return new Promise((resolve) => setTimeout(resolve, 380));
}

function loadComposer(settings) {
  const config = settings || {};
  const page = loadPage({
    scripts: SCRIPTS,
    storage: config.storage,
  });
  if (config.draftState) {
    page.sandbox.sessionStorage.setItem(
      DRAFT_KEY, JSON.stringify(config.draftState)
    );
  }
  const state = {
    submitted: 0,
    draftChanges: 0,
    statuses: [],
    sent: [],
    thinking: Boolean(config.thinking),
    outputBudget: config.outputBudget || 0,
    countReady: Boolean(config.countReady),
  };
  const composer = page.context.generatorComposerCreate({
    onSubmit() {
      state.submitted += 1;
    },
    onDraftChanged() {
      state.draftChanges += 1;
    },
    reportStatus(text, danger) {
      state.statuses.push({ text: text, danger: danger });
    },
    readThinking() {
      return state.thinking;
    },
    readOutputBudget() {
      return state.outputBudget;
    },
    isCountReady() {
      return state.countReady;
    },
    sendCountPrompt(payload) {
      state.sent.push(host(payload));
    },
  });
  composer.wire();
  composer.boot();
  composer.configure({
    modelId: "smollm3",
    inputMode: config.inputMode || "chat",
    contextLength: config.contextLength || null,
  });
  return {
    page,
    composer,
    state,
    input: page.registry.get("prompt-input"),
  };
}

function completeOptions() {
  return {
    onSubmit() {},
    onDraftChanged() {},
    reportStatus() {},
    readThinking() { return false; },
    readOutputBudget() { return 0; },
    isCountReady() { return false; },
    sendCountPrompt() {},
  };
}

test("every page callback is required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const names = Object.keys(completeOptions());

  for (const name of names) {
    const options = completeOptions();
    delete options[name];

    assert.throws(
      () => page.context.generatorComposerCreate(options),
      new RegExp(name)
    );
  }
});

test("prompt DOM and state do not leak out of the factory", () => {
  const h = loadComposer({});

  assert.equal(h.page.context.promptInput, undefined);
  assert.equal(h.page.context.promptHistory, undefined);
  assert.equal(h.page.context.pendingImportFile, undefined);
  assert.equal(h.page.context.promptCountRequest, undefined);
});

test("configuration restores the draft and completion copy", () => {
  const h = loadComposer({
    inputMode: "completion",
    contextLength: 65536,
    draftState: {
      smollm3: {
        experimental: true,
        params: { temperature: "0.8" },
        prompt: "A REST API is",
      },
    },
  });

  assert.equal(h.input.value, "A REST API is");
  assert.equal(
    h.page.registry.get("prompt-label").textContent,
    "Text to continue"
  );
  assert.match(h.input.placeholder, /continue/);
  assert.equal(
    h.page.registry.get("prompt-mode-info").hidden, false
  );
});

test("typing persists only the prompt member", () => {
  const h = loadComposer({
    draftState: {
      smollm3: {
        experimental: true,
        params: { temperature: "0.8" },
        prompt: "old",
      },
      llada: {
        params: { gen_length: "160" },
        prompt: "other model",
      },
    },
  });
  h.input.value = "new draft";

  h.input.dispatch("input");

  const stored = JSON.parse(
    h.page.sandbox.sessionStorage.getItem(DRAFT_KEY)
  );
  assert.equal(stored.smollm3.prompt, "new draft");
  assert.equal(stored.smollm3.experimental, true);
  assert.deepEqual(
    stored.smollm3.params, { temperature: "0.8" }
  );
  assert.equal(stored.llada.prompt, "other model");
  assert.equal(h.state.draftChanges, 1);
});

test("Enter submits while Shift Enter remains text input", () => {
  const h = loadComposer({});
  let prevented = 0;

  h.input.dispatch("keydown", {
    key: "Enter",
    shiftKey: true,
    preventDefault() { prevented += 1; },
  });
  h.input.dispatch("keydown", {
    key: "Enter",
    shiftKey: false,
    preventDefault() { prevented += 1; },
  });

  assert.equal(h.state.submitted, 1);
  assert.equal(prevented, 1);
});

test("generation records the same de-duplicated history", () => {
  const h = loadComposer({
    storage: {
      [HISTORY_KEY]: JSON.stringify(["repeat", "older"]),
    },
  });

  h.composer.prepareGeneration(" repeat ");

  const stored = JSON.parse(
    h.page.sandbox.localStorage.getItem(HISTORY_KEY)
  );
  assert.deepEqual(stored, ["repeat", "older"]);
  assert.equal(
    h.page.registry.get("prompt-history").hidden, false
  );
});

test("an import confirms before replacing a draft", async () => {
  const h = loadComposer({});
  const picker = h.page.registry.get("prompt-file-input");
  const modal = h.page.registry.get("modal-import");
  h.input.value = "keep until confirmed";
  picker.files = [{
    name: "notes.md",
    type: "text/markdown",
    size: 12,
    text() {
      return Promise.resolve("# Imported\n");
    },
  }];

  picker.dispatch("change");

  assert.equal(modal.open, true);
  assert.equal(h.input.value, "keep until confirmed");
  h.page.registry.get("btn-import-confirm").click();
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(h.input.value, "# Imported\n");
  assert.deepEqual(h.state.statuses, [{
    text: "Imported notes.md.",
    danger: false,
  }]);
  assert.equal(h.state.draftChanges, 1);
});

test("context requests are fenced and repaint budget warnings",
  async () => {
    const h = loadComposer({
      thinking: true,
      outputBudget: 20,
      countReady: true,
      contextLength: 100,
    });
    const row = h.page.registry.get("prompt-context");
    const count = h.page.registry.get("prompt-context-count");
    const note = h.page.registry.get("prompt-context-note");
    h.input.value = "Explain diffusion";

    h.input.dispatch("input");
    await waitForCount();

    assert.deepEqual(h.state.sent, [{
      type: "count_prompt",
      text: "Explain diffusion",
      thinking: true,
      request_id: 1,
    }]);
    h.composer.handleCountResult({
      request_id: 1,
      count: 90,
      truncated: false,
    });
    assert.equal(count.textContent, "90 / 100 tokens");
    assert.equal(
      note.textContent,
      "prompt + 20 output exceeds the window"
    );
    assert.equal(row.classList.contains("is-over"), false);

    h.state.outputBudget = 5;
    h.composer.parametersChanged();
    assert.equal(note.textContent, "");

    h.state.thinking = false;
    h.composer.parametersChanged();
    assert.equal(row.classList.contains("is-empty"), true);
    await waitForCount();

    assert.equal(h.state.sent[1].request_id, 2);
    assert.equal(h.state.sent[1].thinking, false);
    h.composer.handleCountResult({
      request_id: 1,
      count: 1,
      truncated: false,
    });
    assert.equal(row.classList.contains("is-empty"), true);
    h.composer.handleCountResult({
      request_id: 2,
      count: 101,
      truncated: false,
    });
    assert.equal(note.textContent, "over the context window");
    assert.equal(row.classList.contains("is-over"), true);
  }
);
