// Durable message actions, driven through their delegated event seam.
//
// Strategy: render bounded card decorations into the shared DOM stub,
// dispatch clicks at the transcript root, and hold mutations where a
// second click could race them. Passing proves role-specific actions,
// clipboard fallback, edit and confirmation state, branch paging,
// focus restoration, and blocking all stay inside this controller.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const BRANCH_A = "b_" + "a".repeat(32);
const BRANCH_B = "b_" + "b".repeat(32);
const BRANCH_C = "b_" + "c".repeat(32);
const OPERATION_ONE = "0".repeat(31) + "1";

function turn(index, overrides) {
  const assistant = index % 2 === 0;
  return Object.assign({
    turn_id: "turn-" + index,
    index,
    version: assistant ? 2 : 1,
    role: assistant ? "assistant" : "user",
    text: assistant ? "answer " + index : "question " + index,
    partial: false,
    model_id: assistant ? "test-model" : null,
    input_mode: assistant ? "chat" : null,
    metadata: {},
    run_link: null,
  }, overrides || {});
}

function initialState() {
  return {
    conversation: {
      id: "d".repeat(32),
      branch_id: BRANCH_A,
      turn_count: 4,
      pending_assistant_id: null,
    },
    selectedBranchId: BRANCH_A,
    turns: [turn(1), turn(2), turn(3), turn(4)],
    branchPoints: [],
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, refuse) => {
    resolve = accept;
    reject = refuse;
  });
  return { promise, resolve, reject };
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function explicitConflict(reason, message) {
  const error = new Error(message);
  error.status = 409;
  error.reason = reason;
  error.conversationReloaded = true;
  error.conversationReplayable = false;
  error.conversationConflict = true;
  return error;
}

function harness(overrides) {
  const settings = overrides || {};
  const page = loadPage({
    scripts: [
      "conversation_action_view.js",
      "conversation_actions.js",
    ],
  });
  const root = page.document.getElementById(
    "conversation-transcript"
  );
  const turnsRoot = page.document.getElementById(
    "conversation-turns"
  );
  const activeMount = page.document.getElementById(
    "active-assistant-actions"
  );
  const activeCard = page.document.getElementById(
    "active-assistant-card"
  );
  root.appendChild(turnsRoot);
  root.appendChild(activeMount);
  root.appendChild(activeCard);
  let state = settings.state || initialState();
  let blockReason = settings.blockReason || "";
  let operationSerial = 0;
  const calls = {
    select: [],
    edit: [],
    delete: [],
    retry: [],
    state: [],
  };
  let controller;

  function render() {
    const cards = state.turns.map((item) => {
      const article = page.document.createElement("article");
      article.setAttribute("data-turn-id", item.turn_id);
      const text = page.document.createElement("div");
      text.className = "conversation-turn-text";
      text.textContent = item.text;
      article.appendChild(text);
      const points = state.branchPoints.filter(
        (point) => point.turn_index === item.index
      );
      controller.decorateTurn(article, item, points);
      return article;
    });
    turnsRoot.replaceChildren(...cards);
  }

  controller = page.context.conversationActionsCreate({
    readState: () => state,
    readConfiguration: () => settings.readConfiguration
      ? settings.readConfiguration()
      : {
        modelId: "test-model",
        modelDisplay: "Test Model",
        inputMode: "chat",
        settingsSummary: "Steps 32, Temperature 0.7",
        parameters: { steps: 32, temperature: 0.7 },
        experimental: false,
        valid: true,
        validationMessage: "",
      },
    readBlockReason: () => blockReason,
    requestRender: render,
    onStateChanged: (value) => calls.state.push(value),
    selectBranch: (branchId) => {
      calls.select.push(branchId);
      return settings.selectBranch
        ? settings.selectBranch(branchId)
        : Promise.resolve(true);
    },
    editUser: (input) => {
      calls.edit.push(input);
      return settings.editUser
        ? settings.editUser(input)
        : Promise.resolve({
          result: {
            user_turn: { turn_id: "opaque-edited-user" },
          },
          launched: true,
        });
    },
    deleteUser: (input) => {
      calls.delete.push(input);
      return settings.deleteUser
        ? settings.deleteUser(input)
        : Promise.resolve({
          branch: { branch_id: BRANCH_B },
          removed_turn_count: 2,
        });
    },
    retryAssistant: (input) => {
      calls.retry.push(input);
      return settings.retryAssistant
        ? settings.retryAssistant(input)
        : Promise.resolve({
          result: {
            assistant_turn: {
              turn_id: "opaque-retried-assistant",
            },
          },
          launched: true,
        });
    },
    createOperationId: settings.createOperationId || (() => {
      operationSerial += 1;
      return operationSerial.toString(16).padStart(32, "0");
    }),
  });
  controller.wire();
  render();

  return {
    page,
    root,
    turnsRoot,
    activeMount,
    activeCard,
    controller,
    calls,
    render,
    state: () => state,
    setState(value) {
      state = value;
      controller.reconcile();
      render();
    },
    setBlockReason(value) {
      blockReason = value;
    },
  };
}

function card(h, turnId) {
  return h.turnsRoot.querySelector(
    '[data-turn-id="' + turnId + '"]'
  );
}

function action(cardElement, name) {
  const buttons = cardElement.querySelectorAll(
    "[data-conversation-action]"
  );
  return buttons.find(
    (button) =>
      button.getAttribute("data-conversation-action") === name
  );
}

function dispatch(h, target) {
  h.root.dispatch("click", { target });
}

function dispatchDialog(h, kind, buttonId) {
  const dialog = h.page.registry.get(
    "conversation-" + kind + "-dialog"
  );
  const button = h.page.registry.get(buttonId);
  const actionName = buttonId.endsWith("-cancel")
    ? "cancel"
    : "confirm-" + kind;
  button.setAttribute(
    "data-conversation-confirmation", actionName
  );
  dialog.dispatch("click", { target: button });
}

test("each durable role gets only its allowed icon actions", () => {
  const h = harness();
  const user = card(h, "turn-1");
  const assistant = card(h, "turn-2");

  assert.deepEqual(
    user.querySelectorAll("[data-conversation-action]")
      .map((button) =>
        button.getAttribute("data-conversation-action")
      ),
    ["copy", "edit", "delete"]
  );
  assert.deepEqual(
    assistant.querySelectorAll("[data-conversation-action]")
      .map((button) =>
        button.getAttribute("data-conversation-action")
      ),
    ["copy", "retry"]
  );
  const icon = action(user, "edit").firstChild;
  assert.equal(icon.tag, "svg");
  assert.equal(icon.getAttribute("aria-hidden"), "true");
  assert.equal(action(user, "edit").tag, "button");
});

test("pending assistants have no message actions", () => {
  const state = initialState();
  state.conversation.pending_assistant_id = "turn-4";
  state.turns[3] = turn(4, {
    version: 1,
    text: "",
  });
  const h = harness({ state });

  assert.equal(
    card(h, "turn-4").querySelectorAll(
      "[data-conversation-action]"
    ).length,
    0
  );
  h.controller.decorateActive(state.turns[3], [], true);
  assert.equal(h.activeMount.hidden, true);
});

test("the completed rich assistant gets Copy and Retry", () => {
  const h = harness();
  const point = {
    turn_index: 4,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_B,
    branch_ids: [BRANCH_A, BRANCH_B],
    deleted_branch_ids: [],
  };
  h.controller.decorateActive(
    h.state().turns[3], [point], true
  );

  assert.equal(h.activeMount.hidden, false);
  assert.deepEqual(
    h.activeMount.querySelector(
      ".conversation-turn-actions"
    ).querySelectorAll(
      "[data-conversation-action]"
    ).map((button) =>
      button.getAttribute("data-conversation-action")
    ),
    ["copy", "retry"]
  );
  assert.equal(
    h.activeMount.querySelector(
      ".conversation-branch-position"
    ).textContent,
    "Path 2 / 2"
  );
});

test("clipboard success reports Copied", async () => {
  const h = harness();
  let copied = "";
  h.page.context.navigator.clipboard.writeText = (text) => {
    copied = text;
    return Promise.resolve();
  };

  dispatch(h, action(card(h, "turn-1"), "copy"));
  await tick();

  assert.equal(copied, "question 1");
  assert.equal(
    h.page.registry.get("conversation-action-status").textContent,
    "Copied"
  );
});

test("clipboard rejection uses one bounded fallback", async () => {
  const h = harness();
  const commands = [];
  h.page.context.navigator.clipboard.writeText = () =>
    Promise.reject(new Error("denied"));
  h.page.document.execCommand = (command) => {
    commands.push(command);
    return true;
  };
  const bodyCount = h.page.document.body.children.length;
  const copy = action(card(h, "turn-2"), "copy");
  copy.focus();

  dispatch(h, copy);
  await tick();

  assert.deepEqual(commands, ["copy"]);
  assert.equal(h.page.document.body.children.length, bodyCount);
  assert.equal(h.page.document.activeElement, copy);
  assert.equal(
    h.page.registry.get("conversation-action-status").textContent,
    "Copied"
  );
});

test("clipboard and fallback failure stays in the live region",
  async () => {
  const h = harness();
  h.page.context.navigator.clipboard.writeText = () =>
    Promise.reject(new Error("denied"));
  h.page.document.execCommand = () => false;
  const copy = action(card(h, "turn-1"), "copy");
  copy.focus();

  dispatch(h, copy);
  await tick();

  const status = h.page.registry.get(
    "conversation-action-status"
  );
  assert.match(status.textContent, /Copy failed/);
  assert.equal(status.classes.has("is-error"), true);
  assert.equal(h.page.document.activeElement, copy);
});

test("clipboard fallback refuses text beyond its fixed bound",
  async () => {
  const h = harness();
  let commands = 0;
  h.page.document.execCommand = () => {
    commands += 1;
    return true;
  };

  await assert.rejects(
    h.page.context.conversationActionsClipboardFallback(
      "x".repeat(1000001)
    ),
    /fallback copy bound/
  );
  assert.equal(commands, 0);
});

test("a stale clipboard rejection cannot overwrite newer copy",
  async () => {
  const first = deferred();
  const h = harness();
  const copied = [];
  h.page.context.navigator.clipboard.writeText = (text) => {
    if (text === "question 1") {
      return first.promise;
    }
    copied.push(text);
    return Promise.resolve();
  };
  h.page.document.execCommand = (command) => {
    copied.push(command);
    return true;
  };
  const copyA = action(card(h, "turn-1"), "copy");
  const copyB = action(card(h, "turn-3"), "copy");

  copyA.focus();
  dispatch(h, copyA);
  copyB.focus();
  dispatch(h, copyB);
  await tick();
  first.reject(new Error("late denial"));
  await tick();

  assert.deepEqual(copied, ["question 3"]);
  assert.equal(h.page.document.activeElement, copyB);
  assert.equal(
    h.page.registry.get("conversation-action-status").textContent,
    "Copied"
  );
});

test("inline edit Cancel restores exact text and focus", () => {
  const h = harness();
  dispatch(h, action(card(h, "turn-1"), "edit"));
  const editing = card(h, "turn-1");
  const input = editing.querySelector(
    '[data-conversation-edit-input="turn-1"]'
  );
  assert.equal(input.value, "question 1");
  assert.match(
    editing.querySelector(
      ".conversation-inline-edit-note"
    ).textContent,
    /Test Model \(test-model\).*Steps 32/
  );
  input.value = "changed only in the draft";
  h.root.dispatch("input", { target: input });

  dispatch(h, action(editing, "edit-cancel"));

  const restored = card(h, "turn-1");
  assert.equal(
    restored.querySelector(".conversation-turn-text").textContent,
    "question 1"
  );
  assert.equal(action(restored, "edit").focused, true);
  assert.equal(h.calls.edit.length, 0);
});

test("Escape cancels inline edit with no mutation", () => {
  const h = harness();
  dispatch(h, action(card(h, "turn-3"), "edit"));
  const input = card(h, "turn-3").querySelector(
    '[data-conversation-edit-input="turn-3"]'
  );
  let prevented = false;

  h.root.dispatch("keydown", {
    target: input,
    key: "Escape",
    preventDefault() {
      prevented = true;
    },
  });

  assert.equal(prevented, true);
  assert.equal(h.calls.edit.length, 0);
  assert.equal(action(card(h, "turn-3"), "edit").focused, true);
});

test("Save forks once with current model and edited text",
  async () => {
  const pending = deferred();
  const h = harness({
    editUser: () => pending.promise,
  });
  dispatch(h, action(card(h, "turn-3"), "edit"));
  const editing = card(h, "turn-3");
  const input = editing.querySelector(
    '[data-conversation-edit-input="turn-3"]'
  );
  input.value = "edited question";
  h.root.dispatch("input", { target: input });

  dispatch(h, action(editing, "edit-save"));
  dispatch(h, action(card(h, "turn-3"), "edit-save"));
  assert.equal(h.calls.edit.length, 0);
  await tick();
  assert.equal(h.calls.edit.length, 1);
  assert.deepEqual(host(h.calls.edit[0]), {
    operationId: OPERATION_ONE,
    userTurnId: "turn-3",
    text: "edited question",
    modelId: "test-model",
    inputMode: "chat",
    metadata: {},
    configuration: {
      modelId: "test-model",
      modelDisplay: "Test Model",
      inputMode: "chat",
      settingsSummary: "Steps 32, Temperature 0.7",
      parameters: { steps: 32, temperature: 0.7 },
      experimental: false,
      valid: true,
      validationMessage: "",
    },
  });
  assert.equal(h.state().turns[2].text, "question 3");
  const status = card(h, "turn-3").querySelector(
    '[data-conversation-edit-status="turn-3"]'
  );
  assert.ok(status);
  assert.equal(status.getAttribute("role"), "status");
  assert.equal(status.getAttribute("aria-busy"), "true");
  assert.equal(status.focused, true);
  assert.equal(
    card(h, "turn-3").querySelector(
      '[data-conversation-edit-input="turn-3"]'
    ),
    null
  );

  pending.resolve({
    result: {
      user_turn: { turn_id: "opaque-edited-user" },
    },
    launched: true,
  });
  await tick();
  assert.equal(h.controller.blocking(), false);
});

test("inline Edit uses the configuration shown when opened",
  async () => {
  let temperature = 0.7;
  const h = harness({
    readConfiguration() {
      return {
        modelId: "test-model",
        modelDisplay: "Test Model",
        inputMode: "chat",
        settingsSummary: "Temperature " + temperature,
        parameters: { temperature },
        experimental: false,
        valid: true,
        validationMessage: "",
      };
    },
  });
  dispatch(h, action(card(h, "turn-1"), "edit"));
  temperature = 0.9;

  dispatch(h, action(card(h, "turn-1"), "edit-save"));
  await tick();
  await tick();

  assert.equal(h.calls.edit[0].configuration.parameters.temperature, 0.7);
  assert.equal(
    h.calls.edit[0].configuration.settingsSummary,
    "Temperature 0.7"
  );
});

test("a final Edit failure keeps its frozen action retryable", async () => {
  const h = harness({
    editUser() {
      return Promise.reject(new Error("edit rejected"));
    },
  });
  dispatch(h, action(card(h, "turn-1"), "edit"));

  dispatch(h, action(card(h, "turn-1"), "edit-save"));
  await tick();
  await tick();

  assert.equal(h.controller.blocking(), true);
  assert.equal(
    card(h, "turn-1")
      .querySelector("[data-conversation-edit-input]").focused,
    true
  );
  assert.match(
    h.page.registry.get("conversation-action-status").textContent,
    /edit rejected/
  );
});

test("a Delete failure stays inside its retryable dialog",
  async () => {
  const h = harness({
    deleteUser() {
      return Promise.reject(new Error("delete rejected"));
    },
  });
  const trigger = action(card(h, "turn-1"), "delete");
  dispatch(h, trigger);
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );

  await tick();
  await tick();

  const dialog = h.page.registry.get(
    "conversation-delete-dialog"
  );
  const status = h.page.registry.get(
    "conversation-delete-status"
  );
  assert.equal(dialog.open, true);
  assert.equal(h.controller.blocking(), true);
  assert.equal(dialog.getAttribute("aria-busy"), "false");
  assert.match(status.textContent, /delete rejected/);
  assert.equal(status.focused, true);
  assert.equal(
    h.page.registry.get(
      "btn-conversation-delete-cancel"
    ).disabled,
    false
  );
  assert.equal(
    h.page.registry.get(
      "btn-conversation-delete-confirm"
    ).disabled,
    false
  );

  dispatchDialog(
    h, "delete", "btn-conversation-delete-cancel"
  );
  assert.equal(dialog.open, false);
  assert.equal(trigger.focused, true);
});

test("Delete names dependent count and Cancel restores focus", () => {
  const h = harness();
  const trigger = action(card(h, "turn-1"), "delete");
  dispatch(h, trigger);

  const dialog = h.page.registry.get(
    "conversation-delete-dialog"
  );
  assert.equal(dialog.open, true);
  assert.match(
    h.page.registry.get("conversation-delete-message").textContent,
    /4 selected and later dependent turns/
  );
  assert.match(
    h.page.registry.get("conversation-delete-message").textContent,
    /original path remains available/i
  );
  const cancel = h.page.registry.get(
    "btn-conversation-delete-cancel"
  );
  const confirm = h.page.registry.get(
    "btn-conversation-delete-confirm"
  );
  assert.equal(cancel.focused, true);
  assert.equal((cancel.listeners.click || []).length, 0);
  assert.equal((confirm.listeners.click || []).length, 0);

  dispatchDialog(
    h, "delete", "btn-conversation-delete-cancel"
  );
  assert.equal(dialog.open, false);
  assert.equal(trigger.focused, true);
  assert.equal(h.calls.delete.length, 0);
});

test("native dialog dismissal restores the invoking control", () => {
  const h = harness();
  const trigger = action(card(h, "turn-2"), "retry");
  dispatch(h, trigger);
  const dialog = h.page.registry.get(
    "conversation-retry-dialog"
  );

  dialog.close("");

  assert.equal(dialog.open, false);
  assert.equal(trigger.focused, true);
  assert.equal(h.calls.retry.length, 0);
});

test("confirmed Delete focuses its non-message marker",
  async () => {
  const h = harness({
    deleteUser(input) {
      const state = h.state();
      state.conversation.branch_id = BRANCH_B;
      state.conversation.turn_count = 2;
      state.selectedBranchId = BRANCH_B;
      state.turns = state.turns.slice(0, 2);
      state.branchPoints = [{
        turn_index: 3,
        source_branch_id: BRANCH_A,
        selected_branch_id: BRANCH_B,
        branch_ids: [BRANCH_A, BRANCH_B],
        deleted_branch_ids: [BRANCH_B],
      }];
      h.render();
      const marker = h.controller.deletionMarker(
        state.branchPoints[0]
      );
      h.turnsRoot.appendChild(marker);
      return Promise.resolve({
        branch: { branch_id: BRANCH_B },
        removed_turn_count: 2,
      });
    },
  });
  dispatch(h, action(card(h, "turn-3"), "delete"));
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );
  await tick();
  await tick();

  assert.deepEqual(host(h.calls.delete), [{
    operationId: OPERATION_ONE,
    userTurnId: "turn-3",
  }]);
  const marker = h.turnsRoot.querySelector(
    '[data-conversation-deletion-branch="' + BRANCH_B + '"]'
  );
  assert.ok(marker);
  assert.equal(marker.classes.has("conversation-turn"), false);
  assert.equal(marker.focused, true);
});

test("Retry dialog freezes the named model and settings",
  async () => {
  const pending = deferred();
  const h = harness({
    retryAssistant: () => pending.promise,
  });
  const trigger = action(card(h, "turn-2"), "retry");
  dispatch(h, trigger);
  const dialog = h.page.registry.get(
    "conversation-retry-dialog"
  );
  assert.equal(dialog.open, true);
  assert.match(
    h.page.registry.get("conversation-retry-message").textContent,
    /Test Model \(test-model\).*Steps 32, Temperature 0.7/
  );
  assert.match(
    h.page.registry.get("conversation-retry-message").textContent,
    /Input mode: chat/
  );

  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  await tick();
  assert.deepEqual(host(h.calls.retry), [{
    operationId: OPERATION_ONE,
    assistantTurnId: "turn-2",
    modelId: "test-model",
    inputMode: "chat",
    configuration: {
      modelId: "test-model",
      modelDisplay: "Test Model",
      inputMode: "chat",
      settingsSummary: "Steps 32, Temperature 0.7",
      parameters: { steps: 32, temperature: 0.7 },
      experimental: false,
      valid: true,
      validationMessage: "",
    },
  }]);
  assert.equal(dialog.open, true);
  assert.equal(dialog.getAttribute("aria-busy"), "true");
  const progress = h.page.registry.get(
    "conversation-retry-status"
  );
  assert.match(progress.textContent, /Creating the retry path/);
  assert.equal(progress.getAttribute("role"), "status");
  assert.equal(progress.getAttribute("aria-busy"), "true");
  assert.equal(progress.focused, true);

  h.activeCard.hidden = false;
  pending.resolve({
    result: {
      assistant_turn: {
        turn_id: "opaque-retried-assistant",
      },
    },
    launched: true,
  });
  await tick();
  assert.equal(dialog.open, false);
  assert.equal(dialog.getAttribute("aria-busy"), "false");
  assert.equal(h.activeCard.focused, true);
});

test("reloaded Edit response reuses one durable operation", async () => {
  let durableForks = 0;
  const result = {
    result: {
      user_turn: { turn_id: "opaque-edited-user" },
    },
    launched: false,
  };
  const h = harness({
    editUser() {
      durableForks += durableForks === 0 ? 1 : 0;
      if (h.calls.edit.length === 1) {
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve(result);
    },
  });
  dispatch(h, action(card(h, "turn-1"), "edit"));
  const editing = card(h, "turn-1");
  const input = editing.querySelector(
    '[data-conversation-edit-input="turn-1"]'
  );
  input.value = "receipt edit";
  h.root.dispatch("input", { target: input });

  dispatch(h, action(editing, "edit-save"));
  await tick();
  await tick();
  await tick();

  assert.equal(durableForks, 1);
  assert.equal(h.calls.edit.length, 2);
  assert.equal(
    h.calls.edit[0].operationId,
    h.calls.edit[1].operationId
  );
  assert.equal(h.calls.edit[1].text, "receipt edit");
});

test("reloaded Delete response replays automatically",
  async () => {
  let durableForks = 0;
  const result = {
    branch: { branch_id: BRANCH_B },
    removed_turn_count: 4,
  };
  const h = harness({
    deleteUser() {
      durableForks += durableForks === 0 ? 1 : 0;
      if (h.calls.delete.length === 1) {
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve(result);
    },
  });
  dispatch(h, action(card(h, "turn-1"), "delete"));
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );
  await tick();
  await tick();
  await tick();

  assert.equal(durableForks, 1);
  assert.equal(h.calls.delete.length, 2);
  assert.equal(
    h.calls.delete[0].operationId,
    h.calls.delete[1].operationId
  );
  assert.equal(
    h.page.registry.get("conversation-delete-dialog").open,
    false
  );
});

test("reloaded Retry response reuses its frozen receipt", async () => {
  let durableForks = 0;
  const result = {
    result: {
      assistant_turn: {
        turn_id: "opaque-retried-assistant",
      },
    },
    launched: false,
  };
  const h = harness({
    retryAssistant() {
      durableForks += durableForks === 0 ? 1 : 0;
      if (h.calls.retry.length === 1) {
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve(result);
    },
  });
  h.activeCard.hidden = true;
  dispatch(h, action(card(h, "turn-2"), "retry"));
  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  await tick();
  await tick();
  await tick();

  assert.equal(durableForks, 1);
  assert.equal(h.calls.retry.length, 2);
  assert.equal(
    h.calls.retry[0].operationId,
    h.calls.retry[1].operationId
  );
  assert.deepEqual(
    host(h.calls.retry[0].configuration),
    host(h.calls.retry[1].configuration)
  );
  assert.equal(
    h.page.document.activeElement,
    h.page.registry.get("btn-generate")
  );
  assert.equal(h.activeCard.focused, false);
});

test("reloaded old-page Edit automatically replays once",
  async () => {
  const h = harness({
    editUser() {
      if (h.calls.edit.length === 1) {
        const state = h.state();
        h.setState(Object.assign({}, state, {
          turns: state.turns.slice(2),
        }));
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve({
        result: {
          user_turn: { turn_id: "replayed-user" },
        },
        launched: false,
      });
    },
  });
  dispatch(h, action(card(h, "turn-1"), "edit"));
  const input = card(h, "turn-1").querySelector(
    '[data-conversation-edit-input="turn-1"]'
  );
  input.value = "frozen old edit";
  h.root.dispatch("input", { target: input });

  dispatch(h, action(card(h, "turn-1"), "edit-save"));
  await tick();
  await tick();
  await tick();

  assert.equal(h.calls.edit.length, 2);
  assert.equal(
    h.calls.edit[0].operationId,
    h.calls.edit[1].operationId
  );
  assert.equal(h.calls.edit[1].text, "frozen old edit");
  assert.equal(h.controller.blocking(), false);
});

test("reloaded old-page Delete automatically replays once",
  async () => {
  const h = harness({
    deleteUser() {
      if (h.calls.delete.length === 1) {
        const state = h.state();
        h.setState(Object.assign({}, state, {
          turns: state.turns.slice(2),
        }));
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve({
        branch: { branch_id: BRANCH_B },
        removed_turn_count: 4,
      });
    },
  });
  dispatch(h, action(card(h, "turn-1"), "delete"));
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );
  await tick();
  await tick();
  await tick();

  assert.equal(h.calls.delete.length, 2);
  assert.equal(
    h.calls.delete[0].operationId,
    h.calls.delete[1].operationId
  );
  assert.equal(h.controller.blocking(), false);
});

test("reloaded old-page Retry automatically replays once",
  async () => {
  const h = harness({
    retryAssistant() {
      if (h.calls.retry.length === 1) {
        const state = h.state();
        h.setState(Object.assign({}, state, {
          turns: state.turns.slice(2),
        }));
        const error = new Error("reply lost");
        error.conversationReloaded = true;
        error.conversationReplayable = true;
        return Promise.reject(error);
      }
      return Promise.resolve({
        result: {
          assistant_turn: { turn_id: "replayed-assistant" },
        },
        launched: false,
      });
    },
  });
  dispatch(h, action(card(h, "turn-2"), "retry"));
  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  await tick();
  await tick();
  await tick();

  assert.equal(h.calls.retry.length, 2);
  assert.equal(
    h.calls.retry[0].operationId,
    h.calls.retry[1].operationId
  );
  assert.deepEqual(
    host(h.calls.retry[0].configuration),
    host(h.calls.retry[1].configuration)
  );
  assert.equal(h.controller.blocking(), false);
});

test("Delete catalog conflict closes stale count before reopening",
  async () => {
  let attempts = 0;
  const h = harness({
    deleteUser() {
      attempts += 1;
      if (attempts === 1) {
        const state = h.state();
        state.conversation.turn_count = 6;
        state.turns.push(turn(5), turn(6));
        h.setState(state);
        return Promise.reject(explicitConflict(
          "catalog_revision_conflict", "path catalog changed"
        ));
      }
      return Promise.resolve({
        branch: { branch_id: BRANCH_B },
        removed_turn_count: 6,
      });
    },
  });
  dispatch(h, action(card(h, "turn-1"), "delete"));
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );
  await tick();
  await tick();

  const dialog = h.page.registry.get(
    "conversation-delete-dialog"
  );
  assert.equal(h.calls.delete.length, 1);
  assert.equal(dialog.open, false);
  assert.equal(h.controller.blocking(), false);

  dispatch(h, action(card(h, "turn-1"), "delete"));
  assert.match(
    h.page.registry.get("conversation-delete-message").textContent,
    /6 selected and later dependent turns/
  );
  dispatchDialog(
    h, "delete", "btn-conversation-delete-confirm"
  );
  await tick();
  await tick();

  assert.equal(h.calls.delete.length, 2);
  assert.notEqual(
    h.calls.delete[0].operationId,
    h.calls.delete[1].operationId
  );
});

test("Edit branch conflict cancels its frozen operation",
  async () => {
  let attempts = 0;
  const h = harness({
    editUser() {
      attempts += 1;
      if (attempts === 1) {
        return Promise.reject(explicitConflict(
          "branch_revision_conflict", "selected path changed"
        ));
      }
      return Promise.resolve({
        result: {
          user_turn: { turn_id: "fresh-edit-user" },
        },
        launched: false,
      });
    },
  });
  dispatch(h, action(card(h, "turn-1"), "edit"));
  dispatch(h, action(card(h, "turn-1"), "edit-save"));
  await tick();
  await tick();

  assert.equal(h.calls.edit.length, 1);
  assert.equal(h.controller.blocking(), false);
  assert.ok(action(card(h, "turn-1"), "edit"));

  dispatch(h, action(card(h, "turn-1"), "edit"));
  dispatch(h, action(card(h, "turn-1"), "edit-save"));
  await tick();
  await tick();

  assert.equal(h.calls.edit.length, 2);
  assert.notEqual(
    h.calls.edit[0].operationId,
    h.calls.edit[1].operationId
  );
});

test("Retry state conflict closes and requires a fresh operation",
  async () => {
  let attempts = 0;
  const h = harness({
    retryAssistant() {
      attempts += 1;
      if (attempts === 1) {
        return Promise.reject(explicitConflict(
          "state_conflict", "assistant state changed"
        ));
      }
      return Promise.resolve({
        result: {
          assistant_turn: { turn_id: "fresh-retry-assistant" },
        },
        launched: false,
      });
    },
  });
  dispatch(h, action(card(h, "turn-2"), "retry"));
  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  await tick();
  await tick();

  const dialog = h.page.registry.get(
    "conversation-retry-dialog"
  );
  assert.equal(h.calls.retry.length, 1);
  assert.equal(dialog.open, false);
  assert.equal(h.controller.blocking(), false);

  dispatch(h, action(card(h, "turn-2"), "retry"));
  dispatchDialog(
    h, "retry", "btn-conversation-retry-confirm"
  );
  await tick();
  await tick();

  assert.equal(h.calls.retry.length, 2);
  assert.notEqual(
    h.calls.retry[0].operationId,
    h.calls.retry[1].operationId
  );
});

test("secure randomness failure blocks every fork", () => {
  const h = harness({
    createOperationId() {
      throw new Error("secure randomness unavailable");
    },
  });

  dispatch(h, action(card(h, "turn-1"), "edit"));
  dispatch(h, action(card(h, "turn-1"), "delete"));
  dispatch(h, action(card(h, "turn-2"), "retry"));

  assert.equal(h.controller.blocking(), false);
  assert.equal(h.calls.edit.length, 0);
  assert.equal(h.calls.delete.length, 0);
  assert.equal(h.calls.retry.length, 0);
  assert.match(
    h.page.registry.get("conversation-action-status").textContent,
    /secure randomness unavailable/
  );
});

test("branch pager labels paths and disables both edges", () => {
  const state = initialState();
  state.branchPoints = [{
    turn_index: 1,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_A,
    branch_ids: [BRANCH_A, BRANCH_B, BRANCH_C],
    deleted_branch_ids: [],
  }];
  const h = harness({ state });
  let pager = card(h, "turn-1").querySelector(
    ".conversation-branch-pager"
  );
  let buttons = pager.querySelectorAll(
    ".conversation-branch-button"
  );
  assert.equal(pager.querySelector(
    ".conversation-branch-position"
  ).textContent, "Path 1 / 3");
  assert.equal(buttons[0].disabled, true);
  assert.equal(buttons[1].disabled, false);
  assert.match(buttons[1].getAttribute("aria-label"), /Next path/);

  state.branchPoints[0] = Object.assign(
    {}, state.branchPoints[0], {
      selected_branch_id: BRANCH_C,
    }
  );
  h.render();
  pager = card(h, "turn-1").querySelector(
    ".conversation-branch-pager"
  );
  buttons = pager.querySelectorAll(
    ".conversation-branch-button"
  );
  assert.equal(pager.querySelector(
    ".conversation-branch-position"
  ).textContent, "Path 3 / 3");
  assert.equal(buttons[0].disabled, false);
  assert.equal(buttons[1].disabled, true);
});

test("rapid branch clicks dispatch one guarded read", async () => {
  const pending = deferred();
  const state = initialState();
  state.branchPoints = [{
    turn_index: 1,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_A,
    branch_ids: [BRANCH_A, BRANCH_B],
    deleted_branch_ids: [],
  }];
  const h = harness({
    state,
    selectBranch: () => pending.promise,
  });
  const next = card(h, "turn-1").querySelectorAll(
    ".conversation-branch-button"
  )[1];

  dispatch(h, next);
  dispatch(h, next);
  await tick();
  assert.deepEqual(h.calls.select, [BRANCH_B]);

  state.branchPoints[0] = Object.assign(
    {}, state.branchPoints[0], {
      selected_branch_id: BRANCH_B,
    }
  );
  h.render();
  pending.resolve(true);
  await tick();
  const focused = card(h, "turn-1").querySelector(
    ".conversation-branch-pager"
  );
  assert.equal(focused.focused, true);
});

test("branch selection falls back when its pager leaves the page",
  async () => {
  const state = initialState();
  state.branchPoints = [{
    turn_index: 1,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_A,
    branch_ids: [BRANCH_A, BRANCH_B],
    deleted_branch_ids: [],
  }];
  const h = harness({
    state,
    selectBranch() {
      state.turns = state.turns.slice(2);
      state.branchPoints = [];
      h.render();
      return Promise.resolve(true);
    },
  });
  const next = card(h, "turn-1").querySelectorAll(
    ".conversation-branch-button"
  )[1];

  dispatch(h, next);
  await tick();
  await tick();

  assert.equal(h.page.document.activeElement, h.root);
});

test("failed branch selection also has a focus fallback",
  async () => {
  const state = initialState();
  state.branchPoints = [{
    turn_index: 1,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_A,
    branch_ids: [BRANCH_A, BRANCH_B],
    deleted_branch_ids: [],
  }];
  const h = harness({
    state,
    selectBranch() {
      state.turns = state.turns.slice(2);
      state.branchPoints = [];
      h.render();
      return Promise.reject(new Error("selection failed"));
    },
  });
  const next = card(h, "turn-1").querySelectorAll(
    ".conversation-branch-button"
  )[1];

  dispatch(h, next);
  await tick();
  await tick();

  assert.equal(h.page.document.activeElement, h.root);
});

test("generation, pending completion, and save reasons block forks",
  () => {
  const reasons = [
    "Wait for generation to finish.",
    "Finish the pending response first.",
    "Wait for the run save to finish.",
  ];
  for (const reason of reasons) {
    const h = harness({ blockReason: reason });
    dispatch(h, action(card(h, "turn-1"), "edit"));
    dispatch(h, action(card(h, "turn-1"), "delete"));
    dispatch(h, action(card(h, "turn-2"), "retry"));

    assert.equal(h.calls.edit.length, 0);
    assert.equal(h.calls.delete.length, 0);
    assert.equal(h.calls.retry.length, 0);
    assert.equal(
      h.page.registry.get(
        "conversation-action-status"
      ).textContent,
      reason
    );
  }
});

test("a one-path point renders no pager", () => {
  const h = harness();
  const article = card(h, "turn-1");
  h.controller.decorateTurn(article, h.state().turns[0], [{
    turn_index: 1,
    source_branch_id: BRANCH_A,
    selected_branch_id: BRANCH_A,
    branch_ids: [BRANCH_A],
    deleted_branch_ids: [],
  }]);

  assert.equal(
    article.querySelector(".conversation-branch-pager"),
    null
  );
});
