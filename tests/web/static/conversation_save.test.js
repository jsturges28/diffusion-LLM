// Save Conversation workflow without loading the generator page.
//
// Strategy: drive the shipped controller through a fake view and
// client. Passing proves the dialog uses server counts, exact heads
// fence confirmation, active XAI saves before publication, one
// operation id survives retries, and partial success is explicit.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..", "..", "..",
  "src", "web", "static",
  "conversation_save.js"
);

function load() {
  const context = vm.createContext({
    Promise,
    Error,
    TypeError,
    Object,
  });
  vm.runInContext(
    fs.readFileSync(SOURCE, "utf8"),
    context,
    { filename: SOURCE }
  );
  return context;
}

function head(revision = 4, version = 2) {
  return {
    conversation_id: "a".repeat(32),
    branch_id: "b_" + "b".repeat(32),
    branch_revision: revision,
    turn_count: 2,
    tail_turn_id: "c".repeat(32),
    tail_version: version,
  };
}

function fakeView() {
  const state = {
    callbacks: null,
    open: false,
    disabled: null,
    pending: false,
    title: "Investigation",
    statuses: [],
    shown: [],
  };
  return {
    state,
    wire(callbacks) {
      state.callbacks = callbacks;
    },
    show(preview, savesTail) {
      state.open = true;
      state.shown.push({ preview, savesTail });
    },
    close() {
      state.open = false;
    },
    setDisabled(value) {
      state.disabled = value;
    },
    setPending(value) {
      state.pending = value;
    },
    showStatus(message) {
      state.statuses.push(message);
    },
    clearStatus() {},
    title() {
      return state.title;
    },
    isOpen() {
      return state.open;
    },
  };
}

function harness(settings = {}) {
  const context = load();
  const view = fakeView();
  const state = {
    head: head(),
    blocked: "",
    needsSave: settings.needsSave === true,
    previewCalls: [],
    createCalls: [],
    saveCalls: 0,
    busy: [],
    saved: [],
    blockedMessages: [],
    tailPinned: false,
  };
  const client = {
    preview(value) {
      state.previewCalls.push(value);
      return Promise.resolve({
        default_title: "Question",
        exchange_count: 1,
        xai_count: 0,
        text_only_count: 1,
        unavailable_count: 0,
        tail_xai_status: state.tailPinned
          ? "pinned"
          : "text_only",
      });
    },
    create(value) {
      state.createCalls.push(value);
      if (settings.createError) {
        return Promise.reject(new Error(settings.createError));
      }
      return Promise.resolve({
        snapshot_id: "d".repeat(32),
        analytics_url: "/analytics.html?conversation=" + "d".repeat(32),
      });
    },
  };
  const controller = context.conversationSaveCreate({
    client,
    view,
    readHead: () => state.head,
    readBlockReason: () => state.blocked,
    shouldSaveActiveTail: () => (
      state.needsSave ? "save" : false
    ),
    saveActiveTail() {
      state.saveCalls += 1;
      if (settings.saveFails) {
        return Promise.resolve(false);
      }
      state.head = head(5, 3);
      state.tailPinned = true;
      return Promise.resolve(true);
    },
    createOperationId: () => "e".repeat(32),
    onSaved: (result) => state.saved.push(result),
    onBlocked: (message) => state.blockedMessages.push(message),
    onBusy: (value) => state.busy.push(value),
  });
  controller.wire();
  return { controller, state, view };
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

test("opening renders server-authoritative summary", async () => {
  const { state, view } = harness();

  view.state.callbacks.open();
  await settle();

  assert.equal(state.previewCalls.length, 1);
  assert.equal(view.state.shown.length, 1);
  assert.equal(view.state.shown[0].savesTail, false);
  assert.deepEqual(state.busy, [true]);
});

test("active tail saves before snapshot publication", async () => {
  const { state, view } = harness({ needsSave: true });
  view.state.callbacks.open();
  await settle();

  view.state.callbacks.confirm();
  await settle();
  await settle();

  assert.equal(state.saveCalls, 1);
  assert.equal(state.createCalls.length, 1);
  assert.equal(state.createCalls[0].branch_revision, 5);
  assert.equal(state.createCalls[0].operationId, "e".repeat(32));
  assert.equal(state.saved.length, 1);
  assert.deepEqual(state.busy, [true, false]);
});

test("head changes block confirmation", async () => {
  const { state, view } = harness();
  view.state.callbacks.open();
  await settle();
  state.head = head(8, 4);

  view.state.callbacks.confirm();
  await settle();

  assert.equal(state.createCalls.length, 0);
  assert.match(view.state.statuses[0], /selected path changed/i);
});

test("failed active save publishes no snapshot", async () => {
  const { state, view } = harness({
    needsSave: true,
    saveFails: true,
  });
  view.state.callbacks.open();
  await settle();

  view.state.callbacks.confirm();
  await settle();
  await settle();

  assert.equal(state.saveCalls, 1);
  assert.equal(state.createCalls.length, 0);
  assert.match(view.state.statuses[0], /could not be saved/i);
});

test("snapshot failure reports the already-saved run", async () => {
  const { state, view } = harness({
    needsSave: true,
    createError: "disk full",
  });
  view.state.callbacks.open();
  await settle();

  view.state.callbacks.confirm();
  await settle();
  await settle();

  assert.equal(state.createCalls.length, 1);
  assert.match(
    view.state.statuses[0],
    /active run was saved.*disk full/i
  );
});

test("blocked workflow never requests a preview", async () => {
  const { state, view } = harness();
  state.blocked = "Wait for generation.";

  view.state.callbacks.open();
  await settle();

  assert.equal(state.previewCalls.length, 0);
  assert.deepEqual(state.blockedMessages, ["Wait for generation."]);
});
