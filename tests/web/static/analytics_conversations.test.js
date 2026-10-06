// Saved-conversation Analytics catalog and exchange viewer.
//
// Strategy: load the shipped controller in the shared DOM stub and
// drive its real delegated events against a fake client. Passing
// proves list rows open bounded exchanges, pinned XAI delegates with
// snapshot-local URLs, rename updates CAS, and delete removes only
// the snapshot row.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { loadPage } = require("./dom_stub");

const SCRIPTS = ["analytics_conversations.js"];
const SNAPSHOT_ID = "a".repeat(32);
const TURN_ID = "b".repeat(32);

function summary() {
  return {
    snapshot_id: SNAPSHOT_ID,
    title: "Investigation",
    title_revision: 1,
    created_at: "2026-10-06T00:00:00Z",
    exchange_count: 1,
    xai_count: 1,
    text_only_count: 0,
    unavailable_count: 0,
  };
}

function pagePayload() {
  return {
    snapshot_id: SNAPSHOT_ID,
    turns: [
      {
        index: 1,
        turn_id: "c".repeat(32),
        role: "user",
        text: "Why?",
        partial: false,
      },
      {
        index: 2,
        turn_id: TURN_ID,
        role: "assistant",
        text: "Because.",
        partial: false,
        model_id: "smollm3",
        xai: {
          status: "pinned",
          backend: "smollm3",
          model_type: "autoregressive",
        },
      },
    ],
    next_before: null,
    has_more: false,
  };
}

function descendants(node) {
  const found = [];
  const pending = node.children.slice();
  while (pending.length > 0) {
    const next = pending.shift();
    found.push(next);
    pending.push(...next.children);
  }
  return found;
}

function harness() {
  const page = loadPage({ scripts: SCRIPTS });
  const calls = {
    renamed: [],
    deleted: [],
    xai: [],
    toasts: [],
  };
  const client = {
    list: () => Promise.resolve([summary()]),
    metadata: () => Promise.resolve(Object.assign(summary(), {
      source: { branch_id: "branch" },
    })),
    turns: () => Promise.resolve(pagePayload()),
    rename(snapshotId, title, revision) {
      calls.renamed.push({ snapshotId, title, revision });
      return Promise.resolve(Object.assign(summary(), {
        title,
        title_revision: revision + 1,
        source: { branch_id: "branch" },
      }));
    },
    delete(snapshotId) {
      calls.deleted.push(snapshotId);
      return Promise.resolve({ deleted: snapshotId });
    },
    pinnedUrl(snapshotId, turnId, resource) {
      return [snapshotId, turnId, resource].join("/");
    },
  };
  const controller = page.context.analyticsConversationsCreate({
    client,
    openXai: (input) => calls.xai.push(input),
    showToast: (message) => calls.toasts.push(message),
    focusFallback: page.document.getElementById("tab-conversations"),
  });
  controller.wire();
  return { page, calls, controller };
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("catalog row opens its exchange and pinned XAI", async () => {
  const run = harness();
  await run.controller.activate();
  const tbody = run.page.registry.get("conversations-tbody");
  const row = tbody.children[0];

  tbody.dispatch("click", { target: row });
  await settle();
  const detail = run.page.registry.get("conversation-detail-modal");
  const exchanges = run.page.registry.get(
    "saved-conversation-exchanges"
  );
  const xai = descendants(exchanges).find((element) =>
    element.getAttribute("data-view-xai-turn") === TURN_ID
  );
  exchanges.dispatch("click", { target: xai });

  assert.equal(detail.open, true);
  assert.equal(exchanges.children.length, 1);
  assert.equal(run.calls.xai.length, 1);
  assert.equal(run.calls.xai[0].summary.model_type, "autoregressive");
  assert.match(run.calls.xai[0].urls.frames, /frames$/);
});

test("rename and delete mutate only snapshot catalog state",
  async () => {
  const run = harness();
  await run.controller.activate();
  const tbody = run.page.registry.get("conversations-tbody");
  tbody.dispatch("click", { target: tbody.children[0] });
  await settle();
  const title = run.page.registry.get("saved-conversation-title");
  title.value = "Renamed";

  run.page.registry.get("btn-rename-conversation")
    .dispatch("click");
  await settle();
  run.page.registry.get("btn-delete-conversation")
    .dispatch("click");
  run.page.registry.get("btn-delete-conversation-confirm")
    .dispatch("click");
  await settle();

  assert.deepEqual(run.calls.renamed, [{
    snapshotId: SNAPSHOT_ID,
    title: "Renamed",
    revision: 1,
  }]);
  assert.deepEqual(run.calls.deleted, [SNAPSHOT_ID]);
  assert.equal(tbody.children.length, 0);
});

test("an older page cannot land in a newly opened snapshot",
  async () => {
  const page = loadPage({ scripts: SCRIPTS });
  const secondId = "d".repeat(32);
  const older = deferred();
  const rows = [
    summary(),
    Object.assign({}, summary(), {
      snapshot_id: secondId,
      title: "Second",
    }),
  ];
  const client = {
    list: () => Promise.resolve(rows),
    metadata: (id) => Promise.resolve(Object.assign(
      {}, rows.find((row) => row.snapshot_id === id),
      { source: { branch_id: "branch" } }
    )),
    turns(id, before) {
      if (id === SNAPSHOT_ID && before) {
        return older.promise;
      }
      const label = id === SNAPSHOT_ID ? "A new" : "B only";
      return Promise.resolve({
        turns: [
          {
            index: 1,
            turn_id: "1",
            role: "user",
            text: label,
            partial: false,
          },
          {
            index: 2,
            turn_id: id.slice(0, 8),
            role: "assistant",
            text: label,
            partial: false,
            xai: { status: "text_only" },
          },
        ],
        next_before: id === SNAPSHOT_ID ? "1" : null,
        has_more: id === SNAPSHOT_ID,
      });
    },
    rename: () => Promise.reject(new Error("unused")),
    delete: () => Promise.reject(new Error("unused")),
    pinnedUrl: () => "",
  };
  const controller = page.context.analyticsConversationsCreate({
    client,
    openXai() {},
    showToast() {},
    focusFallback: page.document.getElementById("tab-conversations"),
  });
  controller.wire();
  await controller.activate();
  await controller.openLinked(SNAPSHOT_ID);
  page.registry.get("btn-load-older-exchanges").dispatch("click");
  await controller.openLinked(secondId);
  older.resolve({
    turns: [
      {
        index: 1,
        turn_id: "old",
        role: "user",
        text: "A old",
        partial: false,
      },
      {
        index: 2,
        turn_id: "old-answer",
        role: "assistant",
        text: "A old",
        partial: false,
        xai: { status: "text_only" },
      },
    ],
    next_before: null,
    has_more: false,
  });
  await settle();
  const texts = descendants(
    page.registry.get("saved-conversation-exchanges")
  ).filter((element) => element.tag === "pre")
    .map((element) => element.textContent);

  assert.deepEqual(texts, ["B only", "B only"]);
});
