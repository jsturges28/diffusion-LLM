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
  });
  controller.wire();
  return { page, calls, controller };
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
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
