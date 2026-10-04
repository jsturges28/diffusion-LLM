// The transcript view, driven against the shared DOM seam.
//
// Strategy: render reducer states with saved, partial, and active
// turns, then inspect only visible DOM and native control behavior.
// Passing proves the workspace's tail assistant is not duplicated,
// frozen text reports whether XAI was saved, Analytics links target
// their run, and older-page loading remains keyboard accessible.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ID = "a".repeat(32);

function turn(index, overrides) {
  const assistant = index % 2 === 0;
  return Object.assign({
    turn_id: String(index).padStart(8, "0"),
    index,
    version: assistant ? 2 : 1,
    role: assistant ? "assistant" : "user",
    text: assistant ? "answer " + index : "question " + index,
    partial: false,
    model_id: assistant ? "llada" : null,
    input_mode: assistant ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  }, overrides || {});
}

function state(page, overrides) {
  const api = page.context;
  return api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: Object.assign({
        id: ID,
        title: "Test",
        revision: 6,
        turn_count: 4,
        tail_turn_id: "00000004",
        tail_version: 2,
        pending_assistant_id: null,
      }, overrides || {}),
      page: {
        conversation_id: ID,
        revision: 6,
        turns: [
          turn(1),
          turn(2, {
            run_link: { run_id: "saved-run", revision: 1 },
          }),
          turn(3),
          turn(4, { partial: true }),
        ],
        next_before: "00000001",
        has_more: true,
      },
    }
  );
}

function harness() {
  const page = loadPage({
    scripts: [
      "conversation_state.js",
      "conversation_view.js",
    ],
  });
  let olderLoads = 0;
  const view = page.context.conversationViewCreate({
    onLoadOlder() {
      olderLoads += 1;
    },
  });
  view.wire();
  return {
    page,
    view,
    olderLoads: () => olderLoads,
  };
}

test("the workspace alone owns the active tail assistant", () => {
  const h = harness();
  h.view.render(state(h.page), {
    workspaceAssistantTurnId: "00000004",
  });
  const turns = h.page.registry.get("conversation-turns");

  assert.equal(turns.children.length, 3);
  assert.equal(
    turns.children.some((node) =>
      node.getAttribute("data-turn-id") === "00000004"
    ),
    false
  );
  assert.equal(
    turns.children.at(-1).getAttribute("data-turn-id"),
    "00000003"
  );
});

test("a completed tail without a workspace remains visible", () => {
  const h = harness();
  h.view.render(state(h.page));
  const turns = h.page.registry.get("conversation-turns");

  assert.equal(turns.children.length, 4);
  assert.equal(
    turns.children.at(-1).getAttribute("data-turn-id"),
    "00000004"
  );
  assert.equal(
    turns.children.at(-1)
      .querySelector(".conversation-turn-text").textContent,
    "answer 4"
  );
});

test("saved frozen responses link directly to Analytics", () => {
  const h = harness();
  h.view.render(state(h.page));
  const turns = h.page.registry.get("conversation-turns");
  const assistant = turns.children[1];
  const link = assistant.querySelector(".conversation-run-link");

  assert.equal(link.textContent, "Open in Analytics");
  assert.equal(link.href, "/analytics.html?run=saved-run");
  assert.ok(
    assistant.querySelector(".conversation-turn-badge-saved")
  );
});

test("an unsaved frozen response says its XAI is text only", () => {
  const h = harness();
  const current = state(h.page);
  const api = h.page.context;
  const appended = api.conversationStateReduce(current, {
    type: "appended",
    conversation: {
      id: ID,
      title: "Test",
      revision: 7,
      turn_count: 6,
      tail_turn_id: "00000006",
      tail_version: 1,
      pending_assistant_id: "00000006",
    },
    userTurn: turn(5),
    assistantTurn: turn(6, {
      version: 1,
      text: "",
      partial: true,
    }),
  });
  h.view.render(appended);
  const turns = h.page.registry.get("conversation-turns");
  const oldTail = turns.children.find((node) =>
    node.getAttribute("data-turn-id") === "00000004"
  );

  assert.ok(
    oldTail.querySelector(".conversation-turn-badge-text-only")
  );
  assert.equal(
    oldTail.querySelector(".conversation-run-link"),
    null
  );
});

test("Load older is a native bounded action", () => {
  const h = harness();
  h.view.render(state(h.page));
  const button = h.page.registry.get("btn-load-older");

  assert.equal(button.hidden, false);
  button.click();
  assert.equal(h.olderLoads(), 1);

  button.disabled = true;
  button.click();
  assert.equal(h.olderLoads(), 1);
});

test("no active conversation has an honest empty state", () => {
  const h = harness();
  h.view.render(h.page.context.conversationStateCreate());

  assert.match(
    h.page.registry.get("conversation-empty").textContent,
    /durable conversation/
  );
  assert.equal(
    h.page.registry.get("conversation-turns").children.length,
    0
  );
});
