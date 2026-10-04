// The bounded conversation reducer, without DOM or network.
//
// Strategy: feed it the same manifests and pages the REST boundary
// returns, then inspect immutable projections and packed messages.
// Passing proves the cache never exceeds four pages, eviction removes
// whole exchanges, context indices remain absolute, pending retries
// reuse their user turn, and only the durable tail is editable.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..", "..", "..",
  "src", "web", "static", "conversation_state.js"
);
const CONVERSATION_ID = "a".repeat(32);

function load() {
  const context = vm.createContext({});
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context);
  return context;
}

function turn(index, overrides) {
  const role = index % 2 === 1 ? "user" : "assistant";
  return Object.assign({
    turn_id: String(index).padStart(8, "0"),
    index,
    version: role === "assistant" ? 2 : 1,
    role,
    text: role + " " + index,
    partial: false,
    model_id: role === "assistant" ? "llada" : null,
    input_mode: role === "assistant" ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  }, overrides || {});
}

function manifest(turnCount, overrides) {
  return Object.assign({
    id: CONVERSATION_ID,
    title: "Test",
    revision: 1 + turnCount,
    turn_count: turnCount,
    tail_turn_id: turnCount > 0
      ? String(turnCount).padStart(8, "0")
      : null,
    tail_version: turnCount > 0 ? 2 : null,
    pending_assistant_id: null,
  }, overrides || {});
}

function page(start, finish, hasMore, revision) {
  const turns = [];
  for (let index = start; index <= finish; index += 1) {
    turns.push(turn(index));
  }
  return {
    conversation_id: CONVERSATION_ID,
    revision: revision || 1 + finish,
    turns,
    next_before: hasMore
      ? String(start).padStart(8, "0")
      : null,
    has_more: hasMore,
  };
}

function loaded(api, start, finish, hasMore) {
  return api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: manifest(finish),
      page: page(start, finish, hasMore),
    }
  );
}

test("four pages are the hard cache and loading bound", () => {
  const api = load();
  let state = loaded(api, 151, 200, true);
  for (const bounds of [[101, 150], [51, 100], [1, 50]]) {
    state = api.conversationStateReduce(state, {
      type: "older_loaded",
      page: page(bounds[0], bounds[1], bounds[0] > 1, 201),
    });
  }

  assert.equal(state.turns.length, 200);
  assert.equal(state.pagesLoaded, 4);
  assert.equal(state.hasMore, false);
  assert.equal(state.nextBefore, null);
});

test("a new pair evicts the oldest whole exchange", () => {
  const api = load();
  let state = loaded(api, 151, 200, true);
  for (const bounds of [[101, 150], [51, 100], [1, 50]]) {
    state = api.conversationStateReduce(state, {
      type: "older_loaded",
      page: page(bounds[0], bounds[1], false, 201),
    });
  }
  state = api.conversationStateReduce(state, {
    type: "appended",
    conversation: manifest(202, {
      pending_assistant_id: "00000202",
      tail_version: 1,
    }),
    userTurn: turn(201),
    assistantTurn: turn(202, {
      version: 1,
      text: "",
      partial: true,
    }),
  });

  assert.equal(state.turns.length, 200);
  assert.equal(state.turns[0].turn_id, "00000003");
  assert.equal(state.turns[1].turn_id, "00000004");
});

test("message packing keeps pending user and absolute offset", () => {
  const api = load();
  let state = loaded(api, 151, 200, false);
  state = api.conversationStateReduce(state, {
    type: "appended",
    conversation: manifest(202, {
      revision: 203,
      pending_assistant_id: "00000202",
      tail_version: 1,
    }),
    userTurn: turn(201, { text: "pending question" }),
    assistantTurn: turn(202, {
      version: 1,
      text: "",
      partial: true,
    }),
  });

  const packed = api.conversationStateMessages(
    state, "different draft"
  );
  assert.equal(packed.messages.at(-1).content, "pending question");
  assert.equal(packed.messages.at(-1).turn_id, "00000201");
  assert.equal(packed.candidate_turn_offset, 150);
  assert.equal(packed.assistant_turn_id, "00000202");
  assert.equal(packed.conversation_revision, 203);
});

test("only the completed durable tail is editable", () => {
  const api = load();
  const complete = loaded(api, 1, 4, false);
  const pending = api.conversationStateReduce(complete, {
    type: "appended",
    conversation: manifest(6, {
      pending_assistant_id: "00000006",
      tail_version: 1,
    }),
    userTurn: turn(5),
    assistantTurn: turn(6, {
      version: 1,
      text: "",
      partial: true,
    }),
  });

  const currentIdentity = api.conversationStateIdentity(complete);
  const oldIdentity = Object.assign({}, currentIdentity, {
    assistant_turn_id: "00000002",
    turn_index: 2,
  });
  assert.equal(
    api.conversationStateCanEdit(complete, currentIdentity), true
  );
  assert.equal(
    api.conversationStateCanEdit(complete, oldIdentity), false
  );
  assert.equal(
    api.conversationStateCanEdit(
      pending, api.conversationStateIdentity(pending)
    ),
    false
  );
});

test("compact turns discard heavy arbitrary metadata", () => {
  const api = load();
  const state = api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: manifest(2),
      page: {
        conversation_id: CONVERSATION_ID,
        revision: 3,
        turns: [
          turn(1),
          turn(2, {
            metadata: {
              status: "completed",
              frames: ["large"],
              candidates: [{ large: true }],
            },
          }),
        ],
        next_before: null,
        has_more: false,
      },
    }
  );

  assert.equal(state.turns[1].metadata.status, "completed");
  assert.equal("frames" in state.turns[1].metadata, false);
  assert.equal("candidates" in state.turns[1].metadata, false);
});
