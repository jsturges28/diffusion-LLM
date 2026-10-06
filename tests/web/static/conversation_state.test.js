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

const STATIC = path.join(
  __dirname,
  "..", "..", "..",
  "src", "web", "static"
);
const CONVERSATION_ID = "a".repeat(32);
const BRANCH_ID = "b_" + "b".repeat(32);
const OTHER_BRANCH_ID = "b_" + "c".repeat(32);

function load() {
  const context = vm.createContext({});
  for (const name of [
    "conversation_generation.js",
    "conversation_state.js",
  ]) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context
    );
  }
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

function pendingGenerationConfiguration(overrides) {
  return Object.assign({
    codec_version: 1,
    model_id: "llada",
    input_mode: "chat",
    device: "cuda",
    schema_id: "1".repeat(64),
    experimental: false,
    parameters: {
      steps: 128,
      temperature: 0.75,
      alternatives: true,
    },
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

test("only message-changing actions request a context recount", () => {
  const api = load();
  for (const type of [
    "clear",
    "created",
    "loaded",
    "forked",
    "older_loaded",
    "appended",
    "assistant_updated",
  ]) {
    assert.equal(
      api.conversationStateActionChangesMessages({ type }),
      true,
      type
    );
  }
  for (const type of [
    "older_started",
    "run_linked",
    "catalog_refreshed",
    "failed",
  ]) {
    assert.equal(
      api.conversationStateActionChangesMessages({ type }),
      false,
      type
    );
  }
});

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
  assert.equal(packed.branch_id, "b_" + CONVERSATION_ID);
  assert.equal(packed.branch_revision, 203);
  assert.equal(packed.assistant_turn_index, 202);
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
    assistant_turn_index: 2,
  });
  assert.equal(
    api.conversationStateCanEdit(complete, currentIdentity), true
  );
  assert.equal(
    api.conversationStateCanEdit(complete, oldIdentity), false
  );
  assert.equal(
    api.conversationStateCanEdit(
      complete,
      Object.assign({}, currentIdentity, {
        branch_id: OTHER_BRANCH_ID,
      })
    ),
    false
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

test("pending configuration is retained by a pure reader", () => {
  const api = load();
  const configuration = pendingGenerationConfiguration();
  const state = api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: manifest(2, {
        schema_version: 2,
        branch_id: BRANCH_ID,
        branch_revision: 3,
        catalog_revision: 0,
        default_branch_id: BRANCH_ID,
        pending_assistant_id: "00000002",
        tail_version: 1,
      }),
      page: {
        schema_version: 2,
        conversation_id: CONVERSATION_ID,
        branch_id: BRANCH_ID,
        branch_revision: 3,
        revision: 3,
        catalog_revision: 0,
        default_branch_id: BRANCH_ID,
        turns: [
          turn(1),
          turn(2, {
            version: 1,
            text: "",
            partial: true,
            metadata: {
              pending_generation_v1: configuration,
            },
          }),
        ],
        next_before: null,
        has_more: false,
        branch_points: [],
      },
    }
  );

  const durable =
    api.conversationStatePendingGenerationConfiguration(state);
  assert.equal(durable.modelId, "llada");
  assert.equal(durable.codecVersion, 1);
  assert.equal(durable.inputMode, "chat");
  assert.equal(durable.device, "cuda");
  assert.equal(durable.experimental, false);
  assert.equal(durable.parameters.temperature, 0.75);
  assert.equal(Object.isFrozen(durable), true);
  assert.equal(Object.isFrozen(durable.parameters), true);
});

test("historical user and completed reserved keys stay ordinary", () => {
  const api = load();
  const user = api.conversationStateTurn(
    turn(1, {
      metadata: {
        pending_generation_v1: { historical: "user" },
      },
    }),
    "b_" + CONVERSATION_ID,
    1
  );
  const compact = api.conversationStateTurn(
    turn(2, {
      metadata: {
        status: "completed",
        pending_generation_v1: { historical: "completed" },
      },
    }),
    "b_" + CONVERSATION_ID,
    2
  );
  const legacyPending = api.conversationStateTurn(
    turn(2, {
      version: 1,
      text: "",
      partial: true,
      metadata: {
        pending_generation_v1: { historical: "pending" },
      },
    }),
    "b_" + CONVERSATION_ID,
    1,
    "00000002"
  );

  assert.equal("pending_generation_v1" in user.metadata, false);
  assert.equal(compact.metadata.status, "completed");
  assert.equal("pending_generation_v1" in compact.metadata, false);
  assert.equal(
    "pending_generation_v1" in legacyPending.metadata,
    false
  );
});

test("pending configuration parser rejects invalid bounded shapes",
  () => {
  const api = load();
  const scenarios = [
    pendingGenerationConfiguration({ unknown: true }),
    pendingGenerationConfiguration({ codec_version: 2 }),
    pendingGenerationConfiguration({ model_id: "mamba3" }),
    pendingGenerationConfiguration({
      experimental: 1,
    }),
    pendingGenerationConfiguration({
      parameters: { temperature: Infinity },
    }),
    pendingGenerationConfiguration({
      parameters: { temperature: 1e101 },
    }),
    pendingGenerationConfiguration({
      parameters: Object.fromEntries(
        Array.from({ length: 65 }, (_, index) => [
          "parameter_" + index,
          index,
        ])
      ),
    }),
  ];

  for (const configuration of scenarios) {
    assert.throws(() => api.conversationStateTurn(
      turn(2, {
        version: 1,
        text: "",
        partial: true,
        metadata: {
          pending_generation_v1: configuration,
        },
      }),
      "b_" + CONVERSATION_ID,
      2,
      "00000002"
    ));
  }
});

function opaqueTurn(index, role, branchId) {
  const slot = String(index + 40).padStart(8, "0");
  return {
    turn_id: "t_" + branchId.slice(2) + "_" + slot
      + "_" + String(index).padStart(16, "0"),
    branch_id: branchId,
    index,
    version: role === "assistant" ? 2 : 1,
    role,
    text: role + " opaque",
    partial: false,
    model_id: role === "assistant" ? "llada" : null,
    input_mode: role === "assistant" ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  };
}

function branchManifest(branchId, revision) {
  const assistant = opaqueTurn(2, "assistant", branchId);
  return {
    schema_version: 2,
    id: CONVERSATION_ID,
    title: "Branches",
    branch_id: branchId,
    branch_revision: revision,
    revision,
    catalog_revision: 4,
    default_branch_id: OTHER_BRANCH_ID,
    turn_count: 2,
    tail_turn_id: assistant.turn_id,
    tail_version: 2,
    pending_assistant_id: null,
  };
}

function branchPage(branchId, revision, withPoint) {
  return {
    schema_version: 2,
    conversation_id: CONVERSATION_ID,
    branch_id: branchId,
    branch_revision: revision,
    revision,
    catalog_revision: 4,
    default_branch_id: OTHER_BRANCH_ID,
    turns: [
      opaqueTurn(1, "user", branchId),
      opaqueTurn(2, "assistant", branchId),
    ],
    next_before: null,
    has_more: false,
    branch_points: withPoint ? [{
      turn_index: 1,
      source_branch_id: BRANCH_ID,
      selected_branch_id: branchId,
      branch_ids: [BRANCH_ID, OTHER_BRANCH_ID],
      deleted_branch_ids: [],
    }] : [],
  };
}

test("schema v2 keeps opaque identity separate from index", () => {
  const api = load();
  const state = api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: branchManifest(BRANCH_ID, 3),
      page: branchPage(BRANCH_ID, 3, true),
    }
  );

  assert.equal(state.selectedBranchId, BRANCH_ID);
  assert.equal(state.turns[0].index, 1);
  assert.notEqual(state.turns[0].turn_id, "00000001");
  assert.equal(state.turns[0].branch_id, BRANCH_ID);
  assert.equal(state.branchPoints.length, 1);
  assert.equal(state.branchPoints[0].turn_index, 1);
});

test("loading another branch drops incompatible cached pages", () => {
  const api = load();
  let state = loaded(api, 1, 50, false);
  assert.equal(state.turns.length, 50);

  state = api.conversationStateReduce(state, {
    type: "loaded",
    conversation: branchManifest(OTHER_BRANCH_ID, 1),
    page: branchPage(OTHER_BRANCH_ID, 1, false),
  });

  assert.equal(state.selectedBranchId, OTHER_BRANCH_ID);
  assert.equal(state.turns.length, 2);
  assert.equal(state.pagesLoaded, 1);
  assert.equal(state.branchPoints.length, 0);
});

test("append removes the selected deletion marker", () => {
  const api = load();
  const manifest = branchManifest(BRANCH_ID, 3);
  const page = branchPage(BRANCH_ID, 3, false);
  page.branch_points = [{
    turn_index: 3,
    source_branch_id: OTHER_BRANCH_ID,
    selected_branch_id: BRANCH_ID,
    branch_ids: [OTHER_BRANCH_ID, BRANCH_ID],
    deleted_branch_ids: [BRANCH_ID],
  }];
  let state = api.conversationStateReduce(
    api.conversationStateCreate(),
    { type: "loaded", conversation: manifest, page: page }
  );
  const user = opaqueTurn(3, "user", BRANCH_ID);
  const assistant = Object.assign(
    {}, opaqueTurn(4, "assistant", BRANCH_ID), {
      version: 1,
      text: "",
      partial: true,
    }
  );
  const appended = Object.assign(
    {}, branchManifest(BRANCH_ID, 4), {
      turn_count: 4,
      tail_turn_id: assistant.turn_id,
      tail_version: 1,
      pending_assistant_id: assistant.turn_id,
    }
  );

  state = api.conversationStateReduce(state, {
    type: "appended",
    conversation: appended,
    userTurn: user,
    assistantTurn: assistant,
  });

  assert.equal(state.branchPoints.length, 1);
  assert.deepEqual(
    Array.from(state.branchPoints[0].deleted_branch_ids),
    []
  );
});

test("catalog refresh preserves turns and their saved run link", () => {
  const api = load();
  const originalPage = branchPage(BRANCH_ID, 3, false);
  originalPage.turns[1].run_link = {
    run_id: "saved-run",
    revision: 7,
  };
  let state = api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: branchManifest(BRANCH_ID, 3),
      page: originalPage,
    }
  );
  const turns = state.turns;
  const refreshedManifest = Object.assign(
    {}, branchManifest(BRANCH_ID, 3), {
      catalog_revision: 5,
      default_branch_id: BRANCH_ID,
    }
  );
  const refreshedPage = Object.assign(
    {}, branchPage(BRANCH_ID, 3, true), {
      catalog_revision: 5,
      default_branch_id: BRANCH_ID,
    }
  );

  state = api.conversationStateReduce(state, {
    type: "catalog_refreshed",
    conversation: refreshedManifest,
    pages: [refreshedPage],
  });

  assert.equal(state.turns, turns);
  assert.deepEqual(
    Object.assign({}, state.turns[1].run_link),
    { run_id: "saved-run", revision: 7 }
  );
  assert.equal(state.conversation.catalog_revision, 5);
  assert.equal(state.branchPoints.length, 1);
});

test("branch point payloads are bounded", () => {
  const api = load();
  const pagePayload = branchPage(BRANCH_ID, 3, false);
  pagePayload.branch_points = Array.from(
    { length: 257 },
    (_, index) => ({
      turn_index: index + 1,
      source_branch_id: BRANCH_ID,
      selected_branch_id: BRANCH_ID,
      branch_ids: [BRANCH_ID],
      deleted_branch_ids: [],
    })
  );

  assert.throws(
    () => api.conversationStateReduce(
      api.conversationStateCreate(),
      {
        type: "loaded",
        conversation: branchManifest(BRANCH_ID, 3),
        page: pagePayload,
      }
    ),
    /exceed 256/
  );
});
