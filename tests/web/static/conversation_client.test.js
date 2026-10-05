// The conversation REST client, with a deterministic fake server.
//
// Strategy: compose the real reducer with scripted HTTP responses and
// issue mutations without awaiting the first. Passing proves CAS
// revisions are read at execution time, failures do not poison the
// queue, stale writes reload but are never repeated, and malformed
// success bodies fail at the boundary.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);
const ID = "a".repeat(32);
const BRANCH_ID = "b_" + "b".repeat(32);
const BRANCH_B = "b_" + "c".repeat(32);
const BRANCH_C = "b_" + "d".repeat(32);
const OPERATION_ID = "1".repeat(32);

function load() {
  const context = vm.createContext({
    Promise,
    Error,
    TypeError,
    RangeError,
    encodeURIComponent,
  });
  for (const name of [
    "conversation_state.js",
    "conversation_client.js",
  ]) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context,
      { filename: name }
    );
  }
  return context;
}

function manifest(revision, version) {
  return {
    schema_version: 2,
    id: ID,
    title: "Test",
    branch_id: BRANCH_ID,
    branch_revision: revision,
    revision,
    catalog_revision: 1,
    default_branch_id: BRANCH_ID,
    turn_count: 2,
    tail_turn_id: turnId(2),
    tail_version: version,
    pending_assistant_id: null,
  };
}

function turnId(index) {
  return "t_" + BRANCH_ID.slice(2) + "_"
    + String(index).padStart(8, "0")
    + "_" + String(index).padStart(16, "0");
}

function turn(index, version, overrides) {
  const assistant = index % 2 === 0;
  return Object.assign({
    turn_id: turnId(index),
    branch_id: BRANCH_ID,
    index,
    version,
    role: assistant ? "assistant" : "user",
    text: assistant ? "answer" : "question",
    partial: false,
    model_id: assistant ? "llada" : null,
    input_mode: assistant ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  }, overrides || {});
}

function page(revision) {
  return {
    schema_version: 2,
    conversation_id: ID,
    branch_id: BRANCH_ID,
    branch_revision: revision,
    revision,
    catalog_revision: 1,
    default_branch_id: BRANCH_ID,
    turns: [turn(1, 1), turn(2, revision - 1)],
    next_before: null,
    has_more: false,
    branch_points: [],
  };
}

function branchManifest(branchId, revision) {
  return Object.assign({}, manifest(revision, revision - 1), {
    branch_id: branchId,
  });
}

function branchPage(branchId, revision) {
  const result = Object.assign({}, page(revision), {
    branch_id: branchId,
    turns: [
      turn(1, 1, { branch_id: branchId }),
      turn(2, revision - 1, { branch_id: branchId }),
    ],
  });
  return result;
}

function response(body, status) {
  const code = status || 200;
  return Promise.resolve({
    ok: code >= 200 && code < 300,
    status: code,
    json: () => Promise.resolve(body),
  });
}

function harness(request) {
  const api = load();
  let state = api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation: manifest(3, 2),
      page: page(3),
    }
  );
  const actions = [];
  function applyAction(action) {
    actions.push(action.type);
    state = api.conversationStateReduce(state, action);
  }
  const client = api.conversationClientCreate({
    request,
    readState: () => state,
    applyAction,
  });
  return {
    api,
    client,
    actions,
    applyAction,
    readState: () => state,
  };
}

test("queued mutations consume prior revisions", async () => {
  let release;
  const calls = [];
  const first = new Promise((resolve) => { release = resolve; });
  const h = harness((url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    if (calls.length === 1) {
      return first;
    }
    return response({
      conversation: manifest(5, 4),
      turn: turn(2, 4, {
        run_link: { run_id: "run-1", revision: 0 },
      }),
    });
  });

  const completion = h.client.updateAssistant({
    branchId: BRANCH_ID,
    branchRevision: 3,
    assistantTurnId: turnId(2),
    text: "answer",
    partial: false,
    contextPack: {},
    metadata: { status: "completed" },
  });
  const link = h.client.linkRun({
    branchId: BRANCH_ID,
    assistantTurnId: turnId(2),
    assistantTurnIndex: 2,
    assistantTurnVersion: 3,
    runId: "run-1",
    runRevision: 0,
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(calls.length, 1);
  release(await response({
    conversation: manifest(4, 3),
    turn: turn(2, 3),
  }));
  await completion;
  await link;

  assert.equal(calls.length, 2);
  assert.equal(calls[0].body.branch_id, BRANCH_ID);
  assert.equal(calls[0].body.branch_revision, 3);
  assert.equal(calls[1].body.branch_revision, 4);
  assert.equal(calls[1].body.assistant_turn_index, 2);
  assert.equal(calls[1].body.assistant_turn_version, 3);
  assert.equal(h.readState().conversation.revision, 5);
});

test("a stale append reloads and is not retried", async () => {
  let posts = 0;
  const h = harness((url, init) => {
    if (init.method === "POST") {
      posts += 1;
      return response({
        error: "stale",
        reason: "branch_revision_conflict",
        conversation_id: ID,
        branch_id: BRANCH_ID,
        expected_branch_revision: 3,
        branch_revision: 4,
      }, 409);
    }
    if (url.split("?")[0].endsWith("/metadata")) {
      return response({ conversation: manifest(4, 3) });
    }
    return response(page(4));
  });

  const conflict = await h.client.appendUser({
      text: "next",
      modelId: "llada",
      inputMode: "chat",
      metadata: {},
    }).then(
      () => null,
      (error) => error
    );

  assert.equal(posts, 1);
  assert.match(conflict.message, /stale/);
  assert.equal(conflict.conversationReloaded, true);
  assert.equal(conflict.conversationReplayable, false);
  assert.equal(conflict.conversationConflict, true);
  assert.equal(h.readState().conversation.revision, 4);
  assert.deepEqual(h.actions, ["loaded"]);
});

test("concurrent fork metadata refreshes after local completion",
  async () => {
  const completedTurn = turn(2, 3, {
    text: "local completion",
    metadata: { status: "completed" },
  });
  const changedManifest = Object.assign({}, manifest(4, 3), {
    catalog_revision: 2,
    default_branch_id: BRANCH_B,
  });
  const changedPage = Object.assign({}, page(4), {
    catalog_revision: 2,
    default_branch_id: BRANCH_B,
    turns: [turn(1, 1), completedTurn],
    branch_points: [{
      turn_index: 1,
      source_branch_id: BRANCH_ID,
      selected_branch_id: BRANCH_ID,
      branch_ids: [BRANCH_ID, BRANCH_B],
      deleted_branch_ids: [],
    }],
  });
  const h = harness((url, init) => {
    if (init.method === "PUT") {
      return response({
        conversation: changedManifest,
        turn: completedTurn,
      });
    }
    return response(changedPage);
  });

  await h.client.updateAssistant({
    branchId: BRANCH_ID,
    branchRevision: 3,
    assistantTurnId: turnId(2),
    text: "local completion",
    partial: false,
    contextPack: {},
    metadata: { status: "completed" },
  });

  const state = h.readState();
  assert.deepEqual(
    h.actions, ["assistant_updated", "catalog_refreshed"]
  );
  assert.equal(state.selectedBranchId, BRANCH_ID);
  assert.equal(state.conversation.default_branch_id, BRANCH_B);
  assert.equal(state.turns.at(-1).text, "local completion");
  assert.equal(state.branchPoints.length, 1);
  assert.deepEqual(
    Array.from(state.branchPoints[0].branch_ids),
    [BRANCH_ID, BRANCH_B]
  );
});

test("concurrent fork metadata refreshes after local append",
  async () => {
  const user = turn(3, 1, { text: "local append" });
  const assistant = turn(4, 1, {
    text: "",
    partial: true,
  });
  const changedManifest = Object.assign({}, manifest(4, 1), {
    catalog_revision: 2,
    default_branch_id: BRANCH_B,
    turn_count: 4,
    tail_turn_id: assistant.turn_id,
    pending_assistant_id: assistant.turn_id,
  });
  const changedPage = Object.assign({}, page(4), {
    catalog_revision: 2,
    default_branch_id: BRANCH_B,
    turns: [turn(1, 1), turn(2, 2), user, assistant],
    branch_points: [{
      turn_index: 1,
      source_branch_id: BRANCH_ID,
      selected_branch_id: BRANCH_ID,
      branch_ids: [BRANCH_ID, BRANCH_B],
      deleted_branch_ids: [],
    }],
  });
  const h = harness((url, init) => {
    if (init.method === "POST") {
      return response({
        conversation: changedManifest,
        user_turn: user,
        assistant_turn: assistant,
      }, 201);
    }
    return response(changedPage);
  });

  await h.client.appendUser({
    text: "local append",
    modelId: "llada",
    inputMode: "chat",
    metadata: {},
  });

  const state = h.readState();
  assert.deepEqual(
    h.actions, ["appended", "catalog_refreshed"]
  );
  assert.equal(state.selectedBranchId, BRANCH_ID);
  assert.equal(state.conversation.catalog_revision, 2);
  assert.equal(state.turns.at(-2).text, "local append");
  assert.equal(state.turns.at(-1).partial, true);
  assert.equal(state.branchPoints.length, 1);
});

test("completion refuses a revision replaced by another window",
  async () => {
  let puts = 0;
  const h = harness((url, init) => {
    if (init.method === "PUT") {
      puts += 1;
      return response({});
    }
    if (url.split("?")[0].endsWith("/metadata")) {
      return response({ conversation: manifest(4, 3) });
    }
    return response(page(4));
  });
  h.applyAction({
    type: "loaded",
    conversation: manifest(4, 3),
    page: page(4),
  });

  await assert.rejects(
    h.client.updateAssistant({
      branchId: BRANCH_ID,
      branchRevision: 3,
      assistantTurnId: turnId(2),
      text: "stale answer",
      partial: false,
      contextPack: {},
      metadata: {},
    }),
    /selected revision/
  );

  assert.equal(puts, 0);
  assert.equal(h.readState().conversation.branch_revision, 4);
});

test("a failed mutation does not poison the queue", async () => {
  let mutations = 0;
  const calls = [];
  const h = harness((url, init) => {
    calls.push(url);
    if (init.method === "PUT") {
      mutations += 1;
    }
    if (mutations === 1 && init.method === "PUT") {
      return Promise.reject(new Error("offline"));
    }
    if (url.split("?")[0].endsWith("/metadata")) {
      return response({ conversation: manifest(3, 2) });
    }
    if (init.method === "GET") {
      return response(page(3));
    }
    return response({
      conversation: manifest(4, 3),
      turn: turn(2, 3),
    });
  });

  await assert.rejects(
    h.client.updateAssistant({
      branchId: BRANCH_ID,
      branchRevision: 3,
      assistantTurnId: turnId(2),
      text: "first",
      partial: true,
      contextPack: {},
      metadata: {},
    }),
    /offline/
  );
  await h.client.updateAssistant({
    branchId: BRANCH_ID,
    branchRevision: 3,
    assistantTurnId: turnId(2),
    text: "second",
    partial: false,
    contextPack: {},
    metadata: {},
  });

  assert.equal(mutations, 2);
  assert.equal(calls.length, 4);
  assert.equal(h.readState().conversation.revision, 4);
});

test("a malformed successful body is refused", async () => {
  const h = harness(() => response({ success: true }, 201));

  await assert.rejects(
    h.client.create("Broken"),
    /manifest/
  );
});

test("a stale older page reloads once without retrying it",
  async () => {
  let olderReads = 0;
  const h = harness((url) => {
    if (url.includes("before=")) {
      olderReads += 1;
      return response(page(4));
    }
    if (url.split("?")[0].endsWith("/metadata")) {
      return response({ conversation: manifest(4, 3) });
    }
    return response(page(4));
  });
  const pageable = page(3);
  pageable.has_more = true;
  pageable.next_before = "00000001";
  h.applyAction({
    type: "loaded",
    conversation: manifest(3, 2),
    page: pageable,
  });

  await h.client.loadOlder();

  assert.equal(olderReads, 1);
  assert.equal(h.readState().conversation.branch_revision, 4);
  assert.equal(h.readState().loadingOlder, false);
});

test("an older-page catalog mismatch reloads the selected branch",
  async () => {
  let olderReads = 0;
  const changedManifest = Object.assign({}, manifest(3, 2), {
    catalog_revision: 2,
  });
  const changedPage = Object.assign({}, page(3), {
    catalog_revision: 2,
  });
  const h = harness((url) => {
    if (url.includes("before=")) {
      olderReads += 1;
      return response(changedPage);
    }
    if (url.split("?")[0].endsWith("/metadata")) {
      return response({ conversation: changedManifest });
    }
    return response(changedPage);
  });
  const pageable = page(3);
  pageable.has_more = true;
  pageable.next_before = "00000001";
  h.applyAction({
    type: "loaded",
    conversation: manifest(3, 2),
    page: pageable,
  });

  await h.client.loadOlder();

  assert.equal(olderReads, 1);
  assert.equal(h.readState().conversation.catalog_revision, 2);
  assert.equal(h.readState().selectedBranchId, BRANCH_ID);
});

test("branch catalog and fork methods carry both CAS revisions",
  async () => {
  const scenarios = [
    {
      suffix: "/edit-user/" + turnId(1),
      invoke(client) {
        return client.editUserFork({
          operationId: OPERATION_ID,
          userTurnId: turnId(1),
          text: "edited",
          modelId: "llada",
          inputMode: "chat",
          metadata: {},
        });
      },
      result: {
        user_turn: turn(1, 1, { text: "edited" }),
        assistant_turn: turn(2, 2),
      },
    },
    {
      suffix: "/delete-from-path/" + turnId(1),
      invoke(client) {
        return client.deleteFromPathFork({
          operationId: OPERATION_ID,
          userTurnId: turnId(1),
        });
      },
      result: {},
    },
    {
      suffix: "/retry-assistant/" + turnId(2),
      invoke(client) {
        return client.retryAssistantFork({
          operationId: OPERATION_ID,
          assistantTurnId: turnId(2),
          modelId: "llada",
          inputMode: "chat",
        });
      },
      result: {
        assistant_turn: turn(2, 2),
      },
    },
  ];

  for (const scenario of scenarios) {
    const calls = [];
    const forkManifest = Object.assign(
      {}, manifest(3, 2), { catalog_revision: 2 }
    );
    const h = harness((url, init) => {
      const route = url.split("?")[0];
      calls.push({ route, init });
      if (init.method === "POST") {
        return response(Object.assign({
          catalog: { catalog_revision: 2 },
          conversation: forkManifest,
          branch: {
            branch_id: BRANCH_ID,
            branch_revision: 3,
          },
        }, scenario.result), 201);
      }
      if (route.endsWith("/metadata")) {
        return response({ conversation: forkManifest });
      }
      return response(Object.assign(
        {}, page(3), { catalog_revision: 2 }
      ));
    });

    await scenario.invoke(h.client);

    const mutation = calls.find((call) =>
      call.init.method === "POST"
    );
    const body = JSON.parse(mutation.init.body);
    assert.ok(mutation.route.endsWith(scenario.suffix));
    assert.equal(body.branch_id, BRANCH_ID);
    assert.equal(body.branch_revision, 3);
    assert.equal(body.catalog_revision, 1);
    assert.equal(body.operation_id, OPERATION_ID);
  }
});

test("fork operation ids are strict lowercase hex", () => {
  const invalid = ["A".repeat(32), "a".repeat(31), "g".repeat(32)];
  for (const operationId of invalid) {
    let requests = 0;
    const h = harness(() => {
      requests += 1;
      return response({});
    });

    assert.throws(
      () => h.client.deleteFromPathFork({
        operationId,
        userTurnId: turnId(1),
      }),
      /operation id/
    );
    assert.equal(requests, 0);
  }
});

test("branch listing validates the bounded catalog", async () => {
  const h = harness((url) => response({
    schema_version: 2,
    conversation_id: ID,
    catalog_revision: 1,
    default_branch_id: BRANCH_ID,
    branch_ids: [BRANCH_ID],
    branches: [{
      branch_id: BRANCH_ID,
      branch_revision: 3,
    }],
  }));

  const body = await h.client.listBranches();

  assert.equal(body.default_branch_id, BRANCH_ID);
  assert.equal(body.branches.length, 1);
});

test("rapid branch selection keeps the newest append target",
  async () => {
  let releaseBranchB;
  const calls = [];
  const heldBranchB = new Promise((resolve) => {
    releaseBranchB = () => resolve(response({
      conversation: branchManifest(BRANCH_B, 3),
    }));
  });
  const h = harness((url, init) => {
    const parsed = new URL(url, "http://test");
    const branchId = parsed.searchParams.get("branch_id");
    calls.push({
      method: init.method,
      branchId,
      body: init.body ? JSON.parse(init.body) : null,
    });
    if (init.method === "POST") {
      const user = turn(3, 1, {
        branch_id: BRANCH_C,
        text: "on C",
      });
      const assistant = turn(4, 1, {
        branch_id: BRANCH_C,
        text: "",
        partial: true,
      });
      const conversation = Object.assign(
        {}, branchManifest(BRANCH_C, 4), {
          turn_count: 4,
          tail_turn_id: assistant.turn_id,
          tail_version: 1,
          pending_assistant_id: assistant.turn_id,
        }
      );
      return response({
        conversation,
        user_turn: user,
        assistant_turn: assistant,
      }, 201);
    }
    if (parsed.pathname.endsWith("/metadata")) {
      if (branchId === BRANCH_B) {
        return heldBranchB;
      }
      return response({
        conversation: branchManifest(branchId, 3),
      });
    }
    return response(branchPage(branchId, 3));
  });

  const selectB = h.client.selectBranch(BRANCH_B);
  const selectC = h.client.selectBranch(BRANCH_C);
  const append = h.client.appendUser({
    text: "on C",
    modelId: "llada",
    inputMode: "chat",
    metadata: {},
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(calls.length, 1);
  releaseBranchB();
  await Promise.all([selectB, selectC, append]);

  const post = calls.find((call) => call.method === "POST");
  assert.equal(post.body.branch_id, BRANCH_C);
  assert.equal(h.readState().selectedBranchId, BRANCH_C);
  assert.equal(h.readState().turns.at(-2).text, "on C");
});
