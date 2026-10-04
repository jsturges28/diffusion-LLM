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
    id: ID,
    title: "Test",
    revision,
    turn_count: 2,
    tail_turn_id: "00000002",
    tail_version: version,
    pending_assistant_id: null,
  };
}

function turn(index, version, overrides) {
  const assistant = index % 2 === 0;
  return Object.assign({
    turn_id: String(index).padStart(8, "0"),
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
    conversation_id: ID,
    revision,
    turns: [turn(1, 1), turn(2, revision - 1)],
    next_before: null,
    has_more: false,
  };
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
  const client = api.conversationClientCreate({
    request,
    readState: () => state,
    applyAction(action) {
      actions.push(action.type);
      state = api.conversationStateReduce(state, action);
    },
  });
  return { api, client, actions, readState: () => state };
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
    assistantTurnId: "00000002",
    text: "answer",
    partial: false,
    contextPack: {},
    metadata: { status: "completed" },
  });
  const link = h.client.linkRun({
    assistantTurnId: "00000002",
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
  assert.equal(calls[0].body.expected_revision, 3);
  assert.equal(calls[1].body.expected_revision, 4);
  assert.equal(h.readState().conversation.revision, 5);
});

test("a stale append reloads and is not retried", async () => {
  let posts = 0;
  const h = harness((url, init) => {
    if (init.method === "POST") {
      posts += 1;
      return response({
        error: "stale",
        reason: "revision_conflict",
        conversation_id: ID,
        expected_revision: 3,
        revision: 4,
      }, 409);
    }
    if (url.endsWith("/metadata")) {
      return response({ conversation: manifest(4, 3) });
    }
    return response(page(4));
  });

  await assert.rejects(
    h.client.appendUser({
      text: "next",
      modelId: "llada",
      inputMode: "chat",
      metadata: {},
    }),
    /stale/
  );

  assert.equal(posts, 1);
  assert.equal(h.readState().conversation.revision, 4);
  assert.deepEqual(h.actions, ["loaded"]);
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
    if (url.endsWith("/metadata")) {
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
      assistantTurnId: "00000002",
      text: "first",
      partial: true,
      contextPack: {},
      metadata: {},
    }),
    /offline/
  );
  await h.client.updateAssistant({
    assistantTurnId: "00000002",
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
