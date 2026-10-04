// Durable chat through the complete generator page.
//
// Strategy: back the shipped page with a tiny in-memory CAS API and
// the shared fake WebSocket. Passing proves first Send creates and
// reserves before generation, the wire receives bounded structured
// messages and durable ids, terminal text is committed, the next user
// freezes the old response without duplicating it, and a failed New
// Conversation never clears the workspace.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const ID_ONE = "a".repeat(32);
const ID_TWO = "b".repeat(32);
const MODEL = {
  id: "llada",
  display_name: "LLaDA",
  capabilities: {
    family: "diffusion",
    generation_shape: "iterative_canvas",
    input_mode: "chat",
    supported_devices: ["cuda"],
    unresolved_char: "\u2591",
    supports_resume: true,
  },
  param_specs: [],
  status: "active",
};
const MODELS = {
  models: [MODEL],
  active: MODEL.id,
  active_device: "cuda",
  active_tokenizer: { name: "test" },
  active_context_length: 4096,
  default: MODEL.id,
  gpu_name: "Test GPU",
};

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function reply(body, status) {
  return Promise.resolve(response(body, status));
}

function response(body, status) {
  const code = status || 200;
  return {
    ok: code >= 200 && code < 300,
    status: code,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

function turn(index, role, text, version) {
  const assistant = role === "assistant";
  return {
    schema_version: 1,
    conversation_id: ID_ONE,
    conversation_revision: 1,
    turn_id: String(index).padStart(8, "0"),
    index,
    version,
    role,
    created_at: "2026-10-04T00:00:00Z",
    updated_at: "2026-10-04T00:00:00Z",
    text,
    partial: assistant,
    model_id: assistant ? MODEL.id : null,
    input_mode: assistant ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  };
}

function conversationApi() {
  const state = {
    id: null,
    revision: 0,
    turns: [],
    failCreate: false,
    dropNextUpdateReply: false,
    dropNextLinkReply: false,
    holdUpdateReplies: false,
    holdLinkReplies: false,
    failNextSave: false,
    updateWaiters: [],
    linkWaiters: [],
    saveRevision: 0,
    calls: [],
  };

  function manifest() {
    const tail = state.turns.at(-1) || null;
    const pending = tail && tail.role === "assistant"
      && tail.version === 1
      ? tail.turn_id
      : null;
    return {
      schema_version: 1,
      id: state.id,
      title: "New conversation",
      revision: state.revision,
      created_at: "2026-10-04T00:00:00Z",
      updated_at: "2026-10-04T00:00:00Z",
      turn_count: state.turns.length,
      tail_role: tail ? tail.role : null,
      tail_turn_id: tail ? tail.turn_id : null,
      tail_version: tail ? tail.version : null,
      pending_assistant_id: pending,
    };
  }

  function create() {
    if (state.failCreate) {
      return reply({
        error: "disk unavailable",
        reason: "store_error",
      }, 500);
    }
    state.id = state.id === null ? ID_ONE : ID_TWO;
    state.revision = 1;
    state.turns = [];
    return reply({ conversation: manifest() }, 201);
  }

  function append(body) {
    assert.equal(body.expected_revision, state.revision);
    const userIndex = state.turns.length + 1;
    const user = turn(userIndex, "user", body.text, 1);
    const assistant = turn(userIndex + 1, "assistant", "", 1);
    state.revision += 1;
    user.conversation_revision = state.revision;
    assistant.conversation_revision = state.revision;
    state.turns.push(user, assistant);
    return reply({
      conversation: manifest(),
      user_turn: user,
      assistant_turn: assistant,
    }, 201);
  }

  function update(body) {
    assert.equal(body.expected_revision, state.revision);
    const assistant = state.turns.at(-1);
    state.revision += 1;
    assistant.version += 1;
    assistant.conversation_revision = state.revision;
    assistant.text = body.text;
    assistant.partial = body.partial;
    assistant.context_pack = body.context_pack;
    assistant.metadata = body.metadata;
    assistant.run_link = null;
    const result = {
      conversation: manifest(),
      turn: assistant,
    };
    if (state.dropNextUpdateReply) {
      state.dropNextUpdateReply = false;
      return Promise.reject(new Error("reply lost"));
    }
    if (state.holdUpdateReplies) {
      return heldReply(result, state.updateWaiters);
    }
    return reply(result);
  }

  function link(body) {
    assert.equal(body.expected_revision, state.revision);
    const assistant = state.turns.at(-1);
    state.revision += 1;
    assistant.version += 1;
    assistant.conversation_revision = state.revision;
    assistant.run_link = {
      run_id: body.run_id,
      revision: body.run_revision,
    };
    const result = {
      conversation: manifest(),
      turn: assistant,
    };
    if (state.dropNextLinkReply) {
      state.dropNextLinkReply = false;
      return Promise.reject(new Error("link reply lost"));
    }
    if (state.holdLinkReplies) {
      return heldReply(result, state.linkWaiters);
    }
    return reply(result);
  }

  function heldReply(body, waiters) {
    const snapshot = JSON.parse(JSON.stringify(body));
    return new Promise((resolve) => {
      waiters.push(() => resolve(response(snapshot)));
    });
  }

  function release(waiters) {
    const pending = waiters.splice(0, waiters.length);
    for (const resolve of pending) {
      resolve();
    }
  }

  function reviseExternally(text) {
    const assistant = state.turns.at(-1);
    state.revision += 1;
    assistant.version += 1;
    assistant.conversation_revision = state.revision;
    assistant.text = text;
    assistant.partial = false;
    assistant.metadata = { status: "completed" };
    assistant.run_link = null;
  }

  function fetchImpl(url, init) {
    const path = String(url).split("?")[0];
    const method = (init && init.method) || "GET";
    const body = init && init.body ? JSON.parse(init.body) : null;
    state.calls.push({ path, method, body });
    if (path === "/api/models") {
      return reply(MODELS);
    }
    if (path === "/api/save" && method === "POST") {
      if (state.failNextSave) {
        state.failNextSave = false;
        return reply({
          success: false,
          message: "disk full",
        });
      }
      state.saveRevision += 1;
      return reply({
        success: true,
        path: "results/run-one",
        run_id: "run-one",
        revision: state.saveRevision,
      });
    }
    if (path.startsWith("/api/ui-state/")) {
      return reply({ success: true });
    }
    if (path === "/api/conversations" && method === "POST") {
      return create();
    }
    if (path.endsWith("/metadata") && method === "GET") {
      return reply({ conversation: manifest() });
    }
    if (path.endsWith("/turns") && method === "GET") {
      return reply({
        conversation_id: state.id,
        revision: state.revision,
        turns: state.turns,
        next_before: null,
        has_more: false,
      });
    }
    if (path.endsWith("/turns") && method === "POST") {
      return append(body);
    }
    if (path.endsWith("/run") && method === "PUT") {
      return link(body);
    }
    if (path.includes("/turns/") && method === "PUT") {
      return update(body);
    }
    return reply({});
  }

  return {
    state,
    fetchImpl,
    releaseUpdates: () => release(state.updateWaiters),
    releaseLinks: () => release(state.linkWaiters),
    reviseExternally,
  };
}

async function pageWithApi(api) {
  const mark = FakeSocket.opened.length;
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: api.fetchImpl,
    conversationApi: false,
    bootState: { ui_state: {}, models: MODELS },
  });
  await tick();
  await tick();
  return {
    page,
    context: page.context,
    socket: FakeSocket.opened[mark],
  };
}

function frame(index, text) {
  return {
    type: "frame",
    index,
    text,
    tokens: [{
      t: text,
      m: false,
      id: index + 1,
      c: 0.8,
    }],
    canvas_index: 0,
    elapsed: (index + 1) / 10,
    revealed: [0],
    total_steps: 3,
  };
}

async function finishRun(run, prompt, answer) {
  run.page.registry.get("prompt-input").value = prompt;
  assert.equal(await run.context.startGeneration(), true);
  run.context.handleFrame(frame(0, answer + " 0"));
  run.context.handleFrame(frame(1, answer + " 1"));
  run.context.handleFrame(frame(2, answer));
  run.context.handleDone({
    type: "done",
    final_text: answer,
    run_token: "nonce:1",
  });
  await run.context.conversationClient.flush();
}

test("boot restores the active conversation and latest page",
  async () => {
  const api = conversationApi();
  api.state.id = ID_ONE;
  api.state.revision = 3;
  const user = turn(1, "user", "Restored question", 1);
  const assistant = turn(2, "assistant", "Restored answer", 2);
  assistant.partial = false;
  api.state.turns = [user, assistant];
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: api.fetchImpl,
    conversationApi: false,
    bootState: {
      ui_state: {
        diffusion_active_conversation: JSON.stringify({
          id: ID_ONE,
          revision: 3,
        }),
      },
      models: MODELS,
    },
  });
  await tick();
  await tick();

  assert.equal(page.context.conversationState.turns.length, 2);
  assert.equal(
    page.context.conversationState.conversation.revision,
    3
  );
  assert.equal(
    page.registry.get("conversation-turns").children.length,
    2
  );
  assert.equal(
    page.registry.get("conversation-turns").children.at(-1)
      .querySelector(".conversation-turn-text").textContent,
    "Restored answer"
  );
  });

test("Send reserves, generates, completes, then appends once",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  const prompt = run.page.registry.get("prompt-input");
  prompt.value = "First question";

  assert.equal(await run.context.startGeneration(), true);
  const first = JSON.parse(run.socket.sent.at(-1));
  assert.equal(first.type, "generate");
  assert.equal(first.prompt, undefined);
  assert.equal(first.messages.length, 1);
  assert.equal(first.messages[0].content, "First question");
  assert.equal(first.conversation_id, ID_ONE);
  assert.equal(first.conversation_revision, 2);
  assert.equal(first.assistant_turn_id, "00000002");
  assert.equal(first.candidate_turn_offset, 0);

  run.context.handleFrame({
    type: "frame",
    index: 0,
    text: "First answer",
    tokens: [{
      t: "First answer",
      m: false,
      id: 1,
      c: 0.8,
    }],
    canvas_index: 0,
    elapsed: 0.1,
    revealed: [0],
  });
  run.context.handleDone({
    type: "done",
    final_text: "First answer",
    provenance: {
      model_id: MODEL.id,
      context_pack: {
        included_turn_ids: ["00000001"],
        first_included_index: 0,
        omitted_turn_count: 0,
        prompt_token_count: 8,
        output_reserve: 64,
        requested_total_budget: 4096,
        effective_total_budget: 4096,
      },
    },
    run_token: "nonce:1",
  });
  await run.context.conversationClient.flush();
  assert.equal(api.state.turns.at(-1).text, "First answer");
  assert.equal(api.state.turns.at(-1).partial, false);
  assert.equal(
    run.page.registry.get("conversation-turns").children.length,
    1
  );

  run.context.generatorCandidatesRequestProbe({
    position: 0,
    tokenId: 7,
    requestId: 1,
  });
  run.context.generatorEditRequestRewind({
    runToken: "nonce:1",
  });
  run.context.generatorEditRequestResume({
    frameIndex: 0,
    remaskPositions: [0],
    targetFrame: null,
    continueRun: false,
    runToken: "nonce:1",
  });
  run.context.generatorEditRequestSubstitute({
    position: 0,
    tokenId: 7,
    typedText: null,
    runToken: "nonce:1",
  });
  const stateful = run.socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) =>
      ["probe", "rewind", "resume", "substitute"]
        .includes(message.type)
    );
  assert.equal(stateful.length, 4);
  for (const message of stateful) {
    assert.equal(message.conversation_id, ID_ONE);
    assert.equal(message.conversation_revision, 3);
    assert.equal(message.assistant_turn_id, "00000002");
  }

  prompt.value = "Second question";
  assert.equal(await run.context.startGeneration(), true);
  const second = JSON.parse(run.socket.sent.at(-1));
  assert.deepEqual(
    second.messages.map((message) => message.content),
    ["First question", "First answer", "Second question"]
  );
  assert.equal(second.assistant_turn_id, "00000004");
  assert.equal(
    api.state.turns.filter((item) =>
      item.text === "Second question"
    ).length,
    1
  );
  assert.equal(run.context.generatorRun.frameCount(), 0);
  });

test("a failed generation retries its reserved assistant",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  run.page.registry.get("prompt-input").value = "Retry me";

  assert.equal(await run.context.startGeneration(), true);
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "generation_failed",
    message: "model reloaded",
  });
  assert.equal(await run.context.startGeneration(), true);

  const appends = api.state.calls.filter((call) =>
    call.method === "POST" && call.path.endsWith("/turns")
  );
  const generations = run.socket.sent.filter((raw) =>
    JSON.parse(raw).type === "generate"
  );
  assert.equal(appends.length, 1);
  assert.equal(generations.length, 2);
  assert.equal(
    JSON.parse(generations[1]).messages.at(-1).content,
    "Retry me"
  );
  });

test("Enter cannot bypass a pending conversation commit",
  async () => {
  const api = conversationApi();
  api.state.holdUpdateReplies = true;
  const run = await pageWithApi(api);
  const prompt = run.page.registry.get("prompt-input");
  prompt.value = "Only once";

  assert.equal(await run.context.startGeneration(), true);
  run.context.handleFrame(frame(0, "answer"));
  run.context.handleDone({
    type: "done",
    final_text: "answer",
    run_token: "nonce:1",
  });
  const readsBefore = api.state.calls.filter((call) =>
    call.method === "GET"
    && call.path.startsWith("/api/conversations/")
  ).length;
  prompt.dispatch("keydown", {
    key: "Enter",
    shiftKey: false,
    preventDefault() {},
  });
  await tick();

  const generations = run.socket.sent.filter((raw) =>
    JSON.parse(raw).type === "generate"
  );
  const readsAfter = api.state.calls.filter((call) =>
    call.method === "GET"
    && call.path.startsWith("/api/conversations/")
  ).length;
  assert.equal(generations.length, 1);
  assert.equal(readsAfter, readsBefore);

  api.releaseUpdates();
  await run.context.conversationClient.flush();
  });

test("a lost completion reply reconciles before retry",
  async () => {
  const api = conversationApi();
  api.state.dropNextUpdateReply = true;
  const run = await pageWithApi(api);
  const prompt = run.page.registry.get("prompt-input");
  prompt.value = "First";

  assert.equal(await run.context.startGeneration(), true);
  run.context.handleFrame(frame(0, "durable answer"));
  run.context.handleDone({
    type: "done",
    final_text: "durable answer",
    run_token: "nonce:1",
  });
  await run.context.conversationClient.flush();
  await run.context.conversationCompletion;

  assert.equal(
    run.context.conversationState.conversation
      .pending_assistant_id,
    null
  );
  assert.equal(
    run.context.conversationState.turns.at(-1).text,
    "durable answer"
  );
  prompt.value = "Second";
  assert.equal(await run.context.startGeneration(), true);
  const generations = run.socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) => message.type === "generate");
  assert.equal(generations.length, 2);
  assert.equal(generations[1].messages.at(-1).content, "Second");
  });

test("edit drafts publish only after Confirm", async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Original answer");

  run.context.generatorEdit.enterFrames();
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "draft 1"));
  run.context.handleFrame(frame(2, "Draft answer"));
  run.context.handleDone({
    type: "done",
    final_text: "Draft answer",
    run_token: "nonce:1",
  });

  assert.equal(api.state.turns.at(-1).text, "Original answer");
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    "review"
  );
  run.context.generatorEdit.retry();
  assert.equal(
    run.context.generatorRun.finalText(),
    "Original answer"
  );
  assert.equal(api.state.turns.at(-1).text, "Original answer");

  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "exit 1"));
  run.context.handleFrame(frame(2, "Exit draft"));
  run.context.handleDone({
    type: "done",
    final_text: "Exit draft",
    run_token: "nonce:1",
  });
  assert.equal(run.context.generatorEdit.exit(), true);
  assert.equal(
    run.context.generatorRun.finalText(),
    "Original answer"
  );
  assert.equal(api.state.turns.at(-1).text, "Original answer");

  run.context.generatorEdit.enterFrames();
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "confirmed 1"));
  run.context.handleFrame(frame(2, "Confirmed answer"));
  run.context.handleDone({
    type: "done",
    final_text: "Confirmed answer",
    run_token: "nonce:1",
  });
  assert.equal(
    await run.context.generatorEdit.confirm(),
    true
  );

  assert.equal(api.state.turns.at(-1).text, "Confirmed answer");
  assert.deepEqual(api.state.turns.at(-1).run_link, {
    run_id: "run-one",
    revision: 1,
  });
  });

test("reload during edit review rolls back to the durable run",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Original answer");
  const originalSnapshot = run.context.sessionStorage.getItem(
    run.context.SESSION_KEY
  );

  run.context.generatorEdit.enterFrames();
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "draft 1"));
  run.context.handleFrame(frame(2, "Draft answer"));
  run.context.handleDone({
    type: "done",
    final_text: "Draft answer",
    run_token: "nonce:1",
  });

  assert.equal(
    run.context.sessionStorage.getItem(run.context.SESSION_KEY),
    originalSnapshot
  );
  run.context.generatorEdit.reset();
  run.context.generatorRun.reset();
  assert.equal(run.context.restoreSessionState(), true);
  assert.equal(
    run.context.generatorRun.finalText(),
    "Original answer"
  );
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    null
  );
  assert.equal(api.state.turns.at(-1).text, "Original answer");
  });

test("a failed Confirm save never offers an unsafe Retry",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Original answer");
  run.context.generatorEdit.enterFrames();
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "edited 1"));
  run.context.handleFrame(frame(2, "Edited answer"));
  run.context.handleDone({
    type: "done",
    final_text: "Edited answer",
    run_token: "nonce:1",
  });
  api.state.failNextSave = true;

  assert.equal(
    await run.context.generatorEdit.confirm(),
    false
  );
  assert.equal(api.state.turns.at(-1).text, "Edited answer");
  assert.equal(api.state.turns.at(-1).run_link, null);
  assert.equal(
    run.context.generatorEdit.phaseState().mode,
    null
  );
  assert.equal(
    run.page.registry.get("btn-save").disabled,
    false
  );
  });

test("a stale same-tail snapshot yields to durable text",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Old answer");
  run.context.saveSessionState();
  const stale = run.context.sessionStorage.getItem(
    run.context.SESSION_KEY
  );
  api.reviseExternally("New durable answer");
  await run.context.conversationClient.restore(ID_ONE);
  let turns = run.page.registry.get("conversation-turns");
  let tail = turns.children.at(-1);
  assert.equal(
    tail.querySelector(".conversation-turn-text").textContent,
    "New durable answer"
  );
  assert.equal(run.context.activeRunCanEdit(), false);

  run.context.generatorEdit.reset();
  run.context.generatorRun.reset();
  run.context.sessionStorage.setItem(
    run.context.SESSION_KEY, stale
  );

  assert.equal(run.context.restoreSessionState(), false);
  turns = run.page.registry.get("conversation-turns");
  tail = turns.children.at(-1);
  assert.equal(
    tail.querySelector(".conversation-turn-text").textContent,
    "New durable answer"
  );
  assert.equal(run.context.activeRunCanEdit(), false);
  });

test("rescue reload waits for the durable run link",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Answer");
  api.state.holdLinkReplies = true;
  let reloads = 0;
  run.context.location.reload = () => {
    reloads += 1;
  };

  run.context.rescueRunThenReload();
  await tick();
  await tick();
  assert.equal(reloads, 0);
  run.page.registry.get("prompt-input").value = "Do not start";
  assert.equal(await run.context.startGeneration(), false);

  api.releaseLinks();
  await tick();
  await tick();
  assert.equal(reloads, 1);
  assert.deepEqual(api.state.turns.at(-1).run_link, {
    run_id: "run-one",
    revision: 1,
  });
  });

test("a lost run-link reply reconciles as saved", async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Answer");
  api.state.dropNextLinkReply = true;

  assert.equal(await run.context.saveRun(), true);
  const link = run.context.conversationState.turns.at(-1).run_link;
  assert.equal(link.run_id, "run-one");
  assert.equal(link.revision, 1);
  assert.equal(run.context.generatorRun.saved(), true);
  });

test("rescue reload discards an unconfirmed edit", async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  await finishRun(run, "Question", "Original answer");
  run.context.generatorEdit.enterFrames();
  run.context.generatorEdit.selectFrame();
  run.context.generatorEdit.togglePosition(0);
  run.context.generatorEdit.lockSelection();
  run.context.generatorEdit.resumeToEnd();
  run.context.handleFrame(frame(1, "draft 1"));
  run.context.handleFrame(frame(2, "Draft answer"));
  run.context.handleDone({
    type: "done",
    final_text: "Draft answer",
    run_token: "nonce:1",
  });
  let reloads = 0;
  run.context.location.reload = () => {
    reloads += 1;
  };

  run.context.rescueRunThenReload();
  await tick();
  await tick();

  const save = api.state.calls.find((call) =>
    call.path === "/api/save"
  );
  assert.ok(save);
  assert.equal(save.body.final_text, "Original answer");
  assert.equal("remask_edits" in save.body, false);
  assert.equal(api.state.turns.at(-1).text, "Original answer");
  assert.equal(reloads, 1);
  });

test("failed New Conversation preserves the composer and run",
  async () => {
  const api = conversationApi();
  const run = await pageWithApi(api);
  const prompt = run.page.registry.get("prompt-input");
  prompt.value = "Keep me";
  assert.equal(await run.context.startGeneration(), true);
  run.context.handleFrame({
    type: "frame",
    index: 0,
    text: "answer",
    tokens: [{ t: "answer", m: false, id: 1, c: 0.8 }],
    canvas_index: 0,
    elapsed: 0.1,
    revealed: [0],
  });
  run.context.handleDone({
    type: "done",
    final_text: "answer",
    run_token: "nonce:1",
  });
  await run.context.conversationClient.flush();
  prompt.value = "unsent draft";
  api.state.failCreate = true;

  assert.equal(await run.context.startNewRun(), false);
  assert.equal(prompt.value, "unsent draft");
  assert.equal(run.context.generatorRun.frameCount(), 1);

  api.state.failCreate = false;
  assert.equal(await run.context.startNewRun(), true);
  assert.equal(prompt.value, "");
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(run.context.conversationState.conversation.id, ID_TWO);
  });
