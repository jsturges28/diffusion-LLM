// Browser-to-API contract for schema-v2 conversation identity.
//
// Strategy: run the shipped generator page against a strict in-memory
// branch API, complete and save one opaque-id response, then select a
// sibling and force both branch and catalog conflicts. Passing proves
// every transport carries the explicit durable location, selection is
// read-only and clears incompatible run state, conflicts reload the
// selected branch without replay, and numeric snapshots still
// migrate.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, FakeSocket } = require("./dom_stub.js");

class OpenSocket extends FakeSocket {}
OpenSocket.OPEN = 1;

const CONVERSATION_ID = "a".repeat(32);
const BRANCH_A = "b_" + "b".repeat(32);
const BRANCH_B = "b_" + "c".repeat(32);
const USER_A =
  "t_" + "b".repeat(32) + "_00000001_3439fbc5590d46cb";
const ASSISTANT_A =
  "t_" + "b".repeat(32) + "_00000002_77a21ee4f3d34877";
const USER_B =
  "t_" + "c".repeat(32) + "_00000001_425e5371282a5cc5";
const ASSISTANT_B =
  "t_" + "c".repeat(32) + "_00000002_2b715ccacb6d6190";
const BRANCH_C = "b_" + "d".repeat(32);
const BRANCH_D = "b_" + "e".repeat(32);
const BRANCH_E = "b_" + "f".repeat(32);
const USER_C =
  "t_" + "d".repeat(32) + "_00000001_62e3f5ae88305e8e";
const ASSISTANT_C =
  "t_" + "d".repeat(32) + "_00000002_6c0277ad4d19d8ab";
const USER_D =
  "t_" + "e".repeat(32) + "_00000001_34539a25cb885b4f";
const ASSISTANT_D =
  "t_" + "e".repeat(32) + "_00000002_ab45a84c532dc218";

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
  param_specs: [
    {
      name: "temperature",
      label: "Temperature",
      type: "float",
      default: 0.7,
      group: "sampling",
      prominence: "primary",
      recommended: [0, 1],
      experimental: [0, 2],
      step: 0.1,
    },
    {
      name: "strategy",
      label: "Strategy",
      type: "select",
      default: "low_confidence",
      group: "sampling",
      prominence: "secondary",
      options: ["low_confidence", "random"],
    },
  ],
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

function response(body, status) {
  const code = status || 200;
  return Promise.resolve({
    ok: code >= 200 && code < 300,
    status: code,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

function turn(branchId, turnId, index, role, text, version) {
  const assistant = role === "assistant";
  return {
    schema_version: 2,
    conversation_id: CONVERSATION_ID,
    branch_id: branchId,
    branch_revision: 1,
    turn_id: turnId,
    index,
    version,
    role,
    text,
    partial: assistant && version === 1,
    model_id: assistant ? MODEL.id : null,
    input_mode: assistant ? "chat" : null,
    context_pack: {},
    metadata: {},
    run_link: null,
  };
}

class BranchApi {
  constructor() {
    this.branches = {
      [BRANCH_A]: {
        revision: 1,
        turns: [],
      },
      [BRANCH_B]: {
        revision: 3,
        turns: [
          turn(
            BRANCH_B,
            USER_B,
            1,
            "user",
            "Other question",
            1
          ),
          turn(
            BRANCH_B,
            ASSISTANT_B,
            2,
            "assistant",
            "Other answer",
            2
          ),
        ],
      },
    };
    this.state = {
      calls: [],
      branchConflict: false,
      catalogConflict: false,
      catalogRevision: 2,
      branchIds: [BRANCH_A, BRANCH_B],
      forkGate: null,
      releaseFork: null,
      dropForkReply: null,
      rejectFork: null,
      receipts: new Map(),
    };
    this.fetchImpl = this.fetchImpl.bind(this);
  }

  manifest(branchId) {
    const branch = this.branches[branchId];
    const tail = branch.turns.at(-1) || null;
    const pending = tail
      && tail.role === "assistant"
      && tail.version === 1
      ? tail.turn_id
      : null;
    return {
      schema_version: 2,
      id: CONVERSATION_ID,
      title: "Contract",
      branch_id: branchId,
      branch_revision: branch.revision,
      revision: branch.revision,
      catalog_revision: this.state.catalogRevision,
      default_branch_id: BRANCH_A,
      turn_count: branch.turns.length,
      tail_turn_id: tail ? tail.turn_id : null,
      tail_version: tail ? tail.version : null,
      pending_assistant_id: pending,
    };
  }

  page(branchId) {
    const branch = this.branches[branchId];
    let branchPoints = [];
    if (branchId === BRANCH_C) {
      branchPoints = [{
        turn_index: 2,
        source_branch_id: BRANCH_A,
        selected_branch_id: BRANCH_C,
        branch_ids: [BRANCH_A, BRANCH_C],
        deleted_branch_ids: [],
      }];
    } else if (branchId === BRANCH_D) {
      branchPoints = [{
        turn_index: 1,
        source_branch_id: BRANCH_A,
        selected_branch_id: BRANCH_D,
        branch_ids: [BRANCH_A, BRANCH_D],
        deleted_branch_ids: [],
      }];
    } else if (branchId === BRANCH_E) {
      branchPoints = [{
        turn_index: 1,
        source_branch_id: BRANCH_A,
        selected_branch_id: BRANCH_E,
        branch_ids: [BRANCH_A, BRANCH_E],
        deleted_branch_ids: [BRANCH_E],
      }];
    }
    return {
      schema_version: 2,
      conversation_id: CONVERSATION_ID,
      branch_id: branchId,
      branch_revision: branch.revision,
      revision: branch.revision,
      catalog_revision: this.state.catalogRevision,
      default_branch_id: BRANCH_A,
      turns: branch.turns,
      next_before: null,
      has_more: false,
      branch_points: branchPoints,
    };
  }

  append(body) {
    const branch = this.branches[body.branch_id];
    assert.equal(body.branch_revision, branch.revision);
    branch.revision += 1;
    const user = turn(
      BRANCH_A, USER_A, 1, "user", body.text, 1
    );
    const assistant = turn(
      BRANCH_A, ASSISTANT_A, 2, "assistant", "", 1
    );
    branch.turns = [user, assistant];
    return response({
      conversation: this.manifest(BRANCH_A),
      user_turn: user,
      assistant_turn: assistant,
    }, 201);
  }

  update(body) {
    const branch = this.branches[body.branch_id];
    if (this.state.branchConflict) {
      this.state.branchConflict = false;
      return response({
        error: "stale branch",
        reason: "branch_revision_conflict",
        conversation_id: CONVERSATION_ID,
        branch_id: body.branch_id,
        branch_revision: branch.revision,
      }, 409);
    }
    assert.equal(body.branch_revision, branch.revision);
    branch.revision += 1;
    const assistant = branch.turns.at(-1);
    assistant.version += 1;
    assistant.text = body.text;
    assistant.partial = body.partial;
    assistant.context_pack = body.context_pack;
    assistant.metadata = body.metadata;
    return response({
      conversation: this.manifest(body.branch_id),
      turn: assistant,
    });
  }

  link(body) {
    const branch = this.branches[body.branch_id];
    assert.equal(body.branch_revision, branch.revision);
    const assistant = branch.turns.at(-1);
    assert.equal(body.assistant_turn_index, assistant.index);
    assert.equal(body.assistant_turn_version, assistant.version);
    branch.revision += 1;
    assistant.version += 1;
    assistant.run_link = {
      run_id: body.run_id,
      revision: body.run_revision,
    };
    return response({
      conversation: this.manifest(body.branch_id),
      turn: assistant,
    });
  }

  branchesBody() {
    return {
      schema_version: 2,
      conversation_id: CONVERSATION_ID,
      catalog_revision: this.state.catalogRevision,
      default_branch_id: BRANCH_A,
      branch_ids: this.state.branchIds.slice(),
      branches: this.state.branchIds.map((branchId) => ({
        branch_id: branchId,
        branch_revision: this.branches[branchId].revision,
      })),
    };
  }

  forkPrelude(body, kind) {
    assert.match(body.operation_id, /^[0-9a-f]{32}$/);
    const receipt = this.state.receipts.get(body.operation_id);
    if (receipt) {
      assert.equal(receipt.kind, kind);
      return response(receipt.payload, 201);
    }
    if (this.state.rejectFork === kind) {
      this.state.rejectFork = null;
      return response({
        error: kind + " rejected",
        reason: "store_error",
      }, 503);
    }
    return null;
  }

  forkCommit(body, kind, payload) {
    const saved = JSON.parse(JSON.stringify(payload));
    this.state.receipts.set(body.operation_id, {
      kind,
      payload: saved,
    });
    if (this.state.dropForkReply === kind) {
      this.state.dropForkReply = null;
      return Promise.reject(new Error(kind + " reply lost"));
    }
    return response(saved, 201);
  }

  forkRetry(body, assistantTurnId) {
    const replay = this.forkPrelude(body, "retry");
    if (replay !== null) {
      return replay;
    }
    const source = this.branches[body.branch_id];
    assert.equal(body.branch_revision, source.revision);
    assert.equal(
      body.catalog_revision, this.state.catalogRevision
    );
    assert.equal(
      source.turns.some((item) =>
        item.turn_id === assistantTurnId
      ),
      true
    );
    const assistant = turn(
      BRANCH_C, ASSISTANT_C, 2, "assistant", "", 1
    );
    this.branches[BRANCH_C] = {
      revision: 1,
      turns: [source.turns[0], assistant],
    };
    this.state.branchIds.push(BRANCH_C);
    this.state.catalogRevision += 1;
    return this.forkCommit(body, "retry", {
      catalog: this.branchesBody(),
      conversation: this.manifest(BRANCH_C),
      branch: {
        branch_id: BRANCH_C,
        branch_revision: 1,
      },
      source_branch_id: body.branch_id,
      retried_assistant_turn_id: assistantTurnId,
      assistant_turn: assistant,
    });
  }

  forkEdit(body, userTurnId) {
    const replay = this.forkPrelude(body, "edit");
    if (replay !== null) {
      return replay;
    }
    const source = this.branches[body.branch_id];
    assert.equal(body.branch_revision, source.revision);
    assert.equal(
      body.catalog_revision, this.state.catalogRevision
    );
    assert.equal(
      source.turns.some((item) => item.turn_id === userTurnId),
      true
    );
    const user = turn(
      BRANCH_D, USER_D, 1, "user", body.text, 1
    );
    const assistant = turn(
      BRANCH_D, ASSISTANT_D, 2, "assistant", "", 1
    );
    this.branches[BRANCH_D] = {
      revision: 1,
      turns: [user, assistant],
    };
    this.state.branchIds.push(BRANCH_D);
    this.state.catalogRevision += 1;
    return this.forkCommit(body, "edit", {
      catalog: this.branchesBody(),
      conversation: this.manifest(BRANCH_D),
      branch: {
        branch_id: BRANCH_D,
        branch_revision: 1,
      },
      source_branch_id: body.branch_id,
      replaced_user_turn_id: userTurnId,
      user_turn: user,
      assistant_turn: assistant,
    });
  }

  forkDelete(body, userTurnId) {
    const replay = this.forkPrelude(body, "delete");
    if (replay !== null) {
      return replay;
    }
    const source = this.branches[body.branch_id];
    assert.equal(body.branch_revision, source.revision);
    assert.equal(
      body.catalog_revision, this.state.catalogRevision
    );
    assert.equal(
      source.turns.some((item) => item.turn_id === userTurnId),
      true
    );
    this.branches[BRANCH_E] = {
      revision: 1,
      turns: [],
    };
    this.state.branchIds.push(BRANCH_E);
    this.state.catalogRevision += 1;
    return this.forkCommit(body, "delete", {
      catalog: this.branchesBody(),
      conversation: this.manifest(BRANCH_E),
      branch: {
        branch_id: BRANCH_E,
        branch_revision: 1,
      },
      source_branch_id: body.branch_id,
      deleted_user_turn_id: userTurnId,
      removed_turn_count: source.turns.length,
    });
  }

  fetchImpl(url, init) {
    const parsed = new URL(String(url), "http://test");
    const path = parsed.pathname;
    const method = (init && init.method) || "GET";
    const body = init && init.body ? JSON.parse(init.body) : {};
    this.state.calls.push({
      path,
      method,
      branch: parsed.searchParams.get("branch_id"),
      body,
    });
    if (path === "/api/models") {
      return response(MODELS);
    }
    if (path.startsWith("/api/ui-state/")) {
      return response({ success: true });
    }
    if (path === "/api/conversations" && method === "POST") {
      return response({
        conversation: this.manifest(BRANCH_A),
      }, 201);
    }
    if (path.endsWith("/branches") && method === "GET") {
      return response(this.branchesBody());
    }
    if (
      path.includes("/branches/retry-assistant/")
      && method === "POST"
      && this.state.catalogConflict
    ) {
      this.state.catalogConflict = false;
      return response({
        error: "stale catalog",
        reason: "catalog_revision_conflict",
        conversation_id: CONVERSATION_ID,
        catalog_revision: 2,
      }, 409);
    }
    if (
      path.includes("/branches/retry-assistant/")
      && method === "POST"
    ) {
      if (this.state.forkGate !== null) {
        const gate = this.state.forkGate;
        this.state.forkGate = null;
        return gate.then(() => this.forkRetry(
          body, decodeURIComponent(path.split("/").at(-1))
        ));
      }
      return this.forkRetry(
        body, decodeURIComponent(path.split("/").at(-1))
      );
    }
    if (
      path.includes("/branches/edit-user/")
      && method === "POST"
    ) {
      if (this.state.forkGate !== null) {
        const gate = this.state.forkGate;
        this.state.forkGate = null;
        return gate.then(() => this.forkEdit(
          body, decodeURIComponent(path.split("/").at(-1))
        ));
      }
      return this.forkEdit(
        body, decodeURIComponent(path.split("/").at(-1))
      );
    }
    if (
      path.includes("/branches/delete-from-path/")
      && method === "POST"
    ) {
      return this.forkDelete(
        body, decodeURIComponent(path.split("/").at(-1))
      );
    }
    if (path.endsWith("/metadata") && method === "GET") {
      const branchId =
        parsed.searchParams.get("branch_id") || BRANCH_A;
      return response({ conversation: this.manifest(branchId) });
    }
    if (path.endsWith("/turns") && method === "GET") {
      return response(
        this.page(parsed.searchParams.get("branch_id"))
      );
    }
    if (path.endsWith("/turns") && method === "POST") {
      return this.append(body);
    }
    if (path.endsWith("/run") && method === "PUT") {
      return this.link(body);
    }
    if (path.includes("/turns/") && method === "PUT") {
      return this.update(body);
    }
    if (path === "/api/save" && method === "POST") {
      return response({
        success: true,
        path: "results/contract-run",
        run_id: "contract-run",
        revision: 1,
      });
    }
    return response({});
  }

  conflictBranch() {
    this.state.branchConflict = true;
  }

  conflictCatalog() {
    this.state.catalogConflict = true;
  }

  holdNextFork() {
    this.state.forkGate = new Promise((resolve) => {
      this.state.releaseFork = resolve;
    });
  }

  releaseFork() {
    const release = this.state.releaseFork;
    assert.equal(typeof release, "function");
    this.state.releaseFork = null;
    release();
  }

  dropNextForkReply(kind) {
    this.state.dropForkReply = kind;
  }

  rejectNextFork(kind) {
    this.state.rejectFork = kind;
  }
}

function branchApi() {
  return new BranchApi();
}

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function frame(index, text) {
  return {
    type: "frame",
    index,
    text,
    tokens: [{ t: text, m: false, id: index + 1, c: 0.8 }],
    canvas_index: 0,
    elapsed: (index + 1) / 10,
    revealed: [0],
    total_steps: 2,
  };
}

function runProvenance() {
  return {
    model_id: MODEL.id,
    device: "cuda",
    context_pack: {
      included_turn_ids: [USER_A],
      first_included_index: 0,
      omitted_turn_count: 0,
      prompt_token_count: 4,
      output_reserve: 64,
      requested_total_budget: 4096,
      effective_total_budget: 4096,
      conversation: {
        conversation_id: CONVERSATION_ID,
        branch_id: BRANCH_A,
        branch_revision: 2,
        assistant_turn_id: ASSISTANT_A,
        assistant_turn_index: 2,
      },
    },
  };
}

async function activeRun() {
  const api = branchApi();
  const mark = FakeSocket.opened.length;
  const page = loadPage({
    WebSocket: OpenSocket,
    fetchImpl: api.fetchImpl,
    conversationApi: false,
    bootState: { ui_state: {}, models: MODELS },
  });
  await tick();
  await tick();
  page.registry.get("prompt-input").value = "Contract question";
  assert.equal(await page.context.startGeneration(), true);
  page.context.handleFrame(frame(0, "draft"));
  page.context.handleFrame(frame(1, "Contract answer"));
  page.context.handleDone({
    type: "done",
    final_text: "Contract answer",
    provenance: runProvenance(),
    run_token: "contract:1",
  });
  await page.context.conversationCompletion;
  await page.context.conversationClient.flush();
  return {
    api,
    page,
    context: page.context,
    socket: FakeSocket.opened[mark],
  };
}

function actionButton(mount, name) {
  const buttons = mount.querySelectorAll(
    "[data-conversation-action]"
  );
  const button = buttons.find((candidate) =>
    candidate.getAttribute("data-conversation-action") === name
  );
  assert.ok(button, "missing " + name + " action");
  return button;
}

function turnCard(run, turnId) {
  const cards = run.page.registry.get(
    "conversation-turns"
  ).children;
  const card = cards.find((candidate) =>
    candidate.getAttribute("data-turn-id") === turnId
  );
  assert.ok(card, "missing turn card " + turnId);
  return card;
}

function clickAction(run, button) {
  run.page.registry.get("conversation-transcript").dispatch(
    "click", { target: button }
  );
}

function clickDialog(run, kind, action) {
  const button = run.page.registry.get(
    "btn-conversation-" + kind + "-" + action
  );
  button.setAttribute(
    "data-conversation-confirmation",
    action === "cancel" ? "cancel" : "confirm-" + kind
  );
  run.page.registry.get(
    "conversation-" + kind + "-dialog"
  ).dispatch("click", { target: button });
}

function temperatureInput(run) {
  return run.page.registry.get("param-fields")
    .querySelector("input");
}

async function settleConversation(run) {
  await run.context.conversationClient.flush();
  await tick();
  await tick();
}

test("opaque branch identity reaches generate, complete, and save",
  async () => {
  const run = await activeRun();
  const generate = run.socket.sent
    .map((raw) => JSON.parse(raw))
    .find((message) => message.type === "generate");
  const append = run.api.state.calls.find((call) =>
    call.path.endsWith("/turns") && call.method === "POST"
  );
  const complete = run.api.state.calls.find((call) =>
    call.path.includes("/turns/" + ASSISTANT_A)
    && call.method === "PUT"
  );

  assert.equal(append.body.branch_id, BRANCH_A);
  assert.equal(append.body.branch_revision, 1);
  assert.equal(generate.conversation_id, CONVERSATION_ID);
  assert.equal(generate.branch_id, BRANCH_A);
  assert.equal(generate.branch_revision, 2);
  assert.equal(generate.assistant_turn_id, ASSISTANT_A);
  assert.equal(generate.assistant_turn_index, 2);
  assert.equal(generate.messages[0].turn_id, USER_A);
  assert.equal(complete.body.branch_id, BRANCH_A);
  assert.equal(complete.body.branch_revision, 2);

  assert.equal(await run.context.saveRun(), true);
  const save = run.api.state.calls.find((call) =>
    call.path === "/api/save"
  );
  const link = run.api.state.calls.find((call) =>
    call.path.endsWith("/run")
  );
  assert.equal(save.body.conversation_id, CONVERSATION_ID);
  assert.equal(save.body.branch_id, BRANCH_A);
  assert.equal(save.body.assistant_turn_id, ASSISTANT_A);
  assert.equal(save.body.turn_index, 2);
  assert.equal(save.body.assistant_turn_version, 2);
  assert.equal(link.body.branch_id, BRANCH_A);
  assert.equal(link.body.branch_revision, 3);
  assert.equal(link.body.assistant_turn_index, 2);
  assert.equal(link.body.assistant_turn_version, 2);
});

test("branch selection clears the incompatible page and run",
  async () => {
  const run = await activeRun();
  assert.equal(run.context.activeRunCanEdit(), true);
  assert.equal(run.context.generatorRun.frameCount(), 2);
  const callsBefore = run.api.state.calls.length;

  await run.context.selectConversationBranch(BRANCH_B);

  const selectionCalls = run.api.state.calls.slice(callsBefore)
    .filter((call) =>
      call.path.startsWith("/api/conversations/")
    );
  assert.ok(selectionCalls.length >= 2);
  assert.equal(
    selectionCalls.every((call) => call.method === "GET"),
    true
  );
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_B
  );
  assert.equal(run.context.conversationState.turns.length, 2);
  assert.equal(
    run.context.conversationState.turns[0].turn_id,
    USER_B
  );
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(run.context.activeRunCanEdit(), false);
  assert.equal(
    run.page.registry.get("active-assistant-card").hidden,
    true
  );
  const stored = JSON.parse(run.context.localStorage.getItem(
    run.context.PERSIST_ACTIVE_CONVERSATION_KEY
  ));
  assert.equal(stored.branch_id, BRANCH_B);
});

test("same-branch reload and cancel preserve exact XAI",
  async () => {
  const run = await activeRun();
  const framesBefore = run.context.generatorRun.frameCount();
  const retry = actionButton(
    run.page.registry.get("active-assistant-actions"),
    "retry"
  );
  clickAction(run, retry);

  await run.context.conversationClient.restore(
    CONVERSATION_ID, BRANCH_A
  );
  clickDialog(run, "retry", "cancel");

  assert.equal(
    run.context.generatorRun.frameCount(), framesBefore
  );
  assert.equal(
    run.page.registry.get("active-assistant-card").hidden,
    false
  );
  assert.equal(run.context.activeRunCanEdit(), true);
});

test("clicked Retry launches with its confirmed parameter snapshot",
  async () => {
  const run = await activeRun();
  run.context.handleModelStatus({ status: "ready" });
  run.api.holdNextFork();
  const retry = actionButton(
    run.page.registry.get("active-assistant-actions"),
    "retry"
  );
  clickAction(run, retry);
  assert.equal(
    run.page.registry.get("conversation-retry-dialog").open,
    true
  );
  assert.equal(
    run.page.document.activeElement,
    run.page.registry.get("btn-conversation-retry-cancel")
  );
  clickDialog(run, "retry", "confirm");
  await tick();

  const temperature = temperatureInput(run);
  temperature.value = "0.9";
  temperature.dispatch("input");
  run.api.releaseFork();
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_C
  );
  const generations = run.socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) => message.type === "generate");
  const retried = generations.at(-1);
  assert.equal(generations.length, 2);
  assert.equal(retried.branch_id, BRANCH_C);
  assert.equal(retried.assistant_turn_id, ASSISTANT_C);
  assert.equal(retried.assistant_turn_index, 2);
  assert.equal(retried.temperature, 0.7);
  assert.equal(retried.messages.at(-1).turn_id, USER_A);
  assert.equal(retried.messages.at(-1).content, "Contract question");
  const mutation = run.api.state.calls.find((call) =>
    call.path.includes("/branches/retry-assistant/")
    && call.method === "POST"
  );
  assert.match(mutation.body.operation_id, /^[0-9a-f]{32}$/);
  assert.equal("parameters" in mutation.body, false);
  assert.equal("configuration" in mutation.body, false);
  assert.equal("experimental" in mutation.body, false);
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(
    run.page.document.activeElement,
    run.page.registry.get("active-assistant-card")
  );
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "test_cleanup",
    message: "cleanup",
  });
});

test("clicked fork holds the shared conversation busy guard",
  async () => {
  const run = await activeRun();
  run.context.handleModelStatus({ status: "ready" });
  run.api.holdNextFork();

  clickAction(
    run,
    actionButton(
      run.page.registry.get("active-assistant-actions"),
      "retry"
    )
  );
  clickDialog(run, "retry", "confirm");
  await tick();

  assert.equal(run.context.conversationBusy, true);
  assert.equal(
    run.page.registry.get("btn-new-conversation").disabled,
    true
  );
  assert.equal(
    run.page.registry.get("btn-save").disabled,
    true
  );
  const savesBefore = run.api.state.calls.filter((call) =>
    call.path === "/api/save"
  ).length;
  assert.equal(await run.context.saveRun(), false);
  assert.equal(
    run.api.state.calls.filter((call) =>
      call.path === "/api/save"
    ).length,
    savesBefore
  );
  assert.equal(
    await run.context.selectConversationBranch(BRANCH_B),
    false
  );

  run.api.releaseFork();
  await settleConversation(run);
  assert.equal(run.context.conversationBusy, false);
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "test_cleanup",
    message: "cleanup",
  });
});

test("open message actions freeze and restore model controls",
  async () => {
  const run = await activeRun();
  const temperature = temperatureInput(run);
  const strategy = run.page.registry.get("param-fields")
    .querySelector(".custom-select");
  const model = run.page.registry.get("model-select");
  const retry = actionButton(
    run.page.registry.get("active-assistant-actions"),
    "retry"
  );
  assert.equal(temperature.disabled, false);

  clickAction(run, retry);
  assert.equal(temperature.disabled, true);
  assert.equal(model.getAttribute("aria-disabled"), "true");
  assert.equal(model.tabIndex, -1);
  assert.equal(strategy.getAttribute("aria-disabled"), "true");
  assert.equal(strategy.tabIndex, -1);
  clickDialog(run, "retry", "cancel");
  assert.equal(temperature.disabled, false);
  assert.equal(model.getAttribute("aria-disabled"), "false");
  assert.equal(model.tabIndex, 0);
  assert.equal(strategy.getAttribute("aria-disabled"), "false");
  assert.equal(strategy.tabIndex, 0);

  const edit = actionButton(turnCard(run, USER_A), "edit");
  clickAction(run, edit);
  assert.equal(temperature.disabled, true);
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "edit-cancel")
  );
  assert.equal(temperature.disabled, false);
});

test("clicked Edit keeps Save-time settings through a slow fork",
  async () => {
  const run = await activeRun();
  run.context.handleModelStatus({ status: "ready" });
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "edit")
  );
  const editing = turnCard(run, USER_A);
  const input = editing.querySelector(
    '[data-conversation-edit-input="' + USER_A + '"]'
  );
  input.value = "Edited contract question";
  run.page.registry.get("conversation-transcript").dispatch(
    "input", { target: input }
  );
  run.api.holdNextFork();
  clickAction(run, actionButton(editing, "edit-save"));
  await tick();

  const temperature = temperatureInput(run);
  temperature.value = "0.9";
  temperature.dispatch("input");
  run.api.releaseFork();
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_D
  );
  const generated = run.socket.sent
    .map((raw) => JSON.parse(raw))
    .filter((message) => message.type === "generate")
    .at(-1);
  assert.equal(generated.branch_id, BRANCH_D);
  assert.equal(generated.assistant_turn_id, ASSISTANT_D);
  assert.equal(generated.temperature, 0.7);
  assert.equal(generated.messages.at(-1).turn_id, USER_D);
  assert.equal(
    generated.messages.at(-1).content,
    "Edited contract question"
  );
  const mutation = run.api.state.calls.find((call) =>
    call.path.includes("/branches/edit-user/")
  );
  assert.match(mutation.body.operation_id, /^[0-9a-f]{32}$/);
  assert.equal("parameters" in mutation.body, false);
  assert.equal("configuration" in mutation.body, false);
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(turnCard(run, USER_D).focused, true);
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "test_cleanup",
    message: "cleanup",
  });
});

test("clicked Delete clears XAI and its path arrow navigates",
  async () => {
  const run = await activeRun();
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "delete")
  );
  clickDialog(run, "delete", "confirm");
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_E
  );
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(
    run.page.registry.get("active-assistant-card").hidden,
    true
  );
  const marker = run.page.registry.get(
    "conversation-turns"
  ).querySelector(".conversation-deletion-marker");
  assert.ok(marker);
  assert.equal(marker.focused, true);
  const mutation = run.api.state.calls.find((call) =>
    call.path.includes("/branches/delete-from-path/")
  );
  assert.match(mutation.body.operation_id, /^[0-9a-f]{32}$/);
  assert.deepEqual(
    Object.keys(mutation.body).sort(),
    [
      "branch_id",
      "branch_revision",
      "catalog_revision",
      "operation_id",
    ]
  );

  const previous = marker.querySelectorAll(
    ".conversation-branch-button"
  )[0];
  assert.equal(previous.disabled, false);
  clickAction(run, previous);
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_A
  );
  assert.equal(run.context.generatorRun.frameCount(), 0);
  assert.equal(
    run.api.state.calls.some((call) =>
      call.method === "GET"
      && call.branch === BRANCH_A
      && call.path.endsWith("/metadata")
    ),
    true
  );
});

test("lost Edit reply auto-replays one branch and operation",
  async () => {
  const run = await activeRun();
  run.context.handleModelStatus({ status: "ready" });
  run.api.dropNextForkReply("edit");
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "edit")
  );
  const editing = turnCard(run, USER_A);
  const input = editing.querySelector(
    '[data-conversation-edit-input="' + USER_A + '"]'
  );
  input.value = "Lost edit";
  run.page.registry.get("conversation-transcript").dispatch(
    "input", { target: input }
  );
  clickAction(run, actionButton(editing, "edit-save"));
  await settleConversation(run);

  const calls = run.api.state.calls.filter((call) =>
    call.path.includes("/branches/edit-user/")
  );
  assert.equal(calls.length, 2);
  assert.equal(
    calls[0].body.operation_id,
    calls[1].body.operation_id
  );
  assert.equal(
    run.api.state.branchIds.filter((id) => id === BRANCH_D).length,
    1
  );
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_D
  );
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "test_cleanup",
    message: "cleanup",
  });
});

test("lost Delete reply auto-replays one branch and operation",
  async () => {
  const run = await activeRun();
  run.api.dropNextForkReply("delete");
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "delete")
  );
  clickDialog(run, "delete", "confirm");
  await settleConversation(run);

  const calls = run.api.state.calls.filter((call) =>
    call.path.includes("/branches/delete-from-path/")
  );
  assert.equal(calls.length, 2);
  assert.equal(
    calls[0].body.operation_id,
    calls[1].body.operation_id
  );
  assert.equal(
    run.api.state.branchIds.filter((id) => id === BRANCH_E).length,
    1
  );
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_E
  );
});

test("lost Retry reply auto-replays one branch and operation",
  async () => {
  const run = await activeRun();
  run.context.handleModelStatus({ status: "ready" });
  run.api.dropNextForkReply("retry");
  clickAction(
    run,
    actionButton(
      run.page.registry.get("active-assistant-actions"),
      "retry"
    )
  );
  clickDialog(run, "retry", "confirm");
  await settleConversation(run);

  const calls = run.api.state.calls.filter((call) =>
    call.path.includes("/branches/retry-assistant/")
  );
  assert.equal(calls.length, 2);
  assert.equal(
    calls[0].body.operation_id,
    calls[1].body.operation_id
  );
  assert.equal(
    run.api.state.branchIds.filter((id) => id === BRANCH_C).length,
    1
  );
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_C
  );
  run.context.handleError({
    type: "error",
    scope: "run",
    code: "test_cleanup",
    message: "cleanup",
  });
});

test("a disconnected Retry remains durably pending", async () => {
  const run = await activeRun();
  run.context.generatorSocket.close();
  const generationsBefore = run.socket.sent.filter((raw) =>
    JSON.parse(raw).type === "generate"
  ).length;

  clickAction(
    run,
    actionButton(
      run.page.registry.get("active-assistant-actions"),
      "retry"
    )
  );
  clickDialog(run, "retry", "confirm");
  await settleConversation(run);

  assert.match(
    run.page.registry.get("conversation-action-status").textContent,
    /pending/
  );
  assert.equal(
    run.context.conversationState.conversation
      .pending_assistant_id,
    ASSISTANT_C
  );
  assert.equal(
    run.socket.sent.filter((raw) =>
      JSON.parse(raw).type === "generate"
    ).length,
    generationsBefore
  );
  assert.equal(run.context.conversationCanNavigate(), false);
  assert.equal(
    run.page.registry.get("active-assistant-card").focused,
    false
  );
  assert.equal(
    run.page.document.activeElement,
    run.page.registry.get("prompt-input")
  );
});

test("a disconnected Edit focuses its selected user card",
  async () => {
  const run = await activeRun();
  run.context.generatorSocket.close();
  const generationsBefore = run.socket.sent.filter((raw) =>
    JSON.parse(raw).type === "generate"
  ).length;
  clickAction(
    run,
    actionButton(turnCard(run, USER_A), "edit")
  );
  const editing = turnCard(run, USER_A);
  const input = editing.querySelector(
    '[data-conversation-edit-input="' + USER_A + '"]'
  );
  input.value = "Deferred edit";
  run.page.registry.get("conversation-transcript").dispatch(
    "input", { target: input }
  );
  clickAction(run, actionButton(editing, "edit-save"));
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_D
  );
  assert.equal(
    run.socket.sent.filter((raw) =>
      JSON.parse(raw).type === "generate"
    ).length,
    generationsBefore
  );
  assert.equal(
    run.page.registry.get("active-assistant-card").hidden,
    true
  );
  assert.equal(
    run.page.registry.get("active-assistant-card").focused,
    false
  );
  assert.equal(turnCard(run, USER_D).focused, true);
});

test("a rejected clicked Retry preserves branch and XAI",
  async () => {
  const run = await activeRun();
  run.api.rejectNextFork("retry");
  const retry = actionButton(
    run.page.registry.get("active-assistant-actions"),
    "retry"
  );
  clickAction(run, retry);
  clickDialog(run, "retry", "confirm");
  await settleConversation(run);

  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_A
  );
  assert.equal(run.context.generatorRun.frameCount(), 2);
  assert.equal(
    run.api.state.branchIds.includes(BRANCH_C),
    false
  );
  const status = run.page.registry.get(
    "conversation-retry-status"
  );
  assert.match(status.textContent, /retry rejected/);
  assert.equal(
    run.page.registry.get("conversation-retry-dialog").open,
    true
  );
  assert.equal(
    temperatureInput(run).disabled,
    true
  );
  assert.equal(run.page.document.activeElement, status);

  clickDialog(run, "retry", "cancel");
  assert.equal(
    run.page.registry.get("conversation-retry-dialog").open,
    false
  );
  assert.equal(temperatureInput(run).disabled, false);
  assert.equal(run.page.document.activeElement, retry);
});

test("fork catalog conflict reloads and closes without replay",
  async () => {
  const run = await activeRun();
  await run.context.selectConversationBranch(BRANCH_B);

  run.api.conflictBranch();
  const putsBefore = run.api.state.calls.filter((call) =>
    call.method === "PUT"
    && call.path.includes("/turns/" + ASSISTANT_B)
  ).length;
  await assert.rejects(
    run.context.conversationClient.updateAssistant({
      branchId: BRANCH_B,
      branchRevision: 3,
      assistantTurnId: ASSISTANT_B,
      text: "stale",
      partial: false,
      contextPack: {},
      metadata: {},
    }),
    /stale branch/
  );
  const putsAfter = run.api.state.calls.filter((call) =>
    call.method === "PUT"
    && call.path.includes("/turns/" + ASSISTANT_B)
  ).length;
  assert.equal(putsAfter - putsBefore, 1);
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_B
  );

  run.api.conflictCatalog();
  const forksBefore = run.api.state.calls.filter((call) =>
    call.path.includes("/branches/retry-assistant/")
  ).length;
  clickAction(
    run,
    actionButton(turnCard(run, ASSISTANT_B), "retry")
  );
  clickDialog(run, "retry", "confirm");
  await settleConversation(run);
  const forksAfter = run.api.state.calls.filter((call) =>
    call.path.includes("/branches/retry-assistant/")
  ).length;
  assert.equal(forksAfter - forksBefore, 1);
  assert.equal(run.context.conversationBusy, false);
  assert.equal(
    run.context.conversationState.selectedBranchId,
    BRANCH_B
  );
  assert.equal(
    run.page.registry.get("conversation-retry-dialog").open,
    false
  );
  const reloads = run.api.state.calls.filter((call) =>
    call.path.endsWith("/metadata")
    && call.branch === BRANCH_B
  );
  assert.ok(reloads.length >= 3);
});

test("numeric snapshots normalize to the synthetic legacy branch",
  async () => {
  const run = await activeRun();
  run.context.saveSessionState();
  const stored = JSON.parse(run.context.sessionStorage.getItem(
    run.context.SESSION_KEY
  ));
  delete stored.branchId;
  delete stored.branchRevision;
  delete stored.assistantTurnIndex;
  stored.conversationRevision = 3;
  stored.assistantTurnId = "00000002";
  stored.conversationTurnIndex = 2;

  const decoded = run.context.runSnapshotDecode(
    JSON.stringify(stored),
    { model: MODEL.id, device: "cuda" }
  );

  assert.equal(
    decoded.branchId,
    "b_" + CONVERSATION_ID
  );
  assert.equal(decoded.branchRevision, 3);
  assert.equal(decoded.assistantTurnIndex, 2);
});
