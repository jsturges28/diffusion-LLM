// The generator run controller, driven without app.js.
//
// Strategy: load the frame, candidate and snapshot codecs before the
// controller in one vm context, then supply narrow model, composer,
// chrome, edit, storage and request adapters. Passing proves the
// controller owns mutable run state privately while preserving frame
// reads, reset, save and session behavior at its public boundary.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);

const SOURCES = [
  "run_frames.js",
  "run_candidates.js",
  "run_snapshot.js",
  "generator_run.js",
];

function load() {
  const context = vm.createContext({});
  for (const name of SOURCES) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context,
      { filename: name }
    );
  }
  return context;
}

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) || null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, refuse) => {
    resolve = accept;
    reject = refuse;
  });
  return { promise, resolve, reject };
}

function harness(results) {
  const context = load();
  const storage = memoryStorage();
  const external = {
    model: {
      id: "llada",
      device: "cuda",
      params: { steps: 8 },
    },
    composer: {
      draft: "draft for next run",
      prompt: "draft for next run",
    },
    chrome: {
      thinking: "reasoning",
      status: {
        step: "Step 2/2",
        elapsed: "Elapsed: 0.2s",
        message: "Done.",
      },
    },
    edit: { remaskEdits: [] },
    conversation: null,
    restoredComposer: null,
    restoredChrome: null,
    restoredEdit: null,
    invalidations: 0,
    sessionRestores: 0,
    lifecycle: [],
    requests: [],
  };
  const replies = results || [];
  const run = context.generatorRunCreate({
    readModel: () => external.model,
    readComposer: () => external.composer,
    restoreComposer: (state) => {
      external.restoredComposer = state;
    },
    readChrome: () => external.chrome,
    restoreChrome: (state) => {
      external.restoredChrome = state;
    },
    readEditArtifacts: () => external.edit,
    restoreEditArtifacts: (state) => {
      external.restoredEdit = state;
      external.edit = state;
    },
    invalidateRender: () => {
      external.invalidations += 1;
    },
    onSessionRestored: () => {
      external.sessionRestores += 1;
    },
    requestSave: (url, init) => {
      external.requests.push({
        url: url,
        body: JSON.parse(init.body),
      });
      const reply = replies.shift() || {
        success: false,
        message: "held by test",
      };
      return Promise.resolve({
        json: () => Promise.resolve(reply),
      });
    },
    onSaveStart: (info) => {
      external.lifecycle.push("start:" + info.label);
      return "status-" + external.lifecycle.length;
    },
    onSaveSuccess: (info) => {
      external.lifecycle.push("success:" + info.runId);
    },
    onSaveFailure: (info) => {
      external.lifecycle.push("failure:" + info.message);
    },
    onSaveRefused: (message) => {
      external.lifecycle.push("refused:" + message);
    },
    onSaveSettled: () => {
      external.lifecycle.push("settled");
    },
    readConversation: () => external.conversation,
    storage: storage,
    sessionKey: "last-run",
  });
  return { run, external, storage };
}

function token(text, id) {
  return {
    t: text,
    m: false,
    id: id,
    c: 0.75,
    e: 0.25,
  };
}

function snapshotFrame(index, text) {
  return {
    index: index,
    total_steps: 2,
    canvas_index: 0,
    mean_conf: 0.75,
    text: text,
    tokens: [token(text, 100 + index)],
    revealed: [0],
    elapsed: (index + 1) / 10,
  };
}

function conversationIdentity(overrides) {
  return Object.assign({
    conversation_id: "a".repeat(32),
    branch_id: "b_" + "b".repeat(32),
    branch_revision: 2,
    assistant_turn_id:
      "t_" + "b".repeat(32) + "_00000002_0000000000000002",
    assistant_turn_index: 2,
    assistant_turn_version: 1,
    assistant_text: "",
  }, overrides || {});
}

function assistantAction(text, version, partial) {
  return {
    type: "assistant_updated",
    turn: {
      turn_id:
        "t_" + "b".repeat(32)
        + "_00000002_0000000000000002",
      index: 2,
      version: version,
      text: text,
      partial: partial === true,
      run_link: null,
    },
  };
}

function appendFrame(index, text) {
  return {
    shape: "append",
    index: index,
    total_steps: 3,
    canvas_index: 0,
    mean_conf: 0.75,
    token: token(text, 200 + index),
    revealed: [index - 1],
    elapsed: index / 10,
  };
}

function finish(run, finalText) {
  run.finish({
    final_text: finalText,
    prompt_len: 5,
    run_token: "nonce:1",
    provenance: {
      model_id: "llada",
      device: "cuda",
    },
  });
}

test("mutable stores stay private behind copied reads", () => {
  const { run } = harness();
  run.appendFrame(snapshotFrame(0, "first"));

  assert.equal(run.frames, undefined);
  assert.equal(run.original, undefined);
  assert.equal(run.positionAlts, undefined);

  const read = run.frameTokens(0);
  read.length = 0;
  assert.equal(run.frameTokens(0)[0].t, "first");
});

test("snapshot and append operations read through one API", () => {
  const snapshot = harness().run;
  snapshot.appendFrame(snapshotFrame(0, "a"));
  snapshot.appendFrame(snapshotFrame(1, "b"));

  assert.equal(snapshot.frameCount(), 2);
  assert.equal(snapshot.frameText(1), "b");
  assert.equal(snapshot.frameIsAppend(), false);

  const append = harness().run;
  append.appendFrame(appendFrame(1, "The"));
  append.appendFrame(appendFrame(2, " cat"));
  const checkpoint = append.captureCheckpoint();
  append.truncate(1);
  append.appendFrame(appendFrame(2, " dog"));
  assert.equal(append.frameText(1), "The dog");

  append.restoreCheckpoint(checkpoint);
  assert.equal(append.frameText(1), "The cat");
  assert.equal(append.resumeElapsedOffset(), 0);
});

test("watermark pressure and sampler candidates survive run state",
  () => {
  const { run, external } = harness();
  run.begin("what ran", { watermark: true, alternatives: true });
  const frame = appendFrame(1, "The");
  Object.assign(frame.token, {
    g: true,
    we: false,
    gb: 0.1,
    gk: 0.2,
    gs: 0.3,
  });
  frame.alts = [{ id: 3, t: "The", p: 0.4, g: true }];
  frame.salts = {
    support: 7,
    candidates: [{
      id: 3, t: "The", p: 0.6, rank: 1, g: true,
    }],
  };
  run.appendFrame(frame);
  const second = appendFrame(2, " cat");
  Object.assign(second.token, {
    g: false,
    we: true,
    gb: 0.2,
    gk: 0.3,
    gs: 0.4,
  });
  second.salts = {
    support: 5,
    candidates: [{
      id: 4, t: " cat", p: 0.7, rank: 1, g: false,
    }],
  };
  run.appendFrame(second);
  finish(run, "The cat");
  external.edit = {
    remaskEdits: [{ frame_index: 0, token_positions: [0] }],
  };

  assert.equal(run.framePositions()[0].gb, 0.1);
  assert.equal(
    run.positionSamplerAlternatives(0, false).support,
    7
  );
  assert.equal(run.hasSamplerAlternatives(false), true);
  assert.equal(run.hasSamplerAlternatives(true), true);
  const payload = run.buildSavePayload();
  assert.equal(payload.frame_positions[0].gk, 0.2);
  assert.equal(payload.alternatives[0][0].g, true);
  assert.equal(payload.sampler_alternatives[0].support, 7);
  assert.equal(
    payload.original_sampler_alternatives[0].candidates[0].rank,
    1
  );
  assert.equal(run.saveSession(), true);
  run.reset();
  assert.equal(run.restoreSession(), true);
  assert.equal(
    run.positionSamplerAlternatives(0, false).support,
    7
  );
});

test("reset retires every run identity and frame", () => {
  const { run } = harness();
  run.begin("what ran", { steps: 2 });
  run.adoptResidentWorker("worker:1");
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "done");

  run.reset();

  assert.equal(run.frameCount(), 0);
  assert.equal(run.finalText(), null);
  assert.equal(run.runToken(), "");
  assert.equal(run.originalCaptured(), false);
  assert.equal(run.interrupted(), false);
  assert.equal(run.saved(), false);
});

test("the save payload preserves the run record shape", () => {
  const { run } = harness();
  run.begin("what ran", { steps: 2, temperature: 0.5 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "finished");

  const payload = run.buildSavePayload();

  assert.equal(payload.model, "llada");
  assert.equal(payload.prompt, "what ran");
  assert.deepEqual(
    JSON.parse(JSON.stringify(payload.params)),
    { steps: 2, temperature: 0.5 }
  );
  assert.equal(payload.final_text, "finished");
  assert.equal(payload.prompt_len, 5);
  assert.equal(payload.run_token, "nonce:1");
  assert.equal(payload.frames.length, 2);
  assert.equal(payload.frame_tokens[1][0].e, 0.25);
  assert.equal("partial" in payload, false);
});

test("watermark token flags survive save and session codecs", () => {
  const { run } = harness();
  const first = appendFrame(1, "a");
  first.token.g = true;
  first.token.we = false;
  const second = appendFrame(2, "b");
  second.token.g = false;
  second.token.we = true;

  run.begin("marked", {
    watermark: true,
    watermark_gamma: 0.25,
    watermark_delta: 2,
  });
  run.appendFrame(first);
  run.appendFrame(second);
  finish(run, "ab");

  let payload = run.buildSavePayload();
  assert.equal(payload.frame_positions[0].g, true);
  assert.equal(payload.frame_positions[0].we, false);
  assert.equal(payload.frame_positions[1].g, false);
  assert.equal(payload.frame_positions[1].we, true);

  assert.equal(run.saveSession(), true);
  run.reset();
  assert.equal(run.restoreSession(), true);
  payload = run.buildSavePayload();
  assert.equal(payload.frame_positions[0].g, true);
  assert.equal(payload.frame_positions[1].we, true);
});

test("branch rollback restores watermark provenance", () => {
  const { run } = harness();
  run.begin("marked", { watermark: true });
  run.appendFrame(appendFrame(1, "a"));
  run.appendFrame(appendFrame(2, "b"));
  run.finish({
    final_text: "ab",
    run_token: "nonce:1",
    provenance: {
      model_id: "smollm3",
      watermark: { key_id: "original-key", scored_count: 1 },
    },
  });
  const checkpoint = run.captureCheckpoint();

  run.finish({
    final_text: "branch",
    run_token: "nonce:1",
    provenance: {
      model_id: "smollm3",
      watermark: { key_id: "branch-key", scored_count: 2 },
    },
  });
  assert.equal(
    run.provenance().watermark.key_id, "branch-key"
  );

  run.restoreCheckpoint(checkpoint);

  assert.equal(
    run.provenance().watermark.key_id, "original-key"
  );
});

test("append frames retain the latest watermark score", () => {
  const { run } = harness();
  run.begin("marked", { watermark: true });
  const first = appendFrame(1, "a");
  first.provenance = {
    model_id: "smollm3",
    watermark: {
      scheme: "kgw",
      key_id: "0123456789abcdef",
      scored_count: 0,
    },
  };
  first.watermark_stats = {
    green_count: 0,
    scored_count: 0,
    z_score: 0,
    p0: 0.25,
    status: "insufficient_evidence",
  };
  const second = appendFrame(2, "b");
  second.watermark_stats = {
    green_count: 1,
    scored_count: 1,
    z_score: 1.5,
    p0: 0.25,
    status: "insufficient_evidence",
  };

  run.appendFrame(first);
  run.appendFrame(second);

  const watermark = run.provenance().watermark;
  assert.equal(watermark.key_id, "0123456789abcdef");
  assert.equal(watermark.green_count, 1);
  assert.equal(watermark.scored_count, 1);
  assert.equal(watermark.z_score, 1.5);
  assert.equal(watermark.p0, 0.25);
});

test("a conversation-bound run saves its durable turn location", () => {
  const { run, external } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "finished");

  const payload = run.buildSavePayload();

  assert.equal(payload.conversation_id, "a".repeat(32));
  assert.equal(payload.branch_id, "b_" + "b".repeat(32));
  assert.equal(
    payload.assistant_turn_id,
    "t_" + "b".repeat(32) + "_00000002_0000000000000002"
  );
  assert.equal(payload.turn_index, 2);
  assert.equal(payload.assistant_turn_version, 1);
});

test("a session snapshot round trips through private state", () => {
  const { run, external } = harness();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  run.setTotalSteps(2);
  finish(run, "finished");
  external.edit = {
    remaskEdits: [
      { frame_index: 1, token_positions: [0] },
    ],
  };

  assert.equal(run.saveSession(), true);
  run.reset();
  assert.equal(run.restoreSession(), true);

  assert.equal(run.frameCount(), 2);
  assert.equal(run.finalText(), "finished");
  assert.equal(run.totalSteps(), 2);
  assert.equal(external.sessionRestores, 1);
  assert.equal(external.restoredComposer.draft, "draft for next run");
  assert.equal(external.restoredChrome.thinking, "reasoning");
  assert.equal(external.restoredEdit.remaskEdits.length, 1);
});

test("session snapshots bind the active run to its conversation", () => {
  const { run, external, storage } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "finished");
  external.conversation.branch_revision = 3;
  external.conversation.assistant_turn_version = 2;
  external.conversation.assistant_text = "finished";
  assert.equal(
    run.refreshConversation(
      assistantAction("finished", 2, false)
    ),
    true
  );

  assert.equal(run.saveSession(), true);
  const stored = JSON.parse(storage.getItem("last-run"));
  assert.equal(stored.conversationId, "a".repeat(32));
  assert.equal(stored.branchId, "b_" + "b".repeat(32));
  assert.equal(stored.branchRevision, 3);
  assert.equal(stored.assistantTurnIndex, 2);

  run.reset();
  assert.equal(run.restoreSession(), true);
  assert.equal(
    run.conversationIdentity().assistant_turn_id,
    "t_" + "b".repeat(32) + "_00000002_0000000000000002"
  );
});

test("an empty completion still advances its conversation identity",
  () => {
  const { run, external } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  finish(run, "");
  external.conversation = conversationIdentity({
    branch_revision: 3,
    assistant_turn_version: 2,
    assistant_text: "",
  });

  assert.equal(run.finalText(), "");
  assert.equal(
    run.refreshConversation(assistantAction("", 2, false)),
    true
  );
  assert.equal(
    run.conversationIdentity().assistant_turn_version,
    2
  );
});

test("a reload cannot relabel frames to a newer same-text turn", () => {
  const { run, external } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "finished"));
  finish(run, "finished");
  external.conversation = conversationIdentity({
    branch_revision: 3,
    assistant_turn_version: 2,
    assistant_text: "finished",
  });

  assert.equal(
    run.refreshConversation({ type: "loaded" }),
    false
  );
  assert.equal(
    run.conversationIdentity().assistant_turn_version,
    1
  );
});

test("a lost completion reload may adopt its exact terminal turn",
  () => {
  const { run, external } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "finished"));
  finish(run, "finished");
  external.conversation = conversationIdentity({
    branch_revision: 3,
    assistant_turn_version: 2,
    assistant_text: "finished",
  });
  const terminal = assistantAction("finished", 2, false).turn;

  assert.equal(
    run.refreshConversation({
      type: "loaded",
      page: { turns: [terminal] },
    }),
    true
  );
  assert.equal(
    run.conversationIdentity().assistant_turn_version,
    2
  );
});

test("numeric conversation snapshots still restore", () => {
  const { run, external, storage } = harness();
  external.conversation = conversationIdentity({
    branch_id: "b_" + "a".repeat(32),
    assistant_turn_id: "00000002",
  });
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "finished");
  assert.equal(run.saveSession(), true);
  const stored = JSON.parse(storage.getItem("last-run"));
  stored.conversationRevision = stored.branchRevision;
  stored.conversationTurnIndex = stored.assistantTurnIndex;
  delete stored.branchId;
  delete stored.branchRevision;
  delete stored.assistantTurnIndex;
  storage.setItem("last-run", JSON.stringify(stored));

  run.reset();

  assert.equal(run.restoreSession(), true);
  assert.equal(
    run.conversationIdentity().branch_id,
    "b_" + "a".repeat(32)
  );
  assert.equal(
    run.conversationIdentity().assistant_turn_id,
    "00000002"
  );
});

test("a snapshot from another conversation is retired", () => {
  const { run, external, storage } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "b"));
  finish(run, "finished");
  assert.equal(run.saveSession(), true);
  run.reset();
  external.conversation = conversationIdentity({
    conversation_id: "b".repeat(32),
    branch_id: "b_" + "c".repeat(32),
    branch_revision: 1,
  });

  assert.equal(run.restoreSession(), false);
  assert.equal(storage.getItem("last-run"), null);
});

test("a stale same-tail snapshot is retired", () => {
  const { run, external, storage } = harness();
  external.conversation = conversationIdentity();
  run.begin("what ran", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "a"));
  run.appendFrame(snapshotFrame(1, "old answer"));
  finish(run, "old answer");
  external.conversation = conversationIdentity({
    branch_revision: 3,
    assistant_turn_version: 2,
    assistant_text: "old answer",
  });
  assert.equal(
    run.refreshConversation(
      assistantAction("old answer", 2, false)
    ),
    true
  );
  assert.equal(run.saveSession(), true);
  run.reset();
  external.conversation = conversationIdentity({
    branch_revision: 4,
    assistant_turn_version: 3,
    assistant_text: "new answer",
  });

  assert.equal(run.restoreSession(), false);
  assert.equal(storage.getItem("last-run"), null);
});

test(
  "save success carries identity into an edited revision",
  async () => {
    const replies = [
      {
        success: true,
        path: "runs/run-a",
        run_id: "run-a",
        revision: 3,
      },
      {
        success: true,
        path: "runs/run-a",
        run_id: "run-a",
        revision: 4,
      },
    ];
    const { run, external } = harness(replies);
    run.begin("what ran", { steps: 2 });
    run.appendFrame(snapshotFrame(0, "a"));
    run.appendFrame(snapshotFrame(1, "b"));
    finish(run, "finished");

    await run.save();
    external.edit = {
      remaskEdits: [
        { frame_index: 1, token_positions: [0] },
      ],
    };
    await run.save();

    assert.equal(run.saved(), true);
    assert.equal(run.editedSaved(), true);
    assert.equal(run.savedRunId(), "run-a");
    assert.equal(run.savedRevision(), 4);
    assert.equal(external.requests[1].body.run_id, "run-a");
    assert.equal(external.requests[1].body.expected_revision, 3);
    assert.deepEqual(external.lifecycle, [
      "start:original",
      "success:run-a",
      "settled",
      "start:edited",
      "success:run-a",
      "settled",
    ]);
  }
);

test("a late old-run save cannot adopt into a reset run",
  async () => {
  const oldReply = deferred();
  const newReply = deferred();
  const { run, external } = harness([
    oldReply.promise,
    newReply.promise,
  ]);
  run.begin("old run", { steps: 2 });
  run.appendFrame(snapshotFrame(0, "old"));
  finish(run, "old result");
  const oldSaving = run.save();
  await Promise.resolve();

  run.reset();
  run.begin("new run", { steps: 4 });
  run.appendFrame(snapshotFrame(0, "new"));
  finish(run, "new result");
  const newSaving = run.save();
  await Promise.resolve();
  oldReply.resolve({
    success: true,
    path: "runs/old-run",
    run_id: "old-run",
    revision: 1,
  });

  assert.equal(await oldSaving, false);
  assert.equal(run.saving(), true);
  newReply.resolve({
    success: true,
    path: "runs/new-run",
    run_id: "new-run",
    revision: 1,
  });

  assert.equal(await newSaving, true);
  assert.equal(run.saved(), true);
  assert.equal(run.savedRunId(), "new-run");
  assert.equal(run.savedRevision(), 1);
  assert.equal(run.finalText(), "new result");
  assert.deepEqual(external.lifecycle, [
    "start:original",
    "start:original",
    "success:new-run",
    "settled",
  ]);
});
