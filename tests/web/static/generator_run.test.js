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
      "start:edited",
      "success:run-a",
    ]);
  }
);
