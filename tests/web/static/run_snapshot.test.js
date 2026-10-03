// The run snapshot's codec, driven without a page.
//
// Strategy: load run_frames.js, run_candidates.js and run_snapshot.js
// into one vm context, in the order the generator loads them, build
// run records from the two families' own constructors, and drive the
// encoder and the decoder directly. There is no DOM and no storage
// here: the tiers are what the page's quota loop is offered, and the
// decoder is handed the text a tier is stored as.
//
// Passing proves the snapshot's rules hold on their own: which tiers
// a run offers and in what order, that a run not worth keeping
// offers none, that each tier decodes back to the run as far as it
// carries it, and that older shapes and foreign snapshots read as
// they always have. The page-level round trips, through app.js and
// the DOM stub, stay in snapshot_budget.test.js,
// generator_candidates.test.js and run_record.test.js.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);

// The generator's own order: the codec reads both families.
const SOURCES = [
  "run_frames.js",
  "run_candidates.js",
  "run_snapshot.js",
];

const RESIDENT = { model: "llada", device: "cuda" };

// The stored fields in the order every snapshot has written them,
// which is the format an older build's snapshot is read against.
const STORED_FIELDS = [
  "model",
  "device",
  "prompt",
  "runPrompt",
  "finalText",
  "params",
  "promptLen",
  "provenance",
  "runToken",
  "worker",
  "thinking",
  "remaskEdits",
  "editedRunSaved",
  "runInterrupted",
  "runLostConnection",
  "runSaved",
  "lastSavedRunId",
  "lastSavedRevision",
  "statusStep",
  "lastRunTotalSteps",
  "statusElapsed",
  "statusMessage",
];

// The per-token detail the light tier drops.
const DETAIL_KEYS = [
  "frameTokens",
  "frameCanvasIndex",
  "frameMeanConf",
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

// One snapshot-shaped frame, distinguishable by index.
function entry(index) {
  return {
    history: "frame " + index,
    tokens: [{ t: "w" + index, m: false, id: 100 + index, c: 0.5 }],
    canvasIndex: 0,
    meanConf: 0.5,
    elapsed: index / 10,
    revealed: [index],
  };
}

function framesOf(api, count) {
  const frames = api.runFramesCreate();
  for (let index = 0; index < count; index += 1) {
    api.runFramesAppend(frames, entry(index));
  }
  return frames;
}

function appendFramesOf(api, count) {
  const frames = api.runFramesCreate();
  for (let index = 1; index <= count; index += 1) {
    api.runFramesAppendPosition(frames, {
      index: index,
      token: { t: "w" + index, m: false, id: 200 + index, c: 0.5 },
      canvasIndex: 0,
      meanConf: 0.5,
      elapsed: index / 10,
      revealed: [index - 1],
    });
  }
  return frames;
}

function setsFor(frame) {
  return [
    {
      h: frame * 10,
      c: [
        { id: frame * 10, t: " lead", p: 0.6 },
        { id: 7, t: " seven", p: 0.2 },
      ],
    },
  ];
}

function storeWith(api, store, offset, frames) {
  return api.runCandidatesAddStream(store, offset, {
    k: 2,
    stride: 1,
    frames: frames,
    sets: frames.map(setsFor),
  });
}

// A finished, unedited run of three frames: its baseline captured,
// and its candidates the live store itself, as the page holds them.
function finishedRecord(api, overrides) {
  const frames = framesOf(api, 3);
  const original = api.originalRunCreate();
  api.originalRunCapture(original, frames, []);
  const candidates = storeWith(
    api, api.runCandidatesCreate(), 0, [1, 2]
  );
  return Object.assign({
    model: "llada",
    device: "cuda",
    prompt: "the box's text",
    runPrompt: "what ran",
    finalText: "w0 w1 w2",
    params: { steps: 3 },
    promptLen: 9,
    provenance: { model_id: "llada", device: "cuda" },
    runToken: "nonce:1",
    worker: "worker-a:1",
    thinking: "",
    remaskEdits: [],
    editedRunSaved: false,
    runInterrupted: false,
    runLostConnection: false,
    runSaved: false,
    lastSavedRunId: null,
    lastSavedRevision: null,
    statusStep: "Step 3/3",
    lastRunTotalSteps: 3,
    statusElapsed: "0.3s",
    statusMessage: "Done",
    frames: frames,
    positionAlts: [],
    original: original,
    candidates: candidates,
    originalCandidates: candidates,
  }, overrides || {});
}

// The same run edited at frame 1 and resumed for two frames, so the
// live store and the baseline's have parted.
function editedRecord(api) {
  const record = finishedRecord(api);
  api.runFramesTruncate(record.frames, 1);
  api.runFramesAppend(record.frames, entry(1));
  api.runFramesAppend(record.frames, entry(2));
  return Object.assign(record, {
    remaskEdits: [{ frame_index: 1, token_positions: [0] }],
    candidates: storeWith(api, record.candidates, 1, [1]),
  });
}

function tierText(api, record, index) {
  return JSON.stringify(api.runSnapshotTiers(record)[index]);
}

// Compared as JSON: objects built in the vm have that realm's
// prototypes, so a strict deepEqual against a host value fails on
// identity even when the contents match.
function same(actual, expected) {
  assert.equal(JSON.stringify(actual), JSON.stringify(expected));
}

// -- what a run offers --

test("a finished run offers candidates, the full, the light", () => {
  const api = load();
  const tiers = api.runSnapshotTiers(finishedRecord(api));

  assert.equal(tiers.length, 3);
  assert.ok("candidates" in tiers[0]);
  assert.equal("originalCandidates" in tiers[0], false);
  assert.equal("candidates" in tiers[1], false);
  assert.ok("frameTokens" in tiers[1]);
  for (const key of DETAIL_KEYS) {
    assert.equal(key in tiers[2], false, key);
  }
});

test("an edited run offers both runs' candidates first", () => {
  const api = load();
  const tiers = api.runSnapshotTiers(editedRecord(api));

  assert.equal(tiers.length, 4);
  assert.ok("originalCandidates" in tiers[0]);
  assert.ok("candidates" in tiers[0]);
  assert.equal("originalCandidates" in tiers[1], false);
  assert.ok("candidates" in tiers[1]);
});

test("a run without candidates offers the two payloads", () => {
  const api = load();
  const empty = api.runCandidatesCreate();
  const record = finishedRecord(api, {
    candidates: empty,
    originalCandidates: empty,
  });

  const tiers = api.runSnapshotTiers(record);

  assert.equal(tiers.length, 2);
  assert.equal("candidates" in tiers[0], false);
});

test("an unedited run writes its candidates once", () => {
  // Until an edit the baseline's candidates are the live run's, and
  // writing them twice could cost the quota the tokens.
  const api = load();
  for (const originalCandidates of [undefined, null]) {
    const record = finishedRecord(api);
    if (originalCandidates === null) {
      record.originalCandidates = null;
    }
    for (const tier of api.runSnapshotTiers(record)) {
      assert.equal("originalCandidates" in tier, false);
    }
  }
});

test("a run not worth keeping offers nothing", () => {
  const api = load();

  same(
    api.runSnapshotTiers(finishedRecord(api, { model: null })),
    []
  );
  same(
    api.runSnapshotTiers(finishedRecord(api, { finalText: "" })),
    []
  );
  same(
    api.runSnapshotTiers(
      finishedRecord(api, { frames: framesOf(api, 1) })
    ),
    []
  );
});

test("a lighter tier only drops what the one before it held", () => {
  const api = load();
  const tiers = api.runSnapshotTiers(editedRecord(api));

  for (let index = 1; index < tiers.length; index += 1) {
    for (const key of Object.keys(tiers[index])) {
      same(tiers[index][key], tiers[index - 1][key]);
    }
    assert.ok(
      Object.keys(tiers[index]).length
        < Object.keys(tiers[index - 1]).length
    );
  }
});

test("the stored fields keep their order", () => {
  const api = load();
  const light = api.runSnapshotTiers(finishedRecord(api))[2];

  same(
    Object.keys(light).slice(0, STORED_FIELDS.length),
    STORED_FIELDS
  );
});

test("encoding leaves the record as it was", () => {
  const api = load();
  const record = editedRecord(api);
  const before = JSON.stringify(record);

  api.runSnapshotTiers(record);

  assert.equal(JSON.stringify(record), before);
});

test("a record missing a field is refused, naming it", () => {
  // A field the page forgot to read would otherwise drop out of
  // every snapshot without a sound.
  const api = load();
  const record = finishedRecord(api);
  delete record.runToken;

  assert.throws(
    () => api.runSnapshotTiers(record), /missing runToken/
  );
});

// -- what comes back --

test("the fullest tier decodes to the run it came from", () => {
  const api = load();
  const record = finishedRecord(api, {
    thinking: "weighing the yeast",
    editedRunSaved: true,
    runSaved: true,
    lastSavedRunId: "2026-10-03_llada",
    lastSavedRevision: 2,
  });

  const state = api.runSnapshotDecode(
    tierText(api, record, 0), RESIDENT
  );

  same(
    api.runFramesToJson(state.frames),
    api.runFramesToJson(record.frames)
  );
  same(
    api.originalRunToJson(state.original),
    api.originalRunToJson(record.original)
  );
  same(
    api.runCandidatesToJson(state.candidates),
    api.runCandidatesToJson(record.candidates)
  );
  for (const field of STORED_FIELDS.slice(2)) {
    same(state[field], record[field]);
  }
});

test("the light tier decodes without per-token detail", () => {
  const api = load();
  const record = finishedRecord(api);

  const state = api.runSnapshotDecode(
    tierText(api, record, 2), RESIDENT
  );

  assert.equal(api.runFramesLength(state.frames), 3);
  assert.equal(state.frames.history.length, 3);
  assert.equal(state.frames.tokens.length, 0);
  assert.ok(api.runCandidatesIsEmpty(state.candidates));
});

test("an append run keeps its positions in the light tier", () => {
  const api = load();
  const record = finishedRecord(api, {
    frames: appendFramesOf(api, 4),
  });
  const tiers = api.runSnapshotTiers(record);

  const state = api.runSnapshotDecode(
    JSON.stringify(tiers[tiers.length - 1]), RESIDENT
  );

  assert.ok(api.runFramesIsAppend(state.frames));
  same(state.frames.positions, record.frames.positions);
});

test("the baseline comes back, with a fallback total", () => {
  const api = load();
  const record = editedRecord(api);
  const text = tierText(api, record, 0);
  const older = JSON.parse(text);
  delete older.originalTotalFrames;

  const state = api.runSnapshotDecode(text, RESIDENT);
  const fallback = api.runSnapshotDecode(
    JSON.stringify(older), RESIDENT
  );

  same(
    api.originalRunToJson(state.original),
    api.originalRunToJson(record.original)
  );
  assert.equal(
    fallback.original.totalFrames, api.runFramesLength(record.frames)
  );
});

test("a stopped run comes back stopped, and says why", () => {
  const api = load();
  const record = finishedRecord(api, {
    runInterrupted: true,
    runLostConnection: true,
  });
  const older = JSON.parse(tierText(api, record, 0));
  delete older.runInterrupted;
  delete older.runLostConnection;

  const state = api.runSnapshotDecode(
    tierText(api, record, 0), RESIDENT
  );
  const before = api.runSnapshotDecode(
    JSON.stringify(older), RESIDENT
  );

  assert.equal(state.runInterrupted, true);
  assert.equal(state.runLostConnection, true);
  assert.equal(before.runInterrupted, false);
  assert.equal(before.runLostConnection, false);
});

// -- the two stores --

test("an unedited run's baseline shares the live store", () => {
  const api = load();

  const state = api.runSnapshotDecode(
    tierText(api, finishedRecord(api), 0), RESIDENT
  );

  assert.equal(state.originalCandidates, state.candidates);
});

test("an edited run's two stores come back apart", () => {
  const api = load();
  const record = editedRecord(api);

  const state = api.runSnapshotDecode(
    tierText(api, record, 0), RESIDENT
  );

  assert.notEqual(state.originalCandidates, state.candidates);
  same(
    api.runCandidatesToJson(state.originalCandidates),
    api.runCandidatesToJson(record.originalCandidates)
  );
});

test("an edited run's lost baseline candidates read as none", () => {
  // The Original page then shows nothing, rather than the edited
  // run's candidates under the original's tokens.
  const api = load();
  const record = editedRecord(api);

  const state = api.runSnapshotDecode(
    tierText(api, record, 1), RESIDENT
  );

  assert.notEqual(state.originalCandidates, state.candidates);
  assert.ok(api.runCandidatesIsEmpty(state.originalCandidates));
  assert.equal(api.runCandidatesIsEmpty(state.candidates), false);
});

// -- older and malformed snapshots --

function decodeChanged(api, change) {
  const stored = JSON.parse(tierText(api, finishedRecord(api), 0));
  change(stored);
  return api.runSnapshotDecode(JSON.stringify(stored), RESIDENT);
}

test("a snapshot with no device still restores", () => {
  const api = load();

  const state = decodeChanged(api, (stored) => {
    delete stored.device;
  });

  assert.notEqual(state, null);
});

test("a snapshot without the run's identity reads as unknown", () => {
  const api = load();

  const missing = decodeChanged(api, (stored) => {
    delete stored.runToken;
    delete stored.worker;
  });
  const wrong = decodeChanged(api, (stored) => {
    stored.runToken = 7;
    stored.worker = 3;
  });

  for (const state of [missing, wrong]) {
    assert.equal(state.runToken, "");
    assert.equal(state.worker, "");
  }
});

test("without the run's prompt, the box text stands in", () => {
  const api = load();

  const padded = decodeChanged(api, (stored) => {
    delete stored.runPrompt;
    stored.prompt = "  explain yeast  ";
  });
  const empty = decodeChanged(api, (stored) => {
    delete stored.runPrompt;
    stored.prompt = "";
  });

  assert.equal(padded.runPrompt, "explain yeast");
  assert.equal(empty.runPrompt, null);
  assert.equal(api.runSnapshotRunPrompt({}), null);
});

test("malformed values read as absent", () => {
  const api = load();

  const state = decodeChanged(api, (stored) => {
    stored.provenance = "llada";
    stored.promptLen = "9";
    stored.lastSavedRevision = "2";
    delete stored.lastRunTotalSteps;
    delete stored.remaskEdits;
    delete stored.params;
  });

  assert.equal(state.provenance, null);
  assert.equal(state.promptLen, null);
  assert.equal(state.lastSavedRevision, null);
  assert.equal(state.lastRunTotalSteps, null);
  same(state.remaskEdits, []);
  assert.equal(state.params, null);
});

test("a foreign or too-short snapshot is refused", () => {
  const api = load();
  const text = tierText(api, finishedRecord(api), 0);

  assert.equal(
    api.runSnapshotDecode(text, { model: "smollm3", device: "cuda" }),
    null
  );
  assert.equal(
    api.runSnapshotDecode(text, { model: "llada", device: "cpu" }),
    null
  );
  assert.equal(
    decodeChanged(api, (stored) => {
      stored.frameCount = 1;
    }),
    null
  );
});

test("text that is not a snapshot restores nothing", () => {
  const api = load();

  for (const text of [null, "", "{not json", "null", "[]", "5"]) {
    assert.equal(api.runSnapshotDecode(text, RESIDENT), null, text);
  }
});
