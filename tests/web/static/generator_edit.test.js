// The generator edit controller, driven without app.js.
//
// Strategy: load the phase table and controller into one vm context,
// then supply a run-controller stand-in, inert render controllers and
// narrow page callbacks. Passing proves mutable edit state and DOM
// references stay private, every read follows the current run, legal
// phase moves preserve the guided workflow, edit records and
// transport intents keep their exact meaning, a stop before the
// first resumed frame rolls back, and reset retires the whole
// session.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const { makeElement } = require("./dom_stub.js");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);

function load() {
  const elements = new Map();
  const document = {
    activeElement: makeElement("active"),
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, makeElement(id));
      }
      return elements.get(id);
    },
    addEventListener() {},
  };
  document.getElementById("btn-retry-edit").title = "Retry";
  document.getElementById("btn-continue-edit").title = "Continue";
  const context = vm.createContext({
    console,
    document,
    Math,
    Object,
    Number,
    String,
    Array,
    Error,
    TypeError,
    RangeError,
  });
  for (const name of ["run_phases.js", "generator_edit.js"]) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context,
      { filename: name }
    );
  }
  return { context, elements };
}

function token(position) {
  return {
    t: String(position),
    m: position === 0,
    id: 100 + position,
    c: 0.5,
  };
}

function frames(count) {
  const output = [];
  for (let frame = 0; frame < count; frame++) {
    output.push([token(0), token(1), token(2)]);
  }
  return output;
}

function fakeRun() {
  let heldFrames = frames(4);
  let originalFrames = heldFrames.length;
  let interrupted = false;
  let savedEdit = false;
  let saving = false;
  let resident = "worker:1";
  let madeBy = "worker:1";
  const checkpoints = new WeakMap();
  return {
    setFrameCount(count) {
      heldFrames = frames(count);
      originalFrames = count;
    },
    setInterrupted(value) {
      interrupted = value;
    },
    setEditedSaved(value) {
      savedEdit = value;
    },
    frameCount: () => heldFrames.length,
    frameTokens: (index) => heldFrames[index].slice(),
    frameCanvas: () => 0,
    frameIsMultiCanvas: () => false,
    totalSteps: () => 8,
    originalCaptured: () => true,
    originalTotalFrames: () => originalFrames,
    captureCheckpoint() {
      const checkpoint = Object.freeze({});
      checkpoints.set(checkpoint, {
        heldFrames: heldFrames.map((row) => row.slice()),
        interrupted,
      });
      return checkpoint;
    },
    restoreCheckpoint(checkpoint) {
      const state = checkpoints.get(checkpoint);
      assert.ok(state, "unknown run checkpoint");
      heldFrames = state.heldFrames.map((row) => row.slice());
      interrupted = state.interrupted;
    },
    truncate(count) {
      heldFrames.length = count;
    },
    truncateAlternatives() {},
    runToken: () => "nonce:1",
    saving: () => saving,
    editedSaved: () => savedEdit,
    interrupted: () => interrupted,
    editIdentity: () => ({
      lostConnection: false,
      madeBy,
      resident,
    }),
    adoptResidentWorker(worker) {
      resident = worker;
    },
  };
}

function harness(settings) {
  const { context, elements } = load();
  const run = fakeRun();
  const external = {
    capabilities: Object.assign({
      supports_resume: true,
      supports_substitution: false,
    }, settings || {}),
    generating: false,
    statuses: [],
    runStatuses: [],
    rewinds: [],
    resumes: [],
    substitutions: [],
    saves: 0,
    commits: 0,
    canEdit: true,
  };
  const canvas = {
    activate() {},
    deactivate() {},
    refreshControls() {},
    renderFrame() {},
    renderTargetPlaceholder() {},
  };
  const readouts = {
    activate() {},
    deactivate() {},
    updateProfile() {},
    refreshStop() {},
  };
  const candidates = {
    alternativesAvailable: () => true,
    hidePopover() {},
  };
  const edit = context.generatorEditCreate({
    run,
    canvas,
    readouts,
    candidates,
    readCapabilities: () => external.capabilities,
    readGenerating: () => external.generating,
    readDiffusionEffect: () => false,
    revealText: (_element, _text, done) => done(),
    dissolveText: (_element, done) => done(),
    renderFrameReadout() {},
    setStatus: (message) => external.statuses.push(message),
    setGenerating: (active) => {
      external.generating = active;
    },
    setSaveAvailable() {},
    resetStatus() {},
    startRunStatus: (label) => {
      external.runStatuses.push(label);
    },
    primaryStateChanged() {},
    requestSave: () => {
      external.saves += 1;
      return Promise.resolve(true);
    },
    requestCommit: () => {
      external.commits += 1;
      return Promise.resolve(true);
    },
    requestRewind: (intent) => {
      external.rewinds.push(host(intent));
      return true;
    },
    requestResume: (intent) => {
      external.resumes.push(host(intent));
      return true;
    },
    requestSubstitute: (intent) => {
      external.substitutions.push(host(intent));
      return true;
    },
    canEditConversation: () => external.canEdit,
  });
  return { edit, run, external, elements };
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function selectAt(edit, frame, position) {
  edit.enterFrames();
  edit.navigate(frame);
  edit.selectFrame();
  edit.togglePosition(position);
}

test("mutable edit state and DOM references stay private", () => {
  const { edit } = harness();
  selectAt(edit, 2, 1);
  edit.lockSelection();

  assert.equal(edit.runPhase, undefined);
  assert.equal(edit.remaskEdits, undefined);
  assert.equal(edit.remaskedPositions, undefined);
  assert.equal(edit.scrubberSlider, undefined);
  assert.equal(edit.outputArea, undefined);

  const phase = edit.phaseState();
  phase.lockedEdits[0].token_positions.length = 0;
  assert.deepEqual(
    host(edit.phaseState().lockedEdits[0].token_positions),
    [1]
  );
});

test("activation always opens the latest current run frame", () => {
  const { edit, run } = harness();
  edit.activate();
  assert.equal(edit.currentFrame(), 3);

  run.setFrameCount(6);
  edit.activate();

  assert.equal(edit.currentFrame(), 5);
});

test("guided actions follow the declared phase transitions", () => {
  const { edit } = harness();
  selectAt(edit, 1, 1);
  assert.equal(edit.phaseState().mode, "edit");

  edit.lockSelection();
  assert.equal(edit.phaseState().mode, "choice");
  edit.chooseAnotherFrame();
  assert.equal(edit.phaseState().mode, "select_target");
  edit.navigate(3);
  edit.runToCurrentFrame();

  assert.equal(edit.phaseState().mode, "generating");
});

test("a resume records one edit and one intent", () => {
  const { edit, external } = harness();
  selectAt(edit, 2, 1);
  edit.lockSelection();

  edit.resumeToEnd();

  assert.deepEqual(host(edit.readArtifacts().remaskEdits), [{
    frame_index: 2,
    token_positions: [1],
  }]);
  assert.deepEqual(external.resumes, [{
    frameIndex: 2,
    remaskPositions: [1],
    targetFrame: null,
    continueRun: false,
    runToken: "nonce:1",
  }]);
  assert.equal("type" in external.resumes[0], false);
});

test("What If preserves typed substitution intent", () => {
  const { edit, external } = harness({
    supports_resume: false,
    supports_substitution: true,
  });
  edit.enterWhatIf();

  edit.substitute({
    position: 1,
    tokenId: 701,
    typedText: " dog",
  });

  assert.deepEqual(external.substitutions, [{
    position: 1,
    tokenId: 701,
    typedText: " dog",
    runToken: "nonce:1",
  }]);
  assert.deepEqual(host(edit.readArtifacts().remaskEdits), [{
    frame_index: 1,
    token_positions: [1],
  }]);
});

test("random selection chooses only resolved positions", () => {
  const { edit, elements } = harness();
  edit.wire();
  edit.enterFrames();
  edit.selectFrame();
  const slider = elements.get("remask-random-slider");
  slider.value = "2";

  elements.get("btn-remask-shuffle").dispatch("click");

  assert.equal(
    elements.get("remask-random-total").textContent,
    "2"
  );
  assert.deepEqual(
    Object.keys(edit.readoutsSettings().remaskedPositions),
    ["1", "2"]
  );
});

test("a saved edit locks both counterfactual entries", () => {
  const { edit, run, elements } = harness();
  run.setEditedSaved(true);

  edit.refreshLocks();

  for (const id of ["btn-edit-frames", "btn-what-if"]) {
    const button = elements.get(id);
    assert.equal(button.classList.contains("is-locked"), true);
    assert.match(button.title, /already has a saved edit/);
  }
});

test("a frozen conversation response refuses editing clearly", () => {
  const { edit, external, elements } = harness();
  external.canEdit = false;

  edit.refreshLocks();
  assert.equal(edit.enterFrames(), false);
  assert.match(
    elements.get("btn-edit-frames").title,
    /latest response/
  );
  assert.equal(edit.phaseState().mode, null);
});

test("a stop before the first resumed frame rolls back", () => {
  const { edit, run, external } = harness();
  selectAt(edit, 2, 1);
  edit.lockSelection();
  edit.resumeToEnd();
  assert.equal(run.frameCount(), 2);

  const handled = edit.finishStream({ cancelled: true });

  assert.equal(handled, true);
  assert.equal(run.frameCount(), 4);
  assert.equal(edit.phaseState().mode, "choice");
  assert.equal(edit.currentFrame(), 2);
  assert.deepEqual(host(edit.readArtifacts().remaskEdits), []);
  assert.match(external.statuses.at(-1), /run is unchanged/);
});

test("reset retires the whole edit session", () => {
  const { edit } = harness();
  selectAt(edit, 2, 1);
  edit.lockSelection();
  edit.resumeToEnd();

  edit.reset();

  assert.equal(edit.phaseState().mode, null);
  assert.equal(edit.active(), false);
  assert.equal(edit.resuming(), false);
  assert.deepEqual(host(edit.readArtifacts().remaskEdits), []);
  assert.deepEqual(
    host(edit.readoutsSettings().remaskedPositions),
    {}
  );
});
