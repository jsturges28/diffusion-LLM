// The generator candidate controller, driven without app.js.
//
// Strategy: compose the classic-script factory around narrow fakes
// for generatorRun, generatorCanvas and generatorReadouts. Drive its
// real DOM listeners and typed-token debounce through the shared DOM
// stub, while recording the three intents handed back to the page.
//
// Passing proves popover and typed-token state stay in the closure,
// both run shapes read through generatorRun, paging follows the
// canvas blend, stale tokenizer replies cannot replace a newer
// draft, and tokenize, probe and substitution remain intents rather
// than transport or run mutation owned by this controller.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "overlays.js",
  "generator_candidates.js",
];

function candidate(id, text, probability) {
  return { id, t: text, p: probability };
}

function defaultState() {
  return {
    append: true,
    frame: 2,
    scrubberActive: true,
    editing: false,
    substituting: true,
    remaskEdits: [{
      frame_index: 1,
      token_positions: [1],
    }],
    tokenizer: {
      "class": "TestTokenizer",
      vocab_size: 128000,
    },
    vocabSize: 128000,
    tokens: [
      { id: 1, t: " first" },
      { id: 2, t: " base" },
    ],
    originalTokens: [
      { id: 1, t: " first" },
      { id: 12, t: " old" },
    ],
    editedAlternatives: {
      1: [
        candidate(21, " offered", 0.6),
        candidate(22, " other", 0.2),
      ],
    },
    originalAlternatives: {
      1: [
        candidate(11, " old", 0.7),
        candidate(13, " prior", 0.1),
      ],
    },
    editedSamplerAlternatives: {
      1: {
        support: 3,
        candidates: [
          { id: 31, t: " sampled", p: 0.7, rank: 1, g: true },
          { id: 32, t: " red", p: 0.2, rank: 2, g: false },
        ],
      },
    },
    originalSamplerAlternatives: {
      1: {
        support: 2,
        candidates: [
          { id: 14, t: " before", p: 0.8, rank: 1, g: true },
        ],
      },
    },
    diffusionEdited: {
      frame: 2,
      set: {
        h: 21,
        c: [candidate(21, " edited", 0.8)],
      },
    },
    diffusionOriginal: {
      frame: 1,
      set: {
        h: 11,
        c: [candidate(11, " original", 0.9)],
      },
    },
    favorsOriginal: true,
    blendActive: true,
    candidateCalls: [],
    candidateReadings: [],
    tokenReadings: [],
    clearTokenHovers: 0,
    profileUpdates: 0,
    tokenizeIntents: [],
    probeIntents: [],
    substituteIntents: [],
    acceptSubstitution: true,
  };
}

function fakeRun(state) {
  return {
    frameIsAppend() {
      return state.append;
    },
    hasAlternatives(original) {
      const source = original
        ? state.originalAlternatives
        : state.editedAlternatives;
      return Object.keys(source).length > 0;
    },
    positionAlternatives(position, original) {
      const source = original
        ? state.originalAlternatives
        : state.editedAlternatives;
      return source[position] || null;
    },
    positionSamplerAlternatives(position, original) {
      const source = original
        ? state.originalSamplerAlternatives
        : state.editedSamplerAlternatives;
      return source[position] || null;
    },
    frameTokens() {
      return state.tokens;
    },
    originalTokensLast() {
      return state.originalTokens;
    },
    originalTokenFrames() {
      return 3;
    },
    candidateSet(frame, position, original) {
      state.candidateCalls.push({
        frame, position, original,
      });
      return original
        ? state.diffusionOriginal
        : state.diffusionEdited;
    },
    frameCanvas() {
      return 0;
    },
  };
}

function fakeCanvas(state) {
  return {
    blendFavorsOriginal() {
      return state.favorsOriginal;
    },
    blendActive() {
      return state.blendActive;
    },
  };
}

function fakeReadouts(state) {
  return {
    setCandidateHover(reading) {
      state.candidateReadings.push(reading);
    },
    setTokenHover(position) {
      state.tokenReadings.push(position);
    },
    clearTokenHover() {
      state.clearTokenHovers += 1;
    },
    updateProfile() {
      state.profileUpdates += 1;
    },
  };
}

function completeOptions(state) {
  return {
    run: fakeRun(state),
    canvas: fakeCanvas(state),
    readouts: fakeReadouts(state),
    readState() {
      return {
        frame: state.frame,
        scrubberActive: state.scrubberActive,
        editing: state.editing,
        substituting: state.substituting,
        remaskEdits: state.remaskEdits,
        tokenizer: state.tokenizer,
        vocabSize: state.vocabSize,
      };
    },
    requestTokenize(intent) {
      state.tokenizeIntents.push(intent);
      return true;
    },
    requestProbe(intent) {
      state.probeIntents.push(intent);
      return true;
    },
    requestSubstitute(intent) {
      state.substituteIntents.push(intent);
      return state.acceptSubstitution;
    },
  };
}

function controller(settings) {
  const state = settings || defaultState();
  const page = loadPage({ scripts: SCRIPTS });
  const instance = page.context.generatorCandidatesCreate(
    completeOptions(state)
  );
  instance.wire();
  return {
    page,
    state,
    instance,
    popover: page.registry.get("token-alts-popover"),
    output: page.registry.get("output-area"),
  };
}

function descendants(node) {
  const found = [];
  const pending = node.children.slice();
  while (pending.length > 0) {
    const next = pending.shift();
    found.push(next);
    pending.push(...next.children);
  }
  return found;
}

function withClass(node, name) {
  return descendants(node).filter(
    (child) => child.classList.contains(name)
  );
}

function titleOf(popover) {
  return withClass(popover, "alt-heading")[0]
    .children[0].textContent;
}

function rowIds(popover) {
  return withClass(popover, "alt-row").map(
    (row) => Number(row.getAttribute("data-alt-id"))
  );
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function waitForDebounce() {
  return new Promise((resolve) => {
    setTimeout(resolve, 140);
  });
}

test("every controller and callback is required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const names = Object.keys(completeOptions(defaultState()));

  for (const name of names) {
    const options = completeOptions(defaultState());
    options[name] = null;
    assert.throws(
      () => page.context.generatorCandidatesCreate(options),
      { name: "TypeError" },
      name
    );
  }
});

test("mutable popover and typed state stay private", () => {
  const { page, instance } = controller();

  assert.equal(typeof instance.showPopover, "function");
  for (const name of [
    "popoverPosition",
    "popoverPage",
    "typedDraft",
    "typedPreview",
    "typedToken",
    "typedMeasure",
    "typedPreviewTimer",
  ]) {
    assert.equal(page.context[name], undefined, name);
  }
});

test("append paging follows the blend and tokenizer", () => {
  const { instance, popover } = controller();

  instance.showPopover(1, null);

  assert.equal(titleOf(popover), "Position 2: Original");
  assert.deepEqual(rowIds(popover), [11, 13]);
  assert.equal(
    withClass(popover, "alt-tokenizer")[0].textContent,
    "TestTokenizer \u00B7 128k vocab"
  );

  const edited = withClass(popover, "alt-pager-btn")
    .find((button) =>
      button.getAttribute("aria-label") === "Edited run"
    );
  edited.dispatch("click", { stopPropagation() {} });

  assert.equal(titleOf(popover), "Position 2: Edited");
  assert.deepEqual(rowIds(popover), [21, 22]);
});

test("Model and Sampler are independent from run paging", () => {
  const state = defaultState();
  state.favorsOriginal = false;
  const { instance, popover } = controller(state);
  instance.showPopover(1, null);
  const sampler = descendants(popover).find((node) =>
    node.getAttribute("data-alt-distribution") === "sampler"
  );

  sampler.dispatch("click", {
    preventDefault() {},
    stopPropagation() {},
  });
  assert.deepEqual(rowIds(popover), [31, 32]);
  const activeSampler = descendants(popover).find((node) =>
    node.getAttribute("data-alt-distribution") === "sampler"
  );
  assert.equal(
    activeSampler.getAttribute("aria-pressed"),
    "true"
  );
  popover.dispatch("click", {
    target: withClass(popover, "alt-row")[0],
  });
  assert.deepEqual(state.substituteIntents, []);

  const original = withClass(popover, "alt-pager-btn")
    .find((button) =>
      button.getAttribute("aria-label") === "Original run"
    );
  original.dispatch("click", { stopPropagation() {} });
  assert.deepEqual(rowIds(popover), [14]);

  instance.hidePopover();
  instance.showPopover(1, null);
  assert.deepEqual(rowIds(popover), [31, 32]);

  instance.outputReset();
  instance.showPopover(1, null);
  assert.deepEqual(rowIds(popover), [31, 32]);

  instance.reset();
  instance.showPopover(1, null);
  assert.deepEqual(rowIds(popover), [21, 22]);
});

test("diffusion candidates read the scrubbed frame", () => {
  const state = defaultState();
  state.append = false;
  state.blendActive = false;
  const { instance, popover } = controller(state);

  instance.showPopover(1, null);

  assert.deepEqual(host(state.candidateCalls), [{
    frame: 2,
    position: 1,
    original: false,
  }]);
  assert.equal(titleOf(popover), "Position 2: candidates");
  assert.deepEqual(rowIds(popover), [21]);
});

test("token hover and touch own the popover lifecycle", () => {
  const { instance, state, popover, output, page } = controller();
  const span = page.document.createElement("span");
  span.className = "token-span";
  span.setAttribute("data-pos", "1");
  output.appendChild(span);

  output.dispatch("mouseover", { target: span });

  assert.deepEqual(state.tokenReadings, [1]);
  assert.equal(popover.hidden, false);
  assert.equal(titleOf(popover), "Position 2: Original");

  output.dispatch("mouseleave");

  assert.equal(popover.hidden, true);
  assert.equal(state.clearTokenHovers, 1);

  output.dispatch("click", { target: span });
  assert.equal(popover.hidden, false);
  page.document.dispatch("pointerdown");
  assert.equal(popover.hidden, true);

  assert.equal(instance.alternativesAvailable(), true);
});

test("typed replies fence callback intents", async () => {
  const state = defaultState();
  state.favorsOriginal = false;
  const { instance, popover } = controller(state);
  instance.showPopover(1, null);
  let field = withClass(popover, "typed-input")[0];
  let confirm = withClass(popover, "typed-confirm")[0];

  assert.equal(field.value, " ");
  field.dispatch("focus");
  field.value = " custom";
  field.dispatch("input");
  await waitForDebounce();

  assert.deepEqual(host(state.tokenizeIntents), [{
    text: " custom",
    requestId: 1,
  }]);

  field.value = " newer";
  field.dispatch("input");
  await waitForDebounce();
  assert.equal(state.tokenizeIntents[1].requestId, 2);

  instance.handleTokenizeResult({
    request_id: 1,
    text: " custom",
    pieces: [{ id: 98, t: " custom" }],
    count: 1,
  });
  assert.equal(confirm.disabled, true);

  instance.handleTokenizeResult({
    request_id: 2,
    text: " stale text",
    pieces: [{ id: 97, t: " stale" }],
    count: 1,
  });
  assert.equal(confirm.disabled, true);

  instance.handleTokenizeResult({
    request_id: 2,
    text: " newer",
    pieces: [{ id: 99, t: " newer" }],
    count: 1,
  });
  assert.equal(confirm.disabled, false);

  confirm.dispatch("click");
  assert.deepEqual(host(state.probeIntents), [{
    position: 1,
    tokenId: 99,
    requestId: 1,
  }]);

  instance.handleProbeResult({
    request_id: 0,
    token_id: 99,
    probability: 0.9,
    rank: 1,
    vocab_size: 128000,
  });
  let probability = withClass(popover, "typed-prob")[0];
  assert.equal(probability.textContent, "\u2026");

  instance.handleProbeResult({
    request_id: 1,
    token_id: 99,
    probability: 0.0005,
    rank: 42000,
    vocab_size: 128000,
  });
  probability = withClass(popover, "typed-prob")[0];
  assert.equal(probability.textContent, "<0.1%");
  assert.match(probability.title, /rank 42,000 of 128,000/);

  const typedRow = withClass(popover, "typed-solid")[0];
  popover.dispatch("click", { target: typedRow });

  assert.deepEqual(host(state.substituteIntents), [{
    position: 1,
    tokenId: 99,
    typedText: " newer",
  }]);
  assert.equal(popover.hidden, true);
  field = withClass(popover, "typed-input")[0];
  assert.equal(field, undefined);
});

test("captured rows send untyped substitution intents", () => {
  const state = defaultState();
  state.favorsOriginal = false;
  const { instance, popover } = controller(state);
  instance.showPopover(1, null);
  const row = withClass(popover, "alt-row")[0];

  popover.dispatch("click", { target: row });

  assert.deepEqual(host(state.substituteIntents), [{
    position: 1,
    tokenId: 21,
    typedText: null,
  }]);
});
