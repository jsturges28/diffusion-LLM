// Tests for the shared token span builder in overlays.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// pattern the other browser tests use, and drive
// overlaysSyncTokenSpan against a fake span. No DOM is needed: that
// function never creates a node, it only reads and writes the five
// properties this fake carries, and overlays.js does no DOM work at
// the top level.
//
// This is the span builder both pages draw every token through, and
// it had no test. The gap that prompted one: the generator's live
// streaming view passed an empty options object, so a mask's opacity
// was graded only when scrubbing back over a finished run and the
// canvas stayed flat while it was being written, which is when the
// reading is most interesting. That was a caller's bug rather than
// this function's, but nothing here pinned the behavior the caller
// was failing to ask for.
//
// What passing proves: opacity is written for a mask when a hook
// asks, cleared rather than left stale on a span being reused for a
// different frame, and never applied to a resolved token. The clear
// is the subtle one, because live rendering keeps one span per
// position for the whole run and updates it in place.
//
// The reveal tests below cover the other half of what this function
// decides: whether an unsettled position draws the glyph or the
// token the model is currently holding there. The invariant worth
// stating once is that the two are independent. Revealing changes
// the text and nothing else, so a revealed position keeps token-mask
// and keeps its confidence fade, and therefore still reads as a
// guess rather than as an answer.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "src",
  "web",
  "static",
  "overlays.js"
);

const MASK = "\u2591";

function load() {
  const sandbox = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "overlays.js",
  });
  return sandbox;
}

// Stands in for one span already on the page. Only the members
// overlaysSyncTokenSpan touches, so a write it makes that this does
// not model would throw rather than pass unnoticed.
function fakeSpan() {
  const attributes = {};
  return {
    className: "",
    textContent: "",
    style: { color: "", opacity: "" },
    getAttribute: (name) =>
      Object.prototype.hasOwnProperty.call(attributes, name)
        ? attributes[name]
        : null,
    setAttribute: (name, value) => {
      attributes[name] = value;
    },
  };
}

function masked(confidence) {
  const token = { t: " the", m: true, id: 7 };
  if (confidence !== null) {
    token.c = confidence;
  }
  return token;
}

function resolved() {
  return { t: " the", m: false, id: 7, c: 0.9 };
}

// The generator's hook, in miniature: grade a mask by confidence,
// leave a resolved token alone.
function gradeMasks(index, tok, isMask) {
  if (!isMask) {
    return null;
  }
  return tok && typeof tok.c === "number" ? tok.c : 0.25;
}

test("a mask is graded when a hook asks", () => {
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.8), MASK, {
    opacityFor: gradeMasks,
  });

  assert.equal(span.style.opacity, "0.8");
  assert.equal(span.textContent, MASK);
});

test("a mask is flat when no hook asks", () => {
  // Analytics passes no callbacks, and the live view used to pass
  // none either, which is the bug this file was added for.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.8), MASK, {});

  assert.equal(span.style.opacity, "");
});

test("a resolved token is never graded", () => {
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, resolved(), MASK, {
    opacityFor: gradeMasks,
  });

  assert.equal(span.style.opacity, "");
  assert.equal(span.textContent, " the");
});

test("a reused span drops the previous frame's grading", () => {
  // The live path keeps one span per position for the whole run, so
  // a value left behind here would follow a token past its reveal
  // and show a resolved word at a mask's opacity.
  const sandbox = load();
  const span = fakeSpan();
  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.3), MASK, {
    opacityFor: gradeMasks,
  });
  assert.equal(span.style.opacity, "0.3");

  sandbox.overlaysSyncTokenSpan(span, 0, resolved(), MASK, {
    opacityFor: gradeMasks,
  });

  assert.equal(span.style.opacity, "");
});

test("a reused span follows a mask's confidence upward", () => {
  const sandbox = load();
  const span = fakeSpan();
  const climb = [0.1, 0.4, 0.9];
  const seen = [];

  for (const value of climb) {
    sandbox.overlaysSyncTokenSpan(span, 0, masked(value), MASK, {
      opacityFor: gradeMasks,
    });
    seen.push(span.style.opacity);
  }

  assert.deepEqual(seen, ["0.1", "0.4", "0.9"]);
});

test("a hole in the canvas is masked without a token", () => {
  // Two stacked layers only line up if both emit a span per
  // position, so a missing token draws the glyph rather than
  // throwing on tok.t.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 3, null, MASK, {
    opacityFor: gradeMasks,
  });

  assert.equal(span.textContent, MASK);
  assert.equal(span.style.opacity, "0.25");
});

test("every span carries the position interactions key off", () => {
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 12, resolved(), MASK, {});

  assert.equal(span.getAttribute("data-pos"), "12");
  assert.match(span.className, /token-span/);
});

test("a mask and a resolved token are classed apart", () => {
  const sandbox = load();
  const maskSpan = fakeSpan();
  const wordSpan = fakeSpan();

  sandbox.overlaysSyncTokenSpan(maskSpan, 0, masked(null), MASK, {});
  sandbox.overlaysSyncTokenSpan(wordSpan, 1, resolved(), MASK, {});

  assert.match(maskSpan.className, /token-mask/);
  assert.match(wordSpan.className, /token-resolved/);
});

// ---- Revealing the candidate behind a mask ----

test("a mask draws the glyph by default", () => {
  // The flag absent is every caller written before the setting
  // existed, and they must be unaffected.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {});

  assert.equal(span.textContent, MASK);
});

test("a mask draws its candidate when the flag is set", () => {
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {
    revealMask: true,
  });

  assert.equal(span.textContent, " the");
});

test("a revealed candidate is still classed as a mask", () => {
  // The whole feature rests on this. A word drawn at a position that
  // has not settled must not be readable as one that has.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {
    revealMask: true,
  });

  assert.match(span.className, /token-mask/);
  assert.doesNotMatch(span.className, /token-resolved/);
});

test("a revealed candidate keeps its confidence fade", () => {
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {
    revealMask: true,
    opacityFor: gradeMasks,
  });

  assert.equal(span.textContent, " the");
  assert.equal(span.style.opacity, "0.4");
});

test("a hole draws the glyph even with the reveal on", () => {
  // There is no candidate to name at a position with no token, and
  // reading tok.t off null would throw.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 3, null, MASK, {
    revealMask: true,
  });

  assert.equal(span.textContent, MASK);
});

test("a run saved before the capture reveals nothing new", () => {
  // LLaDA wrote the glyph into t for years, and frame 0 still does,
  // because the model has not looked at the canvas yet. Substituting
  // it yields the glyph, so the setting is simply inert there rather
  // than a hole to guard against.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(
    span, 0, { t: MASK, m: true, id: 126336 }, MASK,
    { revealMask: true }
  );

  assert.equal(span.textContent, MASK);
  assert.match(span.className, /token-mask/);
});

test("an empty candidate falls back to the glyph", () => {
  // An empty span collapses, and two stacked layers stop lining up
  // the moment one of them is a position short.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(
    span, 0, { t: "", m: true, id: 7 }, MASK, { revealMask: true }
  );

  assert.equal(span.textContent, MASK);
});

test("a hook-masked token is not revealed", () => {
  // A remask selection is the app hiding a settled token to show
  // what the next run will redraw. Revealing it would put the word
  // back and undo the point of the selection.
  const sandbox = load();
  const span = fakeSpan();

  sandbox.overlaysSyncTokenSpan(span, 0, resolved(), MASK, {
    revealMask: true,
    maskedFor: () => true,
  });

  assert.equal(span.textContent, MASK);
  assert.match(span.className, /token-mask/);
});

test("a reused span puts the glyph back when the reveal is off", () => {
  // The live path updates one span per position in place, so a
  // revealed word left behind would outlive the setting that drew it.
  const sandbox = load();
  const span = fakeSpan();
  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {
    revealMask: true,
  });
  assert.equal(span.textContent, " the");

  sandbox.overlaysSyncTokenSpan(span, 0, masked(0.4), MASK, {});

  assert.equal(span.textContent, MASK);
});
