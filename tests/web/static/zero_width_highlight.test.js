// A token that renders to nothing still shows where it is.
//
// Strategy: build real token spans through the shared span builder,
// then ask each page's setTokenHighlight to light a position and read
// the classes it left behind. A newline token has to come away with
// the extra class that CSS turns into a standing marker; a token with
// glyphs must not, because that marker would sit on top of its first
// character.
//
// The bug being pinned: the cross-highlight is a background plus a
// box-shadow on an inline span. A newline occupies no horizontal
// space, so both painted nothing, and sweeping the entropy profile
// onto such a position lit the text area not at all. The column was
// plainly pointing somewhere and the somewhere looked empty, which
// reads as the profile being wrong rather than as the token being
// invisible.
//
// Passing proves the predicate classifies line breaks and the empty
// string as zero-width while leaving spaces and tabs alone, that both
// the generator and Analytics tag such a token when they light it,
// and that clearing a highlight takes the extra class with it.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

// A short canvas with a line break in the middle of it, which is the
// shape the maintainer hit: position 2 is a newline sitting between
// two ordinary words.
const WORDS = ["The", " cat", "\n", " sat"];

function tokens() {
  return WORDS.map((word, at) => ({
    t: word,
    m: false,
    id: 1000 + at,
    c: 0.9,
    e: 0.5,
  }));
}

function bootFetch() {
  return function (url) {
    const body = String(url).indexOf("/api/analytics/runs") === 0
      ? []
      : { success: true, collections: [] };
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    });
  };
}

// Fill a container with real spans, built by the same function both
// pages use, so the classes under test are read off the genuine
// markup rather than off hand-made stand-ins.
function fillWithTokens(context, container) {
  container.textContent = "";
  const options = {
    colorFor: () => null,
    classFor: () => "",
    revealMask: false,
    opacityFor: () => 1,
  };
  const frame = tokens();
  for (let at = 0; at < frame.length; at++) {
    container.appendChild(
      context.overlaysBuildTokenSpan(at, frame[at], "\u2591", options)
    );
  }
}

function classesAt(container, position) {
  const spans = container.querySelectorAll(
    "[data-pos=\"" + position + "\"]"
  );
  assert.equal(spans.length, 1, "expected one span at " + position);
  return spans[0].classes;
}

// -- the predicate --

test("a line break is zero-width", () => {
  const { context } = loadPage({});

  assert.equal(context.overlaysTokenIsZeroWidth("\n"), true);
  assert.equal(context.overlaysTokenIsZeroWidth("\r\n"), true);
  assert.equal(context.overlaysTokenIsZeroWidth("\n\n"), true);
});

test("the empty string is zero-width", () => {
  const { context } = loadPage({});

  assert.equal(context.overlaysTokenIsZeroWidth(""), true);
});

test("whitespace that occupies space is not", () => {
  // The boundary. A space and a tab paint a visible box, so tinting
  // them already worked and standing a marker over them would be a
  // second mark for no reason.
  const { context } = loadPage({});

  assert.equal(context.overlaysTokenIsZeroWidth(" "), false);
  assert.equal(context.overlaysTokenIsZeroWidth("\t"), false);
  assert.equal(context.overlaysTokenIsZeroWidth(" \n"), false);
});

test("a word is not zero-width", () => {
  const { context } = loadPage({});

  assert.equal(context.overlaysTokenIsZeroWidth("cat"), false);
});

test("a missing text is not zero-width", () => {
  // Defensive rather than expected: a token record always carries a
  // string. Answering true would put a marker on every position of a
  // malformed frame, which is a loud failure for a quiet cause.
  const { context } = loadPage({});

  assert.equal(context.overlaysTokenIsZeroWidth(undefined), false);
  assert.equal(context.overlaysTokenIsZeroWidth(null), false);
});

// -- the generator --

test("the generator marks a lit newline", () => {
  const { context } = loadPage({});
  fillWithTokens(context, context.outputArea);

  context.setTokenHighlight(2);

  const classes = classesAt(context.outputArea, 2);
  assert.equal(classes.has("token-cross-highlight"), true);
  assert.equal(classes.has("token-zero-width"), true);
});

test("the generator leaves a lit word unmarked", () => {
  const { context } = loadPage({});
  fillWithTokens(context, context.outputArea);

  context.setTokenHighlight(1);

  const classes = classesAt(context.outputArea, 1);
  assert.equal(classes.has("token-cross-highlight"), true);
  assert.equal(classes.has("token-zero-width"), false);
});

test("clearing takes the marker with it", () => {
  // Both classes are added together and have to leave together: a
  // stranded token-zero-width would stand a marker on a token the
  // pointer had already left.
  const { context } = loadPage({});
  fillWithTokens(context, context.outputArea);
  context.setTokenHighlight(2);

  context.clearTokenHighlight();

  const classes = classesAt(context.outputArea, 2);
  assert.equal(classes.has("token-cross-highlight"), false);
  assert.equal(classes.has("token-zero-width"), false);
});

test("moving the highlight moves the marker", () => {
  // The sweep. Going from the newline to the word beside it must not
  // leave the marker behind, and the guard that skips a repeated
  // position must not skip this.
  const { context } = loadPage({});
  fillWithTokens(context, context.outputArea);
  context.setTokenHighlight(2);

  context.setTokenHighlight(3);

  assert.equal(
    classesAt(context.outputArea, 2).has("token-zero-width"), false
  );
  assert.equal(
    classesAt(context.outputArea, 3).has("token-zero-width"), false
  );
});

// -- Analytics --

test("Analytics marks a lit newline too", () => {
  // The same behaviour on the other surface, which keeps its own copy
  // of setTokenHighlight over its own container.
  const { context } = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  fillWithTokens(context, context.overlayOutput);

  context.setTokenHighlight(2);

  const classes = classesAt(context.overlayOutput, 2);
  assert.equal(classes.has("token-cross-highlight"), true);
  assert.equal(classes.has("token-zero-width"), true);
});

test("Analytics leaves a lit word unmarked", () => {
  const { context } = loadPage({
    scripts: ANALYTICS_SCRIPTS, fetchImpl: bootFetch(),
  });
  fillWithTokens(context, context.overlayOutput);

  context.setTokenHighlight(0);

  const classes = classesAt(context.overlayOutput, 0);
  assert.equal(classes.has("token-cross-highlight"), true);
  assert.equal(classes.has("token-zero-width"), false);
});
