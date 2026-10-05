// Shared KGW overlay primitives, driven without either page.
//
// Strategy: hand the shared helpers keyed memberships and evidence
// exclusions, then render the shared readout in the DOM stub.
// Passing proves both pages use the same neutral membership colors,
// the non-color exclusion class, exact count math, and threshold
// wording without turning the score into an authorship verdict.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

function load() {
  return loadPage({ scripts: ["overlays.js"] });
}

function records(scored, green) {
  const tokens = [{ g: true, we: false }];
  for (let index = 0; index < scored; index++) {
    tokens.push({ g: index < green, we: true });
  }
  return tokens;
}

function textOf(element) {
  if (element.children.length === 0) {
    return element.textContent;
  }
  return element.children.map(textOf).join(" ");
}

test("membership colors do not depend on evidence inclusion", () => {
  const { context } = load();

  assert.equal(
    context.watermarkColor({ g: true, we: true }),
    context.OVERLAYS_WATERMARK_FAVORED
  );
  assert.equal(
    context.watermarkColor({ g: false, we: false }),
    context.OVERLAYS_WATERMARK_COMPLEMENT
  );
  assert.equal(context.watermarkColor({}), null);
});

test("excluded evidence gets a non-color class and wording", () => {
  const { context } = load();

  assert.equal(
    context.overlaysWatermarkTokenClass({ g: true, we: false }),
    "token-watermark-favored token-watermark-excluded"
  );
  assert.equal(
    context.overlaysWatermarkTokenClass({ g: true, we: true }),
    "token-watermark-favored"
  );
  assert.equal(
    context.overlaysWatermarkTokenClass({ g: false, we: true }),
    "token-watermark-complement"
  );
  assert.match(
    context.overlaysWatermarkReading({ g: false, we: false }),
    /keyed complement; excluded from detector score/
  );
});

test("score math excludes the first and forced tokens", () => {
  const { context } = load();
  const tokens = records(4, 3);
  tokens.push({ g: true, we: false });

  const stats = context.overlaysWatermarkStats(tokens, 0.25);

  assert.equal(stats.green_count, 3);
  assert.equal(stats.scored_count, 4);
  assert.equal(stats.green_rate, 0.75);
  assert.equal(stats.status, "insufficient_evidence");
});

test("score math refuses membership without evidence flags", () => {
  const { context } = load();

  assert.equal(
    context.overlaysWatermarkStats([{ g: true }], 0.25),
    null
  );
});

test("score math refuses a partially missing record pair", () => {
  const { context } = load();

  assert.equal(
    context.overlaysWatermarkStats(
      [{ g: true, we: false }, {}], 0.25
    ),
    null
  );
});

test("display status changes at fifty evidence tokens", () => {
  const { context } = load();
  const short = context.overlaysWatermarkStats(
    records(49, 49), 0.25
  );
  const enough = context.overlaysWatermarkStats(
    records(50, 50), 0.25
  );

  assert.equal(
    context.overlaysWatermarkDisplayStatus(short, 4),
    "insufficient_evidence"
  );
  assert.equal(
    context.overlaysWatermarkDisplayStatus(enough, 4),
    "threshold_crossed"
  );
  assert.equal(
    context.overlaysWatermarkDisplayStatus(enough, 100),
    "threshold_not_crossed"
  );
});

test("readout reports counts, p0 and neutral status", () => {
  const page = load();
  const element = page.document.createElement("div");
  page.context.overlaysBuildWatermarkReadout(element);
  const stats = page.context.overlaysWatermarkStats(
    records(50, 30), 0.25
  );

  page.context.overlaysRenderWatermarkReadout(element, {
    stats: stats,
    threshold: 4,
    recordConsistency: "consistent",
  });

  const text = textOf(element);
  assert.match(text, /green\/scored 30\/50/);
  assert.match(text, /green rate 60\.0%/);
  assert.match(text, /p0 0\.25/);
  assert.match(text, /threshold crossed/);
  assert.match(text, /record counts consistent/);
  assert.doesNotMatch(
    text.toLowerCase(), /ai|human|correct|confidence/
  );
});

test("membership description reaches assistive text", () => {
  const page = load();
  const token = { t: " word", m: false, g: false, we: true };
  const span = page.context.overlaysBuildTokenSpan(
    0, token, "?", {
      classFor(index, item) {
        return page.context.overlaysWatermarkTokenClass(item);
      },
      descriptionFor(index, item) {
        return page.context.overlaysWatermarkDescription(item);
      },
    }
  );

  assert.equal(
    span.classList.contains("token-watermark-complement"),
    true
  );
  assert.match(span.getAttribute("aria-label"), /keyed complement/);
  assert.match(span.getAttribute("title"), /included in detector/);
  assert.equal(span.getAttribute("tabindex"), null);
});
