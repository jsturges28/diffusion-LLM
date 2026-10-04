// The prompt history counter rises to the right.
//
// Strategy: seed the generator page with a known history, enter
// browse mode, and read the counter element after each arrow press,
// checking the text it shows against the prompt actually in the box.
// Direction is the whole point, so every test asserts a number and
// the prompt it belongs to together: a counter that counted correctly
// while showing the wrong prompt would be a worse bug than this one.
//
// The bug being fixed: the store is most-recent-first and the counter
// showed index + 1, so the newest prompt was 1 / N. The left arrow
// steps to older prompts, which meant the number climbed as you moved
// left and fell as you moved right. Both arrows and both tooltips
// were already correct (left is older, matching every back button);
// only the numbering ran the other way.
//
// Passing proves browse opens at N / N on the most recent prompt, the
// right arrow raises the number while moving to newer prompts, the
// left arrow lowers it while moving to older ones, and the wrap at
// either end keeps the number and the prompt in step.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// Most recent first, the order the store keeps. Named so the
// assertions below read as time rather than as indices.
const NEWEST = "write a sonnet";
const MIDDLE = "explain diffusion";
const OLDEST = "hello world";
const HISTORY = [NEWEST, MIDDLE, OLDEST];
const HISTORY_KEY = "diffusion_prompt_history";

// A page in browse mode over HISTORY. Seeded by assignment rather
// than through generation so the order under test is stated here and
// not produced by another interaction's behaviour.
function browsing() {
  const { registry } = loadPage({
    storage: { [HISTORY_KEY]: JSON.stringify(HISTORY) },
    bootState: { ui_state: {}, models: { models: [] } },
  });
  registry.get("btn-prompt-history").click();
  return {
    counter: registry.get("prompt-history-counter"),
    input: registry.get("prompt-input"),
    older: registry.get("btn-hist-prev"),
    newer: registry.get("btn-hist-next"),
  };
}

test("browsing opens on the newest prompt at N / N", () => {
  // The opening state, and the one the maintainer sees first.
  const { counter, input } = browsing();

  assert.equal(counter.textContent, "3 / 3");
  assert.equal(input.value, NEWEST);
});

test("the left arrow lowers the count and goes older", () => {
  const { older, counter, input } = browsing();

  older.click();

  assert.equal(counter.textContent, "2 / 3");
  assert.equal(input.value, MIDDLE);
});

test("the right arrow raises the count and goes newer", () => {
  // The press that used to go the wrong way. Two steps back first, so
  // there is room to climb without wrapping.
  const { older, newer, counter, input } = browsing();
  older.click();
  older.click();

  newer.click();

  assert.equal(counter.textContent, "2 / 3");
  assert.equal(input.value, MIDDLE);
});

test("the oldest prompt is 1 / N", () => {
  // The far end. Numbering from the oldest is what makes the count
  // mean "which prompt in the order you typed them", so the oldest
  // has to be 1 and not N.
  const { older, counter, input } = browsing();

  older.click();
  older.click();

  assert.equal(counter.textContent, "1 / 3");
  assert.equal(input.value, OLDEST);
});

test("wrapping past the oldest returns to N / N", () => {
  // History navigation wraps, so the counter has to survive the jump
  // rather than run off either end.
  const { older, counter, input } = browsing();
  older.click();
  older.click();

  older.click();

  assert.equal(counter.textContent, "3 / 3");
  assert.equal(input.value, NEWEST);
});

test("wrapping past the newest returns to 1 / N", () => {
  const { newer, counter, input } = browsing();

  newer.click();

  assert.equal(counter.textContent, "1 / 3");
  assert.equal(input.value, OLDEST);
});

test("a single-prompt history reads 1 / 1", () => {
  // The degenerate case. With one prompt the newest and the oldest
  // are the same entry, so both readings have to agree on it.
  const { registry } = loadPage({
    storage: {
      [HISTORY_KEY]: JSON.stringify([NEWEST]),
    },
    bootState: { ui_state: {}, models: { models: [] } },
  });
  registry.get("btn-prompt-history").click();

  assert.equal(
    registry.get("prompt-history-counter").textContent, "1 / 1"
  );
});

test("clicking the right arrow raises the number", () => {
  // Guarding against the fix that was not made. Swapping the two
  // deltas would also have made the number climb rightward, at the
  // cost of the right arrow meaning "older" and both tooltips
  // becoming lies. Driven through the buttons rather than through
  // cyclePromptHistory so the wiring is part of what passes.
  const { older, newer, counter, input } = browsing();
  older.click();

  newer.click();

  assert.equal(counter.textContent, "3 / 3");
  assert.equal(input.value, NEWEST);
});

test("clicking the left arrow lowers the number", () => {
  const { older, counter, input } = browsing();

  older.click();

  assert.equal(counter.textContent, "2 / 3");
  assert.equal(input.value, MIDDLE);
});
