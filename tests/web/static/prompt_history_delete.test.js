// Deleting a prompt from the history takes two presses of the trash.
//
// Strategy: seed the generator page with a known history, enter
// browse mode, and drive the trash button the way a user does,
// reading back the box, the counter, the button's state and the
// stored history after each press. Every way of leaving the button
// between the two presses is tried, because the second press must
// only ever count on the prompt the first one was made on.
//
// Passing proves one press deletes nothing, a second deletes the
// prompt on show and moves to the next older one (or the newer one
// at the far end), the deletion is stored, anything that takes the
// user away from the button disarms it, and deleting the last prompt
// ends browsing with the user's own text put back.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

// Most recent first, the order the store keeps.
const NEWEST = "write a sonnet";
const MIDDLE = "explain diffusion";
const OLDEST = "hello world";
const DRAFT = "half a thought";

const OLDER = 1;

function browsing(history) {
  const page = loadPage({});
  const { context, registry } = page;
  registry.get("prompt-input").value = DRAFT;
  context.promptHistory = history.slice();
  context.updatePromptHistoryUI();
  context.enterPromptHistory();
  return {
    page,
    context,
    trash: registry.get("btn-hist-delete"),
    input: registry.get("prompt-input"),
    counter: registry.get("prompt-history-counter"),
  };
}

function stored(page, context) {
  const raw = page.sandbox.localStorage.getItem(
    context.PROMPT_HISTORY_KEY
  );
  return JSON.parse(raw);
}

function history(context) {
  return JSON.parse(JSON.stringify(context.promptHistory));
}

test("one press arms the trash and deletes nothing", () => {
  const { context, trash, input } = browsing([NEWEST, MIDDLE]);

  trash.click();

  assert.deepEqual(history(context), [NEWEST, MIDDLE]);
  assert.equal(input.value, NEWEST);
  assert.equal(trash.classList.contains("is-armed"), true);
  assert.match(trash.getAttribute("aria-label"), /again/);
});

test("a second press deletes the prompt on show", () => {
  const { page, context, trash, input, counter } = browsing(
    [NEWEST, MIDDLE, OLDEST]
  );

  trash.click();
  trash.click();

  assert.deepEqual(history(context), [MIDDLE, OLDEST]);
  assert.equal(input.value, MIDDLE);
  assert.equal(counter.textContent, "2 / 2");
  assert.equal(trash.classList.contains("is-armed"), false);
  assert.deepEqual(stored(page, context), [MIDDLE, OLDEST]);
});

test("deleting from the middle moves to the next older one", () => {
  // The case that tells the two directions apart: the prompt either
  // side of it still exists.
  const { context, trash, input, counter } = browsing(
    [NEWEST, MIDDLE, OLDEST]
  );
  context.cyclePromptHistory(OLDER);

  trash.click();
  trash.click();

  assert.deepEqual(history(context), [NEWEST, OLDEST]);
  assert.equal(input.value, OLDEST);
  assert.equal(counter.textContent, "1 / 2");
});

test("deleting the oldest moves to the newer one", () => {
  // There is no older prompt to move to, so the one after it in time
  // takes its place, and it is now the oldest.
  const { context, trash, input, counter } = browsing(
    [NEWEST, MIDDLE, OLDEST]
  );
  context.cyclePromptHistory(OLDER);
  context.cyclePromptHistory(OLDER);

  trash.click();
  trash.click();

  assert.deepEqual(history(context), [NEWEST, MIDDLE]);
  assert.equal(input.value, MIDDLE);
  assert.equal(counter.textContent, "1 / 2");
});

test("stepping to another prompt disarms", () => {
  // Otherwise the first press, made on one prompt, would carry over
  // and let a single press delete a different one.
  const { context, trash } = browsing([NEWEST, MIDDLE, OLDEST]);
  trash.click();

  context.btnHistPrev.click();
  trash.click();

  assert.deepEqual(history(context), [NEWEST, MIDDLE, OLDEST]);
  assert.equal(trash.classList.contains("is-armed"), true);
});

test("moving off the button disarms", () => {
  const { context, trash } = browsing([NEWEST, MIDDLE]);
  trash.click();

  trash.dispatch("mouseleave");
  trash.click();

  assert.deepEqual(history(context), [NEWEST, MIDDLE]);
});

test("focus leaving the button disarms", () => {
  const { context, trash } = browsing([NEWEST, MIDDLE]);
  trash.click();

  trash.dispatch("blur");

  assert.equal(trash.classList.contains("is-armed"), false);
  assert.match(trash.getAttribute("aria-label"), /^Delete/);
  assert.deepEqual(history(context), [NEWEST, MIDDLE]);
});

test("ending browsing disarms", () => {
  const { context, trash } = browsing([NEWEST, MIDDLE]);
  trash.click();

  context.cancelPromptHistory();
  context.enterPromptHistory();
  trash.click();

  assert.deepEqual(history(context), [NEWEST, MIDDLE]);
});

test("deleting the last prompt puts the user's text back", () => {
  // Nothing is left to browse, so browsing ends the way the cross
  // ends it, and the history control goes away with its contents.
  const { page, context, trash, input } = browsing([NEWEST]);

  trash.click();
  trash.click();

  assert.deepEqual(history(context), []);
  assert.equal(input.value, DRAFT);
  assert.equal(input.readOnly, false);
  assert.equal(context.promptHistoryActive, false);
  assert.equal(page.registry.get("prompt-history").hidden, true);
  assert.deepEqual(stored(page, context), []);
});
