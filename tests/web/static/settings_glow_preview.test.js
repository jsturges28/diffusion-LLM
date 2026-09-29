// The Settings page's glow preview, for every model class it tunes.
//
// Strategy: load the Settings page into the DOM stub beside this file
// and ask it, class by class, for the schedule its preview would
// play. The sliders are judged against that playback, so a class with
// no copy, no pace or the wrong order would tune a glow against a
// preview that shows something else.
//
// Passing proves every class the picker offers has copy and a tick,
// that all of them run for the same time so switching compares the
// glow and not the pacing, and that the state-space preview lights
// its words in order, one per tick, the way its tokens arrive.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SETTINGS_SCRIPTS = [
  "custom_select.js",
  "overlays.js",
  "download_client.js",
  "download_toast.js",
  "settings.js",
];

// The pace every class is held to: 38 words at 90ms, or nine
// diffusion bursts at 380ms.
const PREVIEW_MS = 3420;

function settingsPage() {
  const fetchImpl = () => Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(""),
  });
  return loadPage({ scripts: SETTINGS_SCRIPTS, fetchImpl }).context;
}

function schedule(context, glowClass) {
  context.glowClass = glowClass;
  const words = context.GLOW_PREVIEW_COPY[glowClass].split(" ");
  return JSON.parse(
    JSON.stringify(context.glowPreviewSchedule(words.length))
  );
}

test("every class the picker offers has copy and a pace", () => {
  const context = settingsPage();

  for (const option of context.GLOW_CLASS_OPTIONS) {
    const copy = context.GLOW_PREVIEW_COPY[option.value];
    assert.equal(typeof copy, "string", option.value);
    assert.ok(context.GLOW_PREVIEW_TICK_MS[option.value] > 0);
  }
});

test("every class runs for the same time", () => {
  const context = settingsPage();

  for (const option of context.GLOW_CLASS_OPTIONS) {
    const ticks = schedule(context, option.value).length;
    const tick = context.GLOW_PREVIEW_TICK_MS[option.value];
    assert.equal(ticks * tick, PREVIEW_MS, option.value);
  }
});

test("a state-space preview lights one word per tick, in order", () => {
  const context = settingsPage();

  const groups = schedule(context, "state_space");

  assert.deepEqual(
    groups, groups.map((_, at) => [at])
  );
});

test("the diffusion preview still scatters", () => {
  // The negative space: the table that marks the appending classes
  // must not have swept diffusion into them.
  const context = settingsPage();

  const groups = schedule(context, "diffusion");

  assert.ok(groups.some((group) => group.length > 1));
});
