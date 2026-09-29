// Tests for the durable settings model in overlays.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// pattern the other browser tests use, and drive parseSettings and
// settingsEqual directly. Both are pure over a string, so no DOM and
// no storage are needed.
//
// These three functions are the whole contract between the Settings
// page and the two pages that read its work: the page stages a clone,
// compares it with settingsEqual to decide whether Save is live, and
// writes the blob back whole. A key missing from any one of the three
// fails quietly and in a different way each time. Absent from the
// defaults it reads as undefined; absent from parseSettings a stored
// value never comes back; absent from settingsEqual the toggle moves
// and Save stays greyed out.
//
// Written with the mask-candidate reveal, the first setting to be
// read by Analytics as well as the generator, so it is also the first
// one where getting the round trip wrong would show up on two pages.
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

function parse(sandbox, value) {
  return sandbox.parseSettings(JSON.stringify(value));
}

// ---- The mask-candidate reveal ----

test("the reveal is off until it is asked for", () => {
  // A canvas of blocks is what a diffusion run looks like. Reading a
  // page of plausible words that are not the answer yet is a thing
  // to opt into.
  const sandbox = load();

  assert.equal(sandbox.SETTINGS_DEFAULTS.revealMaskCandidate, false);
  assert.equal(
    sandbox.parseSettings(null).revealMaskCandidate,
    false
  );
});

test("the reveal round-trips through storage", () => {
  const sandbox = load();

  const on = parse(sandbox, { revealMaskCandidate: true });
  const off = parse(sandbox, { revealMaskCandidate: false });

  assert.equal(on.revealMaskCandidate, true);
  assert.equal(off.revealMaskCandidate, false);
});

test("a profile saved before the reveal existed keeps blocks", () => {
  // Unlike the hover highlight and the birth glow, which default on
  // when absent: this one changes what the canvas says rather than
  // how it looks, so it is not handed to anyone silently.
  const sandbox = load();

  const older = parse(sandbox, { tokenBirthGlow: true });

  assert.equal(older.revealMaskCandidate, false);
  assert.equal(older.tokenBirthGlow, true);
});

test("a stored reveal is coerced to a boolean", () => {
  // The value reaches the span builder, which asks whether it is
  // truthy. Settling that here keeps a stale string out of a
  // per-token decision.
  const sandbox = load();

  const truthy = parse(sandbox, { revealMaskCandidate: "on" });
  const falsy = parse(sandbox, { revealMaskCandidate: 0 });

  assert.equal(truthy.revealMaskCandidate, true);
  assert.equal(falsy.revealMaskCandidate, false);
});

test("corrupt storage still yields the defaults", () => {
  const sandbox = load();

  const parsed = sandbox.parseSettings("{not json");

  assert.equal(parsed.revealMaskCandidate, false);
});

test("the reveal counts as a change the Save button sees", () => {
  // The Settings page enables Save by comparing the staged clone
  // against the applied one. A key missing here is a toggle that
  // moves and cannot be saved.
  const sandbox = load();
  const before = sandbox.parseSettings(null);
  const after = parse(sandbox, { revealMaskCandidate: true });

  assert.equal(sandbox.settingsEqual(before, before), true);
  assert.equal(sandbox.settingsEqual(before, after), false);
});

test("loading settings tolerates storage being unavailable", () => {
  // Analytics reads the preferences at parse time, so a throwing
  // localStorage would take the page down before it drew anything.
  const sandbox = {
    localStorage: {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {},
    },
    document: { addEventListener: () => {} },
    window: { addEventListener: () => {} },
  };
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "overlays.js",
  });

  const settings = sandbox.overlaysLoadSettings();

  assert.equal(settings.revealMaskCandidate, false);
});

// ---- The state-space glow pair ----
//
// The third class's pair, held to the same three places as every
// other key: the defaults, parseSettings and settingsEqual.

test("state space has its own glow pair, at the defaults", () => {
  const sandbox = load();

  const settings = sandbox.parseSettings(null);

  assert.equal(
    settings.glowBrightnessStateSpace,
    sandbox.GLOW_BRIGHTNESS_DEFAULT
  );
  assert.equal(
    settings.glowFadeMsStateSpace, sandbox.GLOW_FADE_MS_DEFAULT
  );
});

test("a stored state-space pair comes back, clamped", () => {
  const sandbox = load();

  const stored = parse(sandbox, {
    glowBrightnessStateSpace: 150,
    glowFadeMsStateSpace: 5000,
  });

  assert.equal(stored.glowBrightnessStateSpace, 150);
  assert.equal(
    stored.glowFadeMsStateSpace, sandbox.GLOW_FADE_MS_MAX
  );
});

test("a state-space change is one the Save button sees", () => {
  const sandbox = load();
  const before = sandbox.parseSettings(null);
  const after = parse(sandbox, { glowFadeMsStateSpace: 900 });

  assert.equal(sandbox.settingsEqual(before, after), false);
});

test("a state-space model reads its own pair", () => {
  // Not the diffusion pair it would fall back to, and not the
  // autoregressive one it appends like: the class is its own.
  const sandbox = load();
  const settings = parse(sandbox, {
    glowBrightnessStateSpace: 150,
    glowBrightnessAutoregressive: 80,
    glowBrightnessDiffusion: 60,
  });

  const glow = sandbox.overlaysGlowFor(settings, "state_space");

  assert.equal(glow.brightness, 150);
});

test("the Settings picker offers the state-space class", () => {
  const sandbox = load();

  const values = sandbox.GLOW_CLASS_OPTIONS.map(
    (option) => option.value
  );

  assert.deepEqual(
    Object.keys(sandbox.GLOW_KEYS).sort(), [...values].sort()
  );
  assert.ok(values.includes("state_space"));
});
