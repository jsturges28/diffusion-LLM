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
// That toggle is now a three-way choice of what an unsettled position
// shows, and a profile saved while it was a toggle migrates.
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

// ---- What an unsettled position shows ----

test("unsettled positions show the glyph until asked otherwise", () => {
  // A canvas of blocks is what a diffusion run looks like. Reading a
  // page of plausible words that are not the answer yet is a thing
  // to opt into.
  const sandbox = load();

  assert.equal(sandbox.SETTINGS_DEFAULTS.unsettledShows, "glyph");
  assert.equal(sandbox.parseSettings(null).unsettledShows, "glyph");
});

test("each choice round-trips through storage", () => {
  const sandbox = load();

  for (const choice of ["glyph", "guess", "candidates"]) {
    const parsed = parse(sandbox, { unsettledShows: choice });
    assert.equal(parsed.unsettledShows, choice);
  }
});

test("a profile that had the reveal on lands on the guess", () => {
  // The toggle this replaced drew the guess when on. Its old
  // coercion read any truthy value as on, so the migration does too.
  const sandbox = load();

  const on = parse(sandbox, { revealMaskCandidate: true });
  const stringly = parse(sandbox, { revealMaskCandidate: "on" });
  const off = parse(sandbox, { revealMaskCandidate: false });

  assert.equal(on.unsettledShows, "guess");
  assert.equal(stringly.unsettledShows, "guess");
  assert.equal(off.unsettledShows, "glyph");
});

test("a stored choice wins over the old toggle", () => {
  // Once the Settings page saves, the choice is what the profile
  // says; a stale toggle beside it must not override it.
  const sandbox = load();

  const cycling = parse(sandbox, {
    unsettledShows: "candidates", revealMaskCandidate: false,
  });
  const glyph = parse(sandbox, {
    unsettledShows: "glyph", revealMaskCandidate: true,
  });

  assert.equal(cycling.unsettledShows, "candidates");
  assert.equal(glyph.unsettledShows, "glyph");
});

test("a choice this build does not know falls back", () => {
  const sandbox = load();

  const unknown = parse(sandbox, { unsettledShows: "stack" });
  const migrated = parse(sandbox, {
    unsettledShows: "stack", revealMaskCandidate: true,
  });

  assert.equal(unknown.unsettledShows, "glyph");
  assert.equal(migrated.unsettledShows, "guess");
});

test("a profile saved before either keeps the glyph", () => {
  // Unlike the hover highlight and the birth glow, which default on
  // when absent: this one changes what the canvas says rather than
  // how it looks, so it is not handed to anyone silently.
  const sandbox = load();

  const older = parse(sandbox, { tokenBirthGlow: true });

  assert.equal(older.unsettledShows, "glyph");
  assert.equal(older.tokenBirthGlow, true);
});

test("the old toggle is not carried forward", () => {
  // Save writes what parseSettings returns, so the next save drops
  // the toggle and the choice is all a profile holds.
  const sandbox = load();

  const parsed = parse(sandbox, { revealMaskCandidate: true });

  assert.equal("revealMaskCandidate" in parsed, false);
});

test("corrupt storage still yields the defaults", () => {
  const sandbox = load();

  const parsed = sandbox.parseSettings("{not json");

  assert.equal(parsed.unsettledShows, "glyph");
});

test("the choice counts as a change the Save button sees", () => {
  // The Settings page enables Save by comparing the staged clone
  // against the applied one. A key missing here is a control that
  // moves and cannot be saved.
  const sandbox = load();
  const before = sandbox.parseSettings(null);
  const after = parse(sandbox, { unsettledShows: "candidates" });

  assert.equal(sandbox.settingsEqual(before, before), true);
  assert.equal(sandbox.settingsEqual(before, after), false);
});

test("the guess is drawn for every choice but the glyph", () => {
  // Cycling starts from the guess and falls back to it wherever
  // there are no candidates to cycle through.
  const sandbox = load();

  const draws = (choice) => sandbox.overlaysDrawsGuess(
    parse(sandbox, { unsettledShows: choice })
  );

  assert.equal(draws("glyph"), false);
  assert.equal(draws("guess"), true);
  assert.equal(draws("candidates"), true);
});

test("the Settings dropdown offers exactly the three choices", () => {
  const sandbox = load();

  const values = sandbox.UNSETTLED_SHOWS_OPTIONS.map(
    (option) => option.value
  );

  assert.deepEqual([...values], ["glyph", "guess", "candidates"]);
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

  assert.equal(settings.unsettledShows, "glyph");
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

// ---- The revision glow ----
//
// Its toggle held to the same three places, and its flash to the
// birth glow's shape in another colour.

test("the revision glow is on by default", () => {
  const sandbox = load();

  assert.equal(sandbox.parseSettings(null).revisionGlow, true);
});

test("a profile saved before it existed meets it on", () => {
  // Like the birth glow: absent is not a choice to have it off.
  const sandbox = load();

  const older = parse(sandbox, { tokenBirthGlow: true });
  const chose = parse(sandbox, { revisionGlow: false });

  assert.equal(older.revisionGlow, true);
  assert.equal(chose.revisionGlow, false);
});

test("the revision toggle is a change the Save button sees", () => {
  const sandbox = load();
  const before = sandbox.parseSettings(null);
  const after = parse(sandbox, { revisionGlow: false });

  assert.equal(sandbox.settingsEqual(before, after), false);
});

// An element that records the custom properties written onto it.
function styled() {
  const props = {};
  return {
    props,
    style: {
      setProperty: (name, value) => {
        props[name] = value;
      },
    },
  };
}

function radii(shadow) {
  return shadow.match(/[\d.]+px/g);
}

test("the cyan flash has the white one's radii and fade", () => {
  // One brightness drives both, so at any setting the two differ in
  // colour alone, and the fade they share is one property.
  const sandbox = load();
  const el = styled();

  sandbox.overlaysApplyGlowVars(el, 150, 800);

  const white = el.props["--token-birth-shadow"];
  const cyan = el.props["--token-revision-shadow"];
  assert.deepEqual(radii(cyan), radii(white));
  assert.match(cyan, /rgba\(0, 220, 255, /);
  assert.match(white, /rgba\(255, 255, 255, /);
  assert.deepEqual(
    radii(el.props["--token-revision-shadow-off"]), radii(white)
  );
  assert.doesNotMatch(
    el.props["--token-revision-shadow-off"], /, 0\.\d+\)/
  );
  assert.equal(el.props["--token-birth-duration"], "800ms");
});

test("a shadow without its colour is refused", () => {
  const sandbox = load();

  assert.throws(() => sandbox.overlaysGlowShadow(100), /colour/);
});
