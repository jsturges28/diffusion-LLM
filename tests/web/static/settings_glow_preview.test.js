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
// its words in order, one per tick, the way its tokens arrive. For
// the revision glow it proves the Diffusion preview's beats each land
// after their word is born and inside the run, that a broken beat is
// refused, that a beat turns a draft into the copy's word in cyan,
// that each glow plays without the other and neither plays when both
// are off, and that the shared rows stay live while either is on.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage, SETTINGS_SCRIPTS } = require("./dom_stub.js");

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

// -- the revision beats --
//
// The Diffusion preview changes its mind twice, on ticks its schedule
// already has. Each beat's word reads as a draft of the same length,
// is born white, and later turns into the copy's word in cyan.

function diffusionPreview(prepare) {
  const context = settingsPage();
  if (prepare) {
    prepare(context);
  }
  context.glowClass = "diffusion";
  context.buildGlowPreviewCopy();
  return context;
}

function textOf(context, word) {
  return context.glowPreviewWords[word].textContent;
}

// What matchMedia answers for a system that prefers reduced motion.
function prefersStill() {
  return { matches: true, addEventListener() {} };
}

test("each beat lands after its word is born, inside the run", () => {
  const context = diffusionPreview();
  const groups = JSON.parse(
    JSON.stringify(context.glowPreviewGroups)
  );

  for (const beat of context.GLOW_PREVIEW_REVISIONS.diffusion) {
    const born = groups.findIndex(
      (group) => group.includes(beat.word)
    );
    assert.ok(born >= 0, `${beat.draft} is never born`);
    assert.ok(born < beat.tick, `${beat.draft}: ${born}`);
    assert.ok(beat.tick < groups.length, beat.draft);
    assert.equal(
      beat.draft.length, context.glowPreviewText[beat.word].length
    );
  }
});

test("a beat that breaks either rule is refused", () => {
  // Word 9 is born on the last tick, so nothing can revise it in
  // time; "resolves" is a letter longer than the word it drafts.
  const late = settingsPage();
  late.glowClass = "diffusion";
  late.GLOW_PREVIEW_REVISIONS.diffusion = [
    { word: 9, draft: "fresh", tick: 5 },
  ];
  assert.throws(() => late.buildGlowPreviewCopy(), /born/);

  const long = settingsPage();
  long.glowClass = "diffusion";
  long.GLOW_PREVIEW_REVISIONS.diffusion = [
    { word: 2, draft: "resolves", tick: 5 },
  ];
  assert.throws(() => long.buildGlowPreviewCopy(), /fit/);

  const past = settingsPage();
  past.glowClass = "diffusion";
  past.GLOW_PREVIEW_REVISIONS.diffusion = [
    { word: 2, draft: "resolve", tick: 9 },
  ];
  assert.throws(() => past.buildGlowPreviewCopy(), /past/);
});

test("the appending classes have no beats", () => {
  const context = settingsPage();

  for (const glowClass of ["autoregressive", "state_space"]) {
    context.glowClass = glowClass;
    assert.equal(context.glowPreviewBeats().length, 0, glowClass);
  }
});

test("a beat turns its draft into the copy's word, in cyan", () => {
  const context = diffusionPreview();
  context.showGlowPreviewDrafts(context.glowPreviewBeats());
  const span = context.glowPreviewWords[2];
  assert.equal(span.textContent, "resolve");
  span.setAttribute("data-born", "");

  context.reviseGlowPreviewWords(context.glowPreviewBeatsAt(5));

  assert.equal(span.textContent, "denoise");
  assert.equal(span.hasAttribute("data-revised"), true);
  assert.equal(span.hasAttribute("data-born"), false);
});

test("a play starts each beat's word on its draft", () => {
  const context = diffusionPreview();

  context.playGlowPreview();
  context.stopGlowPreview();

  assert.equal(textOf(context, 2), "resolve");
  assert.equal(textOf(context, 17), "screen");
});

test("turning the revision glow off puts the words back", () => {
  const context = diffusionPreview();
  context.playGlowPreview();
  context.stopGlowPreview();
  assert.equal(textOf(context, 2), "resolve");

  context.stagedSettings.revisionGlow = false;
  context.playGlowPreview();
  context.stopGlowPreview();

  assert.equal(textOf(context, 2), "denoise");
});

test("with the revision glow off the copy keeps its words", () => {
  const context = diffusionPreview((page) => {
    page.stagedSettings.revisionGlow = false;
  });

  context.playGlowPreview();
  context.stopGlowPreview();

  assert.equal(textOf(context, 2), "denoise");
  assert.equal(context.glowPreviewBeatsAt(5).length, 0);
});

test("the beats still play with the birth glow off", () => {
  const context = diffusionPreview((page) => {
    page.stagedSettings.tokenBirthGlow = false;
  });
  context.playGlowPreview();
  context.stopGlowPreview();

  context.glowPreviewAt = 5;
  context.stepGlowPreview();
  context.stopGlowPreview();

  const born = context.glowPreviewWords.filter(
    (span) => span.hasAttribute("data-born")
  );
  assert.equal(born.length, 0);
  assert.equal(
    context.glowPreviewWords[2].hasAttribute("data-revised"), true
  );
  assert.equal(textOf(context, 2), "denoise");
});

test("with both glows off nothing plays", () => {
  const context = diffusionPreview((page) => {
    page.stagedSettings.tokenBirthGlow = false;
    page.stagedSettings.revisionGlow = false;
  });

  context.playGlowPreview();

  assert.equal(context.glowPreviewTimer, null);
  assert.equal(textOf(context, 2), "denoise");
});

test("reduced motion holds revised words cyan by the sample", () => {
  const context = diffusionPreview((page) => {
    page.matchMedia = prefersStill;
  });

  context.playGlowPreview();

  const words = [...context.glowPreviewWords];
  const revised = words.filter(
    (span) => span.hasAttribute("data-revised")
  );
  assert.deepEqual(
    revised.map((span) => span.textContent), ["denoise", "canvas"]
  );
  const born = words.filter((span) => span.hasAttribute("data-born"));
  assert.equal(born.length, context.GLOW_PREVIEW_STATIC_COUNT);
});

test("reduced motion with no birth glow holds only revisions", () => {
  const context = diffusionPreview((page) => {
    page.matchMedia = prefersStill;
    page.stagedSettings.tokenBirthGlow = false;
  });

  context.playGlowPreview();

  const words = [...context.glowPreviewWords];
  const born = words.filter((span) => span.hasAttribute("data-born"));
  const revised = words.filter(
    (span) => span.hasAttribute("data-revised")
  );
  assert.equal(born.length, 0);
  assert.equal(revised.length, 2);
});

test("either glow keeps the shared rows live", () => {
  // Brightness and fade tune both flashes, so their rows dim only
  // once neither glow is on.
  const context = settingsPage();
  const row = context.glowBrightnessRow;
  const DISABLED = "settings-row-disabled";
  const dimmed = () => row.classList.contains(DISABLED);

  context.stagedSettings.tokenBirthGlow = false;
  context.stagedSettings.revisionGlow = true;
  context.syncGlowControls();
  assert.equal(dimmed(), false);

  context.stagedSettings.revisionGlow = false;
  context.syncGlowControls();
  assert.equal(dimmed(), true);
});

// The page wires its controls once the persisted settings have been
// fetched, which settles on a later turn of the event loop.
function booted() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("the revision toggle is staged and cloned", async () => {
  const context = settingsPage();
  await booted();
  const checkbox = context.settingRevisionGlowCb;
  assert.equal(checkbox.checked, true);

  checkbox.checked = false;
  checkbox.dispatchEvent(new context.Event("change"));

  assert.equal(context.stagedSettings.revisionGlow, false);
  assert.equal(
    context.cloneSettings(context.stagedSettings).revisionGlow, false
  );
});
