// Settings page: edits the durable user preferences and persists them
// via the shared model in overlays.js (SETTINGS_KEY / SETTINGS_DEFAULTS
// / parseSettings / settingsEqual). It only stages and saves; the
// generator applies the live effects when it next loads (hydrate).

"use strict";

var settingDiffusionCb =
  document.getElementById("setting-diffusion-text");
var settingGpuTickerCb =
  document.getElementById("setting-gpu-ticker");
var settingBirthGlowCb =
  document.getElementById("setting-token-birth-glow");
var settingRevealMaskCb =
  document.getElementById("setting-reveal-mask-candidate");
var diffusionModeRow =
  document.getElementById("diffusion-mode-row");
var diffusionModeMount =
  document.getElementById("diffusion-mode-mount");
var selectDiffusionMode = null;
var glowClassRow =
  document.getElementById("glow-class-row");
var glowClassMount =
  document.getElementById("glow-class-mount");
var selectGlowClass = null;
var glowBrightnessRow =
  document.getElementById("glow-brightness-row");
var glowBrightnessInput =
  document.getElementById("setting-glow-brightness");
var glowBrightnessValue =
  document.getElementById("glow-brightness-value");
var glowFadeRow =
  document.getElementById("glow-fade-row");
var glowFadeInput =
  document.getElementById("setting-glow-fade");
var glowFadeValue =
  document.getElementById("glow-fade-value");
var glowPreviewRow =
  document.getElementById("glow-preview-row");
var glowPreview = document.getElementById("glow-preview");
var glowPreviewCopy =
  document.getElementById("glow-preview-copy");
// Which model class the two sliders are editing. A view state, not a
// preference: it says what you are looking at, and every class keeps
// its own stored pair regardless.
var glowClass = "diffusion";
var btnSettingsSave =
  document.getElementById("btn-settings-save");
var btnSettingsReset =
  document.getElementById("btn-settings-reset");
var settingsStatus =
  document.getElementById("settings-status");
var settingsStatusTimer = null;
var settingsTabs =
  document.querySelectorAll(".settings-tab");
var settingsPanels =
  document.querySelectorAll(".settings-panel");

// appliedSettings is the last-saved state; stagedSettings is the
// working copy, committed to appliedSettings only via Save.
var appliedSettings = parseSettings(null);
var stagedSettings = parseSettings(null);

// highlightTokens is carried but never shown: its control moved to
// each page's overlay drawer. Save writes this whole blob, so keeping
// the field here is what stops a save from clobbering the checkbox.
function cloneSettings(source) {
  return {
    highlightTokens: source.highlightTokens,
    diffusionText: source.diffusionText,
    diffusionTextMode: source.diffusionTextMode,
    gpuTicker: source.gpuTicker,
    tokenBirthGlow: source.tokenBirthGlow,
    revealMaskCandidate: source.revealMaskCandidate,
    glowBrightnessDiffusion: source.glowBrightnessDiffusion,
    glowFadeMsDiffusion: source.glowFadeMsDiffusion,
    glowBrightnessAutoregressive:
      source.glowBrightnessAutoregressive,
    glowFadeMsAutoregressive: source.glowFadeMsAutoregressive,
    tpsMode: source.tpsMode,
  };
}

// Mirror the staged settings into the controls.
function syncControls() {
  if (settingDiffusionCb) {
    settingDiffusionCb.checked = stagedSettings.diffusionText;
  }
  if (settingGpuTickerCb) {
    settingGpuTickerCb.checked = stagedSettings.gpuTicker;
  }
  if (settingBirthGlowCb) {
    settingBirthGlowCb.checked =
      stagedSettings.tokenBirthGlow;
  }
  if (settingRevealMaskCb) {
    settingRevealMaskCb.checked =
      stagedSettings.revealMaskCandidate;
  }
  if (selectDiffusionMode) {
    selectDiffusionMode.value = stagedSettings.diffusionTextMode;
  }
  setSubRowEnabled(
    diffusionModeRow,
    selectDiffusionMode,
    stagedSettings.diffusionText
  );
  syncGlowControls();
}

// Mirror the staged pair for the selected model class into the two
// sliders, their readouts, and the preview.
function syncGlowControls() {
  var glow = overlaysGlowFor(stagedSettings, glowClass);
  if (selectGlowClass) {
    selectGlowClass.value = glowClass;
  }
  if (glowBrightnessInput) {
    glowBrightnessInput.value = String(glow.brightness);
  }
  if (glowFadeInput) {
    glowFadeInput.value = String(glow.fadeMs);
  }
  if (glowBrightnessValue) {
    glowBrightnessValue.textContent = glow.brightness + "%";
  }
  if (glowFadeValue) {
    glowFadeValue.textContent = glow.fadeMs + " ms";
  }
  overlaysApplyGlowVars(
    glowPreview, glow.brightness, glow.fadeMs
  );
  var on = stagedSettings.tokenBirthGlow;
  setSubRowEnabled(glowClassRow, selectGlowClass, on);
  setSubRowEnabled(glowBrightnessRow, glowBrightnessInput, on);
  setSubRowEnabled(glowFadeRow, glowFadeInput, on);
  setSubRowEnabled(glowPreviewRow, glowPreview, on);
  // A sequence in flight has to be stopped, not just left to finish,
  // or it would keep lighting words inside a row that is now dimmed.
  if (!on) {
    stopGlowPreview();
    clearGlowPreviewWords();
  }
}

// ---- The glow preview ----

// The copy is the preview. Each word is a token, and lighting them in
// the order the selected class would produce them is what the two
// sliders are actually being judged against: brightness is legible on
// one word, but fade is only legible as a trail, and a trail needs
// enough lit words at once to have a head and a tail.
var GLOW_PREVIEW_COPY = {
  diffusion:
    "Diffusion models denoise many positions at once, in no "
    + "fixed order, so the glow scatters across the canvas "
    + "instead of trailing behind one point. Drag either "
    + "slider above, or click here, to replay this at the "
    + "current settings.",
  autoregressive:
    "Autoregressive models emit one token at a time, left to "
    + "right, so the glow reads as a trail chasing the word at "
    + "the front. Drag either slider above, or click here, to "
    + "replay this at the current settings.",
};

// Both classes run for exactly 3420ms over the 38 words of copy, so
// switching between them compares the glow and not the pacing. They
// spend it differently: one word per tick for autoregressive, a
// scattered burst per slower tick for diffusion, which is the shape
// of a denoising step. Fixed rather than matched to real hardware,
// since a real run's speed is the thing being compensated for.
//
// The pace is set by the worst case rather than by realism. Lit words
// at any moment is roughly fade over tick, so the sequence has to
// outlast the longest fade or the block saturates and there is no
// tail left to judge. At these values the 2000ms maximum peaks at 23
// of 38 words for autoregressive and 28 for diffusion, and the 200ms
// minimum at 3 and 6.
//
// 380ms is not chosen directly: the burst sizes are drawn, they come
// to nine ticks, and 3420 / 9 is what makes the two totals equal.
// Reseeding or changing the burst range changes the tick count, so
// this has to be recomputed alongside it.
var GLOW_PREVIEW_TICK_MS = {
  diffusion: 380,
  autoregressive: 90,
};

// Real denoising steps do not resolve the same number of positions
// every time, so neither does this. Uniform over the range, which
// averages the 4 the burst was fixed at before.
var GLOW_PREVIEW_BURST_MIN = 2;
var GLOW_PREVIEW_BURST_MAX = 6;

// Chosen by measurement, not taste, so do not treat it as arbitrary.
// Across 800 candidates this is one of eleven that put no two
// adjacent words in the same burst while landing on nine ticks, and
// it has the widest spread of those: 0.83 of the words in a burst
// fall on different rendered lines. Changing the copy changes the
// word count and invalidates all of that.
var GLOW_PREVIEW_SEED = 662;

// Fade cannot be shown without movement, so reduced motion gets
// brightness only: a fixed handful held at peak. Lit words against
// unlit neighbours is what makes the level legible, where running
// the sequence would both animate and end with the block washed out.
var GLOW_PREVIEW_STATIC_COUNT = 8;

// A slider fires input continuously through a drag. Replaying on
// every event would restart the sequence before it cleared its first
// word, so the readouts track the drag live and the replay waits for
// it to settle.
var GLOW_PREVIEW_SETTLE_MS = 180;

var glowPreviewWords = [];
var glowPreviewGroups = [];
var glowPreviewAt = 0;
var glowPreviewTimer = null;
var glowPreviewRestartTimer = null;

// Rebuild the copy for the selected class. Also fixes the schedule,
// which depends only on the word count and the class, so the order is
// settled once here rather than re-derived on every replay.
function buildGlowPreviewCopy() {
  if (!glowPreviewCopy) {
    return;
  }
  stopGlowPreview();
  var text =
    GLOW_PREVIEW_COPY[glowClass] || GLOW_PREVIEW_COPY.diffusion;
  var words = text.split(" ");
  glowPreviewCopy.textContent = "";
  glowPreviewWords = new Array(words.length);
  for (var i = 0; i < words.length; i++) {
    var span = document.createElement("span");
    span.className = "glow-preview-word";
    span.textContent = words[i];
    glowPreviewWords[i] = span;
    glowPreviewCopy.appendChild(span);
    if (i + 1 < words.length) {
      glowPreviewCopy.appendChild(
        document.createTextNode(" ")
      );
    }
  }
  glowPreviewGroups = glowPreviewSchedule(words.length);
}

// Which words light on each tick, as an array of ticks. One seeded
// stream drives both the order and the burst sizes, so the whole
// schedule follows from GLOW_PREVIEW_SEED and is identical on every
// replay, every reload, and every machine. That repeatability is the
// requirement; randomness is only how the shape is reached.
function glowPreviewSchedule(count) {
  var groups = [];
  var i = 0;
  if (glowClass === "autoregressive") {
    for (i = 0; i < count; i++) {
      groups.push([i]);
    }
    return groups;
  }
  var draw = glowPreviewRandom(GLOW_PREVIEW_SEED);
  var order = glowPreviewShuffled(count, draw);
  var at = 0;
  // Bounded by count because every pass takes at least one word.
  for (i = 0; i < count && at < count; i++) {
    var size = glowPreviewBurstSize(count - at, draw);
    groups.push(order.slice(at, at + size));
    at += size;
  }
  return groups;
}

function glowPreviewBurstSize(remaining, draw) {
  var span = GLOW_PREVIEW_BURST_MAX - GLOW_PREVIEW_BURST_MIN + 1;
  var size = GLOW_PREVIEW_BURST_MIN + Math.floor(draw() * span);
  // Absorb a would-be final burst of one rather than ending the
  // sequence on a lone flicker.
  if (remaining - size === 1) {
    size += 1;
  }
  if (size > remaining) {
    return remaining;
  }
  return size;
}

// Fisher-Yates, which is what makes this a permutation: every word
// lights exactly once. The previous version walked the block by a
// fixed stride, which covers evenly but is an arithmetic progression,
// so it read as a rigid motif marching across the rows rather than as
// anything scattered. Even coverage and looking unordered are not the
// same property, and the stride optimised for the wrong one.
function glowPreviewShuffled(count, draw) {
  var order = new Array(count);
  var i = 0;
  for (i = 0; i < count; i++) {
    order[i] = i;
  }
  for (i = count - 1; i > 0; i--) {
    var pick = Math.floor(draw() * (i + 1));
    var held = order[i];
    order[i] = order[pick];
    order[pick] = held;
  }
  return order;
}

// mulberry32: a small deterministic generator, used here only so the
// schedule is reproducible without shipping a hardcoded permutation
// that would have to be regenerated by hand whenever the copy
// changes. Not suitable for anything that needs real randomness.
function glowPreviewRandom(seed) {
  var state = seed | 0;
  return function () {
    state = (state + 0x6d2b79f5) | 0;
    var t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function playGlowPreview() {
  stopGlowPreview();
  if (!glowPreviewCopy || !stagedSettings.tokenBirthGlow) {
    return;
  }
  if (glowPreviewGroups.length === 0) {
    return;
  }
  clearGlowPreviewWords();
  if (prefersReducedMotion()) {
    lightGlowPreviewSample();
    return;
  }
  // One forced reflow for the whole block, so the clear above lands
  // before the first group is re-lit. Without it the browser
  // coalesces the two writes and a repeat play does nothing. This is
  // the same trick the live canvas refuses to use, because there it
  // would be one reflow per token per frame rather than one per
  // click on a page with no generation running.
  void glowPreviewCopy.offsetWidth;
  glowPreviewAt = 0;
  stepGlowPreview();
}

function stepGlowPreview() {
  var group = glowPreviewGroups[glowPreviewAt];
  for (var i = 0; i < group.length; i++) {
    var span = glowPreviewWords[group[i]];
    if (span) {
      span.setAttribute("data-born", "");
    }
  }
  glowPreviewAt += 1;
  if (glowPreviewAt >= glowPreviewGroups.length) {
    glowPreviewTimer = null;
    return;
  }
  glowPreviewTimer = setTimeout(
    stepGlowPreview,
    GLOW_PREVIEW_TICK_MS[glowClass]
      || GLOW_PREVIEW_TICK_MS.diffusion
  );
}

// Debounced replay, for the sliders. Discrete actions (a click, a
// class change) call playGlowPreview directly instead.
function replayGlowPreviewSoon() {
  if (glowPreviewRestartTimer !== null) {
    clearTimeout(glowPreviewRestartTimer);
  }
  glowPreviewRestartTimer = setTimeout(function () {
    glowPreviewRestartTimer = null;
    playGlowPreview();
  }, GLOW_PREVIEW_SETTLE_MS);
}

function stopGlowPreview() {
  if (glowPreviewTimer !== null) {
    clearTimeout(glowPreviewTimer);
    glowPreviewTimer = null;
  }
  if (glowPreviewRestartTimer !== null) {
    clearTimeout(glowPreviewRestartTimer);
    glowPreviewRestartTimer = null;
  }
}

function clearGlowPreviewWords() {
  for (var i = 0; i < glowPreviewWords.length; i++) {
    glowPreviewWords[i].removeAttribute("data-born");
  }
}

// Walks the schedule rather than the copy, so the sample it lights is
// contiguous for autoregressive and scattered for diffusion without
// knowing which is which.
function lightGlowPreviewSample() {
  var lit = 0;
  for (var i = 0; i < glowPreviewGroups.length; i++) {
    var group = glowPreviewGroups[i];
    for (var j = 0; j < group.length; j++) {
      if (lit >= GLOW_PREVIEW_STATIC_COUNT) {
        return;
      }
      glowPreviewWords[group[j]].setAttribute("data-born", "");
      lit += 1;
    }
  }
}

// Dim a sub-setting row and make its control inert when the
// preference it depends on is off. Dimmed rather than hidden: the
// indent only reads as "belongs to the row above" if the row stays.
function setSubRowEnabled(row, control, enabled) {
  if (row) {
    row.classList.toggle("settings-row-disabled", !enabled);
  }
  if (control) {
    control.disabled = !enabled;
  }
}

// Save is enabled only with pending changes; Reset only when the staged
// settings differ from the fresh defaults.
function updateButtons() {
  if (btnSettingsSave) {
    btnSettingsSave.disabled = settingsEqual(
      stagedSettings, appliedSettings
    );
  }
  if (btnSettingsReset) {
    btnSettingsReset.disabled = settingsEqual(
      stagedSettings, SETTINGS_DEFAULTS
    );
  }
}

// Small save-feedback line in the footer. Pass "" to hide.
function setStatus(text, saved) {
  if (!settingsStatus) {
    return;
  }
  if (settingsStatusTimer !== null) {
    clearTimeout(settingsStatusTimer);
    settingsStatusTimer = null;
  }
  if (!text) {
    settingsStatus.hidden = true;
    settingsStatus.textContent = "";
    settingsStatus.classList.remove("is-saved");
    return;
  }
  settingsStatus.textContent = text;
  settingsStatus.classList.toggle("is-saved", !!saved);
  settingsStatus.hidden = false;
}

function saveStaged() {
  if (settingsEqual(stagedSettings, appliedSettings)) {
    return;
  }
  setStatus("Saving\u2026", false);
  appliedSettings = cloneSettings(stagedSettings);
  // persistSet mirrors to the server so settings survive across the
  // browser and desktop app; the generator hydrates them on next load.
  persistSet(SETTINGS_KEY, JSON.stringify(appliedSettings));
  updateButtons();
  if (btnSettingsSave) {
    btnSettingsSave.blur();
  }
  settingsStatusTimer = setTimeout(function () {
    setStatus("Changes saved!", true);
    settingsStatusTimer = setTimeout(function () {
      setStatus("", false);
    }, 2400);
  }, 300);
}

// Reset only what this page shows. highlightTokens and tpsMode are
// carried in the same blob but controlled from the overlay drawers
// and the generator's footer, so resetting them from here would
// silently flip switches the user cannot see.
function resetStaged() {
  var highlight = stagedSettings.highlightTokens;
  var tpsMode = stagedSettings.tpsMode;
  stagedSettings = cloneSettings(SETTINGS_DEFAULTS);
  stagedSettings.highlightTokens = highlight;
  stagedSettings.tpsMode = tpsMode;
  syncControls();
  // Reset moves both glow sliders, so show what they moved to.
  playGlowPreview();
  updateButtons();
  setStatus("", false);
}

function wireControls() {
  if (settingDiffusionCb) {
    settingDiffusionCb.addEventListener("change", function () {
      stagedSettings.diffusionText = settingDiffusionCb.checked;
      setSubRowEnabled(
        diffusionModeRow,
        selectDiffusionMode,
        settingDiffusionCb.checked
      );
      updateButtons();
    });
  }
  if (settingGpuTickerCb) {
    settingGpuTickerCb.addEventListener("change", function () {
      stagedSettings.gpuTicker = settingGpuTickerCb.checked;
      updateButtons();
    });
  }
  if (settingBirthGlowCb) {
    settingBirthGlowCb.addEventListener("change", function () {
      stagedSettings.tokenBirthGlow =
        settingBirthGlowCb.checked;
      syncGlowControls();
      // Turning it back on should show what was turned back on.
      // Off is a no-op: playGlowPreview returns early.
      playGlowPreview();
      updateButtons();
    });
  }
  if (settingRevealMaskCb) {
    settingRevealMaskCb.addEventListener("change", function () {
      stagedSettings.revealMaskCandidate =
        settingRevealMaskCb.checked;
      updateButtons();
    });
  }
  wireGlowControls();
  if (btnSettingsSave) {
    btnSettingsSave.addEventListener("click", saveStaged);
  }
  if (btnSettingsReset) {
    btnSettingsReset.addEventListener("click", resetStaged);
  }
}

// Both sliders write to whichever class is selected, so the handler
// is the same shape twice: read the input, store it under that
// class's key, re-sync, and queue a replay once the drag settles.
function wireGlowControls() {
  if (glowBrightnessInput) {
    glowBrightnessInput.min = String(GLOW_BRIGHTNESS_MIN);
    glowBrightnessInput.max = String(GLOW_BRIGHTNESS_MAX);
    glowBrightnessInput.step = "5";
    glowBrightnessInput.addEventListener("input", function () {
      stageGlowValue(
        "brightness", Number(glowBrightnessInput.value)
      );
    });
  }
  if (glowFadeInput) {
    glowFadeInput.min = String(GLOW_FADE_MS_MIN);
    glowFadeInput.max = String(GLOW_FADE_MS_MAX);
    glowFadeInput.step = String(GLOW_FADE_MS_STEP);
    glowFadeInput.addEventListener("input", function () {
      stageGlowValue("fadeMs", Number(glowFadeInput.value));
    });
  }
  if (glowPreview) {
    glowPreview.addEventListener("click", playGlowPreview);
  }
}

function stageGlowValue(field, value) {
  var keys = GLOW_KEYS[glowClass] || GLOW_KEYS.diffusion;
  stagedSettings[keys[field]] = value;
  syncGlowControls();
  replayGlowPreviewSoon();
  updateButtons();
}

function buildGlowClassSelect() {
  if (!glowClassMount) {
    return;
  }
  selectGlowClass = createCustomSelect(
    GLOW_CLASS_OPTIONS, glowClass
  );
  glowClassMount.appendChild(selectGlowClass);
  sizeCustomSelect(selectGlowClass);
  selectGlowClass.addEventListener("change", function () {
    glowClass = selectGlowClass.value;
    // The copy names the class and its order depends on it, so the
    // whole preview is rebuilt rather than merely replayed.
    buildGlowPreviewCopy();
    syncGlowControls();
    playGlowPreview();
  });
}

// Open on the class of the model that is currently resident, so the
// sliders start on the one the user is about to see the effect from.
// Best-effort: the picker defaults to diffusion and stays there when
// nothing is loaded. Called before the preview is built or played, so
// unlike the fetch it replaced it only has to set the value; it no
// longer has to undo a sequence that already started on the wrong one.
function adoptGlowClassForActiveModel(type) {
  if (type && GLOW_KEYS[type]) {
    glowClass = type;
  }
}

function buildModeSelect() {
  if (!diffusionModeMount) {
    return;
  }
  selectDiffusionMode = createCustomSelect(
    [
      { value: "default", label: "Default" },
      { value: "cycle", label: "Cycle" },
    ],
    stagedSettings.diffusionTextMode
  );
  diffusionModeMount.appendChild(selectDiffusionMode);
  sizeCustomSelect(selectDiffusionMode);
  selectDiffusionMode.addEventListener("change", function () {
    stagedSettings.diffusionTextMode = selectDiffusionMode.value;
    updateButtons();
  });
}

function selectTab(name) {
  for (var i = 0; i < settingsTabs.length; i++) {
    settingsTabs[i].classList.toggle(
      "is-active",
      settingsTabs[i].getAttribute("data-tab") === name
    );
  }
  for (var j = 0; j < settingsPanels.length; j++) {
    settingsPanels[j].hidden =
      settingsPanels[j].getAttribute("data-panel") !== name;
  }
}

function wireTabs() {
  for (var i = 0; i < settingsTabs.length; i++) {
    (function (tab) {
      tab.addEventListener("click", function () {
        selectTab(tab.getAttribute("data-tab"));
      });
    })(settingsTabs[i]);
  }
}

// Which model is resident, from the boot state the server inlined. It
// used to be a /api/models fetch feeding two consumers: the Generation
// nav link, now unhidden in the markup by the server, and the glow
// class below. Reading it here rather than fetching is what lets the
// preview open on the right class instead of correcting itself.
function bootActiveModelType() {
  var boot = window.__BOOT__;
  if (!boot || typeof boot.active_model_type !== "string") {
    return null;
  }
  return boot.active_model_type;
}

function bootSettings() {
  appliedSettings = parseSettings(
    localStorage.getItem(SETTINGS_KEY)
  );
  stagedSettings = cloneSettings(appliedSettings);
  // Ahead of everything that reads glowClass: the picker, the preview
  // copy and the preview itself all open on it.
  adoptGlowClassForActiveModel(bootActiveModelType());
  buildModeSelect();
  buildGlowClassSelect();
  buildGlowPreviewCopy();
  syncControls();
  playGlowPreview();
  updateButtons();
  wireControls();
  wireTabs();
}

// Hydrate durable UI state from the server first (so the synchronous
// localStorage read above sees the persisted settings), then boot.
// persistHydrate always runs its callback, even if the fetch fails.
persistHydrate(bootSettings);
