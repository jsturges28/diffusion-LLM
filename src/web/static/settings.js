// Settings page: edits the durable user preferences and persists them
// via the shared model in overlays.js (SETTINGS_KEY / SETTINGS_DEFAULTS
// / parseSettings / settingsEqual). It only stages and saves; the
// generator applies the live effects when it next loads (hydrate).

"use strict";

var settingHighlightCb =
  document.getElementById("setting-highlight-tokens");
var settingDiffusionCb =
  document.getElementById("setting-diffusion-text");
var settingGpuTickerCb =
  document.getElementById("setting-gpu-ticker");
var diffusionModeRow =
  document.getElementById("diffusion-mode-row");
var diffusionModeMount =
  document.getElementById("diffusion-mode-mount");
var selectDiffusionMode = null;
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

function cloneSettings(source) {
  return {
    highlightTokens: source.highlightTokens,
    diffusionText: source.diffusionText,
    diffusionTextMode: source.diffusionTextMode,
    gpuTicker: source.gpuTicker,
  };
}

// Mirror the staged settings into the controls.
function syncControls() {
  if (settingHighlightCb) {
    settingHighlightCb.checked = stagedSettings.highlightTokens;
  }
  if (settingDiffusionCb) {
    settingDiffusionCb.checked = stagedSettings.diffusionText;
  }
  if (settingGpuTickerCb) {
    settingGpuTickerCb.checked = stagedSettings.gpuTicker;
  }
  if (selectDiffusionMode) {
    selectDiffusionMode.value = stagedSettings.diffusionTextMode;
  }
  // The Mode sub-setting only applies when the effect is on.
  if (diffusionModeRow) {
    diffusionModeRow.hidden = !stagedSettings.diffusionText;
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

function resetStaged() {
  stagedSettings = cloneSettings(SETTINGS_DEFAULTS);
  syncControls();
  updateButtons();
  setStatus("", false);
}

function wireControls() {
  if (settingHighlightCb) {
    settingHighlightCb.addEventListener("change", function () {
      stagedSettings.highlightTokens = settingHighlightCb.checked;
      updateButtons();
    });
  }
  if (settingDiffusionCb) {
    settingDiffusionCb.addEventListener("change", function () {
      stagedSettings.diffusionText = settingDiffusionCb.checked;
      if (diffusionModeRow) {
        diffusionModeRow.hidden = !settingDiffusionCb.checked;
      }
      updateButtons();
    });
  }
  if (settingGpuTickerCb) {
    settingGpuTickerCb.addEventListener("change", function () {
      stagedSettings.gpuTicker = settingGpuTickerCb.checked;
      updateButtons();
    });
  }
  if (btnSettingsSave) {
    btnSettingsSave.addEventListener("click", saveStaged);
  }
  if (btnSettingsReset) {
    btnSettingsReset.addEventListener("click", resetStaged);
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

// Reveal the Generation nav link only when a model is resident (mirrors
// analytics): the generator is gated on an active model, so the link is
// honest only when there is one to generate with.
function revealGenerationLink() {
  var link = document.getElementById("link-generation");
  if (!link) {
    return;
  }
  fetch("/api/models")
    .then(function (response) {
      return response.json();
    })
    .then(function (info) {
      if (info && info.active) {
        link.hidden = false;
      }
    })
    .catch(function () {
      // Leave it hidden on failure; the menu is always reachable.
    });
}

function bootSettings() {
  appliedSettings = parseSettings(
    localStorage.getItem(SETTINGS_KEY)
  );
  stagedSettings = cloneSettings(appliedSettings);
  buildModeSelect();
  syncControls();
  updateButtons();
  wireControls();
  wireTabs();
  revealGenerationLink();
}

// Hydrate durable UI state from the server first (so the synchronous
// localStorage read above sees the persisted settings), then boot.
// persistHydrate always runs its callback, even if the fetch fails.
persistHydrate(bootSettings);
