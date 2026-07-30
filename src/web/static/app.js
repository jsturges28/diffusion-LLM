// LLM Visualizer: client-side logic.

"use strict";

// Unresolved-token glyph; set from the active model.
var MASK_CHAR = "\u2591"; // ░
var RECONNECT_DELAY_MS = 2000;
var MAX_RECONNECT_DELAY_MS = 16000;

// ---- Model registry state (from /api/models) ----

var models = {}; // id -> ModelInfo
var activeModelId = null;
var activeModel = null; // ModelInfo of the active model
var activeDevice = null; // "cuda" | "cpu": the active model's device
var gpuPresent = false; // whether a usable GPU was detected
var suppressReconnect = false;

// Dynamic parameter DOM, rebuilt per model from its schema.
var paramInputs = {}; // name -> input/select element
var paramTooltips = {}; // name -> tooltip span

// ---- DOM refs ----

var promptInput =
  document.getElementById("prompt-input");
var promptHistoryGroup =
  document.getElementById("prompt-history");
var btnPromptHistory =
  document.getElementById("btn-prompt-history");
var promptHistoryNav =
  document.getElementById("prompt-history-nav");
var btnHistPrev =
  document.getElementById("btn-hist-prev");
var btnHistNext =
  document.getElementById("btn-hist-next");
var promptHistoryCounter =
  document.getElementById("prompt-history-counter");
var btnHistConfirm =
  document.getElementById("btn-hist-confirm");
var btnHistCancel =
  document.getElementById("btn-hist-cancel");
var btnGenerate =
  document.getElementById("btn-generate");
var btnGenerateLabel =
  document.getElementById("btn-generate-label");
var btnSave =
  document.getElementById("btn-save");
var outputArea =
  document.getElementById("output-area");
var connectionBadge =
  document.getElementById("connection-badge");
var statusStep =
  document.getElementById("status-step");
var statusElapsed =
  document.getElementById("status-elapsed");
var statusMessage =
  document.getElementById("status-message");
var loadingOverlay =
  document.getElementById("loading-overlay");
var validationHint =
  document.getElementById("validation-hint");
var toggleExperimental =
  document.getElementById("toggle-experimental");

var modelSelect =
  document.getElementById("model-select");
var modelSelectValue =
  document.getElementById("model-select-value");
var modelSelectList =
  document.getElementById("model-select-list");
var modelSelectDisabled = false;
var paramFields =
  document.getElementById("param-fields");
var modeExtra =
  document.getElementById("mode-extra");
var loadingText =
  document.getElementById("loading-text");
var thinkingPanel =
  document.getElementById("thinking-panel");
var thinkingContent =
  document.getElementById("thinking-content");

// Settings are edited on the shared /settings.html page; the generator
// only applies the persisted preferences (loadSettings/applySettings).
var statusHighlight =
  document.getElementById("status-highlight");
// Header "new run saved" cue on the Analytics link.
var linkAnalytics =
  document.getElementById("link-analytics");
var analyticsNewDot =
  document.getElementById("analytics-new-dot");
// Persistent UI preferences, applied live on the generator. The schema,
// defaults, and parsing live in overlays.js (SETTINGS_DEFAULTS /
// parseSettings), shared with the Settings page which edits them.
var appSettings = parseSettings(null);

// Scrubber DOM refs.
var scrubberSection =
  document.getElementById("scrubber-section");
var scrubberControls =
  document.getElementById("scrubber-controls");
var scrubberSlider =
  document.getElementById("scrubber-slider");
var scrubberLabel =
  document.getElementById("scrubber-label");
var btnScrubStart =
  document.getElementById("btn-scrub-start");
var btnScrubPrev =
  document.getElementById("btn-scrub-prev");
var btnScrubNext =
  document.getElementById("btn-scrub-next");
var btnScrubEnd =
  document.getElementById("btn-scrub-end");
var btnEditFrames =
  document.getElementById("btn-edit-frames");
var overlaySelectGroup =
  document.getElementById("overlay-select-group");
var overlayDrawerHandle =
  document.getElementById("overlay-drawer-handle");
var overlaySelectMount =
  document.getElementById("overlay-select-mount");
var overlaySelect = null;
// Track how the picker was last built so it is only rebuilt when
// the option set actually changes (the Diff option appearing after
// a resume), avoiding leaked listeners from createCustomSelect.
var overlaySelectBuilt = false;
var overlaySelectHasDiff = false;
var diffSummary =
  document.getElementById("diff-summary");
var commitLegend =
  document.getElementById("commit-legend");
var diffOverlayControls =
  document.getElementById("diff-overlay-controls");
var diffOriginalSlider =
  document.getElementById("diff-original-opacity");
var diffEditedSlider =
  document.getElementById("diff-edited-opacity");
var diffBlendToggle =
  document.getElementById("diff-blend-toggle");
// Active visual overlay chosen in the picker:
// "none" | "conf" (heatmap) | "diff". Commit-order tinting is a
// separate persistent setting applied only when no overlay is
// selected (see effectiveColorMode).
var overlayMode = "none";
// Memoized per-run commit steps (position index -> settle step),
// null until first needed and invalidated whenever frameTokens
// is replaced (new run, resume, or session restore).
var commitSteps = null;
// Memoized intervention diff (branch vs original final frame),
// null until needed and invalidated alongside commitSteps.
var diffData = null;
// Diff-overlay layer opacities (0-100) and the "difference" blend
// toggle, controlled by the sliders shown in the overlay drawer.
var diffOriginalOpacity = 50;
var diffEditedOpacity = 100;
var diffBlend = false;

// Guided edit mode DOM refs.
var guidedEditControls =
  document.getElementById("guided-edit-controls");
var guidedEditStatus =
  document.getElementById("guided-edit-status");
var btnSelectFrame =
  document.getElementById("btn-select-frame");
var btnLockIn =
  document.getElementById("btn-lock-in");
var btnClearGuided =
  document.getElementById("btn-clear-guided");
var btnEditAnother =
  document.getElementById("btn-edit-another");
var btnRunToHere =
  document.getElementById("btn-run-to-here");
var btnResumeEnd =
  document.getElementById("btn-resume-end");
var btnConfirmEdit =
  document.getElementById("btn-confirm-edit");
var btnRetryEdit =
  document.getElementById("btn-retry-edit");
var btnExitEdit =
  document.getElementById("btn-exit-edit");
var remaskRandomizeRow =
  document.getElementById("remask-randomize-row");
var remaskRandomSlider =
  document.getElementById("remask-random-slider");
var remaskRandomCount =
  document.getElementById("remask-random-count");
var remaskRandomTotal =
  document.getElementById("remask-random-total");
var btnRemaskShuffle =
  document.getElementById("btn-remask-shuffle");
var shuffleLabel =
  document.getElementById("btn-remask-shuffle-label");

// ---- State ----

var ws = null;
var isGenerating = false;
var isSaving = false;
var saveCheckTimer = null;
var modelReady = false;
var paramsValid = true;
var reconnectDelay = RECONNECT_DELAY_MS;
var reconnectTimer = null;

// Accumulated data for the most recent completed run.
var frameHistory = [];
var frameTokens = [];
var perFrameElapsed = [];
var frameCanvasIndex = [];
var frameMeanConf = [];
var lastRunParams = null;
var lastFinalText = null;
var originalTotalFrames = 0;
var originalFrameHistory = [];
var originalFrameTokens = [];

// Scrubber and remasking state.
var scrubberActive = false;
var currentScrubFrame = 0;
var remaskedPositions = {};
var perFrameRemasked = {};
var remaskEdits = [];

// Guided multi-frame edit mode state.
// null | "select" | "edit" | "choice"
//      | "select_target" | "generating" | "review"
var remaskMode = null;
var remaskModeEdits = [];
// True once an edited run has been saved. Locks Edit Frames for the
// current run (until the next Generate) so a run cannot accrue a
// second, conflicting saved edit.
var editedRunSaved = false;
// True once the current run has been saved at least once (manually or
// via the Edit Frames auto-save). Prevents entering an edit session
// from duplicating the original run's saved entry.
var runSaved = false;
// Folder id of this run's last save. An edited/bundled save reuses it
// so the pre-edit run's folder is updated in place (one Analytics row)
// rather than creating a duplicate.
var lastSavedRunId = null;
// Prompt history (localStorage, per-browser): most-recent-first. While
// browsing, the box is read-only and the pre-browse text is held in
// promptHistoryDraft so Cancel can restore it.
var PROMPT_HISTORY_KEY = "diffusion_prompt_history";
var PROMPT_HISTORY_MAX = 30;
var promptHistory = [];
var promptHistoryIndex = -1;
var promptHistoryDraft = null;
var promptHistoryActive = false;
// Snapshot of the complete run taken when Edit Frames is entered.
// Partial resumes ("Run to Here") truncate the live run mid-way, so
// exiting restores this to avoid stranding the user on an
// incomplete run.
var preEditSnapshot = null;
var scrubberMinFrame = 0;
var guidedResumeAction = null;
var guidedTargetFrame = null;

// Resume state: when resuming, incoming frames are
// appended starting at resumeFrameOffset.
var isResuming = false;
var resumeFrameOffset = 0;

// ---- Output placeholder ----

// The resting state of the output area before any generation and
// after a New Run: the same hint index.html ships with. (The former
// idle ASCII scene / donut animations were removed.)
function showOutputPlaceholder() {
  outputArea.textContent = "";
  var placeholder = document.createElement("span");
  placeholder.id = "output-placeholder";
  placeholder.textContent =
    "Diffusion output will appear here...";
  outputArea.appendChild(placeholder);
}

// ---- Background floating characters ----

function spawnFloaters() {
  var container =
    document.getElementById("bg-floaters");
  if (!container) {
    return;
  }
  var chars =
    "01\u2591\u2592\u2593\u2588\u2584\u2580"
    + "\u28FF\u2847\u283F\u28C0\u28E4\u28FF"
    + "\u03A3\u0394\u03A9\u03BB\u2202\u2207";
  var COUNT = 30;

  for (var i = 0; i < COUNT; i++) {
    var el = document.createElement("span");
    el.className = "floater";
    el.textContent = chars[
      Math.floor(Math.random() * chars.length)
    ];
    el.style.left =
      Math.random() * 100 + "%";
    el.style.animationDuration =
      30 + Math.random() * 50 + "s";
    el.style.animationDelay =
      -(Math.random() * 60) + "s";
    el.style.fontSize =
      10 + Math.random() * 8 + "px";
    container.appendChild(el);
  }
}

spawnFloaters();

// ---- Model + schema-driven parameter panel ----

function setMaskChar() {
  if (
    activeModel
    && activeModel.capabilities
    && activeModel.capabilities.unresolved_char
  ) {
    MASK_CHAR =
      activeModel.capabilities.unresolved_char;
  }
}

// Autoregressive models stream a growing left-to-right sequence
// instead of denoising a masked canvas, so diffusion-only affordances
// (Diff overlay, Commit Order, convergence) are gated off for them.
function isAutoregressive() {
  return !!(
    activeModel
    && activeModel.capabilities
    && activeModel.capabilities.model_type === "autoregressive"
  );
}

function setLoadingText(text) {
  if (loadingText) {
    loadingText.textContent = text;
  }
}

function fetchModels() {
  return fetch("/api/models").then(function (r) {
    return r.json();
  });
}

function modelIsAR(model) {
  return !!(
    model
    && model.capabilities
    && model.capabilities.model_type === "autoregressive"
  );
}

// The device a plain row-click (on the name) would target: GPU when
// present, else CPU for the CPU-capable AR models. Diffusion is GPU.
function defaultDeviceFor(model) {
  if (modelIsAR(model)) {
    return gpuPresent ? "cuda" : "cpu";
  }
  return "cuda";
}

// Full VRAM readout shown to the side of a dropdown option on hover
// (the compact dropdown carries no headroom pill; the Main Menu does).
// Returns null when there is nothing quantitative to show.
function buildOptionInfo(model) {
  var required = Math.round(model.min_vram_gib || 0);
  var headroom = model.vram_headroom_gib;
  var pop = document.createElement("div");
  pop.className = "option-info";
  if (typeof headroom === "number") {
    var available = (
      (model.min_vram_gib || 0) + headroom
    ).toFixed(1);
    var positive = headroom >= 0;
    var sign = (positive ? "+" : "\u2212")
      + Math.abs(headroom).toFixed(1);
    // Body stays grey; only the trailing signed headroom is tinted
    // (green when it fits, red when short), matching the border.
    pop.appendChild(document.createTextNode(
      "Required " + required
      + " GiB \u00b7 Available " + available
      + " GiB \u00b7 "
    ));
    var head = document.createElement("span");
    head.className = "option-info-headroom "
      + (positive ? "is-positive" : "is-negative");
    head.textContent = sign;
    pop.appendChild(head);
    pop.classList.add(positive ? "is-positive" : "is-negative");
    return pop;
  }
  if (required > 0) {
    pop.textContent = "Requires ~" + required + " GiB VRAM";
    return pop;
  }
  return null;
}

// A small green device pill (used both in the collapsed value and,
// for AR models, as clickable GPU/CPU buttons in each option row).
function buildDevicePill(label, active) {
  var pill = document.createElement("span");
  pill.className =
    "device-pill" + (active ? " is-active" : "");
  pill.textContent = label;
  return pill;
}

// Interval id for the collapsed-slot device/headroom ticker.
var collapsedTickerTimer = null;

function stopCollapsedTicker() {
  if (collapsedTickerTimer !== null) {
    clearInterval(collapsedTickerTimer);
    collapsedTickerTimer = null;
  }
}

// Cycle the collapsed device pill between the device (GPU/CPU) and the
// signed VRAM headroom (+Z / -Z, no unit), ~2s per side with a fade.
// Static device label when the ticker Setting is off, reduced motion
// is preferred, or there is no headroom to show.
function startCollapsedTicker(pill, model, device) {
  stopCollapsedTicker();
  var deviceLabel = device === "cpu" ? "CPU" : "GPU";
  // Move the label into an inner span so only the text fades; the
  // pill's border and background stay static through the cycle.
  var textEl = document.createElement("span");
  textEl.className = "ticker-text";
  textEl.textContent = deviceLabel;
  pill.textContent = "";
  pill.appendChild(textEl);
  var headroom = model.vram_headroom_gib;
  // Headroom is a GPU-VRAM figure, so it does not apply on CPU; there
  // the pill stays a static "CPU" tag rather than cycling.
  var canTick = appSettings.gpuTicker
    && !prefersReducedMotion()
    && device !== "cpu"
    && typeof headroom === "number";
  if (!canTick) {
    return;
  }
  var headLabel = (headroom >= 0 ? "+" : "\u2212")
    + Math.abs(headroom).toFixed(1);
  var showingDevice = true;
  collapsedTickerTimer = setInterval(function () {
    textEl.classList.add("ticker-fade");
    setTimeout(function () {
      showingDevice = !showingDevice;
      textEl.textContent = showingDevice ? deviceLabel : headLabel;
      // Shrink the wider signed-headroom face so it fits the pill's
      // fixed GPU/CPU width without stretching the border.
      textEl.classList.toggle("is-headroom", !showingDevice);
      textEl.classList.remove("ticker-fade");
    }, 350);
  }, 2000);
}

function setModelSelectValue(id) {
  if (!modelSelectValue) {
    return;
  }
  stopCollapsedTicker();
  var m = models[id];
  modelSelectValue.innerHTML = "";
  var nameEl = document.createElement("span");
  nameEl.className = "model-select-value-name";
  nameEl.textContent = m ? m.display_name : (id || "-");
  // Full name on hover, since a long name ellipsizes to the fixed width.
  nameEl.title = m ? m.display_name : "";
  modelSelectValue.appendChild(nameEl);
  if (m) {
    // The collapsed value shows the active model's current device.
    var dev = id === activeModelId && activeDevice
      ? activeDevice
      : defaultDeviceFor(m);
    var pill = buildDevicePill(
      dev === "cpu" ? "CPU" : "GPU", false
    );
    pill.classList.add("device-pill-collapsed");
    modelSelectValue.appendChild(pill);
    startCollapsedTicker(pill, m, dev);
  }
}

// Device control for one option row: a static GPU pill for diffusion
// models, or a clickable GPU/CPU toggle for AR models. Each button
// routes through requestSwitch so any change goes past the confirm.
function buildOptionDevice(model, activeId) {
  var wrap = document.createElement("span");
  wrap.className = "option-device";
  if (!modelIsAR(model)) {
    wrap.appendChild(buildDevicePill("GPU", true));
    return wrap;
  }
  var isActiveModel = model.id === activeId;
  var current = isActiveModel && activeDevice
    ? activeDevice
    : defaultDeviceFor(model);
  var devices = [
    { value: "cuda", label: "GPU" },
    { value: "cpu", label: "CPU" },
  ];
  for (var i = 0; i < devices.length; i++) {
    (function (dev) {
      var btn = document.createElement("button");
      btn.type = "button";
      // The loaded model's current device is redundant to re-select, so
      // it is locked (is-current); the other device stays switchable.
      var isCurrent = isActiveModel && dev.value === activeDevice;
      btn.className =
        "device-pill device-pill-btn"
        + (dev.value === current ? " is-active" : "")
        + (isCurrent ? " is-current" : "");
      btn.textContent = dev.label;
      if (dev.value === "cuda" && !gpuPresent) {
        btn.disabled = true;
        btn.title = "No GPU detected";
      }
      if (isCurrent) {
        btn.title = "Currently loaded";
      }
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        if (btn.disabled) {
          return;
        }
        requestSwitch(model.id, dev.value);
      });
      wrap.appendChild(btn);
    })(devices[i]);
  }
  return wrap;
}

function setModelSelectDisabled(disabled) {
  modelSelectDisabled = disabled;
  if (modelSelect) {
    modelSelect.classList.toggle("disabled", disabled);
  }
  if (disabled) {
    closeModelList();
  }
}

function openModelList() {
  if (modelSelectDisabled || !modelSelectList) {
    return;
  }
  closeSwitchConfirm();
  modelSelectList.hidden = false;
  modelSelect.classList.add("open");
}

function closeModelList() {
  if (!modelSelectList) {
    return;
  }
  modelSelectList.hidden = true;
  modelSelect.classList.remove("open");
}

function toggleModelList() {
  if (modelSelectList && modelSelectList.hidden) {
    openModelList();
  } else {
    closeModelList();
  }
}

function renderModelSelector(list, activeId) {
  if (!modelSelect || !modelSelectList) {
    return;
  }
  modelSelectList.innerHTML = "";
  for (var i = 0; i < list.length; i++) {
    var m = list[i];
    var li = document.createElement("li");
    li.className =
      "model-select-option"
      + (m.id === activeId ? " is-active" : "");
    li.setAttribute("role", "option");
    li.setAttribute("data-id", m.id);
    var nameEl = document.createElement("span");
    nameEl.className = "model-select-name";
    nameEl.textContent = m.display_name;
    nameEl.title = m.display_name;
    li.appendChild(nameEl);
    li.appendChild(buildOptionDevice(m, activeId));
    // Full VRAM readout to the side of the row on hover (keeps the
    // option compact; the pill/toggle alone stays in the row).
    var info = buildOptionInfo(m);
    if (info) {
      li.appendChild(info);
      // Capture this row's info element per iteration: a plain closure
      // over the loop-scoped ``var info`` would leave every row toggling
      // the last option's popover (SmolLM3's).
      (function (infoEl) {
        li.addEventListener("mouseenter", function () {
          infoEl.classList.add("is-visible");
        });
        li.addEventListener("mouseleave", function () {
          infoEl.classList.remove("is-visible");
        });
      })(info);
    }
    modelSelectList.appendChild(li);
  }
  setModelSelectValue(activeId);
  sizeModelSelect(list);
}

// ---- Model / device switch confirmation ----

var switchConfirmEl = null;

function closeSwitchConfirm() {
  if (switchConfirmEl && switchConfirmEl.parentNode) {
    switchConfirmEl.parentNode.removeChild(switchConfirmEl);
  }
  switchConfirmEl = null;
}

// Any model or device change routes here: no-op if already active on
// that device, otherwise open a small confirm popover on the dropdown.
function requestSwitch(id, device) {
  closeModelList();
  var model = models[id];
  if (!model) {
    return;
  }
  if (id === activeModelId && device === activeDevice) {
    return;
  }
  openSwitchConfirm(id, device);
}

function openSwitchConfirm(id, device) {
  closeSwitchConfirm();
  var model = models[id];
  if (!model || !modelSelect) {
    return;
  }
  var box = document.createElement("div");
  box.className = "switch-confirm";
  // Clicks inside the popover must not bubble to the dropdown's
  // toggle handler (which would open/close the option list).
  box.addEventListener("click", function (e) {
    e.stopPropagation();
  });
  var currentName = activeModel
    ? activeModel.display_name
    : "the current model";
  var msg = document.createElement("span");
  msg.className = "switch-confirm-msg";
  msg.textContent =
    "Unload the current model " + currentName
    + " and load " + model.display_name + " on "
    + (device === "cpu" ? "CPU" : "GPU") + "?";
  var actions = document.createElement("span");
  actions.className = "switch-confirm-actions";
  var yes = document.createElement("button");
  yes.type = "button";
  yes.className = "switch-confirm-yes";
  yes.title = "Confirm switch";
  yes.setAttribute("aria-label", "Confirm switch");
  yes.textContent = "\u2713";
  yes.addEventListener("click", function (e) {
    e.stopPropagation();
    closeSwitchConfirm();
    switchModel(id, device);
  });
  var no = document.createElement("button");
  no.type = "button";
  no.className = "switch-confirm-no";
  no.title = "Cancel";
  no.setAttribute("aria-label", "Cancel switch");
  no.textContent = "\u2717";
  no.addEventListener("click", function (e) {
    e.stopPropagation();
    closeSwitchConfirm();
  });
  actions.appendChild(yes);
  actions.appendChild(no);
  box.appendChild(msg);
  box.appendChild(actions);
  modelSelect.appendChild(box);
  switchConfirmEl = box;
}

function sizeModelSelect(list) {
  if (!modelSelect || !list.length) {
    return;
  }
  var names = [];
  for (var i = 0; i < list.length; i++) {
    names.push(list[i].display_name);
  }
  var width = measureTextWidth(
    names, modelSelectValue || modelSelect
  );
  modelSelect.style.minWidth =
    Math.ceil(width) + 48 + "px";
}

function numericSpecs() {
  var out = [];
  if (!activeModel) {
    return out;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    if (
      specs[i].type === "int"
      || specs[i].type === "float"
    ) {
      out.push(specs[i]);
    }
  }
  return out;
}

// Per-device override for a spec, if one applies to the active device.
function specOverride(spec) {
  if (
    spec.overrides
    && activeDevice
    && spec.overrides[activeDevice]
  ) {
    return spec.overrides[activeDevice];
  }
  return null;
}

// Device-aware (low, high) bounds: an active-device override wins over
// the base recommended/experimental bounds when present.
function specBounds(spec, experimental) {
  var override = specOverride(spec);
  if (override) {
    var ob = experimental
      ? override.experimental
      : override.recommended;
    if (ob) {
      return ob;
    }
  }
  return experimental ? spec.experimental : spec.recommended;
}

// Device-aware default value for a spec.
function specDefault(spec) {
  var override = specOverride(spec);
  if (
    override
    && override.default !== null
    && override.default !== undefined
  ) {
    return override.default;
  }
  return spec.default;
}

function activeLimits() {
  var experimental = toggleExperimental.checked;
  var out = {};
  if (!activeModel) {
    return out;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var s = specs[i];
    var b = specBounds(s, experimental);
    if (b) {
      out[s.name] = { min: b[0], max: b[1] };
    }
  }
  return out;
}

// Tiny "?" icon whose tooltip is filled by updateRangeLabels.
function buildInfoIcon(spec) {
  var info = document.createElement("span");
  info.className = "info-icon info-icon-sm";
  info.textContent = "?";
  info.setAttribute("aria-label", spec.label + " info");
  var tip = document.createElement("span");
  tip.className = "tooltip";
  info.appendChild(tip);
  // Hover-only: clicking must not toggle/focus a bound control.
  info.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
  });
  paramTooltips[spec.name] = tip;
  return info;
}

// Numeric / select params render in the hyperparameter row.
function buildParamField(spec, input) {
  var group = document.createElement("div");
  group.className = "param-group";
  var label = document.createElement("label");
  label.setAttribute("for", "param-" + spec.name);
  label.appendChild(document.createTextNode(spec.label));
  label.appendChild(buildInfoIcon(spec));
  group.appendChild(label);
  group.appendChild(input);
  paramFields.appendChild(group);
}

// Boolean params render as a toggle next to Experimental,
// mirroring its layout (toggle -> label -> info icon).
function buildModeToggle(spec, checkbox) {
  if (!modeExtra) {
    return;
  }
  var wrap = document.createElement("span");
  wrap.className = "mode-toggle";
  var toggle = document.createElement("label");
  toggle.className = "toggle-switch";
  var slider = document.createElement("span");
  slider.className = "toggle-slider";
  toggle.appendChild(checkbox);
  toggle.appendChild(slider);
  var name = document.createElement("span");
  name.className = "toggle-label";
  name.textContent = spec.label;
  wrap.appendChild(toggle);
  wrap.appendChild(name);
  wrap.appendChild(buildInfoIcon(spec));
  modeExtra.appendChild(wrap);
}

function buildParamPanel(model) {
  paramInputs = {};
  paramTooltips = {};
  paramFields.innerHTML = "";
  if (modeExtra) {
    modeExtra.innerHTML = "";
  }
  var specs = model.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var s = specs[i];
    var input = buildParamInput(s);
    paramInputs[s.name] = input;

    if (s.type === "bool") {
      buildModeToggle(s, input);
    } else {
      buildParamField(s, input);
    }

    if (s.type === "int" || s.type === "float") {
      input.addEventListener("input", validateAllParams);
    } else {
      input.addEventListener("change", validateAllParams);
    }
  }
  applyLimits();
}

function buildParamInput(spec) {
  var input;
  if (spec.type === "select") {
    var options = (spec.options || []).map(function (v) {
      return { value: v, label: prettifyOption(v) };
    });
    input = createCustomSelect(options, spec.default);
  } else if (spec.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(specDefault(spec));
  } else {
    input = document.createElement("input");
    input.type = "number";
    if (spec.step !== null && spec.step !== undefined) {
      input.step = String(spec.step);
    }
    input.value = String(specDefault(spec));
  }
  input.id = "param-" + spec.name;
  return input;
}

function paramRangeText(spec, limits) {
  if (spec.type === "select") {
    return (spec.options || [])
      .map(prettifyOption)
      .join(" / ");
  }
  if (spec.type === "bool") {
    return "on / off";
  }
  var b = limits[spec.name];
  if (b) {
    return "(" + b.min + "\u2013" + b.max + ")";
  }
  return "";
}

// Fills each hyperparameter's "?" tooltip: an italic "Range:"
// line on top, a blank line, then the concise description.
function updateRangeLabels() {
  if (!activeModel) {
    return;
  }
  var limits = activeLimits();
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var s = specs[i];
    var tip = paramTooltips[s.name];
    if (!tip) {
      continue;
    }
    tip.innerHTML = "";
    var rangeLine = document.createElement("div");
    var em = document.createElement("em");
    em.textContent = "Range:";
    rangeLine.appendChild(em);
    rangeLine.appendChild(
      document.createTextNode(
        " " + paramRangeText(s, limits)
      )
    );
    tip.appendChild(rangeLine);
    if (s.help) {
      var desc = document.createElement("div");
      desc.className = "tooltip-desc";
      desc.textContent = s.help;
      tip.appendChild(desc);
    }
  }
}

// Uniform width for every hyperparameter box, sized to the
// widest label / select across ALL models so the row spacing is
// consistent within and across model views.
function applyUniformParamWidth(allModels) {
  var refLabel = paramFields.querySelector("label");
  if (!refLabel) {
    return;
  }
  var refControl =
    paramFields.querySelector("input, .custom-select")
    || refLabel;
  // Canvas measureText ignores letter-spacing (0.08em at 10px),
  // so add a per-character fudge for the uppercased labels.
  var letterSpacing = 0.8;
  var maxWidth = 90;
  for (var mi = 0; mi < allModels.length; mi++) {
    var specs = allModels[mi].param_specs || [];
    for (var si = 0; si < specs.length; si++) {
      var s = specs[si];
      if (s.type === "bool") {
        continue;
      }
      var upper = String(s.label).toUpperCase();
      var labelWidth =
        measureTextWidth([upper], refLabel)
        + letterSpacing * Math.max(0, upper.length - 1)
        + 26;
      maxWidth = Math.max(maxWidth, labelWidth);
      if (s.type === "select") {
        var opts = (s.options || []).map(prettifyOption);
        var optWidth =
          measureTextWidth(opts, refControl) + 40;
        maxWidth = Math.max(maxWidth, optWidth);
      }
    }
  }
  document.documentElement.style.setProperty(
    "--param-width", Math.ceil(maxWidth) + "px"
  );
}

function applyLimits() {
  var limits = activeLimits();
  var keys = Object.keys(limits);
  for (var i = 0; i < keys.length; i++) {
    var input = paramInputs[keys[i]];
    if (!input || input.type !== "number") {
      continue;
    }
    var b = limits[keys[i]];
    input.min = b.min;
    input.max = b.max;
    var val = parseFloat(input.value);
    if (!isNaN(val)) {
      if (val < b.min) {
        input.value = b.min;
      } else if (val > b.max) {
        input.value = b.max;
      }
    }
  }
  updateRangeLabels();
  validateAllParams();
}

// ---- Comprehensive validation ----

function validateAllParams() {
  var limits = activeLimits();
  var errors = [];
  var specs = numericSpecs();

  for (var i = 0; i < specs.length; i++) {
    var inp = paramInputs[specs[i].name];
    if (inp) {
      inp.classList.remove("input-warn");
    }
  }

  for (var j = 0; j < specs.length; j++) {
    var s = specs[j];
    var input = paramInputs[s.name];
    if (!input) {
      continue;
    }
    var bound = limits[s.name];
    var raw = input.value.trim();
    var val = parseFloat(raw);

    if (raw === "" || isNaN(val)) {
      input.classList.add("input-warn");
      errors.push(s.label + " is empty or invalid.");
      continue;
    }
    if (bound && val < bound.min) {
      input.classList.add("input-warn");
      errors.push(
        val < 0
          ? s.label + " cannot be negative."
          : s.label + " must be at least "
            + bound.min + "."
      );
      continue;
    }
    if (bound && val > bound.max) {
      input.classList.add("input-warn");
      errors.push(
        s.label + " must be at most "
        + bound.max + "."
      );
    }
  }

  validateDivisibility(errors);

  if (errors.length > 0) {
    validationHint.textContent = errors[0];
    validationHint.hidden = false;
    paramsValid = false;
  } else {
    validationHint.hidden = true;
    validationHint.textContent = "";
    paramsValid = true;
  }
  updateGenerateButton();
}

// LLaDA-style block divisibility, applied only when the
// relevant params exist in the active model's schema.
function validateDivisibility(errors) {
  var g = paramInputs["gen_length"];
  var b = paramInputs["block_length"];
  var st = paramInputs["steps"];
  if (!g || !b || !st) {
    return;
  }
  var genLength = parseInt(g.value, 10);
  var blockLength = parseInt(b.value, 10);
  var steps = parseInt(st.value, 10);
  var genOk = !g.classList.contains("input-warn");
  var blkOk = !b.classList.contains("input-warn");
  var stpOk = !st.classList.contains("input-warn");

  if (
    genOk && blkOk
    && blockLength > 0
    && genLength % blockLength !== 0
  ) {
    g.classList.add("input-warn");
    b.classList.add("input-warn");
    errors.push(
      "Gen Length (" + genLength
      + ") must be divisible by Block Length ("
      + blockLength + ")."
    );
  } else if (
    genOk && blkOk && stpOk
    && blockLength > 0
    && genLength % blockLength === 0
  ) {
    var numBlocks = genLength / blockLength;
    if (numBlocks > 0 && steps % numBlocks !== 0) {
      st.classList.add("input-warn");
      errors.push(
        "Steps (" + steps
        + ") must be divisible by num_blocks ("
        + numBlocks + ")."
      );
    }
  }
}

function getParamValues() {
  var out = {};
  if (!activeModel) {
    return out;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var s = specs[i];
    var input = paramInputs[s.name];
    if (!input) {
      continue;
    }
    if (s.type === "int") {
      out[s.name] = parseInt(input.value, 10);
    } else if (s.type === "float") {
      out[s.name] = parseFloat(input.value);
    } else if (s.type === "bool") {
      out[s.name] = input.checked;
    } else {
      out[s.name] = input.value;
    }
  }
  return out;
}

function switchModel(id, device) {
  // Same model on the same device is a no-op (requestSwitch also
  // guards this before showing the confirm).
  if (id === activeModelId && (device || activeDevice) === activeDevice) {
    return;
  }
  suppressReconnect = true;
  if (ws) {
    try {
      ws.close();
    } catch (_e) {
      // ignore
    }
  }
  var name = models[id] ? models[id].display_name : id;
  setLoadingText("Loading " + name + "\u2026");
  loadingOverlay.classList.remove("hidden");
  setModelSelectDisabled(true);

  var options = { method: "POST" };
  if (device) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify({ device: device });
  }
  fetch(
    "/api/models/" + encodeURIComponent(id) + "/activate",
    options
  )
    .then(function (r) {
      return r.json();
    })
    .then(function (res) {
      // Non-blocking activation: poll until the new worker is ready,
      // then reload so the page picks up the new model/device.
      if (res && res.ok) {
        pollSwitch(name);
      } else {
        throw new Error(
          (res && res.message) || "activation failed"
        );
      }
    })
    .catch(switchFailed);
}

function pollSwitch(name) {
  fetch("/api/models/activation")
    .then(function (r) {
      return r.json();
    })
    .then(function (status) {
      if (status.state === "ready") {
        location.reload();
        return;
      }
      if (status.state === "error") {
        switchFailed(
          new Error(status.message || "load failed")
        );
        return;
      }
      if (
        status.state === "downloading"
        && status.progress
        && typeof status.progress.fraction === "number"
      ) {
        setLoadingText(
          "Downloading " + name + " "
          + Math.round(status.progress.fraction * 100) + "%"
        );
      } else {
        setLoadingText("Loading " + name + "\u2026");
      }
      setTimeout(function () {
        pollSwitch(name);
      }, 500);
    })
    .catch(function () {
      setTimeout(function () {
        pollSwitch(name);
      }, 800);
    });
}

function switchFailed(err) {
  suppressReconnect = false;
  setModelSelectDisabled(false);
  setModelSelectValue(activeModelId);
  loadingOverlay.classList.add("hidden");
  statusMessage.textContent =
    "Model switch failed: " + err.message;
  statusMessage.style.color = "var(--danger)";
}

// ---- WebSocket connection ----

function connect() {
  if (
    ws
    && (
      ws.readyState === WebSocket.OPEN
      || ws.readyState === WebSocket.CONNECTING
    )
  ) {
    return;
  }

  var protocol =
    location.protocol === "https:"
      ? "wss:" : "ws:";
  var url =
    protocol + "//" + location.host + "/ws";
  ws = new WebSocket(url);

  ws.onopen = function () {
    reconnectDelay = RECONNECT_DELAY_MS;
    setBadge("loading");
  };

  ws.onclose = function () {
    setBadge("disconnected");
    modelReady = false;
    updateGenerateButton();
    if (!suppressReconnect) {
      scheduleReconnect();
    }
  };

  ws.onerror = function () {
    ws.close();
  };

  ws.onmessage = function (event) {
    var data;
    try {
      data = JSON.parse(event.data);
    } catch (_unused) {
      return;
    }
    handleMessage(data);
  };
}

function scheduleReconnect() {
  if (reconnectTimer) {
    return;
  }
  reconnectTimer = setTimeout(function () {
    reconnectTimer = null;
    connect();
  }, reconnectDelay);
  reconnectDelay = Math.min(
    reconnectDelay * 2,
    MAX_RECONNECT_DELAY_MS
  );
}

// ---- Message handler ----

function handleMessage(data) {
  switch (data.type) {
    case "model_status":
      handleModelStatus(data);
      break;
    case "frame":
      handleFrame(data);
      break;
    case "done":
      handleDone(data);
      break;
    case "error":
      handleError(data);
      break;
  }
}

function handleModelStatus(data) {
  if (data.status === "loading") {
    setBadge("loading");
    modelReady = false;
    setLoadingText(
      "Loading "
      + (activeModel
        ? activeModel.display_name
        : "model")
      + "\u2026"
    );
    loadingOverlay.classList.remove("hidden");
    updateGenerateButton();
  } else if (data.status === "ready") {
    setBadge("ready");
    modelReady = true;
    loadingOverlay.classList.add("hidden");
    updateGenerateButton();
  }
}

function handleFrame(data) {
  frameHistory.push(data.text);
  if (data.tokens) {
    frameTokens.push(data.tokens);
  } else {
    frameTokens.push(null);
  }

  if (typeof data.elapsed === "number") {
    perFrameElapsed.push(data.elapsed);
  }
  frameCanvasIndex.push(
    typeof data.canvas_index === "number"
      ? data.canvas_index
      : 0
  );
  frameMeanConf.push(
    typeof data.mean_conf === "number"
      ? data.mean_conf
      : null
  );

  renderFrame(data.text);

  var prefix = isResuming ? "Resuming " : "Step ";
  var displayStep;
  if (typeof data.total_steps === "number") {
    displayStep = prefix + data.index
      + "/" + data.total_steps;
  } else {
    // Adaptive-stopping models (DiffusionGemma) have no fixed
    // step total; report the step and its canvas instead.
    // canvas_index is 0-based internally; display it 1-based.
    var canvas = typeof data.canvas_index === "number"
      ? data.canvas_index
      : 0;
    displayStep = prefix + data.index
      + ", Canvas " + (canvas + 1);
  }
  statusStep.textContent = displayStep;

  if (typeof data.elapsed === "number") {
    statusElapsed.textContent =
      "Elapsed: " + data.elapsed.toFixed(1) + "s";
  }
}

function handleDone(data) {
  setGenerating(false);
  isResuming = false;
  stopStatusDots();
  statusMessage.textContent = "Done.";
  if (data.final_text) {
    lastFinalText = data.final_text;
  }
  if (thinkingPanel && thinkingContent) {
    if (data.thinking) {
      thinkingContent.textContent = data.thinking;
      thinkingPanel.hidden = false;
    } else {
      thinkingPanel.hidden = true;
      thinkingContent.textContent = "";
    }
  }
  lastRunParams = getParamValues();
  if (originalTotalFrames === 0) {
    originalTotalFrames = frameHistory.length;
    originalFrameHistory = frameHistory.slice();
    originalFrameTokens = frameTokens.slice();
  }

  setSaveAvailable(true);

  if (remaskMode === "generating") {
    handleGuidedDone();
  } else {
    activateScrubber();
  }

  // Persist the completed run so it survives navigating to
  // Analytics and back (skip while mid guided-edit).
  if (remaskMode === null) {
    saveSessionState();
  }
}

function handleError(data) {
  setGenerating(false);
  isResuming = false;
  stopStatusDots();
  if (remaskMode !== null) {
    resetGuidedMode();
  }
  statusMessage.textContent =
    "Error: " + (data.message || "unknown");
  statusMessage.style.color = "var(--danger)";
  setTimeout(function () {
    statusMessage.style.color = "";
  }, 5000);
  if (frameHistory.length > 1) {
    activateScrubber();
  }
}

// ---- Rendering ----

function renderFrame(text) {
  outputArea.classList.remove("diff-overlay-mode");
  var fragment =
    document.createDocumentFragment();
  for (var i = 0; i < text.length; i++) {
    var ch = text[i];
    if (ch === MASK_CHAR) {
      var span = document.createElement("span");
      span.className = "char-mask";
      span.textContent = ch;
      fragment.appendChild(span);
    } else if (ch === "\n") {
      fragment.appendChild(
        document.createTextNode("\n")
      );
    } else {
      var span2 = document.createElement("span");
      span2.className = "char-resolved";
      span2.textContent = ch;
      fragment.appendChild(span2);
    }
  }
  outputArea.textContent = "";
  outputArea.appendChild(fragment);
}

function renderFinalText(text) {
  outputArea.textContent = "";
  var span = document.createElement("span");
  span.className = "char-resolved";
  span.textContent = text;
  outputArea.appendChild(span);
}

// heatColor now lives in overlays.js (shared with Analytics).

// Per-position commit step for the current run: the step after
// which a position last changed to its final value. Derived
// purely from frameTokens (the final frame is ground truth), so
// it is exact for LLaDA (resolved tokens are frozen) and a
// "settle" proxy for DiffusionGemma. Positions still unresolved
// at the last frame get -1 (left uncolored). Result is memoized
// in commitSteps and invalidated whenever frameTokens changes.
function computeCommitSteps() {
  return overlaysComputeCommitSteps(frameTokens);
}

// commitColor now lives in overlays.js (shared with Analytics).

// Compare the branch's final frame against the retained original
// run's final frame, position-aligned on the shared canvas. Returns
// per-position change flags, the original display text (for
// tooltips), the remask-origin positions, and a divergence summary.
function computeDiff() {
  var cur = frameTokens.length
    ? frameTokens[frameTokens.length - 1]
    : null;
  var orig = originalFrameTokens.length
    ? originalFrameTokens[originalFrameTokens.length - 1]
    : null;
  return overlaysComputeDiff(cur, orig, remaskEdits);
}

// diffColor and the layered-diff builder now live in overlays.js
// (shared with Analytics).

// Draw the original and edited runs at the current frame as two
// stacked layers (independent opacity + optional difference blend)
// so overlaps and divergences can be compared directly. The layer
// construction is shared via overlaysBuildDiffLayers; this wrapper
// resolves the per-frame tokens and owns the output container.
function renderDiffOverlay(frameIndex) {
  if (diffData === null) {
    diffData = computeDiff();
  }
  var editedTokens = frameTokens[frameIndex] || [];
  var oIdx = Math.min(
    frameIndex, originalFrameTokens.length - 1
  );
  var origTokens =
    (oIdx >= 0 ? originalFrameTokens[oIdx] : null) || [];

  outputArea.textContent = "";
  outputArea.classList.add("diff-overlay-mode");
  outputArea.appendChild(
    overlaysBuildDiffLayers(
      origTokens,
      editedTokens,
      diffData,
      {
        originalOpacity: diffOriginalOpacity,
        editedOpacity: diffEditedOpacity,
        blend: diffBlend,
      },
      MASK_CHAR
    )
  );
}

// Which coloring paints tokens: the overlay picker's selection
// (Heatmap/Commit Order/Diff), or none. Commit Order and Diff are
// diffusion-only and omitted from the picker for AR runs; the guard
// keeps a stale selection from tinting them.
function effectiveColorMode() {
  if (overlayMode === "commit" && isAutoregressive()) {
    return "none";
  }
  return overlayMode;
}

// Apply the effective color mode to one resolved-token span,
// mutating its inline color and (where useful) its tooltip. Mask
// and user-remasked tokens never reach here.
function applyTokenColor(span, index, tok) {
  var mode = effectiveColorMode();
  if (mode === "conf") {
    if (typeof tok.c === "number") {
      span.style.color = heatColor(tok.c);
    }
    return;
  }
  if (mode === "commit") {
    if (commitSteps === null) {
      commitSteps = computeCommitSteps();
    }
    var step = commitSteps[index];
    if (typeof step === "number" && step >= 0) {
      span.style.color = commitColor(
        step, frameTokens.length - 1
      );
      span.title += "\nResolved at step: " + step;
    }
    return;
  }
  if (mode === "diff") {
    if (diffData === null) {
      diffData = computeDiff();
    }
    if (diffData.origins[index]) {
      span.style.color = "#ff8a3d";
      span.title += "\n(remasked here)";
    } else if (diffData.changed[index]) {
      span.style.color = diffColor(true);
      span.title += "\nwas: " + diffData.origText[index];
    } else {
      span.style.color = diffColor(false);
    }
  }
}

// Format a token's confidence for the hover tooltip. Masked or
// unlabeled tokens report 0.
function confLabel(c) {
  if (typeof c !== "number") {
    return "0";
  }
  return String(+c.toFixed(3));
}

// First tooltip line: token position. LLaDA has a fixed token
// budget (the gen-length canvas), so it shows X/total; other
// models (DiffusionGemma) just show the index.
function tokenLabel(index, total) {
  if (activeModelId === "llada") {
    return "Token " + (index + 1) + "/" + total;
  }
  return "Token: " + (index + 1);
}

// The per-token hover highlight is now a global preference,
// independent of any coloring overlay.
function updateHoverHighlight() {
  if (!outputArea) {
    return;
  }
  outputArea.classList.toggle(
    "token-hover-highlight",
    appSettings.highlightTokens
  );
}

// Select the active visual overlay from the picker and re-render.
function setOverlayMode(mode) {
  overlayMode = mode;
  updateDiffSummary();
  updateDiffOverlayControls();
  updateCommitLegend();
  if (scrubberActive) {
    renderFrameWithTokens(currentScrubFrame);
  }
}

// The commit-order legend (early -> late gradient) shows only while
// the Commit Order overlay is the active selection.
function updateCommitLegend() {
  if (commitLegend) {
    commitLegend.hidden = overlayMode !== "commit";
  }
}

// The Original/Edited opacity sliders + blend toggle only apply to
// the diff overlay, so they show only while it is selected.
function updateDiffOverlayControls() {
  if (!diffOverlayControls) {
    return;
  }
  diffOverlayControls.hidden = !(
    overlayMode === "diff" && diffAvailable() && remaskMode === null
  );
}

// Reset the diff-overlay sliders/blend to defaults (called per run).
function resetDiffOverlay() {
  diffOriginalOpacity = 50;
  diffEditedOpacity = 100;
  diffBlend = false;
  if (diffOriginalSlider) {
    diffOriginalSlider.value = "50";
  }
  if (diffEditedSlider) {
    diffEditedSlider.value = "100";
  }
  if (diffBlendToggle) {
    diffBlendToggle.checked = false;
  }
}

// Show the "diverged N/total" readout while the diff overlay is on.
function updateDiffSummary() {
  if (!diffSummary) {
    return;
  }
  if (overlayMode !== "diff") {
    diffSummary.hidden = true;
    return;
  }
  if (diffData === null) {
    diffData = computeDiff();
  }
  var total = diffData.totalCount;
  var changed = diffData.changedCount;
  var pct = total > 0
    ? Math.round((changed / total) * 100)
    : 0;
  diffSummary.textContent =
    "Diverged " + changed + "/" + total
    + " (" + pct + "%)";
  diffSummary.hidden = false;
}

// The intervention diff only makes sense once a resume has produced
// a branch to compare against the retained original run.
function diffAvailable() {
  return (
    originalTotalFrames > 0
    && remaskEdits.length > 0
    && originalFrameTokens.length > 0
  );
}

// Slide the overlay drawer open or closed and flip the handle glyph
// (pointing left to invite opening, right to push it back in).
function setOverlayDrawerOpen(open) {
  if (!overlaySelectGroup) {
    return;
  }
  overlaySelectGroup.classList.toggle("open", open);
  if (overlayDrawerHandle) {
    overlayDrawerHandle.textContent = open ? "\u203a" : "\u2039";
    overlayDrawerHandle.title = open
      ? "Collapse overlay options"
      : "Overlay options";
  }
}

// (Re)build the top-right overlay picker. "Diff vs Original" is
// always listed but disabled until a resume branch exists. Rebuilt
// only when that availability flips (to refresh the disabled state).
function buildOverlaySelect() {
  if (!overlaySelectMount) {
    return;
  }
  var hasDiff = diffAvailable();
  if (overlayMode === "diff" && !hasDiff) {
    overlayMode = "none";
  }
  // Commit Order is diffusion-only; drop a stale selection for AR runs.
  if (overlayMode === "commit" && isAutoregressive()) {
    overlayMode = "none";
  }
  // Keep the commit legend in sync with the (possibly reset) mode on
  // every (re)build or reuse, not just on an explicit picker change.
  updateCommitLegend();
  // Option set unchanged: just reset the collapsed selection.
  if (overlaySelectBuilt && hasDiff === overlaySelectHasDiff) {
    if (overlaySelect) {
      overlaySelect.value = overlayMode;
    }
    return;
  }
  var options = [
    { value: "none", label: "None" },
    { value: "conf", label: "Heatmap" },
  ];
  // Commit Order and Diff vs Original are diffusion-only. Commit Order
  // tints by resolution step; Diff needs an edit-and-resume branch,
  // which autoregressive models do not support, so both are omitted for
  // them (Heatmap, the natural per-token confidence view, stays).
  if (!isAutoregressive()) {
    options.push({ value: "commit", label: "Commit Order" });
    options.push({
      value: "diff",
      label: "Diff vs Original",
      disabled: !hasDiff,
      title: hasDiff
        ? undefined
        : "Edit and resume a run (via Edit Frames) to"
          + " compare it against the original.",
    });
  }
  overlaySelectMount.innerHTML = "";
  overlaySelect = createCustomSelect(options, overlayMode);
  overlaySelectMount.appendChild(overlaySelect);
  sizeCustomSelect(overlaySelect);
  overlaySelect.addEventListener("change", function () {
    setOverlayMode(overlaySelect.value);
  });
  overlaySelectBuilt = true;
  overlaySelectHasDiff = hasDiff;
}

// ---- Prompt history (localStorage) ----

function loadPromptHistory() {
  try {
    var raw = localStorage.getItem(PROMPT_HISTORY_KEY);
    if (!raw) {
      return;
    }
    var parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      promptHistory = parsed.filter(function (p) {
        return typeof p === "string" && p.length > 0;
      });
    }
  } catch (_e) {
    // Unavailable/corrupt storage: start with an empty history.
    promptHistory = [];
  }
}

function savePromptHistoryStore() {
  // Write-through to the server (see persistSet) so history survives
  // desktop-app restarts, not just the current window origin.
  persistSet(PROMPT_HISTORY_KEY, JSON.stringify(promptHistory));
}

// Record a used prompt at the front (most recent first), de-duplicated
// and capped. Called when a run starts.
function pushPromptHistory(text) {
  var prompt = (text || "").trim();
  if (!prompt) {
    return;
  }
  promptHistory = promptHistory.filter(function (p) {
    return p !== prompt;
  });
  promptHistory.unshift(prompt);
  if (promptHistory.length > PROMPT_HISTORY_MAX) {
    promptHistory.length = PROMPT_HISTORY_MAX;
  }
  savePromptHistoryStore();
  updatePromptHistoryUI();
}

// Show the history control only when there is something to browse.
function updatePromptHistoryUI() {
  if (promptHistoryGroup) {
    promptHistoryGroup.hidden = promptHistory.length === 0;
  }
}

function _setPromptHistoryCounter() {
  if (promptHistoryCounter) {
    promptHistoryCounter.textContent =
      (promptHistoryIndex + 1) + " / " + promptHistory.length;
  }
}

// Enter browse mode: hold the current text as a draft, show the most
// recent prompt, and make the box read-only while browsing.
function enterPromptHistory() {
  if (promptHistory.length === 0 || promptHistoryActive) {
    return;
  }
  promptHistoryActive = true;
  promptHistoryDraft = promptInput.value;
  promptHistoryIndex = 0;
  promptInput.value = promptHistory[0];
  promptInput.readOnly = true;
  if (btnPromptHistory) {
    btnPromptHistory.classList.add("is-active");
  }
  if (promptHistoryNav) {
    promptHistoryNav.hidden = false;
  }
  _setPromptHistoryCounter();
}

// Step through history (delta +1 = older, -1 = newer), wrapping.
function cyclePromptHistory(delta) {
  if (!promptHistoryActive || promptHistory.length === 0) {
    return;
  }
  var n = promptHistory.length;
  promptHistoryIndex =
    (((promptHistoryIndex + delta) % n) + n) % n;
  promptInput.value = promptHistory[promptHistoryIndex];
  _setPromptHistoryCounter();
}

// Reset the browse UI without changing the box text. Shared by commit
// and by starting a generation from a browsed prompt.
function _exitPromptHistoryUI() {
  promptHistoryActive = false;
  promptHistoryDraft = null;
  promptHistoryIndex = -1;
  if (promptInput) {
    promptInput.readOnly = false;
  }
  if (btnPromptHistory) {
    btnPromptHistory.classList.remove("is-active");
  }
  if (promptHistoryNav) {
    promptHistoryNav.hidden = true;
  }
}

function confirmPromptHistory() {
  // Keep the shown prompt; return to normal (editable) input.
  _exitPromptHistoryUI();
  promptInput.focus();
}

function cancelPromptHistory() {
  // Restore the text the user had before browsing.
  if (promptHistoryDraft !== null) {
    promptInput.value = promptHistoryDraft;
  }
  _exitPromptHistoryUI();
}

// ---- Persistent UI settings (localStorage) ----

// Load persisted preferences into appSettings. The schema, key, and
// parsing live in overlays.js (shared with the Settings page); edits
// happen there and are picked up here on the next load (hydrate).
function loadSettings() {
  appSettings = parseSettings(localStorage.getItem(SETTINGS_KEY));
}

// Reflect the applied settings in the status bar.
function updateStatusPrefs() {
  if (statusHighlight) {
    statusHighlight.textContent =
      "Highlighted Tokens: "
      + (appSettings.highlightTokens ? "On" : "Off");
  }
}

// Apply the (saved) settings to the live app: status bar, hover
// highlight, and any active token coloring.
function applySettings() {
  updateStatusPrefs();
  updateHoverHighlight();
  // Toggling the effect starts/stops the Generate idle cycle live.
  updateGenerateIdleEffect();
  // Restart the collapsed device ticker so the GPU-ticker toggle takes
  // effect immediately.
  setModelSelectValue(activeModelId);
  if (scrubberActive) {
    renderFrameWithTokens(currentScrubFrame);
  }
}

// Token-level rendering for scrubber mode.
// Each token is a clickable span; resolved tokens
// can be clicked to toggle remasking.
// Map a still-masked position's predicted confidence to opacity:
// unresolved/0-confidence masks sit at a solid floor and grow more
// opaque (energized) as the model's confidence rises toward the reveal
// range (~0.3+ for low_confidence), then the token resolves into text.
var MASK_OPACITY_FLOOR = 0.35;
var MASK_OPACITY_CAP = 0.4;

function maskOpacity(c) {
  if (typeof c !== "number" || c <= 0) {
    return MASK_OPACITY_FLOOR;
  }
  var frac = Math.min(c / MASK_OPACITY_CAP, 1);
  return MASK_OPACITY_FLOOR + (1 - MASK_OPACITY_FLOOR) * frac;
}

function renderFrameWithTokens(frameIndex) {
  var tokens = frameTokens[frameIndex];
  if (!tokens) {
    renderFrame(frameHistory[frameIndex]);
    return;
  }

  // Diff overlay takes over rendering (two stacked layers).
  if (
    overlayMode === "diff"
    && remaskMode === null
    && diffAvailable()
  ) {
    renderDiffOverlay(frameIndex);
    return;
  }

  outputArea.classList.remove("diff-overlay-mode");
  var allowClick = remaskMode === "edit";
  var fragment =
    document.createDocumentFragment();

  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    var span = document.createElement("span");
    span.setAttribute("data-pos", String(i));

    var tline = tokenLabel(i, tokens.length) + "\n";
    var isUserRemasked =
      remaskedPositions[i] === true;

    if (isUserRemasked) {
      // Orange edit-selection highlight: kept fully opaque so the
      // selection reads clearly (distinct from output masks).
      span.className =
        "token-span token-remasked";
      span.textContent = MASK_CHAR;
      span.title = tline + "Confidence: 0";
    } else if (tok.m) {
      // Output mask opacity tracks the model's live predicted
      // confidence for the position (0/absent -> solid floor).
      span.className = "token-span token-mask";
      span.textContent = MASK_CHAR;
      span.style.opacity = String(maskOpacity(tok.c));
      span.title = tline + "Confidence: " + confLabel(tok.c);
    } else {
      span.className =
        "token-span token-resolved"
        + (allowClick ? " token-clickable" : "");
      span.textContent = tok.t;
      span.title = tline + "Confidence: " + confLabel(tok.c);
      applyTokenColor(span, i, tok);
    }
    fragment.appendChild(span);
  }

  outputArea.textContent = "";
  outputArea.appendChild(fragment);
}

function renderTargetPlaceholder(frameIndex) {
  outputArea.classList.remove("diff-overlay-mode");
  outputArea.textContent = "";

  var editedFrames = [];
  for (
    var ei = 0; ei < remaskModeEdits.length; ei++
  ) {
    editedFrames.push(
      remaskModeEdits[ei].frame_index
    );
  }
  if (editedFrames.length === 0) {
    editedFrames.push(scrubberMinFrame);
  }

  var frameList = "";
  if (editedFrames.length === 1) {
    frameList = "Frame " + editedFrames[0];
  } else if (editedFrames.length === 2) {
    frameList =
      "Frames " + editedFrames[1]
      + " and " + editedFrames[0];
  } else {
    frameList = "Frames ";
    for (
      var fi = editedFrames.length - 1;
      fi >= 0; fi--
    ) {
      if (fi === 0) {
        frameList += "and " + editedFrames[fi];
      } else {
        frameList += editedFrames[fi] + ", ";
      }
    }
  }

  var notice = document.createElement("span");
  notice.className = "preview-notice";

  var label = document.createElement("span");
  label.className = "preview-frame-label";
  label.textContent = "Frame " + frameIndex;

  notice.appendChild(label);
  notice.appendChild(
    document.createTextNode(
      " will be generated. "
      + "Output will diverge from this "
      + "preview based on edits to "
      + frameList + "."
    )
  );
  outputArea.appendChild(notice);

  var origTokens = originalFrameTokens[frameIndex];
  var origText = originalFrameHistory[frameIndex];

  if (origTokens || origText) {
    var wrapper = document.createElement("div");
    wrapper.className = "preview-content";

    if (origTokens) {
      for (var i = 0; i < origTokens.length; i++) {
        var tok = origTokens[i];
        var span = document.createElement("span");
        span.className = tok.m
          ? "token-span token-mask"
          : "token-span token-resolved";
        span.textContent = tok.m ? MASK_CHAR : tok.t;
        wrapper.appendChild(span);
      }
    } else {
      var textSpan = document.createElement("span");
      textSpan.className = "char-resolved";
      textSpan.textContent = origText;
      wrapper.appendChild(textSpan);
    }
    outputArea.appendChild(wrapper);
  }
}

// ---- Scrubber ----

// True when the current run spans more than one canvas.
// DiffusionGemma resume re-enters a single 256-token canvas, so
// multi-canvas runs cannot be resumed in this version; the editing
// UI stays hidden for them.
function runIsMultiCanvas() {
  for (var i = 0; i < frameCanvasIndex.length; i++) {
    if (frameCanvasIndex[i] > 0) {
      return true;
    }
  }
  return false;
}

// Reflect the "already saved an edit" lock on the Edit Frames button:
// greyed out and non-interactive with an explanatory tooltip, until
// the next Generate clears the lock.
function updateEditFramesLock() {
  if (editedRunSaved) {
    btnEditFrames.classList.add("is-locked");
    btnEditFrames.title =
      "This run already has a saved edit."
      + " Generate again to edit a new run.";
  } else {
    btnEditFrames.classList.remove("is-locked");
    btnEditFrames.removeAttribute("title");
  }
}

// Once a run is finalized (an edited run has been saved), the primary
// button becomes "New Run" (clears the canvas for a fresh prompt);
// otherwise it is "Generate". Keeps the same slot/size.
function currentGenerateLabel() {
  return editedRunSaved ? "New Run" : "Generate";
}

function updateGenerateButton() {
  if (editedRunSaved) {
    btnGenerate.classList.add("is-new-run");
    // New Run is client-side; only a completing save should hold it.
    btnGenerate.disabled = isSaving;
  } else {
    btnGenerate.classList.remove("is-new-run");
    btnGenerate.disabled =
      isGenerating
      || isSaving
      || !(modelReady && paramsValid);
  }
  // The label text is owned by the idle-effect controller (it either
  // sets the static label or drives the looping diffusion reveal).
  updateGenerateIdleEffect();
}

// ---- Generate button idle diffusion cycle ----

// One-time discovery nudge: the Generate button always idles with the
// diffusion cycle before the user's first-ever fresh run, then follows
// the "Render diffusion-style text" setting. Persisted per browser.
var GENERATE_TEASED_KEY = "diffusion_generate_teased";
// The button holds its resolved text longer than the status bar so the
// primary CTA reads calmly rather than flickering.
var GENERATE_CYCLE_HOLD_MS = 2000;
var generateCycleTimer = null;
var generateCycleActive = false;
var generateCycleLabel = "";

function generateTeaserActive() {
  try {
    return localStorage.getItem(GENERATE_TEASED_KEY) !== "1";
  } catch (_e) {
    return false;
  }
}

function markGenerateTeased() {
  // Write-through to the server (see persistSet) so the one-time teaser
  // does not replay every restart on a fresh window origin.
  persistSet(GENERATE_TEASED_KEY, "1");
}

// The button idles with the diffusion cycle while it is clickable:
// always before the first fresh run, and thereafter only when the
// effect setting is on. Reduced motion disables it entirely.
function generateIdleActive() {
  if (!btnGenerateLabel || prefersReducedMotion()) {
    return false;
  }
  if (btnGenerate.disabled || isGenerating) {
    return false;
  }
  return generateTeaserActive() || !!appSettings.diffusionText;
}

function startGenerateCycle() {
  if (generateCycleActive) {
    return;
  }
  generateCycleActive = true;
  var runOnce = function () {
    generateCycleLabel = currentGenerateLabel();
    denoiseReveal(
      btnGenerateLabel,
      generateCycleLabel,
      function () {
        generateCycleTimer = setTimeout(
          runOnce, GENERATE_CYCLE_HOLD_MS
        );
      },
      true
    );
  };
  runOnce();
}

function stopGenerateCycle() {
  generateCycleActive = false;
  cancelDenoise(btnGenerateLabel);
  if (generateCycleTimer !== null) {
    clearTimeout(generateCycleTimer);
    generateCycleTimer = null;
  }
  if (btnGenerateLabel) {
    btnGenerateLabel.textContent = currentGenerateLabel();
  }
}

function updateGenerateIdleEffect() {
  if (generateIdleActive()) {
    // Restart if the label changed (e.g. Generate -> New Run) so the
    // cycle animates the correct word without a lag.
    if (
      generateCycleActive
      && generateCycleLabel !== currentGenerateLabel()
    ) {
      stopGenerateCycle();
    }
    startGenerateCycle();
  } else {
    stopGenerateCycle();
  }
}

function activateScrubber() {
  if (frameHistory.length < 2) {
    return;
  }
  scrubberActive = true;
  currentScrubFrame = frameHistory.length - 1;

  scrubberSlider.min = "0";
  scrubberSlider.max =
    String(frameHistory.length - 1);
  scrubberSlider.value =
    String(currentScrubFrame);
  scrubberSlider.disabled = false;
  updateScrubberLabel();

  scrubberSection.hidden = false;
  btnEditFrames.hidden = !(
    activeModel
    && activeModel.capabilities
    && activeModel.capabilities.supports_resume
    && !runIsMultiCanvas()
  );
  updateEditFramesLock();
  overlayMode = "none";
  resetDiffOverlay();
  buildOverlaySelect();
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = false;
  }
  setOverlayDrawerOpen(false);
  updateDiffSummary();
  updateDiffOverlayControls();
  guidedEditControls.hidden = true;
  clearRemaskedPositions();
  unlockScrubberNav();

  navigateToFrame(currentScrubFrame);
}

function deactivateScrubber() {
  scrubberActive = false;
  scrubberSection.hidden = true;
  guidedEditControls.hidden = true;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }
  clearRemaskedPositions();
}

function updateScrubberLabel() {
  var maxLabel = (
    remaskMode === "select_target"
    && originalTotalFrames > 0
  ) ? originalTotalFrames - 1
    : frameHistory.length - 1;
  scrubberLabel.textContent =
    "Frame " + currentScrubFrame
    + " / " + maxLabel;
}

function navigateToFrame(index) {
  saveFrameSelections(currentScrubFrame);

  var minFrame = (
    remaskMode === "select"
    || remaskMode === "select_target"
  ) ? scrubberMinFrame : 0;
  var maxFrame = (
    remaskMode === "select_target"
    && originalTotalFrames > 0
  ) ? originalTotalFrames - 1
    : frameHistory.length - 1;
  index = Math.max(
    minFrame,
    Math.min(index, maxFrame)
  );
  currentScrubFrame = index;
  scrubberSlider.value = String(index);
  updateScrubberLabel();

  restoreFrameSelections(index);

  if (remaskMode === "select_target") {
    renderTargetPlaceholder(index);
  } else if (index < frameHistory.length) {
    renderFrameWithTokens(index);
  } else {
    renderTargetPlaceholder(index);
  }
  updateGuidedUI();
}

function clearRemaskedPositions() {
  remaskedPositions = {};
  perFrameRemasked = {};
  updateGuidedUI();
}

function saveFrameSelections(frameIndex) {
  if (Object.keys(remaskedPositions).length > 0) {
    perFrameRemasked[frameIndex] =
      Object.assign({}, remaskedPositions);
  } else {
    delete perFrameRemasked[frameIndex];
  }
}

function restoreFrameSelections(frameIndex) {
  if (perFrameRemasked[frameIndex]) {
    remaskedPositions = Object.assign(
      {}, perFrameRemasked[frameIndex]
    );
  } else {
    remaskedPositions = {};
  }
}

function countEditedFrames(excludeFrame) {
  var count = 0;
  var keys = Object.keys(perFrameRemasked);
  for (var i = 0; i < keys.length; i++) {
    if (Number(keys[i]) !== excludeFrame) {
      count++;
    }
  }
  return count;
}

function toggleRemaskPosition(pos) {
  if (remaskedPositions[pos]) {
    delete remaskedPositions[pos];
  } else {
    remaskedPositions[pos] = true;
  }
  saveFrameSelections(currentScrubFrame);
  renderFrameWithTokens(currentScrubFrame);
  updateGuidedUI();
}

// ---- Randomize remasks (Edit Frames) ----

// Frame the randomize row was last seeded for, so the target count is
// re-initialized from the selection only when the frame changes (not
// on every re-render, which would fight the user's slider input).
var randomizeInitFrame = null;

function clampInt(value, low, high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

// Frame-array indices of resolved (non-mask) tokens: the candidates
// that can be remasked. Masked positions are never remaskable.
function resolvedPositions(frameIndex) {
  var tokens = frameTokens[frameIndex];
  var out = [];
  if (!tokens) {
    return out;
  }
  for (var i = 0; i < tokens.length; i++) {
    if (tokens[i] && !tokens[i].m) {
      out.push(i);
    }
  }
  return out;
}

// Sync the randomize row to the current edit frame: total resolved
// count, slider/input bounds, and (on a frame change) seed the target
// N from the frame's existing selection.
function updateRandomizeRow() {
  if (!remaskRandomizeRow) {
    return;
  }
  var total = resolvedPositions(currentScrubFrame).length;
  // Remasking 0 tokens is a no-op, so the target floor is 1 (whenever
  // there is at least one resolved token to pick from).
  var floor = total > 0 ? 1 : 0;
  if (randomizeInitFrame !== currentScrubFrame) {
    randomizeInitFrame = currentScrubFrame;
    var selected = Object.keys(remaskedPositions).length;
    remaskRandomSlider.value = String(
      clampInt(selected, floor, total)
    );
  }
  var target = clampInt(
    parseInt(remaskRandomSlider.value, 10) || floor, floor, total
  );
  remaskRandomTotal.textContent = String(total);
  remaskRandomSlider.min = String(floor);
  remaskRandomSlider.max = String(total);
  remaskRandomSlider.value = String(target);
  remaskRandomCount.min = String(floor);
  remaskRandomCount.max = String(total);
  remaskRandomCount.value = String(target);
  var disabled = total === 0;
  remaskRandomSlider.disabled = disabled;
  remaskRandomCount.disabled = disabled;
  btnRemaskShuffle.disabled = disabled;
}

// Replace the current selection with N random resolved positions on
// the current frame (partial Fisher-Yates), then re-render so they
// show as remasked and Lock In can proceed as usual.
function shuffleRemasks() {
  var candidates = resolvedPositions(currentScrubFrame);
  var total = candidates.length;
  if (total === 0) {
    return;
  }
  var n = clampInt(
    parseInt(remaskRandomSlider.value, 10) || 0, 0, total
  );
  for (var i = 0; i < n; i++) {
    var j = i + Math.floor(
      Math.random() * (total - i)
    );
    var swap = candidates[i];
    candidates[i] = candidates[j];
    candidates[j] = swap;
  }
  remaskedPositions = {};
  for (var k = 0; k < n; k++) {
    remaskedPositions[candidates[k]] = true;
  }
  saveFrameSelections(currentScrubFrame);
  renderFrameWithTokens(currentScrubFrame);
  updateGuidedUI();
}

// Cosmetic press feedback: run the diffusion reveal on the Shuffle
// label with a glow that lingers on the way out (the CSS transition
// handles the lag). Gated on the same effect setting as the status bar.
function playShuffleDiffusion() {
  if (!btnRemaskShuffle || !shuffleLabel) {
    return;
  }
  if (!diffusionEffectActive()) {
    return;
  }
  btnRemaskShuffle.classList.add("is-diffusing");
  denoiseReveal(shuffleLabel, "Shuffle", function () {
    btnRemaskShuffle.classList.remove("is-diffusing");
  });
}

// ---- Guided multi-frame edit mode ----

function resetGuidedMode() {
  remaskMode = null;
  guidedResumeAction = null;
  guidedTargetFrame = null;
  remaskModeEdits = [];
  preEditSnapshot = null;
  randomizeInitFrame = null;
  guidedEditControls.hidden = true;
  scrubberSlider.disabled = false;
  scrubberSlider.min = "0";
  unlockScrubberNav();
}

// Snapshot the current complete run before an edit session begins.
function captureEditSnapshot() {
  preEditSnapshot = {
    frameHistory: frameHistory.slice(),
    frameTokens: frameTokens.slice(),
    frameCanvasIndex: frameCanvasIndex.slice(),
    frameMeanConf: frameMeanConf.slice(),
    perFrameElapsed: perFrameElapsed.slice(),
    finalText: lastFinalText,
    remaskEditsLen: remaskEdits.length,
  };
}

// Restore the pre-edit run, discarding any partial/uncommitted
// resumes made during the session (used when the user exits).
function restoreEditSnapshot() {
  if (!preEditSnapshot) {
    return;
  }
  frameHistory = preEditSnapshot.frameHistory.slice();
  frameTokens = preEditSnapshot.frameTokens.slice();
  frameCanvasIndex = preEditSnapshot.frameCanvasIndex.slice();
  frameMeanConf = preEditSnapshot.frameMeanConf.slice();
  perFrameElapsed = preEditSnapshot.perFrameElapsed.slice();
  lastFinalText = preEditSnapshot.finalText;
  // Drop any edits committed during this (now-cancelled) session.
  remaskEdits.length = Math.min(
    remaskEdits.length, preEditSnapshot.remaskEditsLen
  );
  commitSteps = null;
  diffData = null;
  preEditSnapshot = null;
}

function unlockScrubberNav() {
  btnScrubStart.disabled = false;
  btnScrubPrev.disabled = false;
  btnScrubNext.disabled = false;
  btnScrubEnd.disabled = false;
}

function lockScrubberNav() {
  btnScrubStart.disabled = true;
  btnScrubPrev.disabled = true;
  btnScrubNext.disabled = true;
  btnScrubEnd.disabled = true;
  scrubberSlider.disabled = true;
}

// Freeze scrubber navigation and every guided-edit action while a save
// is in flight, so no save path (auto-save, Confirm, or the standalone
// Save button during review) leaves interactive controls that could
// race the snapshot. The subsequent updateGuidedUI() re-derives each
// button's per-phase state once the save settles.
function setSavingControls(saving) {
  var disabled = !!saving;
  scrubberSlider.disabled = disabled;
  btnScrubStart.disabled = disabled;
  btnScrubPrev.disabled = disabled;
  btnScrubNext.disabled = disabled;
  btnScrubEnd.disabled = disabled;
  btnSelectFrame.disabled = disabled;
  btnEditFrames.disabled = disabled;
  // Guided-edit action buttons (visible only mid edit session) freeze
  // too, so the dimmed slider matches the Confirm-checkmark behavior.
  btnLockIn.disabled = disabled;
  btnClearGuided.disabled = disabled;
  btnEditAnother.disabled = disabled;
  btnRunToHere.disabled = disabled;
  btnResumeEnd.disabled = disabled;
  btnConfirmEdit.disabled = disabled;
  btnRetryEdit.disabled = disabled;
  btnExitEdit.disabled = disabled;
  // Dim the whole scrubber row and surface a tooltip on hover. The
  // title lives on the (non-disabled) container because native
  // tooltips do not fire on disabled controls.
  if (scrubberControls) {
    scrubberControls.classList.toggle("is-saving", disabled);
    if (disabled) {
      scrubberControls.title = "Saving in progress\u2026";
    } else {
      scrubberControls.removeAttribute("title");
    }
  }
  // Reflect the saving state on the primary button too (so Generate is
  // greyed out during a save, then becomes New Run once finalized).
  updateGenerateButton();
}

// Edit Frames entry point. The current run is the "original": if it
// has not been saved yet, save it now so an unsaved original is never
// lost once the edited run is saved. Editing an original implies you
// want to keep it, so this makes the save implicit.
function enterRemaskMode() {
  // Gated once an edited run has been saved for this generation.
  if (btnEditFrames.classList.contains("is-locked")) {
    return;
  }
  // Enter edit mode first, then auto-save the original: saving last
  // means its control-freeze (setSavingControls) is not immediately
  // undone by the edit-session UI setup.
  beginEditSession();
  if (!runSaved) {
    saveRun();
  }
}

// Start a fresh edit session on the current run (no save). Shared by
// Edit Frames (after its auto-save) and Retry (which reuses the
// already-saved original), keeping the save decoupled from re-entry.
function beginEditSession() {
  captureEditSnapshot();
  remaskMode = "select";
  // Start at frame 1: frame 0 is the fully-masked canvas with nothing
  // to remask, so it is never a useful selection. (Guarded for the
  // degenerate single-frame case.)
  var startFrame = frameHistory.length > 1 ? 1 : 0;
  scrubberMinFrame = startFrame;
  remaskModeEdits = [];
  guidedResumeAction = null;
  clearRemaskedPositions();

  scrubberSlider.min = String(startFrame);
  scrubberSlider.max = String(frameHistory.length - 1);
  btnEditFrames.hidden = true;
  guidedEditControls.hidden = false;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }

  navigateToFrame(startFrame);
  updateGuidedUI();
}

// Leaving edit mode cancels the session: any partial resumes made
// during it are discarded by restoring the pre-edit run, then
// activateScrubber returns to the clean scrubber state (overlay
// drawer + Edit Frames shown, guided controls hidden, final frame).
function exitRemaskMode() {
  restoreEditSnapshot();
  resetGuidedMode();
  activateScrubber();
}

// DiffusionGemma resumes by renoising remasked positions rather than
// hard-masking them (as LLaDA does), so committed neighbours may also
// shift on resume. Surface that difference while editing.
function renoiseNote() {
  if (activeModelId === "diffusiongemma") {
    return " Remasked tokens are renoised, so nearby"
      + " tokens may also change on resume.";
  }
  return "";
}

function updateGuidedUI() {
  // The diff-opacity row shares the scrubber area with the guided
  // controls, so keep it hidden whenever a run is being edited
  // (remaskMode !== null); updateDiffOverlayControls restores it on
  // exit once remaskMode is null again.
  updateDiffOverlayControls();

  // Reset every phase button first so no stale state can survive a
  // transition (including the exit back to remaskMode === null). Only
  // the buttons relevant to the current phase are then revealed; the
  // status text sits on the left (flex:1) and the action cluster is
  // right-anchored, so the text never shifts as buttons change.
  btnSelectFrame.hidden = true;
  btnSelectFrame.disabled = false;
  btnLockIn.hidden = true;
  btnClearGuided.hidden = true;
  btnEditAnother.hidden = true;
  btnRunToHere.hidden = true;
  btnResumeEnd.hidden = true;
  btnConfirmEdit.hidden = true;
  btnRetryEdit.hidden = true;
  if (remaskRandomizeRow) {
    remaskRandomizeRow.hidden = true;
  }

  if (remaskMode === null) {
    guidedEditControls.hidden = true;
    return;
  }

  guidedEditControls.hidden = false;
  // The guided flow owns saving during an edit session (auto-save on
  // entry, then Confirm/Retry). Disable the standalone Save so it can
  // never race or double-fire with the review-step Confirm.
  btnSave.disabled = true;

  var count =
    Object.keys(remaskedPositions).length;
  var plural = count !== 1 ? "s" : "";

  switch (remaskMode) {
    case "select":
      guidedEditStatus.textContent =
        "Navigate to a frame, then select it"
        + " for editing.";
      btnSelectFrame.hidden = false;
      scrubberSlider.disabled = false;
      scrubberSlider.min =
        String(scrubberMinFrame);
      unlockScrubberNav();
      break;

    case "edit":
      guidedEditStatus.textContent =
        "Frame " + currentScrubFrame
        + ": click tokens to remask ("
        + count + " selected)." + renoiseNote();
      btnLockIn.hidden = false;
      btnLockIn.disabled = count === 0;
      btnClearGuided.hidden = false;
      btnClearGuided.disabled = count === 0;
      if (remaskRandomizeRow) {
        remaskRandomizeRow.hidden = false;
        updateRandomizeRow();
      }
      lockScrubberNav();
      break;

    case "choice":
      guidedEditStatus.textContent =
        count + " token" + plural
        + " locked on Frame "
        + currentScrubFrame + ".";
      btnEditAnother.hidden = false;
      btnResumeEnd.hidden = false;
      lockScrubberNav();
      break;

    case "select_target":
      guidedEditStatus.textContent =
        "Navigate to the target frame,"
        + " then run to it.";
      btnRunToHere.hidden = false;
      scrubberSlider.disabled = false;
      scrubberSlider.min =
        String(scrubberMinFrame);
      scrubberSlider.max = String(
        (originalTotalFrames > 0)
          ? originalTotalFrames - 1
          : frameHistory.length - 1
      );
      unlockScrubberNav();
      break;

    case "generating":
      guidedEditStatus.textContent =
        "Generating\u2026";
      lockScrubberNav();
      break;

    case "review":
      scrubberSlider.disabled = false;
      scrubberSlider.min = "0";
      scrubberSlider.max =
        String(frameHistory.length - 1);
      unlockScrubberNav();
      if (currentScrubFrame === frameHistory.length - 1) {
        guidedEditStatus.textContent =
          "Edit complete. Confirm to save, or"
          + " retry from the start.";
        btnConfirmEdit.hidden = false;
        btnRetryEdit.hidden = false;
      } else {
        guidedEditStatus.textContent =
          "Reviewing edited run. Return to the last"
          + " frame to confirm or retry.";
      }
      break;
  }

  // A save in flight (e.g. the Edit Frames auto-save) overrides the
  // per-mode state: freeze navigation until it completes.
  if (isSaving) {
    lockScrubberNav();
    btnSelectFrame.disabled = true;
  }
}

function selectCurrentFrame() {
  remaskMode = "edit";
  renderFrameWithTokens(currentScrubFrame);
  updateGuidedUI();
}

function lockInEdits() {
  var positions =
    Object.keys(remaskedPositions).map(Number);
  if (positions.length === 0) {
    return;
  }
  remaskModeEdits.push({
    frame_index: currentScrubFrame,
    token_positions: positions.slice(),
  });
  remaskMode = "choice";
  updateGuidedUI();
}

function doGuidedResume(action) {
  // Guard against a stale click with no locked edits (should be
  // unreachable now that the buttons hide correctly).
  if (remaskModeEdits.length === 0) {
    return;
  }
  guidedResumeAction = action;

  var lastEdit =
    remaskModeEdits[remaskModeEdits.length - 1];
  var positions = lastEdit.token_positions;
  var frameIndex = lastEdit.frame_index;

  remaskEdits.push({
    frame_index: frameIndex,
    token_positions: positions.slice(),
  });

  perFrameRemasked = {};
  remaskedPositions = {};

  resumeFrameOffset = frameIndex;
  frameHistory.length = resumeFrameOffset;
  frameTokens.length = resumeFrameOffset;
  frameCanvasIndex.length = resumeFrameOffset;
  frameMeanConf.length = resumeFrameOffset;
  commitSteps = null;
  diffData = null;
  isResuming = true;

  remaskMode = "generating";
  updateGuidedUI();

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  startStatusDots("Resuming");

  var message = {
    type: "resume",
    frame_index: frameIndex,
    remask_positions: positions,
  };

  if (
    action === "another"
    && guidedTargetFrame !== null
  ) {
    message.max_frames =
      guidedTargetFrame - frameIndex + 1;
  }

  ws.send(JSON.stringify(message));
}

function handleGuidedDone() {
  if (guidedResumeAction === "another") {
    var target = Math.min(
      guidedTargetFrame,
      frameHistory.length - 1
    );

    scrubberActive = true;
    scrubberSection.hidden = false;
    guidedEditControls.hidden = false;
    btnEditFrames.hidden = true;
    if (overlaySelectGroup) {
      overlaySelectGroup.hidden = true;
    }

    scrubberSlider.min = String(target);
    scrubberSlider.max =
      String(frameHistory.length - 1);
    scrubberSlider.value = String(target);

    currentScrubFrame = target;
    remaskMode = "edit";
    guidedResumeAction = null;
    guidedTargetFrame = null;
    remaskedPositions = {};
    perFrameRemasked = {};

    updateScrubberLabel();
    renderFrameWithTokens(target);
    updateGuidedUI();
  } else {
    enterReviewMode();
  }
}

// Resume-to-End finished: rather than dropping straight back to the
// plain scrubber, stay in guided editing at the final frame so the
// user must explicitly Confirm (save) or Retry (redo). Navigation
// stays enabled so the result can be inspected; only the final frame
// exposes the Confirm/Retry actions.
function enterReviewMode() {
  guidedResumeAction = null;
  guidedTargetFrame = null;
  remaskedPositions = {};
  perFrameRemasked = {};
  remaskMode = "review";
  scrubberActive = true;
  scrubberSection.hidden = false;
  guidedEditControls.hidden = false;
  btnEditFrames.hidden = true;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }
  currentScrubFrame = frameHistory.length - 1;
  scrubberSlider.min = "0";
  scrubberSlider.max = String(frameHistory.length - 1);
  scrubberSlider.value = String(currentScrubFrame);
  scrubberSlider.disabled = false;
  unlockScrubberNav();
  updateScrubberLabel();
  renderFrameWithTokens(currentScrubFrame);
  updateGuidedUI();
}

// Confirm the reviewed edit: trigger a save (as the Save button
// would), then leave guided editing. The save-success handler locks
// Edit Frames so the run cannot accrue a second, conflicting edit.
function confirmGuidedEdit() {
  saveRun();
  resetGuidedMode();
  activateScrubber();
}

// Retry: discard this session's edits and restart editing from frame
// 0. Reuses the already-saved original, so (unlike Edit Frames) it
// does not trigger another save.
function retryGuidedEdit() {
  restoreEditSnapshot();
  resetGuidedMode();
  beginEditSession();
}

// ---- UI state helpers ----

function setBadge(state) {
  connectionBadge.className =
    "badge badge-" + state;
  connectionBadge.textContent = state;
}

function setGenerating(active) {
  isGenerating = active;
  // Generate stays visible; it just greys out while the model runs
  // (and whenever the model is not ready, params are invalid, or a
  // save is completing) -- centralized in updateGenerateButton.
  updateGenerateButton();
  promptInput.disabled = active;
  toggleExperimental.disabled = active;
  setModelSelectDisabled(active);
  var keys = Object.keys(paramInputs);
  for (var i = 0; i < keys.length; i++) {
    paramInputs[keys[i]].disabled = active;
  }

  if (active) {
    deactivateScrubber();
  }
}

// Animated "<base>..." status where only the trailing dots cycle
// (3 -> 0 -> 1 -> 2 -> 3 ...), padded with non-breaking spaces so
// the base word stays put in the right-aligned status slot.
var statusDotsTimer = null;
var statusDotsCount = 3;
var statusCycleTimer = null;

// Block glyphs for the optional "diffusion-style text" reveal.
var DENOISE_GLYPHS = "\u2591\u2592\u2593";

function prefersReducedMotion() {
  try {
    return window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
  } catch (_e) {
    return false;
  }
}

// True when the diffusion-text effect should actually animate.
function diffusionEffectActive() {
  return !!appSettings.diffusionText && !prefersReducedMotion();
}

// Per-element reveal timer, stored on the element so independent
// targets (status bar, Shuffle label) can animate simultaneously
// without one cancelling the other.
function cancelDenoise(el) {
  if (el && el._denoiseTimer) {
    clearInterval(el._denoiseTimer);
    el._denoiseTimer = null;
  }
}

// Reveal `text` in `el` with a brief scramble-to-resolve ("denoising")
// pass: characters lock in left-to-right while the rest keep flickering
// through block glyphs, then `onDone` runs. Instant when the effect is
// off or the OS prefers reduced motion (an accessibility escape hatch).
// `force` animates regardless of the setting (still honoring reduced
// motion) for the one-time Generate teaser.
function denoiseReveal(el, text, onDone, force) {
  cancelDenoise(el);
  var active = force
    ? !prefersReducedMotion()
    : diffusionEffectActive();
  if (!active || text.length === 0) {
    el.textContent = text;
    if (onDone) {
      onDone();
    }
    return;
  }
  var steps_total = 8;
  var step = 0;
  var render = function () {
    var revealed = Math.floor(
      (step / steps_total) * text.length
    );
    var out = "";
    for (var i = 0; i < text.length; i++) {
      if (i < revealed || text[i] === " ") {
        out += text[i];
      } else {
        out += DENOISE_GLYPHS[
          Math.floor(Math.random() * DENOISE_GLYPHS.length)
        ];
      }
    }
    el.textContent = out;
    step += 1;
    if (step > steps_total) {
      cancelDenoise(el);
      el.textContent = text;
      if (onDone) {
        onDone();
      }
    }
  };
  render();
  el._denoiseTimer = setInterval(render, 45);
}

// Reverse of denoiseReveal: dissolve `el`'s current text into solid
// mask glyphs (0-confidence "░") left-to-right, then run `onDone`.
// Code-point safe so the lock emoji collapses as one glyph. Instant
// when the effect is off or reduced motion is preferred.
function denoiseDissolve(el, onDone) {
  var chars = Array.from(el.textContent);
  cancelDenoise(el);
  if (!diffusionEffectActive() || chars.length === 0) {
    if (onDone) {
      onDone();
    }
    return;
  }
  var steps_total = 8;
  var step = 1;
  var render = function () {
    var masked = Math.ceil(
      (step / steps_total) * chars.length
    );
    var out = "";
    for (var i = 0; i < chars.length; i++) {
      if (chars[i] === " ") {
        out += " ";
      } else if (i < masked) {
        out += MASK_CHAR;
      } else {
        out += chars[i];
      }
    }
    el.textContent = out;
    step += 1;
    if (step > steps_total) {
      cancelDenoise(el);
      if (onDone) {
        onDone();
      }
    }
  };
  render();
  el._denoiseTimer = setInterval(render, 40);
}

// Cycle mode: re-diffuse the base word, hold the resolved text briefly,
// then diffuse again, looping until stopStatusDots. The re-diffusion is
// the activity indicator, so the trailing dots are omitted here.
var STATUS_CYCLE_HOLD_MS = 700;

function startStatusCycle(base) {
  var runOnce = function () {
    denoiseReveal(statusMessage, base, function () {
      statusCycleTimer = setTimeout(runOnce, STATUS_CYCLE_HOLD_MS);
    });
  };
  runOnce();
}

function startStatusDots(base) {
  stopStatusDots();
  statusDotsCount = 3;
  statusMessage.style.color = "";

  if (
    diffusionEffectActive()
    && appSettings.diffusionTextMode === "cycle"
  ) {
    startStatusCycle(base);
    return;
  }

  // Default: reveal the base word once (denoise effect when enabled),
  // then cycle the trailing dots on the resolved text.
  denoiseReveal(statusMessage, base, function () {
    var render = function () {
      var dots = ".".repeat(statusDotsCount);
      var pad = "\u00A0".repeat(3 - statusDotsCount);
      statusMessage.textContent = base + dots + pad;
      statusDotsCount = (statusDotsCount + 1) % 4;
    };
    render();
    statusDotsTimer = setInterval(render, 400);
  });
}

function stopStatusDots() {
  cancelDenoise(statusMessage);
  if (statusDotsTimer !== null) {
    clearInterval(statusDotsTimer);
    statusDotsTimer = null;
  }
  if (statusCycleTimer !== null) {
    clearTimeout(statusCycleTimer);
    statusCycleTimer = null;
  }
}

function setSaveAvailable(available) {
  // Always visible; greyed out when there is nothing to save.
  btnSave.disabled = !(available && frameHistory.length > 0);
}

function resetStatus() {
  statusStep.textContent =
    "Step -/-";
  statusElapsed.textContent =
    "Elapsed: -";
  statusMessage.textContent = "";
  statusMessage.style.color = "";
}

// ---- Actions ----

// Clear all live-run state (frames, edits, overlays, gates) back to a
// pre-run baseline. Shared by Generate (fresh run) and New Run.
function resetRunState() {
  resetGuidedMode();
  remaskedPositions = {};
  perFrameRemasked = {};
  frameHistory = [];
  frameTokens = [];
  perFrameElapsed = [];
  frameCanvasIndex = [];
  frameMeanConf = [];
  commitSteps = null;
  diffData = null;
  overlayMode = "none";
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }
  lastRunParams = null;
  lastFinalText = null;
  originalTotalFrames = 0;
  originalFrameHistory = [];
  originalFrameTokens = [];
  remaskEdits = [];
  editedRunSaved = false;
  runSaved = false;
  lastSavedRunId = null;
  isResuming = false;
  resumeFrameOffset = 0;
  updateEditFramesLock();
  updateGenerateButton();
  setSaveAvailable(false);
}

// "New Run": reset to a clean slate for a new prompt once a run is
// finalized (Generate has become "New Run"). Clears the canvas and the
// prompt box (revealing its placeholder), but keeps prompt history.
function startNewRun() {
  _exitPromptHistoryUI();
  resetRunState();
  deactivateScrubber();
  promptInput.value = "";
  if (thinkingPanel) {
    thinkingPanel.hidden = true;
  }
  clearSessionState();
  resetStatus();
  setGenerating(false);
  // Return to the pre-generation resting state.
  showOutputPlaceholder();
  promptInput.focus();
}

function startGeneration() {
  if (
    !ws
    || ws.readyState !== WebSocket.OPEN
  ) {
    return;
  }
  if (isGenerating) {
    return;
  }
  if (!paramsValid) {
    return;
  }

  var prompt = promptInput.value.trim();
  if (!prompt) {
    statusMessage.textContent =
      "Prompt is empty.";
    return;
  }

  // The first fresh run retires the Generate teaser: from now on the
  // idle diffusion cycle follows the setting.
  markGenerateTeased();

  // A fresh run abandons any in-progress edit session and clears the
  // previous run's state. Record the prompt in history first.
  _exitPromptHistoryUI();
  pushPromptHistory(prompt);
  resetRunState();

  outputArea.textContent = "";
  if (thinkingPanel) {
    thinkingPanel.hidden = true;
  }
  clearSessionState();
  resetStatus();
  setGenerating(true);
  startStatusDots("Running");

  var payload = getParamValues();
  payload.type = "generate";
  payload.prompt = prompt;
  payload.experimental = toggleExperimental.checked;
  ws.send(JSON.stringify(payload));
}

// Project in-memory frame token objects into the persisted record
// shape {t, m, id, c?}. Confidence is included only when present
// (masked positions carry none), mirroring the live protocol so a
// saved run can drive the durable analytics overlays.
// Return a clean List[int] of length frameCount, or null to omit it.
// The server's canvas_index is List[int] and rejects null entries, so a
// sparse/misaligned array (which can arise from a resumed run) must be
// dropped rather than sent.
function cleanCanvasIndex(arr, frameCount) {
  if (!arr || arr.length !== frameCount) {
    return null;
  }
  for (var i = 0; i < arr.length; i++) {
    if (typeof arr[i] !== "number" || !isFinite(arr[i])) {
      return null;
    }
  }
  return arr.slice();
}

function tokenRecordsFrom(frames) {
  var out = [];
  for (var fi = 0; fi < frames.length; fi++) {
    var ft = frames[fi];
    if (!ft) {
      out.push(null);
      continue;
    }
    var records = [];
    for (var ti = 0; ti < ft.length; ti++) {
      var tok = ft[ti];
      var record = { t: tok.t, m: !!tok.m, id: tok.id };
      if (typeof tok.c === "number") {
        record.c = tok.c;
      }
      records.push(record);
    }
    out.push(records);
  }
  return out;
}

// ---- "New run saved" Analytics cue ----

// The header badge shows how many saved runs have not yet been opened
// in Analytics (the shared set lives in overlays.js). Cleared per run
// when its detail is opened there, so the count stays in sync.
function refreshAnalyticsCue() {
  if (!analyticsNewDot) {
    return;
  }
  var count = overlaysNewRunCount();
  if (count > 0) {
    analyticsNewDot.textContent = String(count);
    analyticsNewDot.hidden = false;
  } else {
    analyticsNewDot.textContent = "";
    analyticsNewDot.hidden = true;
  }
}

// One-shot "+1" that rises and fades above the Analytics link.
function flashAnalyticsPlusOne() {
  if (!linkAnalytics || prefersReducedMotion()) {
    return;
  }
  var plus = document.createElement("span");
  plus.className = "analytics-plus-one";
  plus.textContent = "+1";
  plus.setAttribute("aria-hidden", "true");
  linkAnalytics.appendChild(plus);
  plus.addEventListener("animationend", function () {
    plus.remove();
  });
  // Fallback removal if animationend never fires.
  setTimeout(function () {
    if (plus.parentNode) {
      plus.remove();
    }
  }, 1500);
}

// Register a freshly saved run as "new" and light up the header badge.
// The "+1" flashes only when the run is genuinely new (not an in-place
// update of a run already counted), so editing-and-resaving a run does
// not double-count it.
function showAnalyticsCue(runId) {
  var added = overlaysAddNewRun(runId);
  refreshAnalyticsCue();
  if (added) {
    flashAnalyticsPlusOne();
  }
}

function saveRun() {
  if (isSaving) {
    return;
  }
  if (
    frameHistory.length === 0
    || !lastFinalText
  ) {
    return;
  }

  isSaving = true;
  btnSave.disabled = true;
  setSavingControls(true);
  if (saveCheckTimer !== null) {
    clearTimeout(saveCheckTimer);
    saveCheckTimer = null;
  }
  btnSave.classList.remove("is-saved");
  btnSave.classList.add("is-saving");
  startStatusDots("Saving run");

  var totalElapsed = perFrameElapsed.length > 0
    ? perFrameElapsed[perFrameElapsed.length - 1]
    : null;

  // Captured now so the async success handler locks Edit Frames only
  // when the saved run actually carried edits.
  var wasEdited = remaskEdits.length > 0;

  var payload = {
    model: activeModelId,
    prompt: promptInput.value.trim(),
    params: lastRunParams || getParamValues(),
    frames: frameHistory,
    final_text: lastFinalText,
    elapsed_seconds: totalElapsed,
    per_frame_elapsed: perFrameElapsed.slice(),
    frame_tokens: tokenRecordsFrom(frameTokens),
    mean_conf: frameMeanConf.slice(),
  };

  // canvas_index must be a clean List[int] matching the frame count.
  // If it is sparse or misaligned (e.g. a resumed run whose pre-resume
  // indices were not restored), omit it rather than send nulls that
  // would fail server validation and break the whole save.
  var canvasIndexClean = cleanCanvasIndex(
    frameCanvasIndex, frameHistory.length
  );
  if (canvasIndexClean !== null) {
    payload.canvas_index = canvasIndexClean;
  }

  if (remaskEdits.length > 0) {
    payload.remask_edits = remaskEdits;
    // Persist the pre-edit snapshot so the counterfactual diff is
    // reviewable post-hoc (only meaningful for edited runs).
    if (originalFrameTokens.length > 0) {
      payload.original_frame_tokens =
        tokenRecordsFrom(originalFrameTokens);
    }
    // Update the pre-edit run's folder in place so the bundled edited
    // run replaces its original (one Analytics row, not two).
    if (lastSavedRunId) {
      payload.run_id = lastSavedRunId;
    }
  }

  fetch("/api/save", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (result) {
      isSaving = false;
      btnSave.classList.remove("is-saving");
      setSavingControls(false);
      updateGuidedUI();
      stopStatusDots();
      if (result.success) {
        // Flash a glowing check for half a second, then revert to
        // the (disabled) arrow. It stays disabled to prevent a
        // duplicate save; re-enables on the next run.
        btnSave.classList.add("is-saved");
        saveCheckTimer = setTimeout(function () {
          btnSave.classList.remove("is-saved");
          saveCheckTimer = null;
        }, 500);
        runSaved = true;
        if (wasEdited) {
          editedRunSaved = true;
        }
        updateEditFramesLock();
        updateGenerateButton();
        // Point the user to where the saved run now lives (the run id
        // is the folder name at the end of the returned path). Remember
        // it so a later edited save updates this folder in place.
        var savedParts = String(result.path || "").split("/");
        lastSavedRunId = savedParts[savedParts.length - 1] || null;
        showAnalyticsCue(lastSavedRunId || "");
        statusMessage.textContent =
          "Saved to " + result.path;
        statusMessage.style.color =
          "var(--accent)";
        // Persist LAST, so the session captures the final run id and
        // "Saved to..." status (not the transient "Saving run..." text
        // or a stale run id) and survives a round-trip to Analytics.
        saveSessionState();
      } else {
        btnSave.disabled = false;
        statusMessage.textContent =
          "Save failed: "
          + (result.message || "unknown");
        statusMessage.style.color =
          "var(--danger)";
      }
    })
    .catch(function (error) {
      isSaving = false;
      btnSave.classList.remove("is-saving", "is-saved");
      setSavingControls(false);
      updateGuidedUI();
      stopStatusDots();
      btnSave.disabled = false;
      statusMessage.textContent =
        "Save failed: " + error.message;
      statusMessage.style.color =
        "var(--danger)";
    });
}

// ---- Event listeners ----

btnGenerate.addEventListener(
  "click",
  function () {
    // The primary button is "New Run" once a run is finalized.
    if (editedRunSaved) {
      startNewRun();
    } else {
      startGeneration();
    }
  }
);
btnSave.addEventListener("click", saveRun);

promptInput.addEventListener(
  "keydown",
  function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      // Enter runs a generation; in the finalized "New Run" state it
      // is a no-op so it can't wipe the canvas unexpectedly.
      if (!editedRunSaved) {
        startGeneration();
      }
    }
  }
);

// Prompt history controls.
if (btnPromptHistory) {
  btnPromptHistory.addEventListener("click", function () {
    if (promptHistoryActive) {
      cancelPromptHistory();
    } else {
      enterPromptHistory();
    }
  });
}
if (btnHistPrev) {
  btnHistPrev.addEventListener("click", function () {
    cyclePromptHistory(1);
  });
}
if (btnHistNext) {
  btnHistNext.addEventListener("click", function () {
    cyclePromptHistory(-1);
  });
}
if (btnHistConfirm) {
  btnHistConfirm.addEventListener(
    "click", confirmPromptHistory
  );
}
if (btnHistCancel) {
  btnHistCancel.addEventListener(
    "click", cancelPromptHistory
  );
}

toggleExperimental.addEventListener(
  "change", applyLimits
);

if (modelSelect && modelSelectList) {
  modelSelect.addEventListener("click", function (e) {
    if (e.target.closest(".model-select-option")) {
      return;
    }
    if (modelSelectDisabled) {
      return;
    }
    toggleModelList();
  });
  modelSelectList.addEventListener("click", function (e) {
    // Per-model device buttons handle their own clicks
    // (stopPropagation); a click on the row name targets the
    // model's default device. Both route through the confirm.
    var opt = e.target.closest(".model-select-option");
    if (!opt) {
      return;
    }
    var id = opt.getAttribute("data-id");
    // The loaded model is inert here: re-selecting it is redundant, and
    // any device change goes through its (still enabled) other-device
    // button, so a name-area click on the active row does nothing.
    if (!id || id === activeModelId) {
      return;
    }
    requestSwitch(id, defaultDeviceFor(models[id]));
  });
  modelSelect.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleModelList();
    } else if (e.key === "Escape") {
      closeModelList();
    }
  });
  document.addEventListener("click", function (e) {
    if (!modelSelect.contains(e.target)) {
      closeModelList();
      closeSwitchConfirm();
    }
  });
}

// Scrubber event listeners.
scrubberSlider.addEventListener(
  "input",
  function () {
    saveFrameSelections(currentScrubFrame);
    var val = parseInt(scrubberSlider.value, 10);
    currentScrubFrame = val;
    updateScrubberLabel();
    restoreFrameSelections(val);
    if (remaskMode === "select_target") {
      renderTargetPlaceholder(val);
    } else if (val < frameHistory.length) {
      renderFrameWithTokens(val);
    } else {
      renderTargetPlaceholder(val);
    }
    updateGuidedUI();
  }
);

btnScrubStart.addEventListener(
  "click",
  function () {
    navigateToFrame(0);
  }
);

btnScrubPrev.addEventListener(
  "click",
  function () {
    navigateToFrame(currentScrubFrame - 1);
  }
);

btnScrubNext.addEventListener(
  "click",
  function () {
    navigateToFrame(currentScrubFrame + 1);
  }
);

btnScrubEnd.addEventListener(
  "click",
  function () {
    var endFrame = (
      remaskMode === "select_target"
      && originalTotalFrames > 0
    ) ? originalTotalFrames - 1
      : frameHistory.length - 1;
    navigateToFrame(endFrame);
  }
);

if (overlayDrawerHandle) {
  overlayDrawerHandle.addEventListener("click", function () {
    var open = !overlaySelectGroup.classList.contains("open");
    setOverlayDrawerOpen(open);
  });
}

// Diff-overlay opacity sliders update the live layers directly;
// the blend toggle re-renders (it changes the original layer's
// coloring as well as the blend mode).
if (diffOriginalSlider) {
  diffOriginalSlider.addEventListener("input", function () {
    diffOriginalOpacity = parseInt(
      diffOriginalSlider.value, 10
    );
    var layer = outputArea.querySelector(
      ".diff-layer-original"
    );
    if (layer) {
      layer.style.opacity = String(diffOriginalOpacity / 100);
    }
  });
}

if (diffEditedSlider) {
  diffEditedSlider.addEventListener("input", function () {
    diffEditedOpacity = parseInt(
      diffEditedSlider.value, 10
    );
    var layer = outputArea.querySelector(
      ".diff-layer-edited"
    );
    if (layer) {
      layer.style.opacity = String(diffEditedOpacity / 100);
    }
  });
}

if (diffBlendToggle) {
  diffBlendToggle.addEventListener("change", function () {
    diffBlend = diffBlendToggle.checked;
    if (scrubberActive && overlayMode === "diff") {
      renderFrameWithTokens(currentScrubFrame);
    }
  });
}

// Guided edit mode event listeners.
btnEditFrames.addEventListener(
  "click", enterRemaskMode
);

btnSelectFrame.addEventListener(
  "click", selectCurrentFrame
);

btnLockIn.addEventListener("click", function () {
  if (!diffusionEffectActive()) {
    lockInEdits();
    return;
  }
  // Dissolve the label (letters + lock emoji) into 0-confidence mask
  // glyphs, then commit (which hides the button). Restore the label
  // afterward so it reads correctly the next time it appears.
  var label = btnLockIn.textContent;
  denoiseDissolve(btnLockIn, function () {
    lockInEdits();
    btnLockIn.textContent = label;
  });
});

btnClearGuided.addEventListener(
  "click",
  function () {
    remaskedPositions = {};
    delete perFrameRemasked[currentScrubFrame];
    renderFrameWithTokens(currentScrubFrame);
    updateGuidedUI();
  }
);

// Randomize-remask controls: the slider and number input mirror one
// target count; Shuffle applies it. They only set the target, so they
// never render until Shuffle is pressed.
if (remaskRandomSlider) {
  remaskRandomSlider.addEventListener("input", function () {
    remaskRandomCount.value = remaskRandomSlider.value;
  });
}
if (remaskRandomCount) {
  remaskRandomCount.addEventListener("input", function () {
    var total = resolvedPositions(currentScrubFrame).length;
    var floor = total > 0 ? 1 : 0;
    var n = clampInt(
      parseInt(remaskRandomCount.value, 10) || floor, floor, total
    );
    remaskRandomCount.value = String(n);
    remaskRandomSlider.value = String(n);
  });
}
if (btnRemaskShuffle) {
  btnRemaskShuffle.addEventListener("click", function () {
    shuffleRemasks();
    playShuffleDiffusion();
  });
}

btnEditAnother.addEventListener(
  "click",
  function () {
    if (remaskModeEdits.length === 0) {
      return;
    }
    var lastEdit = remaskModeEdits[
      remaskModeEdits.length - 1
    ];
    scrubberMinFrame =
      lastEdit.frame_index + 1;
    remaskMode = "select_target";
    var maxFrame = (originalTotalFrames > 0)
      ? originalTotalFrames - 1
      : frameHistory.length - 1;
    scrubberSlider.min =
      String(scrubberMinFrame);
    scrubberSlider.max = String(maxFrame);
    scrubberSlider.disabled = false;
    unlockScrubberNav();
    navigateToFrame(scrubberMinFrame);
    updateGuidedUI();
  }
);

btnRunToHere.addEventListener(
  "click",
  function () {
    guidedTargetFrame = currentScrubFrame;
    doGuidedResume("another");
  }
);

btnResumeEnd.addEventListener(
  "click",
  function () {
    doGuidedResume("end");
  }
);

btnConfirmEdit.addEventListener(
  "click", confirmGuidedEdit
);

btnRetryEdit.addEventListener(
  "click", retryGuidedEdit
);

btnExitEdit.addEventListener(
  "click", exitRemaskMode
);

// Token click delegation on the output area.
outputArea.addEventListener(
  "click",
  function (e) {
    if (!scrubberActive) {
      return;
    }
    if (remaskMode !== "edit") {
      return;
    }
    var target = e.target;
    if (
      !target.classList.contains("token-clickable")
      && !target.classList.contains("token-remasked")
    ) {
      return;
    }
    var pos = target.getAttribute("data-pos");
    if (pos === null) {
      return;
    }
    toggleRemaskPosition(parseInt(pos, 10));
  }
);

// Keyboard shortcuts for scrubber navigation.
document.addEventListener(
  "keydown",
  function (e) {
    if (!scrubberActive || isGenerating || isSaving) {
      return;
    }
    if (
      remaskMode === "edit"
      || remaskMode === "choice"
      || remaskMode === "generating"
    ) {
      return;
    }
    // "select" and "select_target" allow navigation.
    var tag = document.activeElement.tagName;
    if (
      tag === "INPUT"
      || tag === "TEXTAREA"
      || tag === "SELECT"
    ) {
      return;
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      navigateToFrame(currentScrubFrame - 1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      navigateToFrame(currentScrubFrame + 1);
    } else if (e.key === "Home") {
      e.preventDefault();
      navigateToFrame(0);
    } else if (e.key === "End") {
      e.preventDefault();
      var endFrame = (
        remaskMode === "select_target"
        && originalTotalFrames > 0
      ) ? originalTotalFrames - 1
        : frameHistory.length - 1;
      navigateToFrame(endFrame);
    }
  }
);

// ---- Modal logic (About / Help / Settings) ----

var linkAbout =
  document.getElementById("link-about");
var linkHelp =
  document.getElementById("link-help");
var modalAbout =
  document.getElementById("modal-about");
var modalHelp =
  document.getElementById("modal-help");

var allModals = [
  modalAbout, modalHelp,
];

function openModal(modal) {
  modal.classList.remove("hidden");
}

function closeModal(modal) {
  modal.classList.add("hidden");
}

linkAbout.addEventListener(
  "click",
  function (e) {
    e.preventDefault();
    openModal(modalAbout);
  }
);

linkHelp.addEventListener(
  "click",
  function (e) {
    e.preventDefault();
    openModal(modalHelp);
  }
);

var closeButtons =
  document.querySelectorAll(".modal-close");
for (var ci = 0; ci < closeButtons.length; ci++) {
  (function (btn) {
    btn.addEventListener("click", function () {
      var overlay =
        btn.closest(".modal-overlay");
      if (overlay) {
        closeModal(overlay);
      }
    });
  })(closeButtons[ci]);
}

allModals.forEach(function (modal) {
  modal.addEventListener(
    "click",
    function (e) {
      if (e.target === modal) {
        closeModal(modal);
      }
    }
  );
});

document.addEventListener(
  "keydown",
  function (e) {
    if (e.key === "Escape") {
      for (
        var i = 0;
        i < allModals.length;
        i++
      ) {
        if (
          !allModals[i].classList.contains(
            "hidden"
          )
        ) {
          closeModal(allModals[i]);
        }
      }
    }
  }
);

// ---- Session persistence (survives Analytics navigation) ----

var SESSION_KEY = "diffusion_last_run";

function saveSessionState() {
  if (
    !activeModelId
    || frameHistory.length < 2
    || !lastFinalText
  ) {
    return;
  }
  var base = {
    model: activeModelId,
    prompt: promptInput.value,
    frameHistory: frameHistory,
    perFrameElapsed: perFrameElapsed,
    finalText: lastFinalText,
    params: lastRunParams,
    thinking:
      thinkingPanel && !thinkingPanel.hidden
        ? thinkingContent.textContent
        : "",
    remaskEdits: remaskEdits,
    originalTotalFrames: originalTotalFrames,
    editedRunSaved: editedRunSaved,
    runSaved: runSaved,
    lastSavedRunId: lastSavedRunId,
    statusStep: statusStep.textContent,
    statusElapsed: statusElapsed.textContent,
    statusMessage: statusMessage.textContent,
  };
  var full = Object.assign({}, base, {
    frameTokens: frameTokens,
    frameCanvasIndex: frameCanvasIndex,
    frameMeanConf: frameMeanConf,
    originalFrameHistory: originalFrameHistory,
    originalFrameTokens: originalFrameTokens,
  });
  // Prefer the token-rich payload; fall back to a lighter one
  // if it exceeds the sessionStorage quota (long runs).
  try {
    sessionStorage.setItem(
      SESSION_KEY, JSON.stringify(full)
    );
  } catch (_e) {
    try {
      sessionStorage.setItem(
        SESSION_KEY, JSON.stringify(base)
      );
    } catch (_e2) {
      // Give up silently; state simply won't persist.
    }
  }
}

function clearSessionState() {
  try {
    sessionStorage.removeItem(SESSION_KEY);
  } catch (_e) {
    // ignore
  }
}

function restoreSessionState() {
  if (!activeModelId) {
    return false;
  }
  var raw;
  try {
    raw = sessionStorage.getItem(SESSION_KEY);
  } catch (_e) {
    return false;
  }
  if (!raw) {
    return false;
  }
  var s;
  try {
    s = JSON.parse(raw);
  } catch (_e) {
    return false;
  }
  if (
    !s
    || s.model !== activeModelId
    || !s.frameHistory
    || s.frameHistory.length < 2
  ) {
    return false;
  }

  frameHistory = s.frameHistory;
  frameTokens = s.frameTokens || [];
  // Canvas index + mean confidence must be restored too: a later
  // Edit-Frames resume truncates them to the resume offset, and if they
  // were left empty they would extend to sparse (null) arrays that fail
  // the save's canvas_index validation.
  frameCanvasIndex = s.frameCanvasIndex || [];
  frameMeanConf = s.frameMeanConf || [];
  commitSteps = null;
  diffData = null;
  perFrameElapsed = s.perFrameElapsed || [];
  lastFinalText = s.finalText || "";
  lastRunParams = s.params || null;
  remaskEdits = s.remaskEdits || [];
  originalTotalFrames =
    s.originalTotalFrames || frameHistory.length;
  originalFrameHistory = s.originalFrameHistory || [];
  originalFrameTokens = s.originalFrameTokens || [];
  editedRunSaved = !!s.editedRunSaved;
  runSaved = !!s.runSaved;
  lastSavedRunId = s.lastSavedRunId || null;
  updateGenerateButton();
  if (s.prompt) {
    promptInput.value = s.prompt;
  }

  if (thinkingPanel && thinkingContent) {
    if (s.thinking) {
      thinkingContent.textContent = s.thinking;
      thinkingPanel.hidden = false;
    } else {
      thinkingPanel.hidden = true;
    }
  }
  // A run already saved (or saved+edited) has nothing left to save,
  // so keep Save disabled; activateScrubber re-applies the Edit
  // Frames lock from the restored editedRunSaved flag.
  setSaveAvailable(!runSaved);
  activateScrubber();

  // Restore the footer readouts (Step / Elapsed / message) so the
  // status bar reflects the completed run rather than resetting.
  if (s.statusStep) {
    statusStep.textContent = s.statusStep;
  }
  if (s.statusElapsed) {
    statusElapsed.textContent = s.statusElapsed;
  }
  if (s.statusMessage) {
    statusMessage.textContent = s.statusMessage;
  }
  return true;
}

// ---- Boot ----

function boot() {
  loadSettings();
  loadPromptHistory();
  updatePromptHistoryUI();
  updateStatusPrefs();
  updateHoverHighlight();
  refreshAnalyticsCue();
  fetchModels()
    .then(function (info) {
      var list = info.models || [];
      for (var i = 0; i < list.length; i++) {
        models[list[i].id] = list[i];
      }
      activeModelId =
        info.active
        || info.default
        || (list[0] && list[0].id);
      activeModel =
        models[activeModelId] || list[0] || null;
      activeDevice = info.active_device || null;
      gpuPresent = !!info.gpu_name;
      renderModelSelector(list, activeModelId);
      if (activeModel) {
        buildParamPanel(activeModel);
        applyUniformParamWidth(list);
      }
      setMaskChar();
      var restored = false;
      try {
        restored = restoreSessionState();
      } catch (_e) {
        restored = false;
      }
      if (!restored) {
        showOutputPlaceholder();
      }
      connect();
    })
    .catch(function () {
      showOutputPlaceholder();
      connect();
    });
}

// Hydrate durable UI state from the server first (so boot's synchronous
// localStorage reads see the persisted values), then boot. persistHydrate
// always runs its callback, even if the fetch fails.
persistHydrate(boot);
