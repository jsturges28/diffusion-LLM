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
var statusTps =
  document.getElementById("status-tps");
var statusMessage =
  document.getElementById("status-message");
var statusStack =
  document.getElementById("status-stack");
var loadingOverlay =
  document.getElementById("loading-overlay");
var validationHint =
  document.getElementById("validation-hint");
var toggleExperimental =
  document.getElementById("toggle-experimental");
var btnParamDefaults =
  document.getElementById("btn-param-defaults");

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
var btnWhatIf =
  document.getElementById("btn-what-if");
var overlaySelectGroup =
  document.getElementById("overlay-select-group");
var overlayDrawerHandle =
  document.getElementById("overlay-drawer-handle");
var overlaySelectMount =
  document.getElementById("overlay-select-mount");
var overlayHighlightCheckbox =
  document.getElementById("overlay-highlight-tokens");
var overlaySelect = null;
// Track how the picker was last built so it is only rebuilt when
// the option set actually changes (the Diff option appearing after
// a resume), avoiding leaked listeners from createCustomSelect.
var overlaySelectBuilt = false;
var overlaySelectHasDiff = false;
var overlaySelectHasEntropy = false;
var diffSummary =
  document.getElementById("diff-summary");
var commitLegend =
  document.getElementById("commit-legend");
var altsPopover =
  document.getElementById("token-alts-popover");
var entropyProfileRow =
  document.getElementById("entropy-profile-row");
var entropyProfileCanvas =
  document.getElementById("entropy-profile");
var entropyProfileReadout =
  document.getElementById("entropy-profile-readout");
var diffOverlayControls =
  document.getElementById("diff-overlay-controls");
var diffOriginalSlider =
  document.getElementById("diff-original-opacity");
var diffEditedSlider =
  document.getElementById("diff-edited-opacity");
var diffBlendToggle =
  document.getElementById("diff-blend-toggle");
var runBlendRow =
  document.getElementById("run-blend-row");
var runBlendInput =
  document.getElementById("run-blend");
// Active visual overlay chosen in the picker:
// "none" | "conf" (heatmap) | "diff". Commit-order tinting is a
// separate persistent setting applied only when no overlay is
// selected (see effectiveColorMode).
var overlayMode = "none";
// Memoized per-run commit steps (position index -> settle step),
// null until first needed and invalidated whenever frameTokens
// is replaced (new run, resume, or session restore).
var commitSteps = null;
// The same for the retained pre-edit run, needed because the ghost
// layer of a crossfade settles its positions on its own schedule and
// would be quietly wrong colored by the branch's.
var originalCommitSteps = null;
// Memoized intervention diff (branch vs original final frame),
// null until needed and invalidated alongside commitSteps.
var diffData = null;
// Diff-overlay layer opacities (0-100) and the "difference" blend
// toggle, controlled by the sliders shown in the overlay drawer.
var diffOriginalOpacity = 50;
var diffEditedOpacity = 100;
var diffBlend = false;
// The run crossfade, from 0 (the retained pre-edit run) to 1 (the
// branch). Governs the stacked token layers and the entropy strip in
// every overlay except Diff, which keeps its own two sliders.
var runBlend = 1;

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
// How many positions each frame produced, from the sampler's reveal
// signal. Kept per frame rather than as a running total so the
// footer can report the last step as well as the run average.
var frameRevealed = [];
var frameCanvasIndex = [];
var frameMeanConf = [];
var lastRunParams = null;
var lastFinalText = null;
var originalTotalFrames = 0;
var originalFrameHistory = [];
var originalFrameTokens = [];
// The pre-edit run's own signals, captured once when the first run
// completes. Timing and mean confidence could in principle be
// recovered from the saved frames, but the candidate sets cannot:
// doSubstitute truncates positionAlts at the edit and the branch
// overwrites the rest, so this is the only chance to keep them.
var originalPerFrameElapsed = [];
var originalMeanConf = [];
var originalPositionAlts = [];

// Per-position competing candidates, indexed by token position (not
// by frame): a position's candidate set is fixed the moment it is
// sampled, so each arrives once, on the frame that introduces it.
// Empty unless the model's Alternatives capture was enabled.
var positionAlts = [];
// Position whose candidate popover is open, or null when closed.
// The page is "original", "edited", or null where only one run
// captured candidates and there is nothing to page between.
var altsPopoverPos = null;
var altsPopoverPage = null;
// Token position under the pointer, or null. Drives the glowing
// column in the entropy profile, so it is tracked for every token,
// independent of whether that position captured alternatives.
var entropyHoverPos = null;
// True while "What If" substitution is armed: the popover's
// candidates become clickable instead of read-only.
var substitutionMode = false;

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
// appended starting at resumeFrameOffset. The worker restarts its
// clock for each generate/resume/substitute segment, so elapsed
// samples from the branch are shifted by resumeElapsedOffset (the
// elapsed value at the last frame kept) to stay cumulative and
// aligned with the frame arrays.
var isResuming = false;
var resumeFrameOffset = 0;
var resumeElapsedOffset = 0;

// ---- Output placeholder ----

// The resting state of the output area before any generation and
// after a New Run. (The former idle ASCII scene / donut animations
// were removed.)
//
// Names the resident model rather than the modality, since the
// playground hosts an autoregressive model too and "Diffusion output"
// was simply wrong under it. The bare fallback matches what
// index.html ships and covers boot's failure path, which paints the
// placeholder with no model resolved.
function showOutputPlaceholder() {
  outputArea.textContent = "";
  var placeholder = document.createElement("span");
  placeholder.id = "output-placeholder";
  var name = activeModel ? activeModel.display_name : "";
  placeholder.textContent = name
    ? name + " output will appear here..."
    : "Output will appear here...";
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

// Draw the loading overlay's bar and phase line from one activation
// poll. The headline stays on the model's name throughout, so the
// only thing moving is the part that is actually changing.
function setLoadingProgress(state, progress) {
  var container = document.getElementById(
    "load-progress-container"
  );
  var fill = document.getElementById("load-progress-fill");
  var detail = document.getElementById("load-progress-detail");
  if (!container || !fill || !detail) {
    return;
  }
  var view = overlaysActivationProgress(state, progress);
  var sweeping = view.mode === "sweep";
  container.hidden = view.mode === "hidden";
  fill.classList.toggle("is-sweep", sweeping);
  // While sweeping, the width belongs to the class: the sweep is a
  // short bar sliding across the track, so an inline width would
  // fight it. Handing back on the way out is also what makes the
  // switch to a real measurement one eased move rather than a jump.
  if (sweeping) {
    fill.style.removeProperty("width");
  } else {
    fill.style.width = view.percent + "%";
  }
  // The label always says which phase is running, which is the whole
  // difference between a slow load and an apparently hung one. The
  // percentage only joins it once there is a real one to show, and
  // both go away with the track, since "hidden" means no activation
  // is in flight and so there is no phase to name.
  detail.hidden = view.mode === "hidden";
  detail.textContent = sweeping
    ? view.label + "\u2026"
    : view.label + ", " + view.percent + "%";
}

// Fill the bar and let it be seen full before `done` navigates away.
//
// Every activation now ends here, because the track is on screen for
// all of one: it sweeps through the phases that cannot be measured
// and fills through the ones that can. That is a change from when an
// unmeasurable checkpoint ran with no bar and this had to avoid
// conjuring one for a fifth of a second at the end. A sweep resolving
// into a full bar is the better close, so the `hidden` check below
// now only guards against the overlay never having been raised.
function finishLoadingProgress(done) {
  var container = document.getElementById(
    "load-progress-container"
  );
  if (!container || container.hidden) {
    done();
    return;
  }
  setLoadingProgress("ready", null);
  setTimeout(done, OVERLAYS_LOAD_COMPLETE_HOLD_MS);
}

// The boot path raises the same overlay without going through
// switchModel, so until now nothing polled for progress there: the
// first load of a session, reliably the slowest, was the one with no
// bar. This drives it from the endpoint pollSwitch already uses.
var loadProgressTimer = null;
// Matched to the supervisor's own sampling of the worker, so the bar
// is told as soon as there is anything to tell it. Both pollers here
// use it; their 800ms error retries stay slower on purpose, since a
// failing poll should back off rather than hammer.
var ACTIVATION_POLL_MS = 250;

function startLoadProgressPoll() {
  if (loadProgressTimer !== null) {
    return;
  }

  function tick() {
    fetch("/api/models/activation")
      .then(function (r) {
        return r.json();
      })
      .then(function (status) {
        if (loadProgressTimer === null) {
          return;
        }
        setLoadingProgress(status.state, status.progress);
      })
      .catch(function () {
        // One dropped poll leaves the last reading on screen; the
        // next one corrects it. The load is unaffected either way.
      })
      .then(function () {
        if (loadProgressTimer !== null) {
          loadProgressTimer = setTimeout(
            tick, ACTIVATION_POLL_MS
          );
        }
      });
  }

  loadProgressTimer = setTimeout(tick, 0);
}

function stopLoadProgressPoll() {
  if (loadProgressTimer !== null) {
    clearTimeout(loadProgressTimer);
    loadProgressTimer = null;
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
      input.addEventListener("input", onParamFormChanged);
    } else {
      input.addEventListener("change", validateAllParams);
      input.addEventListener("change", onParamFormChanged);
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
  // A switch ends in location.reload(), which the restore path cannot
  // tell from a trip to Analytics and back. Dropping the snapshot
  // here, where the reload is known to be a switch, is what makes the
  // new model start on a blank canvas. The identity check alone
  // cannot do it: switching away and back lands on a matching
  // (model, device) pair again, so the stale run would return.
  clearSessionState();
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
  // pollSwitch drives the bar from here on; stop the boot poller so
  // the two are never writing the same overlay. Seeding with the same
  // state the first poll will report keeps the opening frame from
  // saying "Loading" for a poll interval before correcting itself to
  // "Starting worker".
  stopLoadProgressPoll();
  setLoadingProgress("starting", null);
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
        finishLoadingProgress(function () {
          location.reload();
        });
        return;
      }
      if (status.state === "error") {
        switchFailed(
          new Error(status.message || "load failed")
        );
        return;
      }
      setLoadingText("Loading " + name + "\u2026");
      setLoadingProgress(status.state, status.progress);
      setTimeout(function () {
        pollSwitch(name);
      }, ACTIVATION_POLL_MS);
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
  stopLoadProgressPoll();
  // Tear the track down rather than leaving it to the next switch to
  // re-sync. The overlay hides by going transparent, not by leaving
  // the layout, so a sweep left on it would keep animating unseen for
  // the rest of the session. "idle" is the reducer's way of saying no
  // activation is in flight, which is exactly the state after this.
  setLoadingProgress("idle", null);
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
    startLoadProgressPoll();
    updateGenerateButton();
  } else if (data.status === "ready") {
    setBadge("ready");
    modelReady = true;
    stopLoadProgressPoll();
    updateGenerateButton();
    // Only the overlay waits. The model is usable the moment the
    // worker says so, and holding the Generate button for a
    // cosmetic beat would be the wrong trade. Re-checking readiness
    // inside the hold keeps a status that flips back mid-beat from
    // pulling the overlay off a load that is starting again.
    finishLoadingProgress(function () {
      if (modelReady) {
        loadingOverlay.classList.add("hidden");
      }
    });
  }
}

function handleFrame(data) {
  frameHistory.push(data.text);
  if (data.tokens) {
    frameTokens.push(data.tokens);
  } else {
    frameTokens.push(null);
  }

  // Candidate sets ride only the frame that introduces their
  // position (the frame's last token), so accumulate by position.
  if (data.alts && data.tokens && data.tokens.length > 0) {
    positionAlts[data.tokens.length - 1] = data.alts;
  }

  if (typeof data.elapsed === "number") {
    // Shifted by the pre-resume total and re-rounded to the worker's
    // two decimals, so the series stays cumulative across segments
    // instead of dropping back to zero at each branch.
    perFrameElapsed.push(
      +(data.elapsed + resumeElapsedOffset).toFixed(2)
    );
  }
  frameRevealed.push(
    data.revealed ? data.revealed.length : 0
  );
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

  // The token view needs per-position metadata; a model that does not
  // send it still gets the character renderer.
  if (data.tokens) {
    renderLiveFrame(data.tokens, data.revealed);
  } else {
    renderFrame(data.text);
  }

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
  updateRunRateFooter();
}

// ---- Elapsed and tokens per second ----

// Both readouts come off perFrameElapsed rather than off the frame in
// hand. data.elapsed is segment-local: after an edit the worker times
// the branch from zero, so reading it directly made the footer's
// Elapsed jump backwards mid-run. perFrameElapsed is already carrying
// the pre-edit total (see handleFrame), so its tail is the real
// wall-clock time of everything generated so far.
function updateRunRateFooter() {
  var frames = perFrameElapsed.length;
  if (frames === 0) {
    return;
  }
  var seconds = perFrameElapsed[frames - 1];
  statusElapsed.textContent =
    "Elapsed: " + seconds.toFixed(1) + "s";
  renderTpsFooter(currentTokensPerSecond());
}

// null whenever the rate would be meaningless rather than merely
// zero: no tokens yet, or a window too short to have been timed. The
// first frame after an edit is the second case, since it lands at the
// pre-edit total and so shares a timestamp with the frame before it.
function currentTokensPerSecond() {
  var frames = perFrameElapsed.length;
  if (frames === 0) {
    return null;
  }
  if (appSettings.tpsMode === "last") {
    var stepSeconds = frames > 1
      ? perFrameElapsed[frames - 1] - perFrameElapsed[frames - 2]
      : perFrameElapsed[0];
    if (!(stepSeconds > 0)) {
      return null;
    }
    return (frameRevealed[frames - 1] || 0) / stepSeconds;
  }
  var total = perFrameElapsed[frames - 1];
  if (!(total > 0)) {
    return null;
  }
  var produced = 0;
  for (var i = 0; i < frames; i++) {
    produced += frameRevealed[i] || 0;
  }
  return produced / total;
}

function renderTpsFooter(rate) {
  var label = appSettings.tpsMode === "last"
    ? "Last step"
    : "Run average";
  statusTps.title = "Tokens per second (" + label.toLowerCase()
    + "). Click to switch.";
  if (rate === null) {
    statusTps.textContent = "T/s: -";
    return;
  }
  // One decimal below 100, none above: past that the fraction is
  // noise and the extra digit only makes the footer jitter.
  var shown = rate < 100
    ? rate.toFixed(1)
    : String(Math.round(rate));
  statusTps.textContent = "T/s: " + shown;
}

function toggleTpsMode() {
  appSettings.tpsMode =
    appSettings.tpsMode === "last" ? "total" : "last";
  overlaysWriteTpsMode(appSettings.tpsMode);
  renderTpsFooter(currentTokensPerSecond());
}

function handleDone(data) {
  setGenerating(false);
  isResuming = false;
  endRunStatus();
  // The chip is still fading as the line fills in beneath it, so
  // ease the row's new shape instead of snapping the chip sideways.
  statusRowReflow(function () {
    statusMessage.textContent = "Done.";
  });
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
    originalPerFrameElapsed = perFrameElapsed.slice();
    originalMeanConf = frameMeanConf.slice();
    originalPositionAlts = positionAlts.slice();
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
  endRunStatus();
  if (remaskMode !== null) {
    // A resume or substitution truncates the run before the worker
    // answers, so a rejected request would otherwise strand the user
    // with a half-run. Roll back to the pre-session snapshot.
    restoreEditSnapshot();
    resetGuidedMode();
  }
  statusRowReflow(function () {
    statusMessage.textContent =
      "Error: " + (data.message || "unknown");
  });
  statusMessage.style.color = "var(--danger)";
  setTimeout(function () {
    statusMessage.style.color = "";
  }, 5000);
  if (frameHistory.length > 1) {
    activateScrubber();
  }
}

// ---- Rendering ----

// Live rendering keeps one span per token position and updates those
// spans in place. The character-by-character renderer below it is
// still the fallback, but it rebuilt the whole output every step: at
// LLaDA's default length that is several hundred inline boxes torn
// down and laid out again per frame, where a token view needs a
// constant ~160 and touches only the ones that actually changed.
var liveTokenSpans = [];

// No hooks, on purpose. The streaming view draws plain masked and
// resolved tokens with no overlay tint, no confidence grading, and no
// hover tooltip, matching the character renderer exactly. Any of
// those would be a visual change smuggled in with a refactor.
var LIVE_TOKEN_OPTIONS = {};

function renderLiveFrame(tokens, revealed) {
  outputArea.classList.remove("token-layers");
  outputArea.classList.add("live-tokens");
  // Reuse only while our spans are still the ones on the page. Any
  // other render path wipes the container, which detaches them, so
  // reading the parent back is what keeps this self-healing instead
  // of depending on every one of those paths to tell us.
  var reusable = liveTokenSpans.length === tokens.length
    && liveTokenSpans.length > 0
    && liveTokenSpans[0].parentNode === outputArea;
  if (reusable) {
    for (var i = 0; i < tokens.length; i++) {
      overlaysSyncTokenSpan(
        liveTokenSpans[i], i, tokens[i], MASK_CHAR,
        LIVE_TOKEN_OPTIONS
      );
    }
  } else {
    rebuildLiveTokens(tokens);
  }
  markTokenBirths(revealed);
}

function rebuildLiveTokens(tokens) {
  var fragment = document.createDocumentFragment();
  liveTokenSpans = new Array(tokens.length);
  for (var i = 0; i < tokens.length; i++) {
    var span = overlaysBuildTokenSpan(
      i, tokens[i], MASK_CHAR, LIVE_TOKEN_OPTIONS
    );
    liveTokenSpans[i] = span;
    fragment.appendChild(span);
  }
  // The spans the queue was tracking are about to be detached, so
  // their glows end here whether or not they had finished.
  tokenBirthQueue = [];
  outputArea.textContent = "";
  outputArea.appendChild(fragment);
}

// ---- Birth glow ----

// A token flashes once, at apex, the instant it is denoised. Capped
// because a low-step LLaDA run reveals a couple of dozen positions in
// one frame, and each is a blurred repaint region on a renderer with
// a documented history of struggling with exactly that. Past the cap
// the oldest flash is cut short, which is invisible in practice: it
// is already most of the way decayed.
//
// The cap has to follow the fade rather than being fixed. How many
// tokens glow at once is roughly the generation rate times the fade,
// so a long fade at autoregressive speeds would otherwise have the
// queue, not the timer, decide when a flash ends: the trail would
// stop growing exactly when the user lengthened it, and its tail
// would look cut rather than faded.
//
// The rate ceiling is picked so the 500ms default lands on 48, which
// is what this was before it became a function of the fade.
var TOKEN_BIRTH_RATE_CEILING = 96;
var TOKEN_BIRTH_CONCURRENT_MIN = 48;
var TOKEN_BIRTH_CONCURRENT_MAX = 192;
var TOKEN_BIRTH_ANIMATION = "token-birth";
var tokenBirthQueue = [];
var tokenBirthMaxConcurrent = TOKEN_BIRTH_CONCURRENT_MIN;

// Set the live canvas' glow to the active model class' preferences,
// and size the concurrency cap to the fade it asks for. Called once
// per page load: a model switch ends in location.reload(), so the
// active model cannot change under a live canvas.
function applyTokenBirthGlow() {
  var modelType =
    activeModel
    && activeModel.capabilities
    && activeModel.capabilities.model_type;
  var glow = overlaysGlowFor(appSettings, modelType);
  overlaysApplyGlowVars(
    outputArea, glow.brightness, glow.fadeMs
  );
  tokenBirthMaxConcurrent = tokenBirthConcurrentCap(glow.fadeMs);
}

function tokenBirthConcurrentCap(fadeMs) {
  var expected = Math.round(
    (fadeMs / 1000) * TOKEN_BIRTH_RATE_CEILING
  );
  if (expected < TOKEN_BIRTH_CONCURRENT_MIN) {
    return TOKEN_BIRTH_CONCURRENT_MIN;
  }
  if (expected > TOKEN_BIRTH_CONCURRENT_MAX) {
    return TOKEN_BIRTH_CONCURRENT_MAX;
  }
  return expected;
}

function markTokenBirths(revealed) {
  if (!revealed || revealed.length === 0) {
    return;
  }
  if (!appSettings.tokenBirthGlow) {
    return;
  }
  if (prefersReducedMotion()) {
    return;
  }
  for (var i = 0; i < revealed.length; i++) {
    var span = liveTokenSpans[revealed[i]];
    if (span) {
      startTokenBirth(span);
    }
  }
}

function startTokenBirth(span) {
  if (span.hasAttribute("data-born")) {
    // Already mid-flash. Restarting the animation would need a
    // forced reflow per span, which is the cost this whole path
    // exists to avoid, and a token that is already glowing looks
    // the same either way.
    return;
  }
  span.setAttribute("data-born", "");
  tokenBirthQueue.push(span);
  while (tokenBirthQueue.length > tokenBirthMaxConcurrent) {
    tokenBirthQueue.shift().removeAttribute("data-born");
  }
}

// Delegated: animationend bubbles, so one listener on the container
// serves every span and none of them needs its own.
function onTokenBirthEnd(e) {
  if (e.animationName !== TOKEN_BIRTH_ANIMATION) {
    return;
  }
  var span = e.target;
  span.removeAttribute("data-born");
  var at = tokenBirthQueue.indexOf(span);
  if (at !== -1) {
    tokenBirthQueue.splice(at, 1);
  }
}

function renderFrame(text) {
  outputArea.classList.remove("token-layers");
  outputArea.classList.remove("live-tokens");
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
  outputArea.classList.remove("live-tokens");
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

// Drop every memo derived from the frame arrays. Called wherever
// those arrays are replaced or truncated, in one place so the three
// can never fall out of step with each other.
function invalidateRunMemos() {
  commitSteps = null;
  originalCommitSteps = null;
  diffData = null;
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
  var diff = currentDiffData();
  var editedTokens = frameTokens[frameIndex] || [];
  var oIdx = Math.min(
    frameIndex, originalFrameTokens.length - 1
  );
  var origTokens =
    (oIdx >= 0 ? originalFrameTokens[oIdx] : null) || [];

  var total = editedTokens.length;
  outputArea.textContent = "";
  tokenHighlightPos = null;
  outputArea.classList.add("token-layers");
  outputArea.appendChild(
    overlaysBuildDiffLayers(
      origTokens,
      editedTokens,
      diff,
      {
        originalOpacity: diffOriginalOpacity,
        editedOpacity: diffEditedOpacity,
        blend: diffBlend,
        titleFor: function (index, tok) {
          if (!tok || tok.m) {
            return tokenLabel(index, total)
              + "\nConfidence: 0";
          }
          return tokenLabel(index, total) + "\n"
            + "Confidence: " + confLabel(tok.c)
            + tokenExtraLabel(tok);
        },
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

// Trailing tooltip lines for one token: entropy, when captured. No
// nudge toward the candidates, since hovering is what opened this
// tooltip and the popover arrives on the same gesture.
function tokenExtraLabel(tok) {
  if (typeof tok.e !== "number") {
    return "";
  }
  return "\nEntropy: " + String(+tok.e.toFixed(3)) + " nats";
}

// The divergence map, computed on first use and memoized. Both the
// diff coloring and the diff tooltip lines need it, so the lazy
// build lives here rather than in each.
function currentDiffData() {
  if (diffData === null) {
    diffData = computeDiff();
  }
  return diffData;
}

// Commit steps for whichever of the two runs a layer is drawing,
// memoized separately. The runs settle their positions on different
// schedules, so the ghost layer painted from the branch's would
// misreport every position past the edit.
function commitStepsFor(isOriginal) {
  if (isOriginal) {
    if (originalCommitSteps === null) {
      originalCommitSteps =
        overlaysComputeCommitSteps(originalFrameTokens);
    }
    return originalCommitSteps;
  }
  if (commitSteps === null) {
    commitSteps = computeCommitSteps();
  }
  return commitSteps;
}

// Which step resolved a position under the Commit Order overlay, or
// null where the run recorded none.
function tokenCommitStep(index, isOriginal) {
  var step = commitStepsFor(isOriginal)[index];
  if (typeof step !== "number" || step < 0) {
    return null;
  }
  return step;
}

// The active overlay's color for one resolved token, or null to let
// the token's own class color it. Kept separate from the tooltip
// lines the same overlay contributes (tokenTitleExtra below) because
// the span builder takes color and title as independent callbacks.
function tokenColorAt(index, tok, isOriginal) {
  var mode = effectiveColorMode();
  if (mode === "conf") {
    if (typeof tok.c !== "number") {
      return null;
    }
    return heatColor(tok.c);
  }
  if (mode === "entropy") {
    if (typeof tok.e !== "number") {
      return null;
    }
    return entropyColor(tok.e);
  }
  if (mode === "commit") {
    var step = tokenCommitStep(index, isOriginal);
    if (step === null) {
      return null;
    }
    var frames = isOriginal ? originalFrameTokens : frameTokens;
    return commitColor(step, frames.length - 1);
  }
  if (mode === "diff") {
    var diff = currentDiffData();
    if (diff.origins[index]) {
      return "#ff8a3d";
    }
    return diffColor(!!diff.changed[index]);
  }
  return null;
}

// The overlay-specific trailing tooltip lines for one resolved
// token, on top of the confidence and entropy lines every token
// carries.
function tokenTitleExtra(index, isOriginal) {
  var mode = effectiveColorMode();
  if (mode === "commit") {
    var step = tokenCommitStep(index, isOriginal);
    if (step === null) {
      return "";
    }
    return "\nResolved at step: " + step;
  }
  if (mode === "diff") {
    var diff = currentDiffData();
    if (diff.origins[index]) {
      return "\n(remasked here)";
    }
    if (diff.changed[index]) {
      return "\nwas: " + diff.origText[index];
    }
    return "";
  }
  return "";
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

// The per-token hover highlight, independent of any coloring overlay.
// Its control is the overlay drawer's checkbox rather than a Settings
// row, so it sits next to the tokens it acts on; the value is still
// persisted (and shared with Analytics) through the settings blob.
function updateHoverHighlight() {
  if (overlayHighlightCheckbox) {
    overlayHighlightCheckbox.checked = appSettings.highlightTokens;
  }
  if (!outputArea) {
    return;
  }
  outputArea.classList.toggle(
    "token-hover-highlight",
    appSettings.highlightTokens
  );
}

function onOverlayHighlightToggle() {
  appSettings.highlightTokens = overlayHighlightCheckbox.checked;
  overlaysWriteHighlightTokens(appSettings.highlightTokens);
  updateHoverHighlight();
}

// Select the active visual overlay from the picker and re-render.
function setOverlayMode(mode) {
  overlayMode = mode;
  updateDiffSummary();
  updateDiffOverlayControls();
  updateRunBlendControls();
  updateCommitLegend();
  hideAltsPopover();
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

// The run crossfade is the other overlays' answer to the diff
// sliders, so the two rows are mutually exclusive: whichever one
// governs the layers currently on screen is the one that shows.
function updateRunBlendControls() {
  if (!runBlendRow) {
    return;
  }
  runBlendRow.hidden = !(
    overlayMode !== "diff" && runBlendActive()
  );
}

// Back to the branch at full opacity. Called per run, so a resumed
// branch opens on itself rather than on the previous mix.
function resetRunBlend() {
  runBlend = 1;
  if (runBlendInput) {
    runBlendInput.value = "100";
  }
  updateRunBlendControls();
}

// Restyle the stacked layers in place. Rebuilding them would mean
// several hundred spans per slider step, and would drop the
// candidate popover mid-drag. Diff mode is left alone: its own two
// sliders own the layers there.
function onRunBlendInput() {
  runBlend = Number(runBlendInput.value) / 100;
  if (overlayMode !== "diff") {
    applyRunBlendToLayers();
  }
  // Gated rather than drawn directly: the strip must stay hidden for
  // a run that carries no entropy at all.
  updateEntropyProfileVisibility();
}

function applyRunBlendToLayers() {
  var original =
    outputArea.querySelector(".token-layer-original");
  var edited =
    outputArea.querySelector(".token-layer-edited");
  if (!original || !edited) {
    return;
  }
  original.style.opacity = String(1 - runBlend);
  edited.style.opacity = String(runBlend);
  overlaysApplyLayerPointers(
    outputArea, 1 - runBlend, runBlend
  );
}

// Which run the crossfade currently favors. Drives the entropy
// strip's readout and the candidate popover's opening page, so both
// agree with the tokens the user is actually reading.
function runBlendFavorsOriginal() {
  return runBlend < 0.5;
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
  var diff = currentDiffData();
  var total = diff.totalCount;
  var changed = diff.changedCount;
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

// Whether the run carries per-token entropy. Gated on the data
// rather than on model_type, so the overlay appears for any model
// that starts emitting `e` (autoregressive runs are just the first).
function entropyAvailable() {
  var tokens = frameTokens[frameTokens.length - 1];
  if (!tokens) {
    return false;
  }
  for (var i = 0; i < tokens.length; i++) {
    if (typeof tokens[i].e === "number") {
      return true;
    }
  }
  return false;
}

// Whether any position captured competing candidates for the hover
// popover (and, for models that support it, What If substitution).
function alternativesAvailable() {
  return hasAnyAlternatives(positionAlts);
}

// True when at least one position captured a candidate set. Takes
// the array so it answers for the pre-edit run as well as the live
// one.
function hasAnyAlternatives(positions) {
  for (var i = 0; i < positions.length; i++) {
    var alts = positions[i];
    if (alts && alts.length > 0) {
      return true;
    }
  }
  return false;
}

// The earliest position a What If branch could differ at: the
// leftmost remasked position across every edit. Null on an unedited
// run. Left of it the branch reproduces the original verbatim,
// candidate sets included, so there is nothing to compare there.
function editDivergencePosition() {
  var earliest = null;
  for (var e = 0; e < remaskEdits.length; e++) {
    var positions = remaskEdits[e].token_positions || [];
    for (var p = 0; p < positions.length; p++) {
      if (earliest === null || positions[p] < earliest) {
        earliest = positions[p];
      }
    }
  }
  return earliest;
}

// ---- Top-k alternatives popover ----

function hideAltsPopover() {
  if (!altsPopover) {
    return;
  }
  altsPopover.hidden = true;
  altsPopover.textContent = "";
  altsPopoverPos = null;
  altsPopoverPage = null;
}

// Whether this position has a candidate set from each run to page
// between. Only possible at or past the divergence point, and only
// for a branch whose pre-edit candidates were retained.
function altsPageable(pos) {
  var divergence = editDivergencePosition();
  if (divergence === null || pos < divergence) {
    return false;
  }
  var original = originalPositionAlts[pos];
  var edited = positionAlts[pos];
  return !!(
    original && original.length > 0
    && edited && edited.length > 0
  );
}

// Build one row per candidate: the token text, a proportional bar,
// and its probability. The chosen token is marked so the popover
// reads as "what it picked, and what it nearly picked instead".
function buildAltsRows(alts, chosenId) {
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < alts.length; i++) {
    var alt = alts[i];
    var row = document.createElement("div");
    row.className = "alt-row";
    if (alt.id === chosenId) {
      row.classList.add("alt-row-chosen");
    }
    row.setAttribute("data-alt-id", String(alt.id));

    var text = document.createElement("span");
    text.className = "alt-text";
    text.textContent = overlaysAltDisplay(alt.t);
    row.appendChild(text);

    var bar = document.createElement("span");
    bar.className = "alt-bar";
    var fill = document.createElement("span");
    fill.className = "alt-bar-fill";
    fill.style.width =
      Math.round(Math.max(0, Math.min(1, alt.p)) * 100)
      + "%";
    bar.appendChild(fill);
    row.appendChild(bar);

    var prob = document.createElement("span");
    prob.className = "alt-prob";
    prob.textContent =
      (Math.max(0, Math.min(1, alt.p)) * 100).toFixed(1)
      + "%";
    row.appendChild(prob);

    fragment.appendChild(row);
  }
  return fragment;
}

// Which run's candidates a pageable position opens on: the one the
// crossfade is favoring, so the popover agrees with the tokens and
// the entropy strip. Both pages stay reachable through the arrows
// either way, so the midpoint picks a default rather than gating
// access.
function defaultAltsPage() {
  return runBlendFavorsOriginal() ? "original" : "edited";
}

// Show the candidate popover for a token position, anchored to its
// span. The pager reaches the pre-edit set where one was retained.
function showAltsPopover(pos, span) {
  altsPopoverPage = altsPageable(pos)
    ? defaultAltsPage() : null;
  renderAltsPopover(pos, span);
}

// Flip pages in place. Rendered without an anchor deliberately: the
// two pages can differ in height, and re-placing the box under the
// pointer that just clicked an arrow can slide it out from under that
// pointer, firing the mouseleave that closes it.
function setAltsPage(page) {
  if (altsPopoverPos === null) {
    return;
  }
  altsPopoverPage = page;
  renderAltsPopover(altsPopoverPos, null);
}

// With an anchor span, positioned in viewport coordinates (the
// popover is fixed at body level) and flipped above the token when it
// would overflow. Without one, left where it already sits.
function renderAltsPopover(pos, span) {
  if (!altsPopover) {
    return;
  }
  var original = altsPopoverPage === "original";
  var alts = original
    ? originalPositionAlts[pos] : positionAlts[pos];
  if (!alts || alts.length === 0) {
    hideAltsPopover();
    return;
  }
  // Each page marks the token its own run drew, so the Original page
  // does not mark the branch's substitution as chosen.
  var tokens = original
    ? originalFrameTokens[originalFrameTokens.length - 1]
    : frameTokens[currentScrubFrame];
  var chosen = tokens && tokens[pos] ? tokens[pos].id : null;

  altsPopover.textContent = "";
  altsPopover.appendChild(
    overlaysBuildAltHeading(pos, altsPopoverPage, setAltsPage)
  );
  altsPopover.appendChild(buildAltsRows(alts, chosen));
  // Substitution only ever applies to the live run, so the Original
  // page is read-only even while What If is armed.
  var pickable = substitutionMode && !original;
  if (pickable) {
    var hint = document.createElement("div");
    hint.className = "alt-hint";
    hint.textContent = "Click a candidate to substitute";
    altsPopover.appendChild(hint);
  }
  altsPopover.classList.toggle("alt-pickable", pickable);

  // Measure before placing: the popover must be visible for its
  // height to be known, so unhide first, then correct the position.
  altsPopover.hidden = false;
  if (span) {
    var rect = span.getBoundingClientRect();
    var box = altsPopover.getBoundingClientRect();
    altsPopover.style.left =
      overlaysPopoverLeft(rect, box) + "px";
    altsPopover.style.top =
      overlaysPopoverTop(rect, box) + "px";
  }
  altsPopoverPos = pos;
}

// ---- Per-position entropy profile ----

// The app's edit color, shared with .token-remasked in style.css and
// with EDIT_COLOR in analytics.js, so an intervention reads the same
// in the strip as it does in the tokens above it.
var EDIT_MARKER_COLOR = "#ff9f1c";
var EDIT_MARKER_TINT = "rgba(255, 159, 28, 0.15)";

function entropyValuesFrom(tokens) {
  if (!tokens) {
    return [];
  }
  var values = [];
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    values.push(
      typeof tok.e === "number" ? tok.e : 0
    );
  }
  return values;
}

// Entropy per position, read off the final frame's token records
// (each position is sampled once, so its entropy never changes).
function entropyProfileValues() {
  return entropyValuesFrom(
    frameTokens[frameTokens.length - 1]
  );
}

// The same for the retained pre-edit run, so the crossfade can mix
// the two profiles. Empty unless a branch exists to compare against,
// which collapses the strip back to a single series.
function originalEntropyProfileValues() {
  if (!runBlendActive()) {
    return [];
  }
  return entropyValuesFrom(
    originalFrameTokens[originalFrameTokens.length - 1]
  );
}

// How many columns the strip spans: the longer of the two runs, so
// the drawing and the pointer-to-position inverse agree on the step
// even when a branch outran the original.
function entropyProfileColumns() {
  return Math.max(
    entropyProfileValues().length,
    originalEntropyProfileValues().length
  );
}

// Every position an edit touched. Sequential What If rounds each push
// their own entry, so a branch can carry more than one, and a
// diffusion remask contributes a whole group at once. Sibling to
// editDivergencePosition, which reduces the same records to their
// minimum for the popover's pager.
function editedProfilePositions() {
  var seen = {};
  var positions = [];
  for (var e = 0; e < remaskEdits.length; e++) {
    var group = remaskEdits[e].token_positions || [];
    for (var p = 0; p < group.length; p++) {
      var pos = group[p];
      if (seen[pos] !== true) {
        seen[pos] = true;
        positions.push(pos);
      }
    }
  }
  return positions;
}

// Draw the profile: one column per position, height proportional to
// normalized entropy, colored by the same ramp as the overlay. The
// column for the frame under the scrubber is highlighted so the
// profile and the canvas stay tied together. On an edited run the
// pre-edit profile is drawn underneath and the two are mixed by the
// run crossfade, exactly as the token layers above them are.
function drawEntropyProfile() {
  if (!entropyProfileCanvas || !entropyProfileRow) {
    return;
  }
  var values = entropyProfileValues();
  if (values.length === 0) {
    entropyProfileRow.hidden = true;
    return;
  }
  entropyProfileRow.hidden = false;

  // Match the backing store to the CSS box so columns stay crisp on
  // HiDPI displays and after a window resize.
  var ratio = window.devicePixelRatio || 1;
  var cssWidth = entropyProfileCanvas.clientWidth || 1;
  var cssHeight = entropyProfileCanvas.clientHeight || 34;
  entropyProfileCanvas.width = Math.round(cssWidth * ratio);
  entropyProfileCanvas.height = Math.round(
    cssHeight * ratio
  );
  var ctx = entropyProfileCanvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  // Frame index maps straight onto position: the autoregressive
  // worker emits no leading empty canvas (ar_sampler._build_frame
  // runs after the pick is appended), so frameHistory[k] holds k+1
  // tokens and the frame at k is the one that introduced position k.
  // The profile only renders for runs carrying per-token entropy,
  // which is autoregressive-only, so the diffusion all-mask frame 0
  // does not apply here.
  //
  // The scrubber's position is carried by the bar's own opacity
  // rather than a drawn marker. A standing neutral guide reads as an
  // artifact at rest, and drawEntropyProfileGlow already owns that
  // visual language for the column under the pointer. The orange
  // edit marker below is a different statement: it names a position
  // the run was intervened at, which is true whether or not the
  // pointer is anywhere near it.
  var current = currentScrubFrame;
  var original = originalEntropyProfileValues();
  // Stepped off the longer run so the two profiles stay
  // position-aligned when a branch outran or fell short of the
  // original.
  var step = cssWidth / entropyProfileColumns();
  var layout = {
    step: step,
    barWidth: Math.max(1, step - 0.5),
    cssHeight: cssHeight,
  };

  // Tint under the bars, dashed guide over them, hover glow last:
  // the same stacking the Analytics entropy chart gets from its
  // plugin order, so the pointer's guide lays over the edit tint
  // rather than under it.
  var edits = editedProfilePositions();
  drawEntropyProfileEditTint(ctx, layout, edits);

  var paired = original.length > 0;
  if (paired) {
    drawEntropyProfileSeries(ctx, layout, {
      values: original,
      alpha: 1 - runBlend,
      // The scrubber indexes the branch, so the pre-edit run gets no
      // current-position emphasis of its own.
      current: -1,
    });
  }
  drawEntropyProfileSeries(ctx, layout, {
    values: values,
    alpha: paired ? runBlend : 1,
    current: current,
  });
  drawEntropyProfileEditLines(ctx, layout, edits);

  // The glow and the readout speak for one run, so they follow
  // whichever the crossfade is favoring.
  layout.values = (paired && runBlendFavorsOriginal())
    ? original : values;
  drawEntropyProfileGlow(ctx, layout);
  updateEntropyReadout(
    layout.values,
    entropyHoverPos === null ? current : entropyHoverPos
  );
}

// A faint column behind each edited position. Floored at 2px like
// the hover guide: at a few hundred tokens a bar-width tint is too
// thin to notice.
function drawEntropyProfileEditTint(ctx, layout, positions) {
  if (positions.length === 0) {
    return;
  }
  ctx.save();
  ctx.fillStyle = EDIT_MARKER_TINT;
  for (var i = 0; i < positions.length; i++) {
    ctx.fillRect(
      positions[i] * layout.step,
      0,
      Math.max(2, layout.barWidth),
      layout.cssHeight
    );
  }
  ctx.restore();
}

// The dashed guide, drawn over the bars and centered on the column
// it marks so it reads as belonging to that position rather than to
// the gap beside it. Mirrors substitutionMarkerPlugin in
// analytics.js, down to the dash pattern.
function drawEntropyProfileEditLines(ctx, layout, positions) {
  if (positions.length === 0) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = EDIT_MARKER_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  for (var i = 0; i < positions.length; i++) {
    var x = positions[i] * layout.step
      + layout.barWidth / 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, layout.cssHeight);
    ctx.stroke();
  }
  ctx.restore();
}

// One profile's columns, at a shared alpha for the crossfade times
// each bar's own emphasis. ``current`` is the position the scrubber
// sits on, or -1 for a series the scrubber does not index.
function drawEntropyProfileSeries(ctx, layout, series) {
  if (series.alpha <= 0.01) {
    return;
  }
  for (var i = 0; i < series.values.length; i++) {
    var value = series.values[i];
    var frac = overlaysEntropyFraction(value);
    var height = Math.max(1, frac * (layout.cssHeight - 2));
    var emphasis = i === series.current ? 1 : 0.68;
    ctx.globalAlpha = emphasis * series.alpha;
    ctx.fillStyle = entropyColor(value);
    ctx.fillRect(
      i * layout.step,
      layout.cssHeight - height,
      layout.barWidth,
      height
    );
  }
  ctx.globalAlpha = 1;
}

// Light up the column for the token under the pointer: a faint
// full-height guide so a column a few pixels wide is findable at a
// glance, then the bar redrawn brighter with a halo of its own hue.
function drawEntropyProfileGlow(ctx, layout) {
  var pos = entropyHoverPos;
  if (pos === null || pos < 0 || pos >= layout.values.length) {
    return;
  }
  var value = layout.values[pos];
  var left = pos * layout.step;
  ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
  ctx.fillRect(
    left, 0, Math.max(2, layout.barWidth), layout.cssHeight
  );

  var frac = overlaysEntropyFraction(value);
  var height = Math.max(2, frac * (layout.cssHeight - 2));
  var top = layout.cssHeight - height;
  ctx.shadowColor = entropyColor(value);
  ctx.shadowBlur = 8;
  ctx.fillStyle = entropyGlowColor(value);
  // Twice, so the halo builds to something visible against the
  // neighboring columns without washing the bar itself out.
  ctx.fillRect(left, top, layout.barWidth, height);
  ctx.fillRect(left, top, layout.barWidth, height);
  ctx.shadowBlur = 0;
  ctx.shadowColor = "transparent";
}

function updateEntropyReadout(values, index) {
  if (!entropyProfileReadout) {
    return;
  }
  if (index < 0 || index >= values.length) {
    entropyProfileReadout.textContent = "";
    return;
  }
  entropyProfileReadout.textContent =
    String(+values[index].toFixed(2)) + " nats";
}

// Track the hovered token and repaint the profile when it changes.
// Cheap: the profile is one canvas of a few hundred rects, and the
// early return keeps mouseover from redrawing on every pixel of
// movement within a single token.
function setEntropyHoverPosition(pos) {
  // Hover state only exists while the profile is on screen, so a
  // token hovered mid-generation cannot leave a stale column lit when
  // the scrubber later appears.
  var visible = entropyProfileRow && !entropyProfileRow.hidden;
  var next = visible ? pos : null;
  if (entropyHoverPos === next) {
    return;
  }
  entropyHoverPos = next;
  if (visible) {
    drawEntropyProfile();
  }
}

// ---- Cross-highlighting: entropy profile -> token view ----
//
// The token -> column direction already exists (the output area's
// mouseover feeds setEntropyHoverPosition). These close the loop, so
// a tall warm column can be read back to the word behind it.

// Position currently lit from the profile, so sweeping the pointer
// across one column does not re-query the DOM on every pixel. Reset
// by the render paths below, which drop the class with the spans.
var tokenHighlightPos = null;

// Light the token(s) at a position. There are two while the diff
// overlay is stacked, and lighting both keeps the mark visible
// whichever layer is on top.
function setTokenHighlight(pos) {
  if (tokenHighlightPos === pos) {
    return;
  }
  clearTokenHighlight();
  tokenHighlightPos = pos;
  if (pos === null || !outputArea) {
    return;
  }
  var spans = outputArea.querySelectorAll(
    "[data-pos=\"" + pos + "\"]"
  );
  for (var i = 0; i < spans.length; i++) {
    spans[i].classList.add("token-cross-highlight");
  }
}

function clearTokenHighlight() {
  tokenHighlightPos = null;
  if (!outputArea) {
    return;
  }
  var lit = outputArea.querySelectorAll(
    ".token-cross-highlight"
  );
  for (var i = 0; i < lit.length; i++) {
    lit[i].classList.remove("token-cross-highlight");
  }
}

// Map a pointer x on the profile back to a token position by
// inverting the layout drawEntropyProfile lays down. Columns are
// contiguous at `step` (the half-pixel gap is taken out of the bar,
// not the slot), so the floor of x/step names the column drawn there.
// Uses clientWidth, the same measure the draw does.
function entropyProfilePosition(event) {
  if (!entropyProfileCanvas) {
    return null;
  }
  var columns = entropyProfileColumns();
  if (columns === 0) {
    return null;
  }
  var cssWidth = entropyProfileCanvas.clientWidth || 1;
  var step = cssWidth / columns;
  var rect = entropyProfileCanvas.getBoundingClientRect();
  var index = Math.floor((event.clientX - rect.left) / step);
  if (index < 0 || index >= columns) {
    return null;
  }
  return index;
}

// Show the profile only when the run carries entropy and the
// scrubber is driving a token view.
function updateEntropyProfileVisibility() {
  if (!entropyProfileRow) {
    return;
  }
  if (!scrubberActive || !entropyAvailable()) {
    entropyProfileRow.hidden = true;
    return;
  }
  drawEntropyProfile();
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
  var hasEntropy = entropyAvailable();
  if (overlayMode === "diff" && !hasDiff) {
    overlayMode = "none";
  }
  if (overlayMode === "entropy" && !hasEntropy) {
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
  if (
    overlaySelectBuilt
    && hasDiff === overlaySelectHasDiff
    && hasEntropy === overlaySelectHasEntropy
  ) {
    if (overlaySelect) {
      overlaySelect.value = overlayMode;
    }
    return;
  }
  var options = [
    { value: "none", label: "None" },
    { value: "conf", label: "Heatmap" },
  ];
  // Entropy answers a different question than the confidence
  // Heatmap: how undecided the model was over the whole vocabulary,
  // not how likely the token it chose was.
  if (hasEntropy) {
    options.push({ value: "entropy", label: "Entropy" });
  }
  // Commit Order tints by resolution step, which a left-to-right model
  // does not have (its commit order is just position order), so it
  // stays diffusion-only.
  if (!isAutoregressive()) {
    options.push({ value: "commit", label: "Commit Order" });
  }
  // Diff needs a branch to compare against. Diffusion runs list it
  // up front (disabled until Edit Frames produces one); autoregressive
  // runs list it only once a What If substitution has, since there is
  // no equivalent standing invitation for them.
  if (!isAutoregressive() || hasDiff) {
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
  overlaySelectHasEntropy = hasEntropy;
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

// Apply the (saved) settings to the live app: hover highlight and any
// active token coloring.
function applySettings() {
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

// A user-remasked position draws the mask glyph even though its
// token is still resolved: the selection is a statement about what
// the next run will redraw, not about what this frame holds.
function tokenMaskedFn(index) {
  return remaskedPositions[index] === true;
}

// Beyond token-mask / token-resolved, a span can be a remask
// selection, or invite the click that makes one, or invite the hover
// that opens its candidates. The latter two are edit-mode
// affordances, so they never appear on a stacked layer.
function tokenClassFn(index, tok, masked) {
  if (remaskedPositions[index] === true) {
    return "token-remasked";
  }
  if (masked) {
    return "";
  }
  var classes = [];
  if (remaskMode === "edit") {
    classes.push("token-clickable");
  }
  if (substitutionMode && positionAlts[index]) {
    classes.push("token-substitutable");
  }
  return classes.join(" ");
}

// Mask opacity tracks the model's live predicted confidence for the
// position (0 or absent gives the solid floor). A remask selection
// is held fully opaque instead, so it reads as a choice rather than
// as one more low-confidence mask.
function tokenOpacityFn(index, tok, masked) {
  if (!masked || remaskedPositions[index] === true) {
    return null;
  }
  return maskOpacity(tok ? tok.c : null);
}

function tokenColorFn(isOriginal) {
  return function (index, tok) {
    if (!tok || tok.m || remaskedPositions[index] === true) {
      return null;
    }
    return tokenColorAt(index, tok, isOriginal);
  };
}

// The hover tooltip for a token in a layer of ``total`` positions.
// Closed over the total because the two stacked runs can differ in
// length, and each should count its own.
function tokenTitleFn(total, isOriginal) {
  return function (index, tok) {
    var line = tokenLabel(index, total) + "\n";
    if (remaskedPositions[index] === true || !tok) {
      // Whatever confidence the old token carried says nothing about
      // a position queued for remasking.
      return line + "Confidence: 0";
    }
    if (tok.m) {
      return line + "Confidence: " + confLabel(tok.c);
    }
    return line + "Confidence: " + confLabel(tok.c)
      + tokenExtraLabel(tok)
      + tokenTitleExtra(index, isOriginal);
  };
}

// The full callback set for one layer of ``total`` tokens drawn from
// either the branch or the retained pre-edit run.
function tokenLayerOptions(total, isOriginal) {
  return {
    maskChar: MASK_CHAR,
    maskedFor: tokenMaskedFn,
    classFor: tokenClassFn,
    opacityFor: tokenOpacityFn,
    colorFor: tokenColorFn(isOriginal),
    titleFor: tokenTitleFn(total, isOriginal),
  };
}

// Whether the run crossfade governs the token view: only once a
// branch exists to compare against, and never mid-edit, where the
// tokens are a target for clicks rather than something to read.
function runBlendActive() {
  return diffAvailable() && remaskMode === null;
}

function renderFrameWithTokens(frameIndex) {
  // Leaving the live view: the mask glow this class restores is for
  // streaming only, and every branch below owns the container now.
  outputArea.classList.remove("live-tokens");
  var tokens = frameTokens[frameIndex];
  if (!tokens) {
    renderFrame(frameHistory[frameIndex]);
    return;
  }

  // Diff overlay takes over rendering (two stacked layers of its
  // own, under two independent opacity sliders).
  if (overlayMode === "diff" && runBlendActive()) {
    renderDiffOverlay(frameIndex);
    return;
  }

  tokenHighlightPos = null;
  outputArea.textContent = "";
  if (runBlendActive()) {
    outputArea.classList.add("token-layers");
    outputArea.appendChild(
      buildCrossfadedLayers(frameIndex, tokens)
    );
    return;
  }

  outputArea.classList.remove("token-layers");
  var options = tokenLayerOptions(tokens.length, false);
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < tokens.length; i++) {
    fragment.appendChild(
      overlaysBuildTokenSpan(i, tokens[i], MASK_CHAR, options)
    );
  }
  outputArea.appendChild(fragment);
}

// The pre-edit run and the branch drawn on top of each other, mixed
// by the run crossfade. The original layer clamps to its final frame
// past its own end, so a branch that outran it keeps a stable ghost
// rather than emptying out.
function buildCrossfadedLayers(frameIndex, editedTokens) {
  var oIdx = Math.min(
    frameIndex, originalFrameTokens.length - 1
  );
  var origTokens =
    (oIdx >= 0 ? originalFrameTokens[oIdx] : null) || [];
  var editedTakes = overlaysEditedOwnsPointer(
    1 - runBlend, runBlend
  );

  var fragment = document.createDocumentFragment();
  var origOptions = tokenLayerOptions(origTokens.length, true);
  origOptions.layerClass = "token-layer-original";
  origOptions.opacity = 1 - runBlend;
  origOptions.interactive = !editedTakes;
  fragment.appendChild(
    overlaysBuildTokenLayer(origTokens, origOptions)
  );

  var editOptions = tokenLayerOptions(
    editedTokens.length, false
  );
  editOptions.layerClass = "token-layer-edited";
  editOptions.opacity = runBlend;
  editOptions.interactive = editedTakes;
  fragment.appendChild(
    overlaysBuildTokenLayer(editedTokens, editOptions)
  );
  return fragment;
}

function renderTargetPlaceholder(frameIndex) {
  outputArea.classList.remove("token-layers");
  outputArea.classList.remove("live-tokens");
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
// greyed out and non-interactive until the next Generate clears the
// lock. Either way the button carries a tooltip, explaining the lock
// when locked and what the mode does when not, matching What If.
function updateEditFramesLock() {
  // An edited save in flight locks too, not just a completed one:
  // confirmGuidedEdit fires the save and re-shows the buttons before
  // its async handler can set editedRunSaved, which would otherwise
  // leave a live window where a second edit could be started.
  var locked = editedRunSaved
    || (isSaving && remaskEdits.length > 0);
  if (locked) {
    setButtonLocked(
      btnEditFrames,
      "This run already has a saved edit."
      + " Generate again to edit a new run."
    );
  } else {
    setButtonUnlocked(
      btnEditFrames,
      "Remask tokens at any frame, then resume the run"
      + " from there"
    );
  }
  if (!btnWhatIf) {
    return;
  }
  // What If writes the same single saved edit per generation, so it
  // locks on the same condition as Edit Frames.
  if (locked) {
    setButtonLocked(
      btnWhatIf,
      "This run already has a saved edit."
      + " Generate again to try another branch."
    );
  } else {
    setButtonUnlocked(
      btnWhatIf,
      "Replace a token with one the model nearly"
      + " chose, then regenerate"
    );
  }
}

// The lock is both visual and behavioural: pointer-events is off in
// CSS, aria-disabled announces it, and the callers keep their own
// is-locked guard so a programmatic click still cannot slip through.
function setButtonLocked(button, title) {
  button.classList.add("is-locked");
  button.setAttribute("aria-disabled", "true");
  button.title = title;
}

function setButtonUnlocked(button, title) {
  button.classList.remove("is-locked");
  button.removeAttribute("aria-disabled");
  button.title = title;
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
  // What If needs captured candidates to substitute from, so it stays
  // hidden when the run was generated with Alternatives off.
  if (btnWhatIf) {
    btnWhatIf.hidden = !(
      supportsSubstitution() && alternativesAvailable()
    );
  }
  updateEditFramesLock();
  overlayMode = "none";
  resetDiffOverlay();
  resetRunBlend();
  buildOverlaySelect();
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = false;
  }
  setOverlayDrawerOpen(false);
  updateDiffSummary();
  updateDiffOverlayControls();
  updateRunBlendControls();
  guidedEditControls.hidden = true;
  clearRemaskedPositions();
  unlockScrubberNav();

  navigateToFrame(currentScrubFrame);
  updateEntropyProfileVisibility();
}

function deactivateScrubber() {
  scrubberActive = false;
  scrubberSection.hidden = true;
  guidedEditControls.hidden = true;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }
  if (entropyProfileRow) {
    entropyProfileRow.hidden = true;
  }
  entropyHoverPos = null;
  clearTokenHighlight();
  hideAltsPopover();
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
  // The token spans were just replaced, so any open popover now
  // points at a detached element.
  hideAltsPopover();
  if (scrubberActive) {
    updateEntropyProfileVisibility();
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
  substitutionMode = false;
  hideAltsPopover();
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
    frameRevealed: frameRevealed.slice(),
    resumeElapsedOffset: resumeElapsedOffset,
    positionAlts: positionAlts.slice(),
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
  frameRevealed = preEditSnapshot.frameRevealed.slice();
  resumeElapsedOffset = preEditSnapshot.resumeElapsedOffset;
  positionAlts = preEditSnapshot.positionAlts.slice();
  lastFinalText = preEditSnapshot.finalText;
  // Drop any edits committed during this (now-cancelled) session.
  remaskEdits.length = Math.min(
    remaskEdits.length, preEditSnapshot.remaskEditsLen
  );
  invalidateRunMemos();
  preEditSnapshot = null;
}

// Cut every per-frame array back to `offset` so the branch about to
// be generated appends cleanly at that index. perFrameElapsed is cut
// with the rest: leaving it whole made the saved timing array longer
// than the frame arrays, which knocked the Timing chart's x axis out
// of step with every other chart. The elapsed value at the last kept
// frame carries forward, because the worker restarts its clock for
// the new segment.
function truncateRunArraysAt(offset) {
  resumeFrameOffset = offset;
  resumeElapsedOffset = offset > 0
    ? (perFrameElapsed[offset - 1] || 0)
    : 0;
  frameHistory.length = offset;
  frameTokens.length = offset;
  frameCanvasIndex.length = offset;
  frameMeanConf.length = offset;
  perFrameElapsed.length = offset;
  frameRevealed.length = offset;
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

// ---- What If: top-k substitution (autoregressive) ----

function supportsSubstitution() {
  return !!(
    activeModel
    && activeModel.capabilities
    && activeModel.capabilities.supports_substitution
  );
}

// Arm substitution on the completed run. Mirrors Edit Frames: the
// current run becomes the "original", auto-saved if unsaved so it
// survives the branch that replaces it.
function enterSubstitutionMode() {
  if (btnWhatIf && btnWhatIf.classList.contains("is-locked")) {
    return;
  }
  beginSubstitutionSession();
  if (!runSaved) {
    saveRun();
  }
}

// Frame index and token position are the same choice for a
// left-to-right model, so there is no frame-selection phase: the run
// opens at its final frame and every captured position is clickable.
function beginSubstitutionSession() {
  captureEditSnapshot();
  remaskMode = "substitute";
  substitutionMode = true;
  scrubberMinFrame = 0;
  remaskModeEdits = [];
  guidedResumeAction = null;
  clearRemaskedPositions();

  scrubberSlider.min = "0";
  scrubberSlider.max = String(frameHistory.length - 1);
  btnEditFrames.hidden = true;
  if (btnWhatIf) {
    btnWhatIf.hidden = true;
  }
  guidedEditControls.hidden = false;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }

  navigateToFrame(frameHistory.length - 1);
  updateGuidedUI();
}

// Commit a substitution: truncate the run at the position, then let
// the worker regenerate from the forced token. Reuses the diffusion
// resume splice path (resumeFrameOffset + isResuming), so handleFrame
// appends the branch onto the truncation unchanged.
function doSubstitute(position, tokenId) {
  if (!substitutionMode || remaskMode !== "substitute") {
    return;
  }
  if (position < 0 || position >= frameHistory.length) {
    return;
  }
  hideAltsPopover();
  substitutionMode = false;

  // Recorded as an ordinary remask edit so the analytics Edited
  // column, the durable diff, and the saved metadata all work with
  // no schema change. For a left-to-right model the edited frame and
  // the edited position are the same index.
  remaskEdits.push({
    frame_index: position,
    token_positions: [position],
  });

  perFrameRemasked = {};
  remaskedPositions = {};

  truncateRunArraysAt(position);
  // Positions from the substituted one onward are about to be
  // resampled, so their captured candidates no longer apply.
  positionAlts.length = position;
  invalidateRunMemos();
  isResuming = true;

  remaskMode = "generating";
  updateGuidedUI();

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  // A substitution always resamples to the end of the run.
  startRunStatus(editRunLabel(position, null));

  ws.send(JSON.stringify({
    type: "substitute",
    position: position,
    token_id: tokenId,
  }));
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
  // The two blend rows share the scrubber area with the guided
  // controls, so keep them hidden whenever a run is being edited
  // (remaskMode !== null); both updates restore the right one on exit
  // once remaskMode is null again.
  updateDiffOverlayControls();
  updateRunBlendControls();

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

    case "substitute":
      guidedEditStatus.textContent =
        "Hover a token to see what the model nearly"
        + " chose, then click a candidate to"
        + " regenerate from it.";
      btnClearGuided.hidden = true;
      lockScrubberNav();
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

  truncateRunArraysAt(frameIndex);
  invalidateRunMemos();
  isResuming = true;

  remaskMode = "generating";
  updateGuidedUI();

  // One source for where the branch stops, so the message on screen
  // and the request on the wire cannot drift apart. Null means run to
  // the end, which is both the "resume to end" action and the
  // fallback when no target frame was captured.
  var resumeTarget = (
    action === "another" && guidedTargetFrame !== null
  ) ? guidedTargetFrame : null;

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  startRunStatus(editRunLabel(frameIndex, resumeTarget));

  var message = {
    type: "resume",
    frame_index: frameIndex,
    remask_positions: positions,
  };

  if (resumeTarget !== null) {
    message.max_frames = resumeTarget - frameIndex + 1;
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

// Retry: discard this session's edits and restart editing from the
// beginning. Reuses the already-saved original, so (unlike Edit
// Frames / What If) it does not trigger another save. Autoregressive
// runs re-enter substitution, whose session has no frame-selection
// phase to restart into.
function retryGuidedEdit() {
  var wasSubstitution = supportsSubstitution();
  restoreEditSnapshot();
  resetGuidedMode();
  if (wasSubstitution) {
    beginSubstitutionSession();
  } else {
    beginEditSession();
  }
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
  updateParamDefaultsButton();

  if (active) {
    deactivateScrubber();
  }
}

// Block glyphs for the optional "diffusion-style text" reveal.
var DENOISE_GLYPHS = "\u2591\u2592\u2593";

// True when the diffusion-text effect should actually animate.
// prefersReducedMotion comes from overlays.js, which every page
// loads ahead of its own script.
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

var STATUS_DOTS_MS = 400;

// How long the resolved word rests before re-diffusing in cycle mode.
// It no longer has to line up with the dots: they tick on their own
// continuous timer, in their own span, so the two animations cannot
// interfere however their periods fall.
var STATUS_CYCLE_HOLD_MS = 700;

// Diffuse the word into the chip, and in cycle mode keep re-diffusing
// it on a loop. Touches only the word's span, never the ellipsis
// beside it.
//
// Cycle mode used to suppress the dots entirely, on the grounds that
// re-diffusing the word was indicator enough. That left the ellipsis
// present on two text settings and absent on the third for no reason
// a user could see, and routing both through one text node was what
// forced the choice: rewriting the word meant rewriting the dots.
function statusWordPass(el, base, cycle) {
  denoiseReveal(el._textEl, base, function () {
    if (!cycle) {
      return;
    }
    el._cycleTimer = setTimeout(function () {
      statusWordPass(el, base, true);
    }, STATUS_CYCLE_HOLD_MS);
  });
}

// Animated "<base>..." activity text: the word reveals (once, or
// repeatedly in cycle mode) while the trailing dots run 3 -> 0 -> 1
// -> 2 -> 3 beside it, independently and without pause.
//
// Every timer lives on the element rather than on the module, for the
// same reason denoiseReveal's does: the status row runs one of these
// per chip, and a save animating its dots must not cancel a run
// animating its own.
function startStatusDots(el, base) {
  stopStatusDots(el);
  el._dotsCount = 3;
  var render = function () {
    el._dotsEl.textContent = ".".repeat(el._dotsCount);
    el._dotsCount = (el._dotsCount + 1) % 4;
  };
  render();
  el._dotsTimer = setInterval(render, STATUS_DOTS_MS);
  var cycle = diffusionEffectActive()
    && appSettings.diffusionTextMode === "cycle";
  statusWordPass(el, base, cycle);
}

function stopStatusDots(el) {
  cancelDenoise(el._textEl);
  if (el._dotsTimer) {
    clearInterval(el._dotsTimer);
    el._dotsTimer = null;
  }
  if (el._cycleTimer) {
    clearTimeout(el._cycleTimer);
    el._cycleTimer = null;
  }
}

// ---- Status stack ----
//
// The footer message is a single slot, so two operations running at
// once used to overwrite each other: auto-saving the pre-edit run and
// then picking a What If candidate left only "Resuming" on screen,
// with no sign the save was still in flight.
//
// The split that fixes it is by lifetime, not by category. A chip
// says only what is *happening*; where the run *stands* (Done, Saved
// to..., an error) belongs to the footer, which is also what
// saveSessionState persists.
//
// Chips deliberately do not report their own outcome. Letting them
// meant "Done" sat on top of "Done." and "Saved" on top of "Saved to
// results/...", which read as stutter, and it was the only thing that
// ever put a second line into an already crowded corner. A chip
// leaving as the footer fills in is the handoff, so silence is the
// success signal and the footer carries every word of the result.
//
// Chips are inserted directly before the resting message and so
// extend leftward from it, newest nearest the footer line, oldest
// furthest out. The row clips and fades at its left edge, against the
// gutter the footer's own gap leaves before the readouts.

// Two concurrent operations is the real ceiling (one run, and saveRun
// guards itself with isSaving), so this bound is slack. It exists so
// an unforeseen caller cannot push an unbounded run of chips out
// under the fade, where they cost layout while being unreadable.
var STATUS_STACK_MAX = 4;

// Must match the .status-chip.is-leaving transition in style.css: the
// chip is removed from the DOM only once its fade has finished.
var STATUS_CHIP_FADE_MS = 150;

// Chips currently owning a slot. A chip leaves this list the moment
// it starts to dismiss, not when its node is finally removed, so a
// late retire cannot bring a departing chip back.
var statusChips = [];

// Run `mutate`, then animate away the sideways jump it caused.
//
// Flex offers no transition for "the item beside me changed width",
// so a chip arriving, a chip's node finally leaving, or the resting
// message growing all snap their neighbours across instantly. This is
// the standard first-last-invert-play: measure, mutate, hand each
// moved chip its former position as a transform, then release it so
// the CSS transition carries it home.
//
// The offset goes on `transform` while the chips' own entrances and
// exits use the `translate` longhand, so a chip can be sliding
// sideways and rising at the same time without either being lost.
function statusRowReflow(mutate) {
  if (!statusStack || prefersReducedMotion()) {
    mutate();
    return;
  }
  // Read from the DOM rather than statusChips: a chip that is midway
  // through its fade has already left that list but still holds row
  // width, and it is the one most likely to be shoved, since the
  // resting line usually fills in as it goes.
  var moved = Array.prototype.slice.call(
    statusStack.querySelectorAll(".status-chip")
  );
  var before = moved.map(function (chip) {
    return chip.getBoundingClientRect().left;
  });
  mutate();
  for (var i = 0; i < moved.length; i++) {
    var chip = moved[i];
    var delta =
      before[i] - chip.getBoundingClientRect().left;
    if (delta === 0) {
      continue;
    }
    chip.style.transition = "none";
    chip.style.transform =
      "translateX(" + delta + "px)";
    // Commit the inverted position before the transition returns,
    // or the browser coalesces both into no visible movement.
    void chip.offsetWidth;
    chip.style.transition = "";
    chip.style.transform = "";
  }
}

// Raise a chip for an operation that has just started, and hand back
// the handle its caller retires when the operation ends.
function statusPush(text) {
  if (!statusStack) {
    return null;
  }
  var chip = document.createElement("span");
  chip.className = "status-chip";
  // The word and the ellipsis get separate spans so each can animate
  // without rewriting the other, and so the dots occupy a slot sized
  // in CSS instead of being padded out with spaces. The chip is then
  // one fixed width for its whole life, which a right-anchored row
  // needs: any width change here shoves every chip to its left.
  chip._textEl = document.createElement("span");
  chip._textEl.className = "status-chip-text";
  chip._dotsEl = document.createElement("span");
  chip._dotsEl.className = "status-chip-dots";
  chip.appendChild(chip._textEl);
  chip.appendChild(chip._dotsEl);
  // Wrapped so the chips already up slide aside rather than jumping.
  // The new chip is outside the snapshot either way (it is not in the
  // DOM yet), which is right: it has its own entrance.
  statusRowReflow(function () {
    statusStack.insertBefore(chip, statusMessage);
    statusChips.push(chip);
  });
  statusStackTrim();
  startStatusDots(chip, text);
  // Force a reflow so the browser has the hidden state to animate
  // from; without it the chip is painted visible from the start.
  void chip.offsetWidth;
  chip.classList.add("is-visible");
  return chip;
}

// Take a chip down, because its operation finished or was superseded.
// A no-op for a handle that has already left, so a promise landing
// after its chip was retired or trimmed does nothing.
function statusRetire(chip) {
  if (!chip) {
    return;
  }
  if (statusChips.indexOf(chip) === -1) {
    return;
  }
  statusChipDismiss(chip);
}

function statusStackTrim() {
  while (statusChips.length > STATUS_STACK_MAX) {
    statusChipDismiss(statusChips[0]);
  }
}

// The run's chip. Generating, substituting, and guided resume are
// mutually exclusive and all finish in handleDone or handleError,
// which are socket handlers with no closure to carry a handle, so the
// one in flight is tracked here instead. A save's handle is a local,
// which is what lets the two coexist.
var runStatusHandle = null;

function startRunStatus(text) {
  // A retry that never got a terminal message would otherwise leave
  // its chip animating forever.
  statusRetire(runStatusHandle);
  runStatusHandle = statusPush(text);
}

// Names the stretch a resume is about to regenerate. "Resuming" said
// only that something had restarted, which reads as ambiguous next to
// a plain run; the frame range says which part of the output is being
// replaced, and that is what you are waiting to watch change. A null
// target means the branch runs to the end, which is always the case
// for a left-to-right substitution.
function editRunLabel(fromFrame, toFrame) {
  var target = toFrame === null ? "end" : String(toFrame);
  return "Running edit from frame " + fromFrame
    + " to " + target;
}

// Called from handleDone and handleError alike: the chip says nothing
// about the outcome, so both endings look the same here and the
// footer is left to draw the distinction.
function endRunStatus() {
  statusRetire(runStatusHandle);
  runStatusHandle = null;
}

// Give up the slot now, release the node after the fade. Every timer
// the chip owns is cleared here, so nothing can write to a node on
// its way out of the document.
function statusChipDismiss(chip) {
  var at = statusChips.indexOf(chip);
  if (at !== -1) {
    statusChips.splice(at, 1);
  }
  stopStatusDots(chip);
  // is-leaving rather than merely dropping is-visible: the two states
  // need different offsets, and one rule cannot serve both without
  // sending a departing chip back the way it came in.
  chip.classList.remove("is-visible");
  chip.classList.add("is-leaving");
  chip._exitTimer = setTimeout(function () {
    chip._exitTimer = null;
    if (chip.parentNode) {
      statusRowReflow(function () {
        chip.parentNode.removeChild(chip);
      });
    }
  }, STATUS_CHIP_FADE_MS);
}

function setSaveAvailable(available) {
  // Always visible; greyed out when there is nothing to save.
  btnSave.disabled = !(available && frameHistory.length > 0);
}

// Clears the footer readouts only, never the stack. doSubstitute and
// doGuidedResume both call this immediately before starting a resume,
// which is exactly when the pre-edit run's auto-save may still be in
// flight; clearing the chips here would put back the overwriting this
// stack exists to fix.
function resetStatus() {
  statusStep.textContent =
    "Step -/-";
  statusElapsed.textContent =
    "Elapsed: -";
  renderTpsFooter(null);
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
  frameRevealed = [];
  frameCanvasIndex = [];
  frameMeanConf = [];
  invalidateRunMemos();
  overlayMode = "none";
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }
  lastRunParams = null;
  lastFinalText = null;
  originalTotalFrames = 0;
  originalFrameHistory = [];
  originalFrameTokens = [];
  originalPerFrameElapsed = [];
  originalMeanConf = [];
  originalPositionAlts = [];
  positionAlts = [];
  entropyHoverPos = null;
  clearTokenHighlight();
  hideAltsPopover();
  remaskEdits = [];
  editedRunSaved = false;
  runSaved = false;
  lastSavedRunId = null;
  isResuming = false;
  resumeFrameOffset = 0;
  resumeElapsedOffset = 0;
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
  startRunStatus("Running");

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
      if (typeof tok.e === "number") {
        record.e = tok.e;
      }
      records.push(record);
    }
    out.push(records);
  }
  return out;
}

// Attach the pre-edit run's own timing, confidence, and candidate
// sets to an edited run's save. These let Analytics compare the two
// runs on every axis rather than only on token text, and each is
// omitted when the original never recorded it (older sessions, or
// Alternatives left off), so the reader can tell absent from empty.
function addOriginalRunSignals(payload) {
  if (originalPerFrameElapsed.length > 0) {
    payload.original_per_frame_elapsed =
      originalPerFrameElapsed.slice();
    payload.original_elapsed_seconds =
      originalPerFrameElapsed[
        originalPerFrameElapsed.length - 1
      ];
  }
  if (originalMeanConf.length > 0) {
    payload.original_mean_conf = originalMeanConf.slice();
  }
  var originalAlts = alternativeRecordsFrom(
    originalPositionAlts
  );
  if (originalAlts !== null) {
    payload.original_alternatives = originalAlts;
  }
}

// Project accumulated candidate sets into the persisted shape, one
// entry per token position. Returns null when nothing was captured,
// so the run simply omits alternatives.json.
function alternativeRecordsFrom(positions) {
  if (!hasAnyAlternatives(positions)) {
    return null;
  }
  var out = [];
  for (var i = 0; i < positions.length; i++) {
    var alts = positions[i];
    if (!alts || alts.length === 0) {
      out.push(null);
      continue;
    }
    var records = [];
    for (var k = 0; k < alts.length; k++) {
      records.push({
        id: alts[k].id,
        t: alts[k].t,
        p: alts[k].p,
      });
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
  // Locks the edit entry points for the whole save, not just after it
  // succeeds, so confirming an edit cannot be immediately followed by
  // starting another one.
  updateEditFramesLock();
  if (saveCheckTimer !== null) {
    clearTimeout(saveCheckTimer);
    saveCheckTimer = null;
  }
  // Captured now so the async success handler locks Edit Frames only
  // when the saved run actually carried edits, and so the messages
  // below name which of the two runs is being written. That is worth
  // stating: entering What If or Edit Frames triggers a save of the
  // pre-edit run on your behalf, and an unlabelled "Saving run" gives
  // no clue that is what you are watching.
  var wasEdited = remaskEdits.length > 0;
  var runLabel = wasEdited ? "edited" : "original";

  btnSave.classList.remove("is-saved");
  btnSave.classList.add("is-saving");
  // A local, so this chip survives a run starting underneath it. That
  // is the whole point: entering What If auto-saves the pre-edit run,
  // and picking a candidate before the POST lands used to overwrite
  // this with the resume's message.
  var saveStatus = statusPush("Saving " + runLabel + " run");

  var totalElapsed = perFrameElapsed.length > 0
    ? perFrameElapsed[perFrameElapsed.length - 1]
    : null;

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

  var altRecords = alternativeRecordsFrom(positionAlts);
  if (altRecords !== null) {
    payload.alternatives = altRecords;
  }

  if (remaskEdits.length > 0) {
    payload.remask_edits = remaskEdits;
    // Persist the pre-edit snapshot so the counterfactual diff is
    // reviewable post-hoc (only meaningful for edited runs).
    if (originalFrameTokens.length > 0) {
      payload.original_frame_tokens =
        tokenRecordsFrom(originalFrameTokens);
    }
    addOriginalRunSignals(payload);
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
      // Releases the in-flight lock; the success branch below
      // re-applies it permanently once editedRunSaved is set.
      updateEditFramesLock();
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
        statusRetire(saveStatus);
        // The longest line the row ever shows, arriving while the
        // save's chip is still on screen. Easing it is what keeps the
        // chip from being flung left in a single frame.
        statusRowReflow(function () {
          statusMessage.textContent =
            "Saved " + runLabel + " run to "
            + result.path;
        });
        statusMessage.style.color =
          "var(--accent)";
        // Persist LAST, so the session captures the final run id and
        // the "Saved ... to ..." line rather than a stale run id, and
        // survives a round-trip to Analytics. The in-flight text is
        // never at risk here: it lives on a chip, not in the footer.
        saveSessionState();
      } else {
        btnSave.disabled = false;
        statusRetire(saveStatus);
        statusRowReflow(function () {
          statusMessage.textContent =
            "Save failed: "
            + (result.message || "unknown");
        });
        statusMessage.style.color =
          "var(--danger)";
      }
    })
    .catch(function (error) {
      isSaving = false;
      btnSave.classList.remove("is-saving", "is-saved");
      setSavingControls(false);
      updateGuidedUI();
      updateEditFramesLock();
      btnSave.disabled = false;
      statusRetire(saveStatus);
      statusRowReflow(function () {
        statusMessage.textContent =
          "Save failed: " + error.message;
      });
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
toggleExperimental.addEventListener(
  "change", onParamFormChanged
);

if (btnParamDefaults) {
  btnParamDefaults.addEventListener(
    "click", resetParamsToDefaults
  );
}

// The prompt's only other listener is Enter-to-generate, so the draft
// needs its own to reach the session snapshot as it is typed.
promptInput.addEventListener("input", onParamFormChanged);

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

// The shared helper owns the handle click as well as the drag, so
// this binds none of its own (see overlaysMakeDrawerDraggable).
overlaysMakeDrawerDraggable({
  group: overlaySelectGroup,
  handle: overlayDrawerHandle,
  container: document.getElementById("output-section"),
  storageKey: "diffusion_overlay_drawer_top_generator",
  onToggle: setOverlayDrawerOpen,
});

if (overlayHighlightCheckbox) {
  overlayHighlightCheckbox.addEventListener(
    "change", onOverlayHighlightToggle
  );
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
      ".token-layer-original"
    );
    if (layer) {
      layer.style.opacity = String(diffOriginalOpacity / 100);
    }
    applyDiffLayerPointers();
  });
}

if (diffEditedSlider) {
  diffEditedSlider.addEventListener("input", function () {
    diffEditedOpacity = parseInt(
      diffEditedSlider.value, 10
    );
    var layer = outputArea.querySelector(
      ".token-layer-edited"
    );
    if (layer) {
      layer.style.opacity = String(diffEditedOpacity / 100);
    }
    applyDiffLayerPointers();
  });
}

// Hand the pointer to whichever layer a drag just made the more
// opaque, so hover and the candidate popover follow the layer the
// user is reading.
function applyDiffLayerPointers() {
  overlaysApplyLayerPointers(
    outputArea, diffOriginalOpacity, diffEditedOpacity
  );
}

if (diffBlendToggle) {
  diffBlendToggle.addEventListener("change", function () {
    diffBlend = diffBlendToggle.checked;
    if (scrubberActive && overlayMode === "diff") {
      renderFrameWithTokens(currentScrubFrame);
    }
  });
}

if (runBlendInput) {
  runBlendInput.addEventListener("input", onRunBlendInput);
}

// Guided edit mode event listeners.
btnEditFrames.addEventListener(
  "click", enterRemaskMode
);

if (btnWhatIf) {
  btnWhatIf.addEventListener(
    "click", enterSubstitutionMode
  );
}

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

outputArea.addEventListener(
  "animationend", onTokenBirthEnd
);

statusTps.addEventListener("click", toggleTpsMode);
statusTps.addEventListener("keydown", function (e) {
  // It is exposed as a button, so it owes the keyboard the two keys
  // a real button answers to.
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    toggleTpsMode();
  }
});

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

// Token hover, delegated like the click handler above. Drives two
// things: the entropy profile's glowing column, which follows every
// token, and the candidate popover, which is suppressed during guided
// remask editing so it never covers the tokens being selected.
outputArea.addEventListener(
  "mouseover",
  function (e) {
    var target = e.target;
    var pos = hoveredTokenPosition(target);
    setEntropyHoverPosition(pos);
    if (pos === null || !scrubberActive || !altsPopover) {
      return;
    }
    if (remaskMode !== null && !substitutionMode) {
      return;
    }
    if (pos === altsPopoverPos) {
      return;
    }
    showAltsPopover(pos, target);
  }
);

// The profile's own hover, the mirror of the handler above: moving
// along the columns lights both the column and the token it belongs
// to. A direct token hover reaches the same look through CSS
// (.token-hover-highlight), so the two directions match without
// this having to touch the class the pointer already applies.
if (entropyProfileCanvas) {
  entropyProfileCanvas.addEventListener(
    "mousemove",
    function (e) {
      var pos = entropyProfilePosition(e);
      setEntropyHoverPosition(pos);
      setTokenHighlight(pos);
    }
  );
  entropyProfileCanvas.addEventListener(
    "mouseleave",
    function () {
      setEntropyHoverPosition(null);
      setTokenHighlight(null);
    }
  );
}

// The token position an event target represents, or null when the
// pointer is over the output area's padding rather than a token.
function hoveredTokenPosition(target) {
  if (!target.classList || !target.classList.contains("token-span")) {
    return null;
  }
  var raw = target.getAttribute("data-pos");
  if (raw === null) {
    return null;
  }
  return parseInt(raw, 10);
}

outputArea.addEventListener(
  "mouseleave",
  function () {
    // Keep the popover open, and its position glowing, while the
    // pointer is inside the popover itself: it sits above the token,
    // so reaching a candidate to click means leaving the output area.
    if (altsPopover && altsPopover.matches(":hover")) {
      return;
    }
    setEntropyHoverPosition(null);
    hideAltsPopover();
  }
);

if (altsPopover) {
  altsPopover.addEventListener("mouseleave", function () {
    setEntropyHoverPosition(null);
    hideAltsPopover();
  });
  // Picking a candidate commits the substitution. Only armed in What
  // If mode; the popover is read-only otherwise.
  altsPopover.addEventListener("click", function (e) {
    if (!substitutionMode || altsPopoverPos === null) {
      return;
    }
    // The Original page shows the pre-edit run's candidates. There is
    // nothing to substitute into on that side, and the worker only
    // holds the live run's state anyway.
    if (altsPopoverPage === "original") {
      return;
    }
    var row = e.target.closest(".alt-row");
    if (!row) {
      return;
    }
    var raw = row.getAttribute("data-alt-id");
    if (raw === null) {
      return;
    }
    doSubstitute(altsPopoverPos, parseInt(raw, 10));
  });
}

// A scroll moves the anchoring token out from under a fixed popover.
window.addEventListener(
  "scroll",
  function () {
    if (altsPopoverPos !== null) {
      hideAltsPopover();
    }
  },
  true
);

window.addEventListener("resize", function () {
  hideAltsPopover();
  if (scrubberActive) {
    updateEntropyProfileVisibility();
  }
});

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
      || remaskMode === "substitute"
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

// Shared with the menu, which clears the same snapshot when it
// activates a model (see overlaysClearLastRun for why both pages do).
var SESSION_KEY = OVERLAYS_LAST_RUN_KEY;

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
    // Part of the snapshot's identity, not decoration. The same model
    // on the other device is a different worker with its own output,
    // and every activation path (the header selector, the menu) ends
    // in a reload, so this is the only thing that tells a CPU/GPU
    // switch apart from a page navigation.
    device: activeDevice,
    prompt: promptInput.value,
    frameHistory: frameHistory,
    perFrameElapsed: perFrameElapsed,
    // Carried so the footer can still answer a mode switch after a
    // round trip. Restoring only the rendered text would leave the
    // first click with nothing to recompute from.
    frameRevealed: frameRevealed,
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
    originalPerFrameElapsed: originalPerFrameElapsed,
    originalMeanConf: originalMeanConf,
    originalPositionAlts: originalPositionAlts,
    positionAlts: positionAlts,
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
  overlaysClearLastRun();
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
  if (!s) {
    return false;
  }
  // Snapshots written before the device joined the identity have no
  // `device` key at all. Treating that as a mismatch would silently
  // drop one in-flight run per upgrade, so it is read as "matches",
  // and the clear-on-switch covers the case it cannot.
  var sameDevice =
    s.device === undefined || s.device === activeDevice;
  if (
    s.model !== activeModelId
    || !sameDevice
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
  invalidateRunMemos();
  perFrameElapsed = s.perFrameElapsed || [];
  frameRevealed = s.frameRevealed || [];
  lastFinalText = s.finalText || "";
  lastRunParams = s.params || null;
  remaskEdits = s.remaskEdits || [];
  originalTotalFrames =
    s.originalTotalFrames || frameHistory.length;
  originalFrameHistory = s.originalFrameHistory || [];
  originalFrameTokens = s.originalFrameTokens || [];
  originalPerFrameElapsed = s.originalPerFrameElapsed || [];
  originalMeanConf = s.originalMeanConf || [];
  originalPositionAlts = s.originalPositionAlts || [];
  positionAlts = s.positionAlts || [];
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
  // Recomputed rather than replayed from stored text, so it honors
  // the mode in effect now: the setting is global and may have been
  // switched on another page since this run finished.
  renderTpsFooter(currentTokensPerSecond());
  if (s.statusMessage) {
    statusMessage.textContent = s.statusMessage;
  }
  return true;
}

// ---- Session-scoped form state (params + prompt draft) ----
//
// Leaving the page and coming back used to reset every
// hyperparameter, because boot() rebuilds the panel from specDefault
// and nothing put the user's values back. This restores them for the
// life of the app.
//
// It cannot ride in SESSION_KEY: saveSessionState bails unless a run
// completed, and clearSessionState fires at the *start* of every
// generate, so params would be wiped by Generate and lost entirely if
// you navigated mid-run. This is form state, not a run artifact, and
// it gets its own key with its own lifetime.
//
// Keyed by model id, because param_specs differ per model and a model
// switch ends in a location.reload() that sessionStorage survives, so
// each model keeps its own values. Deliberately not in PERSIST_KEYS:
// it is meant to die with the app, since a fresh launch should start
// from the recommended defaults.

var PARAM_STATE_KEY = "diffusion_param_state";

function readParamStateAll() {
  var raw = null;
  try {
    raw = sessionStorage.getItem(PARAM_STATE_KEY);
  } catch (_e) {
    return {};
  }
  if (!raw) {
    return {};
  }
  try {
    var parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch (_e) {
    // Corrupt storage: fall back to the defaults.
  }
  return {};
}

// Raw control values rather than getParamValues() output, so a
// half-typed entry round-trips instead of being silently rewritten by
// a parseFloat. validateAllParams does its usual job on the way back.
function currentParamRawValues() {
  var out = {};
  var names = Object.keys(paramInputs);
  for (var i = 0; i < names.length; i++) {
    var input = paramInputs[names[i]];
    if (input.type === "checkbox") {
      out[names[i]] = input.checked;
    } else {
      out[names[i]] = input.value;
    }
  }
  return out;
}

function saveParamState() {
  if (!activeModelId) {
    return;
  }
  var all = readParamStateAll();
  all[activeModelId] = {
    experimental: toggleExperimental.checked,
    params: currentParamRawValues(),
    prompt: promptInput.value,
  };
  try {
    sessionStorage.setItem(
      PARAM_STATE_KEY, JSON.stringify(all)
    );
  } catch (_e) {
    // Quota or private mode: the form simply will not persist.
  }
}

// Every path that mutates the form funnels through here, so the
// stored snapshot and the Reset button's enabled state cannot drift
// apart.
function onParamFormChanged() {
  saveParamState();
  updateParamDefaultsButton();
}

function restoreParamState() {
  if (!activeModelId) {
    return;
  }
  var state = readParamStateAll()[activeModelId];
  if (!state) {
    return;
  }
  // Experimental goes first: specDefault and specBounds both read it,
  // so the values have to land against the right set of bounds.
  toggleExperimental.checked = !!state.experimental;
  applyParamRawValues(state.params);
  // Clamps the restored values, refreshes the range tooltips, and
  // validates, all of which the new bounds require. This also covers
  // the awkward case of the device changing between save and restore,
  // where a stored value can fall outside the override's range.
  applyLimits();
  if (state.prompt) {
    promptInput.value = state.prompt;
  }
}

// Restore by spec name so a spec set that changed between sessions
// degrades instead of breaking: stored names the model no longer has
// are ignored, and specs with nothing stored keep the default that
// buildParamPanel already applied.
function applyParamRawValues(values) {
  if (!values || !activeModel) {
    return;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var stored = values[specs[i].name];
    if (stored !== undefined) {
      applyParamRawValue(specs[i], stored);
    }
  }
}

// One control, one stored value. A select is checked against the
// spec's current options, since an option removed since the value was
// stored would otherwise be forwarded to the server verbatim.
function applyParamRawValue(spec, stored) {
  var input = paramInputs[spec.name];
  if (!input) {
    return;
  }
  if (spec.type === "bool") {
    input.checked = !!stored;
  } else if (spec.type === "select") {
    if ((spec.options || []).indexOf(stored) >= 0) {
      input.value = stored;
    }
  } else {
    input.value = String(stored);
  }
}

// ---- Reset to defaults ----

// Whether every control already sits at its default, which is what
// disables the Reset button. Experimental is part of the question,
// since it moves the bounds the defaults are drawn from.
function paramsAtDefaults() {
  if (toggleExperimental.checked) {
    return false;
  }
  if (!activeModel) {
    return true;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    if (!paramAtDefault(specs[i])) {
      return false;
    }
  }
  return true;
}

function paramAtDefault(spec) {
  var input = paramInputs[spec.name];
  if (!input) {
    return true;
  }
  if (spec.type === "bool") {
    return input.checked === Boolean(specDefault(spec));
  }
  if (spec.type === "select") {
    return input.value === spec.default;
  }
  // String comparison against the same String() the builder used, so
  // a retyped "0.30" reads as changed. Harmless: Reset just
  // normalizes it.
  return input.value === String(specDefault(spec));
}

function resetParamsToDefaults() {
  if (!activeModel || isGenerating) {
    return;
  }
  toggleExperimental.checked = false;
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    resetParamToDefault(specs[i]);
  }
  applyLimits();
  // Re-save rather than clearing the entry: the prompt draft lives in
  // the same record and is not the button's business.
  onParamFormChanged();
}

function resetParamToDefault(spec) {
  var input = paramInputs[spec.name];
  if (!input) {
    return;
  }
  if (spec.type === "bool") {
    input.checked = Boolean(specDefault(spec));
  } else if (spec.type === "select") {
    input.value = spec.default;
  } else {
    input.value = String(specDefault(spec));
  }
}

function updateParamDefaultsButton() {
  if (!btnParamDefaults) {
    return;
  }
  btnParamDefaults.disabled = isGenerating || paramsAtDefaults();
}

// ---- Boot ----

function boot() {
  loadSettings();
  loadPromptHistory();
  updatePromptHistoryUI();
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
        // Before restoreSessionState, so a completed run's prompt
        // still overwrites the draft with no special casing.
        restoreParamState();
        updateParamDefaultsButton();
      }
      // Needs the active model, since the glow is tuned per model
      // class. Outside the guard above because it falls back to the
      // diffusion pair, which is the right reading when the active
      // model could not be identified at all.
      applyTokenBirthGlow();
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
