// Diffusion LLM Visualizer — client-side logic.

"use strict";

// Unresolved-token glyph; set from the active model.
var MASK_CHAR = "\u2591"; // ░
var RECONNECT_DELAY_MS = 2000;
var MAX_RECONNECT_DELAY_MS = 16000;

// ---- Model registry state (from /api/models) ----

var models = {}; // id -> ModelInfo
var activeModelId = null;
var activeModel = null; // ModelInfo of the active model
var suppressReconnect = false;

// Dynamic parameter DOM, rebuilt per model from its schema.
var paramInputs = {}; // name -> input/select element
var paramTooltips = {}; // name -> tooltip span

// ---- DOM refs ----

var promptInput =
  document.getElementById("prompt-input");
var btnGenerate =
  document.getElementById("btn-generate");
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

// Settings DOM refs.
var linkSettings =
  document.getElementById("link-settings");
var modalSettings =
  document.getElementById("modal-settings");
var idleDisplayMount =
  document.getElementById("idle-display-mount");
var selectIdleDisplay = null;
var settingHighlightCb =
  document.getElementById("setting-highlight-tokens");
var settingCommitCb =
  document.getElementById("setting-commit-order");
var btnSettingsSave =
  document.getElementById("btn-settings-save");
var btnSettingsReset =
  document.getElementById("btn-settings-reset");
var settingsStatus =
  document.getElementById("settings-status");
var settingsStatusTimer = null;
var statusHighlight =
  document.getElementById("status-highlight");
var statusCommitText =
  document.getElementById("status-commit-text");
// Persistent UI preferences (localStorage-backed). appSettings is
// the applied/saved state; stagedSettings is the modal's working
// copy, committed to appSettings only via the Save button.
var DEFAULT_SETTINGS = {
  idleDisplay: "default",
  highlightTokens: false,
  commitOrder: false,
};
var appSettings = {
  idleDisplay: "default",
  highlightTokens: false,
  commitOrder: false,
};
var stagedSettings = {
  idleDisplay: "default",
  highlightTokens: false,
  commitOrder: false,
};

// Scrubber DOM refs.
var scrubberSection =
  document.getElementById("scrubber-section");
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
var btnExitEdit =
  document.getElementById("btn-exit-edit");

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
//      | "select_target" | "generating"
var remaskMode = null;
var remaskModeEdits = [];
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

// Idle animation state.
var idleDisplayMode = "default";
var donutTimer = null;
var donutA = 1;
var donutB = 1;
var hasEverGenerated = false;

// ---- Idle ASCII donut (donut.c) ----

var DONUT_COLS = 80;
var DONUT_ROWS = 22;
var DONUT_LUMINANCE = ".,-~:;=!*#$@";

function renderDonutFrame() {
  var size = DONUT_COLS * DONUT_ROWS;
  var buffer = new Array(size);
  var zbuffer = new Array(size);

  donutA += 0.07;
  donutB += 0.03;

  var cosA = Math.cos(donutA);
  var sinA = Math.sin(donutA);
  var cosB = Math.cos(donutB);
  var sinB = Math.sin(donutB);

  for (var k = 0; k < size; k++) {
    buffer[k] = " ";
    zbuffer[k] = 0;
  }

  for (
    var theta = 0;
    theta < 6.28;
    theta += 0.07
  ) {
    var cosTheta = Math.cos(theta);
    var sinTheta = Math.sin(theta);

    for (
      var phi = 0;
      phi < 6.28;
      phi += 0.02
    ) {
      var sinPhi = Math.sin(phi);
      var cosPhi = Math.cos(phi);
      var circleX = cosTheta + 2;
      var oneOverZ = 1 / (
        sinPhi * circleX * sinA
        + sinTheta * cosA + 5
      );
      var t = (
        sinPhi * circleX * cosA
        - sinTheta * sinA
      );

      var xp = 0 | (
        40 + 30 * oneOverZ
        * (cosPhi * circleX * cosB - t * sinB)
      );
      var yp = 0 | (
        11 + 15 * oneOverZ
        * (cosPhi * circleX * sinB + t * cosB)
      );
      var idx = xp + DONUT_COLS * yp;
      var luminance = 0 | (8 * (
        (
          sinTheta * sinA
          - sinPhi * cosTheta * cosA
        ) * cosB
        - sinPhi * cosTheta * sinA
        - sinTheta * cosA
        - cosPhi * cosTheta * sinB
      ));

      if (
        yp >= 0 && yp < DONUT_ROWS
        && xp >= 0 && xp < DONUT_COLS
        && oneOverZ > zbuffer[idx]
      ) {
        zbuffer[idx] = oneOverZ;
        buffer[idx] = DONUT_LUMINANCE[
          luminance > 0 ? luminance : 0
        ];
      }
    }
  }

  var lines = [];
  for (var row = 0; row < DONUT_ROWS; row++) {
    var start = row * DONUT_COLS;
    lines.push(
      buffer
        .slice(start, start + DONUT_COLS)
        .join("")
    );
  }
  return lines.join("\n");
}

function startDonut() {
  if (donutTimer !== null || hasEverGenerated) {
    return;
  }

  var pre = document.createElement("pre");
  pre.id = "donut-pre";
  outputArea.textContent = "";
  outputArea.appendChild(pre);
  outputArea.classList.add("donut-active");

  pre.textContent = renderDonutFrame();

  donutTimer = setInterval(function () {
    pre.textContent = renderDonutFrame();
  }, 50);
}

function stopDonut() {
  if (donutTimer === null) {
    return;
  }
  clearInterval(donutTimer);
  donutTimer = null;
  outputArea.classList.remove("donut-active");
  var pre = document.getElementById("donut-pre");
  if (pre) {
    pre.remove();
  }
}

// ---- Idle animation dispatchers ----

function startIdleAnimation() {
  if (hasEverGenerated) {
    return;
  }
  outputArea.textContent = "";
  if (idleDisplayMode === "donut") {
    startDonut();
  } else {
    window.startAsciiScene(outputArea);
  }
}

function stopIdleAnimation() {
  stopDonut();
  window.stopAsciiScene();
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

function setModelSelectValue(id) {
  if (!modelSelectValue) {
    return;
  }
  var m = models[id];
  modelSelectValue.textContent =
    m ? m.display_name : (id || "\u2014");
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
    li.textContent = m.display_name;
    modelSelectList.appendChild(li);
  }
  setModelSelectValue(activeId);
  sizeModelSelect(list);
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

function activeLimits() {
  var experimental = toggleExperimental.checked;
  var out = {};
  if (!activeModel) {
    return out;
  }
  var specs = activeModel.param_specs;
  for (var i = 0; i < specs.length; i++) {
    var s = specs[i];
    var b = experimental
      ? s.experimental
      : s.recommended;
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
    input.checked = Boolean(spec.default);
  } else {
    input = document.createElement("input");
    input.type = "number";
    if (spec.step !== null && spec.step !== undefined) {
      input.step = String(spec.step);
    }
    input.value = String(spec.default);
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
    if (modelReady && !isGenerating) {
      btnGenerate.disabled = true;
    }
  } else {
    validationHint.hidden = true;
    validationHint.textContent = "";
    paramsValid = true;
    if (modelReady && !isGenerating) {
      btnGenerate.disabled = false;
    }
  }
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

function switchModel(id) {
  if (id === activeModelId) {
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

  fetch(
    "/api/models/" + encodeURIComponent(id) + "/activate",
    { method: "POST" }
  )
    .then(function (r) {
      return r.json();
    })
    .then(function (res) {
      if (res.ok) {
        location.reload();
      } else {
        throw new Error(
          res.message || "activation failed"
        );
      }
    })
    .catch(function (err) {
      suppressReconnect = false;
      setModelSelectDisabled(false);
      setModelSelectValue(activeModelId);
      loadingOverlay.classList.add("hidden");
      statusMessage.textContent =
        "Model switch failed: " + err.message;
      statusMessage.style.color = "var(--danger)";
    });
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
    btnGenerate.disabled = true;
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
    btnGenerate.disabled = true;
  } else if (data.status === "ready") {
    setBadge("ready");
    modelReady = true;
    loadingOverlay.classList.add("hidden");
    if (paramsValid && !isGenerating) {
      btnGenerate.disabled = false;
    }
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

// Map confidence in [0,1] to a green intensity for the heatmap.
function heatColor(c) {
  var clamped = Math.max(0, Math.min(1, c));
  var sat = Math.round(35 + 55 * clamped);
  var light = Math.round(32 + 30 * clamped);
  return "hsl(135, " + sat + "%, " + light + "%)";
}

// Per-position commit step for the current run: the step after
// which a position last changed to its final value. Derived
// purely from frameTokens (the final frame is ground truth), so
// it is exact for LLaDA (resolved tokens are frozen) and a
// "settle" proxy for DiffusionGemma. Positions still unresolved
// at the last frame get -1 (left uncolored). Result is memoized
// in commitSteps and invalidated whenever frameTokens changes.
function computeCommitSteps() {
  var frameCount = frameTokens.length;
  if (frameCount === 0) {
    return [];
  }
  var finalTokens = frameTokens[frameCount - 1];
  if (!finalTokens) {
    return [];
  }
  var width = finalTokens.length;
  var steps = new Array(width);
  for (var i = 0; i < width; i++) {
    var finalTok = finalTokens[i];
    if (!finalTok || finalTok.m) {
      steps[i] = -1;
      continue;
    }
    var finalId = finalTok.id;
    var settle = 0;
    for (var f = 0; f < frameCount; f++) {
      var ft = frameTokens[f];
      if (!ft || i >= ft.length) {
        continue;
      }
      var tk = ft[i];
      if (!tk || tk.id !== finalId) {
        settle = f + 1;
      }
    }
    steps[i] = settle;
  }
  return steps;
}

// Map a commit step to an early->late hue: early settles read
// light green, late settles read red-orange. maxStep normalizes to
// the run length so the scale means "early vs late in the run".
function commitColor(step, maxStep) {
  var frac = maxStep > 0 ? step / maxStep : 0;
  frac = Math.max(0, Math.min(1, frac));
  var hue = Math.round(130 - 115 * frac);
  var sat = Math.round(60 + 22 * frac);
  var light = Math.round(62 - 10 * frac);
  return "hsl(" + hue + ", " + sat + "%, " + light + "%)";
}

// Compare the branch's final frame against the retained original
// run's final frame, position-aligned on the shared canvas. Returns
// per-position change flags, the original display text (for
// tooltips), the remask-origin positions, and a divergence summary.
function computeDiff() {
  var result = {
    changed: [],
    origText: [],
    origins: {},
    changedCount: 0,
    totalCount: 0,
  };
  var cur = frameTokens.length
    ? frameTokens[frameTokens.length - 1]
    : null;
  var orig = originalFrameTokens.length
    ? originalFrameTokens[originalFrameTokens.length - 1]
    : null;
  if (!cur || !orig) {
    return result;
  }
  for (var e = 0; e < remaskEdits.length; e++) {
    var positions = remaskEdits[e].token_positions || [];
    for (var p = 0; p < positions.length; p++) {
      result.origins[positions[p]] = true;
    }
  }
  var width = Math.min(cur.length, orig.length);
  for (var i = 0; i < width; i++) {
    var c = cur[i];
    var o = orig[i];
    var cResolved = !!c && !c.m;
    var oResolved = !!o && !o.m;
    var changed = false;
    if (cResolved && oResolved) {
      result.totalCount++;
      changed = c.id !== o.id;
    } else if (cResolved !== oResolved) {
      result.totalCount++;
      changed = true;
    }
    if (changed) {
      result.changedCount++;
    }
    result.changed[i] = changed;
    result.origText[i] = o ? (o.m ? MASK_CHAR : o.t) : "";
  }
  return result;
}

// Divergence coloring: changed tokens glow magenta, unchanged
// tokens fade to a dim neutral so the intervention's footprint
// stands out.
function diffColor(changed) {
  if (changed) {
    return "hsl(320, 80%, 66%)";
  }
  return "hsl(0, 0%, 45%)";
}

// Palette for the diff-overlay layers. The original layer is cyan in
// ghost mode (blend off); with "Difference blend" on it adopts the
// edited layer's diff colors so matching tokens cancel to black.
function diffLayerColor(index, isOriginal) {
  if (isOriginal && !diffBlend) {
    return "#2dd4ff";
  }
  if (diffData && diffData.origins[index]) {
    return "#ff8a3d";
  }
  if (diffData && diffData.changed[index]) {
    return "hsl(320, 80%, 66%)";
  }
  return "#e6e6e6";
}

function buildDiffLayerSpans(tokens, isOriginal) {
  var frag = document.createDocumentFragment();
  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    if (!tok) { continue; }
    var span = document.createElement("span");
    if (tok.m) {
      span.textContent = MASK_CHAR;
      span.style.color = "var(--mask-color)";
    } else {
      span.textContent = tok.t;
      span.style.color = diffLayerColor(i, isOriginal);
    }
    frag.appendChild(span);
  }
  return frag;
}

// Draw the original and edited runs at the current frame as two
// stacked layers (independent opacity + optional difference blend)
// so overlaps and divergences can be compared directly.
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

  var origLayer = document.createElement("div");
  origLayer.className = "diff-layer diff-layer-original";
  origLayer.style.opacity = String(diffOriginalOpacity / 100);
  origLayer.appendChild(
    buildDiffLayerSpans(origTokens, true)
  );

  var editLayer = document.createElement("div");
  editLayer.className = "diff-layer diff-layer-edited";
  editLayer.style.opacity = String(diffEditedOpacity / 100);
  if (diffBlend) {
    editLayer.style.mixBlendMode = "difference";
  }
  editLayer.appendChild(
    buildDiffLayerSpans(editedTokens, false)
  );

  outputArea.textContent = "";
  outputArea.classList.add("diff-overlay-mode");
  outputArea.appendChild(origLayer);
  outputArea.appendChild(editLayer);
}

// Which coloring actually paints tokens: an explicit overlay
// selection (heatmap/diff) wins; otherwise the Commit Order
// preference applies as an ambient tint; otherwise none.
function effectiveColorMode() {
  if (overlayMode === "conf" || overlayMode === "diff") {
    return overlayMode;
  }
  if (appSettings.commitOrder) {
    return "commit";
  }
  return "none";
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
  if (scrubberActive) {
    renderFrameWithTokens(currentScrubFrame);
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
    {
      value: "diff",
      label: "Diff vs Original",
      disabled: !hasDiff,
      title: hasDiff
        ? undefined
        : "Edit and resume a run (via Edit Frames) to"
          + " compare it against the original.",
    },
  ];
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

// ---- Persistent UI settings (localStorage) ----

var SETTINGS_KEY = "diffusion_settings";

// Read one string setting from a parsed object, falling back to the
// default when absent or of the wrong type.
function _readStr(parsed, key, fallback) {
  var value = parsed[key];
  return typeof value === "string" ? value : fallback;
}

function loadSettings() {
  try {
    var raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) {
      return;
    }
    var parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      appSettings.idleDisplay = _readStr(
        parsed, "idleDisplay", DEFAULT_SETTINGS.idleDisplay
      );
      appSettings.highlightTokens = !!parsed.highlightTokens;
      appSettings.commitOrder = !!parsed.commitOrder;
    }
  } catch (_e) {
    // Unavailable or corrupt storage: keep defaults. Note that
    // clearing browser site data removes this key, reverting to
    // the defaults on next load.
  }
}

function saveSettings() {
  try {
    localStorage.setItem(
      SETTINGS_KEY, JSON.stringify(appSettings)
    );
  } catch (_e) {
    // Ignore quota/availability errors; just won't persist.
  }
}

function settingsEqual(a, b) {
  return (
    a.idleDisplay === b.idleDisplay
    && a.highlightTokens === b.highlightTokens
    && a.commitOrder === b.commitOrder
  );
}

// Reflect the applied settings in the status bar and legend.
function updateStatusPrefs() {
  if (statusHighlight) {
    statusHighlight.textContent =
      "Highlighted Tokens: "
      + (appSettings.highlightTokens ? "On" : "Off");
  }
  if (statusCommitText) {
    statusCommitText.textContent =
      "Show Commit Order: "
      + (appSettings.commitOrder ? "On" : "Off");
  }
  if (commitLegend) {
    commitLegend.hidden = !appSettings.commitOrder;
  }
}

// Apply the (saved) settings to the live app: status bar, hover
// highlight, idle animation choice, and any active token coloring.
function applySettings() {
  updateStatusPrefs();
  updateHoverHighlight();
  if (idleDisplayMode !== appSettings.idleDisplay) {
    stopIdleAnimation();
    idleDisplayMode = appSettings.idleDisplay;
    startIdleAnimation();
  }
  if (scrubberActive) {
    renderFrameWithTokens(currentScrubFrame);
  }
}

// Mirror the staged (in-modal) settings into the modal controls.
function syncSettingsControls() {
  if (settingHighlightCb) {
    settingHighlightCb.checked = stagedSettings.highlightTokens;
  }
  if (settingCommitCb) {
    settingCommitCb.checked = stagedSettings.commitOrder;
  }
  if (selectIdleDisplay) {
    selectIdleDisplay.value = stagedSettings.idleDisplay;
  }
}

// Save is enabled only when there are pending changes; Reset only
// when the staged settings differ from the fresh defaults.
function updateSettingsButtons() {
  if (btnSettingsSave) {
    btnSettingsSave.disabled = settingsEqual(
      stagedSettings, appSettings
    );
  }
  if (btnSettingsReset) {
    btnSettingsReset.disabled = settingsEqual(
      stagedSettings, DEFAULT_SETTINGS
    );
  }
}

// Small save-feedback line in the settings footer. Pass "" to hide.
function setSettingsStatus(text, saved) {
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

// Token-level rendering for scrubber mode.
// Each token is a clickable span; resolved tokens
// can be clicked to toggle remasking.
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
      span.className =
        "token-span token-remasked";
      span.textContent = MASK_CHAR;
      span.title = tline + "Confidence: 0";
    } else if (tok.m) {
      span.className = "token-span token-mask";
      span.textContent = MASK_CHAR;
      span.title = tline + "Confidence: 0";
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
      " \u2014 will be generated. "
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

// ---- Guided multi-frame edit mode ----

function resetGuidedMode() {
  remaskMode = null;
  guidedResumeAction = null;
  guidedTargetFrame = null;
  remaskModeEdits = [];
  preEditSnapshot = null;
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

function enterRemaskMode() {
  captureEditSnapshot();
  remaskMode = "select";
  scrubberMinFrame = 0;
  remaskModeEdits = [];
  guidedResumeAction = null;
  clearRemaskedPositions();

  scrubberSlider.min = "0";
  btnEditFrames.hidden = true;
  guidedEditControls.hidden = false;
  if (overlaySelectGroup) {
    overlaySelectGroup.hidden = true;
  }

  navigateToFrame(0);
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
  btnLockIn.hidden = true;
  btnClearGuided.hidden = true;
  btnEditAnother.hidden = true;
  btnRunToHere.hidden = true;
  btnResumeEnd.hidden = true;

  if (remaskMode === null) {
    guidedEditControls.hidden = true;
    return;
  }

  guidedEditControls.hidden = false;

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
        + " \u2014 click tokens to remask ("
        + count + " selected)." + renoiseNote();
      btnLockIn.hidden = false;
      btnLockIn.disabled = count === 0;
      btnClearGuided.hidden = false;
      btnClearGuided.disabled = count === 0;
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
    resetGuidedMode();
    activateScrubber();
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
  // (and whenever the model is not ready or params are invalid).
  btnGenerate.disabled = active || !(modelReady && paramsValid);
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

function startStatusDots(base) {
  stopStatusDots();
  statusDotsCount = 3;
  statusMessage.style.color = "";
  var render = function () {
    var dots = ".".repeat(statusDotsCount);
    var pad = "\u00A0".repeat(3 - statusDotsCount);
    statusMessage.textContent = base + dots + pad;
    statusDotsCount = (statusDotsCount + 1) % 4;
  };
  render();
  statusDotsTimer = setInterval(render, 400);
}

function stopStatusDots() {
  if (statusDotsTimer !== null) {
    clearInterval(statusDotsTimer);
    statusDotsTimer = null;
  }
}

function setSaveAvailable(available) {
  // Always visible; greyed out when there is nothing to save.
  btnSave.disabled = !(available && frameHistory.length > 0);
}

function resetStatus() {
  statusStep.textContent =
    "Step \u2014/\u2014";
  statusElapsed.textContent =
    "Elapsed: \u2014";
  statusMessage.textContent = "";
  statusMessage.style.color = "";
}

// ---- Actions ----

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

  if (!hasEverGenerated) {
    hasEverGenerated = true;
    stopIdleAnimation();
  }

  // A fresh run abandons any in-progress edit session so its
  // controls do not linger once the new run completes.
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
  perFrameRemasked = {};
  resetGuidedMode();
  isResuming = false;
  resumeFrameOffset = 0;
  setSaveAvailable(false);

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
  if (saveCheckTimer !== null) {
    clearTimeout(saveCheckTimer);
    saveCheckTimer = null;
  }
  btnSave.classList.remove("is-saved");
  btnSave.classList.add("is-saving");
  statusMessage.textContent = "";
  statusMessage.style.color = "";

  var totalElapsed = perFrameElapsed.length > 0
    ? perFrameElapsed[perFrameElapsed.length - 1]
    : null;

  var tokenIds = [];
  for (var fi = 0; fi < frameTokens.length; fi++) {
    var ft = frameTokens[fi];
    if (ft) {
      var ids = [];
      for (var ti = 0; ti < ft.length; ti++) {
        ids.push(ft[ti].id);
      }
      tokenIds.push(ids);
    } else {
      tokenIds.push(null);
    }
  }

  var payload = {
    model: activeModelId,
    prompt: promptInput.value.trim(),
    params: lastRunParams || getParamValues(),
    frames: frameHistory,
    final_text: lastFinalText,
    elapsed_seconds: totalElapsed,
    per_frame_elapsed: perFrameElapsed.slice(),
    frame_token_ids: tokenIds,
    canvas_index: frameCanvasIndex.slice(),
    mean_conf: frameMeanConf.slice(),
  };

  if (remaskEdits.length > 0) {
    payload.remask_edits = remaskEdits;
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
      if (result.success) {
        // Flash a glowing check for half a second, then revert to
        // the (disabled) arrow. It stays disabled to prevent a
        // duplicate save; re-enables on the next run.
        btnSave.classList.add("is-saved");
        saveCheckTimer = setTimeout(function () {
          btnSave.classList.remove("is-saved");
          saveCheckTimer = null;
        }, 500);
        statusMessage.textContent =
          "Saved to " + result.path;
        statusMessage.style.color =
          "var(--accent)";
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
      btnSave.disabled = false;
      statusMessage.textContent =
        "Save failed: " + error.message;
      statusMessage.style.color =
        "var(--danger)";
    });
}

// ---- Event listeners ----

btnGenerate.addEventListener(
  "click", startGeneration
);
btnSave.addEventListener("click", saveRun);

promptInput.addEventListener(
  "keydown",
  function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      startGeneration();
    }
  }
);

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
    var opt = e.target.closest(".model-select-option");
    if (!opt) {
      return;
    }
    var id = opt.getAttribute("data-id");
    closeModelList();
    if (id && id !== activeModelId) {
      switchModel(id);
    }
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

// Settings toggles only stage changes; nothing applies until Save.
if (settingHighlightCb) {
  settingHighlightCb.addEventListener("change", function () {
    stagedSettings.highlightTokens = settingHighlightCb.checked;
    updateSettingsButtons();
  });
}

if (settingCommitCb) {
  settingCommitCb.addEventListener("change", function () {
    stagedSettings.commitOrder = settingCommitCb.checked;
    updateSettingsButtons();
  });
}

if (btnSettingsSave) {
  btnSettingsSave.addEventListener("click", function () {
    if (settingsEqual(stagedSettings, appSettings)) {
      return;
    }
    setSettingsStatus("Saving\u2026", false);
    appSettings.idleDisplay = stagedSettings.idleDisplay;
    appSettings.highlightTokens = stagedSettings.highlightTokens;
    appSettings.commitOrder = stagedSettings.commitOrder;
    saveSettings();
    applySettings();
    // Disable + blur the button so it visibly de-presses.
    updateSettingsButtons();
    btnSettingsSave.blur();
    settingsStatusTimer = setTimeout(function () {
      setSettingsStatus("Changes saved!", true);
      settingsStatusTimer = setTimeout(function () {
        setSettingsStatus("", false);
      }, 2400);
    }, 300);
  });
}

if (btnSettingsReset) {
  btnSettingsReset.addEventListener("click", function () {
    stagedSettings.idleDisplay = DEFAULT_SETTINGS.idleDisplay;
    stagedSettings.highlightTokens =
      DEFAULT_SETTINGS.highlightTokens;
    stagedSettings.commitOrder = DEFAULT_SETTINGS.commitOrder;
    syncSettingsControls();
    updateSettingsButtons();
    setSettingsStatus("", false);
  });
}

// Guided edit mode event listeners.
btnEditFrames.addEventListener(
  "click", enterRemaskMode
);

btnSelectFrame.addEventListener(
  "click", selectCurrentFrame
);

btnLockIn.addEventListener(
  "click", lockInEdits
);

btnClearGuided.addEventListener(
  "click",
  function () {
    remaskedPositions = {};
    delete perFrameRemasked[currentScrubFrame];
    renderFrameWithTokens(currentScrubFrame);
    updateGuidedUI();
  }
);

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
    if (!scrubberActive || isGenerating) {
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
  modalAbout, modalHelp, modalSettings,
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

linkSettings.addEventListener(
  "click",
  function (e) {
    e.preventDefault();
    // Start editing from a fresh copy of the applied settings.
    stagedSettings.idleDisplay = appSettings.idleDisplay;
    stagedSettings.highlightTokens = appSettings.highlightTokens;
    stagedSettings.commitOrder = appSettings.commitOrder;
    syncSettingsControls();
    updateSettingsButtons();
    setSettingsStatus("", false);
    openModal(modalSettings);
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

// ---- Settings: idle display toggle ----

if (idleDisplayMount) {
  selectIdleDisplay = createCustomSelect(
    [
      { value: "default", label: "Default (ASCII Scene)" },
      { value: "donut", label: "donut.c (Spinning Torus)" },
    ],
    appSettings.idleDisplay
  );
  idleDisplayMount.appendChild(selectIdleDisplay);
  sizeCustomSelect(selectIdleDisplay);
  selectIdleDisplay.addEventListener(
    "change",
    function () {
      // Stage only; applied on Save like the other preferences.
      stagedSettings.idleDisplay = selectIdleDisplay.value;
      updateSettingsButtons();
    }
  );
}

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
    statusStep: statusStep.textContent,
    statusElapsed: statusElapsed.textContent,
    statusMessage: statusMessage.textContent,
  };
  var full = Object.assign({}, base, {
    frameTokens: frameTokens,
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
  if (s.prompt) {
    promptInput.value = s.prompt;
  }

  hasEverGenerated = true;
  if (thinkingPanel && thinkingContent) {
    if (s.thinking) {
      thinkingContent.textContent = s.thinking;
      thinkingPanel.hidden = false;
    } else {
      thinkingPanel.hidden = true;
    }
  }
  setSaveAvailable(true);
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
  // Seed the idle-animation choice and reflect prefs without
  // starting the animation here (boot decides that below based on
  // whether a prior session is restored).
  idleDisplayMode = appSettings.idleDisplay;
  updateStatusPrefs();
  updateHoverHighlight();
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
        startIdleAnimation();
      }
      connect();
    })
    .catch(function () {
      startIdleAnimation();
      connect();
    });
}

boot();
