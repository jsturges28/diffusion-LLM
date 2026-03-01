// LLaDA Diffusion Visualizer — client-side logic.

"use strict";

var MASK_CHAR = "\u2591"; // ░
var RECONNECT_DELAY_MS = 2000;
var MAX_RECONNECT_DELAY_MS = 16000;

// ---- Parameter bounds (two tiers) ----

var LIMITS_RECOMMENDED = {
  steps:        { min: 8,   max: 150  },
  gen_length:   { min: 16,  max: 160  },
  block_length: { min: 8,   max: 160  },
  temperature:  { min: 0.0, max: 1.0  },
  cfg_scale:    { min: 0.0, max: 2.0  },
};

var LIMITS_EXPERIMENTAL = {
  steps:        { min: 1,   max: 1024 },
  gen_length:   { min: 1,   max: 1024 },
  block_length: { min: 1,   max: 1024 },
  temperature:  { min: 0.0, max: 10.0 },
  cfg_scale:    { min: 0.0, max: 20.0 },
};

var PARAM_LABELS = {
  steps: "Steps",
  gen_length: "Gen Length",
  block_length: "Block Length",
  temperature: "Temperature",
  cfg_scale: "CFG Scale",
};

// ---- DOM refs ----

var promptInput =
  document.getElementById("prompt-input");
var btnGenerate =
  document.getElementById("btn-generate");
var btnCancel =
  document.getElementById("btn-cancel");
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

var paramSteps =
  document.getElementById("param-steps");
var paramGenLength =
  document.getElementById("param-gen-length");
var paramBlockLength =
  document.getElementById("param-block-length");
var paramTemperature =
  document.getElementById("param-temperature");
var paramCfgScale =
  document.getElementById("param-cfg-scale");
var paramRemasking =
  document.getElementById("param-remasking");

var rangeSteps =
  document.getElementById("range-steps");
var rangeGenLength =
  document.getElementById("range-gen-length");
var rangeBlockLength =
  document.getElementById("range-block-length");
var rangeTemperature =
  document.getElementById("range-temperature");
var rangeCfgScale =
  document.getElementById("range-cfg-scale");

var PARAM_INPUTS = {
  steps: paramSteps,
  gen_length: paramGenLength,
  block_length: paramBlockLength,
  temperature: paramTemperature,
  cfg_scale: paramCfgScale,
};

var RANGE_LABELS = {
  steps: rangeSteps,
  gen_length: rangeGenLength,
  block_length: rangeBlockLength,
  temperature: rangeTemperature,
  cfg_scale: rangeCfgScale,
};

// Settings DOM refs.
var linkSettings =
  document.getElementById("link-settings");
var modalSettings =
  document.getElementById("modal-settings");
var selectIdleDisplay =
  document.getElementById("select-idle-display");

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
var remaskControls =
  document.getElementById("remask-controls");
var remaskCount =
  document.getElementById("remask-count");
var btnClearRemask =
  document.getElementById("btn-clear-remask");
var btnResume =
  document.getElementById("btn-resume");

// ---- State ----

var ws = null;
var isGenerating = false;
var isSaving = false;
var modelReady = false;
var paramsValid = true;
var reconnectDelay = RECONNECT_DELAY_MS;
var reconnectTimer = null;

// Accumulated data for the most recent completed run.
var frameHistory = [];
var frameTokens = [];
var lastRunParams = null;
var lastFinalText = null;

// Scrubber and remasking state.
var scrubberActive = false;
var currentScrubFrame = 0;
var remaskedPositions = {};
var remaskEdits = [];

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

// ---- Limits helpers ----

function activeLimits() {
  if (toggleExperimental.checked) {
    return LIMITS_EXPERIMENTAL;
  }
  return LIMITS_RECOMMENDED;
}

function updateRangeLabels() {
  var limits = activeLimits();
  var keys = Object.keys(RANGE_LABELS);
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var bound = limits[key];
    RANGE_LABELS[key].textContent =
      "(" + bound.min + "\u2013" + bound.max + ")";
  }
}

function applyLimits() {
  var limits = activeLimits();
  var keys = Object.keys(PARAM_INPUTS);
  for (var i = 0; i < keys.length; i++) {
    var key = keys[i];
    var input = PARAM_INPUTS[key];
    var bound = limits[key];
    input.min = bound.min;
    input.max = bound.max;

    var val = parseFloat(input.value);
    if (!isNaN(val)) {
      if (val < bound.min) {
        input.value = bound.min;
      } else if (val > bound.max) {
        input.value = bound.max;
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

  var keys = Object.keys(PARAM_INPUTS);
  for (var i = 0; i < keys.length; i++) {
    PARAM_INPUTS[keys[i]].classList.remove(
      "input-warn"
    );
  }

  for (var j = 0; j < keys.length; j++) {
    var key = keys[j];
    var input = PARAM_INPUTS[key];
    var raw = input.value.trim();
    var val = parseFloat(raw);
    var bound = limits[key];
    var label = PARAM_LABELS[key];

    if (raw === "" || isNaN(val)) {
      input.classList.add("input-warn");
      errors.push(
        label + " is empty or invalid."
      );
      continue;
    }

    if (val < bound.min) {
      input.classList.add("input-warn");
      if (val < 0) {
        errors.push(
          label + " cannot be negative."
        );
      } else {
        errors.push(
          label + " must be at least "
          + bound.min + "."
        );
      }
      continue;
    }

    if (val > bound.max) {
      input.classList.add("input-warn");
      errors.push(
        label + " must be at most "
        + bound.max + "."
      );
    }
  }

  var genLength =
    parseInt(paramGenLength.value, 10);
  var blockLength =
    parseInt(paramBlockLength.value, 10);
  var steps =
    parseInt(paramSteps.value, 10);

  var genOk = !paramGenLength.classList.contains(
    "input-warn"
  );
  var blkOk = !paramBlockLength.classList.contains(
    "input-warn"
  );
  var stpOk = !paramSteps.classList.contains(
    "input-warn"
  );

  if (
    genOk && blkOk
    && blockLength > 0
    && genLength % blockLength !== 0
  ) {
    paramGenLength.classList.add("input-warn");
    paramBlockLength.classList.add("input-warn");
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
    if (
      numBlocks > 0
      && steps % numBlocks !== 0
    ) {
      paramSteps.classList.add("input-warn");
      errors.push(
        "Steps (" + steps
        + ") must be divisible by num_blocks ("
        + numBlocks + ")."
      );
    }
  }

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
    scheduleReconnect();
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

  renderFrame(data.text);

  var globalIndex =
    frameHistory.length - 1;
  var displayStep = isResuming
    ? "Resuming " + data.index
      + "/" + data.total_steps
    : "Step " + data.index
      + "/" + data.total_steps;
  statusStep.textContent = displayStep;

  if (typeof data.elapsed === "number") {
    statusElapsed.textContent =
      "Elapsed: " + data.elapsed.toFixed(1) + "s";
  }
}

function handleDone(data) {
  setGenerating(false);
  isResuming = false;
  statusMessage.textContent = "Done.";
  if (data.final_text) {
    lastFinalText = data.final_text;
  }
  lastRunParams = {
    steps: parseInt(paramSteps.value, 10),
    gen_length:
      parseInt(paramGenLength.value, 10),
    block_length:
      parseInt(paramBlockLength.value, 10),
    temperature:
      parseFloat(paramTemperature.value),
    cfg_scale:
      parseFloat(paramCfgScale.value),
    remasking: paramRemasking.value,
  };
  setSaveAvailable(true);
  activateScrubber();
}

function handleError(data) {
  setGenerating(false);
  isResuming = false;
  statusMessage.textContent =
    "Error: " + (data.message || "unknown");
  statusMessage.style.color = "var(--danger)";
  setTimeout(function () {
    statusMessage.style.color = "";
  }, 5000);
}

// ---- Rendering ----

function renderFrame(text) {
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

// Token-level rendering for scrubber mode.
// Each token is a clickable span; resolved tokens
// can be clicked to toggle remasking.
function renderFrameWithTokens(frameIndex) {
  var tokens = frameTokens[frameIndex];
  if (!tokens) {
    renderFrame(frameHistory[frameIndex]);
    return;
  }

  var fragment =
    document.createDocumentFragment();

  for (var i = 0; i < tokens.length; i++) {
    var tok = tokens[i];
    var span = document.createElement("span");
    span.setAttribute("data-pos", String(i));

    var isUserRemasked =
      remaskedPositions[i] === true;

    if (isUserRemasked) {
      span.className =
        "token-span token-remasked";
      span.textContent = MASK_CHAR;
    } else if (tok.m) {
      span.className = "token-span token-mask";
      span.textContent = MASK_CHAR;
    } else {
      span.className =
        "token-span token-resolved"
        + " token-clickable";
      span.textContent = tok.t;
    }
    fragment.appendChild(span);
  }

  outputArea.textContent = "";
  outputArea.appendChild(fragment);
}

// ---- Scrubber ----

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
  updateScrubberLabel();

  scrubberSection.hidden = false;
  clearRemaskedPositions();

  navigateToFrame(currentScrubFrame);
}

function deactivateScrubber() {
  scrubberActive = false;
  scrubberSection.hidden = true;
  remaskControls.hidden = true;
  clearRemaskedPositions();
}

function updateScrubberLabel() {
  scrubberLabel.textContent =
    "Frame " + currentScrubFrame
    + " / " + (frameHistory.length - 1);
}

function navigateToFrame(index) {
  index = Math.max(
    0,
    Math.min(index, frameHistory.length - 1)
  );
  currentScrubFrame = index;
  scrubberSlider.value = String(index);
  updateScrubberLabel();

  clearRemaskedPositions();
  renderFrameWithTokens(index);
  updateRemaskControls();
}

function updateRemaskControls() {
  var count = Object.keys(remaskedPositions).length;
  if (count > 0) {
    remaskControls.hidden = false;
    remaskCount.textContent =
      count + " token"
      + (count !== 1 ? "s" : "")
      + " selected";
    btnResume.disabled = false;
  } else {
    remaskControls.hidden = true;
    btnResume.disabled = true;
  }
}

function clearRemaskedPositions() {
  remaskedPositions = {};
  updateRemaskControls();
}

function toggleRemaskPosition(pos) {
  if (remaskedPositions[pos]) {
    delete remaskedPositions[pos];
  } else {
    remaskedPositions[pos] = true;
  }
  renderFrameWithTokens(currentScrubFrame);
  updateRemaskControls();
}

// ---- UI state helpers ----

function setBadge(state) {
  connectionBadge.className =
    "badge badge-" + state;
  connectionBadge.textContent = state;
}

function setGenerating(active) {
  isGenerating = active;
  btnGenerate.hidden = active;
  btnCancel.hidden = !active;
  promptInput.disabled = active;
  paramSteps.disabled = active;
  paramGenLength.disabled = active;
  paramBlockLength.disabled = active;
  paramTemperature.disabled = active;
  paramCfgScale.disabled = active;
  paramRemasking.disabled = active;
  toggleExperimental.disabled = active;

  if (active) {
    deactivateScrubber();
  }

  if (!active && modelReady && paramsValid) {
    btnGenerate.disabled = false;
  }
}

function setSaveAvailable(available) {
  if (available && frameHistory.length > 0) {
    btnSave.hidden = false;
    btnSave.disabled = false;
    btnSave.textContent = "Save";
  } else {
    btnSave.hidden = true;
    btnSave.disabled = true;
  }
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

  frameHistory = [];
  frameTokens = [];
  lastRunParams = null;
  lastFinalText = null;
  remaskEdits = [];
  isResuming = false;
  resumeFrameOffset = 0;
  setSaveAvailable(false);

  outputArea.textContent = "";
  resetStatus();
  setGenerating(true);

  ws.send(JSON.stringify({
    type: "generate",
    prompt: prompt,
    steps: parseInt(paramSteps.value, 10),
    gen_length:
      parseInt(paramGenLength.value, 10),
    block_length:
      parseInt(paramBlockLength.value, 10),
    temperature:
      parseFloat(paramTemperature.value),
    cfg_scale:
      parseFloat(paramCfgScale.value),
    remasking: paramRemasking.value,
    experimental: toggleExperimental.checked,
  }));
}

function cancelGeneration() {
  if (
    !ws
    || ws.readyState !== WebSocket.OPEN
  ) {
    return;
  }
  ws.send(JSON.stringify({ type: "cancel" }));
  setGenerating(false);
  isResuming = false;
  statusMessage.textContent = "Cancelled.";
}

function startResume() {
  if (
    !ws
    || ws.readyState !== WebSocket.OPEN
  ) {
    return;
  }
  if (isGenerating) {
    return;
  }

  var positions =
    Object.keys(remaskedPositions).map(Number);
  if (positions.length === 0) {
    return;
  }

  remaskEdits.push({
    frame_index: currentScrubFrame,
    token_positions: positions.slice(),
  });

  resumeFrameOffset = currentScrubFrame;
  frameHistory.length = resumeFrameOffset;
  frameTokens.length = resumeFrameOffset;
  isResuming = true;

  setSaveAvailable(false);
  resetStatus();
  statusMessage.textContent = "Resuming...";
  setGenerating(true);

  ws.send(JSON.stringify({
    type: "resume",
    frame_index: currentScrubFrame,
    remask_positions: positions,
  }));
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
  btnSave.textContent = "Saving\u2026";
  statusMessage.textContent = "";
  statusMessage.style.color = "";

  var payload = {
    prompt: promptInput.value.trim(),
    params: lastRunParams,
    frames: frameHistory,
    final_text: lastFinalText,
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
      if (result.success) {
        btnSave.textContent = "Saved";
        statusMessage.textContent =
          "Saved to " + result.path;
        statusMessage.style.color =
          "var(--accent)";
      } else {
        btnSave.disabled = false;
        btnSave.textContent = "Save";
        statusMessage.textContent =
          "Save failed: "
          + (result.message || "unknown");
        statusMessage.style.color =
          "var(--danger)";
      }
    })
    .catch(function (error) {
      isSaving = false;
      btnSave.disabled = false;
      btnSave.textContent = "Save";
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
btnCancel.addEventListener(
  "click", cancelGeneration
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

paramSteps.addEventListener(
  "input", validateAllParams
);
paramGenLength.addEventListener(
  "input", validateAllParams
);
paramBlockLength.addEventListener(
  "input", validateAllParams
);
paramTemperature.addEventListener(
  "input", validateAllParams
);
paramCfgScale.addEventListener(
  "input", validateAllParams
);

// Scrubber event listeners.
scrubberSlider.addEventListener(
  "input",
  function () {
    var val = parseInt(scrubberSlider.value, 10);
    currentScrubFrame = val;
    updateScrubberLabel();
    clearRemaskedPositions();
    renderFrameWithTokens(val);
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
    navigateToFrame(frameHistory.length - 1);
  }
);

btnClearRemask.addEventListener(
  "click",
  function () {
    clearRemaskedPositions();
    renderFrameWithTokens(currentScrubFrame);
  }
);

btnResume.addEventListener(
  "click", startResume
);

// Token click delegation on the output area.
outputArea.addEventListener(
  "click",
  function (e) {
    if (!scrubberActive) {
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
      navigateToFrame(
        frameHistory.length - 1
      );
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

selectIdleDisplay.addEventListener(
  "change",
  function () {
    var newMode = selectIdleDisplay.value;
    if (newMode === idleDisplayMode) {
      return;
    }
    stopIdleAnimation();
    idleDisplayMode = newMode;
    startIdleAnimation();
  }
);

// ---- Boot ----

updateRangeLabels();
validateAllParams();
startIdleAnimation();
connect();
