// LLaDA Diffusion Visualizer — client-side logic.

"use strict";

const MASK_CHAR = "\u2591"; // ░
const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_DELAY_MS = 16000;

// ---- DOM refs ----

const promptInput = document.getElementById("prompt-input");
const btnGenerate = document.getElementById("btn-generate");
const btnCancel = document.getElementById("btn-cancel");
const outputArea = document.getElementById("output-area");
const outputPlaceholder = document.getElementById("output-placeholder");
const connectionBadge = document.getElementById("connection-badge");
const statusStep = document.getElementById("status-step");
const statusElapsed = document.getElementById("status-elapsed");
const statusMessage = document.getElementById("status-message");
const loadingOverlay = document.getElementById("loading-overlay");

const paramSteps = document.getElementById("param-steps");
const paramGenLength = document.getElementById("param-gen-length");
const paramBlockLength = document.getElementById("param-block-length");
const paramTemperature = document.getElementById("param-temperature");
const paramCfgScale = document.getElementById("param-cfg-scale");

// ---- State ----

let ws = null;
let isGenerating = false;
let reconnectDelay = RECONNECT_DELAY_MS;
let reconnectTimer = null;

// ---- Background floating characters ----

function spawnFloaters() {
  const container = document.getElementById("bg-floaters");
  if (!container) {
    return;
  }
  const chars = "01░▒▓█▄▀⣿⡇⠿⣀⣤⣿ΣΔΩλ∂∇";
  const COUNT = 30;

  for (let i = 0; i < COUNT; i++) {
    const el = document.createElement("span");
    el.className = "floater";
    el.textContent = chars[Math.floor(Math.random() * chars.length)];
    el.style.left = Math.random() * 100 + "%";
    el.style.animationDuration = 30 + Math.random() * 50 + "s";
    el.style.animationDelay = -(Math.random() * 60) + "s";
    el.style.fontSize = 10 + Math.random() * 8 + "px";
    container.appendChild(el);
  }
}

spawnFloaters();

// ---- WebSocket connection ----

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = protocol + "//" + location.host + "/ws";
  ws = new WebSocket(url);

  ws.onopen = function () {
    reconnectDelay = RECONNECT_DELAY_MS;
    setBadge("loading");
  };

  ws.onclose = function () {
    setBadge("disconnected");
    btnGenerate.disabled = true;
    scheduleReconnect();
  };

  ws.onerror = function () {
    ws.close();
  };

  ws.onmessage = function (event) {
    let data;
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
  reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
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
    loadingOverlay.classList.remove("hidden");
    btnGenerate.disabled = true;
  } else if (data.status === "ready") {
    setBadge("ready");
    loadingOverlay.classList.add("hidden");
    btnGenerate.disabled = false;
  }
}

function handleFrame(data) {
  renderFrame(data.text);
  const step = data.index;
  const total = data.total_steps;
  statusStep.textContent = "Step " + step + "/" + total;
  if (typeof data.elapsed === "number") {
    statusElapsed.textContent = "Elapsed: " + data.elapsed.toFixed(1) + "s";
  }
}

function handleDone(data) {
  setGenerating(false);
  statusMessage.textContent = "Done.";
  if (data.final_text) {
    renderFinalText(data.final_text);
  }
}

function handleError(data) {
  setGenerating(false);
  statusMessage.textContent = "Error: " + (data.message || "unknown");
  statusMessage.style.color = "var(--danger)";
  setTimeout(function () {
    statusMessage.style.color = "";
  }, 5000);
}

// ---- Rendering ----

function renderFrame(text) {
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === MASK_CHAR) {
      const span = document.createElement("span");
      span.className = "char-mask";
      span.textContent = ch;
      fragment.appendChild(span);
    } else if (ch === "\n") {
      fragment.appendChild(document.createTextNode("\n"));
    } else {
      const span = document.createElement("span");
      span.className = "char-resolved";
      span.textContent = ch;
      fragment.appendChild(span);
    }
  }
  outputArea.textContent = "";
  outputArea.appendChild(fragment);
}

function renderFinalText(text) {
  outputArea.textContent = "";
  const span = document.createElement("span");
  span.className = "char-resolved";
  span.textContent = text;
  outputArea.appendChild(span);
}

// ---- UI state helpers ----

function setBadge(state) {
  connectionBadge.className = "badge badge-" + state;
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

  if (!active) {
    btnGenerate.disabled = false;
  }
}

function resetStatus() {
  statusStep.textContent = "Step \u2014/\u2014";
  statusElapsed.textContent = "Elapsed: \u2014";
  statusMessage.textContent = "";
}

// ---- Actions ----

function startGeneration() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  if (isGenerating) {
    return;
  }

  const prompt = promptInput.value.trim();
  if (!prompt) {
    statusMessage.textContent = "Prompt is empty.";
    return;
  }

  outputArea.textContent = "";
  resetStatus();
  setGenerating(true);

  ws.send(JSON.stringify({
    type: "generate",
    prompt: prompt,
    steps: parseInt(paramSteps.value, 10),
    gen_length: parseInt(paramGenLength.value, 10),
    block_length: parseInt(paramBlockLength.value, 10),
    temperature: parseFloat(paramTemperature.value),
    cfg_scale: parseFloat(paramCfgScale.value),
  }));
}

function cancelGeneration() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  ws.send(JSON.stringify({ type: "cancel" }));
  setGenerating(false);
  statusMessage.textContent = "Cancelled.";
}

// ---- Event listeners ----

btnGenerate.addEventListener("click", startGeneration);
btnCancel.addEventListener("click", cancelGeneration);

promptInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    startGeneration();
  }
});

// ---- Boot ----

connect();
