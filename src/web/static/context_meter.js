// Exact next-inference context meter.
//
// The composer owns when a count is requested. This controller owns
// what one bounded result means and every DOM surface that presents
// it: the Send-adjacent ring and its native detail dialog.

"use strict";

var CONTEXT_METER_ERROR_CHARS_MAX = 512;
var CONTEXT_METER_TURNS_MAX = 200;

function contextMeterReadyState(options) {
  if (!options || typeof options !== "object") {
    throw new TypeError("Context meter needs count options");
  }
  var promptTokens = contextMeterInteger(
    options.count, "prompt token count", 0
  );
  var outputTokens = contextMeterInteger(
    options.outputReserve, "output reserve", 1
  );
  if (typeof options.truncated !== "boolean") {
    throw new TypeError("Context truncation flag must be boolean");
  }
  var pack = contextMeterPack(
    options.contextPack,
    promptTokens,
    outputTokens,
    options.contextLength
  );
  var usedTokens = pack.promptTokens + pack.outputTokens;
  var remainingTokens = Math.max(
    0, pack.effectiveTokens - usedTokens
  );
  var promptEnd = contextMeterPercent(
    pack.promptTokens, pack.effectiveTokens
  );
  var outputEnd = contextMeterPercent(
    usedTokens, pack.effectiveTokens
  );
  var usedPercent = Math.round(outputEnd);
  if (usedTokens > 0 && usedPercent === 0) {
    usedPercent = 1;
  }
  return Object.freeze({
    phase: "ready",
    structured: pack.structured,
    promptTokens: pack.promptTokens,
    outputTokens: pack.outputTokens,
    remainingTokens: remainingTokens,
    effectiveTokens: pack.effectiveTokens,
    requestedTokens: pack.requestedTokens,
    includedTurns: pack.includedTurns,
    omittedTurns: pack.omittedTurns,
    firstIncludedTurn: pack.firstIncludedTurn,
    promptEnd: promptEnd,
    outputEnd: outputEnd,
    usedPercent: usedPercent,
    over: (
      options.truncated === true
      || usedTokens > pack.effectiveTokens
    ),
    truncated: options.truncated === true,
    errorMessage: "",
  });
}

function contextMeterPack(
  raw, promptTokens, outputTokens, contextLength
) {
  if (!raw || typeof raw !== "object") {
    return contextMeterLegacyPack(
      promptTokens, outputTokens, contextLength
    );
  }
  contextMeterRequirePackFields(raw);
  var packedPrompt = contextMeterInteger(
    raw.prompt_token_count,
    "packed prompt token count",
    1
  );
  if (packedPrompt !== promptTokens) {
    throw new Error(
      "Context count disagrees with its packed prompt count"
    );
  }
  var effective = contextMeterInteger(
    raw.effective_total_budget,
    "effective context budget",
    1
  );
  var packedOutput = contextMeterInteger(
    raw.output_reserve, "packed output reserve", 1
  );
  if (packedOutput !== outputTokens) {
    throw new Error(
      "Context count disagrees with its output reserve"
    );
  }
  var requested = contextMeterInteger(
    raw.requested_total_budget,
    "requested context budget",
    1
  );
  if (requested < effective) {
    throw new Error(
      "Requested context budget is below the effective budget"
    );
  }
  var omitted = contextMeterInteger(
    raw.omitted_turn_count, "omitted turn count", 0
  );
  var first = contextMeterInteger(
    raw.first_included_index,
    "first included turn", 0
  );
  if (first !== omitted || omitted % 2 !== 0) {
    throw new Error("Context omission coordinates are inconsistent");
  }
  if (!Array.isArray(raw.included_turn_ids)) {
    throw new TypeError("Included turn ids must be an array");
  }
  if (
    raw.included_turn_ids.length === 0
    || raw.included_turn_ids.length > CONTEXT_METER_TURNS_MAX
  ) {
    throw new RangeError("Included turns exceed the meter bound");
  }
  for (
    var index = 0;
    index < raw.included_turn_ids.length;
    index++
  ) {
    if (typeof raw.included_turn_ids[index] !== "string") {
      throw new TypeError("Included turn ids must be strings");
    }
  }
  var included = raw.included_turn_ids.length;
  if (packedPrompt + packedOutput > effective) {
    throw new Error("Packed context exceeds its effective budget");
  }
  return {
    structured: true,
    promptTokens: packedPrompt,
    outputTokens: packedOutput,
    effectiveTokens: effective,
    requestedTokens: requested,
    includedTurns: included,
    omittedTurns: omitted,
    firstIncludedTurn: first + 1,
  };
}

function contextMeterRequirePackFields(raw) {
  var required = [
    "included_turn_ids",
    "first_included_index",
    "omitted_turn_count",
    "prompt_token_count",
    "output_reserve",
    "requested_total_budget",
    "effective_total_budget",
  ];
  for (var index = 0; index < required.length; index++) {
    if (!Object.prototype.hasOwnProperty.call(raw, required[index])) {
      throw new TypeError(
        "Context pack is missing " + required[index]
      );
    }
  }
}

function contextMeterLegacyPack(
  promptTokens, outputTokens, contextLength
) {
  var effective = contextMeterInteger(
    contextLength, "context window", 1
  );
  return {
    structured: false,
    promptTokens: promptTokens,
    outputTokens: outputTokens,
    effectiveTokens: effective,
    requestedTokens: effective,
    includedTurns: null,
    omittedTurns: 0,
    firstIncludedTurn: null,
  };
}

function contextMeterInteger(value, name, minimum) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new TypeError(name + " is outside its integer bound");
  }
  return value;
}

function contextMeterPercent(value, total) {
  return Math.min(100, Math.max(0, value * 100 / total));
}

function contextMeterCreate() {
  var button = contextMeterElement("btn-context-meter");
  var value = contextMeterElement("context-meter-value");
  var omitted = contextMeterElement("context-meter-omitted");
  var dialog = contextMeterElement("modal-context-meter");
  var closeButton = contextMeterElement(
    "btn-context-meter-close"
  );
  var detailStatus = contextMeterElement(
    "context-meter-detail-status"
  );
  var details = contextMeterDetailElements();
  var facts = contextMeterFacts({});
  var state = contextMeterPhase("unavailable", "");
  var returnFocus = null;
  var wired = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    button.addEventListener("click", open);
    closeButton.addEventListener("click", close);
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) {
        close();
      }
    });
    dialog.addEventListener("close", dialogClosed);
    render();
  }

  function configure(configuration) {
    close();
    facts = contextMeterFacts(configuration);
    state = contextMeterPhase("unavailable", "");
    render();
  }

  function pending() {
    close();
    state = contextMeterPhase("pending", "");
    render();
  }

  function unavailable() {
    close();
    state = contextMeterPhase("unavailable", "");
    render();
  }

  function error(message) {
    var text = typeof message === "string"
      ? message.slice(0, CONTEXT_METER_ERROR_CHARS_MAX)
      : "Context could not be counted.";
    state = contextMeterPhase("error", text);
    render();
  }

  function ready(result) {
    state = contextMeterReadyState({
      count: result.count,
      outputReserve: result.outputReserve,
      contextPack: result.contextPack,
      contextLength: facts.contextLength,
      truncated: result.truncated,
    });
    render();
  }

  function render() {
    contextMeterResetClasses(button);
    if (state.phase === "ready") {
      contextMeterRenderReady(
        button, value, omitted, state
      );
    } else {
      contextMeterRenderPhase(
        button, value, omitted, state
      );
    }
    contextMeterRenderDialog(details, detailStatus, facts, state);
  }

  function open() {
    if (
      button.getAttribute("aria-disabled") === "true"
      || dialog.open
    ) {
      return;
    }
    returnFocus = button;
    button.setAttribute("aria-expanded", "true");
    dialog.showModal();
  }

  function close() {
    if (dialog.open) {
      dialog.close();
    }
  }

  function dialogClosed() {
    button.setAttribute("aria-expanded", "false");
    if (
      returnFocus
      && returnFocus.isConnected
      && typeof returnFocus.focus === "function"
    ) {
      returnFocus.focus();
    }
    returnFocus = null;
  }

  return Object.freeze({
    wire: wire,
    configure: configure,
    pending: pending,
    unavailable: unavailable,
    error: error,
    ready: ready,
    close: close,
  });
}

function contextMeterFacts(raw) {
  var source = raw && typeof raw === "object" ? raw : {};
  var modelId = typeof source.modelId === "string"
    ? source.modelId
    : "";
  var modelDisplay = typeof source.modelDisplay === "string"
    ? source.modelDisplay
    : modelId;
  var inputMode = source.inputMode === "completion"
    ? "completion"
    : "chat";
  var device = typeof source.device === "string"
    ? source.device
    : "";
  var contextLength = (
    Number.isSafeInteger(source.contextLength)
    && source.contextLength > 0
  ) ? source.contextLength : null;
  return Object.freeze({
    modelId: modelId,
    modelDisplay: modelDisplay,
    inputMode: inputMode,
    device: device,
    contextLength: contextLength,
  });
}

function contextMeterPhase(phase, errorMessage) {
  return Object.freeze({
    phase: phase,
    structured: false,
    promptTokens: 0,
    outputTokens: 0,
    remainingTokens: 0,
    effectiveTokens: 0,
    requestedTokens: 0,
    includedTurns: null,
    omittedTurns: 0,
    firstIncludedTurn: null,
    promptEnd: 0,
    outputEnd: 0,
    usedPercent: 0,
    over: false,
    truncated: false,
    errorMessage: errorMessage,
  });
}

function contextMeterResetClasses(button) {
  button.classList.remove(
    "is-pending",
    "is-ready",
    "is-error",
    "has-omissions",
    "is-over"
  );
  button.style.setProperty("--context-prompt-end", "0%");
  button.style.setProperty("--context-output-end", "0%");
}

function contextMeterRenderReady(button, value, omitted, state) {
  var label = contextMeterReadyLabel(state);
  button.disabled = false;
  button.setAttribute("aria-disabled", "false");
  button.classList.add("is-ready");
  button.classList.toggle(
    "has-omissions", state.omittedTurns > 0
  );
  button.classList.toggle("is-over", state.over);
  button.style.setProperty(
    "--context-prompt-end",
    state.promptEnd.toFixed(2) + "%"
  );
  button.style.setProperty(
    "--context-output-end",
    state.outputEnd.toFixed(2) + "%"
  );
  value.textContent = state.over
    ? "!"
    : state.usedPercent + "%";
  omitted.hidden = state.omittedTurns === 0;
  button.setAttribute("aria-label", label);
  button.title = label;
}

function contextMeterRenderPhase(button, value, omitted, state) {
  omitted.hidden = true;
  if (state.phase === "error") {
    button.disabled = false;
    button.setAttribute("aria-disabled", "false");
    button.classList.add("is-error");
    value.textContent = "!";
    button.setAttribute(
      "aria-label", "Context count failed. Open details."
    );
    button.title = "Context count failed";
    return;
  }
  button.disabled = false;
  button.setAttribute("aria-disabled", "true");
  if (state.phase === "pending") {
    button.classList.add("is-pending");
    value.textContent = "\u2026";
    button.setAttribute(
      "aria-label", "Recalculating next-inference context"
    );
    button.title = "Recalculating context";
    return;
  }
  value.textContent = "--";
  button.setAttribute(
    "aria-label", "Next-inference context unavailable"
  );
  button.title = "Context unavailable";
}

function contextMeterReadyLabel(state) {
  if (state.truncated) {
    return "Context count exceeded its safety limit. Open details.";
  }
  var label = "Context: "
    + contextMeterNumber(state.promptTokens)
    + " prompt plus "
    + contextMeterNumber(state.outputTokens)
    + " reserved of "
    + contextMeterNumber(state.effectiveTokens)
    + " tokens";
  if (state.omittedTurns > 0) {
    label += ", " + contextMeterNumber(state.omittedTurns)
      + " earlier turns omitted";
  }
  return label;
}

function contextMeterRenderDialog(
  details, status, facts, state
) {
  details.model.textContent = contextMeterModelLabel(facts);
  details.mode.textContent = contextMeterModeLabel(facts.inputMode);
  details.device.textContent = contextMeterDeviceLabel(facts.device);
  status.classList.toggle(
    "is-error", state.phase === "error" || state.over
  );
  status.classList.toggle(
    "has-omissions",
    state.phase === "ready" && state.omittedTurns > 0
  );
  if (state.phase !== "ready") {
    contextMeterRenderDialogPhase(details, status, state);
    return;
  }
  if (state.truncated) {
    status.textContent =
      "The prompt count reached its safety limit."
      + " The displayed prompt count is a lower bound.";
  } else {
    status.textContent = state.over
      ? "The requested context exceeds the effective budget."
      : state.usedPercent
        + "% of the effective context is reserved for the next run.";
  }
  details.prompt.textContent = state.truncated
    ? "At least " + contextMeterNumber(state.promptTokens)
    : contextMeterNumber(state.promptTokens);
  details.output.textContent = contextMeterNumber(
    state.outputTokens
  );
  details.remaining.textContent = state.truncated
    ? "Not available"
    : contextMeterNumber(state.remainingTokens);
  details.effective.textContent = contextMeterNumber(
    state.effectiveTokens
  );
  details.requested.textContent = contextMeterNumber(
    state.requestedTokens
  );
  details.included.textContent = state.includedTurns === null
    ? "Not reported"
    : contextMeterNumber(state.includedTurns);
  details.omitted.textContent = contextMeterNumber(
    state.omittedTurns
  );
  details.first.textContent = state.firstIncludedTurn === null
    ? "Not reported"
    : "Turn " + contextMeterNumber(state.firstIncludedTurn);
  details.limit.textContent = contextMeterLimitText(state);
}

function contextMeterRenderDialogPhase(details, status, state) {
  var placeholder = "Not available";
  var names = [
    "prompt",
    "output",
    "remaining",
    "effective",
    "requested",
    "included",
    "omitted",
    "first",
    "limit",
  ];
  for (var index = 0; index < names.length; index++) {
    details[names[index]].textContent = placeholder;
  }
  if (state.phase === "error") {
    status.textContent = state.errorMessage;
  } else if (state.phase === "pending") {
    status.textContent = "Recalculating context.";
  } else {
    status.textContent =
      "Type a message with a ready model to calculate context.";
  }
}

function contextMeterLimitText(state) {
  if (!state.structured) {
    return "The loaded checkpoint supplies this window.";
  }
  if (state.effectiveTokens < state.requestedTokens) {
    return "The checkpoint window lowered the policy budget.";
  }
  return "The effective budget follows model and device policy.";
}

function contextMeterModelLabel(facts) {
  if (facts.modelDisplay === "") {
    return "Not available";
  }
  if (
    facts.modelId === ""
    || facts.modelId === facts.modelDisplay
  ) {
    return facts.modelDisplay;
  }
  return facts.modelDisplay + " (" + facts.modelId + ")";
}

function contextMeterModeLabel(mode) {
  return mode === "completion" ? "Completion" : "Chat";
}

function contextMeterDeviceLabel(device) {
  if (device === "cuda") {
    return "GPU";
  }
  if (device === "cpu") {
    return "CPU";
  }
  return device === "" ? "Not available" : device.toUpperCase();
}

function contextMeterNumber(value) {
  return value.toLocaleString();
}

function contextMeterDetailElements() {
  return {
    model: contextMeterElement("context-meter-model"),
    mode: contextMeterElement("context-meter-mode"),
    device: contextMeterElement("context-meter-device"),
    prompt: contextMeterElement("context-meter-prompt"),
    output: contextMeterElement("context-meter-output"),
    remaining: contextMeterElement("context-meter-remaining"),
    effective: contextMeterElement("context-meter-effective"),
    requested: contextMeterElement("context-meter-requested"),
    included: contextMeterElement("context-meter-included"),
    omitted: contextMeterElement("context-meter-omitted-detail"),
    first: contextMeterElement("context-meter-first"),
    limit: contextMeterElement("context-meter-limit-note"),
  };
}

function contextMeterElement(id) {
  var element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing context meter element #" + id);
  }
  return element;
}
