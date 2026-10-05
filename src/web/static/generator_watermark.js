// The generator's tokenizer-only KGW detector dialog.
//
// The controller owns its request ids so a reply for text that has
// since changed or for a dialog that closed cannot overwrite the
// current result. The worker performs raw tokenization and scoring;
// this file only validates bounded display inputs and renders the
// returned statistical evidence without an authorship label.

"use strict";

function generatorWatermarkCreate(options) {
  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorWatermarkCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing generator watermark element #" + id
      );
    }
    return element;
  }

  var sendRequest = requiredCallback("sendRequest");
  var readParameters = requiredCallback("readParameters");

  var openButton = requiredElement("btn-watermark-detector");
  var dialog = requiredElement("modal-watermark-detector");
  var closeButton = requiredElement("btn-watermark-close");
  var detectButton = requiredElement("btn-watermark-detect");
  var textInput = requiredElement("watermark-detector-text");
  var gammaInput = requiredElement("watermark-detector-gamma");
  var thresholdInput =
    requiredElement("watermark-detector-threshold");
  var keyInput = requiredElement("watermark-detector-key");
  var result = requiredElement("watermark-detector-result");

  var requestId = 0;
  var pendingId = null;
  var supported = false;
  var wired = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    openButton.addEventListener("click", open);
    closeButton.addEventListener("click", close);
    detectButton.addEventListener("click", detect);
    textInput.addEventListener("input", inputsChanged);
    gammaInput.addEventListener("input", inputsChanged);
    thresholdInput.addEventListener("input", inputsChanged);
    keyInput.addEventListener("input", inputsChanged);
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) {
        close();
      }
    });
    dialog.addEventListener("close", invalidate);
  }

  function configure(capabilities) {
    supported = !!(
      capabilities && capabilities.supports_watermark
    );
    openButton.hidden = !supported;
    if (!supported) {
      close();
    }
  }

  function open() {
    if (!supported) {
      return;
    }
    applyParameterDefaults();
    if (!dialog.open) {
      dialog.showModal();
    }
    textInput.focus();
  }

  function close() {
    if (dialog.open) {
      dialog.close();
    } else {
      invalidate();
    }
  }

  function invalidate() {
    requestId += 1;
    pendingId = null;
    detectButton.disabled = false;
    result.className = "watermark-detector-result";
  }

  function inputsChanged() {
    if (pendingId === null && result.textContent === "") {
      return;
    }
    requestId += 1;
    pendingId = null;
    detectButton.disabled = false;
    result.className =
      "watermark-detector-result is-stale";
    result.textContent = "Inputs changed. Detect again.";
  }

  function applyParameterDefaults() {
    var parameters = readParameters() || {};
    var gamma = parameters.watermark_gamma;
    var threshold = parameters.watermark_z_threshold;
    if (typeof gamma === "number" && isFinite(gamma)) {
      gammaInput.value = String(gamma);
    }
    if (typeof threshold === "number" && isFinite(threshold)) {
      thresholdInput.value = String(threshold);
    }
  }

  function detect() {
    var request;
    try {
      request = detectorRequest();
    } catch (error) {
      showError(error.message);
      return;
    }
    requestId += 1;
    pendingId = requestId;
    request.request_id = pendingId;
    detectButton.disabled = true;
    result.className =
      "watermark-detector-result is-pending";
    result.textContent = "Tokenizing and scoring...";
    if (!sendRequest(request)) {
      pendingId = null;
      detectButton.disabled = false;
      showError("The model worker is not connected.");
    }
  }

  function detectorRequest() {
    var text = textInput.value;
    if (typeof text !== "string" || text.length === 0) {
      throw new Error("Paste text to detect first.");
    }
    if (text.length > 100000) {
      throw new Error(
        "Detector text must be at most 100,000 characters."
      );
    }
    var gamma = Number(gammaInput.value);
    if (!isFinite(gamma) || gamma <= 0 || gamma >= 1) {
      throw new Error(
        "Green fraction must be between 0 and 1."
      );
    }
    var threshold = Number(thresholdInput.value);
    if (!isFinite(threshold) || threshold < 0) {
      throw new Error(
        "Display z threshold must be non-negative."
      );
    }
    var expected = keyInput.value.trim();
    if (expected && !/^[0-9a-f]{16}$/.test(expected)) {
      throw new Error(
        "Expected key id must be 16 lowercase hex characters."
      );
    }
    var request = {
      type: "detect_watermark",
      text: text,
      gamma: gamma,
      z_threshold: threshold,
    };
    if (expected) {
      request.expected_key_id = expected;
    }
    return request;
  }

  function handleResult(message) {
    if (
      !message
      || message.type !== "detect_watermark_result"
    ) {
      return false;
    }
    if (message.request_id !== pendingId) {
      return true;
    }
    pendingId = null;
    detectButton.disabled = false;
    result.className = "watermark-detector-result";
    result.textContent = detectorResultText(message);
    return true;
  }

  function detectorResultText(message) {
    var rate = Number(message.green_rate) * 100;
    var fingerprint = String(
      message.tokenizer_fingerprint || ""
    );
    var status = detectorStatusText(
      message.status, message.display_threshold
    );
    return (
      "Key " + message.key_id
      + " | model " + message.model_id
      + " | gamma " + message.gamma
      + " | vocab " + message.vocab_size
      + " | tokenizer " + fingerprint.slice(0, 16)
      + " | tokens " + message.token_count
      + " | green/scored " + message.green_count
      + "/" + message.scored_count
      + " | green rate " + rate.toFixed(1) + "%"
      + " | z " + Number(message.z_score).toFixed(2)
      + " | p0 "
      + overlaysWatermarkExactProbability(Number(message.p0))
      + " | " + status
    );
  }

  function detectorStatusText(status, threshold) {
    if (status === "insufficient_evidence") {
      return "insufficient evidence";
    }
    if (status === "threshold_crossed") {
      return "threshold crossed at display z " + threshold;
    }
    return "threshold not crossed at display z " + threshold;
  }

  function handleError(message) {
    if (
      !message
      || message.request_type !== "detect_watermark"
    ) {
      return false;
    }
    if (message.request_id !== pendingId) {
      return true;
    }
    pendingId = null;
    detectButton.disabled = false;
    if (message.code === "watermark_key_missing") {
      showError(
        "No local KGW key exists. Enable watermarking for a run"
        + " first."
      );
    } else if (message.code === "watermark_key_mismatch") {
      showError(
        "Key mismatch: "
        + (
          message.message
          || "the loaded key does not match the expected key id."
        )
      );
    } else {
      showError(message.message || "Detection failed.");
    }
    return true;
  }

  function showError(message) {
    result.className = "watermark-detector-result is-error";
    result.textContent = message;
  }

  return Object.freeze({
    wire: wire,
    configure: configure,
    close: close,
    handleResult: handleResult,
    handleError: handleError,
  });
}
