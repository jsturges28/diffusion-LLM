// The generator's surrounding chrome: connection and loading state,
// output placeholder, run readouts, resource meter, status operations
// and the saved-run cue.
//
// Loaded as a classic script before app.js. The page supplies the
// settings and animation callbacks that remain page-owned. DOM
// references, timers, operation handles and bounded histories stay
// private to the returned controller.

"use strict";

function generatorChromeCreate(options) {
  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorChromeCreate needs options." + name
      );
    }
    return options[name];
  }

  var onTpsToggle = requiredCallback("onTpsToggle");
  var readReducedMotion =
    requiredCallback("readReducedMotion");
  var readDiffusionEffect =
    requiredCallback("readDiffusionEffect");
  var readDiffusionTextMode =
    requiredCallback("readDiffusionTextMode");
  var revealText = requiredCallback("revealText");
  var cancelReveal = requiredCallback("cancelReveal");

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Missing generator chrome element #" + id);
    }
    return element;
  }

  var outputArea = requiredElement("output-area");
  var connectionBadge = requiredElement("connection-badge");
  var statusStep = requiredElement("status-step");
  var statusElapsed = requiredElement("status-elapsed");
  var statusTps = requiredElement("status-tps");
  var statusResource = requiredElement("status-resource");
  var statusResourceLabel =
    requiredElement("status-resource-label");
  var statusResourceSpark =
    requiredElement("status-resource-spark");
  var statusResourceValue =
    requiredElement("status-resource-value");
  var statusStack = requiredElement("status-stack");
  var statusMessage = requiredElement("status-message");
  var loadingOverlay = requiredElement("loading-overlay");
  var loadingText = requiredElement("loading-text");
  var loadProgressContainer =
    requiredElement("load-progress-container");
  var loadProgressFill =
    requiredElement("load-progress-fill");
  var loadProgressDetail =
    requiredElement("load-progress-detail");
  var linkAnalytics = requiredElement("link-analytics");
  var analyticsNewDot = requiredElement("analytics-new-dot");

  var RESOURCE_HISTORY_MAX = 120;
  var RESOURCE_LABELS = {
    vram: "VRAM",
    cpu: "CPU",
  };
  var RESOURCE_LINE_COLOR = "#00ff41";
  var ELAPSED_TICK_MS = 100;
  var STATUS_DOTS_MS = 400;
  var STATUS_CYCLE_HOLD_MS = 700;
  var STATUS_STACK_MAX = 4;
  var STATUS_CHIP_FADE_MS = 150;
  var ANALYTICS_PLUS_ONE_MS = 1500;

  var resourceHistory = [];
  var resourceLatest = null;
  var elapsedStampSeconds = null;
  var elapsedStampAt = 0;
  var elapsedTimer = null;
  var statusChips = [];
  var runStatusHandle = null;
  var wired = false;
  var booted = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    statusTps.addEventListener("click", onTpsToggle);
    statusTps.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      onTpsToggle();
    });
  }

  function boot() {
    if (booted) {
      return;
    }
    booted = true;
    watchStatusMessage();
    refreshAnalyticsCue();
  }

  function showOutputPlaceholder(displayName) {
    showOutputNotice(
      displayName
        ? displayName + " output will appear here..."
        : "Output will appear here..."
    );
  }

  function showNoOutput(stopped) {
    showOutputNotice(
      stopped === true
        ? "The run stopped before producing any text."
        : "The model ended before producing any text."
          + " Retry to sample another continuation."
    );
  }

  function showOutputNotice(message) {
    var placeholder = document.createElement("span");
    placeholder.id = "output-placeholder";
    placeholder.textContent = message;
    outputArea.replaceChildren(placeholder);
  }

  function setConnection(state) {
    connectionBadge.className = "badge badge-" + state;
    connectionBadge.textContent = state;
  }

  function setLoadingText(text) {
    loadingText.textContent = text;
  }

  function setLoadingProgress(state, progress) {
    var view = activationProgressView(state, progress);
    var sweeping = view.mode === "sweep";
    loadProgressContainer.hidden = view.mode === "hidden";
    loadProgressFill.classList.toggle("is-sweep", sweeping);
    if (sweeping) {
      loadProgressFill.style.removeProperty("width");
    } else {
      loadProgressFill.style.width = view.percent + "%";
    }
    loadProgressDetail.hidden = view.mode === "hidden";
    loadProgressDetail.textContent = sweeping
      ? view.label + "\u2026"
      : view.label + ", " + view.percent + "%";
  }

  function finishLoadingProgress(done) {
    if (typeof done !== "function") {
      throw new TypeError(
        "generatorChrome.finishLoadingProgress needs a callback"
      );
    }
    if (loadProgressContainer.hidden) {
      done();
      return;
    }
    setLoadingProgress("ready", null);
    setTimeout(done, ACTIVATION_PROGRESS_HOLD_MS);
  }

  function showLoading() {
    loadingOverlay.classList.remove("hidden");
  }

  function hideLoading() {
    loadingOverlay.classList.add("hidden");
  }

  function handleResourceSample(data) {
    var label = data ? RESOURCE_LABELS[data.kind] : null;
    if (!label) {
      return;
    }
    if (typeof data.fraction !== "number") {
      return;
    }
    if (
      resourceLatest !== null
      && resourceLatest.kind !== data.kind
    ) {
      resourceHistory = [];
    }
    resourceLatest = data;
    resourceHistory.push(
      Math.max(0, Math.min(1, data.fraction))
    );
    if (resourceHistory.length > RESOURCE_HISTORY_MAX) {
      resourceHistory.shift();
    }
    statusResource.hidden = false;
    statusResourceLabel.textContent = label;
    statusResourceValue.textContent = resourceValueText(data);
    drawResourceSpark();
  }

  function resourceValueText(sample) {
    if (sample.kind === "cpu") {
      return Math.round(sample.fraction * 100) + "% of "
        + sample.total_cores + " cores";
    }
    return formatVramGib(sample.used_bytes)
      + " / " + formatVramGib(sample.total_bytes);
  }

  function formatVramGib(bytes) {
    if (typeof bytes !== "number") {
      return "?";
    }
    return (bytes / (1024 * 1024 * 1024)).toFixed(1)
      + " GiB";
  }

  function drawResourceSpark() {
    if (resourceHistory.length === 0) {
      return;
    }
    var ratio = window.devicePixelRatio || 1;
    var cssWidth = statusResourceSpark.clientWidth || 60;
    var cssHeight = statusResourceSpark.clientHeight || 11;
    statusResourceSpark.width = Math.round(cssWidth * ratio);
    statusResourceSpark.height = Math.round(cssHeight * ratio);
    var context = statusResourceSpark.getContext("2d");
    if (!context) {
      return;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    var span = Math.max(1, RESOURCE_HISTORY_MAX - 1);
    var first = RESOURCE_HISTORY_MAX - resourceHistory.length;
    context.beginPath();
    for (var index = 0; index < resourceHistory.length; index++) {
      var x = ((first + index) / span) * cssWidth;
      var y =
        cssHeight - resourceHistory[index] * cssHeight;
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }
    context.strokeStyle = RESOURCE_LINE_COLOR;
    context.lineWidth = 1;
    context.stroke();
  }

  function clearResourceMeter() {
    resourceHistory = [];
    resourceLatest = null;
    statusResource.hidden = true;
  }

  function setStep(text) {
    statusStep.textContent = text;
  }

  function updateRateFooter(state) {
    if (
      !state
      || typeof state.elapsedSeconds !== "number"
      || !isFinite(state.elapsedSeconds)
    ) {
      throw new TypeError(
        "generatorChrome.updateRateFooter needs elapsedSeconds"
      );
    }
    elapsedStampSeconds = state.elapsedSeconds;
    elapsedStampAt = Date.now();
    renderElapsed(elapsedStampSeconds);
    elapsedTick();
    renderTpsFooter(state.rate, state.tpsMode);
  }

  function renderElapsed(seconds) {
    statusElapsed.textContent =
      "Elapsed: " + seconds.toFixed(1) + "s";
  }

  function elapsedTick() {
    if (elapsedTimer !== null) {
      return;
    }
    elapsedTimer = setInterval(function () {
      if (elapsedStampSeconds === null) {
        return;
      }
      var since = (Date.now() - elapsedStampAt) / 1000;
      renderElapsed(elapsedStampSeconds + since);
    }, ELAPSED_TICK_MS);
  }

  function elapsedStop() {
    if (elapsedTimer !== null) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
    elapsedStampSeconds = null;
  }

  function elapsedSettle() {
    if (elapsedStampSeconds !== null) {
      renderElapsed(elapsedStampSeconds);
    }
    elapsedStop();
  }

  function renderTpsFooter(rate, mode) {
    var label = mode === "last"
      ? "Last step"
      : "Run average";
    statusTps.title = "Tokens per second ("
      + label.toLowerCase() + "). Click to switch.";
    if (rate === null) {
      statusTps.textContent = "T/s: -";
      return;
    }
    var shown = rate < 100
      ? rate.toFixed(1)
      : String(Math.round(rate));
    statusTps.textContent = "T/s: " + shown;
  }

  function watchStatusMessage() {
    if (typeof MutationObserver !== "function") {
      return;
    }
    var observer =
      new MutationObserver(applyStatusMessageTitle);
    observer.observe(statusMessage, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    applyStatusMessageTitle();
  }

  function applyStatusMessageTitle() {
    var text = statusMessage.textContent || "";
    var clipped =
      statusMessage.scrollWidth > statusMessage.clientWidth + 1;
    statusMessage.title = clipped ? text : "";
    statusMessage.classList.toggle("is-clipped", clipped);
  }

  function statusRowReflow(mutate) {
    if (typeof mutate !== "function") {
      throw new TypeError(
        "generatorChrome status mutation must be a callback"
      );
    }
    if (readReducedMotion()) {
      mutate();
      return;
    }
    var moved = Array.prototype.slice.call(
      statusStack.querySelectorAll(".status-chip")
    );
    var before = moved.map(function (chip) {
      return chip.getBoundingClientRect().left;
    });
    mutate();
    for (var index = 0; index < moved.length; index++) {
      var chip = moved[index];
      var delta =
        before[index] - chip.getBoundingClientRect().left;
      if (delta === 0) {
        continue;
      }
      chip.style.transition = "none";
      chip.style.transform =
        "translateX(" + delta + "px)";
      void chip.offsetWidth;
      chip.style.transition = "";
      chip.style.transform = "";
    }
  }

  function setMessage(text, settings) {
    var config = settings || {};
    statusRowReflow(function () {
      statusMessage.textContent = text;
    });
    if (
      Object.prototype.hasOwnProperty.call(config, "color")
    ) {
      statusMessage.style.color = config.color;
    }
    if (
      typeof config.clearColorAfterMs === "number"
      && config.clearColorAfterMs > 0
    ) {
      setTimeout(function () {
        statusMessage.style.color = "";
      }, config.clearColorAfterMs);
    }
  }

  function statusWordPass(chip, base, cycle) {
    revealText(chip._textEl, base, function () {
      if (!cycle) {
        return;
      }
      chip._cycleTimer = setTimeout(function () {
        statusWordPass(chip, base, true);
      }, STATUS_CYCLE_HOLD_MS);
    }, false);
  }

  function startStatusDots(chip, base) {
    stopStatusDots(chip);
    chip._dotsCount = 3;
    var render = function () {
      chip._dotsEl.textContent =
        ".".repeat(chip._dotsCount);
      chip._dotsCount = (chip._dotsCount + 1) % 4;
    };
    render();
    chip._dotsTimer =
      setInterval(render, STATUS_DOTS_MS);
    var cycle = readDiffusionEffect()
      && readDiffusionTextMode() === "cycle";
    statusWordPass(chip, base, cycle);
  }

  function stopStatusDots(chip) {
    cancelReveal(chip._textEl);
    if (chip._dotsTimer) {
      clearInterval(chip._dotsTimer);
      chip._dotsTimer = null;
    }
    if (chip._cycleTimer) {
      clearTimeout(chip._cycleTimer);
      chip._cycleTimer = null;
    }
  }

  function pushStatus(text) {
    var chip = document.createElement("span");
    chip.className = "status-chip";
    chip._textEl = document.createElement("span");
    chip._textEl.className = "status-chip-text";
    chip._dotsEl = document.createElement("span");
    chip._dotsEl.className = "status-chip-dots";
    chip.appendChild(chip._textEl);
    chip.appendChild(chip._dotsEl);
    statusRowReflow(function () {
      statusStack.insertBefore(chip, statusMessage);
      statusChips.push(chip);
    });
    statusStackTrim();
    startStatusDots(chip, text);
    void chip.offsetWidth;
    chip.classList.add("is-visible");
    return chip;
  }

  function retireStatus(chip) {
    if (!chip || statusChips.indexOf(chip) === -1) {
      return;
    }
    statusChipDismiss(chip);
  }

  function statusStackTrim() {
    while (statusChips.length > STATUS_STACK_MAX) {
      statusChipDismiss(statusChips[0]);
    }
  }

  function startRunStatus(text) {
    retireStatus(runStatusHandle);
    runStatusHandle = pushStatus(text);
  }

  function endRunStatus() {
    elapsedSettle();
    retireStatus(runStatusHandle);
    runStatusHandle = null;
  }

  function statusChipDismiss(chip) {
    var at = statusChips.indexOf(chip);
    if (at !== -1) {
      statusChips.splice(at, 1);
    }
    stopStatusDots(chip);
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

  function resetStatus(tpsMode) {
    elapsedStop();
    statusStep.textContent = "Step -/-";
    statusElapsed.textContent = "Elapsed: -";
    renderTpsFooter(null, tpsMode);
    statusMessage.textContent = "";
    statusMessage.style.color = "";
  }

  function readStatus() {
    return {
      step: statusStep.textContent,
      elapsed: statusElapsed.textContent,
      message: statusMessage.textContent,
    };
  }

  function restoreStatus(state) {
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorChrome.restoreStatus needs status state"
      );
    }
    if (state.step) {
      statusStep.textContent = state.step;
    }
    if (state.elapsed) {
      statusElapsed.textContent = state.elapsed;
    }
    renderTpsFooter(state.rate, state.tpsMode);
    if (state.message) {
      statusMessage.textContent = state.message;
    }
  }

  function refreshAnalyticsCue() {
    var count = persistNewRunCount();
    analyticsNewDot.textContent =
      count > 0 ? String(count) : "";
    analyticsNewDot.classList.toggle("is-empty", count === 0);
  }

  function flashAnalyticsPlusOne() {
    if (readReducedMotion()) {
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
    setTimeout(function () {
      if (plus.parentNode) {
        plus.remove();
      }
    }, ANALYTICS_PLUS_ONE_MS);
  }

  function showAnalyticsCue(runId) {
    var added = persistAddNewRun(runId);
    refreshAnalyticsCue();
    if (added) {
      flashAnalyticsPlusOne();
    }
  }

  return {
    wire: wire,
    boot: boot,
    showOutputPlaceholder: showOutputPlaceholder,
    showNoOutput: showNoOutput,
    setConnection: setConnection,
    setLoadingText: setLoadingText,
    setLoadingProgress: setLoadingProgress,
    finishLoadingProgress: finishLoadingProgress,
    showLoading: showLoading,
    hideLoading: hideLoading,
    handleResourceSample: handleResourceSample,
    clearResourceMeter: clearResourceMeter,
    setStep: setStep,
    updateRateFooter: updateRateFooter,
    renderTpsFooter: renderTpsFooter,
    setMessage: setMessage,
    pushStatus: pushStatus,
    retireStatus: retireStatus,
    startRunStatus: startRunStatus,
    endRunStatus: endRunStatus,
    resetStatus: resetStatus,
    readStatus: readStatus,
    restoreStatus: restoreStatus,
    refreshAnalyticsCue: refreshAnalyticsCue,
    showAnalyticsCue: showAnalyticsCue,
  };
}
