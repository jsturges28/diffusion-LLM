// The generator's token canvas and visual-overlay controls.
//
// Loaded as a classic script after overlays.js, candidate_flicker.js
// and generator_run.js, and before app.js. The returned controller
// owns output rendering, overlay selection, comparison layers and
// every memo derived from run frames. It reads run records only
// through generatorRun's public API. The page supplies narrow model,
// settings, edit, candidate-cycle and readout callbacks.

"use strict";

function generatorCanvasCreate(options) {
  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorCanvasCreate needs options." + name
      );
    }
    return options[name];
  }

  if (!options || !options.run) {
    throw new TypeError(
      "generatorCanvasCreate needs options.run"
    );
  }

  var run = options.run;
  var readModel = requiredCallback("readModel");
  var readSettings = requiredCallback("readSettings");
  var readEdit = requiredCallback("readEdit");
  var readReducedMotion =
    requiredCallback("readReducedMotion");
  var writeHighlight =
    requiredCallback("writeHighlight");
  var startCandidates =
    requiredCallback("startCandidates");
  var stopCandidates =
    requiredCallback("stopCandidates");
  var onOutputReset =
    requiredCallback("onOutputReset");
  var onRender = requiredCallback("onRender");
  var onOverlayChanged =
    requiredCallback("onOverlayChanged");
  var onLayerChanged =
    requiredCallback("onLayerChanged");

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing generator canvas element #" + id
      );
    }
    return element;
  }

  var outputArea = requiredElement("output-area");
  var overlaySelectGroup =
    requiredElement("overlay-select-group");
  var overlayDrawerHandle =
    requiredElement("overlay-drawer-handle");
  var overlaySelectMount =
    requiredElement("overlay-select-mount");
  var overlayHighlightCheckbox =
    requiredElement("overlay-highlight-tokens");
  var diffSummary = requiredElement("diff-summary");
  var commitLegend = requiredElement("commit-legend");
  var revisionLegend = requiredElement("revision-legend");
  var watermarkLegend = requiredElement("watermark-legend");
  var diffOverlayControls =
    requiredElement("diff-overlay-controls");
  var diffOriginalSlider =
    requiredElement("diff-original-opacity");
  var diffEditedSlider =
    requiredElement("diff-edited-opacity");
  var diffBlendToggle =
    requiredElement("diff-blend-toggle");
  var runBlendRow = requiredElement("run-blend-row");
  var runBlendInput = requiredElement("run-blend");
  var outputSection = requiredElement("output-section");

  var ENTROPY_SHAPES = ["position", "frame|position"];
  var TOKEN_BIRTH_RATE_CEILING = 96;
  var TOKEN_BIRTH_CONCURRENT_MIN = 48;
  var TOKEN_BIRTH_CONCURRENT_MAX = 192;
  var TOKEN_BIRTH_ANIMATION = "token-birth";
  var TOKEN_REVISION_ANIMATION = "token-revision";

  var overlayMode = "none";
  var overlaySelect = null;
  var diffOriginalOpacity = 50;
  var diffEditedOpacity = 100;
  var diffBlend = false;
  var runBlend = 1;

  var commitSteps = null;
  var originalCommitSteps = null;
  var diffData = null;
  var runRevisions = null;
  var originalRevisions = null;
  var revisionCounts = {
    original: null,
    branch: null,
  };
  var entropyBorrowSlots = {
    edited: null,
    original: null,
  };
  var editedMarksCache = {
    log: null,
    count: -1,
    marks: {},
  };

  var liveTokenSpans = [];
  var tokenGlowQueue = [];
  var tokenGlowMaxConcurrent = TOKEN_BIRTH_CONCURRENT_MIN;
  var liveRevisionFold = null;
  var liveRevisionFrames = 0;
  var viewMode = "none";
  var drawnFrame = -1;
  var wired = false;

  var liveTokenOptions = {
    revealMask: false,
    opacityFor: tokenOpacity,
  };

  function modelState() {
    var state = readModel();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorCanvas readModel must return an object"
      );
    }
    return {
      capabilities: state.capabilities || {},
      maskChar:
        typeof state.maskChar === "string"
          ? state.maskChar
          : "\u2591",
    };
  }

  function settingsState() {
    var state = readSettings();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorCanvas readSettings must return an object"
      );
    }
    return state;
  }

  function editState() {
    var state = readEdit();
    if (!state || !Array.isArray(state.remaskEdits)) {
      throw new TypeError(
        "generatorCanvas edit state needs remaskEdits"
      );
    }
    return {
      remaskEdits: state.remaskEdits,
      remaskedPositions: state.remaskedPositions || {},
      mode:
        typeof state.mode === "string" ? state.mode : null,
      substituting: state.substituting === true,
      generating: state.generating === true,
    };
  }

  function maskChar() {
    return modelState().maskChar;
  }

  function isAppendOnly() {
    var capabilities = modelState().capabilities;
    return capabilities.generation_shape === "append_only";
  }

  function singleCanvas() {
    return 0;
  }

  function canvasAt(frame) {
    return run.frameCanvas(frame);
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    overlaysMakeDrawerDraggable({
      group: overlaySelectGroup,
      handle: overlayDrawerHandle,
      container: outputSection,
      storageKey: "diffusion_overlay_drawer_top_generator",
      onToggle: setDrawerOpen,
    });
    overlayHighlightCheckbox.addEventListener(
      "change", highlightChanged
    );
    diffOriginalSlider.addEventListener(
      "input", diffOriginalChanged
    );
    diffEditedSlider.addEventListener(
      "input", diffEditedChanged
    );
    diffBlendToggle.addEventListener(
      "change", diffBlendChanged
    );
    runBlendInput.addEventListener(
      "input", runBlendChanged
    );
    outputArea.addEventListener(
      "animationend", tokenGlowEnded
    );
  }

  function highlightChanged() {
    writeHighlight(overlayHighlightCheckbox.checked);
    updateHoverHighlight();
  }

  function diffOriginalChanged() {
    diffOriginalOpacity =
      parseInt(diffOriginalSlider.value, 10);
    setLayerOpacity(
      ".token-layer-original", diffOriginalOpacity
    );
    applyDiffLayerPointers();
  }

  function diffEditedChanged() {
    diffEditedOpacity =
      parseInt(diffEditedSlider.value, 10);
    setLayerOpacity(
      ".token-layer-edited", diffEditedOpacity
    );
    applyDiffLayerPointers();
  }

  function setLayerOpacity(selector, percent) {
    var layer = outputArea.querySelector(selector);
    if (layer) {
      layer.style.opacity = String(percent / 100);
    }
  }

  function applyDiffLayerPointers() {
    overlaysApplyLayerPointers(
      outputArea,
      diffOriginalOpacity,
      diffEditedOpacity
    );
    onLayerChanged({ profile: false });
  }

  function diffBlendChanged() {
    diffBlend = diffBlendToggle.checked;
    if (viewMode === "scrub" && overlayMode === "diff") {
      renderFrame(drawnFrame);
    }
  }

  function runBlendChanged() {
    setBlend(Number(runBlendInput.value) / 100);
  }

  function setBlend(value) {
    if (typeof value !== "number" || !isFinite(value)) {
      throw new TypeError(
        "generatorCanvas.setBlend needs a number"
      );
    }
    runBlend = Math.max(0, Math.min(1, value));
    runBlendInput.value = String(Math.round(runBlend * 100));
    if (overlayMode !== "diff") {
      applyRunBlendToLayers();
    }
    onLayerChanged({ profile: true });
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

  function applyModel() {
    var capabilities = modelState().capabilities;
    var glow = overlaysGlowFor(
      settingsState(), capabilities.family
    );
    overlaysApplyGlowVars(
      outputArea, glow.brightness, glow.fadeMs
    );
    tokenGlowMaxConcurrent =
      tokenGlowConcurrentCap(glow.fadeMs);
  }

  function applySettings() {
    var settings = settingsState();
    liveTokenOptions.revealMask =
      overlaysDrawsGuess(settings);
    updateHoverHighlight();
    applyModel();
    if (viewMode === "scrub" && drawnFrame >= 0) {
      renderFrame(drawnFrame);
    }
  }

  function updateHoverHighlight() {
    var highlighted = !!settingsState().highlightTokens;
    overlayHighlightCheckbox.checked = highlighted;
    outputArea.classList.toggle(
      "token-hover-highlight", highlighted
    );
  }

  function tokenGlowConcurrentCap(fadeMs) {
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

  function renderLiveFrame(tokens, revealed, live) {
    stopCandidates();
    viewMode = "live";
    drawnFrame = run.frameCount() - 1;
    outputArea.classList.remove("token-layers");
    outputArea.classList.add("live-tokens");
    var reusable =
      liveTokenSpans.length === tokens.length
      && liveTokenSpans.length > 0
      && liveTokenSpans[0].parentNode === outputArea;
    if (reusable) {
      syncLiveTokens(tokens);
    } else {
      rebuildLiveTokens(tokens);
    }
    markTokenBirths(revealed);
    if (!run.frameIsAppend()) {
      markTokenRevisions(
        liveRevisionsAt(drawnFrame, tokens)
      );
    }
    onRender();
    startLiveCycling(tokens, live);
  }

  function syncLiveTokens(tokens) {
    for (var index = 0; index < tokens.length; index++) {
      overlaysSyncTokenSpan(
        liveTokenSpans[index],
        index,
        tokens[index],
        maskChar(),
        liveTokenOptions
      );
    }
  }

  function rebuildLiveTokens(tokens) {
    var fragment = document.createDocumentFragment();
    liveTokenSpans = new Array(tokens.length);
    for (var index = 0; index < tokens.length; index++) {
      var span = overlaysBuildTokenSpan(
        index,
        tokens[index],
        maskChar(),
        liveTokenOptions
      );
      liveTokenSpans[index] = span;
      fragment.appendChild(span);
    }
    tokenGlowQueue = [];
    onOutputReset();
    outputArea.textContent = "";
    outputArea.appendChild(fragment);
  }

  function markTokenBirths(revealed) {
    if (!revealed || revealed.length === 0) {
      return;
    }
    if (!settingsState().tokenBirthGlow) {
      return;
    }
    if (readReducedMotion()) {
      return;
    }
    for (var index = 0; index < revealed.length; index++) {
      var span = liveTokenSpans[revealed[index]];
      if (span) {
        startTokenGlow(
          span, "data-born", "data-revised"
        );
      }
    }
  }

  function markTokenRevisions(revised) {
    if (!revised || revised.length === 0) {
      return;
    }
    if (!settingsState().revisionGlow) {
      return;
    }
    if (readReducedMotion()) {
      return;
    }
    for (var index = 0; index < revised.length; index++) {
      var span = liveTokenSpans[revised[index]];
      if (span) {
        startTokenGlow(
          span, "data-revised", "data-born"
        );
      }
    }
  }

  function startTokenGlow(span, attribute, other) {
    if (span.hasAttribute(attribute)) {
      return;
    }
    if (span.hasAttribute(other)) {
      endTokenGlow(span);
    }
    span.setAttribute(attribute, "");
    tokenGlowQueue.push(span);
    while (tokenGlowQueue.length > tokenGlowMaxConcurrent) {
      endTokenGlow(tokenGlowQueue[0]);
    }
  }

  function endTokenGlow(span) {
    span.removeAttribute("data-born");
    span.removeAttribute("data-revised");
    var index = tokenGlowQueue.indexOf(span);
    if (index !== -1) {
      tokenGlowQueue.splice(index, 1);
    }
  }

  function tokenGlowEnded(event) {
    if (
      event.animationName !== TOKEN_BIRTH_ANIMATION
      && event.animationName !== TOKEN_REVISION_ANIMATION
    ) {
      return;
    }
    endTokenGlow(event.target);
  }

  function liveRevisionsAt(index, tokens) {
    if (
      liveRevisionFold === null
      || liveRevisionFrames !== index
    ) {
      liveRevisionFold = liveRevisionFoldThrough(index);
    }
    var step = overlaysRevisionStep(
      liveRevisionFold,
      tokens,
      canvasAt(index),
      overlaysRemaskedAt(editState().remaskEdits, index)
    );
    liveRevisionFold = step.fold;
    liveRevisionFrames = index + 1;
    return step.revised;
  }

  function liveRevisionFoldThrough(count) {
    var fold = overlaysRevisionFold();
    var edits = editState().remaskEdits;
    for (var frame = 0; frame < count; frame++) {
      fold = overlaysRevisionStep(
        fold,
        run.frameTokens(frame) || [],
        canvasAt(frame),
        overlaysRemaskedAt(edits, frame)
      ).fold;
    }
    return fold;
  }

  function renderTextFrame(text) {
    viewMode = "live";
    drawnFrame = run.frameCount() - 1;
    drawText(text);
    onRender();
  }

  function drawText(text) {
    stopCandidates();
    outputArea.classList.remove("token-layers");
    outputArea.classList.remove("live-tokens");
    var fragment = document.createDocumentFragment();
    var unresolved = maskChar();
    for (var index = 0; index < text.length; index++) {
      var character = text[index];
      if (character === unresolved) {
        fragment.appendChild(
          textSpan("char-mask", character)
        );
      } else if (character === "\n") {
        fragment.appendChild(
          document.createTextNode("\n")
        );
      } else {
        fragment.appendChild(
          textSpan("char-resolved", character)
        );
      }
    }
    onOutputReset();
    outputArea.textContent = "";
    outputArea.appendChild(fragment);
  }

  function textSpan(className, text) {
    var span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    return span;
  }

  function renderFinalText(text) {
    viewMode = "final";
    drawnFrame = run.frameCount() - 1;
    stopCandidates();
    outputArea.classList.remove("token-layers");
    outputArea.classList.remove("live-tokens");
    onOutputReset();
    outputArea.textContent = "";
    outputArea.appendChild(
      textSpan("char-resolved", text)
    );
    onRender();
  }

  function invalidate() {
    commitSteps = null;
    originalCommitSteps = null;
    diffData = null;
    runRevisions = null;
    originalRevisions = null;
    revisionCounts = {
      original: null,
      branch: null,
    };
    entropyBorrowSlots = {
      edited: null,
      original: null,
    };
    editedMarksCache = {
      log: null,
      count: -1,
      marks: {},
    };
    liveRevisionFold = null;
    liveRevisionFrames = 0;
  }

  function computeCommitSteps() {
    if (run.frameIsAppend()) {
      return overlaysAppendCommitSteps(
        run.framePositions()
      );
    }
    var series = run.frameTokenSeries();
    return overlaysComputeCommitSteps(
      overlaysFrameReader(series), series.length
    );
  }

  function commitStepsFor(isOriginal) {
    if (isOriginal) {
      if (originalCommitSteps === null) {
        originalCommitSteps =
          computeOriginalCommitSteps();
      }
      return originalCommitSteps;
    }
    if (commitSteps === null) {
      commitSteps = computeCommitSteps();
    }
    return commitSteps;
  }

  function computeOriginalCommitSteps() {
    if (run.originalIsAppend()) {
      return overlaysAppendCommitSteps(
        run.originalPositions()
      );
    }
    var series = run.originalTokenSeries();
    return overlaysComputeCommitSteps(
      overlaysFrameReader(series), series.length
    );
  }

  function tokenCommitStep(index, isOriginal) {
    var step = commitStepsFor(isOriginal)[index];
    if (typeof step !== "number" || step < 0) {
      return null;
    }
    return step;
  }

  function computeRevisions() {
    if (run.frameIsAppend()) {
      return [];
    }
    var series = run.frameTokenSeries();
    return overlaysComputeRevisions(
      overlaysFrameReader(series),
      series.length,
      canvasAt,
      editState().remaskEdits
    );
  }

  function revisionsFor(isOriginal) {
    if (isOriginal) {
      if (originalRevisions === null) {
        originalRevisions = computeOriginalRevisions();
      }
      return originalRevisions;
    }
    if (
      runRevisions === null
      || runRevisions.length !== run.frameCount()
    ) {
      runRevisions = computeRevisions();
    }
    return runRevisions;
  }

  function computeOriginalRevisions() {
    if (run.originalIsAppend()) {
      return [];
    }
    var series = run.originalTokenSeries();
    return overlaysComputeRevisions(
      overlaysFrameReader(series),
      series.length,
      singleCanvas,
      []
    );
  }

  function revisionsAvailable() {
    if (isAppendOnly()) {
      return false;
    }
    return overlaysHasRevisions(revisionsFor(false));
  }

  function layerFrame(isOriginal) {
    if (!isOriginal) {
      return drawnFrame;
    }
    return Math.min(
      drawnFrame, run.originalTokenFrames() - 1
    );
  }

  function revisionCountsFor(isOriginal) {
    var key = isOriginal ? "original" : "branch";
    var frame = layerFrame(isOriginal);
    var held = revisionCounts[key];
    if (held === null || held.frame !== frame) {
      held = {
        frame: frame,
        counts: overlaysRevisionCounts(
          revisionsFor(isOriginal),
          frame,
          isOriginal ? singleCanvas : canvasAt
        ),
      };
      revisionCounts[key] = held;
    }
    return held.counts;
  }

  function tokenRevisionCount(index, isOriginal) {
    if (editState().generating) {
      return 0;
    }
    var count = revisionCountsFor(isOriginal)[index];
    return typeof count === "number" ? count : 0;
  }

  function computeDiff() {
    return overlaysComputeDiff(
      run.frameTokensLast(),
      run.originalTokensLast(),
      editState().remaskEdits
    );
  }

  function currentDiffData() {
    if (diffData === null) {
      diffData = computeDiff();
    }
    return diffData;
  }

  function readDiffData() {
    var diff = currentDiffData();
    return {
      changed: diff.changed.slice(),
      origText: diff.origText.slice(),
      origins: Object.assign({}, diff.origins),
      changedCount: diff.changedCount,
      totalCount: diff.totalCount,
    };
  }

  function diffAvailable() {
    return (
      run.originalCaptured()
      && editState().remaskEdits.length > 0
      && run.originalTokenFrames() > 0
    );
  }

  function declaredChannel(name) {
    var signals = modelState().capabilities.signals || [];
    for (var index = 0; index < signals.length; index++) {
      if (signals[index] && signals[index].name === name) {
        return signals[index];
      }
    }
    return null;
  }

  function entropyDeclared() {
    var channel = declaredChannel("entropy");
    if (!channel) {
      return false;
    }
    return ENTROPY_SHAPES.indexOf(
      (channel.axes || []).join("|")
    ) !== -1;
  }

  function entropyAvailable() {
    var channel = declaredChannel("entropy");
    if (channel) {
      var shape = (channel.axes || []).join("|");
      if (ENTROPY_SHAPES.indexOf(shape) === -1) {
        return false;
      }
    }
    return runEntropyFrame(
      run.frameCount() - 1, singleCanvas
    ) >= 0;
  }

  function runEntropyFrame(index, canvasOf) {
    return overlaysEntropyFrame(
      function (frame) {
        return run.frameTokens(frame);
      },
      canvasOf,
      index,
      run.frameIsAppend()
    );
  }

  function originalEntropyFrame(index) {
    return overlaysEntropyFrame(
      function (frame) {
        return run.originalTokens(frame);
      },
      singleCanvas,
      index,
      run.originalIsAppend()
    );
  }

  function drawnEntropyFrame(isOriginal) {
    var frame = viewMode === "scrub"
      ? drawnFrame
      : run.frameCount() - 1;
    if (!isOriginal) {
      return frame;
    }
    return Math.min(
      frame, run.originalTokenFrames() - 1
    );
  }

  function layerEntropyBorrow(isOriginal) {
    var key = isOriginal ? "original" : "edited";
    var frame = drawnEntropyFrame(isOriginal);
    var slot = entropyBorrowSlots[key];
    if (slot !== null && slot.frame === frame) {
      return slot.borrow;
    }
    var borrow = entropyBorrowAt(isOriginal, frame);
    entropyBorrowSlots[key] = {
      frame: frame,
      borrow: borrow,
    };
    return borrow;
  }

  function entropyBorrowAt(isOriginal, frame) {
    var source = isOriginal
      ? originalEntropyFrame(frame)
      : runEntropyFrame(frame, canvasAt);
    if (source < 0 || source === frame) {
      return null;
    }
    var tokens = isOriginal
      ? run.originalTokens(source)
      : run.frameTokens(source);
    return { step: source, tokens: tokens };
  }

  function entropyReading(index, token, isOriginal) {
    if (token && typeof token.e === "number") {
      return { value: token.e, asOfStep: null };
    }
    var borrow = layerEntropyBorrow(isOriginal);
    var other =
      borrow && borrow.tokens
        ? borrow.tokens[index]
        : null;
    if (other && typeof other.e === "number") {
      return {
        value: other.e,
        asOfStep: borrow.step,
      };
    }
    return { value: null, asOfStep: null };
  }

  function entropyValues(tokens) {
    if (!tokens) {
      return [];
    }
    var values = [];
    for (var index = 0; index < tokens.length; index++) {
      values.push(
        typeof tokens[index].e === "number"
          ? tokens[index].e
          : 0
      );
    }
    return values;
  }

  function entropyFollowsFrame() {
    var channel = declaredChannel("entropy");
    if (channel) {
      return (
        (channel.axes || []).join("|")
        === "frame|position"
      );
    }
    return !run.frameIsAppend();
  }

  function entropyLayer(isOriginal) {
    var borrow = layerEntropyBorrow(isOriginal);
    var source = borrow
      ? borrow.step
      : drawnEntropyFrame(isOriginal);
    var tokens = isOriginal
      ? run.originalTokens(source)
      : run.frameTokens(source);
    return {
      values: entropyValues(tokens),
      asOfStep: borrow ? borrow.step : null,
    };
  }

  function entropyProfile() {
    if (!entropyFollowsFrame()) {
      return entropyPositionProfile();
    }
    var edited = entropyLayer(false);
    var original = blendActive()
      ? entropyLayer(true)
      : { values: [], asOfStep: null };
    return {
      values: edited.values,
      original: original.values,
      current: -1,
      filled: -1,
      asOfStep: edited.asOfStep,
      originalAsOfStep: original.asOfStep,
    };
  }

  function entropyPositionProfile() {
    return {
      values: entropyValues(run.frameTokensLast()),
      original: blendActive()
        ? entropyValues(run.originalTokensLast())
        : [],
      current: drawnFrame,
      filled: drawnFrame,
      asOfStep: null,
      originalAsOfStep: null,
    };
  }

  function forgettingAvailable() {
    var channel = declaredChannel("forgetting");
    if (!channel) {
      return false;
    }
    if ((channel.axes || []).join("|") !== "position") {
      return false;
    }
    return runCarriesTokenValue("f");
  }

  function watermarkAvailable() {
    var membership = declaredChannel("watermark_membership");
    var evidence = declaredChannel("watermark_evidence");
    if (!membership || !evidence) {
      return false;
    }
    if ((membership.axes || []).join("|") !== "position") {
      return false;
    }
    if ((evidence.axes || []).join("|") !== "position") {
      return false;
    }
    return overlaysTokensCarryWatermark(run.frameTokensLast());
  }

  function runCarriesTokenValue(key) {
    var tokens = run.frameTokensLast();
    if (!tokens) {
      return false;
    }
    for (var index = 0; index < tokens.length; index++) {
      if (
        tokens[index]
        && typeof tokens[index][key] === "number"
      ) {
        return true;
      }
    }
    return false;
  }

  function effectiveColorMode() {
    if (overlayMode === "commit" && isAppendOnly()) {
      return "none";
    }
    if (overlayMode === "revisions" && isAppendOnly()) {
      return "none";
    }
    return overlayMode;
  }

  function tokenColor(index, token, isOriginal) {
    var mode = effectiveColorMode();
    if (mode === "conf") {
      return typeof token.c === "number"
        ? heatColor(token.c)
        : null;
    }
    if (mode === "entropy") {
      var entropy =
        entropyReading(index, token, isOriginal).value;
      return entropy === null ? null : entropyColor(entropy);
    }
    if (mode === "forgetting") {
      return typeof token.f === "number"
        ? forgettingColor(token.f)
        : null;
    }
    if (mode === "watermark") {
      return watermarkColor(token);
    }
    return tokenFrameColor(index, isOriginal, mode);
  }

  function tokenFrameColor(index, isOriginal, mode) {
    if (mode === "commit") {
      var step = tokenCommitStep(index, isOriginal);
      if (step === null) {
        return null;
      }
      var count = isOriginal
        ? run.originalTokenFrames()
        : run.frameCount();
      return commitColor(step, count - 1);
    }
    if (mode === "revisions") {
      return revisionColor(
        tokenRevisionCount(index, isOriginal)
      );
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

  function tokenExtra(index, token, isOriginal) {
    var mode = effectiveColorMode();
    if (mode === "forgetting") {
      return overlaysForgettingReading(token);
    }
    if (mode === "watermark") {
      return overlaysWatermarkReading(token);
    }
    if (mode === "commit") {
      var step = tokenCommitStep(index, isOriginal);
      return step === null
        ? ""
        : "Resolved at step: " + step;
    }
    if (mode === "revisions") {
      return overlaysRevisionReading(
        tokenRevisionCount(index, isOriginal)
      );
    }
    return diffExtra(index, mode);
  }

  function diffExtra(index, mode) {
    if (mode !== "diff" || !diffAvailable()) {
      return "";
    }
    var diff = currentDiffData();
    if (diff.origins[index]) {
      return "(remasked here)";
    }
    if (diff.changed[index]) {
      return "was: " + diff.origText[index];
    }
    return "";
  }

  function editedPositionMarks() {
    var edits = editState().remaskEdits;
    if (
      editedMarksCache.log === edits
      && editedMarksCache.count === edits.length
    ) {
      return editedMarksCache.marks;
    }
    var marks = {};
    for (var edit = 0; edit < edits.length; edit++) {
      var positions = edits[edit].token_positions || [];
      for (
        var position = 0;
        position < positions.length;
        position++
      ) {
        marks[positions[position]] =
          edits[edit].frame_index;
      }
    }
    editedMarksCache = {
      log: edits,
      count: edits.length,
      marks: marks,
    };
    return marks;
  }

  function readEditedPositionMarks() {
    return Object.assign({}, editedPositionMarks());
  }

  function positionWasEdited(position) {
    return (
      typeof editedPositionMarks()[position] === "number"
    );
  }

  function tokenLayerOptions(isOriginal) {
    var edit = editState();
    return {
      maskChar: maskChar(),
      revealMask: overlaysDrawsGuess(settingsState()),
      maskedFor: function (index) {
        return edit.remaskedPositions[index] === true;
      },
      classFor: function (index, token, masked) {
        return tokenClass(index, token, masked, edit);
      },
      opacityFor: function (index, token, masked) {
        return tokenOpacityWithEdit(
          index, token, masked, edit
        );
      },
      colorFor: function (index, token) {
        if (
          !token
          || token.m
          || edit.remaskedPositions[index] === true
        ) {
          return null;
        }
        return tokenColor(index, token, isOriginal);
      },
      descriptionFor: function (index, token, masked) {
        if (masked || effectiveColorMode() !== "watermark") {
          return "";
        }
        return overlaysWatermarkDescription(token);
      },
    };
  }

  function tokenClass(index, token, masked, edit) {
    if (edit.remaskedPositions[index] === true) {
      return "token-remasked";
    }
    if (masked) {
      return "";
    }
    var classes = [];
    if (overlayMode === "watermark") {
      var watermarkClass = overlaysWatermarkTokenClass(token);
      if (watermarkClass) {
        classes.push(watermarkClass);
      }
    }
    if (positionWasEdited(index)) {
      classes.push("token-edited");
    }
    if (edit.mode === "edit") {
      classes.push("token-clickable");
    }
    if (
      edit.substituting
      && run.positionAlternatives(index, false)
    ) {
      classes.push("token-substitutable");
    }
    return classes.join(" ");
  }

  function tokenOpacity(index, token, masked) {
    return tokenOpacityWithEdit(
      index, token, masked, editState()
    );
  }

  function tokenOpacityWithEdit(index, token, masked, edit) {
    if (
      !masked
      || edit.remaskedPositions[index] === true
    ) {
      return null;
    }
    if (!token) {
      return MASK_OPACITY_FLOOR;
    }
    return overlaysMaskOpacity(token.c);
  }

  function blendActive() {
    return diffAvailable() && editState().mode === null;
  }

  function blendFavorsOriginal() {
    return runBlend < 0.5;
  }

  function layersActive() {
    return viewMode === "scrub" && blendActive();
  }

  function layerIsOriginal(target) {
    if (target && target.closest) {
      var layer = target.closest(".token-layer");
      if (layer) {
        return layer.classList.contains(
          "token-layer-original"
        );
      }
    }
    if (!layersActive()) {
      return false;
    }
    if (overlayMode === "diff") {
      return !overlaysEditedOwnsPointer(
        diffOriginalOpacity, diffEditedOpacity
      );
    }
    return !overlaysEditedOwnsPointer(
      1 - runBlend, runBlend
    );
  }

  function drawnTokens(isOriginal) {
    if (viewMode !== "scrub") {
      return run.frameTokensLast();
    }
    var frame = layerFrame(isOriginal);
    if (frame < 0) {
      return null;
    }
    return isOriginal
      ? run.originalTokens(frame)
      : run.frameTokens(frame);
  }

  function renderFrame(frameIndex) {
    viewMode = "scrub";
    drawnFrame = frameIndex;
    stopCandidates();
    outputArea.classList.remove("live-tokens");
    var tokens = run.frameTokens(frameIndex);
    if (!tokens) {
      drawText(run.frameText(frameIndex));
      onRender();
      return;
    }
    if (overlayMode === "diff" && blendActive()) {
      renderDiffOverlay(frameIndex, tokens);
      onRender();
      return;
    }
    renderTokenFrame(frameIndex, tokens);
    onRender();
  }

  function renderTokenFrame(frameIndex, tokens) {
    onOutputReset();
    outputArea.textContent = "";
    if (blendActive()) {
      renderCrossfadedFrame(frameIndex, tokens);
      return;
    }
    outputArea.classList.remove("token-layers");
    var options = tokenLayerOptions(false);
    var fragment = document.createDocumentFragment();
    var spans = [];
    for (var index = 0; index < tokens.length; index++) {
      var span = overlaysBuildTokenSpan(
        index, tokens[index], maskChar(), options
      );
      spans.push(span);
      fragment.appendChild(span);
    }
    outputArea.appendChild(fragment);
    startSingleFlicker(spans, tokens, frameIndex);
  }

  function startSingleFlicker(spans, tokens, frameIndex) {
    if (!candidatesCycle()) {
      return;
    }
    startCandidates([{
      spans: spans,
      tokens: tokens,
      sets: run.candidateSets(
        frameIndex, false, canvasAt
      ),
    }], maskChar());
  }

  function renderCrossfadedFrame(frameIndex, editedTokens) {
    outputArea.classList.add("token-layers");
    var layered =
      buildCrossfadedLayers(frameIndex, editedTokens);
    var stacked = [
      layered.children[0],
      layered.children[1],
    ];
    outputArea.appendChild(layered);
    startStackedFlicker(
      stacked, frameIndex, editedTokens
    );
  }

  function buildCrossfadedLayers(frameIndex, editedTokens) {
    var originalFrame = Math.min(
      frameIndex, run.originalTokenFrames() - 1
    );
    var originalTokens =
      (
        originalFrame >= 0
          ? run.originalTokens(originalFrame)
          : null
      ) || [];
    var editedTakes = overlaysEditedOwnsPointer(
      1 - runBlend, runBlend
    );
    var fragment = document.createDocumentFragment();
    var originalOptions = tokenLayerOptions(true);
    originalOptions.layerClass = "token-layer-original";
    originalOptions.opacity = 1 - runBlend;
    originalOptions.interactive = !editedTakes;
    fragment.appendChild(
      overlaysBuildTokenLayer(
        originalTokens, originalOptions
      )
    );
    var editedOptions = tokenLayerOptions(false);
    editedOptions.layerClass = "token-layer-edited";
    editedOptions.opacity = runBlend;
    editedOptions.interactive = editedTakes;
    fragment.appendChild(
      overlaysBuildTokenLayer(editedTokens, editedOptions)
    );
    return fragment;
  }

  function renderDiffOverlay(frameIndex, editedTokens) {
    onOutputReset();
    outputArea.textContent = "";
    outputArea.classList.add("token-layers");
    var originalFrame = Math.min(
      frameIndex, run.originalTokenFrames() - 1
    );
    var originalTokens =
      (
        originalFrame >= 0
          ? run.originalTokens(originalFrame)
          : null
      ) || [];
    var layered = overlaysBuildDiffLayers(
      originalTokens,
      editedTokens,
      currentDiffData(),
      {
        originalOpacity: diffOriginalOpacity,
        editedOpacity: diffEditedOpacity,
        blend: diffBlend,
        revealMask: overlaysDrawsGuess(settingsState()),
        opacityFor: tokenOpacity,
      },
      maskChar()
    );
    var stacked = [
      layered.children[0],
      layered.children[1],
    ];
    outputArea.appendChild(layered);
    startStackedFlicker(
      stacked, frameIndex, editedTokens
    );
  }

  function candidatesCycle() {
    var edit = editState();
    return (
      settingsState().unsettledShows === "candidates"
      && viewMode === "scrub"
      && !edit.generating
      && edit.mode === null
    );
  }

  function startStackedFlicker(
    layers, frameIndex, editedTokens
  ) {
    if (!candidatesCycle()) {
      return;
    }
    var originalFrame = Math.min(
      frameIndex, run.originalTokenFrames() - 1
    );
    var originalTokens =
      run.originalTokens(originalFrame) || [];
    startCandidates([
      {
        spans: layers[0].children,
        tokens: originalTokens,
        sets: run.candidateSets(
          originalFrame, true, singleCanvas
        ),
      },
      {
        spans: layers[1].children,
        tokens: editedTokens,
        sets: run.candidateSets(
          frameIndex, false, canvasAt
        ),
      },
    ], maskChar());
  }

  function startLiveCycling(tokens, live) {
    if (
      settingsState().unsettledShows !== "candidates"
    ) {
      return;
    }
    var sets = liveCyclingSets(live, tokens.length);
    if (sets === null) {
      return;
    }
    startCandidates([{
      spans: liveTokenSpans,
      tokens: tokens,
      sets: sets,
    }], maskChar());
  }

  function liveCyclingSets(live, width) {
    if (!live || !Array.isArray(live.positions)) {
      return null;
    }
    if (!Array.isArray(live.sets)) {
      return null;
    }
    if (live.positions.length !== live.sets.length) {
      return null;
    }
    var sets = new Array(width).fill(null);
    for (var index = 0; index < live.positions.length; index++) {
      var position = live.positions[index];
      if (!Number.isInteger(position)) {
        return null;
      }
      if (position < 0 || position >= width) {
        return null;
      }
      sets[position] = live.sets[index];
    }
    return sets;
  }

  function setOverlayMode(mode) {
    overlayMode = mode;
    updateOverlayChrome();
    onOverlayChanged(mode);
    if (viewMode === "scrub" && drawnFrame >= 0) {
      renderFrame(drawnFrame);
    }
  }

  function updateOverlayChrome() {
    updateDiffSummary();
    updateDiffControls();
    updateRunBlendControls();
    updateOverlayLegends();
  }

  function updateOverlayLegends() {
    commitLegend.hidden = overlayMode !== "commit";
    revisionLegend.hidden = overlayMode !== "revisions";
    watermarkLegend.hidden = overlayMode !== "watermark";
  }

  function updateDiffControls() {
    diffOverlayControls.hidden = !(
      overlayMode === "diff"
      && diffAvailable()
      && editState().mode === null
    );
  }

  function updateRunBlendControls() {
    runBlendRow.hidden = !(
      overlayMode !== "diff" && blendActive()
    );
  }

  function resetRunBlend() {
    runBlend = 1;
    runBlendInput.value = "100";
    updateRunBlendControls();
  }

  function resetDiffOverlay() {
    diffOriginalOpacity = 50;
    diffEditedOpacity = 100;
    diffBlend = false;
    diffOriginalSlider.value = "50";
    diffEditedSlider.value = "100";
    diffBlendToggle.checked = false;
  }

  function updateDiffSummary() {
    if (overlayMode !== "diff") {
      diffSummary.hidden = true;
      return;
    }
    var diff = currentDiffData();
    var total = diff.totalCount;
    var changed = diff.changedCount;
    var percent = total > 0
      ? Math.round((changed / total) * 100)
      : 0;
    diffSummary.textContent =
      "Diverged " + changed + "/" + total
      + " (" + percent + "%)";
    diffSummary.hidden = false;
  }

  function setDrawerOpen(open) {
    overlaySelectGroup.classList.toggle("open", open);
    overlayDrawerHandle.textContent =
      open ? "\u203a" : "\u2039";
    overlayDrawerHandle.title = open
      ? "Collapse overlay options"
      : "Overlay options";
  }

  function rebuildOverlaySelect() {
    var availability = overlayAvailability();
    normalizeOverlayMode(availability);
    updateOverlayLegends();
    var selectOptions =
      overlayOptions(availability);
    overlaySelectMount.innerHTML = "";
    overlaySelect = createCustomSelect(
      selectOptions, overlayMode
    );
    overlaySelectMount.appendChild(overlaySelect);
    sizeCustomSelect(overlaySelect);
    overlaySelect.addEventListener("change", function () {
      setOverlayMode(overlaySelect.value);
    });
  }

  function overlayAvailability() {
    return {
      diff: diffAvailable(),
      entropy: entropyAvailable(),
      forgetting: forgettingAvailable(),
      watermark: watermarkAvailable(),
      revisions: revisionsAvailable(),
      append: isAppendOnly(),
    };
  }

  function normalizeOverlayMode(available) {
    if (overlayMode === "diff" && !available.diff) {
      overlayMode = "none";
    }
    if (overlayMode === "entropy" && !available.entropy) {
      overlayMode = "none";
    }
    if (
      overlayMode === "forgetting"
      && !available.forgetting
    ) {
      overlayMode = "none";
    }
    if (
      overlayMode === "watermark"
      && !available.watermark
    ) {
      overlayMode = "none";
    }
    if (
      overlayMode === "revisions"
      && !available.revisions
    ) {
      overlayMode = "none";
    }
    if (overlayMode === "commit" && available.append) {
      overlayMode = "none";
    }
  }

  function overlayOptions(available) {
    var selectOptions = [
      { value: "none", label: "None" },
      { value: "conf", label: "Heatmap" },
    ];
    if (available.entropy) {
      selectOptions.push({
        value: "entropy",
        label: "Entropy",
      });
    }
    if (available.forgetting) {
      selectOptions.push({
        value: "forgetting",
        label: "Forgetting",
      });
    }
    if (available.watermark) {
      selectOptions.push({
        value: "watermark",
        label: "Watermark",
      });
    }
    if (!available.append) {
      selectOptions.push({
        value: "commit",
        label: "Commit Order",
      });
    }
    if (available.revisions) {
      selectOptions.push({
        value: "revisions",
        label: "Revisions",
      });
    }
    appendDiffOption(selectOptions, available);
    return selectOptions;
  }

  function appendDiffOption(selectOptions, available) {
    if (available.append && !available.diff) {
      return;
    }
    selectOptions.push({
      value: "diff",
      label: "Diff vs Original",
      disabled: !available.diff,
      title: available.diff
        ? undefined
        : "Edit and resume a run (via Edit Frames) to"
          + " compare it against the original.",
    });
  }

  function activate() {
    overlayMode = "none";
    resetDiffOverlay();
    resetRunBlend();
    rebuildOverlaySelect();
    overlaySelectGroup.hidden = false;
    setDrawerOpen(false);
    updateOverlayChrome();
  }

  function deactivate() {
    overlaySelectGroup.hidden = true;
    stopCandidates();
  }

  function releaseOutput() {
    stopCandidates();
    liveTokenSpans = [];
    tokenGlowQueue = [];
    viewMode = "none";
    drawnFrame = -1;
    outputArea.classList.remove("token-layers");
    outputArea.classList.remove("live-tokens");
  }

  function clearOutput() {
    releaseOutput();
    onOutputReset();
    outputArea.textContent = "";
    onRender();
  }

  function reset() {
    invalidate();
    overlayMode = "none";
    resetDiffOverlay();
    resetRunBlend();
    updateOverlayChrome();
    releaseOutput();
  }

  function renderTargetPlaceholder(state) {
    if (
      !state
      || !Number.isInteger(state.frameIndex)
      || !Array.isArray(state.editedFrames)
    ) {
      throw new TypeError(
        "generatorCanvas target needs frame and edits"
      );
    }
    stopCandidates();
    viewMode = "target";
    drawnFrame = state.frameIndex;
    outputArea.classList.remove("token-layers");
    outputArea.classList.remove("live-tokens");
    onOutputReset();
    outputArea.textContent = "";
    var editedFrames = state.editedFrames.slice();
    if (editedFrames.length === 0) {
      editedFrames.push(state.minimumFrame);
    }
    appendTargetNotice(state.frameIndex, editedFrames);
    appendTargetPreview(state.frameIndex);
    onRender();
  }

  function appendTargetNotice(frameIndex, editedFrames) {
    var notice = document.createElement("span");
    notice.className = "preview-notice";
    var label = document.createElement("span");
    label.className = "preview-frame-label";
    label.textContent = "Frame " + frameIndex;
    notice.appendChild(label);
    notice.appendChild(
      document.createTextNode(
        " will be generated. "
        + "Output will diverge from this preview based on edits to "
        + targetFrameList(editedFrames) + "."
      )
    );
    outputArea.appendChild(notice);
  }

  function targetFrameList(frames) {
    if (frames.length === 1) {
      return "Frame " + frames[0];
    }
    if (frames.length === 2) {
      return "Frames " + frames[1] + " and " + frames[0];
    }
    var text = "Frames ";
    for (var index = frames.length - 1; index >= 0; index--) {
      text += index === 0
        ? "and " + frames[index]
        : frames[index] + ", ";
    }
    return text;
  }

  function appendTargetPreview(frameIndex) {
    var tokens = run.originalTokens(frameIndex);
    var text = run.originalText(frameIndex);
    if (!tokens && !text) {
      return;
    }
    var wrapper = document.createElement("div");
    wrapper.className = "preview-content";
    if (tokens) {
      appendTargetTokens(wrapper, tokens);
    } else {
      wrapper.appendChild(
        textSpan("char-resolved", text)
      );
    }
    outputArea.appendChild(wrapper);
  }

  function appendTargetTokens(wrapper, tokens) {
    var options = {
      revealMask: overlaysDrawsGuess(settingsState()),
    };
    for (var index = 0; index < tokens.length; index++) {
      wrapper.appendChild(
        overlaysBuildTokenSpan(
          index, tokens[index], maskChar(), options
        )
      );
    }
  }

  return {
    wire: wire,
    applyModel: applyModel,
    applySettings: applySettings,
    activate: activate,
    deactivate: deactivate,
    reset: reset,
    clearOutput: clearOutput,
    releaseOutput: releaseOutput,
    renderLiveFrame: renderLiveFrame,
    renderTextFrame: renderTextFrame,
    renderFinalText: renderFinalText,
    renderFrame: renderFrame,
    renderTargetPlaceholder: renderTargetPlaceholder,
    invalidate: invalidate,
    rebuildOverlaySelect: rebuildOverlaySelect,
    refreshControls: updateOverlayChrome,
    setOverlayMode: setOverlayMode,
    overlayMode: function () {
      return overlayMode;
    },
    maskChar: maskChar,
    diff: readDiffData,
    diffAvailable: diffAvailable,
    revisionsAvailable: revisionsAvailable,
    entropyAvailable: entropyAvailable,
    entropyDeclared: entropyDeclared,
    forgettingAvailable: forgettingAvailable,
    watermarkAvailable: watermarkAvailable,
    blend: function () {
      return runBlend;
    },
    setBlend: setBlend,
    blendActive: blendActive,
    blendFavorsOriginal: blendFavorsOriginal,
    layersActive: layersActive,
    layerIsOriginal: layerIsOriginal,
    drawnTokens: drawnTokens,
    entropyReading: entropyReading,
    entropyProfile: entropyProfile,
    tokenExtra: tokenExtra,
    revisionCount: tokenRevisionCount,
    editedPositionMarks: readEditedPositionMarks,
  };
}
