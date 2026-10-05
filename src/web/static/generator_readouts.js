// Generator entropy profile, token metrics and stopping readout.
//
// Loaded as a classic script after overlays.js, generator_run.js and
// generator_canvas.js, and before app.js. The returned controller
// owns the readout DOM, hover state and profile layout. It reads run
// history through generatorRun and rendered-layer facts through
// generatorCanvas. The page supplies narrow model, readout-setting
// and scrubber snapshots.

"use strict";

function generatorReadoutsCreate(options) {
  function requiredController(name) {
    if (!options || !options[name]) {
      throw new TypeError(
        "generatorReadoutsCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorReadoutsCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing generator readout element #" + id
      );
    }
    return element;
  }

  var run = requiredController("run");
  var canvas = requiredController("canvas");
  var readModel = requiredCallback("readModel");
  var readSettings = requiredCallback("readSettings");
  var readScrubber = requiredCallback("readScrubber");

  var outputArea = requiredElement("output-area");
  var entropyProfileRow =
    requiredElement("entropy-profile-row");
  var entropyProfileCanvas =
    requiredElement("entropy-profile");
  var entropyProfileReadout =
    requiredElement("entropy-profile-readout");
  var tokenMetricsStrip = requiredElement("token-metrics");
  var stopReadout = requiredElement("stop-readout");
  var watermarkReadout =
    requiredElement("watermark-readout");

  var ENTROPY_PROFILE_CURRENT = 1;
  var ENTROPY_PROFILE_FILLED = 0.68;
  var ENTROPY_PROFILE_UNFILLED = 0.2;

  var entropyHoverPosition = null;
  var tokenHighlightPosition = null;
  var metricsHoverPosition = null;
  var metricsHoverOriginal = false;
  var metricsCandidate = null;
  var profileLayers = null;
  var profileLayout = null;
  var wired = false;

  function modelState() {
    var state = readModel();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorReadouts readModel must return an object"
      );
    }
    return {
      capabilities: state.capabilities || {},
      parameterDefaults: state.parameterDefaults || {},
      vocabSize:
        typeof state.vocabSize === "number"
          ? state.vocabSize
          : null,
    };
  }

  function settingsState() {
    var state = readSettings();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorReadouts readSettings must return an object"
      );
    }
    if (
      !state.remaskedPositions
      || typeof state.remaskedPositions !== "object"
    ) {
      throw new TypeError(
        "generatorReadouts settings need remaskedPositions"
      );
    }
    if (!Array.isArray(state.segmentStarts)) {
      throw new TypeError(
        "generatorReadouts settings need segmentStarts"
      );
    }
    return state;
  }

  function scrubberState() {
    var state = readScrubber();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorReadouts readScrubber must return an object"
      );
    }
    if (!Number.isInteger(state.frame) || state.frame < 0) {
      throw new TypeError(
        "generatorReadouts scrubber needs a frame"
      );
    }
    return {
      active: state.active === true,
      frame: state.frame,
      selectingTarget: state.selectingTarget === true,
    };
  }

  function boot() {
    overlaysBuildTokenMetrics(tokenMetricsStrip);
    overlaysBuildStopReadout(stopReadout);
    overlaysBuildWatermarkReadout(watermarkReadout);
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    entropyProfileCanvas.addEventListener(
      "mousemove", profilePointerMoved
    );
    entropyProfileCanvas.addEventListener(
      "mouseleave", profilePointerLeft
    );
  }

  function profilePointerMoved(event) {
    var position = entropyProfilePosition(event);
    setEntropyHoverPosition(position);
    setTokenHighlight(position);
    setTokenMetricsHover(position, null);
  }

  function profilePointerLeft() {
    setEntropyHoverPosition(null);
    setTokenHighlight(null);
    clearMetrics();
  }

  // The longer run sets the shared slots, so unequal branches stay
  // position-aligned and the pointer inverse uses the same geometry.
  function entropyProfileColumns(layers) {
    return Math.max(
      layers.values.length, layers.original.length
    );
  }

  // Every touched position maps to its most recent edit frame. The
  // canvas owns that fold because it already derives render facts
  // from the page's edit log.
  function editedProfilePositions() {
    var marks = canvas.editedPositionMarks();
    var positions = [];
    for (var key in marks) {
      if (typeof marks[key] === "number") {
        positions.push(Number(key));
      }
    }
    return positions;
  }

  function drawEntropyProfile() {
    var layers = canvas.entropyProfile();
    profileLayers = layers;
    var values = layers.values;
    if (values.length === 0) {
      profileLayout = null;
      setProfileVisible(false);
      return;
    }
    setProfileVisible(true);

    var ratio = window.devicePixelRatio || 1;
    var cssWidth = entropyProfileCanvas.clientWidth || 1;
    var cssHeight = entropyProfileCanvas.clientHeight || 34;
    entropyProfileCanvas.width = Math.round(cssWidth * ratio);
    entropyProfileCanvas.height = Math.round(
      cssHeight * ratio
    );
    var context = entropyProfileCanvas.getContext("2d");
    if (!context) {
      profileLayout = null;
      return;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);

    // Autoregressive frame indices map to positions. A diffusion
    // profile has no current column, which the canvas expresses as
    // -1. Filled is separate because both comparison layers share
    // the branch's reached boundary.
    var columns = entropyProfileColumns(layers);
    var layout = {
      step: cssWidth / columns,
      barWidth: Math.max(1, cssWidth / columns - 0.5),
      cssWidth: cssWidth,
      cssHeight: cssHeight,
      columns: columns,
      values: values,
    };
    profileLayout = layout;

    var edits = editedProfilePositions();
    var editColors = editMarkerColors(edits);
    drawEntropyProfileEditTint(
      context, layout, edits, editColors
    );

    var paired = layers.original.length > 0;
    if (paired) {
      drawEntropyProfileSeries(context, layout, {
        values: layers.original,
        alpha: 1 - canvas.blend(),
        current: -1,
        filled: layers.filled,
      });
    }
    drawEntropyProfileSeries(context, layout, {
      values: values,
      alpha: paired ? canvas.blend() : 1,
      current: layers.current,
      filled: layers.filled,
    });
    drawEntropyProfileEditLines(
      context, layout, edits, editColors
    );

    // The glow and readout follow the layer which currently owns
    // the pointer, matching the stacked token canvas above.
    var readsOriginal =
      paired && canvas.blendFavorsOriginal();
    layout.values = readsOriginal
      ? layers.original
      : values;
    drawEntropyProfileGlow(context, layout);
    updateEntropyReadout(
      layout.values,
      entropyHoverPosition === null
        ? layers.current
        : entropyHoverPosition,
      readsOriginal
        ? layers.originalAsOfStep
        : layers.asOfStep
    );
  }

  function editMarkerColors(positions) {
    var marks = canvas.editedPositionMarks();
    var maxFrame = run.frameCount() - 1;
    var colors = [];
    for (var index = 0; index < positions.length; index++) {
      colors.push(
        overlaysEditColor(marks[positions[index]], maxFrame)
      );
    }
    return colors;
  }

  function drawEntropyProfileEditTint(
    context, layout, positions, colors
  ) {
    if (positions.length === 0) {
      return;
    }
    context.save();
    context.globalAlpha = OVERLAYS_EDIT_TINT_ALPHA;
    for (var index = 0; index < positions.length; index++) {
      context.fillStyle = colors[index];
      context.fillRect(
        positions[index] * layout.step,
        0,
        Math.max(2, layout.barWidth),
        layout.cssHeight
      );
    }
    context.restore();
  }

  function drawEntropyProfileEditLines(
    context, layout, positions, colors
  ) {
    if (positions.length === 0) {
      return;
    }
    context.save();
    context.globalAlpha = OVERLAYS_EDIT_LINE_ALPHA;
    context.lineWidth = 1;
    context.setLineDash([4, 4]);
    for (var index = 0; index < positions.length; index++) {
      context.strokeStyle = colors[index];
      var x = positions[index] * layout.step
        + layout.barWidth / 2;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, layout.cssHeight);
      context.stroke();
    }
    context.restore();
  }

  function drawEntropyProfileSeries(context, layout, series) {
    if (series.alpha <= 0.01) {
      return;
    }
    for (var index = 0; index < series.values.length; index++) {
      var value = series.values[index];
      var fraction = overlaysEntropyFraction(value);
      var height = Math.max(
        1, fraction * (layout.cssHeight - 2)
      );
      var emphasis =
        entropyProfileEmphasis(series, index);
      context.globalAlpha = emphasis * series.alpha;
      context.fillStyle = entropyColor(value);
      context.fillRect(
        index * layout.step,
        layout.cssHeight - height,
        layout.barWidth,
        height
      );
    }
    context.globalAlpha = 1;
  }

  function entropyProfileEmphasis(series, index) {
    if (index === series.current) {
      return ENTROPY_PROFILE_CURRENT;
    }
    if (
      typeof series.filled !== "number"
      || series.filled < 0
    ) {
      return ENTROPY_PROFILE_FILLED;
    }
    if (index <= series.filled) {
      return ENTROPY_PROFILE_FILLED;
    }
    return ENTROPY_PROFILE_UNFILLED;
  }

  function drawEntropyProfileGlow(context, layout) {
    var position = entropyHoverPosition;
    if (
      position === null
      || position < 0
      || position >= layout.values.length
    ) {
      return;
    }
    var value = layout.values[position];
    var left = position * layout.step;
    context.fillStyle = "rgba(255, 255, 255, 0.1)";
    context.fillRect(
      left,
      0,
      Math.max(2, layout.barWidth),
      layout.cssHeight
    );

    var fraction = overlaysEntropyFraction(value);
    var height = Math.max(
      2, fraction * (layout.cssHeight - 2)
    );
    var top = layout.cssHeight - height;
    context.shadowColor = entropyColor(value);
    context.shadowBlur = 8;
    context.fillStyle = entropyGlowColor(value);
    context.fillRect(left, top, layout.barWidth, height);
    context.fillRect(left, top, layout.barWidth, height);
    context.shadowBlur = 0;
    context.shadowColor = "transparent";
  }

  function updateEntropyReadout(values, index, asOfStep) {
    if (index < 0 || index >= values.length) {
      entropyProfileReadout.textContent = "";
      return;
    }
    var text =
      String(+values[index].toFixed(2)) + " nats";
    if (typeof asOfStep === "number") {
      text += ", " + overlaysEntropyAsOf(asOfStep);
    }
    entropyProfileReadout.textContent = text;
  }

  function setEntropyHoverPosition(position) {
    var visible = profileShowing();
    var next = visible ? position : null;
    if (entropyHoverPosition === next) {
      return;
    }
    entropyHoverPosition = next;
    if (visible) {
      drawEntropyProfile();
    }
  }

  function setTokenHighlight(position) {
    if (tokenHighlightPosition === position) {
      return;
    }
    clearTokenHighlight();
    tokenHighlightPosition = position;
    if (position === null) {
      return;
    }
    var spans = outputArea.querySelectorAll(
      "[data-pos=\"" + position + "\"]"
    );
    for (var index = 0; index < spans.length; index++) {
      spans[index].classList.add("token-cross-highlight");
      if (overlaysTokenIsZeroWidth(spans[index].textContent)) {
        spans[index].classList.add("token-zero-width");
      }
    }
  }

  function clearTokenHighlight() {
    tokenHighlightPosition = null;
    var highlighted = outputArea.querySelectorAll(
      ".token-cross-highlight"
    );
    for (
      var index = 0;
      index < highlighted.length;
      index++
    ) {
      highlighted[index].classList.remove(
        "token-cross-highlight"
      );
      highlighted[index].classList.remove(
        "token-zero-width"
      );
    }
  }

  function setTokenMetricsHover(position, target) {
    metricsHoverPosition = position;
    metricsHoverOriginal = position === null
      ? false
      : canvas.layerIsOriginal(target);
    refreshMetrics();
  }

  function setCandidateHover(reading) {
    metricsCandidate = reading === null
      ? null
      : {
        text: reading.t,
        probability: reading.p,
        rank: reading.rank || null,
        vocabSize:
          reading.vocab_size || metricsVocabSize(),
      };
    refreshMetrics();
  }

  function metricsVocabSize() {
    return modelState().vocabSize || null;
  }

  function refreshMetrics() {
    overlaysRenderTokenMetrics(
      tokenMetricsStrip, buildTokenMetricsReading()
    );
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  function clearMetrics() {
    metricsHoverPosition = null;
    metricsHoverOriginal = false;
    metricsCandidate = null;
    overlaysRenderTokenMetrics(tokenMetricsStrip, null);
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  function refreshLayer() {
    if (metricsHoverPosition !== null) {
      metricsHoverOriginal = canvas.layerIsOriginal(null);
    }
    refreshMetrics();
    refreshStop();
  }

  function buildTokenMetricsReading() {
    if (metricsHoverPosition === null) {
      return null;
    }
    var scrubber = scrubberState();
    if (scrubber.active && scrubber.selectingTarget) {
      return null;
    }
    var tokens = canvas.drawnTokens(metricsHoverOriginal);
    if (
      !tokens
      || metricsHoverPosition >= tokens.length
    ) {
      return null;
    }
    var index = metricsHoverPosition;
    var token = tokens[index];
    var remasked =
      settingsState().remaskedPositions[index] === true;
    var masked = !token || !!token.m || remasked;
    var entropy = canvas.entropyReading(
      index, token, metricsHoverOriginal
    );
    return {
      position: index,
      total: tokens.length,
      tokenText: token ? token.t : "",
      masked: masked,
      maskChar: canvas.maskChar(),
      confidence:
        metricsConfidence(token, masked, remasked),
      entropy: entropy.value,
      extra: overlaysEntropyNote(
        canvas.tokenExtra(
          index, token, metricsHoverOriginal
        ),
        entropy.asOfStep
      ),
      candidate: metricsCandidate,
      runLabel: metricsRunLabel(),
    };
  }

  function metricsConfidence(token, masked, remasked) {
    if (remasked || !token) {
      return 0;
    }
    if (typeof token.c === "number") {
      return token.c;
    }
    return masked ? 0 : null;
  }

  function metricsRunLabel() {
    if (!canvas.layersActive()) {
      return "";
    }
    return metricsHoverOriginal ? "Original" : "Edited";
  }

  function refreshStop() {
    overlaysRenderStopReadout(
      stopReadout, stopReadoutReading()
    );
    refreshWatermark();
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  function refreshWatermark() {
    overlaysRenderWatermarkReadout(
      watermarkReadout, watermarkReadoutReading()
    );
  }

  function watermarkReadoutReading() {
    if (
      typeof canvas.watermarkAvailable !== "function"
      || !canvas.watermarkAvailable()
    ) {
      return null;
    }
    var provenance = typeof run.provenance === "function"
      ? run.provenance()
      : null;
    var watermark = provenance && provenance.watermark;
    if (!watermark || typeof watermark.p0 !== "number") {
      return null;
    }
    var original =
      canvas.layersActive()
      && canvas.layerIsOriginal(null);
    var tokens = canvas.drawnTokens(original);
    var stats = overlaysWatermarkStats(tokens, watermark.p0);
    if (stats === null) {
      return null;
    }
    return {
      stats: stats,
      threshold: watermarkDisplayThreshold(),
    };
  }

  function watermarkDisplayThreshold() {
    var params = run.parameters() || {};
    var value = params.watermark_z_threshold;
    if (
      typeof value === "number"
      && isFinite(value)
      && value >= 0
    ) {
      return value;
    }
    var fallback = modelState().parameterDefaults;
    var defaultValue = fallback.watermark_z_threshold;
    if (
      typeof defaultValue === "number"
      && isFinite(defaultValue)
      && defaultValue >= 0
    ) {
      return defaultValue;
    }
    return 4;
  }

  function stopReadoutReading() {
    var rule = stopReadoutRule();
    if (rule === null) {
      return null;
    }
    var original =
      canvas.layersActive()
      && canvas.layerIsOriginal(null);
    var source = original
      ? stopReadoutOriginalSource()
      : stopReadoutRunSource();
    if (source.count === 0) {
      return null;
    }
    var scrubber = scrubberState();
    var index = scrubber.active
      ? Math.min(scrubber.frame, source.count - 1)
      : source.count - 1;
    return overlaysStopReadingAt(
      overlaysStopTrack(source), index, rule
    );
  }

  function stopReadoutRule() {
    var model = modelState();
    if (!model.capabilities.adaptive_stopping) {
      return null;
    }
    return overlaysStopRuleFrom(
      run.parameters(), model.parameterDefaults
    );
  }

  function stopReadoutRunSource() {
    return {
      count: run.frameCount(),
      readFrame: function (frame) {
        return run.frameTokens(frame);
      },
      canvasAt: function (frame) {
        return run.frameCanvas(frame);
      },
      segmentStarts: settingsState().segmentStarts,
    };
  }

  function stopReadoutOriginalSource() {
    return {
      count: run.originalTokenFrames(),
      readFrame: function (frame) {
        return run.originalTokens(frame);
      },
      canvasAt: function () {
        return 0;
      },
      segmentStarts: [],
    };
  }

  function entropyProfilePosition(event) {
    var layers = canvas.entropyProfile();
    profileLayers = layers;
    var columns = entropyProfileColumns(layers);
    if (columns === 0) {
      return null;
    }
    var cssWidth = entropyProfileCanvas.clientWidth || 1;
    var step = cssWidth / columns;
    if (
      profileLayout
      && profileLayout.cssWidth === cssWidth
      && profileLayout.columns === columns
    ) {
      step = profileLayout.step;
    }
    var rectangle =
      entropyProfileCanvas.getBoundingClientRect();
    var index = Math.floor(
      (event.clientX - rectangle.left) / step
    );
    if (index < 0 || index >= columns) {
      return null;
    }
    return index;
  }

  function setProfileVisible(visible) {
    entropyProfileRow.hidden =
      !visible && !canvas.entropyDeclared();
    entropyProfileRow.classList.toggle(
      "is-empty", !visible
    );
  }

  function profileShowing() {
    return (
      !entropyProfileRow.hidden
      && !entropyProfileRow.classList.contains("is-empty")
    );
  }

  function updateProfile() {
    var scrubber = scrubberState();
    if (!scrubber.active || !canvas.entropyAvailable()) {
      setProfileVisible(false);
      return;
    }
    drawEntropyProfile();
  }

  function applyModel() {
    setProfileVisible(false);
    overlaysRenderWatermarkReadout(watermarkReadout, null);
  }

  function setTokenHover(position, target) {
    setEntropyHoverPosition(position);
    setTokenMetricsHover(position, target);
  }

  function clearTokenHover() {
    setEntropyHoverPosition(null);
    clearMetrics();
  }

  function deactivate() {
    setProfileVisible(false);
    entropyHoverPosition = null;
    clearTokenHighlight();
    clearMetrics();
  }

  function reset() {
    deactivate();
    profileLayers = null;
    profileLayout = null;
    overlaysRenderStopReadout(stopReadout, null);
    overlaysRenderWatermarkReadout(watermarkReadout, null);
  }

  function outputReset() {
    tokenHighlightPosition = null;
  }

  function rendered() {
    refreshMetrics();
  }

  function layerChanged(change) {
    if (
      change
      && change.profile
      && scrubberState().active
    ) {
      updateProfile();
    }
    refreshLayer();
  }

  return {
    boot: boot,
    wire: wire,
    applyModel: applyModel,
    updateProfile: updateProfile,
    profileShowing: profileShowing,
    setTokenHover: setTokenHover,
    clearTokenHover: clearTokenHover,
    setCandidateHover: setCandidateHover,
    clearMetrics: clearMetrics,
    refreshStop: refreshStop,
    refreshWatermark: refreshWatermark,
    deactivate: deactivate,
    reset: reset,
    outputReset: outputReset,
    rendered: rendered,
    layerChanged: layerChanged,
  };
}
