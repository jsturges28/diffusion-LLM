// Generator candidate popover and typed-token controller.
//
// Loaded as a classic script after overlays.js, candidate_flicker.js,
// generator_run.js, generator_canvas.js and generator_readouts.js,
// and before app.js. The returned controller owns the candidate
// popover's DOM, paging, hover lifecycle and typed-token state. It
// reads run and render facts through the supplied controllers. The
// page remains responsible for transport, run mutation and edit
// phase transitions through the three request callbacks.

"use strict";

function generatorCandidatesCreate(options) {
  function requiredController(name) {
    if (!options || !options[name]) {
      throw new TypeError(
        "generatorCandidatesCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorCandidatesCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing generator candidate element #" + id
      );
    }
    return element;
  }

  var run = requiredController("run");
  var canvas = requiredController("canvas");
  var readouts = requiredController("readouts");
  var readState = requiredCallback("readState");
  var requestTokenize =
    requiredCallback("requestTokenize");
  var requestProbe = requiredCallback("requestProbe");
  var requestSubstitute =
    requiredCallback("requestSubstitute");

  var outputArea = requiredElement("output-area");
  var popover = requiredElement("token-alts-popover");

  var TYPED_PREVIEW_DEBOUNCE_MS = 120;
  var TYPED_INPUT_MAX_CHARS = 200;

  var popoverPosition = null;
  var popoverPage = null;

  var typedPosition = null;
  var typedDraft = "";
  var typedActive = false;
  var typedSeeded = false;
  var typedPreview = null;
  var typedToken = null;
  var typedMeasure = null;
  var typedPreviewRequest = 0;
  var typedProbeRequest = 0;
  var typedPreviewTimer = null;
  var wired = false;

  function pageState() {
    var state = readState();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorCandidates readState must return an object"
      );
    }
    if (!Number.isInteger(state.frame) || state.frame < 0) {
      throw new TypeError(
        "generatorCandidates state needs a frame"
      );
    }
    if (!Array.isArray(state.remaskEdits)) {
      throw new TypeError(
        "generatorCandidates state needs remaskEdits"
      );
    }
    return {
      frame: state.frame,
      scrubberActive: state.scrubberActive === true,
      editing: state.editing === true,
      substituting: state.substituting === true,
      remaskEdits: state.remaskEdits,
      tokenizer: state.tokenizer || null,
      vocabSize:
        typeof state.vocabSize === "number"
          ? state.vocabSize
          : null,
    };
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    outputArea.addEventListener(
      "mouseover", outputPointerEntered
    );
    outputArea.addEventListener(
      "mouseleave", outputPointerLeft
    );
    popover.addEventListener(
      "mouseleave", popoverPointerLeft
    );
    popover.addEventListener("click", popoverClicked);
    document.addEventListener("keydown", documentKeyDown);
    document.addEventListener(
      "pointerdown", documentPointerDown
    );
    window.addEventListener("scroll", windowScrolled, true);
    window.addEventListener("resize", windowResized);
  }

  function outputPointerEntered(event) {
    var target = event.target;
    var position = hoveredTokenPosition(target);
    if (position === null) {
      return;
    }
    readouts.setTokenHover(position, target);
    var state = pageState();
    if (!state.scrubberActive) {
      return;
    }
    if (state.editing && !state.substituting) {
      return;
    }
    if (position === popoverPosition || pinned()) {
      return;
    }
    showPopover(position, target);
  }

  function hoveredTokenPosition(target) {
    if (
      !target.classList
      || !target.classList.contains("token-span")
    ) {
      return null;
    }
    var raw = target.getAttribute("data-pos");
    if (raw === null) {
      return null;
    }
    return parseInt(raw, 10);
  }

  function outputPointerLeft() {
    if (popover.matches(":hover") || pinned()) {
      return;
    }
    readouts.clearTokenHover();
    hidePopover();
  }

  function popoverPointerLeft() {
    if (pinned()) {
      return;
    }
    readouts.clearTokenHover();
    hidePopover();
  }

  function popoverClicked(event) {
    var state = pageState();
    if (!state.substituting || popoverPosition === null) {
      return;
    }
    if (popoverPage === "original") {
      return;
    }
    var row = event.target.closest(".alt-row");
    if (!row) {
      return;
    }
    var raw = row.getAttribute("data-alt-id");
    if (raw === null) {
      return;
    }
    if (row.classList.contains("alt-row-outside")) {
      return;
    }
    var typed = row.getAttribute("data-typed") === "1"
      ? typedDraft
      : null;
    var accepted = requestSubstitute({
      position: popoverPosition,
      tokenId: parseInt(raw, 10),
      typedText: typed,
    });
    if (accepted === true) {
      hidePopover();
    }
  }

  function documentKeyDown(event) {
    if (event.key !== "Escape" || !pinned()) {
      return;
    }
    event.preventDefault();
    cancelTypedEntry();
  }

  function documentPointerDown(event) {
    if (!pinned()) {
      return;
    }
    if (popover.contains(event.target)) {
      return;
    }
    cancelTypedEntry();
  }

  function windowScrolled() {
    if (popoverPosition !== null && !pinned()) {
      hidePopover();
    }
  }

  function windowResized() {
    if (!pinned()) {
      hidePopover();
    }
    if (pageState().scrubberActive) {
      readouts.updateProfile();
    }
  }

  // Whether any position captured alternatives for What If.
  function alternativesAvailable() {
    return run.hasAlternatives(false);
  }

  // True while the popover is a surface the user is working in.
  function pinned() {
    return typedActive || typedToken !== null;
  }

  function hidePopover() {
    popover.hidden = true;
    popover.replaceChildren();
    popoverPosition = null;
    popoverPage = null;
    readouts.setCandidateHover(null);
    clearTypedEntry();
  }

  function clearTypedEntry() {
    typedPosition = null;
    typedDraft = "";
    typedActive = false;
    typedSeeded = false;
    typedPreview = null;
    typedToken = null;
    typedMeasure = null;
    if (typedPreviewTimer !== null) {
      clearTimeout(typedPreviewTimer);
      typedPreviewTimer = null;
    }
  }

  // Show a position's candidates, anchored to its token span.
  function showPopover(position, span) {
    var state = pageState();
    if (run.frameIsAppend()) {
      popoverPage = appendPageable(position, state)
        ? defaultPage()
        : null;
    } else {
      popoverPage = diffusionPage(state);
    }
    renderPopover(position, span, state);
  }

  function setPage(page) {
    if (popoverPosition === null) {
      return;
    }
    popoverPage = page;
    renderPopover(
      popoverPosition, null, pageState()
    );
  }

  function renderPopover(position, span, state) {
    if (!run.frameIsAppend()) {
      renderDiffusionPopover(position, span, state);
      return;
    }
    renderAppendPopover(position, span, state);
  }

  function renderAppendPopover(position, span, state) {
    var original = popoverPage === "original";
    var alternatives = run.positionAlternatives(
      position, original
    );
    if (!alternatives || alternatives.length === 0) {
      hidePopover();
      return;
    }
    var tokens = original
      ? run.originalTokensLast()
      : run.frameTokens(state.frame);
    var chosen = tokens && tokens[position]
      ? tokens[position].id
      : null;

    clearCandidateReading();
    popover.replaceChildren();
    popover.appendChild(
      overlaysBuildAltHeading(
        position, popoverPage, setPage
      )
    );
    popover.appendChild(
      buildRows(alternatives, chosen)
    );
    appendTypedInvitation(position, original, state);
    appendTokenizer(state.tokenizer);
    popover.classList.toggle(
      "alt-pickable",
      state.substituting && !original
    );
    placePopover(span);
    popoverPosition = position;
  }

  function appendTypedInvitation(position, original, state) {
    if (!state.substituting || original) {
      return;
    }
    popover.appendChild(buildTypedEntry(position, state));
    var hint = document.createElement("div");
    hint.className = "alt-hint";
    hint.textContent = "Click a candidate to substitute";
    popover.appendChild(hint);
  }

  function renderDiffusionPopover(position, span, state) {
    var page = popoverPage;
    var reading = diffusionReading(
      page, position, state.frame
    );
    if (reading === null) {
      hidePopover();
      return;
    }
    var other = page === null
      ? null
      : diffusionReading(
        otherPage(page), position, state.frame
      );
    clearCandidateReading();
    popover.replaceChildren();
    popover.appendChild(
      overlaysBuildStepHeading(
        position,
        reading.frame,
        reading.shown,
        page,
        other === null ? null : setPage
      )
    );
    popover.appendChild(
      buildRows(reading.set.c, reading.set.h)
    );
    appendTokenizer(state.tokenizer);
    popover.classList.remove("alt-pickable");
    placePopover(span);
    popoverPosition = position;
  }

  function clearCandidateReading() {
    // Removed rows never fire mouseleave, so clear their reading.
    readouts.setCandidateHover(null);
  }

  function appendTokenizer(tokenizer) {
    var footer = overlaysBuildAltTokenizer(tokenizer);
    if (footer) {
      popover.appendChild(footer);
    }
  }

  function buildRows(alternatives, chosenId) {
    var fragment = document.createDocumentFragment();
    for (
      var index = 0;
      index < alternatives.length;
      index++
    ) {
      fragment.appendChild(
        overlaysBuildAltRow(
          alternatives[index],
          chosenId,
          readouts.setCandidateHover,
          index
        )
      );
    }
    return fragment;
  }

  function appendPageable(position, state) {
    var divergence = editDivergencePosition(
      state.remaskEdits
    );
    if (divergence === null || position < divergence) {
      return false;
    }
    var original = run.positionAlternatives(
      position, true
    );
    var edited = run.positionAlternatives(
      position, false
    );
    return !!(
      original
      && original.length > 0
      && edited
      && edited.length > 0
    );
  }

  function editDivergencePosition(edits) {
    var earliest = null;
    for (var edit = 0; edit < edits.length; edit++) {
      var positions = edits[edit].token_positions || [];
      for (
        var index = 0;
        index < positions.length;
        index++
      ) {
        var position = positions[index];
        if (earliest === null || position < earliest) {
          earliest = position;
        }
      }
    }
    return earliest;
  }

  function defaultPage() {
    return canvas.blendFavorsOriginal()
      ? "original"
      : "edited";
  }

  function diffusionPage(state) {
    var divergence = editDivergenceFrame(
      state.remaskEdits
    );
    if (!canvas.blendActive() || divergence === null) {
      return null;
    }
    return state.frame >= divergence
      ? defaultPage()
      : null;
  }

  function editDivergenceFrame(edits) {
    var earliest = null;
    for (var index = 0; index < edits.length; index++) {
      var frame = edits[index].frame_index;
      if (earliest === null || frame < earliest) {
        earliest = frame;
      }
    }
    return earliest;
  }

  function otherPage(page) {
    return page === "original" ? "edited" : "original";
  }

  function diffusionReading(page, position, frame) {
    if (page === "original") {
      var shown = Math.min(
        frame, run.originalTokenFrames() - 1
      );
      return diffusionReadingOf(
        run.candidateSet(
          shown, position, true, singleCanvas
        ),
        shown
      );
    }
    return diffusionReadingOf(
      run.candidateSet(
        frame, position, false, runFrameCanvas
      ),
      frame
    );
  }

  function diffusionReadingOf(found, shown) {
    if (found === null) {
      return null;
    }
    return {
      frame: found.frame,
      set: found.set,
      shown: shown,
    };
  }

  function runFrameCanvas(frame) {
    return run.frameCanvas(frame);
  }

  function singleCanvas() {
    return 0;
  }

  function placePopover(span) {
    popover.hidden = false;
    if (!span) {
      return;
    }
    var rectangle = span.getBoundingClientRect();
    var box = popover.getBoundingClientRect();
    popover.style.left =
      overlaysPopoverLeft(rectangle, box) + "px";
    popover.style.top =
      overlaysPopoverTop(
        rectangle,
        box,
        outputArea.getBoundingClientRect().top
      ) + "px";
  }

  function buildTypedEntry(position, state) {
    if (typedPosition !== position) {
      clearTypedEntry();
      typedPosition = position;
    }
    if (typedToken !== null) {
      return buildTypedSolidified();
    }
    return buildTypedInput(position, state);
  }

  function typedSeedText(position, state) {
    var tokens = run.frameTokens(state.frame);
    var token = tokens && tokens[position]
      ? tokens[position]
      : null;
    var text = token && typeof token.t === "string"
      ? token.t
      : "";
    return text.charAt(0) === " " ? " " : "";
  }

  function buildTypedInput(position, state) {
    var wrap = document.createElement("div");
    wrap.className = "typed-entry";
    if (!typedSeeded) {
      typedSeeded = true;
      typedDraft = typedSeedText(position, state);
    }

    var field = document.createElement("input");
    field.type = "text";
    field.className = "typed-input";
    field.maxLength = TYPED_INPUT_MAX_CHARS;
    field.value = typedDraft;
    field.spellcheck = false;
    field.autocomplete = "off";
    field.addEventListener("input", typedInputChanged);
    field.addEventListener("focus", typedInputFocused);

    var box = document.createElement("div");
    box.className = "typed-field";
    box.appendChild(field);
    box.appendChild(buildTypedPlaceholder());
    wrap.appendChild(box);
    wrap.appendChild(buildTypedActions());

    var preview = document.createElement("div");
    preview.className = "typed-preview";
    wrap.appendChild(preview);
    renderTypedPieces(preview);
    refocusTypedField(field);
    return wrap;
  }

  function refocusTypedField(field) {
    if (!typedActive) {
      return;
    }
    setTimeout(function () {
      if (!typedActive || !document.contains(field)) {
        return;
      }
      field.focus();
      var end = field.value.length;
      field.setSelectionRange(end, end);
    }, 0);
  }

  function buildTypedPlaceholder() {
    var hint = document.createElement("span");
    hint.className = "typed-placeholder";
    syncTypedPlaceholder(hint);
    return hint;
  }

  function typedPlaceholderText() {
    if (typedDraft === "") {
      return "Enter your own";
    }
    if (typedDraft === " ") {
      return "\u00B7Enter your own";
    }
    return "";
  }

  function syncTypedPlaceholder(hint) {
    var text = typedPlaceholderText();
    hint.textContent = text;
    hint.hidden = text === "";
  }

  function buildTypedActions() {
    var actions = document.createElement("div");
    actions.className = "typed-actions";
    if (typedActive) {
      actions.classList.add("is-open");
    }

    var confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "typed-confirm";
    confirm.textContent = "\u2713";
    syncTypedConfirm(confirm);
    confirm.addEventListener("click", confirmTypedEntry);
    actions.appendChild(confirm);

    var cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "typed-cancel";
    cancel.textContent = "\u2715";
    cancel.title = "Discard";
    cancel.addEventListener("click", cancelTypedEntry);
    actions.appendChild(cancel);
    return actions;
  }

  function typedEntryResolvesToOne() {
    return (
      typedPreview !== null
      && typedPreview.count === 1
      && typedDraft !== ""
    );
  }

  function syncTypedConfirm(confirm) {
    confirm.disabled = !typedEntryResolvesToOne();
    confirm.title = confirm.disabled
      ? "Type text that resolves to exactly one token"
      : "Use this token";
  }

  function refreshTypedControls() {
    var actions = popover.querySelector(".typed-actions");
    if (actions) {
      actions.classList.toggle("is-open", typedActive);
    }
    var confirm = popover.querySelector(".typed-confirm");
    if (confirm) {
      syncTypedConfirm(confirm);
    }
    var hint = popover.querySelector(".typed-placeholder");
    if (hint) {
      syncTypedPlaceholder(hint);
    }
    var preview = popover.querySelector(".typed-preview");
    if (preview) {
      renderTypedPieces(preview);
    }
  }

  function renderTypedPieces(host) {
    host.replaceChildren();
    if (typedPreview === null || typedDraft === "") {
      return;
    }
    for (
      var index = 0;
      index < typedPreview.pieces.length;
      index++
    ) {
      host.appendChild(
        buildTypedPiece(typedPreview.pieces[index], index)
      );
    }
    var note = document.createElement("span");
    note.className = "typed-count";
    if (typedPreview.count === 1) {
      note.textContent = "1 token";
    } else if (typedPreview.count === 0) {
      note.textContent = "no tokens";
      note.classList.add("typed-count-over");
    } else {
      note.textContent = typedPreview.count + " tokens";
      note.classList.add("typed-count-over");
    }
    host.appendChild(note);
  }

  function buildTypedPiece(piece, index) {
    var element = document.createElement("span");
    element.className = "typed-piece";
    if (index % 2 === 1) {
      element.classList.add("typed-piece-alt");
    }
    var text = document.createElement("span");
    text.className = "typed-piece-text";
    text.textContent = overlaysAltDisplay(piece.t);
    element.appendChild(text);
    var id = document.createElement("span");
    id.className = "typed-piece-id";
    id.textContent = String(piece.id);
    element.appendChild(id);
    return element;
  }

  function buildTypedSolidified() {
    var row = document.createElement("div");
    row.className = "alt-row typed-solid";
    row.setAttribute("data-alt-id", String(typedToken.id));
    row.setAttribute("data-typed", "1");

    var text = document.createElement("span");
    text.className = "alt-text";
    text.textContent = overlaysAltDisplay(typedToken.t);
    row.appendChild(text);

    var tag = document.createElement("span");
    tag.className = "typed-tag";
    tag.textContent = "yours";
    row.appendChild(tag);

    var probability = document.createElement("span");
    probability.className = "typed-prob";
    renderTypedMeasure(probability);
    row.appendChild(probability);

    var retry = document.createElement("button");
    retry.type = "button";
    retry.className = "typed-retry";
    retry.textContent = "\u21BA";
    retry.title = "Type a different token";
    retry.addEventListener("click", retryTypedEntry);
    row.appendChild(retry);

    row.addEventListener("mouseenter", function () {
      readouts.setCandidateHover(
        typedCandidateReading()
      );
    });
    row.addEventListener("mouseleave", function () {
      readouts.setCandidateHover(null);
    });
    return row;
  }

  function typedCandidateReading() {
    if (typedToken === null) {
      return null;
    }
    var reading = { t: typedToken.t, p: null };
    if (typedMeasure !== null) {
      reading.p = typedMeasure.probability;
      reading.rank = typedMeasure.rank;
      reading.vocab_size = typedMeasure.vocabSize;
    }
    return reading;
  }

  function renderTypedMeasure(slot) {
    if (typedMeasure === null) {
      slot.textContent = "\u2026";
      slot.classList.add("is-pending");
      slot.title = "Measuring what the model gave this token";
      return;
    }
    slot.classList.remove("is-pending");
    slot.textContent = typedProbabilityText(
      typedMeasure.probability
    );
    slot.title = typedMeasureTitle(typedMeasure);
  }

  function typedProbabilityText(probability) {
    var value = Number(probability);
    if (!isFinite(value) || value <= 0) {
      return "0.0%";
    }
    if (value < 0.001) {
      return "<0.1%";
    }
    return (value * 100).toFixed(1) + "%";
  }

  function typedMeasureTitle(measure) {
    var text = "Probability "
      + Number(measure.probability).toPrecision(3);
    if (measure.rank) {
      text += ", rank "
        + Number(measure.rank).toLocaleString();
    }
    if (measure.rank && measure.vocabSize) {
      text += " of "
        + Number(measure.vocabSize).toLocaleString();
    }
    return text;
  }

  function typedInputFocused() {
    if (typedActive) {
      return;
    }
    typedActive = true;
    refreshTypedControls();
  }

  function typedInputChanged(event) {
    typedDraft = event.target.value;
    typedPreview = null;
    refreshTypedControls();
    if (typedPreviewTimer !== null) {
      clearTimeout(typedPreviewTimer);
    }
    typedPreviewTimer = setTimeout(
      sendTypedPreview,
      TYPED_PREVIEW_DEBOUNCE_MS
    );
  }

  function sendTypedPreview() {
    typedPreviewTimer = null;
    if (typedDraft === "") {
      typedPreview = null;
      refreshTypedControls();
      return;
    }
    typedPreviewRequest += 1;
    requestTokenize({
      text: typedDraft,
      requestId: typedPreviewRequest,
    });
  }

  function handleTokenizeResult(message) {
    if (message.request_id !== typedPreviewRequest) {
      return;
    }
    if (message.text !== typedDraft) {
      return;
    }
    typedPreview = {
      pieces: message.pieces || [],
      count: message.count || 0,
    };
    refreshTypedControls();
  }

  function confirmTypedEntry() {
    if (!typedEntryResolvesToOne()) {
      return;
    }
    typedToken = typedPreview.pieces[0];
    typedMeasure = null;
    redrawTypedEntry();
    sendTypedProbe();
  }

  function sendTypedProbe() {
    if (typedToken === null || typedPosition === null) {
      return;
    }
    typedProbeRequest += 1;
    if (fillMeasureFromRecord()) {
      return;
    }
    requestProbe({
      position: typedPosition,
      tokenId: typedToken.id,
      requestId: typedProbeRequest,
    });
  }

  function fillMeasureFromRecord() {
    var alternatives = run.positionAlternatives(
      typedPosition, false
    );
    if (!alternatives) {
      return false;
    }
    for (
      var index = 0;
      index < alternatives.length;
      index++
    ) {
      if (alternatives[index].id !== typedToken.id) {
        continue;
      }
      typedMeasure = {
        probability: alternatives[index].p,
        rank: overlaysAltRank(
          alternatives[index], index
        ),
        vocabSize: pageState().vocabSize,
      };
      refreshTypedMeasure();
      return true;
    }
    return false;
  }

  function handleProbeResult(message) {
    if (message.request_id !== typedProbeRequest) {
      return;
    }
    if (typedToken === null) {
      return;
    }
    if (message.token_id !== typedToken.id) {
      return;
    }
    typedMeasure = {
      probability: message.probability,
      rank: message.rank,
      vocabSize: message.vocab_size,
    };
    refreshTypedMeasure();
  }

  function refreshTypedMeasure() {
    var slot = popover.querySelector(".typed-prob");
    if (slot) {
      renderTypedMeasure(slot);
    }
  }

  function retryTypedEntry(event) {
    event.stopPropagation();
    typedToken = null;
    typedMeasure = null;
    typedPreview = null;
    typedDraft = "";
    typedSeeded = false;
    typedActive = true;
    redrawTypedEntry();
  }

  function cancelTypedEntry() {
    var position = popoverPosition;
    clearTypedEntry();
    if (position === null) {
      return;
    }
    if (popover.matches(":hover")) {
      renderPopover(position, null, pageState());
      return;
    }
    readouts.clearTokenHover();
    hidePopover();
  }

  function redrawTypedEntry() {
    if (popoverPosition === null) {
      return;
    }
    renderPopover(
      popoverPosition, null, pageState()
    );
  }

  function startFlicker(layers, mask) {
    flickerStart(layers, mask);
  }

  function stopFlicker() {
    flickerStop();
  }

  return {
    wire: wire,
    alternativesAvailable: alternativesAvailable,
    showPopover: showPopover,
    hidePopover: hidePopover,
    outputReset: hidePopover,
    handleTokenizeResult: handleTokenizeResult,
    handleProbeResult: handleProbeResult,
    startFlicker: startFlicker,
    stopFlicker: stopFlicker,
  };
}
