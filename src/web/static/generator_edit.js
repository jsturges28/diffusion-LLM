// Generator scrubber and counterfactual-edit controller.
//
// Loaded as a classic script after generator_run.js,
// generator_canvas.js, generator_readouts.js and
// generator_candidates.js, and before app.js. The returned
// controller owns every mutable scrubber and edit-session fact plus
// the DOM that presents them. It reads and mutates the active run
// only through generatorRun. Transport, generation lifecycle,
// status presentation, saving and fresh-run policy stay with the
// page through the callbacks below.

"use strict";

function generatorEditCreate(options) {
  function requiredController(name) {
    if (!options || !options[name]) {
      throw new TypeError(
        "generatorEditCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorEditCreate needs options." + name
      );
    }
    return options[name];
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing generator edit element #" + id
      );
    }
    return element;
  }

  var run = requiredController("run");
  var canvas = requiredController("canvas");
  var readouts = requiredController("readouts");
  var candidates = requiredController("candidates");

  var readCapabilities =
    requiredCallback("readCapabilities");
  var readGenerating = requiredCallback("readGenerating");
  var readDiffusionEffect =
    requiredCallback("readDiffusionEffect");
  var revealText = requiredCallback("revealText");
  var dissolveText = requiredCallback("dissolveText");
  var renderFrameReadout =
    requiredCallback("renderFrameReadout");
  var setStatus = requiredCallback("setStatus");
  var setGenerating = requiredCallback("setGenerating");
  var setSaveAvailable =
    requiredCallback("setSaveAvailable");
  var resetStatus = requiredCallback("resetStatus");
  var startRunStatus = requiredCallback("startRunStatus");
  var primaryStateChanged =
    requiredCallback("primaryStateChanged");
  var requestSave = requiredCallback("requestSave");
  var requestRewind = requiredCallback("requestRewind");
  var requestResume = requiredCallback("requestResume");
  var requestSubstitute =
    requiredCallback("requestSubstitute");

  var outputArea = requiredElement("output-area");
  var scrubberSection =
    requiredElement("scrubber-section");
  var scrubberControls =
    requiredElement("scrubber-controls");
  var scrubberSlider =
    requiredElement("scrubber-slider");
  var scrubberLabel =
    requiredElement("scrubber-label");
  var btnScrubStart =
    requiredElement("btn-scrub-start");
  var btnScrubPrev =
    requiredElement("btn-scrub-prev");
  var btnScrubNext =
    requiredElement("btn-scrub-next");
  var btnScrubEnd =
    requiredElement("btn-scrub-end");
  var btnEditFrames =
    requiredElement("btn-edit-frames");
  var btnWhatIf = requiredElement("btn-what-if");

  var guidedEditControls =
    requiredElement("guided-edit-controls");
  var guidedEditStatus =
    requiredElement("guided-edit-status");
  var btnSelectFrame =
    requiredElement("btn-select-frame");
  var btnBackFrame =
    requiredElement("btn-back-frame");
  var btnLockIn = requiredElement("btn-lock-in");
  var btnClearGuided =
    requiredElement("btn-clear-guided");
  var btnEditAnother =
    requiredElement("btn-edit-another");
  var btnRunToHere =
    requiredElement("btn-run-to-here");
  var btnResumeEnd =
    requiredElement("btn-resume-end");
  var btnConfirmEdit =
    requiredElement("btn-confirm-edit");
  var btnRetryEdit =
    requiredElement("btn-retry-edit");
  var btnContinueEdit =
    requiredElement("btn-continue-edit");
  var btnExitEdit =
    requiredElement("btn-exit-edit");

  var remaskRandomizeRow =
    requiredElement("remask-randomize-row");
  var remaskRandomSlider =
    requiredElement("remask-random-slider");
  var remaskRandomCount =
    requiredElement("remask-random-count");
  var remaskRandomTotal =
    requiredElement("remask-random-total");
  var btnRemaskShuffle =
    requiredElement("btn-remask-shuffle");
  var shuffleLabel =
    requiredElement("btn-remask-shuffle-label");

  var retryEditTitle = btnRetryEdit.title;
  var continueEditTitle = btnContinueEdit.title;
  var stoppedBeforeFrameMessage =
    "Stopped before the edit produced a frame."
    + " The run is unchanged.";

  var scrubberActive = false;
  var currentFrame = 0;
  var remaskedPositions = {};
  var perFrameRemasked = {};
  var remaskEdits = [];
  var runPhase = runPhasesCreate();
  var preEditCheckpoint = null;
  var minimumFrame = 0;
  var pendingResume = null;
  var isResuming = false;
  var randomizeInitialFrame = null;
  var editsReadSource = null;
  var editsReadSnapshot = Object.freeze([]);
  var positionsReadSource = null;
  var positionsReadSnapshot = Object.freeze({});
  var segmentStartsReadSnapshot = Object.freeze([]);
  var wired = false;

  function capabilities() {
    var state = readCapabilities();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorEdit capabilities must be an object"
      );
    }
    return state;
  }

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    wireScrubber();
    wireGuidedActions();
    wireRandomSelection();
    outputArea.addEventListener("click", outputClicked);
    document.addEventListener("keydown", documentKeyDown);
  }

  function wireScrubber() {
    scrubberSlider.addEventListener("input", function () {
      navigate(parseInt(scrubberSlider.value, 10));
    });
    btnScrubStart.addEventListener("click", function () {
      navigate(0);
    });
    btnScrubPrev.addEventListener("click", function () {
      navigate(currentFrame - 1);
    });
    btnScrubNext.addEventListener("click", function () {
      navigate(currentFrame + 1);
    });
    btnScrubEnd.addEventListener("click", function () {
      navigate(navigationMaximum());
    });
    btnEditFrames.addEventListener("click", enterFrames);
    btnWhatIf.addEventListener("click", enterWhatIf);
  }

  function wireGuidedActions() {
    btnSelectFrame.addEventListener("click", selectFrame);
    btnBackFrame.addEventListener("click", backToFrame);
    btnLockIn.addEventListener("click", lockClicked);
    btnClearGuided.addEventListener("click", clearSelection);
    btnEditAnother.addEventListener(
      "click", chooseAnotherFrame
    );
    btnRunToHere.addEventListener("click", function () {
      runPhase.targetFrame = currentFrame;
      resumeGuided("another");
    });
    btnResumeEnd.addEventListener("click", function () {
      resumeGuided("end");
    });
    btnConfirmEdit.addEventListener("click", confirm);
    btnRetryEdit.addEventListener("click", retry);
    btnContinueEdit.addEventListener("click", continueBranch);
    btnExitEdit.addEventListener("click", exit);
  }

  function wireRandomSelection() {
    remaskRandomSlider.addEventListener("input", function () {
      remaskRandomCount.value = remaskRandomSlider.value;
    });
    remaskRandomCount.addEventListener("input", function () {
      var total = resolvedPositions(currentFrame).length;
      var floor = total > 0 ? 1 : 0;
      var count = clampInt(
        parseInt(remaskRandomCount.value, 10) || floor,
        floor,
        total
      );
      remaskRandomCount.value = String(count);
      remaskRandomSlider.value = String(count);
    });
    btnRemaskShuffle.addEventListener("click", function () {
      shuffleRemasks();
      playShuffleEffect();
    });
  }

  function outputClicked(event) {
    if (!scrubberActive || runPhase.mode !== RUN_PHASE_EDIT) {
      return;
    }
    var target = event.target;
    if (
      !target.classList.contains("token-clickable")
      && !target.classList.contains("token-remasked")
    ) {
      return;
    }
    var raw = target.getAttribute("data-pos");
    if (raw === null) {
      return;
    }
    togglePosition(parseInt(raw, 10));
  }

  function documentKeyDown(event) {
    if (
      !scrubberActive
      || readGenerating()
      || run.saving()
    ) {
      return;
    }
    if (!keyboardPhaseAllowsNavigation()) {
      return;
    }
    var active = document.activeElement;
    var tag = active && active.tagName
      ? active.tagName
      : "";
    if (
      tag === "INPUT"
      || tag === "TEXTAREA"
      || tag === "SELECT"
    ) {
      return;
    }
    navigateFromKey(event);
  }

  function keyboardPhaseAllowsNavigation() {
    return (
      runPhase.mode !== RUN_PHASE_EDIT
      && runPhase.mode !== RUN_PHASE_CHOICE
      && runPhase.mode !== RUN_PHASE_GENERATING
      && runPhase.mode !== RUN_PHASE_SUBSTITUTE
    );
  }

  function navigateFromKey(event) {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigate(currentFrame - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      navigate(currentFrame + 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      navigate(0);
    } else if (event.key === "End") {
      event.preventDefault();
      navigate(navigationMaximum());
    }
  }

  function activate() {
    if (run.frameCount() < 2) {
      return false;
    }
    scrubberActive = true;
    currentFrame = run.frameCount() - 1;
    scrubberSlider.min = "0";
    scrubberSlider.max = String(currentFrame);
    scrubberSlider.value = String(currentFrame);
    scrubberSlider.disabled = false;
    updateScrubberLabel();
    setScrubberVisible(true);
    applyEntryGates();
    refreshLocks();
    canvas.activate();
    guidedEditControls.hidden = true;
    clearSelections();
    unlockNavigation();
    navigate(currentFrame);
    readouts.updateProfile();
    return true;
  }

  function applyEntryGates() {
    var declared = capabilities();
    btnEditFrames.hidden = !(
      declared.supports_resume
      && !run.frameIsMultiCanvas()
    );
    btnWhatIf.hidden = !(
      declared.supports_substitution
      && candidates.alternativesAvailable()
    );
  }

  function deactivate() {
    scrubberActive = false;
    setScrubberVisible(false);
    guidedEditControls.hidden = true;
    canvas.deactivate();
    readouts.deactivate();
    candidates.hidePopover();
    clearSelections();
  }

  function generationChanged(active) {
    if (active) {
      deactivate();
    }
  }

  function reset() {
    runPhasesReset(runPhase);
    scrubberActive = false;
    currentFrame = 0;
    remaskedPositions = {};
    perFrameRemasked = {};
    remaskEdits = [];
    preEditCheckpoint = null;
    minimumFrame = 0;
    pendingResume = null;
    isResuming = false;
    randomizeInitialFrame = null;
    guidedEditControls.hidden = true;
    scrubberSlider.disabled = false;
    scrubberSlider.min = "0";
    setScrubberVisible(false);
    unlockNavigation();
    canvas.deactivate();
    readouts.deactivate();
    candidates.hidePopover();
  }

  function setScrubberVisible(visible) {
    scrubberSection.classList.toggle("is-idle", !visible);
  }

  function navigationMinimum() {
    if (
      runPhase.mode === RUN_PHASE_SELECT
      || runPhase.mode === RUN_PHASE_SELECT_TARGET
    ) {
      return minimumFrame;
    }
    return 0;
  }

  function navigationMaximum() {
    if (
      runPhase.mode === RUN_PHASE_SELECT_TARGET
      && run.originalCaptured()
    ) {
      return run.originalTotalFrames() - 1;
    }
    return run.frameCount() - 1;
  }

  function updateScrubberLabel() {
    scrubberLabel.textContent =
      "Frame " + currentFrame
      + " / " + navigationMaximum();
  }

  function navigate(index) {
    if (!Number.isInteger(index)) {
      throw new TypeError(
        "generatorEdit.navigate needs an integer frame"
      );
    }
    saveFrameSelection(currentFrame);
    var bounded = Math.max(
      navigationMinimum(),
      Math.min(index, navigationMaximum())
    );
    currentFrame = bounded;
    scrubberSlider.value = String(bounded);
    updateScrubberLabel();
    renderFrameReadout({
      frame: bounded,
      canvasIndex: run.frameCanvas(bounded),
      totalSteps: run.totalSteps(),
    });
    restoreFrameSelection(bounded);
    renderNavigationFrame(bounded);
    readouts.refreshStop();
    if (scrubberActive) {
      readouts.updateProfile();
    }
    updateGuidedUi();
  }

  function renderNavigationFrame(index) {
    if (runPhase.mode === RUN_PHASE_SELECT_TARGET) {
      renderTargetFrame(index);
    } else if (index < run.frameCount()) {
      canvas.renderFrame(index);
    } else {
      renderTargetFrame(index);
    }
  }

  function renderTargetFrame(frameIndex) {
    var editedFrames = [];
    for (
      var index = 0;
      index < runPhase.lockedEdits.length;
      index++
    ) {
      editedFrames.push(
        runPhase.lockedEdits[index].frame_index
      );
    }
    canvas.renderTargetPlaceholder({
      frameIndex: frameIndex,
      editedFrames: editedFrames,
      minimumFrame: minimumFrame,
    });
  }

  function clearSelections() {
    remaskedPositions = {};
    perFrameRemasked = {};
    updateGuidedUi();
  }

  function saveFrameSelection(frameIndex) {
    if (Object.keys(remaskedPositions).length > 0) {
      perFrameRemasked[frameIndex] =
        copyPositionMap(remaskedPositions);
    } else {
      delete perFrameRemasked[frameIndex];
    }
  }

  function restoreFrameSelection(frameIndex) {
    if (perFrameRemasked[frameIndex]) {
      remaskedPositions = copyPositionMap(
        perFrameRemasked[frameIndex]
      );
    } else {
      remaskedPositions = {};
    }
  }

  function togglePosition(position) {
    if (!Number.isInteger(position) || position < 0) {
      throw new RangeError(
        "generatorEdit.togglePosition needs a position"
      );
    }
    var next = copyPositionMap(remaskedPositions);
    if (next[position]) {
      delete next[position];
    } else {
      next[position] = true;
    }
    remaskedPositions = next;
    saveFrameSelection(currentFrame);
    canvas.renderFrame(currentFrame);
    updateGuidedUi();
  }

  function clearSelection() {
    remaskedPositions = {};
    delete perFrameRemasked[currentFrame];
    canvas.renderFrame(currentFrame);
    updateGuidedUi();
  }

  function clampInt(value, low, high) {
    if (value < low) {
      return low;
    }
    if (value > high) {
      return high;
    }
    return value;
  }

  function resolvedPositions(frameIndex) {
    var tokens = run.frameTokens(frameIndex);
    var output = [];
    if (!tokens) {
      return output;
    }
    for (var index = 0; index < tokens.length; index++) {
      if (tokens[index] && !tokens[index].m) {
        output.push(index);
      }
    }
    return output;
  }

  function updateRandomizeRow() {
    var total = resolvedPositions(currentFrame).length;
    var floor = total > 0 ? 1 : 0;
    if (randomizeInitialFrame !== currentFrame) {
      randomizeInitialFrame = currentFrame;
      var selected = Object.keys(remaskedPositions).length;
      remaskRandomSlider.value = String(
        clampInt(selected, floor, total)
      );
    }
    var target = clampInt(
      parseInt(remaskRandomSlider.value, 10) || floor,
      floor,
      total
    );
    remaskRandomTotal.textContent = String(total);
    setRandomBounds(floor, total, target);
    var disabled = total === 0;
    remaskRandomSlider.disabled = disabled;
    remaskRandomCount.disabled = disabled;
    btnRemaskShuffle.disabled = disabled;
  }

  function setRandomBounds(floor, total, target) {
    remaskRandomSlider.min = String(floor);
    remaskRandomSlider.max = String(total);
    remaskRandomSlider.value = String(target);
    remaskRandomCount.min = String(floor);
    remaskRandomCount.max = String(total);
    remaskRandomCount.value = String(target);
  }

  function shuffleRemasks() {
    var available = resolvedPositions(currentFrame);
    var total = available.length;
    if (total === 0) {
      return;
    }
    var count = clampInt(
      parseInt(remaskRandomSlider.value, 10) || 0,
      0,
      total
    );
    for (var index = 0; index < count; index++) {
      var chosen = index + Math.floor(
        Math.random() * (total - index)
      );
      var swap = available[index];
      available[index] = available[chosen];
      available[chosen] = swap;
    }
    remaskedPositions = {};
    for (var selected = 0; selected < count; selected++) {
      remaskedPositions[available[selected]] = true;
    }
    saveFrameSelection(currentFrame);
    canvas.renderFrame(currentFrame);
    updateGuidedUi();
  }

  function playShuffleEffect() {
    if (!readDiffusionEffect()) {
      return;
    }
    btnRemaskShuffle.classList.add("is-diffusing");
    revealText(shuffleLabel, "Shuffle", function () {
      btnRemaskShuffle.classList.remove("is-diffusing");
    });
  }

  function resetGuidedMode() {
    runPhasesReset(runPhase);
    candidates.hidePopover();
    preEditCheckpoint = null;
    pendingResume = null;
    randomizeInitialFrame = null;
    guidedEditControls.hidden = true;
    scrubberSlider.disabled = false;
    scrubberSlider.min = "0";
    unlockNavigation();
  }

  function capturePreEditCheckpoint() {
    preEditCheckpoint = {
      run: run.captureCheckpoint(),
      remaskEditsLength: remaskEdits.length,
    };
    rewindWorkerRun();
  }

  function rewindWorkerRun() {
    var token = run.runToken();
    if (!token || editBlockReason()) {
      return false;
    }
    return requestRewind({ runToken: token }) === true;
  }

  function restorePreEditCheckpoint() {
    if (preEditCheckpoint === null) {
      return false;
    }
    run.restoreCheckpoint(preEditCheckpoint.run);
    var length = Math.min(
      remaskEdits.length,
      preEditCheckpoint.remaskEditsLength
    );
    remaskEdits = remaskEdits.slice(0, length);
    preEditCheckpoint = null;
    return true;
  }

  function unlockNavigation() {
    btnScrubStart.disabled = false;
    btnScrubPrev.disabled = false;
    btnScrubNext.disabled = false;
    btnScrubEnd.disabled = false;
  }

  function lockNavigation() {
    btnScrubStart.disabled = true;
    btnScrubPrev.disabled = true;
    btnScrubNext.disabled = true;
    btnScrubEnd.disabled = true;
    scrubberSlider.disabled = true;
  }

  function setSavingControls(saving) {
    var disabled = saving === true;
    scrubberSlider.disabled = disabled;
    btnScrubStart.disabled = disabled;
    btnScrubPrev.disabled = disabled;
    btnScrubNext.disabled = disabled;
    btnScrubEnd.disabled = disabled;
    btnSelectFrame.disabled = disabled;
    btnEditFrames.disabled = disabled;
    btnBackFrame.disabled = disabled;
    btnLockIn.disabled = disabled;
    btnClearGuided.disabled = disabled;
    btnEditAnother.disabled = disabled;
    btnRunToHere.disabled = disabled;
    btnResumeEnd.disabled = disabled;
    btnConfirmEdit.disabled = disabled;
    btnRetryEdit.disabled = disabled;
    btnContinueEdit.disabled = disabled;
    btnExitEdit.disabled = disabled;
    scrubberControls.classList.toggle(
      "is-saving", disabled
    );
    if (disabled) {
      scrubberControls.title = "Saving in progress\u2026";
    } else {
      scrubberControls.removeAttribute("title");
    }
    primaryStateChanged();
    if (!disabled) {
      updateGuidedUi();
    }
  }

  function enterWhatIf() {
    if (btnWhatIf.classList.contains("is-locked")) {
      return false;
    }
    beginWhatIfSession();
    return true;
  }

  function beginWhatIfSession() {
    capturePreEditCheckpoint();
    runPhasesEnter(runPhase, RUN_PHASE_SUBSTITUTE);
    runPhase.substituting = true;
    minimumFrame = 0;
    runPhase.lockedEdits = [];
    runPhase.guidedAction = null;
    clearSelections();
    scrubberSlider.min = "0";
    scrubberSlider.max =
      String(run.frameCount() - 1);
    btnEditFrames.hidden = true;
    btnWhatIf.hidden = true;
    guidedEditControls.hidden = false;
    canvas.deactivate();
    navigate(run.frameCount() - 1);
    updateGuidedUi();
  }

  function substitute(intent) {
    validateSubstitutionIntent(intent);
    if (
      !runPhase.substituting
      || runPhase.mode !== RUN_PHASE_SUBSTITUTE
    ) {
      return false;
    }
    if (
      intent.position < 0
      || intent.position >= run.frameCount()
    ) {
      return false;
    }
    if (editRequestRefused()) {
      return false;
    }
    runPhase.substituting = false;
    remaskEdits = remaskEdits.concat([{
      frame_index: intent.position,
      token_positions: [intent.position],
    }]);
    perFrameRemasked = {};
    remaskedPositions = {};
    run.truncate(intent.position);
    run.truncateAlternatives(intent.position);
    isResuming = true;
    runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
    updateGuidedUi();
    beginBranch(intent.position, null);
    requestSubstitute({
      position: intent.position,
      tokenId: intent.tokenId,
      typedText: intent.typedText,
      runToken: run.runToken(),
    });
    return true;
  }

  function validateSubstitutionIntent(intent) {
    if (
      !intent
      || !Number.isInteger(intent.position)
      || !Number.isInteger(intent.tokenId)
    ) {
      throw new TypeError(
        "generatorEdit substitution needs position and token"
      );
    }
    if (
      intent.typedText !== null
      && typeof intent.typedText !== "string"
    ) {
      throw new TypeError(
        "generatorEdit typed text must be a string or null"
      );
    }
  }

  function enterFrames() {
    if (btnEditFrames.classList.contains("is-locked")) {
      return false;
    }
    beginFrameSession();
    return true;
  }

  function beginFrameSession() {
    capturePreEditCheckpoint();
    runPhasesEnter(runPhase, RUN_PHASE_SELECT);
    var startFrame = run.frameCount() > 1 ? 1 : 0;
    minimumFrame = startFrame;
    runPhase.lockedEdits = [];
    runPhase.guidedAction = null;
    clearSelections();
    scrubberSlider.min = String(startFrame);
    scrubberSlider.max =
      String(run.frameCount() - 1);
    btnEditFrames.hidden = true;
    guidedEditControls.hidden = false;
    canvas.deactivate();
    navigate(startFrame);
    updateGuidedUi();
  }

  function exit() {
    restorePreEditCheckpoint();
    resetGuidedMode();
    activate();
  }

  function renoiseNote() {
    if (capabilities().remask_renoises) {
      return " Remasked tokens are renoised, so nearby"
        + " tokens may also change on resume.";
    }
    return "";
  }

  function updateGuidedUi() {
    canvas.refreshControls();
    hidePhaseControls();
    if (runPhase.mode === RUN_PHASE_IDLE) {
      guidedEditControls.hidden = true;
      return;
    }
    guidedEditControls.hidden = false;
    setSaveAvailable(false);
    renderPhaseUi();
    if (run.saving()) {
      lockNavigation();
      btnSelectFrame.disabled = true;
    }
  }

  function hidePhaseControls() {
    btnSelectFrame.hidden = true;
    btnSelectFrame.disabled = false;
    btnBackFrame.hidden = true;
    btnLockIn.hidden = true;
    btnClearGuided.hidden = true;
    btnEditAnother.hidden = true;
    btnRunToHere.hidden = true;
    btnResumeEnd.hidden = true;
    btnConfirmEdit.hidden = true;
    btnRetryEdit.hidden = true;
    btnContinueEdit.hidden = true;
    remaskRandomizeRow.hidden = true;
  }

  function renderPhaseUi() {
    switch (runPhase.mode) {
      case RUN_PHASE_SELECT:
        renderSelectPhase();
        break;
      case RUN_PHASE_EDIT:
        renderEditPhase();
        break;
      case RUN_PHASE_CHOICE:
        renderChoicePhase();
        break;
      case RUN_PHASE_SELECT_TARGET:
        renderTargetPhase();
        break;
      case RUN_PHASE_SUBSTITUTE:
        renderSubstitutePhase();
        break;
      case RUN_PHASE_GENERATING:
        guidedEditStatus.textContent =
          "Generating\u2026";
        lockNavigation();
        break;
      case RUN_PHASE_REVIEW:
        renderReviewPhase();
        break;
      default:
        throw new Error(
          "Unknown generator edit phase " + runPhase.mode
        );
    }
  }

  function renderSelectPhase() {
    guidedEditStatus.textContent =
      "Navigate to a frame, then select it for editing.";
    btnSelectFrame.hidden = false;
    scrubberSlider.disabled = false;
    scrubberSlider.min = String(minimumFrame);
    unlockNavigation();
  }

  function renderEditPhase() {
    var count = Object.keys(remaskedPositions).length;
    guidedEditStatus.textContent =
      "Frame " + currentFrame
      + ": click tokens to remask ("
      + count + " selected)." + renoiseNote();
    btnBackFrame.hidden = false;
    btnLockIn.hidden = false;
    btnLockIn.disabled = count === 0;
    btnClearGuided.hidden = false;
    btnClearGuided.disabled = count === 0;
    remaskRandomizeRow.hidden = false;
    updateRandomizeRow();
    lockNavigation();
  }

  function renderChoicePhase() {
    var count = Object.keys(remaskedPositions).length;
    var plural = count !== 1 ? "s" : "";
    guidedEditStatus.textContent =
      count + " token" + plural
      + " locked on Frame " + currentFrame + ".";
    btnEditAnother.hidden = false;
    btnResumeEnd.hidden = false;
    lockNavigation();
  }

  function renderTargetPhase() {
    guidedEditStatus.textContent =
      "Navigate to the target frame, then run to it.";
    btnRunToHere.hidden = false;
    scrubberSlider.disabled = false;
    scrubberSlider.min = String(minimumFrame);
    scrubberSlider.max = String(navigationMaximum());
    unlockNavigation();
  }

  function renderSubstitutePhase() {
    guidedEditStatus.textContent =
      "Hover a token to see what the model nearly chose,"
      + " then click a candidate to regenerate from it.";
    btnClearGuided.hidden = true;
    lockNavigation();
  }

  function renderReviewPhase() {
    scrubberSlider.disabled = false;
    scrubberSlider.min = "0";
    scrubberSlider.max =
      String(run.frameCount() - 1);
    unlockNavigation();
    btnConfirmEdit.hidden = false;
    btnRetryEdit.hidden = false;
    btnContinueEdit.hidden = !reviewCanContinue();
    if (currentFrame === run.frameCount() - 1) {
      guidedEditStatus.textContent =
        reviewEndText(currentFrame);
      return;
    }
    guidedEditStatus.textContent =
      "Reviewing frame " + currentFrame + " of the "
      + (run.interrupted()
        ? "stopped edit"
        : "edited run")
      + ". " + reviewChoices();
  }

  function selectFrame() {
    runPhasesEnter(runPhase, RUN_PHASE_EDIT);
    canvas.renderFrame(currentFrame);
    updateGuidedUi();
  }

  function backToFrame() {
    remaskedPositions = {};
    delete perFrameRemasked[currentFrame];
    runPhasesEnter(runPhase, RUN_PHASE_SELECT);
    navigate(currentFrame);
  }

  function lockClicked() {
    if (!readDiffusionEffect()) {
      lockSelection();
      return;
    }
    var label = btnLockIn.textContent;
    dissolveText(btnLockIn, function () {
      lockSelection();
      btnLockIn.textContent = label;
    });
  }

  function lockSelection() {
    var positions = Object.keys(
      remaskedPositions
    ).map(Number);
    if (positions.length === 0) {
      return false;
    }
    runPhase.lockedEdits.push({
      frame_index: currentFrame,
      token_positions: positions.slice(),
    });
    runPhasesEnter(runPhase, RUN_PHASE_CHOICE);
    updateGuidedUi();
    return true;
  }

  function chooseAnotherFrame() {
    if (runPhase.lockedEdits.length === 0) {
      return;
    }
    var last = runPhase.lockedEdits[
      runPhase.lockedEdits.length - 1
    ];
    minimumFrame = last.frame_index + 1;
    runPhasesEnter(runPhase, RUN_PHASE_SELECT_TARGET);
    scrubberSlider.min = String(minimumFrame);
    scrubberSlider.max = String(navigationMaximum());
    scrubberSlider.disabled = false;
    unlockNavigation();
    navigate(minimumFrame);
    updateGuidedUi();
  }

  function resumeGuided(action) {
    if (runPhase.lockedEdits.length === 0) {
      return false;
    }
    if (editRequestRefused()) {
      return false;
    }
    runPhase.guidedAction = action;
    var last = runPhase.lockedEdits[
      runPhase.lockedEdits.length - 1
    ];
    var positions = last.token_positions;
    var frameIndex = last.frame_index;
    pendingResume = captureResumeCut(frameIndex);
    remaskEdits = remaskEdits.concat([{
      frame_index: frameIndex,
      token_positions: positions.slice(),
    }]);
    perFrameRemasked = {};
    remaskedPositions = {};
    run.truncate(frameIndex);
    isResuming = true;
    runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
    updateGuidedUi();
    var target = resumeTarget(action);
    beginBranch(frameIndex, target);
    requestResume({
      frameIndex: frameIndex,
      remaskPositions: positions.slice(),
      targetFrame: target,
      continueRun: false,
      runToken: run.runToken(),
    });
    return true;
  }

  function resumeTarget(action) {
    if (
      action === "another"
      && runPhase.targetFrame !== null
    ) {
      return runPhase.targetFrame;
    }
    return null;
  }

  function beginBranch(fromFrame, targetFrame) {
    setSaveAvailable(false);
    resetStatus();
    setGenerating(true);
    startRunStatus(editRunLabel(fromFrame, targetFrame));
  }

  function editRunLabel(fromFrame, toFrame) {
    var target = toFrame === null
      ? "end"
      : String(toFrame);
    return "Running edit from frame " + fromFrame
      + " to " + target;
  }

  function captureResumeCut(cutAt) {
    var mode = runPhase.mode;
    var valid = (
      mode === RUN_PHASE_CHOICE
      || mode === RUN_PHASE_SELECT_TARGET
      || mode === RUN_PHASE_REVIEW
    );
    if (!valid) {
      throw new Error(
        "a resume is sent from choice, target or review"
      );
    }
    return {
      cutAt: cutAt,
      run: run.captureCheckpoint(),
      remaskEditsLength: remaskEdits.length,
      mode: mode,
      frame: currentFrame,
      minimumFrame: minimumFrame,
      remasked: copyPositionMap(remaskedPositions),
      perFrame: copyPerFrame(perFrameRemasked),
    };
  }

  function finishStream(data) {
    var resumed = pendingResume;
    pendingResume = null;
    isResuming = false;
    if (!resumeStoppedBeforeFrame(resumed, data)) {
      return false;
    }
    landBeforeResume(resumed);
    return true;
  }

  function resumeStoppedBeforeFrame(resumed, data) {
    if (resumed === null || data.cancelled !== true) {
      return false;
    }
    return run.frameCount() === resumed.cutAt;
  }

  function landBeforeResume(saved) {
    run.restoreCheckpoint(saved.run);
    remaskEdits = remaskEdits.slice(
      0, saved.remaskEditsLength
    );
    runPhase.guidedAction = null;
    runPhase.targetFrame = null;
    landBeforeResumePhase(saved.mode);
    minimumFrame = saved.minimumFrame;
    remaskedPositions = saved.remasked;
    perFrameRemasked = saved.perFrame;
    scrubberActive = true;
    setScrubberVisible(true);
    navigate(saved.frame);
    setStatus(stoppedBeforeFrameMessage);
  }

  function landBeforeResumePhase(mode) {
    if (mode === RUN_PHASE_CHOICE) {
      runPhasesEnter(runPhase, RUN_PHASE_CHOICE);
    } else if (mode === RUN_PHASE_SELECT_TARGET) {
      runPhasesEnter(runPhase, RUN_PHASE_SELECT_TARGET);
    } else {
      runPhasesEnter(runPhase, RUN_PHASE_REVIEW);
    }
  }

  function completeStream() {
    if (runPhase.mode === RUN_PHASE_GENERATING) {
      handleGuidedDone();
    } else {
      activate();
    }
  }

  function handleGuidedDone() {
    if (runPhase.guidedAction !== "another") {
      enterReview();
      return;
    }
    var target = Math.min(
      runPhase.targetFrame,
      run.frameCount() - 1
    );
    scrubberActive = true;
    setScrubberVisible(true);
    guidedEditControls.hidden = false;
    btnEditFrames.hidden = true;
    canvas.deactivate();
    scrubberSlider.min = String(target);
    scrubberSlider.max =
      String(run.frameCount() - 1);
    scrubberSlider.value = String(target);
    currentFrame = target;
    runPhase.guidedAction = null;
    runPhase.targetFrame = null;
    runPhasesEnter(runPhase, RUN_PHASE_EDIT);
    remaskedPositions = {};
    perFrameRemasked = {};
    updateScrubberLabel();
    canvas.renderFrame(target);
    updateGuidedUi();
  }

  function enterReview() {
    runPhase.guidedAction = null;
    runPhase.targetFrame = null;
    remaskedPositions = {};
    perFrameRemasked = {};
    runPhasesEnter(runPhase, RUN_PHASE_REVIEW);
    scrubberActive = true;
    setScrubberVisible(true);
    guidedEditControls.hidden = false;
    btnEditFrames.hidden = true;
    canvas.deactivate();
    currentFrame = run.frameCount() - 1;
    scrubberSlider.min = "0";
    scrubberSlider.max = String(currentFrame);
    scrubberSlider.value = String(currentFrame);
    scrubberSlider.disabled = false;
    unlockNavigation();
    updateScrubberLabel();
    canvas.renderFrame(currentFrame);
    updateGuidedUi();
  }

  function confirm() {
    requestSave();
    resetGuidedMode();
    activate();
  }

  function retry() {
    if (editRequestRefused()) {
      return false;
    }
    var substitution = supportsSubstitution();
    restorePreEditCheckpoint();
    resetGuidedMode();
    if (substitution) {
      beginWhatIfSession();
    } else {
      beginFrameSession();
    }
    return true;
  }

  function supportsSubstitution() {
    return capabilities().supports_substitution === true;
  }

  function reviewCanContinue() {
    return (
      run.interrupted()
      && capabilities().supports_resume === true
    );
  }

  function reviewEndText(frame) {
    if (run.interrupted()) {
      return "Stopped at frame " + frame + ". "
        + reviewChoices();
    }
    return "Edit complete. " + reviewChoices();
  }

  function reviewChoices() {
    if (reviewCanContinue()) {
      return "Continue, confirm to save it as it is, or retry"
        + " from the start.";
    }
    if (run.interrupted()) {
      return "Confirm to save it as it is, or retry"
        + " from the start.";
    }
    return "Confirm to save, or retry from the start.";
  }

  function continueBranch() {
    if (!reviewCanContinue() || editRequestRefused()) {
      return false;
    }
    var from = run.frameCount() - 1;
    pendingResume = captureResumeCut(from);
    run.truncate(from);
    isResuming = true;
    runPhase.guidedAction = "end";
    runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
    updateGuidedUi();
    beginBranch(from, null);
    requestResume({
      frameIndex: from,
      remaskPositions: [],
      targetFrame: null,
      continueRun: true,
      runToken: run.runToken(),
    });
    return true;
  }

  function interruptStream() {
    isResuming = false;
    pendingResume = null;
  }

  function unwindRunError() {
    isResuming = false;
    if (!runPhasesEditing(runPhase)) {
      return false;
    }
    restorePreEditCheckpoint();
    resetGuidedMode();
    return true;
  }

  function adoptResidentWorker(worker) {
    var wasBlocked = editBlockReason() !== "";
    run.adoptResidentWorker(worker);
    var blocked = editBlockReason();
    if (
      blocked
      && !wasBlocked
      && runPhasesEditing(runPhase)
      && !runPhasesKeepsWork(runPhase)
    ) {
      exit();
    }
    refreshLocks();
    return blocked && !wasBlocked ? blocked : "";
  }

  function editBlockReason() {
    return runPhasesEditBlock(run.editIdentity());
  }

  function editRequestRefused() {
    var blocked = editBlockReason();
    if (!blocked) {
      return false;
    }
    setStatus(blocked);
    return true;
  }

  function refreshLocks() {
    var blocked = editBlockReason();
    if (blocked) {
      setButtonLocked(btnEditFrames, blocked);
      setButtonLocked(btnWhatIf, blocked);
      setButtonLocked(btnRetryEdit, blocked);
      setButtonLocked(btnContinueEdit, blocked);
      return;
    }
    setButtonUnlocked(btnRetryEdit, retryEditTitle);
    setButtonUnlocked(btnContinueEdit, continueEditTitle);
    var savedLock = run.editedSaved()
      || (run.saving() && remaskEdits.length > 0);
    refreshEntryLocks(savedLock);
  }

  function refreshEntryLocks(locked) {
    if (locked) {
      setButtonLocked(
        btnEditFrames,
        "This run already has a saved edit."
        + " Generate again to edit a new run."
      );
      setButtonLocked(
        btnWhatIf,
        "This run already has a saved edit."
        + " Generate again to try another branch."
      );
      return;
    }
    setButtonUnlocked(
      btnEditFrames,
      "Remask tokens at any frame, then resume the run"
      + " from there"
    );
    setButtonUnlocked(
      btnWhatIf,
      "Replace a token with one the model nearly"
      + " chose, then regenerate"
    );
  }

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

  function readArtifacts() {
    return { remaskEdits: copyEdits(remaskEdits) };
  }

  function restoreArtifacts(state) {
    if (!state || !Array.isArray(state.remaskEdits)) {
      throw new TypeError(
        "generatorEdit artifacts need remaskEdits"
      );
    }
    remaskEdits = copyEdits(state.remaskEdits);
  }

  function canvasState() {
    return {
      remaskEdits: immutableEdits(),
      remaskedPositions: immutablePositions(),
      mode: runPhase.mode,
      substituting: runPhase.substituting,
      generating: readGenerating() === true,
    };
  }

  function candidatesState() {
    return {
      frame: currentFrame,
      scrubberActive: scrubberActive,
      editing: runPhasesEditing(runPhase),
      substituting: runPhase.substituting,
      remaskEdits: immutableEdits(),
    };
  }

  function readoutsSettings() {
    return {
      remaskedPositions: immutablePositions(),
      segmentStarts: immutableSegmentStarts(),
    };
  }

  function scrubberState() {
    return {
      active: scrubberActive,
      frame: currentFrame,
      selectingTarget:
        runPhase.mode === RUN_PHASE_SELECT_TARGET,
    };
  }

  function phaseState() {
    return {
      mode: runPhase.mode,
      substituting: runPhase.substituting,
      guidedAction: runPhase.guidedAction,
      targetFrame: runPhase.targetFrame,
      lockedEdits: copyEdits(runPhase.lockedEdits),
    };
  }

  function shouldPersistRun() {
    return runPhase.mode === RUN_PHASE_IDLE;
  }

  function copyEdits(edits) {
    var output = [];
    for (var index = 0; index < edits.length; index++) {
      var copied = Object.assign({}, edits[index]);
      copied.token_positions =
        (edits[index].token_positions || []).slice();
      output.push(copied);
    }
    return output;
  }

  function immutableEdits() {
    if (editsReadSource === remaskEdits) {
      return editsReadSnapshot;
    }
    var copied = copyEdits(remaskEdits);
    for (var index = 0; index < copied.length; index++) {
      Object.freeze(copied[index].token_positions);
      Object.freeze(copied[index]);
    }
    editsReadSource = remaskEdits;
    editsReadSnapshot = Object.freeze(copied);
    segmentStartsReadSnapshot = Object.freeze(
      copied.map(function (edit) {
        return edit.frame_index;
      })
    );
    return editsReadSnapshot;
  }

  function immutablePositions() {
    if (positionsReadSource === remaskedPositions) {
      return positionsReadSnapshot;
    }
    positionsReadSource = remaskedPositions;
    positionsReadSnapshot = Object.freeze(
      copyPositionMap(remaskedPositions)
    );
    return positionsReadSnapshot;
  }

  function immutableSegmentStarts() {
    immutableEdits();
    return segmentStartsReadSnapshot;
  }

  function copyPositionMap(source) {
    return Object.assign({}, source);
  }

  function copyPerFrame(source) {
    var output = {};
    var keys = Object.keys(source);
    for (var index = 0; index < keys.length; index++) {
      output[keys[index]] = copyPositionMap(
        source[keys[index]]
      );
    }
    return output;
  }

  return Object.freeze({
    wire: wire,
    activate: activate,
    deactivate: deactivate,
    generationChanged: generationChanged,
    reset: reset,
    navigate: navigate,
    enterFrames: enterFrames,
    enterWhatIf: enterWhatIf,
    substitute: substitute,
    selectFrame: selectFrame,
    backToFrame: backToFrame,
    togglePosition: togglePosition,
    lockSelection: lockSelection,
    resumeToEnd: function () {
      return resumeGuided("end");
    },
    runToCurrentFrame: function () {
      runPhase.targetFrame = currentFrame;
      return resumeGuided("another");
    },
    chooseAnotherFrame: chooseAnotherFrame,
    confirm: confirm,
    retry: retry,
    continueBranch: continueBranch,
    exit: exit,
    finishStream: finishStream,
    completeStream: completeStream,
    interruptStream: interruptStream,
    unwindRunError: unwindRunError,
    adoptResidentWorker: adoptResidentWorker,
    requestAllowed: function () {
      return !editRequestRefused();
    },
    refreshLocks: refreshLocks,
    setSavingControls: setSavingControls,
    readArtifacts: readArtifacts,
    restoreArtifacts: restoreArtifacts,
    canvasState: canvasState,
    candidatesState: candidatesState,
    readoutsSettings: readoutsSettings,
    scrubberState: scrubberState,
    phaseState: phaseState,
    currentFrame: function () {
      return currentFrame;
    },
    active: function () {
      return scrubberActive;
    },
    resuming: function () {
      return isResuming;
    },
    editing: function () {
      return runPhasesEditing(runPhase);
    },
    keepsWork: function () {
      return runPhasesKeepsWork(runPhase);
    },
    shouldPersistRun: shouldPersistRun,
    blockReason: editBlockReason,
    runIsMultiCanvas: function () {
      return run.frameIsMultiCanvas();
    },
  });
}
