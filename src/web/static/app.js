// LLM Visualizer: client-side logic.

"use strict";

// ---- DOM refs ----

var btnGenerate =
  document.getElementById("btn-generate");
var btnGenerateLabel =
  document.getElementById("btn-generate-label");
var btnSave =
  document.getElementById("btn-save");
var outputArea =
  document.getElementById("output-area");
var generatorComposer = generatorComposerCreate({
  onSubmit: submitComposer,
  onDraftChanged: composerDraftChanged,
  reportStatus: setPromptImportStatus,
  readThinking: composerThinking,
  readOutputBudget: composerOutputBudget,
  isCountReady: composerCountReady,
  sendCountPrompt: sendComposerCount,
});
var thinkingPanel =
  document.getElementById("thinking-panel");
var thinkingContent =
  document.getElementById("thinking-content");

// Persistent UI preferences, applied live on the generator. The schema,
// defaults, and parsing live in overlays.js (SETTINGS_DEFAULTS /
// parseSettings), shared with the Settings page which edits them.
var appSettings = parseSettings(null);
var generatorModelPanel = generatorModelPanelCreate({
  onSwitchRequested: switchModel,
  onValidationChanged: modelPanelValidationChanged,
  onParametersChanged: modelPanelParametersChanged,
  readReducedMotion: prefersReducedMotion,
  readGpuTicker: function () {
    return appSettings.gpuTicker;
  },
});
var generatorChrome = generatorChromeCreate({
  onTpsToggle: toggleTpsMode,
  readReducedMotion: prefersReducedMotion,
  readDiffusionEffect: diffusionEffectActive,
  readDiffusionTextMode: function () {
    return appSettings.diffusionTextMode;
  },
  revealText: denoiseReveal,
  cancelReveal: cancelDenoise,
});
var generatorRun = generatorRunCreate({
  readModel: generatorRunReadModel,
  readComposer: generatorRunReadComposer,
  restoreComposer: generatorRunRestoreComposer,
  readChrome: generatorRunReadChrome,
  restoreChrome: generatorRunRestoreChrome,
  readEditArtifacts: generatorRunReadEditArtifacts,
  restoreEditArtifacts: generatorRunRestoreEditArtifacts,
  invalidateRender: invalidateGeneratorCanvas,
  onSessionRestored: generatorRunSessionRestored,
  requestSave: function (url, init) {
    return fetch(url, init);
  },
  onSaveStart: generatorRunSaveStart,
  onSaveSuccess: generatorRunSaveSuccess,
  onSaveFailure: generatorRunSaveFailure,
  onSaveRefused: generatorRunSaveRefused,
  storage: sessionStorage,
  sessionKey: PERSIST_LAST_RUN_KEY,
});
var generatorCanvas = generatorCanvasCreate({
  run: generatorRun,
  readModel: generatorCanvasReadModel,
  readSettings: generatorCanvasReadSettings,
  readEdit: generatorCanvasReadEdit,
  readReducedMotion: prefersReducedMotion,
  writeHighlight: generatorCanvasWriteHighlight,
  startCandidates: flickerStart,
  stopCandidates: flickerStop,
  onOutputReset: generatorCanvasOutputReset,
  onRender: generatorCanvasRendered,
  onOverlayChanged: generatorCanvasOverlayChanged,
  onLayerChanged: generatorCanvasLayerChanged,
});
var generatorReadouts = generatorReadoutsCreate({
  run: generatorRun,
  canvas: generatorCanvas,
  readModel: generatorReadoutsReadModel,
  readSettings: generatorReadoutsReadSettings,
  readScrubber: generatorReadoutsReadScrubber,
});
var generatorSocket = generatorSocketCreate({
  onOpen: generatorSocketOpened,
  onClose: generatorSocketClosed,
  onMessage: handleMessage,
  onMalformed: generatorSocketMalformed,
  onFatal: generatorSocketFatal,
});

// Scrubber DOM refs.
var scrubberSection =
  document.getElementById("scrubber-section");
var scrubberControls =
  document.getElementById("scrubber-controls");
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
var btnWhatIf =
  document.getElementById("btn-what-if");
var altsPopover =
  document.getElementById("token-alts-popover");

// Guided edit mode DOM refs.
var guidedEditControls =
  document.getElementById("guided-edit-controls");
var guidedEditStatus =
  document.getElementById("guided-edit-status");
var btnSelectFrame =
  document.getElementById("btn-select-frame");
var btnBackFrame =
  document.getElementById("btn-back-frame");
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
var btnConfirmEdit =
  document.getElementById("btn-confirm-edit");
var btnRetryEdit =
  document.getElementById("btn-retry-edit");
// The markup's own tooltip, put back when a lock on Retry lifts.
var RETRY_EDIT_TITLE = btnRetryEdit.title;
var btnContinueEdit =
  document.getElementById("btn-continue-edit");
var CONTINUE_EDIT_TITLE = btnContinueEdit.title;
var btnExitEdit =
  document.getElementById("btn-exit-edit");
var remaskRandomizeRow =
  document.getElementById("remask-randomize-row");
var remaskRandomSlider =
  document.getElementById("remask-random-slider");
var remaskRandomCount =
  document.getElementById("remask-random-count");
var remaskRandomTotal =
  document.getElementById("remask-random-total");
var btnRemaskShuffle =
  document.getElementById("btn-remask-shuffle");
var shuffleLabel =
  document.getElementById("btn-remask-shuffle-label");

// ---- State ----

var isGenerating = false;
var saveCheckTimer = null;
var modelReady = false;

// generator_run.js owns every mutable fact and store about the run
// on screen. generator_socket.js owns transport; this file keeps
// view state and edit phases.
// Position whose candidate popover is open, or null when closed.
// The page names the run it reads, "original" or "edited", and is
// null where there is only one set to show, as on an unedited run.
var altsPopoverPos = null;
var altsPopoverPage = null;

// ---- Typed token entry state (What If) ----
//
// Held outside the DOM on purpose. The popover rebuilds all of its
// children on every hover and page flip, so a draft stored only in
// the input element would be destroyed by a pointer twitch. Keeping
// it here lets buildTypedRow rehydrate the field after any rebuild,
// which is what makes the popover safe to type into at all.
//
// Position the draft belongs to. A draft is meaningless at any
// other position, so this is also how a stale one is detected.
var typedEntryPos = null;
var typedEntryDraft = "";
// Whether the user has engaged with the field at all. Set on focus
// and never cleared by blur, because blur is not an exit: clicking
// confirm blurs the input, and so does clicking anywhere in the
// popover. The deliberate exits clear it instead.
//
// Note this is not "the draft is non-empty". The field arrives
// pre-seeded with a leading space, so a draft-based test would pin
// the popover the instant the pointer crossed a mid-sentence token.
var typedEntryActive = false;
// Whether the leading space has been offered for this position yet.
// One offer only, so a backspace that removes it stays removed
// through the next rebuild instead of being helpfully undone.
var typedEntrySeeded = false;
// Last preview from the worker: {pieces: [{id, t}], count} for the
// text in typedEntryDraft, or null while none has arrived.
var typedEntryPreview = null;
// The confirmed single token, {id, t}, once the user solidifies it.
// Clicking it runs the substitution, exactly like a candidate row.
var typedEntryToken = null;
// What the model gave that token at this position, measured on
// confirm: {probability, rank, vocabSize}, or null while the probe is
// still out. Separate from typedEntryToken because the token is
// clickable immediately and the figure arrives a forward pass later;
// waiting for it would make confirm feel like it had failed.
var typedEntryMeasure = null;
// Monotonic, so a slow reply for an older draft can be discarded.
// Debouncing alone does not guarantee replies arrive in order.
var typedEntryRequest = 0;
// Separate counter for probes. A retry after a confirm has to be able
// to invalidate the outstanding probe without disturbing the preview
// sequence, which is still counting keystrokes.
var typedProbeRequest = 0;
// True while "What If" substitution is armed: the popover's
// candidates become clickable instead of read-only.

// Scrubber and remasking state.
var scrubberActive = false;
var currentScrubFrame = 0;
var remaskedPositions = {};
var perFrameRemasked = {};
var remaskEdits = [];

// Guided multi-frame edit mode state.
// null | "select" | "edit" | "choice"
//      | "select_target" | "generating" | "review"
// Which editing phase the run is in, plus the values describing
// an edit in progress. Held as one thing so a move between
// phases can be checked against the ones that are reachable
// from where it started; see run_phases.js, which owns the
// table. Never reassigned.
var runPhase = runPhasesCreate();
// Snapshot of the complete run taken when Edit Frames is entered.
// Partial resumes ("Run to Here") truncate the live run mid-way, so
// exiting restores this to avoid stranding the user on an
// incomplete run.
var preEditSnapshot = null;
var scrubberMinFrame = 0;
// What a resume in flight cut, and where it was sent from, kept
// until it ends. See captureResumeCut.
var pendingResume = null;
var RESUME_STOPPED_BEFORE_FRAME =
  "Stopped before the edit produced a frame. The run is unchanged.";

// Whether the edit phase is currently receiving a resumed stream.
// The run controller owns the corresponding frame and elapsed
// offsets; this boolean stays with the page's edit-phase state.
var isResuming = false;

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

// The boot path raises the same overlay without going through
// switchModel, so until now nothing polled for progress there: the
// first load of a session, reliably the slowest, was the one with no
// bar. It observes rather than starts, because the worker was
// already coming up when this page opened; nothing here owns that
// activation, so nothing here may cancel or navigate for it.
var bootWatch = activationClientCreate({
  onProgress: generatorChrome.setLoadingProgress,
});

// The switch's own watch, made per switch so an abandoned one cannot
// keep writing the overlay under its replacement. Null until the
// first switch of the session.
var switchWatch = null;

function startLoadProgressPoll() {
  bootWatch.observe();
}

function stopLoadProgressPoll() {
  bootWatch.stop();
}

function fetchModels() {
  return modelClientLoad();
}

function switchModel(id, device) {
  // Same model on the same device is a no-op (requestSwitch also
  // guards this before showing the confirm).
  var activeId = generatorModelPanel.activeModelId();
  var activeDevice = generatorModelPanel.activeDevice();
  if (id === activeId && (device || activeDevice) === activeDevice) {
    return;
  }
  generatorSocket.setReconnectSuppressed(true);
  generatorSocket.close();
  var name = generatorModelPanel.modelDisplayName(id);
  generatorChrome.setLoadingText(
    "Loading " + name + "\u2026"
  );
  // The switch's own watch drives the bar from here on; stop the
  // boot one so the two are never writing the same overlay. Seeding
  // with the same state the first poll will report keeps the opening
  // frame from saying "Loading" for a poll interval before
  // correcting itself to "Starting worker".
  stopLoadProgressPoll();
  generatorChrome.setLoadingProgress("starting", null);
  raiseLoadingOverlay();
  generatorModelPanel.setDisabled(true);

  switchWatch = activationClientCreate({
    onProgress: function (state, progress) {
      generatorChrome.setLoadingText(
        "Loading " + name + "\u2026"
      );
      generatorChrome.setLoadingProgress(state, progress);
    },
    onReady: function () {
      // Dropped here rather than before the request, which is where
      // it used to happen. A switch ends in a reload, and the
      // restore path cannot tell that from a trip to Analytics and
      // back, so the snapshot has to go before the reload; the
      // identity check alone cannot do it, since switching away and
      // back lands on a matching (model, device) pair again and the
      // stale run would return. But clearing it up front meant a
      // switch that was refused, for a missing venv or a model that
      // could not fit, threw away the run on screen for nothing.
      clearSessionState();
      generatorChrome.finishLoadingProgress(function () {
        location.reload();
      });
    },
    onFailed: function (message) {
      switchFailed(new Error(message));
    },
  });
  switchWatch.start(id, { device: device }).catch(switchFailed);
}

function switchFailed(err) {
  generatorSocket.setReconnectSuppressed(false);
  generatorModelPanel.setDisabled(false);
  generatorModelPanel.refreshSelector();
  stopLoadProgressPoll();
  if (switchWatch) {
    switchWatch.stop();
    switchWatch = null;
  }
  // Tear the track down rather than leaving it to the next switch to
  // re-sync. The overlay hides by going transparent, not by leaving
  // the layout, so a sweep left on it would keep animating unseen for
  // the rest of the session. "idle" is the reducer's way of saying no
  // activation is in flight, which is exactly the state after this.
  generatorChrome.setLoadingProgress("idle", null);
  generatorChrome.hideLoading();
  generatorChrome.setMessage(
    "Model switch failed: " + err.message,
    { color: "var(--danger)" }
  );
}

// ---- WebSocket connection ----

function generatorSocketOpened() {
  generatorChrome.setConnection("loading");
}

function generatorSocketClosed() {
  generatorChrome.setConnection("disconnected");
  modelReady = false;
  // Nothing is sampling this machine any more, so the meter must
  // stop claiming to. A switch between models comes through here.
  generatorChrome.clearResourceMeter();
  // A run in flight when the socket drops has stopped: the worker
  // treats the disconnect as a cancel, so there is no terminal
  // frame coming and nothing left computing. Leaving the
  // generating state here is not the same as merely clearing the
  // flag, which the report rejected: the run reaches the labelled
  // stopped state, keeping its frames while refusing to present
  // them as complete.
  if (isGenerating) {
    enterInterruptedState();
  }
  updateGenerateButton();
}

function generatorSocketMalformed(error) {
  console.warn(
    "Ignoring malformed generator socket frame:",
    error
  );
}

function generatorSocketFatal(error) {
  console.error(
    "Generator socket failure:",
    error
  );
}

// ---- Message handler ----

function handleMessage(data) {
  switch (data.type) {
    case "resident":
      handleResident(data);
      break;
    case "model_status":
      handleModelStatus(data);
      break;
    case "frame":
      handleFrame(data);
      break;
    case "candidates":
      handleCandidates(data);
      break;
    case "done":
      handleDone(data);
      break;
    case "error":
      handleError(data);
      break;
    case "tokenize_result":
      handleTokenizeResult(data);
      break;
    case "probe_result":
      handleProbeResult(data);
      break;
    case "count_prompt_result":
      generatorComposer.handleCountResult(data);
      break;
    case "resource_sample":
      generatorChrome.handleResourceSample(data);
      break;
  }
}

function handleModelStatus(data) {
  if (data.status === "loading") {
    generatorChrome.setConnection("loading");
    modelReady = false;
    generatorChrome.setLoadingText(
      "Loading "
      + (generatorModelPanel.activeDisplayName() || "model")
      + "\u2026"
    );
    raiseLoadingOverlay();
    startLoadProgressPoll();
    updateGenerateButton();
  } else if (data.status === "ready") {
    generatorChrome.setConnection("ready");
    modelReady = true;
    stopLoadProgressPoll();
    updateGenerateButton();
    // The first count the page can make: the prompt was there from
    // boot (a default, a restored session, or a saved draft), but only
    // a ready worker can tokenize it.
    generatorComposer.textChanged();
    // Only the overlay waits. The model is usable the moment the
    // worker says so, and holding the Generate button for a
    // cosmetic beat would be the wrong trade. Re-checking readiness
    // inside the hold keeps a status that flips back mid-beat from
    // pulling the overlay off a load that is starting again.
    generatorChrome.finishLoadingProgress(function () {
      if (modelReady) {
        generatorChrome.hideLoading();
      }
    });
  }
}

// The supervisor's statement of who this socket reaches, sent before
// any worker traffic. Almost always the model this page was built
// for, in which case the only question left is whether it is still
// the worker that made the run on screen (adoptResidentWorker).
//
// When it is not, another window switched the model out from under
// us. This page's cached model, device, capability gates and entire
// parameter form describe a worker that no longer exists, so a
// Generate from here would be labelled and parameterised for one
// model and answered by another, often accepted through defaults
// rather than refused. Reloading is what makes the page describe
// what is actually there.
function handleResident(data) {
  if (!data || !data.model) {
    return;
  }
  var activeId = generatorModelPanel.activeModelId();
  var activeDevice = generatorModelPanel.activeDevice();
  var sameModel = data.model === activeId;
  var sameDevice =
    !data.device || !activeDevice || data.device === activeDevice;
  if (sameModel && sameDevice) {
    adoptResidentWorker(data.worker);
    return;
  }
  // Nothing here may be generated against, and the reconnect loop
  // must not race the reload by pulling the page back onto the new
  // worker as though it belonged here.
  modelReady = false;
  generatorSocket.setReconnectSuppressed(true);
  updateGenerateButton();
  var name = generatorModelPanel.modelDisplayName(data.model);
  generatorChrome.setConnection("loading");
  generatorChrome.setLoadingText(
    "Model changed to " + name + "\u2026"
  );
  generatorChrome.setLoadingProgress("idle", null);
  raiseLoadingOverlay();
  generatorChrome.setMessage(
    "The model was changed to " + name + " in another window."
  );
  rescueRunThenReload();
}

// The same model and device, which may still be a different worker:
// loaded again from another window, or after the supervisor restarted.
// Everything this page was built for still holds, its model, device
// and form, so nothing reloads. Only the run on screen goes stale,
// held by no live worker, and it locks in place, still savable.
//
// An open session closes the way Exit does, since nothing in it can
// run now, unless it holds a branch the page can still save
// (runPhasesKeepsWork): Confirm needs no worker.
function adoptResidentWorker(worker) {
  if (typeof worker !== "string" || worker === "") {
    return;
  }
  var wasBlocked = runEditBlock() !== "";
  generatorRun.adoptResidentWorker(worker);
  var blocked = runEditBlock();
  if (!blocked || wasBlocked) {
    updateEditFramesLock();
    return;
  }
  if (runPhasesEditing(runPhase) && !runPhasesKeepsWork(runPhase)) {
    exitRemaskMode();
  }
  updateEditFramesLock();
  generatorChrome.setMessage(blocked);
}

// How long to let a rescue save finish before reloading anyway. The
// page is describing a worker that no longer exists, so it cannot be
// left here indefinitely on the chance that a request completes.
var RESCUE_SAVE_TIMEOUT_MS = 8000;

// Save an unsaved run, then reload onto the model that is actually
// resident.
//
// The run itself cannot be continued: resume, What If and probe all
// read state held by the worker that produced it, and that process
// is gone. Keeping it on screen would preserve something to look at
// and nothing to act on. It can still be *saved*, though, because a
// disconnect does not disable saving and, since `DATA-04`, a save
// carries its own provenance instead of reading whatever worker is
// resident. So the run outlives its worker exactly long enough to be
// written down, which is the one thing worth rescuing.
//
// Auto-saving rather than asking follows what entering What If
// already does with an unsaved run. An unwanted run can be deleted
// from Analytics; a lost one cannot be recovered.
function rescueRunThenReload() {
  if (
    generatorRun.saved()
    || generatorRun.frameCount() === 0
    || !generatorRun.finalText()
  ) {
    location.reload();
    return;
  }
  generatorChrome.startRunStatus(
    "Saving run before reloading"
  );
  var timeout = new Promise(function (resolve) {
    setTimeout(resolve, RESCUE_SAVE_TIMEOUT_MS);
  });
  Promise.race([generatorRun.save(), timeout]).then(
    function () {
      location.reload();
    },
    function () {
      location.reload();
    }
  );
}

// The status bar's step reading, in one place.
//
// Two shapes, because two kinds of model. A fixed schedule reports
// its position out of a known total; an adaptive-stopping model has
// no total to report against, so it names the canvas instead.
//
// Shared by the live path and the scrubber, which is the point:
// scrubbing used to leave this frozen on whatever the run ended at,
// so a finished DiffusionGemma run read "Step 87, Canvas 4" at every
// frame. Each caller supplies its own step number, because during a
// resume the live view counts the branch's steps while the scrubber
// counts the whole run.
function stepReadout(step, canvasIndex, totalSteps, prefix) {
  if (typeof totalSteps === "number") {
    return prefix + step + "/" + totalSteps;
  }
  var canvas = typeof canvasIndex === "number" ? canvasIndex : 0;
  // canvas_index is 0-based internally; display it 1-based.
  return prefix + step + ", Canvas " + (canvas + 1);
}

// A diffusion run's candidates, sent once just before its done frame.
// Their frame numbers are the stream's own, so they go where that
// stream's frames went: after the point a resume branched from, and
// from 0 for a fresh run.
function handleCandidates(data) {
  generatorRun.addCandidates(data);
}

function handleFrame(data) {
  var appended;
  try {
    appended = generatorRun.appendFrame(data);
  } catch (error) {
    reportRunDesync(error);
    return;
  }
  if (appended.append) {
    handleAppendFrame(data, appended);
    return;
  }
  // The token view needs per-position metadata; a model that does not
  // send it still gets the character renderer.
  if (appended.tokens) {
    generatorCanvas.renderLiveFrame(
      appended.tokens,
      data.revealed,
      data.live_candidates
    );
  } else {
    generatorCanvas.renderTextFrame(appended.text);
  }
  generatorReadouts.refreshStop();

  updateLiveFrameStatus(data);
}

// One position arrives; the run grows by one frame.
//
// The rendering half is deliberately unchanged: `renderLiveFrame`
// still gets the whole sequence, because that is what drawing a
// canvas needs, and assembling it from the store is a slice. What
// changed is that the slice happens here instead of arriving over
// the wire N times.
function handleAppendFrame(data, appended) {
  generatorCanvas.renderLiveFrame(
    appended.tokens, data.revealed, null
  );
  updateLiveFrameStatus(data);
}

// The step reading and the footer, for whichever shape delivered the
// frame. Shared rather than written twice: the two numbers below
// were told apart once already, and a second copy of that reasoning
// is a second place for them to drift back together.
//
// Two numbers wearing one name until they were told apart. A frame
// carries its own segment's budget, which is what makes
// "Resuming 12/64" mean something while a branch runs. The scrubber
// wants the whole run's total, so only a generation writes that one
// down: a resume reports the steps it has left, and letting that
// through left a finished run measured against its last branch. An
// edit at frame 64 of 128 then scrubbed to "Step 128/64", and a
// branch abandoned with Retry left the denominator of a run that no
// longer existed on screen.
function updateLiveFrameStatus(data) {
  var frameSteps =
    typeof data.total_steps === "number"
      ? data.total_steps
      : null;
  if (!isResuming) {
    generatorRun.setTotalSteps(frameSteps);
  }
  generatorChrome.setStep(
    stepReadout(
      data.index,
      data.canvas_index,
      frameSteps,
      isResuming ? "Resuming " : "Step "
    )
  );
  updateRunRateFooter();
}

// Stop the run and say why, rather than carrying on with a canvas
// that no longer describes the generation. Routed through the same
// scoped-error path a worker error takes, so it ends the run and
// leaves the page usable instead of tearing down the session.
function reportRunDesync(error) {
  var detail = error && error.message ? error.message : "unknown";
  handleError({
    type: "error",
    scope: "run",
    code: "frame_desync",
    message:
      "The run's frames arrived out of order, so it was stopped"
      + " rather than shown incorrectly (" + detail + ").",
  });
}

// Rebuild the step reading for a frame the user scrubbed to. The
// scrubber counts the whole run, so the array index is the step,
// which is the same number the "Frame N / M" label beside it shows.
function renderScrubStepReadout(index) {
  var total = generatorRun.totalSteps();
  if (
    total === null
    && generatorRun.frameCanvasSeries().length === 0
  ) {
    return;
  }
  generatorChrome.setStep(
    stepReadout(
      index,
      generatorRun.frameCanvas(index),
      total,
      "Step "
    )
  );
}

// ---- Elapsed and tokens per second ----

// Both readouts come off the run's elapsed series, not the frame in
// hand. data.elapsed is segment-local: after an edit the worker times
// the branch from zero, so reading it directly made the footer's
// Elapsed jump backwards mid-run. The controller already carries
// the pre-edit total (see handleFrame), so its tail is the real
// wall-clock time of everything generated so far.
function updateRunRateFooter() {
  if (generatorRun.frameCount() === 0) {
    return;
  }
  generatorChrome.updateRateFooter({
    elapsedSeconds: generatorRun.lastElapsed(),
    rate: currentTokensPerSecond(),
    tpsMode: appSettings.tpsMode,
  });
}

// null whenever the rate would be meaningless rather than merely
// zero: no tokens yet, or a window too short to have been timed. The
// first frame after an edit is the second case, since it lands at the
// pre-edit total and so shares a timestamp with the frame before it.
function currentTokensPerSecond() {
  return generatorRun.tokensPerSecond(appSettings.tpsMode);
}

function toggleTpsMode() {
  appSettings.tpsMode =
    appSettings.tpsMode === "last" ? "total" : "last";
  overlaysWriteTpsMode(appSettings.tpsMode);
  generatorChrome.renderTpsFooter(
    currentTokensPerSecond(), appSettings.tpsMode
  );
}

// The run on screen stopped without a terminal frame to say so,
// which happens when the connection drops mid-run rather than when
// Stop was pressed. Reached from the socket close callback only:
// every other stop arrives as a cancelled ``done`` and goes through
// handleDone, which has the run's own text and token to record too.
function enterInterruptedState() {
  setGenerating(false);
  isResuming = false;
  pendingResume = null;
  generatorChrome.endRunStatus();
  var hasFrames = generatorRun.interruptConnection();
  updateEditFramesLock();
  generatorChrome.setMessage(
    "Stopped: lost the connection mid-run."
  );
  // The frames already on screen are real and worth keeping, so
  // the scrubber and Save stay available. What the run cannot do
  // is claim it finished.
  if (hasFrames) {
    setSaveAvailable(true);
    activateScrubber();
    // Kept the way a finished run is, so leaving for Analytics
    // before saving does not lose it (skip while mid guided-edit).
    if (runPhase.mode === null) {
      generatorRun.saveSession();
    }
  }
}

function handleDone(data) {
  var resumed = pendingResume;
  pendingResume = null;
  setGenerating(false);
  isResuming = false;
  generatorChrome.endRunStatus();
  // A resume that sent nothing changed nothing, on either side. The
  // rest of this is skipped on purpose: the thinking panel is what
  // Save reads, and this frame's empty thinking would clear it.
  if (resumeStoppedBeforeAFrame(resumed, data)) {
    landBeforeResume(resumed);
    return;
  }
  // A stopped run is still a run: it keeps its frames, its scrubber
  // and its edit tools. What it must not do is claim it finished,
  // because the text simply ends either way and nothing else on
  // screen says which. The flag also rides along to the save, so
  // the record cannot outlive the distinction.
  var completed = generatorRun.finish(data);
  var terminalMessage =
    completed.interrupted ? "Stopped." : "Done.";
  // The chip is still fading as the line fills in beneath it, so
  // ease the row's new shape instead of snapping the chip sideways.
  generatorChrome.setMessage(terminalMessage);
  if (thinkingPanel && thinkingContent) {
    if (data.thinking) {
      thinkingContent.textContent = data.thinking;
      thinkingPanel.hidden = false;
    } else {
      thinkingPanel.hidden = true;
      thinkingContent.textContent = "";
    }
  }
  setSaveAvailable(true);

  if (runPhase.mode === "generating") {
    handleGuidedDone();
  } else {
    activateScrubber();
  }

  // Persist the completed run so it survives navigating to
  // Analytics and back (skip while mid guided-edit).
  if (runPhase.mode === null) {
    generatorRun.saveSession();
  }
}

function handleError(data) {
  // How far this reaches is the worker's to say (see wire_errors.js).
  // Everything below used to run for every error, which meant a probe
  // refused because a generation was busy closed What If and threw
  // away the edit being composed.
  var routed = wireErrorsRoute(data);
  if (routed.unwindsRun) {
    setGenerating(false);
    isResuming = false;
    generatorChrome.endRunStatus();
    if (runPhasesEditing(runPhase)) {
      // A resume or substitution truncates the run before the worker
      // answers, so a rejected request would otherwise strand the
      // user with a half-run. Roll back to the pre-session snapshot.
      restoreEditSnapshot();
      resetGuidedMode();
    }
  }
  generatorChrome.setMessage(
    "Error: " + routed.message,
    {
      color: "var(--danger)",
      clearColorAfterMs: 5000,
    }
  );
  // Said either way: an auxiliary failure is still worth reading, and
  // the change here is what gets undone, not what gets shown.
  if (routed.unwindsRun && generatorRun.frameCount() > 1) {
    activateScrubber();
  }
}

// ---- Rendering ----

// Whether any position captured competing candidates for the hover
// popover (and, for models that support it, What If substitution).
function alternativesAvailable() {
  return generatorRun.hasAlternatives(false);
}

// The earliest position a What If branch could differ at: the
// leftmost remasked position across every edit. Null on an unedited
// run. Left of it the branch reproduces the original verbatim,
// candidate sets included, so there is nothing to compare there.
function editDivergencePosition() {
  var earliest = null;
  for (var e = 0; e < remaskEdits.length; e++) {
    var positions = remaskEdits[e].token_positions || [];
    for (var p = 0; p < positions.length; p++) {
      if (earliest === null || positions[p] < earliest) {
        earliest = positions[p];
      }
    }
  }
  return earliest;
}

// ---- Top-k alternatives popover ----

// Whether the popover is currently a surface the user is working
// in rather than a readout they are passing over. True from the
// first keystroke (or focus) until the entry is confirmed, cancelled
// or run.
//
// This is the whole reason the typed token needed groundwork: the
// popover is hover-scoped, and four separate listeners tear it down
// when the pointer or the viewport moves. Every one of them would
// otherwise erase a half-typed word. While pinned they stand down,
// and the only ways out are deliberate: Escape, the cancel button,
// a click outside, or running the substitution.
function altsPopoverPinned() {
  return typedEntryActive || typedEntryToken !== null;
}

function hideAltsPopover() {
  if (!altsPopover) {
    return;
  }
  altsPopover.hidden = true;
  altsPopover.textContent = "";
  altsPopoverPos = null;
  altsPopoverPage = null;
  // Same reason as in renderAltsPopover: the rows go without firing
  // the mouseleave that would have cleared their readout. Needed here
  // too, because scroll and resize close the popover on their own
  // rather than through a pointer leaving it.
  generatorReadouts.setCandidateHover(null);
  // The popover is the draft's only home, so closing it discards
  // the draft. Callers that must not do that check altsPopoverPinned
  // first; the rest (a new run, a frame change, a mode reset) are
  // meant to clear everything.
  clearTypedEntry();
}

// Drop the draft and any confirmed token, without touching the
// popover itself. Split from hideAltsPopover so cancelling an entry
// can leave the candidates on screen.
function clearTypedEntry() {
  typedEntryPos = null;
  typedEntryDraft = "";
  typedEntryActive = false;
  typedEntrySeeded = false;
  typedEntryPreview = null;
  typedEntryToken = null;
  typedEntryMeasure = null;
}

// Whether this position has a candidate set from each run to page
// between. Only possible at or past the divergence point, and only
// for a branch whose pre-edit candidates were retained.
function altsPageable(pos) {
  var divergence = editDivergencePosition();
  if (divergence === null || pos < divergence) {
    return false;
  }
  var original =
    generatorRun.positionAlternatives(pos, true);
  var edited =
    generatorRun.positionAlternatives(pos, false);
  return !!(
    original && original.length > 0
    && edited && edited.length > 0
  );
}

// Build one row per candidate: the token text, a proportional bar,
// and its probability. The chosen token is marked so the popover
// reads as "what it picked, and what it nearly picked instead".
function buildAltsRows(alts, chosenId) {
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < alts.length; i++) {
    fragment.appendChild(
      overlaysBuildAltRow(
        alts[i],
        chosenId,
        generatorReadouts.setCandidateHover,
        i
      )
    );
  }
  return fragment;
}

// Which run's candidates a pageable position opens on: the one the
// crossfade is favoring, so the popover agrees with the tokens and
// the entropy strip. Both pages stay reachable through the arrows
// either way, so the midpoint picks a default rather than gating
// access.
function defaultAltsPage() {
  return generatorCanvas.blendFavorsOriginal()
    ? "original"
    : "edited";
}

// Show the candidate popover for a token position, anchored to its
// span. The pager reaches the pre-edit set where one was retained:
// by position on an append run, by frame on a canvas run.
function showAltsPopover(pos, span) {
  if (generatorRun.frameIsAppend()) {
    altsPopoverPage = altsPageable(pos) ? defaultAltsPage() : null;
  } else {
    altsPopoverPage = candidatesPage();
  }
  renderAltsPopover(pos, span);
}

// Flip pages in place. Rendered without an anchor deliberately: the
// two pages can differ in height, and re-placing the box under the
// pointer that just clicked an arrow can slide it out from under that
// pointer, firing the mouseleave that closes it.
function setAltsPage(page) {
  if (altsPopoverPos === null) {
    return;
  }
  altsPopoverPage = page;
  renderAltsPopover(altsPopoverPos, null);
}

// With an anchor span, positioned in viewport coordinates (the
// popover is fixed at body level) and flipped above the token when it
// would overflow. Without one, left where it already sits.
function renderAltsPopover(pos, span) {
  if (!altsPopover) {
    return;
  }
  if (!generatorRun.frameIsAppend()) {
    renderCandidatesPopover(pos, span);
    return;
  }
  var original = altsPopoverPage === "original";
  var alts = generatorRun.positionAlternatives(
    pos, original
  );
  if (!alts || alts.length === 0) {
    hideAltsPopover();
    return;
  }
  // Each page marks the token its own run drew, so the Original page
  // does not mark the branch's substitution as chosen.
  var tokens = original
    ? generatorRun.originalTokensLast()
    : generatorRun.frameTokens(currentScrubFrame);
  var chosen = tokens && tokens[pos] ? tokens[pos].id : null;

  // Discarding the rows discards their pending mouseleave: a removed
  // node never fires one, so a readout for a row that no longer
  // exists would sit in the strip until the next hover. Cleared here
  // rather than per caller, since every rebuild comes through here.
  generatorReadouts.setCandidateHover(null);
  altsPopover.textContent = "";
  altsPopover.appendChild(
    overlaysBuildAltHeading(pos, altsPopoverPage, setAltsPage)
  );
  altsPopover.appendChild(buildAltsRows(alts, chosen));
  // Substitution only ever applies to the live run, so the Original
  // page is read-only even while What If is armed.
  var pickable = runPhase.substituting && !original;
  if (pickable) {
    altsPopover.appendChild(buildTypedEntry(pos));
    var hint = document.createElement("div");
    hint.className = "alt-hint";
    hint.textContent = "Click a candidate to substitute";
    altsPopover.appendChild(hint);
  }
  var tokenizer = overlaysBuildAltTokenizer(
    generatorModelPanel.activeTokenizer()
  );
  if (tokenizer) {
    altsPopover.appendChild(tokenizer);
  }
  altsPopover.classList.toggle("alt-pickable", pickable);
  placeAltsPopover(span);
  altsPopoverPos = pos;
}

// A canvas run's popover: what the position was weighing at the frame
// on screen, or at the latest captured frame before it when the
// stride skipped this one. From the frame an edit branched at on it
// pages between the two runs, opening on the one the crossfade
// favours; if that run has nothing there it stays closed rather than
// show the other run's candidates under this one's tokens.
function renderCandidatesPopover(pos, span) {
  var page = altsPopoverPage;
  var reading = candidatesReading(page, pos);
  if (reading === null) {
    hideAltsPopover();
    return;
  }
  var other = page === null
    ? null
    : candidatesReading(otherAltsPage(page), pos);
  generatorReadouts.setCandidateHover(null);
  altsPopover.textContent = "";
  altsPopover.appendChild(
    overlaysBuildStepHeading(
      pos, reading.frame, reading.shown, page,
      other === null ? null : setAltsPage
    )
  );
  altsPopover.appendChild(
    buildAltsRows(reading.set.c, reading.set.h)
  );
  var tokenizer = overlaysBuildAltTokenizer(
    generatorModelPanel.activeTokenizer()
  );
  if (tokenizer) {
    altsPopover.appendChild(tokenizer);
  }
  altsPopover.classList.remove("alt-pickable");
  placeAltsPopover(span);
  altsPopoverPos = pos;
}

// The canvas a frame belongs to, 0 for a model with only one.
function runFrameCanvas(frame) {
  return generatorRun.frameCanvas(frame);
}

// The earliest frame any edit branched at, or null on an unedited
// run. Before it both runs hold the same frames, so there is only
// one set of candidates to show.
function editDivergenceFrame() {
  var earliest = null;
  for (var e = 0; e < remaskEdits.length; e++) {
    var frame = remaskEdits[e].frame_index;
    if (earliest === null || frame < earliest) {
      earliest = frame;
    }
  }
  return earliest;
}

// The page a canvas run's popover opens on: the run the crossfade
// favours, from the frame an edit branched at on, and null before it,
// on an unedited run, or while an edit phase has the crossfade off.
function candidatesPage() {
  var divergence = editDivergenceFrame();
  if (
    !generatorCanvas.blendActive()
    || divergence === null
  ) {
    return null;
  }
  return currentScrubFrame >= divergence ? defaultAltsPage() : null;
}

function otherAltsPage(page) {
  return page === "original" ? "edited" : "original";
}

// One run's set for a position, at the frame that run is showing, as
// {frame, set, shown}, or null when it has none there. The edited run
// shows the scrubbed frame; the original shows the frame its layer
// clamps to past its own end. An edited run is single-canvas, since
// Edit Frames is off for a run that chains canvases, so the pre-edit
// run's lookups are on canvas 0.
function candidatesReading(page, pos) {
  if (page === "original") {
    var shown = Math.min(
      currentScrubFrame,
      generatorRun.originalTokenFrames() - 1
    );
    return candidatesReadingOf(
      generatorRun.candidateSet(
        shown, pos, true, singleCanvas
      ),
      shown
    );
  }
  return candidatesReadingOf(
    generatorRun.candidateSet(
      currentScrubFrame,
      pos,
      false,
      runFrameCanvas
    ),
    currentScrubFrame
  );
}

function candidatesReadingOf(found, shown) {
  if (found === null) {
    return null;
  }
  return { frame: found.frame, set: found.set, shown: shown };
}

function singleCanvas() {
  return 0;
}

// Measure before placing: the popover must be visible for its height
// to be known, so unhide first, then correct the position. Without a
// span it stays where it already sits.
function placeAltsPopover(span) {
  altsPopover.hidden = false;
  if (!span) {
    return;
  }
  var rect = span.getBoundingClientRect();
  var box = altsPopover.getBoundingClientRect();
  altsPopover.style.left =
    overlaysPopoverLeft(rect, box) + "px";
  altsPopover.style.top =
    overlaysPopoverTop(
      rect, box, outputArea.getBoundingClientRect().top
    ) + "px";
}

// ---- Typed token entry ----
//
// The row under the candidates that lets you force a token the model
// never offered. Everything it shows is derived from the state block
// near the top of this file rather than from its own DOM, so the
// popover can rebuild around it without losing a draft.
//
// Debounce for the preview. Long enough that a normal typing burst
// sends one request instead of one per key, short enough that the
// pieces feel like they track the text.
var TYPED_PREVIEW_DEBOUNCE_MS = 120;

// Matches TOKENIZE_TEXT_MAX_CHARS in worker_base.py. Enforced here
// too so the field simply stops accepting rather than silently
// diverging from what the worker would resolve.
var TYPED_INPUT_MAX_CHARS = 200;

var typedPreviewTimer = null;

// Build whatever the entry should look like right now: the confirmed
// token if there is one, otherwise the input.
function buildTypedEntry(pos) {
  if (typedEntryPos !== pos) {
    clearTypedEntry();
    typedEntryPos = pos;
  }
  if (typedEntryToken !== null) {
    return buildTypedSolidified();
  }
  return buildTypedInput(pos);
}

// A leading space is decided by the token being replaced, not by
// guessing at sentence position. Mid-sentence words carry their
// space inside the token, so replacing one without it silently
// welds the result onto the previous word. Reading the original is
// exact where a "looks mid-sentence" rule would only be usually
// right, and a single backspace overrides it.
function typedEntrySeedText(pos) {
  var tokens = generatorRun.frameTokens(currentScrubFrame);
  var token = tokens && tokens[pos] ? tokens[pos] : null;
  var text = token && typeof token.t === "string" ? token.t : "";
  return text.charAt(0) === " " ? " " : "";
}

function buildTypedInput(pos) {
  var wrap = document.createElement("div");
  wrap.className = "typed-entry";

  if (!typedEntrySeeded) {
    typedEntrySeeded = true;
    typedEntryDraft = typedEntrySeedText(pos);
  }

  var field = document.createElement("input");
  field.type = "text";
  field.className = "typed-input";
  field.maxLength = TYPED_INPUT_MAX_CHARS;
  field.value = typedEntryDraft;
  field.spellcheck = false;
  field.autocomplete = "off";
  field.addEventListener("input", onTypedInput);
  field.addEventListener("focus", onTypedFocus);

  // The field and its drawn hint share a positioning context, so the
  // hint can sit over the text origin without measuring the wrap's
  // own border and padding.
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

  // Deferred: the popover is still being assembled, and focusing a
  // node mid-build scrolls it into view before it has been placed.
  if (typedEntryActive) {
    setTimeout(function () {
      if (!typedEntryActive || !document.contains(field)) {
        return;
      }
      field.focus();
      var end = field.value.length;
      field.setSelectionRange(end, end);
    }, 0);
  }
  return wrap;
}

// The greyed hint inside the field, drawn rather than delegated to a
// native ``placeholder``. The native one renders only on the empty
// string, and a mid-sentence position seeds the field with a leading
// space, so it was invisible in precisely the case that needed it.
function buildTypedPlaceholder() {
  var hint = document.createElement("span");
  hint.className = "typed-placeholder";
  syncTypedPlaceholder(hint);
  return hint;
}

// Derived from the live draft, not from whether the field was seeded.
// That is what makes the dot a lesson rather than a label: backspace
// the space away and the dot goes with it, type it back and it
// returns, so the hint keeps saying whether a leading space is
// currently there.
function typedPlaceholderText() {
  if (typedEntryDraft === "") {
    return "Enter your own";
  }
  // The same stand-in overlaysAltDisplay gives a space, so the hint
  // reads continuously with the candidate rows above it.
  if (typedEntryDraft === " ") {
    return "\u00B7Enter your own";
  }
  return "";
}

function syncTypedPlaceholder(hint) {
  var text = typedPlaceholderText();
  hint.textContent = text;
  hint.hidden = text === "";
}

// Confirm and cancel, which slide down from behind the field's right
// edge once it is in use. Present in the DOM either way so the drop
// is a CSS state change rather than an insertion.
function buildTypedActions() {
  var actions = document.createElement("div");
  actions.className = "typed-actions";
  if (typedEntryActive) {
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

// Only exactly one token can be confirmed. Everything downstream of
// a substitution is position-indexed (overlaysComputeDiff, the
// entropy chart, the edit marker), so a replacement of any other
// length would shift every index after it.
function typedEntryResolvesToOne() {
  return (
    typedEntryPreview !== null
    && typedEntryPreview.count === 1
    && typedEntryDraft !== ""
  );
}

function syncTypedConfirm(confirm) {
  confirm.disabled = !typedEntryResolvesToOne();
  confirm.title = confirm.disabled
    ? "Type text that resolves to exactly one token"
    : "Use this token";
}

// Update the parts of a live entry that change as you type, in
// place. Deliberately not a rebuild: recreating the input on every
// keystroke would tear the focused element out from under the
// caret, and the characters typed before it was put back would go
// nowhere. Only a structural change (confirming, retrying,
// cancelling) goes through redrawTypedEntry.
function refreshTypedControls() {
  if (!altsPopover) {
    return;
  }
  var actions = altsPopover.querySelector(".typed-actions");
  if (actions) {
    actions.classList.toggle("is-open", typedEntryActive);
  }
  var confirm = altsPopover.querySelector(".typed-confirm");
  if (confirm) {
    syncTypedConfirm(confirm);
  }
  var hint = altsPopover.querySelector(".typed-placeholder");
  if (hint) {
    syncTypedPlaceholder(hint);
  }
  var preview = altsPopover.querySelector(".typed-preview");
  if (preview) {
    renderTypedPieces(preview);
  }
}

// The pieces the draft resolves to, with their vocabulary ids. Shown
// rather than merely counted: a rejection that says "3 tokens" is a
// verdict, while showing which three is an explanation.
function renderTypedPieces(host) {
  host.textContent = "";
  var preview = typedEntryPreview;
  if (preview === null || typedEntryDraft === "") {
    return;
  }
  for (var i = 0; i < preview.pieces.length; i++) {
    host.appendChild(buildTypedPiece(preview.pieces[i], i));
  }
  var note = document.createElement("span");
  note.className = "typed-count";
  if (preview.count === 1) {
    note.textContent = "1 token";
  } else if (preview.count === 0) {
    note.textContent = "no tokens";
    note.classList.add("typed-count-over");
  } else {
    note.textContent = preview.count + " tokens";
    // Orange only here. It means edit or remask everywhere else in
    // the app, so spending it on the ordinary multi-piece case would
    // read as a state rather than as a problem.
    note.classList.add("typed-count-over");
  }
  host.appendChild(note);
}

function buildTypedPiece(piece, index) {
  var el = document.createElement("span");
  el.className = "typed-piece";
  // Alternating tints so adjacent pieces stay distinguishable when
  // a split falls inside a word and the boundary is the whole point.
  if (index % 2 === 1) {
    el.classList.add("typed-piece-alt");
  }
  var text = document.createElement("span");
  text.className = "typed-piece-text";
  text.textContent = overlaysAltDisplay(piece.t);
  el.appendChild(text);
  var id = document.createElement("span");
  id.className = "typed-piece-id";
  id.textContent = String(piece.id);
  el.appendChild(id);
  return el;
}

// The confirmed token, dressed as a candidate row so it clicks the
// same way. data-typed tells the click handler to send it down the
// worker's typed path rather than the captured one.
function buildTypedSolidified() {
  var row = document.createElement("div");
  row.className = "alt-row typed-solid";
  row.setAttribute(
    "data-alt-id", String(typedEntryToken.id)
  );
  row.setAttribute("data-typed", "1");

  var text = document.createElement("span");
  text.className = "alt-text";
  text.textContent = overlaysAltDisplay(typedEntryToken.t);
  row.appendChild(text);

  var tag = document.createElement("span");
  tag.className = "typed-tag";
  tag.textContent = "yours";
  row.appendChild(tag);

  // Left of the retry icon, where the candidate rows keep theirs, so
  // the six rows read as one column of odds rather than five plus an
  // exception.
  var prob = document.createElement("span");
  prob.className = "typed-prob";
  renderTypedMeasure(prob);
  row.appendChild(prob);

  var retry = document.createElement("button");
  retry.type = "button";
  retry.className = "typed-retry";
  retry.textContent = "\u21BA";
  retry.title = "Type a different token";
  retry.addEventListener("click", retryTypedEntry);
  row.appendChild(retry);

  // Bound directly rather than through overlaysBindAltHover, which
  // captures its candidate when the row is built. This one has to be
  // read at hover time, because the probe can land after the row is
  // drawn and would otherwise never reach the strip.
  row.addEventListener("mouseenter", function () {
    generatorReadouts.setCandidateHover(
      typedCandidateReading()
    );
  });
  row.addEventListener("mouseleave", function () {
    generatorReadouts.setCandidateHover(null);
  });
  return row;
}

// The typed token in the shape a candidate row hands the strip. The
// rank arrives with the measurement here rather than from the row's
// position, since a typed token has no position in a sorted list.
function typedCandidateReading() {
  if (typedEntryToken === null) {
    return null;
  }
  var reading = { t: typedEntryToken.t, p: null };
  if (typedEntryMeasure !== null) {
    reading.p = typedEntryMeasure.probability;
    reading.rank = typedEntryMeasure.rank;
    reading.vocab_size = typedEntryMeasure.vocabSize;
  }
  return reading;
}

// The measured odds, or a pending mark while the probe is out. The
// mark matters on CPU, where the pass takes long enough that a blank
// slot would read as "this row has no number" rather than "not yet".
function renderTypedMeasure(slot) {
  if (typedEntryMeasure === null) {
    slot.textContent = "\u2026";
    slot.classList.add("is-pending");
    slot.title = "Measuring what the model gave this token";
    return;
  }
  slot.classList.remove("is-pending");
  slot.textContent = typedProbabilityText(
    typedEntryMeasure.probability
  );
  slot.title = typedMeasureTitle(typedEntryMeasure);
}

// One decimal, like the candidate rows, but never a bare "0.0%". A
// typed token is most interesting precisely where it is improbable,
// so the one reading this must not produce is a flat zero for
// something the model did give a little weight to.
function typedProbabilityText(probability) {
  var p = Number(probability);
  if (!isFinite(p) || p <= 0) {
    return "0.0%";
  }
  if (p < 0.001) {
    return "<0.1%";
  }
  return (p * 100).toFixed(1) + "%";
}

// The precision the row has no width for. Rank is the honest answer
// where the percentage has collapsed: "one of 41,203 the model liked
// better" says what "<0.1%" cannot.
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

function onTypedFocus() {
  if (typedEntryActive) {
    return;
  }
  typedEntryActive = true;
  refreshTypedControls();
}

function onTypedInput(e) {
  typedEntryDraft = e.target.value;
  // The old pieces describe text that is no longer on screen, so
  // they go immediately rather than lingering until the reply.
  typedEntryPreview = null;
  refreshTypedControls();
  if (typedPreviewTimer !== null) {
    clearTimeout(typedPreviewTimer);
  }
  typedPreviewTimer = setTimeout(
    requestTypedPreview, TYPED_PREVIEW_DEBOUNCE_MS
  );
}

// Ask the worker what the draft resolves to. Fire and forget: the
// reply is matched by request id, since a slow answer for an earlier
// draft must not overwrite a newer one.
function requestTypedPreview() {
  typedPreviewTimer = null;
  if (!generatorSocket.isReady()) {
    return;
  }
  if (typedEntryDraft === "") {
    typedEntryPreview = null;
    refreshTypedControls();
    return;
  }
  typedEntryRequest += 1;
  generatorSocket.send({
    type: "tokenize",
    text: typedEntryDraft,
    request_id: typedEntryRequest,
  });
}

// Accept a preview only if it is both the newest request and still
// about the text on screen. The second test matters because the
// draft can change without a new request going out.
function handleTokenizeResult(msg) {
  if (msg.request_id !== typedEntryRequest) {
    return;
  }
  if (msg.text !== typedEntryDraft) {
    return;
  }
  typedEntryPreview = {
    pieces: msg.pieces || [],
    count: msg.count || 0,
  };
  refreshTypedControls();
}

function confirmTypedEntry() {
  if (!typedEntryResolvesToOne()) {
    return;
  }
  typedEntryToken = typedEntryPreview.pieces[0];
  typedEntryMeasure = null;
  redrawTypedEntry();
  requestTypedProbe();
}

// Ask the model what it actually gave this token here. Fired on
// confirm rather than while typing: it is a forward pass, not a
// vocabulary lookup, so one per decision is the right budget where
// one per keystroke would not be.
//
// The same figure the substitution will report, because both read the
// same distribution, so the row cannot promise a number the run then
// contradicts.
function requestTypedProbe() {
  if (typedEntryToken === null || typedEntryPos === null) {
    return;
  }
  if (fillMeasureFromRecord()) {
    return;
  }
  if (!generatorSocket.isReady()) {
    return;
  }
  if (editRequestRefused()) {
    return;
  }
  typedProbeRequest += 1;
  generatorSocket.send({
    type: "probe",
    position: typedEntryPos,
    token_id: typedEntryToken.id,
    request_id: typedProbeRequest,
    run_token: generatorRun.runToken(),
  });
}

// Answer from what the run already recorded, when it can. Typing a
// token that is one of the candidates on screen is the common case,
// and the stored figure is better than a measured one, not merely
// cheaper: the run sampled position n from a single decode step, and
// a probe rebuilds that with a fresh prefill, which in bf16 lands an
// ulp away. Two paths to one number is one number too many when the
// row sits directly above the rows it would disagree with.
//
// Returns whether it answered, so the caller knows to send nothing.
function fillMeasureFromRecord() {
  var alts = generatorRun.positionAlternatives(
    typedEntryPos, false
  );
  if (!alts) {
    return false;
  }
  for (var i = 0; i < alts.length; i++) {
    if (alts[i].id !== typedEntryToken.id) {
      continue;
    }
    typedEntryMeasure = {
      probability: alts[i].p,
      rank: overlaysAltRank(alts[i], i),
      vocabSize: metricsVocabSize(),
    };
    refreshTypedMeasure();
    return true;
  }
  return false;
}

// Accept a measurement only for the newest probe and only while the
// token it describes is still the confirmed one. A retry between
// request and reply would otherwise label the new token with the old
// token's odds, which is worse than showing nothing.
function handleProbeResult(msg) {
  if (msg.request_id !== typedProbeRequest) {
    return;
  }
  if (typedEntryToken === null) {
    return;
  }
  if (msg.token_id !== typedEntryToken.id) {
    return;
  }
  typedEntryMeasure = {
    probability: msg.probability,
    rank: msg.rank,
    vocabSize: msg.vocab_size,
  };
  refreshTypedMeasure();
}

// Fill the figure in place rather than rebuilding the row. The row is
// a click target, and replacing the node the pointer is resting on
// would drop the hover that says so.
function refreshTypedMeasure() {
  if (!altsPopover) {
    return;
  }
  var slot = altsPopover.querySelector(".typed-prob");
  if (slot) {
    renderTypedMeasure(slot);
  }
}

// Back to an empty field, keeping the position. Distinct from
// cancel: the user wants a different token, not to stop.
function retryTypedEntry(e) {
  // The retry icon lives inside a row that clicks through to
  // doSubstitute. Without this, changing your mind would run the
  // very substitution you were backing out of.
  e.stopPropagation();
  typedEntryToken = null;
  typedEntryMeasure = null;
  typedEntryPreview = null;
  typedEntryDraft = "";
  // Back to a fresh field, leading space and all, which is what the
  // user gets on a first visit and so what they expect on a redo.
  typedEntrySeeded = false;
  typedEntryActive = true;
  redrawTypedEntry();
}

function cancelTypedEntry() {
  var pos = altsPopoverPos;
  clearTypedEntry();
  if (typedPreviewTimer !== null) {
    clearTimeout(typedPreviewTimer);
    typedPreviewTimer = null;
  }
  if (pos === null || !altsPopover) {
    return;
  }
  // Where the pointer is decides what a cancel leaves behind. Over
  // the popover (the cancel button) it goes back to showing the
  // candidates. Anywhere else (Escape, a click outside) the pointer
  // has already left, so nothing would ever come along to close the
  // box, and it would sit there stranded.
  if (altsPopover.matches(":hover")) {
    renderAltsPopover(pos, null);
    return;
  }
  generatorReadouts.clearTokenHover();
  hideAltsPopover();
}

// Rebuild the popover around a structural change: the entry
// becoming a solidified row, or going back to a field. Unanchored on
// purpose, since re-placing the box against its token would walk it
// around under the pointer.
function redrawTypedEntry() {
  if (altsPopoverPos === null) {
    return;
  }
  renderAltsPopover(altsPopoverPos, null);
}

// Prompt composition lives in generator_composer.js. The page passes
// model-panel values and transports its count requests, and otherwise
// reaches it only through the controller API created above.

// The resting status line, which is where this page spells out
// results. An import is a one-off action with an outcome to report,
// not ongoing work, so it belongs here rather than in a status chip.
function setPromptImportStatus(text, danger) {
  generatorChrome.setMessage(text, {
    color: danger ? "var(--danger)" : "",
    clearColorAfterMs: danger ? 5000 : 0,
  });
}

// ---- Context window readout ----

// The page mediates between the two form controllers. Each controller
// writes only its members of the shared draft record.
function composerThinking() {
  return generatorModelPanel.thinking();
}

function composerOutputBudget() {
  return generatorModelPanel.outputBudget();
}

function composerCountReady() {
  return generatorSocket.isReady();
}

function sendComposerCount(payload) {
  if (!composerCountReady()) {
    return;
  }
  generatorSocket.send(payload);
}

function submitComposer() {
  // Enter runs a generation; in the finalized "New Run" state it is
  // a no-op so it cannot wipe the canvas unexpectedly.
  if (!generatorRun.editedSaved()) {
    startGeneration();
  }
}

function composerDraftChanged() {
  generatorModelPanel.saveDraft();
}

function modelPanelValidationChanged() {
  updateGenerateButton();
}

function modelPanelParametersChanged() {
  generatorComposer.saveDraft();
  generatorComposer.parametersChanged();
}

// ---- Persistent UI settings (localStorage) ----

// Load persisted preferences into appSettings. The schema, key, and
// parsing live in overlays.js (shared with the Settings page); edits
// happen there and are picked up here on the next load (hydrate).
function loadSettings() {
  appSettings = parseSettings(localStorage.getItem(SETTINGS_KEY));
}

// Apply the (saved) settings to the live app: hover highlight and any
// active token coloring.
function applySettings() {
  generatorCanvas.applySettings();
  // Toggling the effect starts/stops the Generate idle cycle live.
  updateGenerateIdleEffect();
  // Restart the collapsed device ticker so the GPU-ticker toggle takes
  // effect immediately.
  generatorModelPanel.refreshSelector();
}

// ---- Scrubber ----

// True when the current run spans more than one canvas.
// DiffusionGemma resume re-enters a single 256-token canvas, so
// multi-canvas runs cannot be resumed in this version; the editing
// UI stays hidden for them.
function runIsMultiCanvas() {
  return generatorRun.frameIsMultiCanvas();
}

// Reflect the "already saved an edit" lock on the Edit Frames button:
// greyed out and non-interactive until the next Generate clears the
// lock. Either way the button carries a tooltip, explaining the lock
// when locked and what the mode does when not, matching What If.
function updateEditFramesLock() {
  // A run the worker cannot answer for is locked whatever else holds,
  // and with its own reason, since that is the one that applies.
  // Retry and Continue lock too: each would run on that worker.
  var blocked = runEditBlock();
  if (blocked) {
    setButtonLocked(btnEditFrames, blocked);
    if (btnWhatIf) {
      setButtonLocked(btnWhatIf, blocked);
    }
    setButtonLocked(btnRetryEdit, blocked);
    setButtonLocked(btnContinueEdit, blocked);
    return;
  }
  setButtonUnlocked(btnRetryEdit, RETRY_EDIT_TITLE);
  setButtonUnlocked(btnContinueEdit, CONTINUE_EDIT_TITLE);
  // An edited save in flight locks too, not just a completed one:
  // confirmGuidedEdit fires the save and re-shows the buttons before
  // its async handler can set the edited-save flag, which otherwise
  // leave a live window where a second edit could be started.
  var locked = generatorRun.editedSaved()
    || (generatorRun.saving() && remaskEdits.length > 0);
  if (locked) {
    setButtonLocked(
      btnEditFrames,
      "This run already has a saved edit."
      + " Generate again to edit a new run."
    );
  } else {
    setButtonUnlocked(
      btnEditFrames,
      "Remask tokens at any frame, then resume the run"
      + " from there"
    );
  }
  if (!btnWhatIf) {
    return;
  }
  // What If writes the same single saved edit per generation, so it
  // locks on the same condition as Edit Frames.
  if (locked) {
    setButtonLocked(
      btnWhatIf,
      "This run already has a saved edit."
      + " Generate again to try another branch."
    );
  } else {
    setButtonUnlocked(
      btnWhatIf,
      "Replace a token with one the model nearly"
      + " chose, then regenerate"
    );
  }
}

// Why the run on screen cannot be edited at all, or "" when it can.
function runEditBlock() {
  return runPhasesEditBlock(generatorRun.editIdentity());
}

// Whether a request about the run on screen must not be sent, saying
// why when it must not. The edit buttons lock as well; this is for a
// session already open when the run stopped being editable, and is
// asked before anything is cut from the run for a branch.
function editRequestRefused() {
  var blocked = runEditBlock();
  if (!blocked) {
    return false;
  }
  generatorChrome.setMessage(blocked);
  return true;
}

// The lock is both visual and behavioural: pointer-events is off in
// CSS, aria-disabled announces it, and the callers keep their own
// is-locked guard so a programmatic click still cannot slip through.
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

// The primary button has three jobs, in priority order. While a run
// is in flight it is "Stop", because that is the only thing worth
// doing then and the slot was otherwise greyed out for the whole
// run. Once an edited run has been saved it is "New Run". Otherwise
// it is "Generate". Same slot and size throughout.
function currentGenerateLabel() {
  if (isGenerating) {
    return "Stop";
  }
  return generatorRun.editedSaved()
    ? "New Run"
    : "Generate";
}

function updateGenerateButton() {
  if (isGenerating) {
    btnGenerate.classList.remove("is-new-run");
    btnGenerate.classList.add("is-stop");
    // Live precisely when the old code greyed it out: a run in
    // flight is the one moment Stop means anything.
    btnGenerate.disabled = false;
  } else if (generatorRun.editedSaved()) {
    btnGenerate.classList.remove("is-stop");
    btnGenerate.classList.add("is-new-run");
    // New Run is client-side; only a completing save should hold it.
    btnGenerate.disabled = generatorRun.saving();
  } else {
    btnGenerate.classList.remove("is-new-run");
    btnGenerate.classList.remove("is-stop");
    btnGenerate.disabled =
      generatorRun.saving()
      || !(
        modelReady
        && generatorModelPanel.validation().valid
      );
  }
  // The label text is owned by the idle-effect controller (it either
  // sets the static label or drives the looping diffusion reveal).
  updateGenerateIdleEffect();
}

// ---- Generate button idle diffusion cycle ----

// One-time discovery nudge: the Generate button always idles with the
// diffusion cycle before the user's first-ever fresh run, then follows
// the "Render diffusion-style text" setting. Persisted per browser.
var GENERATE_TEASED_KEY = "diffusion_generate_teased";
// The button holds its resolved text longer than the status bar so the
// primary CTA reads calmly rather than flickering.
var GENERATE_CYCLE_HOLD_MS = 2000;
var generateCycleTimer = null;
var generateCycleActive = false;
var generateCycleLabel = "";

function generateTeaserActive() {
  try {
    return localStorage.getItem(GENERATE_TEASED_KEY) !== "1";
  } catch (_e) {
    return false;
  }
}

function markGenerateTeased() {
  // Write-through to the server (see persistSet) so the one-time teaser
  // does not replay every restart on a fresh window origin.
  persistSet(GENERATE_TEASED_KEY, "1");
}

// The button idles with the diffusion cycle while it is clickable:
// always before the first fresh run, and thereafter only when the
// effect setting is on. Reduced motion disables it entirely.
function generateIdleActive() {
  if (!btnGenerateLabel || prefersReducedMotion()) {
    return false;
  }
  if (btnGenerate.disabled || isGenerating) {
    return false;
  }
  return generateTeaserActive() || !!appSettings.diffusionText;
}

function startGenerateCycle() {
  if (generateCycleActive) {
    return;
  }
  generateCycleActive = true;
  var runOnce = function () {
    generateCycleLabel = currentGenerateLabel();
    denoiseReveal(
      btnGenerateLabel,
      generateCycleLabel,
      function () {
        generateCycleTimer = setTimeout(
          runOnce, GENERATE_CYCLE_HOLD_MS
        );
      },
      true
    );
  };
  runOnce();
}

function stopGenerateCycle() {
  generateCycleActive = false;
  cancelDenoise(btnGenerateLabel);
  if (generateCycleTimer !== null) {
    clearTimeout(generateCycleTimer);
    generateCycleTimer = null;
  }
  if (btnGenerateLabel) {
    btnGenerateLabel.textContent = currentGenerateLabel();
  }
}

function updateGenerateIdleEffect() {
  if (generateIdleActive()) {
    // Restart if the label changed (e.g. Generate -> New Run) so the
    // cycle animates the correct word without a lag.
    if (
      generateCycleActive
      && generateCycleLabel !== currentGenerateLabel()
    ) {
      stopGenerateCycle();
    }
    startGenerateCycle();
  } else {
    stopGenerateCycle();
  }
}

function activateScrubber() {
  if (generatorRun.frameCount() < 2) {
    return;
  }
  scrubberActive = true;
  currentScrubFrame = generatorRun.frameCount() - 1;

  scrubberSlider.min = "0";
  scrubberSlider.max =
    String(generatorRun.frameCount() - 1);
  scrubberSlider.value =
    String(currentScrubFrame);
  scrubberSlider.disabled = false;
  updateScrubberLabel();

  setScrubberVisible(true);
  var capabilities = generatorModelPanel.capabilities();
  btnEditFrames.hidden = !(
    capabilities.supports_resume
    && !runIsMultiCanvas()
  );
  // What If needs captured candidates to substitute from, so it stays
  // hidden when the run was generated with Alternatives off.
  if (btnWhatIf) {
    btnWhatIf.hidden = !(
      supportsSubstitution() && alternativesAvailable()
    );
  }
  updateEditFramesLock();
  generatorCanvas.activate();
  guidedEditControls.hidden = true;
  clearRemaskedPositions();
  unlockScrubberNav();

  navigateToFrame(currentScrubFrame);
  generatorReadouts.updateProfile();
}

// Show or hide the scrubber without moving anything around it. It
// keeps its height either way, so the output canvas above it is the
// same size before and after a run: see the `is-idle` note in
// index.html for why that matters on a page that restores a run
// after its first paint.
function setScrubberVisible(visible) {
  scrubberSection.classList.toggle("is-idle", !visible);
}

function deactivateScrubber() {
  scrubberActive = false;
  setScrubberVisible(false);
  guidedEditControls.hidden = true;
  generatorCanvas.deactivate();
  generatorReadouts.deactivate();
  hideAltsPopover();
  clearRemaskedPositions();
}

function updateScrubberLabel() {
  var maxLabel = (
    runPhase.mode === "select_target"
    && generatorRun.originalCaptured()
  ) ? generatorRun.originalTotalFrames() - 1
    : generatorRun.frameCount() - 1;
  scrubberLabel.textContent =
    "Frame " + currentScrubFrame
    + " / " + maxLabel;
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
  generatorCanvas.renderTargetPlaceholder({
    frameIndex: frameIndex,
    editedFrames: editedFrames,
    minimumFrame: scrubberMinFrame,
  });
}

function navigateToFrame(index) {
  saveFrameSelections(currentScrubFrame);

  var minFrame = (
    runPhase.mode === "select"
    || runPhase.mode === "select_target"
  ) ? scrubberMinFrame : 0;
  var maxFrame = (
    runPhase.mode === "select_target"
    && generatorRun.originalCaptured()
  ) ? generatorRun.originalTotalFrames() - 1
    : generatorRun.frameCount() - 1;
  index = Math.max(
    minFrame,
    Math.min(index, maxFrame)
  );
  currentScrubFrame = index;
  scrubberSlider.value = String(index);
  updateScrubberLabel();
  renderScrubStepReadout(index);

  restoreFrameSelections(index);

  if (runPhase.mode === "select_target") {
    renderTargetFrame(index);
  } else if (index < generatorRun.frameCount()) {
    generatorCanvas.renderFrame(index);
  } else {
    renderTargetFrame(index);
  }
  generatorReadouts.refreshStop();
  // The token spans were just replaced, so any open popover now
  // points at a detached element.
  hideAltsPopover();
  if (scrubberActive) {
    generatorReadouts.updateProfile();
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
  generatorCanvas.renderFrame(currentScrubFrame);
  updateGuidedUI();
}

// ---- Randomize remasks (Edit Frames) ----

// Frame the randomize row was last seeded for, so the target count is
// re-initialized from the selection only when the frame changes (not
// on every re-render, which would fight the user's slider input).
var randomizeInitFrame = null;

function clampInt(value, low, high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

// Frame-array indices of resolved (non-mask) tokens: the candidates
// that can be remasked. Masked positions are never remaskable.
function resolvedPositions(frameIndex) {
  var tokens = generatorRun.frameTokens(frameIndex);
  var out = [];
  if (!tokens) {
    return out;
  }
  for (var i = 0; i < tokens.length; i++) {
    if (tokens[i] && !tokens[i].m) {
      out.push(i);
    }
  }
  return out;
}

// Sync the randomize row to the current edit frame: total resolved
// count, slider/input bounds, and (on a frame change) seed the target
// N from the frame's existing selection.
function updateRandomizeRow() {
  if (!remaskRandomizeRow) {
    return;
  }
  var total = resolvedPositions(currentScrubFrame).length;
  // Remasking 0 tokens is a no-op, so the target floor is 1 (whenever
  // there is at least one resolved token to pick from).
  var floor = total > 0 ? 1 : 0;
  if (randomizeInitFrame !== currentScrubFrame) {
    randomizeInitFrame = currentScrubFrame;
    var selected = Object.keys(remaskedPositions).length;
    remaskRandomSlider.value = String(
      clampInt(selected, floor, total)
    );
  }
  var target = clampInt(
    parseInt(remaskRandomSlider.value, 10) || floor, floor, total
  );
  remaskRandomTotal.textContent = String(total);
  remaskRandomSlider.min = String(floor);
  remaskRandomSlider.max = String(total);
  remaskRandomSlider.value = String(target);
  remaskRandomCount.min = String(floor);
  remaskRandomCount.max = String(total);
  remaskRandomCount.value = String(target);
  var disabled = total === 0;
  remaskRandomSlider.disabled = disabled;
  remaskRandomCount.disabled = disabled;
  btnRemaskShuffle.disabled = disabled;
}

// Replace the current selection with N random resolved positions on
// the current frame (partial Fisher-Yates), then re-render so they
// show as remasked and Lock In can proceed as usual.
function shuffleRemasks() {
  var candidates = resolvedPositions(currentScrubFrame);
  var total = candidates.length;
  if (total === 0) {
    return;
  }
  var n = clampInt(
    parseInt(remaskRandomSlider.value, 10) || 0, 0, total
  );
  for (var i = 0; i < n; i++) {
    var j = i + Math.floor(
      Math.random() * (total - i)
    );
    var swap = candidates[i];
    candidates[i] = candidates[j];
    candidates[j] = swap;
  }
  remaskedPositions = {};
  for (var k = 0; k < n; k++) {
    remaskedPositions[candidates[k]] = true;
  }
  saveFrameSelections(currentScrubFrame);
  generatorCanvas.renderFrame(currentScrubFrame);
  updateGuidedUI();
}

// Cosmetic press feedback: run the diffusion reveal on the Shuffle
// label with a glow that lingers on the way out (the CSS transition
// handles the lag). Gated on the same effect setting as the status bar.
function playShuffleDiffusion() {
  if (!btnRemaskShuffle || !shuffleLabel) {
    return;
  }
  if (!diffusionEffectActive()) {
    return;
  }
  btnRemaskShuffle.classList.add("is-diffusing");
  denoiseReveal(shuffleLabel, "Shuffle", function () {
    btnRemaskShuffle.classList.remove("is-diffusing");
  });
}

// ---- Guided multi-frame edit mode ----

function resetGuidedMode() {
  runPhasesReset(runPhase);
  hideAltsPopover();
  preEditSnapshot = null;
  pendingResume = null;
  randomizeInitFrame = null;
  guidedEditControls.hidden = true;
  scrubberSlider.disabled = false;
  scrubberSlider.min = "0";
  unlockScrubberNav();
}

// Snapshot the current complete run before an edit session begins.
function captureEditSnapshot() {
  preEditSnapshot = {
    run: generatorRun.captureCheckpoint(),
    remaskEditsLen: remaskEdits.length,
  };
  rewindWorkerRun();
}

// Tell the worker to discard any branch a previous session left it
// holding, so it re-enters this session from the run on screen.
//
// Sent when a session opens rather than when one is abandoned,
// because abandoning has too many doors. Retry and Exit both restore
// the snapshot above and send nothing; so does a run-scoped error;
// and a reload or a closed tab cannot send anything at all, since
// preEditSnapshot lives only in memory and the session snapshot is
// deliberately not written while an edit is in progress. Opening a
// session is the one moment the browser is known to be showing the
// un-edited run, so one message here covers every one of those.
//
// Harmless when there is nothing to undo: rewinding a run that has
// committed no branch restores what the worker already holds.
function rewindWorkerRun() {
  if (!generatorSocket.isReady()) {
    return;
  }
  if (!generatorRun.runToken()) {
    return;
  }
  // Silently: a rewind is housekeeping nobody asked for by name.
  if (runEditBlock()) {
    return;
  }
  generatorSocket.send({
    type: "rewind",
    run_token: generatorRun.runToken(),
  });
}

// Restore the pre-edit run, discarding any partial/uncommitted
// resumes made during the session (used when the user exits).
function restoreEditSnapshot() {
  if (!preEditSnapshot) {
    return;
  }
  generatorRun.restoreCheckpoint(preEditSnapshot.run);
  // Drop any edits committed during this (now-cancelled) session.
  remaskEdits.length = Math.min(
    remaskEdits.length, preEditSnapshot.remaskEditsLen
  );
  preEditSnapshot = null;
}

// Cut the run back to `offset` frames so the branch about to be
// generated appends cleanly at that index. The elapsed value at the
// last kept frame carries forward, because the worker restarts its
// clock for the new segment.
//
// Which arrays get cut is no longer a decision made here. It used to
// be, and leaving one out made the saved timing array longer than the
// frame arrays, which knocked the Timing chart's x axis out of step
// with every other chart.
function truncateRunArraysAt(offset) {
  generatorRun.truncate(offset);
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

// Freeze scrubber navigation and every guided-edit action while a save
// is in flight, so neither save path, Confirm or the standalone Save
// button, leaves interactive controls that could race the snapshot. The subsequent updateGuidedUI() re-derives each
// button's per-phase state once the save settles.
function setSavingControls(saving) {
  var disabled = !!saving;
  scrubberSlider.disabled = disabled;
  btnScrubStart.disabled = disabled;
  btnScrubPrev.disabled = disabled;
  btnScrubNext.disabled = disabled;
  btnScrubEnd.disabled = disabled;
  btnSelectFrame.disabled = disabled;
  btnEditFrames.disabled = disabled;
  // Guided-edit action buttons (visible only mid edit session) freeze
  // too, so the dimmed slider matches the Confirm-checkmark behavior.
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
  // Dim the whole scrubber row and surface a tooltip on hover. The
  // title lives on the (non-disabled) container because native
  // tooltips do not fire on disabled controls.
  if (scrubberControls) {
    scrubberControls.classList.toggle("is-saving", disabled);
    if (disabled) {
      scrubberControls.title = "Saving in progress\u2026";
    } else {
      scrubberControls.removeAttribute("title");
    }
  }
  // Reflect the saving state on the primary button too (so Generate is
  // greyed out during a save, then becomes New Run once finalized).
  updateGenerateButton();
}

// ---- What If: top-k substitution (autoregressive) ----

function supportsSubstitution() {
  var capabilities = generatorModelPanel.capabilities();
  return !!capabilities.supports_substitution;
}

// Arm substitution on the completed run. Mirrors Edit Frames,
// including in writing nothing: the current run becomes the
// "original" in memory, and the branch that replaces it carries the
// original with it when it is saved.
function enterSubstitutionMode() {
  if (btnWhatIf && btnWhatIf.classList.contains("is-locked")) {
    return;
  }
  beginSubstitutionSession();
}

// Frame index and token position are the same choice for a
// left-to-right model, so there is no frame-selection phase: the run
// opens at its final frame and every captured position is clickable.
function beginSubstitutionSession() {
  captureEditSnapshot();
  runPhasesEnter(runPhase, RUN_PHASE_SUBSTITUTE);
  runPhase.substituting = true;
  scrubberMinFrame = 0;
  runPhase.lockedEdits = [];
  runPhase.guidedAction = null;
  clearRemaskedPositions();

  scrubberSlider.min = "0";
  scrubberSlider.max =
    String(generatorRun.frameCount() - 1);
  btnEditFrames.hidden = true;
  if (btnWhatIf) {
    btnWhatIf.hidden = true;
  }
  guidedEditControls.hidden = false;
  generatorCanvas.deactivate();

  navigateToFrame(generatorRun.frameCount() - 1);
  updateGuidedUI();
}

// Commit a substitution: truncate the run at the position, then let
// the worker regenerate from the forced token. Reuses the diffusion
// controller's resume splice path, so handleFrame
// appends the branch onto the truncation unchanged.
// ``typedText`` is the raw string for a token the user typed, or
// null for one the model actually offered. The worker validates the
// two differently on purpose, so the distinction has to survive the
// trip rather than being inferred from the id.
function doSubstitute(position, tokenId, typedText) {
  if (!runPhase.substituting || runPhase.mode !== "substitute") {
    return;
  }
  if (position < 0 || position >= generatorRun.frameCount()) {
    return;
  }
  if (editRequestRefused()) {
    return;
  }
  hideAltsPopover();
  runPhase.substituting = false;

  // Recorded as an ordinary remask edit so the analytics Edited
  // column, the durable diff, and the saved metadata all work with
  // no schema change. For a left-to-right model the edited frame and
  // the edited position are the same index.
  remaskEdits.push({
    frame_index: position,
    token_positions: [position],
  });

  perFrameRemasked = {};
  remaskedPositions = {};

  truncateRunArraysAt(position);
  // Positions from the substituted one onward are about to be
  // resampled, so their captured candidates no longer apply.
  generatorRun.truncateAlternatives(position);
  isResuming = true;

  runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
  updateGuidedUI();

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  // A substitution always resamples to the end of the run.
  generatorChrome.startRunStatus(
    editRunLabel(position, null)
  );

  var request = {
    type: "substitute",
    position: position,
    token_id: tokenId,
    run_token: generatorRun.runToken(),
  };
  if (typedText) {
    request.typed = true;
    request.typed_text = typedText;
  }
  generatorSocket.send(request);
}

// Edit Frames entry point. The current run is the "original": if it
// has not been saved yet, save it now so an unsaved original is never
// lost once the edited run is saved. Editing an original implies you
// want to keep it, so this makes the save implicit.
function enterRemaskMode() {
  // Gated once an edited run has been saved for this generation.
  if (btnEditFrames.classList.contains("is-locked")) {
    return;
  }
  // Opening the editor writes nothing. Selecting a frame and
  // remasking tokens are reversible and entirely local; the run is
  // only destroyed by the resume, and Confirm is itself a save. So
  // nothing here is worth writing to disk on the user's behalf, and
  // this used to do exactly that: a full save fired on merely
  // opening the panel, which on a long run is megabytes, and which
  // raced any navigation that followed it.
  beginEditSession();
}

// Start a fresh edit session on the current run. Shared by Edit
// Frames and by Retry, which restores the pre-edit run first.
function beginEditSession() {
  captureEditSnapshot();
  runPhasesEnter(runPhase, RUN_PHASE_SELECT);
  // Start at frame 1: frame 0 is the fully-masked canvas with nothing
  // to remask, so it is never a useful selection. (Guarded for the
  // degenerate single-frame case.)
  var startFrame = generatorRun.frameCount() > 1 ? 1 : 0;
  scrubberMinFrame = startFrame;
  runPhase.lockedEdits = [];
  runPhase.guidedAction = null;
  clearRemaskedPositions();

  scrubberSlider.min = String(startFrame);
  scrubberSlider.max =
    String(generatorRun.frameCount() - 1);
  btnEditFrames.hidden = true;
  guidedEditControls.hidden = false;
  generatorCanvas.deactivate();

  navigateToFrame(startFrame);
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

// Some models resume by renoising remasked positions rather than
// hard-masking them, so committed neighbours may also shift. Surface
// that difference while editing.
//
// Read from the declared capability rather than from the model id,
// which is what this used to do: the note would have gone missing for
// the next renoising model to arrive under a different id.
function renoiseNote() {
  var capabilities = generatorModelPanel.capabilities();
  if (capabilities.remask_renoises) {
    return " Remasked tokens are renoised, so nearby"
      + " tokens may also change on resume.";
  }
  return "";
}

function updateGuidedUI() {
  // The two blend rows share the scrubber area with the guided
  // controls, so keep them hidden whenever a run is being edited
  // (runPhase.mode !== null); both updates restore the right one on exit
  // once runPhase.mode is null again.
  generatorCanvas.refreshControls();

  // Reset every phase button first so no stale state can survive a
  // transition (including the exit back to runPhase.mode === null). Only
  // the buttons relevant to the current phase are then revealed; the
  // status text sits on the left (flex:1) and the action cluster is
  // right-anchored, so the text never shifts as buttons change.
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
  if (remaskRandomizeRow) {
    remaskRandomizeRow.hidden = true;
  }

  if (runPhase.mode === null) {
    guidedEditControls.hidden = true;
    return;
  }

  guidedEditControls.hidden = false;
  // Confirm owns saving during an edit session. Disable the
  // standalone Save so it can never race or double-fire with it.
  btnSave.disabled = true;

  var count =
    Object.keys(remaskedPositions).length;
  var plural = count !== 1 ? "s" : "";

  switch (runPhase.mode) {
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
        + ": click tokens to remask ("
        + count + " selected)." + renoiseNote();
      btnBackFrame.hidden = false;
      btnLockIn.hidden = false;
      btnLockIn.disabled = count === 0;
      btnClearGuided.hidden = false;
      btnClearGuided.disabled = count === 0;
      if (remaskRandomizeRow) {
        remaskRandomizeRow.hidden = false;
        updateRandomizeRow();
      }
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
        generatorRun.originalCaptured()
          ? generatorRun.originalTotalFrames() - 1
          : generatorRun.frameCount() - 1
      );
      unlockScrubberNav();
      break;

    case "substitute":
      guidedEditStatus.textContent =
        "Hover a token to see what the model nearly"
        + " chose, then click a candidate to"
        + " regenerate from it.";
      btnClearGuided.hidden = true;
      lockScrubberNav();
      break;

    case "generating":
      guidedEditStatus.textContent =
        "Generating\u2026";
      lockScrubberNav();
      break;

    case "review":
      scrubberSlider.disabled = false;
      scrubberSlider.min = "0";
      scrubberSlider.max =
        String(generatorRun.frameCount() - 1);
      unlockScrubberNav();
      // Both actions stay reachable from any frame. Neither reads the
      // scrubber: Confirm saves the whole run and then jumps to the
      // last frame itself, and Retry restores the pre-edit arrays and
      // goes back to the first editable frame. Hiding them mid-review
      // only made scrubbing back look like it had cancelled the edit.
      btnConfirmEdit.hidden = false;
      btnRetryEdit.hidden = false;
      btnContinueEdit.hidden = !reviewCanContinue();
      if (
        currentScrubFrame === generatorRun.frameCount() - 1
      ) {
        guidedEditStatus.textContent =
          reviewEndText(currentScrubFrame);
      } else {
        guidedEditStatus.textContent =
          "Reviewing frame " + currentScrubFrame + " of the "
          + (generatorRun.interrupted()
            ? "stopped edit"
            : "edited run")
          + ". " + reviewChoices();
      }
      break;
  }

  // A save in flight overrides the per-mode state: freeze
  // navigation until it completes.
  if (generatorRun.saving()) {
    lockScrubberNav();
    btnSelectFrame.disabled = true;
  }
}

function selectCurrentFrame() {
  runPhasesEnter(runPhase, RUN_PHASE_EDIT);
  generatorCanvas.renderFrame(currentScrubFrame);
  updateGuidedUI();
}

// Back from choosing tokens to choosing a frame. Nothing on this
// frame was locked in, so its selection goes, as Clear would drop
// it; the session's earlier steps and its forward-only floor stay.
function backToFrameSelection() {
  remaskedPositions = {};
  delete perFrameRemasked[currentScrubFrame];
  runPhasesEnter(runPhase, RUN_PHASE_SELECT);
  navigateToFrame(currentScrubFrame);
}

function lockInEdits() {
  var positions =
    Object.keys(remaskedPositions).map(Number);
  if (positions.length === 0) {
    return;
  }
  runPhase.lockedEdits.push({
    frame_index: currentScrubFrame,
    token_positions: positions.slice(),
  });
  runPhasesEnter(runPhase, RUN_PHASE_CHOICE);
  updateGuidedUI();
}

function doGuidedResume(action) {
  // Guard against a stale click with no locked edits (should be
  // unreachable now that the buttons hide correctly).
  if (runPhase.lockedEdits.length === 0) {
    return;
  }
  if (editRequestRefused()) {
    return;
  }
  runPhase.guidedAction = action;

  var lastEdit =
    runPhase.lockedEdits[runPhase.lockedEdits.length - 1];
  var positions = lastEdit.token_positions;
  var frameIndex = lastEdit.frame_index;

  pendingResume = captureResumeCut(frameIndex);
  remaskEdits.push({
    frame_index: frameIndex,
    token_positions: positions.slice(),
  });

  perFrameRemasked = {};
  remaskedPositions = {};

  truncateRunArraysAt(frameIndex);
  isResuming = true;

  runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
  updateGuidedUI();

  // One source for where the branch stops, so the message on screen
  // and the request on the wire cannot drift apart. Null means run to
  // the end, which is both the "resume to end" action and the
  // fallback when no target frame was captured.
  var resumeTarget = (
    action === "another" && runPhase.targetFrame !== null
  ) ? runPhase.targetFrame : null;

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  generatorChrome.startRunStatus(
    editRunLabel(frameIndex, resumeTarget)
  );

  var message = {
    type: "resume",
    frame_index: frameIndex,
    remask_positions: positions,
    run_token: generatorRun.runToken(),
  };

  if (resumeTarget !== null) {
    message.max_frames = resumeTarget - frameIndex + 1;
  }

  generatorSocket.send(message);
}

// What a resume is about to cut, and where it is being sent from.
// A resume stopped before it sends a frame has changed nothing on
// the worker, which keeps the run it had, so the page puts this
// back and returns there (landBeforeResume) rather than staying cut
// back to the edited frame, where Confirm would save the run that
// way. Only a Stop is answered this way: a connection lost before
// the first frame is still handled by enterInterruptedState, which
// drops the copy. Review sends one too, for Continue.
function captureResumeCut(cutAt) {
  var mode = runPhase.mode;
  var sender = mode === RUN_PHASE_CHOICE
    || mode === RUN_PHASE_SELECT_TARGET
    || mode === RUN_PHASE_REVIEW;
  if (!sender) {
    throw new Error("a resume is sent from choice, target or review");
  }
  var perFrame = {};
  var keys = Object.keys(perFrameRemasked);
  for (var i = 0; i < keys.length; i++) {
    perFrame[keys[i]] = Object.assign({}, perFrameRemasked[keys[i]]);
  }
  return {
    cutAt: cutAt,
    run: generatorRun.captureCheckpoint(),
    remaskEditsLen: remaskEdits.length,
    mode: mode,
    frame: currentScrubFrame,
    minFrame: scrubberMinFrame,
    remasked: Object.assign({}, remaskedPositions),
    perFrame: perFrame,
  };
}

// Whether the resume that just ended sent nothing before its Stop:
// cancelled, with the run still at the length it was cut to.
function resumeStoppedBeforeAFrame(resumed, data) {
  if (resumed === null) {
    return false;
  }
  if (data.cancelled !== true) {
    return false;
  }
  return generatorRun.frameCount() === resumed.cutAt;
}

// Put back what the resume cut and return to where it was sent from:
// the run whole, the locked edit and its selection in place, ready to
// resume again or exit.
function landBeforeResume(saved) {
  generatorRun.restoreCheckpoint(saved.run);
  remaskEdits.length = saved.remaskEditsLen;

  runPhase.guidedAction = null;
  runPhase.targetFrame = null;
  if (saved.mode === RUN_PHASE_CHOICE) {
    runPhasesEnter(runPhase, RUN_PHASE_CHOICE);
  } else if (saved.mode === RUN_PHASE_SELECT_TARGET) {
    runPhasesEnter(runPhase, RUN_PHASE_SELECT_TARGET);
  } else {
    runPhasesEnter(runPhase, RUN_PHASE_REVIEW);
  }
  scrubberMinFrame = saved.minFrame;
  remaskedPositions = saved.remasked;
  perFrameRemasked = saved.perFrame;
  scrubberActive = true;
  setScrubberVisible(true);
  navigateToFrame(saved.frame);
  generatorChrome.setMessage(RESUME_STOPPED_BEFORE_FRAME);
}

function handleGuidedDone() {
  if (runPhase.guidedAction === "another") {
    var target = Math.min(
      runPhase.targetFrame,
      generatorRun.frameCount() - 1
    );

    scrubberActive = true;
    setScrubberVisible(true);
    guidedEditControls.hidden = false;
    btnEditFrames.hidden = true;
    generatorCanvas.deactivate();

    scrubberSlider.min = String(target);
    scrubberSlider.max =
      String(generatorRun.frameCount() - 1);
    scrubberSlider.value = String(target);

    currentScrubFrame = target;
    runPhase.guidedAction = null;
    runPhase.targetFrame = null;
    runPhasesEnter(runPhase, RUN_PHASE_EDIT);
    remaskedPositions = {};
    perFrameRemasked = {};

    updateScrubberLabel();
    generatorCanvas.renderFrame(target);
    updateGuidedUI();
  } else {
    enterReviewMode();
  }
}

// Resume-to-End finished: rather than dropping straight back to the
// plain scrubber, stay in guided editing at the final frame so the
// user must explicitly Confirm (save) or Retry (redo). Navigation
// stays enabled so the result can be inspected; only the final frame
// exposes the Confirm/Retry actions.
function enterReviewMode() {
  runPhase.guidedAction = null;
  runPhase.targetFrame = null;
  remaskedPositions = {};
  perFrameRemasked = {};
  runPhasesEnter(runPhase, RUN_PHASE_REVIEW);
  scrubberActive = true;
  setScrubberVisible(true);
  guidedEditControls.hidden = false;
  btnEditFrames.hidden = true;
  generatorCanvas.deactivate();
  currentScrubFrame = generatorRun.frameCount() - 1;
  scrubberSlider.min = "0";
  scrubberSlider.max =
    String(generatorRun.frameCount() - 1);
  scrubberSlider.value = String(currentScrubFrame);
  scrubberSlider.disabled = false;
  unlockScrubberNav();
  updateScrubberLabel();
  generatorCanvas.renderFrame(currentScrubFrame);
  updateGuidedUI();
}

// Confirm the reviewed edit: trigger a save (as the Save button
// would), then leave guided editing. The save-success handler locks
// Edit Frames so the run cannot accrue a second, conflicting edit.
function confirmGuidedEdit() {
  generatorRun.save();
  resetGuidedMode();
  activateScrubber();
}

// Retry: discard this session's edits and restart editing from the
// beginning. Writes nothing, like every other way of entering an
// edit session. Autoregressive runs re-enter substitution, whose
// session has no frame-selection phase to restart into.
function retryGuidedEdit() {
  if (editRequestRefused()) {
    return;
  }
  var wasSubstitution = supportsSubstitution();
  restoreEditSnapshot();
  resetGuidedMode();
  if (wasSubstitution) {
    beginSubstitutionSession();
  } else {
    beginEditSession();
  }
}

// Whether review offers Continue: only on a branch that stopped, of
// a model whose worker keeps the frames the page received and can
// resume from them. A stopped What If branch is not one, since its
// worker keeps no branch, and a finished branch has nothing left.
function reviewCanContinue() {
  var capabilities = generatorModelPanel.capabilities();
  return generatorRun.interrupted()
    && !!capabilities.supports_resume;
}

// Review's status line at the branch's last frame: where it stopped,
// or that it finished.
function reviewEndText(frame) {
  if (generatorRun.interrupted()) {
    return "Stopped at frame " + frame + ". " + reviewChoices();
  }
  return "Edit complete. " + reviewChoices();
}

// What review offers, in the words of its status line.
function reviewChoices() {
  if (reviewCanContinue()) {
    return "Continue, confirm to save it as it is, or retry"
      + " from the start.";
  }
  if (generatorRun.interrupted()) {
    return "Confirm to save it as it is, or retry from the start.";
  }
  return "Confirm to save, or retry from the start.";
}

// Carry a stopped branch on from its last frame. No edit is
// recorded: the request remasks nothing, flagged as a continue
// (RESUME_CONTINUE in protocol.py), and the worker re-enters that
// frame as it was. It ends in review, as Resume to End does.
function continueGuidedEdit() {
  if (!reviewCanContinue()) {
    return;
  }
  if (editRequestRefused()) {
    return;
  }
  var from = generatorRun.frameCount() - 1;
  pendingResume = captureResumeCut(from);
  truncateRunArraysAt(from);
  isResuming = true;
  runPhase.guidedAction = "end";
  runPhasesEnter(runPhase, RUN_PHASE_GENERATING);
  updateGuidedUI();

  setSaveAvailable(false);
  resetStatus();
  setGenerating(true);
  generatorChrome.startRunStatus(editRunLabel(from, null));

  generatorSocket.send({
    type: "resume",
    frame_index: from,
    remask_positions: [],
    "continue": true,
    run_token: generatorRun.runToken(),
  });
}

// ---- UI state helpers ----

function setGenerating(active) {
  isGenerating = active;
  // Generate stays visible; it just greys out while the model runs
  // (and whenever the model is not ready, params are invalid, or a
  // save is completing) -- centralized in updateGenerateButton.
  updateGenerateButton();
  generatorComposer.setDisabled(active);
  generatorModelPanel.setDisabled(active);

  if (active) {
    deactivateScrubber();
  }
}

// Block glyphs for the optional "diffusion-style text" reveal.
var DENOISE_GLYPHS = "\u2591\u2592\u2593";

// True when the diffusion-text effect should actually animate.
// prefersReducedMotion comes from reduced_motion.js, which every page
// that animates loads ahead of its own script.
function diffusionEffectActive() {
  return !!appSettings.diffusionText && !prefersReducedMotion();
}

// Per-element reveal timer, stored on the element so independent
// targets (status bar, Shuffle label) can animate simultaneously
// without one cancelling the other.
function cancelDenoise(el) {
  if (el && el._denoiseTimer) {
    clearInterval(el._denoiseTimer);
    el._denoiseTimer = null;
  }
}

// Reveal `text` in `el` with a brief scramble-to-resolve ("denoising")
// pass: characters lock in left-to-right while the rest keep flickering
// through block glyphs, then `onDone` runs. Instant when the effect is
// off or the OS prefers reduced motion (an accessibility escape hatch).
// `force` animates regardless of the setting (still honoring reduced
// motion) for the one-time Generate teaser.
function denoiseReveal(el, text, onDone, force) {
  cancelDenoise(el);
  var active = force
    ? !prefersReducedMotion()
    : diffusionEffectActive();
  if (!active || text.length === 0) {
    el.textContent = text;
    if (onDone) {
      onDone();
    }
    return;
  }
  var steps_total = 8;
  var step = 0;
  var render = function () {
    var revealed = Math.floor(
      (step / steps_total) * text.length
    );
    var out = "";
    for (var i = 0; i < text.length; i++) {
      if (i < revealed || text[i] === " ") {
        out += text[i];
      } else {
        out += DENOISE_GLYPHS[
          Math.floor(Math.random() * DENOISE_GLYPHS.length)
        ];
      }
    }
    el.textContent = out;
    step += 1;
    if (step > steps_total) {
      cancelDenoise(el);
      el.textContent = text;
      if (onDone) {
        onDone();
      }
    }
  };
  render();
  el._denoiseTimer = setInterval(render, 45);
}

// Reverse of denoiseReveal: dissolve `el`'s current text into solid
// mask glyphs (0-confidence "░") left-to-right, then run `onDone`.
// Code-point safe so the lock emoji collapses as one glyph. Instant
// when the effect is off or reduced motion is preferred.
function denoiseDissolve(el, onDone) {
  var chars = Array.from(el.textContent);
  cancelDenoise(el);
  if (!diffusionEffectActive() || chars.length === 0) {
    if (onDone) {
      onDone();
    }
    return;
  }
  var steps_total = 8;
  var step = 1;
  var render = function () {
    var masked = Math.ceil(
      (step / steps_total) * chars.length
    );
    var out = "";
    for (var i = 0; i < chars.length; i++) {
      if (chars[i] === " ") {
        out += " ";
      } else if (i < masked) {
        out += generatorCanvas.maskChar();
      } else {
        out += chars[i];
      }
    }
    el.textContent = out;
    step += 1;
    if (step > steps_total) {
      cancelDenoise(el);
      if (onDone) {
        onDone();
      }
    }
  };
  render();
  el._denoiseTimer = setInterval(render, 40);
}

// Names the stretch a resume is about to regenerate. "Resuming" said
// only that something had restarted, which reads as ambiguous next to
// a plain run; the frame range says which part of the output is being
// replaced, and that is what you are waiting to watch change. A null
// target means the branch runs to the end, which is always the case
// for a left-to-right substitution.
function editRunLabel(fromFrame, toFrame) {
  var target = toFrame === null ? "end" : String(toFrame);
  return "Running edit from frame " + fromFrame
    + " to " + target;
}

function setSaveAvailable(available) {
  // Always visible; greyed out when there is nothing to save.
  btnSave.disabled = !(
    available && generatorRun.frameCount() > 0
  );
}

// Clears the footer readouts only, never the stack. doSubstitute and
// doGuidedResume both call this immediately before starting a resume,
// which is a moment a save the user started by hand may still be in
// flight; clearing the chips here would put back the overwriting this
// stack exists to fix.
function resetStatus() {
  generatorChrome.resetStatus(appSettings.tpsMode);
  generatorReadouts.clearMetrics();
}

// ---- Actions ----

// Clear all live-run state (frames, edits, overlays, gates) back to a
// pre-run baseline. Shared by Generate (fresh run) and New Run.
function resetRunState() {
  resetGuidedMode();
  remaskedPositions = {};
  perFrameRemasked = {};
  generatorRun.reset();
  remaskEdits = [];
  generatorCanvas.reset();
  generatorCanvas.deactivate();
  generatorReadouts.reset();
  hideAltsPopover();
  isResuming = false;
  pendingResume = null;
  updateEditFramesLock();
  updateGenerateButton();
  setSaveAvailable(false);
}

// "New Run": reset to a clean slate for a new prompt once a run is
// finalized (Generate has become "New Run"). Clears the canvas and the
// prompt box (revealing its placeholder), but keeps prompt history.
function startNewRun() {
  resetRunState();
  deactivateScrubber();
  generatorComposer.clear();
  if (thinkingPanel) {
    thinkingPanel.hidden = true;
  }
  generatorRun.clearSession();
  resetStatus();
  setGenerating(false);
  // Return to the pre-generation resting state.
  generatorChrome.showOutputPlaceholder(
    generatorModelPanel.activeDisplayName()
  );
}

// Ask the worker to stop the run it is on.
//
// The reply is the run's ordinary terminal frame carrying
// "cancelled", not a separate acknowledgement, so there is exactly
// one way a run ends however it ended. Nothing is torn down here:
// the worker still owns the frames in flight, and tidying up before
// it has answered would discard tokens that are still arriving.
function requestCancel() {
  if (!generatorSocket.isReady()) {
    return;
  }
  if (!isGenerating) {
    return;
  }
  generatorChrome.setMessage("Stopping...");
  generatorSocket.send({ type: "cancel" });
}

function startGeneration() {
  if (!generatorSocket.isReady()) {
    return;
  }
  if (isGenerating) {
    return;
  }
  if (!generatorModelPanel.validation().valid) {
    return;
  }

  var prompt = generatorComposer.trimmedValue();
  if (!prompt) {
    generatorChrome.setMessage("Prompt is empty.");
    return;
  }

  // The first fresh run retires the Generate teaser: from now on the
  // idle diffusion cycle follows the setting.
  markGenerateTeased();

  // A fresh run abandons any in-progress edit session and clears the
  // previous run's state. Record the prompt in history first.
  generatorComposer.prepareGeneration(prompt);
  resetRunState();
  var params = generatorModelPanel.parameterValues();
  generatorRun.begin(prompt, params);

  generatorCanvas.clearOutput();
  if (thinkingPanel) {
    thinkingPanel.hidden = true;
  }
  generatorRun.clearSession();
  resetStatus();
  setGenerating(true);
  generatorChrome.startRunStatus("Running");

  var payload = Object.assign({}, params);
  payload.type = "generate";
  payload.prompt = prompt;
  payload.experimental = generatorModelPanel.experimental();
  generatorSocket.send(payload);
}

// Run serialization and save request ownership live in
// generator_run.js. The page only supplies presentation callbacks.
function saveRun() {
  return generatorRun.save();
}

function invalidateGeneratorCanvas() {
  if (generatorCanvas) {
    generatorCanvas.invalidate();
  }
}

function generatorCanvasReadModel() {
  var capabilities = generatorModelPanel.capabilities();
  return {
    capabilities: capabilities,
    maskChar: capabilities.unresolved_char || "\u2591",
  };
}

function generatorCanvasReadSettings() {
  return appSettings;
}

function generatorCanvasReadEdit() {
  return {
    remaskEdits: remaskEdits,
    remaskedPositions: remaskedPositions,
    mode: runPhase.mode,
    substituting: runPhase.substituting,
    generating: isGenerating,
  };
}

function generatorCanvasWriteHighlight(value) {
  appSettings.highlightTokens = value;
  overlaysWriteHighlightTokens(value);
}

function generatorCanvasOutputReset() {
  generatorReadouts.outputReset();
}

function generatorCanvasRendered() {
  generatorReadouts.rendered();
}

function generatorCanvasOverlayChanged() {
  hideAltsPopover();
}

function generatorCanvasLayerChanged(change) {
  generatorReadouts.layerChanged(change);
}

function generatorReadoutsReadModel() {
  var tokenizer = generatorModelPanel.activeTokenizer();
  return {
    capabilities: generatorModelPanel.capabilities(),
    parameterDefaults:
      generatorModelPanel.parameterDefaults(),
    vocabSize: tokenizer.model_vocab_size || null,
  };
}

function generatorReadoutsReadSettings() {
  return {
    remaskedPositions: remaskedPositions,
    segmentStarts: remaskEdits.map(function (edit) {
      return edit.frame_index;
    }),
  };
}

function generatorReadoutsReadScrubber() {
  return {
    active: scrubberActive,
    frame: currentScrubFrame,
    selectingTarget: runPhase.mode === "select_target",
  };
}

function generatorRunReadModel() {
  return {
    id: generatorModelPanel.activeModelId(),
    device: generatorModelPanel.activeDevice(),
    params: generatorModelPanel.parameterValues(),
  };
}

function generatorRunReadComposer() {
  return {
    draft: generatorComposer.value(),
    prompt: generatorComposer.trimmedValue(),
  };
}

function generatorRunRestoreComposer(state) {
  generatorComposer.restore(state.draft);
}

function generatorRunReadChrome() {
  return {
    thinking:
      thinkingPanel && !thinkingPanel.hidden
        ? thinkingContent.textContent
        : "",
    status: generatorChrome.readStatus(),
  };
}

function generatorRunRestoreChrome(state) {
  if (thinkingPanel && thinkingContent) {
    if (state.thinking) {
      thinkingContent.textContent = state.thinking;
      thinkingPanel.hidden = false;
    } else {
      thinkingContent.textContent = "";
      thinkingPanel.hidden = true;
    }
  }
  generatorChrome.restoreStatus({
    step: state.status.step,
    elapsed: state.status.elapsed,
    message: state.status.message,
    rate: currentTokensPerSecond(),
    tpsMode: appSettings.tpsMode,
  });
}

function generatorRunReadEditArtifacts() {
  return { remaskEdits: remaskEdits };
}

function generatorRunRestoreEditArtifacts(state) {
  remaskEdits = state.remaskEdits;
}

function generatorRunSessionRestored() {
  updateGenerateButton();
  setSaveAvailable(!generatorRun.saved());
  activateScrubber();
}

function generatorRunSaveStart(info) {
  btnSave.disabled = true;
  setSavingControls(true);
  updateEditFramesLock();
  if (saveCheckTimer !== null) {
    clearTimeout(saveCheckTimer);
    saveCheckTimer = null;
  }
  btnSave.classList.remove("is-saved");
  btnSave.classList.add("is-saving");
  return generatorChrome.pushStatus(
    "Saving " + info.label + " run"
  );
}

function generatorRunSaveSettled() {
  btnSave.classList.remove("is-saving");
  setSavingControls(false);
  updateGuidedUI();
  updateEditFramesLock();
}

function generatorRunSaveSuccess(info) {
  generatorRunSaveSettled();
  btnSave.classList.add("is-saved");
  saveCheckTimer = setTimeout(function () {
    btnSave.classList.remove("is-saved");
    saveCheckTimer = null;
  }, 500);
  updateEditFramesLock();
  updateGenerateButton();
  generatorChrome.showAnalyticsCue(info.runId || "");
  generatorChrome.retireStatus(info.status);
  generatorChrome.setMessage(
    "Saved " + info.label + " run to " + info.result.path,
    { color: "var(--accent)" }
  );
}

function generatorRunSaveFailure(info) {
  btnSave.classList.remove("is-saving", "is-saved");
  generatorRunSaveSettled();
  btnSave.disabled = false;
  generatorChrome.retireStatus(info.status);
  generatorChrome.setMessage(
    "Save failed: " + info.message,
    { color: "var(--danger)" }
  );
}

function generatorRunSaveRefused(message) {
  generatorChrome.setMessage(
    message, { color: "var(--danger)" }
  );
}

// ---- Event listeners ----

btnGenerate.addEventListener(
  "click",
  function () {
    // Same order as currentGenerateLabel, so what the button says
    // and what it does cannot drift apart.
    if (isGenerating) {
      requestCancel();
    } else if (generatorRun.editedSaved()) {
      startNewRun();
    } else {
      startGeneration();
    }
  }
);
btnSave.addEventListener("click", saveRun);

generatorComposer.wire();
generatorModelPanel.wire();
generatorChrome.wire();
generatorCanvas.wire();
generatorReadouts.wire();

// Scrubber event listeners.
//
// Dragging goes through navigateToFrame, the same path the arrow
// buttons and the keyboard take. It used to have its own copy of that
// body, which drifted: the copy never hid the candidate popover and
// never repainted the entropy profile, so a drag left the profile
// showing the frame the arrows had last selected.
scrubberSlider.addEventListener(
  "input",
  function () {
    navigateToFrame(parseInt(scrubberSlider.value, 10));
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
      runPhase.mode === "select_target"
      && generatorRun.originalCaptured()
    ) ? generatorRun.originalTotalFrames() - 1
      : generatorRun.frameCount() - 1;
    navigateToFrame(endFrame);
  }
);

// Guided edit mode event listeners.
btnEditFrames.addEventListener(
  "click", enterRemaskMode
);

if (btnWhatIf) {
  btnWhatIf.addEventListener(
    "click", enterSubstitutionMode
  );
}

btnSelectFrame.addEventListener(
  "click", selectCurrentFrame
);

btnBackFrame.addEventListener(
  "click", backToFrameSelection
);

btnLockIn.addEventListener("click", function () {
  if (!diffusionEffectActive()) {
    lockInEdits();
    return;
  }
  // Dissolve the label (letters + lock emoji) into 0-confidence mask
  // glyphs, then commit (which hides the button). Restore the label
  // afterward so it reads correctly the next time it appears.
  var label = btnLockIn.textContent;
  denoiseDissolve(btnLockIn, function () {
    lockInEdits();
    btnLockIn.textContent = label;
  });
});

btnClearGuided.addEventListener(
  "click",
  function () {
    remaskedPositions = {};
    delete perFrameRemasked[currentScrubFrame];
    generatorCanvas.renderFrame(currentScrubFrame);
    updateGuidedUI();
  }
);

// Randomize-remask controls: the slider and number input mirror one
// target count; Shuffle applies it. They only set the target, so they
// never render until Shuffle is pressed.
if (remaskRandomSlider) {
  remaskRandomSlider.addEventListener("input", function () {
    remaskRandomCount.value = remaskRandomSlider.value;
  });
}
if (remaskRandomCount) {
  remaskRandomCount.addEventListener("input", function () {
    var total = resolvedPositions(currentScrubFrame).length;
    var floor = total > 0 ? 1 : 0;
    var n = clampInt(
      parseInt(remaskRandomCount.value, 10) || floor, floor, total
    );
    remaskRandomCount.value = String(n);
    remaskRandomSlider.value = String(n);
  });
}
if (btnRemaskShuffle) {
  btnRemaskShuffle.addEventListener("click", function () {
    shuffleRemasks();
    playShuffleDiffusion();
  });
}

btnEditAnother.addEventListener(
  "click",
  function () {
    if (runPhase.lockedEdits.length === 0) {
      return;
    }
    var lastEdit = runPhase.lockedEdits[
      runPhase.lockedEdits.length - 1
    ];
    scrubberMinFrame =
      lastEdit.frame_index + 1;
    runPhasesEnter(runPhase, RUN_PHASE_SELECT_TARGET);
    var maxFrame = generatorRun.originalCaptured()
      ? generatorRun.originalTotalFrames() - 1
      : generatorRun.frameCount() - 1;
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
    runPhase.targetFrame = currentScrubFrame;
    doGuidedResume("another");
  }
);

btnResumeEnd.addEventListener(
  "click",
  function () {
    doGuidedResume("end");
  }
);

btnConfirmEdit.addEventListener(
  "click", confirmGuidedEdit
);

btnRetryEdit.addEventListener(
  "click", retryGuidedEdit
);

btnContinueEdit.addEventListener(
  "click", continueGuidedEdit
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
    if (runPhase.mode !== "edit") {
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

// Token hover, delegated like the click handler above. Drives three
// things: the metrics strip above the canvas, the entropy profile's
// glowing column, which follows every token, and the candidate
// popover, which is suppressed during guided remask editing so it
// never covers the tokens being selected.
outputArea.addEventListener(
  "mouseover",
  function (e) {
    var target = e.target;
    var pos = hoveredTokenPosition(target);
    // The canvas's own padding and the leading between lines are not
    // tokens, and the trip from a token up into the popover crosses
    // them. Reporting that as "nothing hovered" wiped the readout
    // mid-journey, so the last token stands until the pointer leaves
    // the canvas, which the mouseleave below handles. Analytics has
    // always behaved this way; this brought the generator in line.
    if (pos === null) {
      return;
    }
    generatorReadouts.setTokenHover(pos, target);
    if (!scrubberActive || !altsPopover) {
      return;
    }
    if (runPhasesEditing(runPhase) && !runPhase.substituting) {
      return;
    }
    if (pos === altsPopoverPos) {
      return;
    }
    // A live draft owns the popover. Retargeting it to whatever the
    // pointer wandered onto would throw the draft away and move the
    // field out from under the user mid-word.
    if (altsPopoverPinned()) {
      return;
    }
    showAltsPopover(pos, target);
  }
);

// The token position an event target represents, or null when the
// pointer is over the output area's padding rather than a token.
function hoveredTokenPosition(target) {
  if (!target.classList || !target.classList.contains("token-span")) {
    return null;
  }
  var raw = target.getAttribute("data-pos");
  if (raw === null) {
    return null;
  }
  return parseInt(raw, 10);
}

outputArea.addEventListener(
  "mouseleave",
  function () {
    // Keep the popover open, and its position glowing, while the
    // pointer is inside the popover itself: it sits above the token,
    // so reaching a candidate to click means leaving the output area.
    if (altsPopover && altsPopover.matches(":hover")) {
      return;
    }
    if (altsPopoverPinned()) {
      return;
    }
    generatorReadouts.clearTokenHover();
    hideAltsPopover();
  }
);

if (altsPopover) {
  altsPopover.addEventListener("mouseleave", function () {
    if (altsPopoverPinned()) {
      return;
    }
    generatorReadouts.clearTokenHover();
    hideAltsPopover();
  });
  // Picking a candidate commits the substitution. Only armed in What
  // If mode; the popover is read-only otherwise.
  altsPopover.addEventListener("click", function (e) {
    if (!runPhase.substituting || altsPopoverPos === null) {
      return;
    }
    // The Original page shows the pre-edit run's candidates. There is
    // nothing to substitute into on that side, and the worker only
    // holds the live run's state anyway.
    if (altsPopoverPage === "original") {
      return;
    }
    var row = e.target.closest(".alt-row");
    if (!row) {
      return;
    }
    var raw = row.getAttribute("data-alt-id");
    if (raw === null) {
      return;
    }
    // The appended row is the token already in this position, so
    // forcing it would spend a full re-generation arriving back
    // where it started.
    if (row.classList.contains("alt-row-outside")) {
      return;
    }
    // A solidified typed row looks and clicks like a candidate, but
    // the worker has to be told which it is: the captured path
    // rejects anything the model did not consider, and that
    // rejection is deliberate (see _validate_substitute).
    var typed = row.getAttribute("data-typed") === "1"
      ? typedEntryDraft : null;
    doSubstitute(altsPopoverPos, parseInt(raw, 10), typed);
  });
}

// Escape and a click outside are the two ways out of a typed entry
// that do not involve reaching for the cancel button. Both are plain
// cancels: the draft goes, the candidates stay.
document.addEventListener("keydown", function (e) {
  if (e.key !== "Escape" || !altsPopoverPinned()) {
    return;
  }
  e.preventDefault();
  cancelTypedEntry();
});

// pointerdown rather than click, so the cancel lands before the
// press can do anything else with the element underneath.
document.addEventListener("pointerdown", function (e) {
  if (!altsPopoverPinned() || !altsPopover) {
    return;
  }
  if (altsPopover.contains(e.target)) {
    return;
  }
  cancelTypedEntry();
});

// A scroll moves the anchoring token out from under a fixed popover.
// Not while pinned: a stray wheel click should not cost a draft, and
// re-anchoring instead would slide the text field away from the
// pointer. The box stays where it is, which is the least surprising
// of the three, and the anchor is only decorative once you are
// typing into it.
window.addEventListener(
  "scroll",
  function () {
    if (altsPopoverPos !== null && !altsPopoverPinned()) {
      hideAltsPopover();
    }
  },
  true
);

window.addEventListener("resize", function () {
  if (!altsPopoverPinned()) {
    hideAltsPopover();
  }
  if (scrubberActive) {
    generatorReadouts.updateProfile();
  }
});

// Keyboard shortcuts for scrubber navigation.
document.addEventListener(
  "keydown",
  function (e) {
    if (
      !scrubberActive
      || isGenerating
      || generatorRun.saving()
    ) {
      return;
    }
    if (
      runPhase.mode === "edit"
      || runPhase.mode === "choice"
      || runPhase.mode === "generating"
      || runPhase.mode === "substitute"
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
        runPhase.mode === "select_target"
        && generatorRun.originalCaptured()
      ) ? generatorRun.originalTotalFrames() - 1
        : generatorRun.frameCount() - 1;
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
  modalAbout, modalHelp,
];

// ---- Help tabs ----
//
// Mirrors the Settings page's rail rather than inventing a second
// idiom: same data attributes, same is-active class, same shape of
// toggle. Duplicated here rather than extracted, because it is twenty
// lines with two callers and a shared module would cost a third file
// plus a load-order change on four pages.
//
// One thing this does that Settings does not: it sets aria-selected.
// Settings declares role="tab" and then marks the active one with a
// class alone, which is invisible to a screen reader.
var helpTabs =
  document.querySelectorAll(".help-tab");
var helpPanels =
  document.querySelectorAll(".help-panel");

function selectHelpTab(name) {
  for (var i = 0; i < helpTabs.length; i++) {
    var active = helpTabs[i].getAttribute("data-help-tab") === name;
    helpTabs[i].classList.toggle("is-active", active);
    helpTabs[i].setAttribute("aria-selected", active ? "true" : "false");
  }
  for (var j = 0; j < helpPanels.length; j++) {
    helpPanels[j].hidden =
      helpPanels[j].getAttribute("data-help-panel") !== name;
  }
}

function wireHelpTabs() {
  for (var i = 0; i < helpTabs.length; i++) {
    (function (tab) {
      tab.addEventListener("click", function () {
        selectHelpTab(tab.getAttribute("data-help-tab"));
        // Back to the top of the new panel. The body is what scrolls,
        // so without this a reader who was deep in one panel lands
        // mid-way down the next one with no idea why.
        var body = tab.closest(".help-layout");
        if (body) {
          var pane = body.querySelector(".help-body");
          if (pane) {
            pane.scrollTop = 0;
          }
        }
      });
    })(helpTabs[i]);
  }
}

wireHelpTabs();

// Raising the loading curtain has to clear the modals first. They are
// native dialogs now, so an open one is in the top layer, which sits
// above every z-index including this overlay's 100. That is reachable
// rather than theoretical: another window swapping the model raises
// the curtain with About open, and the About box would float over it.
// A swap invalidates the page underneath anyway, so the dialog goes.
function raiseLoadingOverlay() {
  for (var mi = 0; mi < allModals.length; mi++) {
    closeModal(allModals[mi]);
  }
  generatorComposer.closeImport();
  generatorChrome.showLoading();
}

function openModal(modal) {
  // showModal, not show: it is the modal form that traps focus, makes
  // the rest of the document inert, and answers Escape. Guarded
  // because opening an already-open dialog throws.
  if (!modal.open) {
    modal.showModal();
  }
}

function closeModal(modal) {
  if (modal.open) {
    modal.close();
  }
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

// The dialog fills the viewport and centres .modal-box inside it, so
// a click landing on the dialog itself is a click beside the box.
// The ::backdrop cannot be hit directly, which is why this tests the
// element rather than the pseudo.
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

// Escape is the dialog's own now. The hand-rolled listener that used
// to do this closed every open modal at once and, being on the
// document, fired for any Escape anywhere; the native one closes the
// topmost dialog and nothing else.

// ---- Session persistence (survives Analytics navigation) ----

// Shared with the menu, which clears the same snapshot when it
// activates a model (see persistClearLastRun for why both pages do).
var SESSION_KEY = PERSIST_LAST_RUN_KEY;

function saveSessionState() {
  return generatorRun.saveSession();
}

function clearSessionState() {
  generatorRun.clearSession();
}

function restoreSessionState() {
  return generatorRun.restoreSession();
}

// ---- Boot ----

// The model snapshot the server inlined when it served this page, or
// null when there is none. It is shaped exactly like the `/api/models`
// body, so both paths below hand it to the same code.
function bootModelInfo() {
  var boot = window.__BOOT__;
  var models = boot ? boot.models : null;
  // The typeof is the load-bearing half. A string or a number here
  // would otherwise be handed to `applyModelInfo`, which reads an
  // empty list out of it and draws a page with no models on it,
  // having already decided not to fetch the real ones.
  if (!models || typeof models !== "object") {
    return null;
  }
  return models;
}

// Everything that depends on which model is resident. Read in one
// place because the order matters: the panel is built from the active
// model, the saved run is restored over the panel, and the glow and
// mask character are tuned per model class.
function applyModelInfo(info) {
  generatorModelPanel.configure(info);
  var capabilities = generatorModelPanel.capabilities();
  generatorComposer.configure({
    modelId: generatorModelPanel.activeModelId(),
    inputMode: capabilities.input_mode,
    contextLength: generatorModelPanel.activeContext(),
  });
  // Needs the active model, since the glow is tuned per model
  // class. Outside the guard above because it falls back to the
  // diffusion pair, which is the right reading when the active
  // model could not be identified at all.
  generatorCanvas.applyModel();
  // Same reason: whether the entropy row is reserved or absent
  // depends on the model, and the markup starts it absent.
  generatorReadouts.applyModel();
}

function finishBoot() {
  var restored = false;
  try {
    restored = restoreSessionState();
  } catch (_e) {
    restored = false;
  }
  if (!restored) {
    generatorChrome.showOutputPlaceholder(
      generatorModelPanel.activeDisplayName()
    );
  }
  generatorSocket.connect();
}

// Fill in the fields a GPU probe has to answer for. They feed the
// hover popover on a model row, which is inside a dropdown that is
// closed at first paint, so paying for them before drawing anything
// would be putting an nvidia-smi call in front of the whole page.
function refreshModelVram() {
  fetchModels()
    .then(function (info) {
      generatorModelPanel.refresh(info);
    })
    .catch(function () {
      // The rows already drew without headroom, which is the same
      // reading `buildOptionInfo` gives an unknown value.
    });
}

function boot() {
  loadSettings();
  generatorComposer.boot();
  generatorChrome.boot();
  generatorCanvas.applySettings();
  generatorReadouts.boot();
  var inlined = bootModelInfo();
  if (inlined !== null) {
    applyModelInfo(inlined);
    finishBoot();
    refreshModelVram();
    return;
  }
  fetchModels()
    .then(function (info) {
      applyModelInfo(info);
      finishBoot();
    })
    .catch(function () {
      generatorChrome.showOutputPlaceholder("");
      generatorSocket.connect();
    });
}

// Durable UI state has to be in localStorage before boot's synchronous
// reads, which is why boot is a callback rather than a next statement.
// The server inlines both that state and the model snapshot, so the
// common path runs straight through with no fetch in front of it;
// persistHydrate still always runs its callback either way.
persistHydrate(boot);
