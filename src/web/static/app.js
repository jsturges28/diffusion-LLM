// LLM Visualizer: client-side logic.

"use strict";

var CONVERSATION_OPERATION_BYTES = 16;

// ---- DOM refs ----

var btnGenerate =
  document.getElementById("btn-generate");
var btnGenerateLabel =
  document.getElementById("btn-generate-label");
var btnSave =
  document.getElementById("btn-save");
var btnNewConversation =
  document.getElementById("btn-new-conversation");
var conversationState = conversationStateCreate();
var conversationBusy = false;
var conversationActions = conversationActionsCreate({
  readState: function () {
    return conversationState;
  },
  readConfiguration: function () {
    return generatorModelPanel.conversationConfiguration();
  },
  readBlockReason: conversationActionBlockReason,
  requestRender: function () {
    renderConversation(null);
  },
  onStateChanged: conversationActionStateChanged,
  selectBranch: selectConversationBranch,
  editUser: editConversationUserFork,
  deleteUser: deleteConversationFromPathFork,
  retryAssistant: retryConversationAssistantFork,
  createOperationId: createConversationOperationId,
});
var conversationView = conversationViewCreate({
  onLoadOlder: loadOlderConversation,
  decorateTurn: conversationActions.decorateTurn,
  decorateActive: conversationActions.decorateActive,
  deletionMarker: conversationActions.deletionMarker,
});
var conversationShell = conversationShellCreate();
var conversationClient = conversationClientCreate({
  request: function (url, init) {
    return fetch(url, init);
  },
  readState: function () {
    return conversationState;
  },
  applyAction: applyConversationAction,
  onConflict: conversationConflict,
});
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
var generatorModals = generatorModalsCreate({
  initialHelpTab: "start",
});
var generatorEdit = null;
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
  onSaveSettled: generatorRunSaveSettled,
  readConversation: currentConversationIdentity,
  storage: sessionStorage,
  sessionKey: PERSIST_LAST_RUN_KEY,
});
var generatorCandidates = null;
var generatorCanvas = generatorCanvasCreate({
  run: generatorRun,
  readModel: generatorCanvasReadModel,
  readSettings: generatorCanvasReadSettings,
  readEdit: generatorCanvasReadEdit,
  readReducedMotion: prefersReducedMotion,
  writeHighlight: generatorCanvasWriteHighlight,
  startCandidates: function (layers, mask) {
    generatorCandidates.startFlicker(layers, mask);
  },
  stopCandidates: function () {
    generatorCandidates.stopFlicker();
  },
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
generatorCandidates = generatorCandidatesCreate({
  run: generatorRun,
  canvas: generatorCanvas,
  readouts: generatorReadouts,
  readState: generatorCandidatesReadState,
  requestTokenize: generatorCandidatesRequestTokenize,
  requestProbe: generatorCandidatesRequestProbe,
  requestSubstitute: generatorCandidatesRequestSubstitute,
});
generatorEdit = generatorEditCreate({
  run: generatorRun,
  canvas: generatorCanvas,
  readouts: generatorReadouts,
  candidates: generatorCandidates,
  readCapabilities: function () {
    return generatorModelPanel.capabilities();
  },
  readGenerating: function () {
    return isGenerating;
  },
  readDiffusionEffect: diffusionEffectActive,
  revealText: denoiseReveal,
  dissolveText: denoiseDissolve,
  renderFrameReadout: generatorEditRenderFrameReadout,
  setStatus: function (message) {
    generatorChrome.setMessage(message);
  },
  setGenerating: setGenerating,
  setSaveAvailable: setSaveAvailable,
  resetStatus: resetStatus,
  startRunStatus: generatorChrome.startRunStatus,
  primaryStateChanged: updateGenerateButton,
  requestSave: function () {
    return saveRun({ editConfirmation: true });
  },
  requestCommit: confirmConversationEdit,
  requestRewind: generatorEditRequestRewind,
  requestResume: generatorEditRequestResume,
  requestSubstitute: generatorEditRequestSubstitute,
  canEditConversation: activeRunCanEdit,
});
var generatorSocket = generatorSocketCreate({
  onOpen: generatorSocketOpened,
  onClose: generatorSocketClosed,
  onMessage: handleMessage,
  onMalformed: generatorSocketMalformed,
  onFatal: generatorSocketFatal,
});
var generatorWatermark = generatorWatermarkCreate({
  sendRequest: function (payload) {
    return generatorSocket.send(payload);
  },
  readParameters: function () {
    return generatorModelPanel.parameterValues();
  },
});

// ---- State ----

var isGenerating = false;
var saveCheckTimer = null;
var saveAvailable = false;
var modelReady = false;
var conversationCompletion = Promise.resolve(true);
var conversationTransitionBusy = false;

// Every controller keeps its mutable state in its factory closure.
// This composition root retains only page-wide generation and boot
// lifecycle flags.

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
      generatorCanvas.clearOutputHeight();
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
  stopLoadProgressPoll();
  if (switchWatch) {
    switchWatch.stop();
    switchWatch = null;
  }
  syncModelPanelDisabled();
  generatorModelPanel.refreshSelector();
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
      generatorCandidates.handleTokenizeResult(data);
      break;
    case "probe_result":
      generatorCandidates.handleProbeResult(data);
      break;
    case "count_prompt_result":
      generatorComposer.handleCountResult(data);
      break;
    case "detect_watermark_result":
      generatorWatermark.handleResult(data);
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
  generatorCanvas.clearOutputHeight();
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
// The edit controller closes an open session unless it holds a
// branch the page can still save. Confirm needs no worker.
function adoptResidentWorker(worker) {
  if (typeof worker !== "string" || worker === "") {
    return;
  }
  var message = generatorEdit.adoptResidentWorker(worker);
  if (message) {
    generatorChrome.setMessage(message);
  }
}

// How long to let a rescue save finish before reloading anyway. The
// page is describing a worker that no longer exists, so it cannot be
// left here indefinitely on the chance that a request completes.
var RESCUE_SAVE_TIMEOUT_MS = 8000;
var RESCUE_NOTICE_KEY = "diffusion_rescue_notice";

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
  if (generatorEdit.editing()) {
    generatorEdit.exit();
  }
  var shouldSave = !generatorRun.saved()
    && generatorRun.frameCount() > 0
    && Boolean(generatorRun.finalText());
  if (shouldSave) {
    generatorChrome.startRunStatus(
      "Saving run before reloading"
    );
  }
  var save = shouldSave
    ? saveRun()
    : generatorRun.flushSave();
  var work = Promise.all([
    conversationCompletion,
    generatorEdit.flushConfirmation(),
    save,
  ]).then(function (results) {
    return conversationClient.flush().then(function () {
      return (
        results[0] !== false
        && results[1] !== false
        && results[2] !== false
      );
    });
  }).then(function (success) {
    return { finished: true, success: success };
  }, function (error) {
    return {
      finished: true,
      success: false,
      error: error,
    };
  });
  var timeoutId = null;
  var timeout = new Promise(function (resolve) {
    timeoutId = setTimeout(function () {
      resolve({ finished: false, success: false });
    }, RESCUE_SAVE_TIMEOUT_MS);
  });
  Promise.race([work, timeout]).then(function (outcome) {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    if (!outcome.success) {
      var reason = outcome.finished
        ? "conversation bookkeeping failed"
        : "conversation bookkeeping timed out";
      persistRescueNotice(
        outcome.finished
          ? "The run or its conversation link could not be fully"
            + " saved before reload. Check Analytics and the"
            + " transcript."
          : "Reload happened before run and conversation saving"
            + " finished. Check Analytics and the transcript."
      );
      console.warn(
        "Reloading after rescue save, but " + reason + ".",
        outcome.error || ""
      );
    }
    location.reload();
  });
}

function persistRescueNotice(message) {
  try {
    sessionStorage.setItem(RESCUE_NOTICE_KEY, message);
  } catch (_error) {
    // The console warning remains if storage is unavailable.
  }
}

function showRescueNotice() {
  var message = "";
  try {
    message = sessionStorage.getItem(RESCUE_NOTICE_KEY) || "";
    sessionStorage.removeItem(RESCUE_NOTICE_KEY);
  } catch (_error) {
    return;
  }
  if (message) {
    generatorChrome.setMessage(
      message,
      { color: "var(--danger)", clearColorAfterMs: 8000 }
    );
  }
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
    generatorReadouts.refreshWatermark();
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
  if (!generatorEdit.resuming()) {
    generatorRun.setTotalSteps(frameSteps);
  }
  generatorChrome.setStep(
    stepReadout(
      data.index,
      data.canvas_index,
      frameSteps,
      generatorEdit.resuming() ? "Resuming " : "Step "
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
function generatorEditRenderFrameReadout(state) {
  var total = state.totalSteps;
  if (
    total === null
    && generatorRun.frameCanvasSeries().length === 0
  ) {
    return;
  }
  generatorChrome.setStep(
    stepReadout(
      state.frame,
      state.canvasIndex,
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
  var editing = generatorEdit.editing();
  setGenerating(false);
  generatorEdit.interruptStream();
  generatorChrome.endRunStatus();
  var hasFrames = generatorRun.interruptConnection();
  if (!editing) {
    queueConversationAssistant({
      text: generatorRun.finalText() || "",
      partial: true,
      status: "disconnected",
    });
  }
  generatorEdit.refreshLocks();
  generatorChrome.setMessage(
    "Stopped: lost the connection mid-run."
  );
  // The frames already on screen are real and worth keeping, so
  // the scrubber and Save stay available. What the run cannot do
  // is claim it finished.
  if (hasFrames) {
    setSaveAvailable(true);
    generatorEdit.activate();
    // Kept the way a finished run is, so leaving for Analytics
    // before saving does not lose it (skip while mid guided-edit).
    if (generatorEdit.shouldPersistRun()) {
      generatorRun.saveSession();
    }
  }
  generatorCanvas.finishRunSegment();
}

function handleDone(data) {
  var editing = generatorEdit.editing();
  setGenerating(false);
  generatorChrome.endRunStatus();
  // A resume that sent nothing changed nothing, on either side. The
  // rest of this is skipped on purpose: the thinking panel is what
  // Save reads, and this frame's empty thinking would clear it.
  if (generatorEdit.finishStream(data)) {
    generatorCanvas.finishRunSegment();
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
  if (!editing) {
    queueConversationAssistant({
      text: generatorRun.finalText() || "",
      partial: completed.interrupted,
      status: completed.interrupted ? "cancelled" : "completed",
    });
  }
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

  generatorEdit.completeStream();
  generatorCanvas.finishRunSegment();

  // Persist the completed run so it survives navigating to
  // Analytics and back (skip while mid guided-edit).
  if (generatorEdit.shouldPersistRun()) {
    generatorRun.saveSession();
  }
}

function handleError(data) {
  if (generatorWatermark.handleError(data)) {
    return;
  }
  // How far this reaches is the worker's to say (see wire_errors.js).
  // Everything below used to run for every error, which meant a probe
  // refused because a generation was busy closed What If and threw
  // away the edit being composed.
  var routed = wireErrorsRoute(data);
  if (routed.unwindsRun) {
    setGenerating(false);
    generatorChrome.endRunStatus();
    generatorEdit.unwindRunError();
    generatorCanvas.finishRunSegment();
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
    generatorEdit.activate();
  }
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
  var messages = conversationStateMessages(
    conversationState, payload.text
  );
  if (messages === null || conversationState.conversation === null) {
    generatorSocket.send(payload);
    return;
  }
  var structured = Object.assign({}, messages, {
    type: "count_prompt",
    thinking: payload.thinking,
    request_id: payload.request_id,
    output_reserve: Math.max(1, composerOutputBudget()),
  });
  generatorSocket.send(structured);
}

function submitComposer() {
  startGeneration();
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

// The composer action has two jobs: Stop while a run is in flight,
// and Generate at rest. Starting over has its own toolbar button.
function currentGenerateLabel() {
  return isGenerating ? "Stop" : "Send";
}

function updateGenerateButton() {
  var actionBlocking = conversationActions.blocking();
  var editBlocking = generatorEdit && (
    generatorEdit.editing() || generatorEdit.confirming()
  );
  if (isGenerating) {
    btnGenerate.classList.add("is-stop");
    btnGenerate.disabled = false;
  } else {
    btnGenerate.classList.remove("is-stop");
    btnGenerate.disabled =
      actionBlocking
      || editBlocking
      || conversationBusy
      || generatorRun.saving()
      || !(
        modelReady
        && generatorModelPanel.validation().valid
      );
  }
  btnNewConversation.disabled =
    !conversationCanStartNew();
  updateSaveButton();
  updateGenerateIdleEffect();
}

function conversationActionStateChanged() {
  syncModelPanelDisabled();
  updateGenerateButton();
}

function syncModelPanelDisabled() {
  generatorModelPanel.setDisabled(Boolean(
    isGenerating
    || conversationBusy
    || conversationActions.blocking()
    || switchWatch !== null
  ));
}

// One-time discovery nudge: Generate idles with the diffusion cycle
// before the first fresh run, then follows the visual-effect setting.
var GENERATE_TEASED_KEY = "diffusion_generate_teased";
var GENERATE_CYCLE_HOLD_MS = 2000;
var generateCycleTimer = null;
var generateCycleActive = false;
var generateCycleLabel = "";

function generateTeaserActive() {
  try {
    return localStorage.getItem(GENERATE_TEASED_KEY) !== "1";
  } catch (_error) {
    return false;
  }
}

function markGenerateTeased() {
  persistSet(GENERATE_TEASED_KEY, "1");
}

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

// ---- UI state helpers ----

function setGenerating(active) {
  if (active) {
    generatorCanvas.startRunSegment({
      preserveCurrentHeight:
        generatorEdit !== null && generatorEdit.resuming(),
    });
  }
  isGenerating = active;
  // Generate stays visible; it just greys out while the model runs
  // (and whenever the model is not ready, params are invalid, or a
  // save is completing) -- centralized in updateGenerateButton.
  updateGenerateButton();
  generatorComposer.setDisabled(active);
  syncModelPanelDisabled();
  generatorEdit.generationChanged(active);
}

function setConversationBusy(active) {
  conversationBusy = active === true;
  syncModelPanelDisabled();
  updateGenerateButton();
}

function renderConversation(action) {
  var identity = generatorRun
    ? generatorRun.conversationIdentity()
    : null;
  conversationShell.render({
    action: action,
    runIdentity: identity,
    conversationIdentity:
      conversationStateIdentity(conversationState),
    runFrameCount: generatorRun
      ? generatorRun.frameCount()
      : 0,
    renderTranscript: function (workspaceVisible) {
      conversationView.render(conversationState, {
        workspaceVisible: workspaceVisible,
        workspaceAssistantTurnId:
          workspaceVisible && identity
            ? identity.assistant_turn_id
            : null,
      });
    },
  });
}

function applyConversationAction(action) {
  var previousBranchId = conversationState.selectedBranchId;
  var nextState = conversationStateReduce(
    conversationState, action
  );
  conversationState = nextState;
  conversationActions.reconcile();
  var runIdentity = generatorRun
    ? generatorRun.conversationIdentity()
    : null;
  var runMatches = generatorRun
    ? generatorRun.refreshConversation(action)
    : false;
  var hasWorkspace = generatorRun && (
    generatorRun.frameCount() > 0 || runIdentity !== null
  );
  var branchChanged = (
    previousBranchId !== null
    && previousBranchId !== nextState.selectedBranchId
  );
  if (
    hasWorkspace
    && (branchChanged || !runMatches)
  ) {
    clearConversationBranchWorkspace();
  }
  renderConversation(action);
  persistActiveConversation();
  if (generatorEdit) {
    generatorEdit.refreshLocks();
  }
  updateGenerateButton();
  if (
    generatorRun
    && generatorRun.frameCount() >= RUN_SNAPSHOT_FRAMES_MIN
    && (!generatorEdit || generatorEdit.shouldPersistRun())
  ) {
    generatorRun.saveSession();
  }
}

function persistActiveConversation() {
  var conversation = conversationState.conversation;
  if (conversation === null) {
    persistSet(PERSIST_ACTIVE_CONVERSATION_KEY, "");
    return;
  }
  persistSet(
    PERSIST_ACTIVE_CONVERSATION_KEY,
    JSON.stringify({
      id: conversation.id,
      revision: conversation.branch_revision,
      branch_id: conversation.branch_id,
    })
  );
}

function storedActiveConversation() {
  var raw = null;
  try {
    raw = localStorage.getItem(
      PERSIST_ACTIVE_CONVERSATION_KEY
    );
  } catch (_error) {
    return null;
  }
  if (!raw) {
    return null;
  }
  try {
    var parsed = JSON.parse(raw);
    if (
      parsed
      && /^[0-9a-f]{32}$/.test(parsed.id)
      && Number.isInteger(parsed.revision)
      && parsed.revision > 0
      && (
        parsed.branch_id === undefined
        || /^b_[0-9a-f]{32}$/.test(parsed.branch_id)
      )
    ) {
      return {
        id: parsed.id,
        revision: parsed.revision,
        branch_id: parsed.branch_id || null,
      };
    }
  } catch (_error) {
    // A malformed pointer is no conversation, not a legacy format.
  }
  persistSet(PERSIST_ACTIVE_CONVERSATION_KEY, "");
  return null;
}

function restoreActiveConversation() {
  var stored = storedActiveConversation();
  renderConversation(null);
  if (stored === null) {
    return Promise.resolve(false);
  }
  return conversationClient.restore(
    stored.id, stored.branch_id
  )
    .then(function () {
      return true;
    })
    .catch(function (error) {
      if (error && error.status === 404) {
        applyConversationAction({ type: "clear" });
      } else {
        applyConversationAction({
          type: "failed",
          error: error,
        });
      }
      return false;
    });
}

function loadOlderConversation() {
  conversationClient.loadOlder().catch(function (error) {
    reportConversationFailure("Could not load older messages", error);
  });
}

function conversationConflict() {
  generatorChrome.setMessage(
    "This conversation changed in another window. Its latest"
    + " durable state has been reloaded.",
    { color: "var(--danger)", clearColorAfterMs: 5000 }
  );
}

function reportConversationFailure(prefix, error) {
  var detail = error && error.message
    ? error.message
    : "network request failed";
  generatorChrome.setMessage(
    prefix + ": " + detail,
    { color: "var(--danger)", clearColorAfterMs: 5000 }
  );
}

function conversationPromptForSend() {
  var conversation = conversationState.conversation;
  if (
    conversation
    && conversation.pending_assistant_id !== null
  ) {
    var activeUser = conversationStateActiveUser(conversationState);
    return activeUser ? activeUser.text : "";
  }
  return generatorComposer.trimmedValue();
}

function conversationPendingMatchesModel() {
  var conversation = conversationState.conversation;
  if (
    !conversation
    || conversation.pending_assistant_id === null
  ) {
    return true;
  }
  var assistant = conversationStateTailAssistant(
    conversationState
  );
  var modelId = generatorModelPanel.activeModelId();
  if (assistant && assistant.model_id === modelId) {
    return true;
  }
  generatorChrome.setMessage(
    "This pending response belongs to "
    + (assistant ? assistant.model_id : "another model")
    + ". Switch back to retry it, or start a new conversation.",
    { color: "var(--danger)" }
  );
  return false;
}

function reserveConversationAssistant(prompt) {
  var conversation = conversationState.conversation;
  if (
    conversation
    && conversation.pending_assistant_id !== null
  ) {
    return Promise.resolve(prompt);
  }
  var ensure = conversation === null
    ? conversationClient.create("New conversation")
    : Promise.resolve(conversation);
  return ensure.then(function () {
    var capabilities = generatorModelPanel.capabilities();
    return conversationClient.appendUser({
      text: prompt,
      modelId: generatorModelPanel.activeModelId(),
      inputMode: capabilities.input_mode,
      metadata: {},
    });
  }).then(function (result) {
    return result.user_turn.text;
  });
}

function currentConversationIdentity() {
  return conversationStateIdentity(conversationState);
}

function generatorRunReadConversation() {
  return generatorRun.conversationIdentity();
}

function activeRunCanEdit() {
  var identity = generatorRunReadConversation();
  if (identity === null) {
    return conversationState.conversation === null;
  }
  return conversationStateCanEdit(conversationState, identity);
}

function addConversationIdentity(message) {
  var identity = generatorRunReadConversation();
  if (identity === null) {
    return message;
  }
  message.conversation_id = identity.conversation_id;
  message.branch_id = identity.branch_id;
  message.branch_revision = identity.branch_revision;
  message.assistant_turn_id = identity.assistant_turn_id;
  message.assistant_turn_index =
    identity.assistant_turn_index;
  return message;
}

function completeConversationAssistant(result) {
  var identity = generatorRunReadConversation();
  if (identity === null) {
    return Promise.resolve(false);
  }
  setConversationBusy(true);
  var provenance = generatorRun.provenance();
  var contextPack =
    provenance && provenance.context_pack
      ? provenance.context_pack
      : {};
  return conversationClient.updateAssistant({
    branchId: identity.branch_id,
    branchRevision: identity.branch_revision,
    assistantTurnId: identity.assistant_turn_id,
    text: result.text,
    partial: result.partial === true,
    contextPack: contextPack,
    metadata: { status: result.status },
  }).then(function () {
    setConversationBusy(false);
    return true;
  }).catch(function (error) {
    setConversationBusy(false);
    if (conversationAssistantMatches(identity, result)) {
      return true;
    }
    reportConversationFailure(
      "Response text could not be saved", error
    );
    return false;
  });
}

function queueConversationAssistant(result) {
  conversationCompletion = completeConversationAssistant(result);
  return conversationCompletion;
}

function conversationAssistantMatches(identity, result) {
  var conversation = conversationState.conversation;
  if (!conversation || conversation.pending_assistant_id !== null) {
    return false;
  }
  if (conversation.id !== identity.conversation_id) {
    return false;
  }
  if (conversation.branch_id !== identity.branch_id) {
    return false;
  }
  var assistant = conversationStateTailAssistant(conversationState);
  if (
    !assistant
    || assistant.turn_id !== identity.assistant_turn_id
    || assistant.index !== identity.assistant_turn_index
  ) {
    return false;
  }
  if (assistant.text !== result.text) {
    return false;
  }
  if (assistant.partial !== (result.partial === true)) {
    return false;
  }
  return assistant.metadata.status === result.status;
}

function confirmConversationEdit() {
  return queueConversationAssistant({
    text: generatorRun.finalText() || "",
    partial: generatorRun.interrupted(),
    status: generatorRun.interrupted()
      ? "cancelled"
      : "completed",
  });
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

function setSaveAvailable(available) {
  saveAvailable = available === true;
  updateSaveButton();
}

function updateSaveButton() {
  var editBlocking = generatorEdit && (
    generatorEdit.editing() || generatorEdit.confirming()
  );
  btnSave.disabled = !saveAvailable
    || generatorRun.frameCount() === 0
    || generatorRun.saving()
    || conversationActions.blocking()
    || conversationTransitionBusy
    || editBlocking;
}

// Clears the footer readouts only, never the stack. The edit
// controller calls this before a branch starts, which can overlap a
// save the user started by hand; clearing the chips here would put
// back the overwriting this stack exists to fix.
function resetStatus() {
  generatorChrome.resetStatus(appSettings.tpsMode);
  generatorReadouts.clearMetrics();
}

// ---- Actions ----

// Clear all live-run state (frames, edits, overlays, gates) back to a
// pre-run baseline. Shared by Generate and New Conversation.
function resetRunState() {
  if (saveCheckTimer !== null) {
    clearTimeout(saveCheckTimer);
    saveCheckTimer = null;
  }
  btnSave.classList.remove("is-saving", "is-saved");
  generatorWatermark.close();
  generatorEdit.reset();
  generatorEdit.setSavingControls(false);
  generatorRun.reset();
  generatorCanvas.reset();
  generatorReadouts.reset();
  generatorEdit.refreshLocks();
  renderConversation(null);
  updateGenerateButton();
  setSaveAvailable(false);
}

function clearConversationBranchWorkspace() {
  if (
    !generatorRun
    || (
      generatorRun.frameCount() === 0
      && generatorRun.conversationIdentity() === null
    )
  ) {
    return;
  }
  resetRunState();
  generatorRun.clearSession();
  if (thinkingPanel) {
    thinkingPanel.hidden = true;
    thinkingContent.textContent = "";
  }
  resetStatus();
  generatorChrome.showOutputPlaceholder(
    generatorModelPanel.activeDisplayName()
  );
}

// Clear the active XAI workspace only after its durable transition
// succeeded. A failed create or append leaves the run recoverable.
function clearActiveRun() {
  resetRunState();
  generatorComposer.clear();
  generatorComposer.saveDraft();
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

function conversationActionBlockReason() {
  if (isGenerating) {
    return "Wait for generation to finish before changing paths.";
  }
  if (conversationBusy) {
    return "Wait for the conversation update to finish.";
  }
  var conversation = conversationState.conversation;
  if (generatorEdit && generatorEdit.confirming()) {
    return "Wait for the Edit Frames confirmation to finish.";
  }
  if (generatorEdit && generatorEdit.editing()) {
    return "Cancel or confirm the open edit before changing paths.";
  }
  if (generatorRun && generatorRun.saving()) {
    return "Wait for the run save to finish.";
  }
  if (
    conversation
    && conversation.pending_assistant_id !== null
  ) {
    return (
      "Finish or retry the pending response before changing paths."
    );
  }
  return "";
}

function conversationCanNavigate() {
  return conversationActionBlockReason() === "";
}

function conversationCanStartNew() {
  if (conversationActions.blocking()) {
    return false;
  }
  if (isGenerating || conversationBusy) {
    return false;
  }
  if (generatorRun && generatorRun.saving()) {
    return false;
  }
  if (
    generatorEdit
    && (generatorEdit.editing() || generatorEdit.confirming())
  ) {
    return false;
  }
  return true;
}

function createConversationOperationId() {
  var provider = typeof crypto === "object" ? crypto : null;
  if (
    provider === null
    || typeof provider.getRandomValues !== "function"
  ) {
    throw new Error(
      "secure random operation IDs are unavailable in this browser"
    );
  }
  var bytes = new Uint8Array(CONVERSATION_OPERATION_BYTES);
  var filled = provider.getRandomValues(bytes);
  if (filled !== bytes || bytes.length !== 16) {
    throw new Error("secure random operation ID creation failed");
  }
  var operationId = "";
  for (var index = 0; index < bytes.length; index++) {
    operationId += bytes[index].toString(16).padStart(2, "0");
  }
  if (!/^[0-9a-f]{32}$/.test(operationId)) {
    throw new Error("secure random operation ID is not canonical");
  }
  return operationId;
}

function runConversationTransition(operation) {
  if (!conversationCanNavigate()) {
    return Promise.resolve(false);
  }
  conversationTransitionBusy = true;
  setConversationBusy(true);
  return Promise.resolve().then(operation).then(
    function (result) {
      conversationTransitionBusy = false;
      setConversationBusy(false);
      return result;
    },
    function (error) {
      conversationTransitionBusy = false;
      setConversationBusy(false);
      throw error;
    }
  );
}

function selectConversationBranch(branchId) {
  return runConversationTransition(function () {
    return conversationClient.selectBranch(branchId);
  });
}

function editConversationUserFork(input) {
  return runConversationTransition(function () {
    return conversationClient.editUserFork({
      operationId: input.operationId,
      userTurnId: input.userTurnId,
      text: input.text,
      modelId: input.modelId,
      inputMode: input.inputMode,
      metadata: input.metadata,
    });
  }).then(function (result) {
    return conversationForkGenerationOutcome(
      result, "edited path", input.configuration
    );
  });
}

function deleteConversationFromPathFork(input) {
  return runConversationTransition(function () {
    return conversationClient.deleteFromPathFork({
      operationId: input.operationId,
      userTurnId: input.userTurnId,
    });
  });
}

function retryConversationAssistantFork(input) {
  return runConversationTransition(function () {
    return conversationClient.retryAssistantFork({
      operationId: input.operationId,
      assistantTurnId: input.assistantTurnId,
      modelId: input.modelId,
      inputMode: input.inputMode,
    });
  }).then(function (result) {
    return conversationForkGenerationOutcome(
      result, "retry path", input.configuration
    );
  });
}

function conversationForkGenerationOutcome(
  result, label, configuration
) {
  if (result === false) {
    return false;
  }
  var prompt = conversationPromptForSend();
  var unavailable = conversationForkLaunchUnavailable(
    prompt, configuration
  );
  if (unavailable !== "") {
    return conversationForkDeferred(result, unavailable);
  }
  var launched = launchReservedGeneration(
    prompt, configuration
  );
  if (!launched) {
    return conversationForkDeferred(
      result,
      "The " + label + " is pending. Press Send to retry it."
    );
  }
  return {
    result: result,
    launched: true,
    feedback:
      "Created the " + label + " and started generation.",
  };
}

function conversationForkLaunchUnavailable(
  prompt, configuration
) {
  if (!prompt) {
    return (
      "The new path is pending because its user message is missing."
    );
  }
  if (!modelReady || !generatorSocket.isReady()) {
    return (
      "The new path is pending because the model connection is not"
      + " ready. Press Send when it reconnects."
    );
  }
  var validation = configuration
    ? {
      valid: configuration.valid,
      message: configuration.validationMessage,
    }
    : generatorModelPanel.validation();
  if (!validation.valid) {
    return (
      "The new path is pending because its confirmed Run settings"
      + " are invalid. Correct them, then press Send."
    );
  }
  if (!conversationPendingMatchesModel()) {
    return (
      "The new path is pending for its reserved model. Switch back,"
      + " then press Send."
    );
  }
  return "";
}

function conversationForkDeferred(result, message) {
  generatorChrome.setMessage(
    message,
    { color: "var(--danger)", clearColorAfterMs: 8000 }
  );
  return {
    result: result,
    launched: false,
    feedback: message,
  };
}

function startNewRun() {
  if (!conversationCanStartNew()) {
    return Promise.resolve(false);
  }
  conversationTransitionBusy = true;
  setConversationBusy(true);
  return conversationClient.create("New conversation")
    .then(function () {
      clearActiveRun();
      return true;
    })
    .catch(function (error) {
      reportConversationFailure("New conversation failed", error);
      return false;
    })
    .then(function (created) {
      conversationTransitionBusy = false;
      setConversationBusy(false);
      return created;
    });
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
    return Promise.resolve(false);
  }
  if (conversationActions.blocking()) {
    return Promise.resolve(false);
  }
  if (generatorEdit.editing() || generatorEdit.confirming()) {
    return Promise.resolve(false);
  }
  if (conversationBusy) {
    return Promise.resolve(false);
  }
  if (generatorRun.saving()) {
    return Promise.resolve(false);
  }
  if (isGenerating) {
    return Promise.resolve(false);
  }
  if (!generatorModelPanel.validation().valid) {
    return Promise.resolve(false);
  }
  var conversation = conversationState.conversation;
  if (
    conversation
    && conversation.pending_assistant_id !== null
  ) {
    return retryPendingConversation(conversation.id);
  }

  var prompt = conversationPromptForSend();
  if (!prompt) {
    generatorChrome.setMessage("Prompt is empty.");
    return Promise.resolve(false);
  }
  if (!conversationPendingMatchesModel()) {
    return Promise.resolve(false);
  }
  setConversationBusy(true);
  return reserveConversationAssistant(prompt)
    .then(function (reservedPrompt) {
      setConversationBusy(false);
      return launchReservedGeneration(reservedPrompt);
    })
    .catch(function (error) {
      setConversationBusy(false);
      reportConversationFailure("Send failed", error);
      return false;
    });
}

function retryPendingConversation(conversationId) {
  setConversationBusy(true);
  return conversationClient.restore(
    conversationId, conversationState.selectedBranchId
  )
    .then(function () {
      var conversation = conversationState.conversation;
      if (
        !conversation
        || conversation.pending_assistant_id === null
      ) {
        setConversationBusy(false);
        generatorChrome.setMessage(
          "The response was already saved. Type a new message"
          + " to continue."
        );
        return false;
      }
      var prompt = conversationPromptForSend();
      if (!prompt || !conversationPendingMatchesModel()) {
        setConversationBusy(false);
        return false;
      }
      setConversationBusy(false);
      return launchReservedGeneration(prompt);
    })
    .catch(function (error) {
      setConversationBusy(false);
      reportConversationFailure(
        "Pending response could not be checked", error
      );
      return false;
    });
}

function launchReservedGeneration(prompt, confirmedConfiguration) {
  // The first fresh run retires the Generate teaser: from now on the
  // idle diffusion cycle follows the setting.
  markGenerateTeased();

  // A fresh run abandons any in-progress edit session and clears the
  // previous run's state. Record the prompt in history first.
  generatorComposer.prepareGeneration(prompt);
  resetRunState();
  var params = confirmedConfiguration
    ? Object.assign({}, confirmedConfiguration.parameters)
    : generatorModelPanel.parameterValues();
  generatorRun.begin(prompt, params);
  renderConversation(null);
  generatorComposer.clear();
  generatorComposer.saveDraft();

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
  payload.experimental = confirmedConfiguration
    ? confirmedConfiguration.experimental
    : generatorModelPanel.experimental();
  var context = conversationStateMessages(
    conversationState, prompt
  );
  if (context === null || !context.assistant_turn_id) {
    setGenerating(false);
    generatorChrome.endRunStatus();
    generatorChrome.setMessage(
      "Send failed: the assistant reservation is missing.",
      { color: "var(--danger)" }
    );
    return false;
  }
  Object.assign(payload, context);
  if (!generatorSocket.send(payload)) {
    setGenerating(false);
    generatorChrome.endRunStatus();
    resetRunState();
    generatorChrome.setMessage(
      "Send failed before generation started. Press Send to retry"
      + " the reserved response.",
      { color: "var(--danger)" }
    );
    return false;
  }
  return true;
}

// Run serialization and save request ownership live in
// generator_run.js. The page only supplies presentation callbacks.
function saveRun(options) {
  var editConfirmation = Boolean(
    options && options.editConfirmation === true
  );
  var blocked = conversationSaveBlockReason(editConfirmation);
  if (blocked !== "") {
    generatorRunSaveRefused(blocked);
    return Promise.resolve(false);
  }
  var completion = conversationCompletion;
  return Promise.resolve(completion).then(function (completed) {
    var identity = generatorRun.conversationIdentity();
    if (completed === false && identity !== null) {
      generatorRunSaveRefused(
        "The response is not durable yet, so it was not saved."
      );
      return false;
    }
    if (
      identity !== null
      && !generatorRun.refreshConversation(null)
    ) {
      generatorRunSaveRefused(
        "The durable response changed, so this run was not saved."
      );
      return false;
    }
    return generatorRun.save();
  });
}

function conversationSaveBlockReason(editConfirmation) {
  if (
    conversationActions.blocking()
    || conversationTransitionBusy
  ) {
    return (
      "Finish the open conversation action before saving."
    );
  }
  if (
    generatorEdit.confirming()
    && !editConfirmation
  ) {
    return "Wait for the Edit Frames confirmation to finish.";
  }
  if (
    generatorEdit.editing()
    && !editConfirmation
  ) {
    return "Cancel or confirm the open edit before saving.";
  }
  return "";
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
  return generatorEdit.canvasState();
}

function generatorCandidatesReadState() {
  var tokenizer = generatorModelPanel.activeTokenizer();
  return Object.assign(
    generatorEdit.candidatesState(),
    {
      tokenizer: tokenizer,
      vocabSize: tokenizer.model_vocab_size || null,
    }
  );
}

function generatorCandidatesRequestTokenize(intent) {
  if (!generatorSocket.isReady()) {
    return false;
  }
  generatorSocket.send({
    type: "tokenize",
    text: intent.text,
    request_id: intent.requestId,
  });
  return true;
}

function generatorCandidatesRequestProbe(intent) {
  if (!generatorSocket.isReady()) {
    return false;
  }
  if (!generatorEdit.requestAllowed()) {
    return false;
  }
  generatorSocket.send(addConversationIdentity({
    type: "probe",
    position: intent.position,
    token_id: intent.tokenId,
    request_id: intent.requestId,
    run_token: generatorRun.runToken(),
  }));
  return true;
}

function generatorCandidatesRequestSubstitute(intent) {
  return generatorEdit.substitute(intent);
}

function generatorEditRequestRewind(intent) {
  if (!generatorSocket.isReady()) {
    return false;
  }
  return generatorSocket.send(addConversationIdentity({
    type: "rewind",
    run_token: intent.runToken,
  }));
}

function generatorEditRequestResume(intent) {
  var message = {
    type: "resume",
    frame_index: intent.frameIndex,
    remask_positions: intent.remaskPositions,
    run_token: intent.runToken,
  };
  if (intent.targetFrame !== null) {
    message.max_frames =
      intent.targetFrame - intent.frameIndex + 1;
  }
  if (intent.continueRun) {
    message["continue"] = true;
  }
  return generatorSocket.send(addConversationIdentity(message));
}

function generatorEditRequestSubstitute(intent) {
  var message = {
    type: "substitute",
    position: intent.position,
    token_id: intent.tokenId,
    run_token: intent.runToken,
  };
  if (intent.typedText) {
    message.typed = true;
    message.typed_text = intent.typedText;
  }
  return generatorSocket.send(addConversationIdentity(message));
}

function generatorCanvasWriteHighlight(value) {
  appSettings.highlightTokens = value;
  overlaysWriteHighlightTokens(value);
}

function generatorCanvasOutputReset() {
  generatorCandidates.outputReset();
  generatorReadouts.outputReset();
}

function generatorCanvasRendered() {
  generatorReadouts.rendered();
}

function generatorCanvasOverlayChanged() {
  generatorCandidates.hidePopover();
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
  return generatorEdit.readoutsSettings();
}

function generatorReadoutsReadScrubber() {
  return generatorEdit.scrubberState();
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
  return generatorEdit.readArtifacts();
}

function generatorRunRestoreEditArtifacts(state) {
  generatorEdit.restoreArtifacts(state);
}

function generatorRunSessionRestored() {
  renderConversation(null);
  updateGenerateButton();
  setSaveAvailable(!generatorRun.saved());
  generatorEdit.activate();
  generatorCanvas.restoreRunHeight();
}

function generatorRunSaveStart(info) {
  btnSave.disabled = true;
  updateGenerateButton();
  generatorEdit.setSavingControls(true);
  generatorEdit.refreshLocks();
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
  updateGenerateButton();
  generatorEdit.setSavingControls(false);
  generatorEdit.refreshLocks();
}

function generatorRunSaveSuccess(info) {
  return linkSavedRun(info).then(function (linked) {
    saveAvailable = !linked;
    btnSave.classList.add("is-saved");
    saveCheckTimer = setTimeout(function () {
      btnSave.classList.remove("is-saved");
      saveCheckTimer = null;
    }, 500);
    generatorEdit.refreshLocks();
    updateGenerateButton();
    generatorChrome.showAnalyticsCue(info.runId || "");
    generatorChrome.retireStatus(info.status);
    if (linked) {
      generatorChrome.setMessage(
        "Saved " + info.label + " run to " + info.result.path,
        { color: "var(--accent)" }
      );
    }
    updateSaveButton();
    return linked;
  });
}

function linkSavedRun(info) {
  var identity = info.conversationIdentity;
  if (
    identity === null
    || !info.runId
    || !Number.isInteger(info.revision)
  ) {
    return Promise.resolve(identity === null);
  }
  if (!conversationStateCanEdit(conversationState, identity)) {
    reportConversationFailure(
      "Run saved, but its response changed before linking",
      new Error("the exact assistant version is no longer selected")
    );
    return Promise.resolve(false);
  }
  return conversationClient.linkRun({
    branchId: identity.branch_id,
    assistantTurnId: identity.assistant_turn_id,
    assistantTurnIndex: identity.assistant_turn_index,
    assistantTurnVersion: identity.assistant_turn_version,
    runId: info.runId,
    runRevision: info.revision,
  }).then(function () {
    return true;
  }).catch(function (error) {
    if (conversationRunLinkMatches(identity, info)) {
      return true;
    }
    reportConversationFailure(
      "Run saved, but its conversation link failed", error
    );
    return false;
  });
}

function conversationRunLinkMatches(identity, info) {
  var conversation = conversationState.conversation;
  if (!conversation || conversation.id !== identity.conversation_id) {
    return false;
  }
  if (conversation.branch_id !== identity.branch_id) {
    return false;
  }
  var assistant = conversationStateTailAssistant(conversationState);
  if (
    !assistant
    || assistant.turn_id !== identity.assistant_turn_id
    || assistant.index !== identity.assistant_turn_index
  ) {
    return false;
  }
  var link = assistant.run_link;
  return Boolean(
    link
    && link.run_id === info.runId
    && link.revision === info.revision
  );
}

function generatorRunSaveFailure(info) {
  btnSave.classList.remove("is-saving", "is-saved");
  updateSaveButton();
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
    if (isGenerating) {
      requestCancel();
    } else {
      startGeneration();
    }
  }
);
btnNewConversation.addEventListener("click", function () {
  if (!btnNewConversation.disabled) {
    startNewRun();
  }
});
btnSave.addEventListener("click", saveRun);

conversationActions.wire();
conversationView.wire();
generatorComposer.wire();
generatorModelPanel.wire();
generatorChrome.wire();
generatorCanvas.wire();
generatorReadouts.wire();
generatorCandidates.wire();
generatorEdit.wire();
generatorWatermark.wire();
generatorModals.wire();

// Raising the loading curtain has to clear the modals first. They are
// native dialogs now, so an open one is in the top layer, which sits
// above every z-index including this overlay's 100. That is reachable
// rather than theoretical: another window swapping the model raises
// the curtain with About open, and the About box would float over it.
// A swap invalidates the page underneath anyway, so the dialog goes.
function raiseLoadingOverlay() {
  conversationActions.closeAll();
  generatorModals.closeAll();
  generatorComposer.closeImport();
  generatorWatermark.close();
  generatorChrome.showLoading();
}

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
  generatorWatermark.configure(capabilities);
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
  restoreActiveConversation().then(finishGeneratorBoot);
}

function finishGeneratorBoot() {
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
  showRescueNotice();
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
