// The generator's active run: frames, identity, persistence and save.
//
// Loaded as a classic script after run_snapshot.js and before app.js.
// The returned controller keeps every mutable run store in its
// closure. The page supplies narrow reads and callbacks for the model
// panel, composer, chrome, edit log and render memos.

"use strict";

function generatorRunCreate(options) {
  function requiredCallback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "generatorRunCreate needs options." + name
      );
    }
    return options[name];
  }

  var readModel = requiredCallback("readModel");
  var readComposer = requiredCallback("readComposer");
  var restoreComposer =
    requiredCallback("restoreComposer");
  var readChrome = requiredCallback("readChrome");
  var restoreChrome = requiredCallback("restoreChrome");
  var readEditArtifacts =
    requiredCallback("readEditArtifacts");
  var restoreEditArtifacts =
    requiredCallback("restoreEditArtifacts");
  var invalidateRender =
    requiredCallback("invalidateRender");
  var onSessionRestored =
    requiredCallback("onSessionRestored");
  var requestSave = requiredCallback("requestSave");
  var onSaveStart = requiredCallback("onSaveStart");
  var onSaveSuccess = requiredCallback("onSaveSuccess");
  var onSaveFailure = requiredCallback("onSaveFailure");
  var onSaveRefused = requiredCallback("onSaveRefused");

  if (!options.storage) {
    throw new TypeError(
      "generatorRunCreate needs options.storage"
    );
  }
  if (
    typeof options.sessionKey !== "string"
    || options.sessionKey === ""
  ) {
    throw new TypeError(
      "generatorRunCreate needs options.sessionKey"
    );
  }

  var storage = options.storage;
  var sessionKey = options.sessionKey;
  var checkpoints = new WeakMap();

  var frames = runFramesCreate();
  var original = originalRunCreate();
  var positionAlts = [];
  var candidates = runCandidatesCreate();
  var originalCandidates = null;

  var runPrompt = null;
  var runParams = null;
  var finalText = null;
  var promptLength = null;
  var provenance = null;
  var totalSteps = null;

  var runToken = "";
  var residentWorker = "";
  var runWorker = "";

  var interrupted = false;
  var lostConnection = false;
  var saved = false;
  var editedSaved = false;
  var savedRunId = null;
  var savedRevision = null;
  var saving = false;

  var frameOffset = 0;
  var elapsedOffset = 0;

  function reset() {
    runFramesClear(frames);
    originalRunClear(original);
    positionAlts = [];
    candidates = runCandidatesCreate();
    originalCandidates = null;
    runPrompt = null;
    runParams = null;
    finalText = null;
    promptLength = null;
    provenance = null;
    totalSteps = null;
    runToken = "";
    runWorker = "";
    interrupted = false;
    lostConnection = false;
    saved = false;
    editedSaved = false;
    savedRunId = null;
    savedRevision = null;
    frameOffset = 0;
    elapsedOffset = 0;
    invalidateRender();
  }

  function begin(prompt, params) {
    if (typeof prompt !== "string" || prompt === "") {
      throw new TypeError(
        "generatorRun.begin needs a non-empty prompt"
      );
    }
    if (!params || typeof params !== "object") {
      throw new TypeError(
        "generatorRun.begin needs parameter values"
      );
    }
    runPrompt = prompt;
    runParams = copyObject(params);
  }

  function adoptProvenance(data) {
    if (data.provenance && typeof data.provenance === "object") {
      provenance = data.provenance;
    }
  }

  function appendFrame(data) {
    if (!data || typeof data !== "object") {
      throw new TypeError(
        "generatorRun.appendFrame needs frame data"
      );
    }
    adoptProvenance(data);
    if (data.shape === RUN_FRAME_SHAPE_APPEND) {
      return appendPositionFrame(data);
    }
    return appendSnapshotFrame(data);
  }

  function appendSnapshotFrame(data) {
    if (data.alts && data.tokens && data.tokens.length > 0) {
      positionAlts[data.tokens.length - 1] = data.alts;
    }
    var storedTokens = sealTokens(data.tokens || null);
    runFramesAppend(frames, {
      history: data.text,
      tokens: storedTokens,
      canvasIndex: numberOr(data.canvas_index, 0),
      meanConf: numberOr(data.mean_conf, null),
      elapsed: elapsedFrom(data.elapsed),
      revealed: data.revealed ? data.revealed.length : 0,
    });
    return {
      append: false,
      index: runFramesLength(frames) - 1,
      tokens: copyTokens(storedTokens),
      text: data.text,
    };
  }

  function appendPositionFrame(data) {
    var position = data.index - 1;
    if (data.alts) {
      positionAlts[position] = data.alts;
    }
    runFramesAppendPosition(frames, {
      index: data.index,
      token: sealToken(data.token),
      canvasIndex: numberOr(data.canvas_index, 0),
      meanConf: numberOr(data.mean_conf, null),
      elapsed: elapsedFrom(data.elapsed),
      revealed: data.revealed ? data.revealed.length : 0,
    });
    return {
      append: true,
      index: runFramesLength(frames) - 1,
      tokens: frameTokensLast(),
      text: null,
    };
  }

  function numberOr(value, fallback) {
    return typeof value === "number" ? value : fallback;
  }

  function elapsedFrom(value) {
    if (typeof value === "number") {
      return +(value + elapsedOffset).toFixed(2);
    }
    return lastElapsedOr(elapsedOffset);
  }

  function lastElapsedOr(fallback) {
    if (frames.elapsed.length === 0) {
      return fallback;
    }
    return frames.elapsed[frames.elapsed.length - 1];
  }

  function addCandidates(message) {
    candidates = runCandidatesAddStream(
      candidates, frameOffset, message
    );
  }

  function finish(data) {
    if (!data || typeof data !== "object") {
      throw new TypeError(
        "generatorRun.finish needs terminal data"
      );
    }
    interrupted = data.cancelled === true;
    if (data.final_text) {
      finalText = data.final_text;
    }
    if (typeof data.prompt_len === "number") {
      promptLength = data.prompt_len;
    }
    adoptProvenance(data);
    if (typeof data.run_token === "string") {
      runToken = data.run_token;
    }
    runWorker = residentWorker;
    originalRunCapture(original, frames, positionAlts);
    if (originalCandidates === null) {
      originalCandidates = candidates;
    }
    return { interrupted: interrupted };
  }

  function interruptConnection() {
    interrupted = true;
    lostConnection = true;
    if (frameCount() > 0) {
      finalText = runFramesLatestText(frames);
    }
    return frameCount() > 0;
  }

  function setTotalSteps(value) {
    totalSteps =
      typeof value === "number" ? value : null;
  }

  function truncate(offset) {
    if (!Number.isInteger(offset) || offset < 0) {
      throw new RangeError(
        "generatorRun.truncate needs a non-negative frame"
      );
    }
    frameOffset = offset;
    elapsedOffset = offset > 0
      ? (frames.elapsed[offset - 1] || 0)
      : 0;
    runFramesTruncate(frames, offset);
    candidates = runCandidatesTruncate(candidates, offset);
    invalidateRender();
  }

  function truncateAlternatives(count) {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(
        "generatorRun alternatives need a non-negative count"
      );
    }
    positionAlts.length = count;
  }

  function captureCheckpoint() {
    var checkpoint = Object.freeze({});
    checkpoints.set(checkpoint, {
      frames: runFramesSnapshot(frames),
      positionAlts: positionAlts.slice(),
      candidates: candidates,
      finalText: finalText,
      interrupted: interrupted,
      frameOffset: frameOffset,
      elapsedOffset: elapsedOffset,
    });
    return checkpoint;
  }

  function restoreCheckpoint(checkpoint) {
    var state = checkpoints.get(checkpoint);
    if (!state) {
      throw new Error(
        "generatorRun.restoreCheckpoint needs its checkpoint"
      );
    }
    runFramesRestore(frames, state.frames);
    positionAlts = state.positionAlts.slice();
    candidates = state.candidates;
    finalText = state.finalText;
    interrupted = state.interrupted;
    frameOffset = state.frameOffset;
    elapsedOffset = state.elapsedOffset;
    invalidateRender();
  }

  function frameCount() {
    return runFramesLength(frames);
  }

  function frameIsAppend() {
    return runFramesIsAppend(frames);
  }

  function frameTokens(index) {
    return copyTokens(runFramesTokensAt(frames, index));
  }

  function frameTokensLast() {
    return copyTokens(runFramesTokensLast(frames));
  }

  function frameText(index) {
    return runFramesTextAt(frames, index);
  }

  function frameTokenSeries() {
    return copyTokenSeries(frames.tokens);
  }

  function framePositions() {
    return copyTokens(frames.positions) || [];
  }

  function frameCanvas(index) {
    var value = frames.canvasIndex[index];
    return typeof value === "number" ? value : 0;
  }

  function frameCanvasSeries() {
    return frames.canvasIndex.slice();
  }

  function frameElapsedSeries() {
    return frames.elapsed.slice();
  }

  function frameMeanConfidenceSeries() {
    return frames.meanConf.slice();
  }

  function frameRevealedSeries() {
    return frames.revealed.slice();
  }

  function frameLacksDetail() {
    return runFramesLackDetail(frames);
  }

  function frameIsMultiCanvas() {
    for (var i = 0; i < frames.canvasIndex.length; i++) {
      if (frames.canvasIndex[i] > 0) {
        return true;
      }
    }
    return false;
  }

  function tokensPerSecond(mode) {
    var count = frames.elapsed.length;
    if (count === 0) {
      return null;
    }
    if (mode === "last") {
      return lastStepRate(count);
    }
    return totalRate(count);
  }

  function lastStepRate(count) {
    var seconds = count > 1
      ? frames.elapsed[count - 1] - frames.elapsed[count - 2]
      : frames.elapsed[0];
    if (!(seconds > 0)) {
      return null;
    }
    return (frames.revealed[count - 1] || 0) / seconds;
  }

  function totalRate(count) {
    var seconds = frames.elapsed[count - 1];
    if (!(seconds > 0)) {
      return null;
    }
    var produced = 0;
    for (var i = 0; i < count; i++) {
      produced += frames.revealed[i] || 0;
    }
    return produced / seconds;
  }

  function originalCaptured() {
    return originalRunCaptured(original);
  }

  function originalTotalFrames() {
    return original.totalFrames;
  }

  function originalIsAppend() {
    return originalRunIsAppend(original);
  }

  function originalTokenFrames() {
    return originalRunTokenFrames(original);
  }

  function originalTokens(index) {
    return copyTokens(originalRunTokensAt(original, index));
  }

  function originalTokensLast() {
    return copyTokens(originalRunTokensLast(original));
  }

  function originalText(index) {
    return originalRunTextAt(original, index);
  }

  function originalTokenSeries() {
    return copyTokenSeries(original.tokens);
  }

  function originalPositions() {
    return copyTokens(original.positions) || [];
  }

  function positionAlternatives(position, fromOriginal) {
    var source = fromOriginal
      ? original.positionAlts
      : positionAlts;
    return copyAlternatives(source[position]);
  }

  function hasAlternatives(fromOriginal) {
    var source = fromOriginal
      ? original.positionAlts
      : positionAlts;
    return alternativesExist(source);
  }

  function candidateSet(
    frame, position, fromOriginal, canvasOf
  ) {
    var store = candidateStore(fromOriginal);
    var found = runCandidatesSetAt(
      store, frame, position, canvasOf
    );
    if (found === null) {
      return null;
    }
    return {
      frame: found.frame,
      set: copyCandidateSet(found.set),
    };
  }

  function candidateSets(frame, fromOriginal, canvasOf) {
    var found = runCandidatesAt(
      candidateStore(fromOriginal), frame, canvasOf
    );
    if (found === null) {
      return null;
    }
    var copied = [];
    for (var i = 0; i < found.sets.length; i++) {
      copied.push(copyCandidateSet(found.sets[i]));
    }
    return copied;
  }

  function candidateStore(fromOriginal) {
    if (fromOriginal) {
      return originalCandidates || runCandidatesCreate();
    }
    return candidates;
  }

  function candidateFrames(fromOriginal) {
    return candidateStore(fromOriginal).frames.slice();
  }

  function candidateSegments(fromOriginal) {
    return candidateStore(fromOriginal).segments.slice();
  }

  function candidateRecord(fromOriginal) {
    return copyJson(
      runCandidatesToJson(candidateStore(fromOriginal))
    );
  }

  function candidatesEmpty(fromOriginal) {
    return runCandidatesIsEmpty(candidateStore(fromOriginal));
  }

  function runParameters() {
    return runParams === null ? null : copyObject(runParams);
  }

  function runProvenance() {
    return provenance === null ? null : copyJson(provenance);
  }

  function adoptResidentWorker(worker) {
    if (typeof worker !== "string" || worker === "") {
      return false;
    }
    residentWorker = worker;
    return true;
  }

  function editIdentity() {
    return {
      lostConnection: lostConnection,
      madeBy: runWorker,
      resident: residentWorker,
    };
  }

  function buildSavePayload() {
    var model = modelState();
    var composer = composerState();
    var edit = editState();
    var elapsed = frameElapsedSeries();
    var payload = {
      model: model.id,
      prompt: runPrompt !== null
        ? runPrompt
        : composer.prompt,
      params: copyObject(runParams || model.params),
      final_text: finalText,
      elapsed_seconds:
        elapsed.length > 0 ? elapsed[elapsed.length - 1] : null,
      per_frame_elapsed: elapsed,
      mean_conf: frameMeanConfidenceSeries(),
    };
    addFrameFields(payload);
    addMeasuredFacts(payload);
    addCandidatesFields(payload);
    if (edit.remaskEdits.length > 0) {
      addEditedFields(payload, edit.remaskEdits);
    }
    return payload;
  }

  function addFrameFields(payload) {
    if (!runFramesIsAppend(frames)) {
      payload.frames = frames.history.slice();
      payload.frame_tokens = tokenRecordsFrom(frames.tokens);
    } else {
      payload.frame_positions =
        positionRecordsFrom(frames.positions);
    }
    var canvas = cleanCanvasIndex(
      frames.canvasIndex, frameCount()
    );
    if (canvas !== null) {
      payload.canvas_index = canvas;
    }
  }

  function addMeasuredFacts(payload) {
    if (promptLength !== null) {
      payload.prompt_len = promptLength;
    }
    if (interrupted) {
      payload.partial = true;
    }
    if (provenance !== null) {
      payload.provenance = copyJson(provenance);
    }
    if (runToken) {
      payload.run_token = runToken;
    }
  }

  function addCandidatesFields(payload) {
    var alternatives = alternativeRecordsFrom(positionAlts);
    if (alternatives !== null) {
      payload.alternatives = alternatives;
    }
    var record = candidatesRecordFrom(candidates);
    if (record !== null) {
      payload.candidates = record;
    }
  }

  function addEditedFields(payload, edits) {
    payload.remask_edits = edits;
    if (originalRunIsAppend(original)) {
      if (original.positions.length > 0) {
        payload.original_frame_positions =
          positionRecordsFrom(original.positions);
      }
    } else if (original.tokens.length > 0) {
      payload.original_frame_tokens =
        tokenRecordsFrom(original.tokens);
    }
    addOriginalSignals(payload);
    if (savedRunId) {
      payload.run_id = savedRunId;
      if (savedRevision !== null) {
        payload.expected_revision = savedRevision;
      }
    }
  }

  function addOriginalSignals(payload) {
    if (original.elapsed.length > 0) {
      payload.original_per_frame_elapsed =
        original.elapsed.slice();
      payload.original_elapsed_seconds =
        original.elapsed[original.elapsed.length - 1];
    }
    if (original.meanConf.length > 0) {
      payload.original_mean_conf = original.meanConf.slice();
    }
    var alternatives = alternativeRecordsFrom(
      original.positionAlts
    );
    if (alternatives !== null) {
      payload.original_alternatives = alternatives;
    }
    var record = candidatesRecordFrom(originalCandidates);
    if (record !== null) {
      payload.original_candidates = record;
    }
  }

  function cleanCanvasIndex(values, count) {
    if (!values || values.length !== count) {
      return null;
    }
    for (var i = 0; i < values.length; i++) {
      if (
        typeof values[i] !== "number"
        || !isFinite(values[i])
      ) {
        return null;
      }
    }
    return values.slice();
  }

  function positionRecordsFrom(positions) {
    var records = [];
    for (var i = 0; i < positions.length; i++) {
      records.push(tokenRecord(positions[i]));
    }
    return records;
  }

  function tokenRecordsFrom(tokenFrames) {
    var output = [];
    for (var frame = 0; frame < tokenFrames.length; frame++) {
      var tokens = tokenFrames[frame];
      if (!tokens) {
        output.push(null);
        continue;
      }
      output.push(positionRecordsFrom(tokens));
    }
    return output;
  }

  function tokenRecord(token) {
    var record = {
      t: token.t,
      m: !!token.m,
      id: token.id,
    };
    if (typeof token.c === "number") {
      record.c = token.c;
    }
    if (typeof token.e === "number") {
      record.e = token.e;
    }
    if (typeof token.f === "number") {
      record.f = token.f;
    }
    return record;
  }

  function alternativeRecordsFrom(positions) {
    if (!alternativesExist(positions)) {
      return null;
    }
    var output = [];
    for (var i = 0; i < positions.length; i++) {
      output.push(alternativeSetRecord(positions[i]));
    }
    return output;
  }

  function alternativeSetRecord(alternatives) {
    if (!alternatives || alternatives.length === 0) {
      return null;
    }
    var records = [];
    for (var i = 0; i < alternatives.length; i++) {
      var record = {
        id: alternatives[i].id,
        t: alternatives[i].t,
        p: alternatives[i].p,
      };
      if (typeof alternatives[i].rank === "number") {
        record.rank = alternatives[i].rank;
      }
      records.push(record);
    }
    return records;
  }

  function alternativesExist(positions) {
    for (var i = 0; i < positions.length; i++) {
      if (positions[i] && positions[i].length > 0) {
        return true;
      }
    }
    return false;
  }

  function candidatesRecordFrom(store) {
    if (store === null || runCandidatesIsEmpty(store)) {
      return null;
    }
    var thinned = runCandidatesThin(
      store, RUN_CANDIDATES_BUDGET
    );
    if (thinned === null || runCandidatesIsEmpty(thinned)) {
      return null;
    }
    return copyJson(runCandidatesToJson(thinned));
  }

  function save() {
    if (saving) {
      return Promise.resolve();
    }
    if (frameCount() === 0 || !finalText) {
      return refuseSave(
        "This run produced nothing to save."
      );
    }
    if (frameLacksDetail()) {
      return refuseSave(
        "This run came back without its per-token detail and"
        + " cannot be saved in full. Generate it again to save"
        + " it."
      );
    }
    var edit = editState();
    var wasEdited = edit.remaskEdits.length > 0;
    var label = wasEdited ? "edited" : "original";
    saving = true;
    var status = onSaveStart({
      edited: wasEdited,
      label: label,
    });
    var payload = buildSavePayload();
    return requestSave("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (response) {
      return response.json();
    }).then(function (result) {
      saveResult(result, wasEdited, label, status);
    }).catch(function (error) {
      saveFailed(error.message, label, status);
    });
  }

  function refuseSave(message) {
    onSaveRefused(message);
    return Promise.resolve();
  }

  function saveResult(result, wasEdited, label, status) {
    saving = false;
    if (!result.success) {
      onSaveFailure({
        label: label,
        message: result.message || "unknown",
        status: status,
      });
      return;
    }
    saved = true;
    if (wasEdited) {
      editedSaved = true;
    }
    adoptSaveIdentity(result);
    onSaveSuccess({
      edited: wasEdited,
      label: label,
      result: result,
      runId: savedRunId,
      revision: savedRevision,
      status: status,
    });
    saveSession();
  }

  function saveFailed(message, label, status) {
    saving = false;
    onSaveFailure({
      label: label,
      message: message,
      status: status,
    });
  }

  function adoptSaveIdentity(result) {
    var parts = String(result.path || "").split("/");
    savedRunId =
      result.run_id || parts[parts.length - 1] || null;
    savedRevision =
      typeof result.revision === "number"
        ? result.revision
        : null;
  }

  function saveSession() {
    var tiers = runSnapshotTiers(sessionRecord());
    for (var i = 0; i < tiers.length; i++) {
      try {
        storage.setItem(
          sessionKey, JSON.stringify(tiers[i])
        );
        return true;
      } catch (_error) {
        // The next tier is deliberately smaller.
      }
    }
    return false;
  }

  function sessionRecord() {
    var model = modelState();
    var composer = composerState();
    var chrome = chromeState();
    var edit = editState();
    return {
      model: model.id,
      device: model.device,
      prompt: composer.draft,
      runPrompt: runPrompt,
      finalText: finalText,
      params: runParams,
      promptLen: promptLength,
      provenance: provenance,
      runToken: runToken,
      worker: runWorker,
      thinking: chrome.thinking,
      remaskEdits: edit.remaskEdits,
      editedRunSaved: editedSaved,
      runInterrupted: interrupted,
      runLostConnection: lostConnection,
      runSaved: saved,
      lastSavedRunId: savedRunId,
      lastSavedRevision: savedRevision,
      statusStep: chrome.status.step,
      lastRunTotalSteps: totalSteps,
      statusElapsed: chrome.status.elapsed,
      statusMessage: chrome.status.message,
      frames: frames,
      positionAlts: positionAlts,
      original: original,
      candidates: candidates,
      originalCandidates: originalCandidates,
    };
  }

  function restoreSession() {
    var model = modelState();
    if (!model.id) {
      return false;
    }
    var stored = null;
    try {
      stored = storage.getItem(sessionKey);
    } catch (_error) {
      return false;
    }
    var restored = runSnapshotDecode(stored, {
      model: model.id,
      device: model.device,
    });
    if (restored === null) {
      return false;
    }
    applyRestored(restored);
    return true;
  }

  function applyRestored(restored) {
    runFramesRestore(frames, restored.frames);
    originalRunAssign(original, restored.original);
    sealRunTokens();
    positionAlts = restored.positionAlts;
    candidates = restored.candidates;
    originalCandidates = restored.originalCandidates;
    runPrompt = restored.runPrompt;
    runParams = restored.params;
    finalText = restored.finalText;
    promptLength = restored.promptLen;
    provenance = restored.provenance;
    totalSteps = restored.lastRunTotalSteps;
    runToken = restored.runToken;
    runWorker = restored.worker;
    editedSaved = restored.editedRunSaved;
    interrupted = restored.runInterrupted;
    lostConnection = restored.runLostConnection;
    saved = restored.runSaved;
    savedRunId = restored.lastSavedRunId;
    savedRevision = restored.lastSavedRevision;
    frameOffset = 0;
    elapsedOffset = 0;
    restoreEditArtifacts({
      remaskEdits: restored.remaskEdits,
    });
    if (restored.prompt) {
      restoreComposer({ draft: restored.prompt });
    }
    invalidateRender();
    onSessionRestored();
    restoreChrome({
      thinking: restored.thinking,
      status: {
        step: restored.statusStep,
        elapsed: restored.statusElapsed,
        message: restored.statusMessage,
      },
    });
  }

  function clearSession() {
    try {
      storage.removeItem(sessionKey);
    } catch (_error) {
      // Storage unavailable means there is nothing to clear.
    }
  }

  function modelState() {
    var state = readModel();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorRun readModel must return an object"
      );
    }
    return {
      id: state.id || null,
      device: state.device || null,
      params: state.params || {},
    };
  }

  function composerState() {
    var state = readComposer();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorRun readComposer must return an object"
      );
    }
    return {
      draft: typeof state.draft === "string"
        ? state.draft
        : "",
      prompt: typeof state.prompt === "string"
        ? state.prompt
        : "",
    };
  }

  function chromeState() {
    var state = readChrome();
    if (!state || typeof state !== "object") {
      throw new TypeError(
        "generatorRun readChrome must return an object"
      );
    }
    return {
      thinking: typeof state.thinking === "string"
        ? state.thinking
        : "",
      status: state.status || {
        step: "",
        elapsed: "",
        message: "",
      },
    };
  }

  function editState() {
    var state = readEditArtifacts();
    if (!state || !Array.isArray(state.remaskEdits)) {
      throw new TypeError(
        "generatorRun edit artifacts need remaskEdits"
      );
    }
    return { remaskEdits: copyEdits(state.remaskEdits) };
  }

  function copyEdits(edits) {
    var output = [];
    for (var i = 0; i < edits.length; i++) {
      var copied = copyObject(edits[i]);
      copied.token_positions =
        (edits[i].token_positions || []).slice();
      output.push(copied);
    }
    return output;
  }

  function copyTokens(tokens) {
    if (tokens === null || tokens === undefined) {
      return null;
    }
    return tokens.slice();
  }

  function copyTokenSeries(series) {
    var output = [];
    for (var i = 0; i < series.length; i++) {
      output.push(copyTokens(series[i]));
    }
    return output;
  }

  function copyAlternatives(alternatives) {
    if (!alternatives) {
      return null;
    }
    var output = [];
    for (var i = 0; i < alternatives.length; i++) {
      output.push(copyObject(alternatives[i]));
    }
    return output;
  }

  function copyCandidateSet(set) {
    if (!set) {
      return null;
    }
    return {
      h: set.h,
      c: copyAlternatives(set.c) || [],
    };
  }

  function copyObject(value) {
    return Object.assign({}, value);
  }

  function sealRunTokens() {
    for (var i = 0; i < frames.tokens.length; i++) {
      sealTokens(frames.tokens[i]);
    }
    for (var p = 0; p < frames.positions.length; p++) {
      sealToken(frames.positions[p]);
    }
    for (var j = 0; j < original.tokens.length; j++) {
      sealTokens(original.tokens[j]);
    }
    for (var q = 0; q < original.positions.length; q++) {
      sealToken(original.positions[q]);
    }
  }

  function sealTokens(tokens) {
    if (tokens === null) {
      return null;
    }
    for (var i = 0; i < tokens.length; i++) {
      sealToken(tokens[i]);
    }
    return Object.freeze(tokens);
  }

  function sealToken(token) {
    if (token && typeof token === "object") {
      return Object.freeze(token);
    }
    return token;
  }

  function copyJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  return {
    reset: reset,
    begin: begin,
    appendFrame: appendFrame,
    addCandidates: addCandidates,
    finish: finish,
    interruptConnection: interruptConnection,
    setTotalSteps: setTotalSteps,
    truncate: truncate,
    truncateAlternatives: truncateAlternatives,
    captureCheckpoint: captureCheckpoint,
    restoreCheckpoint: restoreCheckpoint,
    frameCount: frameCount,
    frameIsAppend: frameIsAppend,
    frameTokens: frameTokens,
    frameTokensLast: frameTokensLast,
    frameText: frameText,
    frameTokenSeries: frameTokenSeries,
    framePositions: framePositions,
    frameCanvas: frameCanvas,
    frameCanvasSeries: frameCanvasSeries,
    frameElapsedSeries: frameElapsedSeries,
    frameMeanConfidenceSeries: frameMeanConfidenceSeries,
    frameRevealedSeries: frameRevealedSeries,
    frameLacksDetail: frameLacksDetail,
    frameIsMultiCanvas: frameIsMultiCanvas,
    tokensPerSecond: tokensPerSecond,
    lastElapsed: function () {
      return lastElapsedOr(null);
    },
    originalCaptured: originalCaptured,
    originalTotalFrames: originalTotalFrames,
    originalIsAppend: originalIsAppend,
    originalTokenFrames: originalTokenFrames,
    originalTokens: originalTokens,
    originalTokensLast: originalTokensLast,
    originalText: originalText,
    originalTokenSeries: originalTokenSeries,
    originalPositions: originalPositions,
    positionAlternatives: positionAlternatives,
    hasAlternatives: hasAlternatives,
    candidateSet: candidateSet,
    candidateSets: candidateSets,
    candidateFrames: candidateFrames,
    candidateSegments: candidateSegments,
    candidateRecord: candidateRecord,
    candidatesEmpty: candidatesEmpty,
    parameters: runParameters,
    finalText: function () {
      return finalText;
    },
    promptLength: function () {
      return promptLength;
    },
    provenance: runProvenance,
    totalSteps: function () {
      return totalSteps;
    },
    runToken: function () {
      return runToken;
    },
    adoptResidentWorker: adoptResidentWorker,
    editIdentity: editIdentity,
    interrupted: function () {
      return interrupted;
    },
    lostConnection: function () {
      return lostConnection;
    },
    saved: function () {
      return saved;
    },
    editedSaved: function () {
      return editedSaved;
    },
    saving: function () {
      return saving;
    },
    savedRunId: function () {
      return savedRunId;
    },
    savedRevision: function () {
      return savedRevision;
    },
    resumeFrameOffset: function () {
      return frameOffset;
    },
    resumeElapsedOffset: function () {
      return elapsedOffset;
    },
    buildSavePayload: buildSavePayload,
    save: save,
    saveSession: saveSession,
    restoreSession: restoreSession,
    clearSession: clearSession,
  };
}
