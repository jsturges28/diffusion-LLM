// The generator's run snapshot: the stored form of a finished run,
// which a trip to Analytics or a reload brings back.
//
// Loaded as a classic global script before app.js and after the two
// families it serialises, run_frames.js and run_candidates.js, and
// like them it reaches for no page and no storage. It turns one
// explicit record of the run into the payloads storage is offered,
// most complete first, and turns stored text back into validated
// state. app.js reads the page into the record, makes the storage
// calls, and applies what comes back, so every change to the page's
// own state still happens in one place.
//
// The stored keys and their order are a format, not a detail. A
// snapshot outlives the build that wrote it for as long as the app
// stays open, so every default and legacy alias below states how an
// older snapshot is read.

"use strict";

// Fewer frames than this leaves nothing to scrub between, so a run
// that short is neither written nor restored.
var RUN_SNAPSHOT_FRAMES_MIN = 2;
var RUN_SNAPSHOT_CONVERSATION_INVALID = Object.freeze({});

// The fields stored as the current record holds them, in their
// serialized order. The decoder below also reads the superseded
// numeric conversation identity fields.
var RUN_SNAPSHOT_FIELDS = [
  "model",
  "device",
  "conversationId",
  "branchId",
  "branchRevision",
  "assistantTurnId",
  "assistantTurnIndex",
  "conversationTurnVersion",
  "conversationTailText",
  "prompt",
  "runPrompt",
  "finalText",
  "params",
  "promptLen",
  "provenance",
  "runToken",
  "saveRunToken",
  "worker",
  "thinking",
  "remaskEdits",
  "editedRunSaved",
  "runInterrupted",
  "runLostConnection",
  "runSaved",
  "lastSavedRunId",
  "lastSavedRevision",
  "statusStep",
  "lastRunTotalSteps",
  "statusElapsed",
  "statusMessage",
];

// The stores the record carries as the page holds them, serialised
// through their own families rather than stored as they are.
var RUN_SNAPSHOT_STORES = [
  "frames",
  "positionAlts",
  "positionSamplerAlts",
  "original",
  "candidates",
  "originalCandidates",
];

// The payloads storage is offered for one run, most complete first,
// or none when the run is not worth keeping. Each drops what the one
// after it can live without, for when the storage quota refuses it,
// which long runs do.
function runSnapshotTiers(record) {
  runSnapshotAssertRecord(record);
  if (!runSnapshotWorthKeeping(record)) {
    return [];
  }
  var light = runSnapshotLight(record);
  var full = Object.assign(
    {},
    light,
    runFramesToJson(record.frames),
    {
      positionAlts: record.positionAlts,
      positionSamplerAlts: record.positionSamplerAlts,
    },
    originalRunToJson(record.original)
  );
  // Candidates go first: packed, a default LLaDA run's are about a
  // million characters against the desktop app's 5.2 million for the
  // whole snapshot, and losing them costs less than losing the
  // per-token detail they explain.
  var tiers = runSnapshotCandidateTiers(record, full);
  return tiers.concat([full, light]);
}

// Every field present, even when its value is empty. One the page
// forgot to read would otherwise drop out of every snapshot without
// a sound.
function runSnapshotAssertRecord(record) {
  if (!record || typeof record !== "object") {
    throw new Error("run snapshot: a record is one object");
  }
  var names = RUN_SNAPSHOT_FIELDS.concat(RUN_SNAPSHOT_STORES);
  var missing = [];
  for (var i = 0; i < names.length; i++) {
    if (!(names[i] in record)) {
      missing.push(names[i]);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      "run snapshot: the record is missing " + missing.join(", ")
    );
  }
}

function runSnapshotWorthKeeping(record) {
  if (!record.model) {
    return false;
  }
  if (runFramesLength(record.frames) < RUN_SNAPSHOT_FRAMES_MIN) {
    return false;
  }
  return !!record.finalText;
}

// The stored fields, then the three frame arrays that survive a
// quota refusal: enough to redraw the run and report its timings.
// The other three are per-token detail and ride the full payload.
function runSnapshotLight(record) {
  var light = {};
  for (var i = 0; i < RUN_SNAPSHOT_FIELDS.length; i++) {
    var name = RUN_SNAPSHOT_FIELDS[i];
    light[name] = record[name];
  }
  return Object.assign(
    light, runFramesToJson(record.frames, RUN_FRAME_LIGHT_FIELDS)
  );
}

// Both runs' candidates, then the live run's alone, so the
// baseline's give way first. The baseline's are written only once an
// edit has made them a store of their own; before that they are the
// live run's, and writing them twice could cost the quota the tokens.
function runSnapshotCandidateTiers(record, full) {
  var tiers = [];
  var withLive = Object.assign({}, full, {
    candidates: runCandidatesToSnapshot(record.candidates),
  });
  if (runSnapshotOriginalKeptApart(record)) {
    tiers.push(Object.assign({}, withLive, {
      originalCandidates: runCandidatesToSnapshot(
        record.originalCandidates
      ),
    }));
  }
  if (!runCandidatesIsEmpty(record.candidates)) {
    tiers.push(withLive);
  }
  return tiers;
}

function runSnapshotOriginalKeptApart(record) {
  return record.originalCandidates !== null
    && record.originalCandidates !== record.candidates
    && !runCandidatesIsEmpty(record.originalCandidates);
}

// Stored text back into the state the page applies, or null when
// there is nothing here to restore: no snapshot, text that is not
// one, a snapshot of another model or device, or a run too short to
// keep. `resident` is the model and device the page is running.
function runSnapshotDecode(text, resident) {
  if (!resident || !resident.model) {
    throw new Error(
      "run snapshot: decoding needs the resident model"
    );
  }
  var source = runSnapshotParse(text);
  if (source === null) {
    return null;
  }
  if (!runSnapshotIsResident(source, resident)) {
    return null;
  }
  var conversation = runSnapshotConversationFacts(source);
  if (conversation === RUN_SNAPSHOT_CONVERSATION_INVALID) {
    return null;
  }
  // A snapshot that hit the storage quota carries only three of the
  // six, so the run comes back renderable but without its per-token
  // detail. That is allowed, and the first Edit-Frames truncate
  // squares the missing three up to the same length as the rest.
  var frames = runFramesFromJson(source);
  if (runFramesLength(frames) < RUN_SNAPSHOT_FRAMES_MIN) {
    return null;
  }
  return runSnapshotState({
    source: source,
    frames: frames,
    conversation: conversation,
  });
}

function runSnapshotParse(text) {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text) || null;
  } catch (_error) {
    // Not a snapshot at all, so there is nothing to restore.
    return null;
  }
}

// Snapshots written before the device joined the identity have no
// `device` key at all. Treating that as a mismatch would silently
// drop one in-flight run per upgrade, so it is read as "matches",
// and the clear-on-switch covers the case it cannot.
function runSnapshotIsResident(source, resident) {
  if (source.model !== resident.model) {
    return false;
  }
  return source.device === undefined
    || source.device === resident.device;
}

// The run's frames and stores, over the facts runSnapshotFacts reads.
function runSnapshotState(options) {
  var source = options.source;
  var frames = options.frames;
  var state = runSnapshotFacts(source, options.conversation);
  state.frames = frames;
  state.original = originalRunFromJson(
    source, runFramesLength(frames)
  );
  state.positionAlts = source.positionAlts || [];
  state.positionSamplerAlts = source.positionSamplerAlts || [];
  state.candidates = runCandidatesFromSnapshot(source.candidates);
  state.originalCandidates = runSnapshotOriginalCandidates(
    source, state.candidates, state.remaskEdits
  );
  return state;
}

// The stored fields, each read the way an older snapshot needs it:
// a missing or malformed value reads as absent rather than as
// whatever it happened to be.
function runSnapshotFacts(source, conversation) {
  return {
    conversationId: conversation.conversationId,
    branchId: conversation.branchId,
    branchRevision: conversation.branchRevision,
    assistantTurnId: conversation.assistantTurnId,
    assistantTurnIndex: conversation.assistantTurnIndex,
    conversationTurnVersion:
      conversation.conversationTurnVersion,
    conversationTailText: conversation.conversationTailText,
    finalText: source.finalText || "",
    runPrompt: runSnapshotRunPrompt(source),
    params: source.params || null,
    promptLen: typeof source.promptLen === "number"
      ? source.promptLen
      : null,
    // Absent in snapshots written before runs had identities, which
    // reads as "no token" and costs one refused edit on the upgrade
    // rather than an edit answered from the wrong run.
    runToken: typeof source.runToken === "string"
      ? source.runToken
      : "",
    saveRunToken: typeof source.saveRunToken === "string"
      ? source.saveRunToken
      : null,
    // Absent in snapshots older than worker names, which reads as
    // unknown and keeps the run editable, as such runs always were.
    worker: typeof source.worker === "string" ? source.worker : "",
    provenance:
      source.provenance && typeof source.provenance === "object"
        ? source.provenance
        : null,
    remaskEdits: source.remaskEdits || [],
    editedRunSaved: !!source.editedRunSaved,
    runInterrupted: !!source.runInterrupted,
    // Absent in snapshots written before it existed, which reads as
    // "kept its connection", as every run was treated then.
    runLostConnection: !!source.runLostConnection,
    runSaved: !!source.runSaved,
    lastSavedRunId: source.lastSavedRunId || null,
    lastSavedRevision: typeof source.lastSavedRevision === "number"
      ? source.lastSavedRevision
      : null,
    prompt: source.prompt || "",
    thinking: source.thinking || "",
    lastRunTotalSteps: typeof source.lastRunTotalSteps === "number"
      ? source.lastRunTotalSteps
      : null,
    statusStep: source.statusStep || "",
    statusElapsed: source.statusElapsed || "",
    statusMessage: source.statusMessage || "",
  };
}

function runSnapshotConversationFacts(source) {
  var modernNames = [
    "branchId", "branchRevision", "assistantTurnIndex",
  ];
  var legacyNames = [
    "conversationRevision", "conversationTurnIndex",
  ];
  var sharedNames = [
    "conversationId",
    "assistantTurnId",
    "conversationTurnVersion",
    "conversationTailText",
  ];
  var hasModern = runSnapshotHasIdentityValue(
    source, modernNames
  );
  var hasLegacy = runSnapshotHasIdentityValue(
    source, legacyNames
  );
  var hasShared = runSnapshotHasIdentityValue(
    source, sharedNames
  );
  if (!hasModern && !hasLegacy && !hasShared) {
    return runSnapshotEmptyConversationFacts();
  }
  if (hasModern && !hasLegacy) {
    return runSnapshotModernConversationFacts(source);
  }
  if (hasLegacy && !hasModern) {
    return runSnapshotLegacyConversationFacts(source);
  }
  return RUN_SNAPSHOT_CONVERSATION_INVALID;
}

function runSnapshotModernConversationFacts(source) {
  var facts = {
    conversationId: runSnapshotConversationId(
      source.conversationId
    ),
    branchId: runSnapshotBranchId(source.branchId),
    branchRevision: runSnapshotPositive(source.branchRevision),
    assistantTurnId: runSnapshotTurnId(source.assistantTurnId),
    assistantTurnIndex: runSnapshotPositive(
      source.assistantTurnIndex
    ),
    conversationTurnVersion: runSnapshotPositive(
      source.conversationTurnVersion
    ),
    conversationTailText:
      typeof source.conversationTailText === "string"
        ? source.conversationTailText
        : null,
  };
  if (!runSnapshotConversationFactsComplete(facts)) {
    return RUN_SNAPSHOT_CONVERSATION_INVALID;
  }
  if (!runSnapshotTurnBelongsToBranch(facts)) {
    return RUN_SNAPSHOT_CONVERSATION_INVALID;
  }
  return facts;
}

function runSnapshotLegacyConversationFacts(source) {
  var conversationId = runSnapshotConversationId(
    source.conversationId
  );
  var facts = {
    conversationId: conversationId,
    branchId: conversationId === null
      ? null
      : "b_" + conversationId,
    branchRevision: runSnapshotPositive(
      source.conversationRevision
    ),
    assistantTurnId: runSnapshotTurnId(source.assistantTurnId),
    assistantTurnIndex: runSnapshotPositive(
      source.conversationTurnIndex
    ),
    conversationTurnVersion: runSnapshotPositive(
      source.conversationTurnVersion
    ),
    conversationTailText:
      typeof source.conversationTailText === "string"
        ? source.conversationTailText
        : null,
  };
  if (!runSnapshotConversationFactsComplete(facts)) {
    return RUN_SNAPSHOT_CONVERSATION_INVALID;
  }
  if (
    !/^[0-9]{8}$/.test(facts.assistantTurnId)
    || parseInt(facts.assistantTurnId, 10)
      !== facts.assistantTurnIndex
  ) {
    return RUN_SNAPSHOT_CONVERSATION_INVALID;
  }
  return facts;
}

function runSnapshotEmptyConversationFacts() {
  return {
    conversationId: null,
    branchId: null,
    branchRevision: null,
    assistantTurnId: null,
    assistantTurnIndex: null,
    conversationTurnVersion: null,
    conversationTailText: null,
  };
}

function runSnapshotHasIdentityValue(source, names) {
  for (var index = 0; index < names.length; index++) {
    var value = source[names[index]];
    if (value !== undefined && value !== null) {
      return true;
    }
  }
  return false;
}

function runSnapshotConversationFactsComplete(facts) {
  return (
    facts.conversationId !== null
    && facts.branchId !== null
    && facts.branchRevision !== null
    && facts.assistantTurnId !== null
    && facts.assistantTurnIndex !== null
    && facts.conversationTurnVersion !== null
    && facts.conversationTailText !== null
  );
}

function runSnapshotTurnBelongsToBranch(facts) {
  if (/^[0-9]{8}$/.test(facts.assistantTurnId)) {
    return (
      parseInt(facts.assistantTurnId, 10)
        === facts.assistantTurnIndex
      && facts.branchId === "b_" + facts.conversationId
    );
  }
  var matched = /^t_([0-9a-f]{32})_[0-9]{8}_[0-9a-f]{16}$/
    .exec(facts.assistantTurnId);
  return matched !== null && facts.branchId === "b_" + matched[1];
}

function runSnapshotConversationId(value) {
  if (typeof value !== "string") {
    return null;
  }
  return /^[0-9a-f]{32}$/.test(value) ? value : null;
}

function runSnapshotBranchId(value) {
  if (typeof value !== "string") {
    return null;
  }
  return /^b_[0-9a-f]{32}$/.test(value) ? value : null;
}

function runSnapshotTurnId(value) {
  if (typeof value !== "string") {
    return null;
  }
  var legacy = /^[0-9]{8}$/.test(value);
  var opaque = /^t_[0-9a-f]{32}_[0-9]{8}_[0-9a-f]{16}$/
    .test(value);
  return legacy || opaque ? value : null;
}

function runSnapshotPositive(value) {
  if (!Number.isInteger(value) || value < 1) {
    return null;
  }
  return value;
}

// A snapshot written before the run carried its own prompt has only
// the box text from when it was taken, which is the best record left
// of what ran.
function runSnapshotRunPrompt(source) {
  if (typeof source.runPrompt === "string") {
    return source.runPrompt;
  }
  if (typeof source.prompt === "string" && source.prompt !== "") {
    return source.prompt.trim();
  }
  return null;
}

// The baseline's candidates as a snapshot left them. An unedited run
// never writes them, being the live run's own, so they come back as
// the live store itself. An edited run's may have given way to the
// storage quota, and that one gets an empty store, so its Original
// page shows nothing rather than the edited run's candidates under
// the original's tokens.
function runSnapshotOriginalCandidates(
  source, candidates, remaskEdits
) {
  if (source.originalCandidates) {
    return runCandidatesFromSnapshot(source.originalCandidates);
  }
  if (remaskEdits.length > 0) {
    return runCandidatesCreate();
  }
  return candidates;
}
