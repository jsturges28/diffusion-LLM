// Pure bounded state for the durable conversation transcript.
//
// The server owns the complete conversation. This reducer owns only
// the newest four pages, enough to build a useful recent context and
// draw a bounded transcript. It stores text and small attestations,
// never run frames, token candidates, or any other XAI artifact.

"use strict";

var CONVERSATION_PAGE_SIZE = 50;
var CONVERSATION_PAGES_MAX = 4;
var CONVERSATION_TURNS_MAX =
  CONVERSATION_PAGE_SIZE * CONVERSATION_PAGES_MAX;
var CONVERSATION_MESSAGE_TURNS_MAX =
  CONVERSATION_TURNS_MAX - 1;
var CONVERSATION_BRANCH_POINTS_MAX = 256;
var CONVERSATION_BRANCHES_MAX = 256;
var CONVERSATION_IDENTIFIER_CHARS_MAX = 128;
var CONVERSATION_SCHEMA_LEGACY = 1;
var CONVERSATION_SCHEMA_BRANCHES = 2;
var CONVERSATION_PENDING_GENERATION_KEY =
  "pending_generation_v1";

if (CONVERSATION_TURNS_MAX !== 200) {
  throw new Error("Conversation cache must hold 200 turns");
}
if (CONVERSATION_MESSAGE_TURNS_MAX % 2 !== 1) {
  throw new Error("Conversation context must end on a user turn");
}

function conversationStateCreate() {
  return conversationStateSeal({
    conversation: null,
    selectedBranchId: null,
    turns: [],
    branchPoints: [],
    nextBefore: null,
    hasMore: false,
    pagesLoaded: 0,
    loadingOlder: false,
    error: "",
  });
}

function conversationStateReduce(state, action) {
  conversationStateAssert(state);
  if (!action || typeof action.type !== "string") {
    throw new TypeError("Conversation action needs a type");
  }
  var next;
  switch (action.type) {
    case "clear":
      return conversationStateCreate();
    case "created":
      next = conversationStateCreated(action.conversation);
      break;
    case "loaded":
    case "forked":
      next = conversationStateLoaded(action);
      break;
    case "older_started":
      next = conversationStateCopy(state, {
        loadingOlder: true,
        error: "",
      });
      break;
    case "older_loaded":
      next = conversationStateOlder(state, action.page);
      break;
    case "appended":
      next = conversationStateAppended(state, action);
      break;
    case "assistant_updated":
    case "run_linked":
      next = conversationStateTailChanged(state, action);
      break;
    case "catalog_refreshed":
      next = conversationStateCatalogRefreshed(state, action);
      break;
    case "failed":
      next = conversationStateCopy(state, {
        loadingOlder: false,
        error: conversationStateErrorText(action.error),
      });
      break;
    default:
      throw new Error(
        "Unknown conversation action: " + action.type
      );
  }
  conversationStateAssert(next);
  return conversationStateSeal(next);
}

function conversationStateActionChangesMessages(action) {
  if (!action || typeof action.type !== "string") {
    return false;
  }
  return (
    action.type === "clear"
    || action.type === "created"
    || action.type === "loaded"
    || action.type === "forked"
    || action.type === "older_loaded"
    || action.type === "appended"
    || action.type === "assistant_updated"
  );
}

function conversationStateCreated(conversation) {
  var manifest = conversationStateManifest(conversation);
  return {
    conversation: manifest,
    selectedBranchId: manifest.branch_id,
    turns: [],
    branchPoints: [],
    nextBefore: null,
    hasMore: false,
    pagesLoaded: 0,
    loadingOlder: false,
    error: "",
  };
}

function conversationStateLoaded(action) {
  var manifest = conversationStateManifest(action.conversation);
  var page = conversationStatePage(
    action.page,
    manifest.id,
    manifest.branch_id,
    manifest.pending_assistant_id
  );
  conversationStatePageMatchesManifest(page, manifest);
  if (page.branchRevision !== manifest.branch_revision) {
    throw new Error(
      "Conversation page and manifest revisions differ"
    );
  }
  return {
    conversation: manifest,
    selectedBranchId: manifest.branch_id,
    turns: conversationStateTrim(page.turns),
    branchPoints: page.branchPoints,
    nextBefore: page.nextBefore,
    hasMore: page.hasMore,
    pagesLoaded: page.turns.length > 0 ? 1 : 0,
    loadingOlder: false,
    error: "",
  };
}

function conversationStateOlder(state, rawPage) {
  if (state.conversation === null) {
    throw new Error("Cannot add a page without a conversation");
  }
  var page = conversationStatePage(
    rawPage,
    state.conversation.id,
    state.selectedBranchId,
    state.conversation.pending_assistant_id
  );
  conversationStatePageMatchesManifest(
    page, state.conversation
  );
  if (
    page.branchRevision
    !== state.conversation.branch_revision
  ) {
    throw new Error("Older page has a stale conversation revision");
  }
  var turns = conversationStateMerge(page.turns, state.turns);
  var branchPoints = conversationStateMergeBranchPoints(
    page.branchPoints, state.branchPoints
  );
  var pages = Math.min(
    CONVERSATION_PAGES_MAX,
    state.pagesLoaded + (page.turns.length > 0 ? 1 : 0)
  );
  var canLoad = page.hasMore
    && pages < CONVERSATION_PAGES_MAX
    && turns.length < CONVERSATION_TURNS_MAX;
  return conversationStateCopy(state, {
    turns: conversationStateTrim(turns),
    branchPoints: branchPoints,
    nextBefore: canLoad ? page.nextBefore : null,
    hasMore: canLoad,
    pagesLoaded: pages,
    loadingOlder: false,
    error: "",
  });
}

function conversationStateAppended(state, action) {
  var manifest = conversationStateManifest(action.conversation);
  conversationStateSameSelectedBranch(state, manifest);
  var user = conversationStateTurn(
    action.userTurn,
    manifest.branch_id,
    manifest.schema_version,
    manifest.pending_assistant_id
  );
  var assistant = conversationStateTurn(
    action.assistantTurn,
    manifest.branch_id,
    manifest.schema_version,
    manifest.pending_assistant_id
  );
  var turns = conversationStateMerge(
    state.turns, [user, assistant]
  );
  return conversationStateCopy(state, {
    conversation: manifest,
    turns: conversationStateTrim(turns),
    branchPoints: conversationStateAppendBranchPoints({
      points: state.branchPoints,
      branchId: manifest.branch_id,
      userIndex: user.index,
    }),
    loadingOlder: false,
    error: "",
  });
}

function conversationStateAppendBranchPoints(options) {
  return options.points.map(function (point) {
    if (
      point.turn_index !== options.userIndex
      || point.deleted_branch_ids.indexOf(options.branchId) === -1
    ) {
      return point;
    }
    return conversationStateBranchPoint({
      turn_index: point.turn_index,
      source_branch_id: point.source_branch_id,
      selected_branch_id: point.selected_branch_id,
      branch_ids: point.branch_ids,
      deleted_branch_ids: point.deleted_branch_ids.filter(
        function (deletedId) {
          return deletedId !== options.branchId;
        }
      ),
    });
  });
}

function conversationStateTailChanged(state, action) {
  var manifest = conversationStateManifest(action.conversation);
  conversationStateSameSelectedBranch(state, manifest);
  var changed = conversationStateTurn(
    action.turn,
    manifest.branch_id,
    manifest.schema_version,
    manifest.pending_assistant_id
  );
  var turns = state.turns.filter(function (turn) {
    return turn.turn_id !== changed.turn_id;
  });
  turns.push(changed);
  turns.sort(conversationStateTurnOrder);
  return conversationStateCopy(state, {
    conversation: manifest,
    turns: conversationStateTrim(turns),
    error: "",
  });
}

function conversationStateCatalogRefreshed(state, action) {
  if (state.conversation === null) {
    throw new Error("Cannot refresh an empty conversation");
  }
  var manifest = conversationStateManifest(action.conversation);
  conversationStateSameSelectedBranch(state, manifest);
  if (
    manifest.branch_revision
    !== state.conversation.branch_revision
  ) {
    throw new Error(
      "Catalog refresh changed the selected branch revision"
    );
  }
  if (
    !Array.isArray(action.pages)
    || action.pages.length < 1
    || action.pages.length > CONVERSATION_PAGES_MAX
  ) {
    throw new RangeError(
      "Catalog refresh page count is out of bounds"
    );
  }
  var points = [];
  for (var index = 0; index < action.pages.length; index++) {
    var page = conversationStatePage(
      action.pages[index],
      manifest.id,
      manifest.branch_id,
      manifest.pending_assistant_id
    );
    conversationStatePageMatchesManifest(page, manifest);
    if (page.branchRevision !== manifest.branch_revision) {
      throw new Error(
        "Catalog refresh page has another branch revision"
      );
    }
    points = conversationStateMergeBranchPoints(
      points, page.branchPoints
    );
  }
  return conversationStateCopy(state, {
    conversation: manifest,
    branchPoints: points,
    error: "",
  });
}

function conversationStateCopy(state, changes) {
  return Object.assign({}, state, changes);
}

function conversationStateMerge(earlier, later) {
  var byId = {};
  var combined = earlier.concat(later);
  for (var index = 0; index < combined.length; index++) {
    var turn = combined[index];
    if (!turn || typeof turn.turn_id !== "string") {
      throw new TypeError("Conversation merge needs parsed turns");
    }
    byId[turn.turn_id] = turn;
  }
  var turns = Object.keys(byId).map(function (turnId) {
    return byId[turnId];
  });
  turns.sort(conversationStateTurnOrder);
  return turns;
}

function conversationStateMergeBranchPoints(earlier, later) {
  var byLocation = {};
  var combined = earlier.concat(later);
  for (var index = 0; index < combined.length; index++) {
    var point = conversationStateBranchPoint(combined[index]);
    var key = point.turn_index + ":" + point.source_branch_id;
    byLocation[key] = point;
  }
  var points = Object.keys(byLocation).map(function (key) {
    return byLocation[key];
  });
  points.sort(conversationStateBranchPointOrder);
  if (points.length > CONVERSATION_BRANCH_POINTS_MAX) {
    throw new RangeError("Conversation branch points exceed 256");
  }
  return points;
}

function conversationStateTrim(turns) {
  var bounded = turns.slice();
  while (bounded.length > CONVERSATION_TURNS_MAX) {
    if (bounded.length < 2) {
      throw new Error("Cannot evict a partial exchange");
    }
    if (
      bounded[0].role !== "user"
      || bounded[1].role !== "assistant"
    ) {
      throw new Error("Conversation cache is not exchange-aligned");
    }
    bounded.splice(0, 2);
  }
  return bounded;
}

function conversationStateManifest(raw) {
  if (!raw || typeof raw !== "object") {
    throw new TypeError("Conversation manifest must be an object");
  }
  var schemaVersion = conversationStateSchemaVersion(raw);
  var id = conversationStateString(raw.id, "conversation id");
  if (!/^[0-9a-f]{32}$/.test(id)) {
    throw new TypeError("Conversation id has an invalid format");
  }
  var branchId = conversationStateSelectedBranchId(
    raw.branch_id, id, schemaVersion
  );
  var branchRevision = conversationStateRevision(
    raw, schemaVersion, "conversation"
  );
  var catalogRevision = conversationStateCatalogRevision(
    raw.catalog_revision, schemaVersion
  );
  var defaultBranchId = conversationStateDefaultBranchId(
    raw.default_branch_id,
    branchId,
    schemaVersion
  );
  var turnCount = conversationStateNonnegative(
    raw.turn_count, "conversation turn count"
  );
  var tailId = conversationStateOptionalTurnId(
    raw.tail_turn_id
  );
  var pending = conversationStateOptionalTurnId(
    raw.pending_assistant_id
  );
  var tailVersion = raw.tail_version === null
    ? null
    : conversationStatePositive(
      raw.tail_version, "tail version"
    );
  conversationStateManifestInvariants(
    turnCount,
    tailId,
    tailVersion,
    pending,
    schemaVersion
  );
  return Object.freeze({
    schema_version: schemaVersion,
    id: id,
    title: typeof raw.title === "string" ? raw.title : "",
    branch_id: branchId,
    branch_revision: branchRevision,
    revision: branchRevision,
    catalog_revision: catalogRevision,
    default_branch_id: defaultBranchId,
    turn_count: turnCount,
    tail_turn_id: tailId,
    tail_version: tailVersion,
    pending_assistant_id: pending,
  });
}

function conversationStateTurn(
  raw, selectedBranchId, schemaVersion, pendingAssistantId
) {
  if (!raw || typeof raw !== "object") {
    throw new TypeError("Conversation turn must be an object");
  }
  var role = raw.role;
  if (role !== "user" && role !== "assistant") {
    throw new TypeError("Conversation turn has an invalid role");
  }
  var branchId = raw.branch_id;
  if (branchId === undefined || branchId === null) {
    branchId = selectedBranchId;
  }
  branchId = conversationStateBranchId(branchId, "turn branch id");
  var turnId = conversationStateTurnIdValue(raw.turn_id);
  var index = conversationStatePositive(raw.index, "turn index");
  var version = conversationStatePositive(
    raw.version, "turn version"
  );
  var text = conversationStateText(raw.text);
  var partial = raw.partial === true;
  var modelId = conversationStateOptionalString(raw.model_id);
  var inputMode = conversationStateOptionalString(raw.input_mode);
  var pendingReservation = (
    schemaVersion === CONVERSATION_SCHEMA_BRANCHES
    && pendingAssistantId === turnId
    && role === "assistant"
    && version === 1
    && text === ""
    && partial
  );
  var turn = {
    turn_id: turnId,
    branch_id: branchId,
    index: index,
    version: version,
    role: role,
    text: text,
    partial: partial,
    model_id: modelId,
    input_mode: inputMode,
    context_pack: conversationStateContextPack(raw.context_pack),
    metadata: conversationStateMetadata(raw.metadata, {
      pendingReservation: pendingReservation,
      modelId: modelId,
      inputMode: inputMode,
    }),
    run_link: conversationStateRunLink(raw.run_link),
  };
  if (
    schemaVersion === CONVERSATION_SCHEMA_LEGACY
    && turn.turn_id !== conversationStateTurnId(turn.index)
  ) {
    throw new Error("Conversation turn id and index differ");
  }
  var expectedRole = turn.index % 2 === 1
    ? "user"
    : "assistant";
  if (turn.role !== expectedRole) {
    throw new Error("Conversation turn breaks role order");
  }
  return Object.freeze(turn);
}

function conversationStateManifestInvariants(
  turnCount, tailId, tailVersion, pending, schemaVersion
) {
  if (turnCount % 2 !== 0) {
    throw new Error("Conversation turn count must be even");
  }
  if (turnCount === 0) {
    if (
      tailId !== null
      || tailVersion !== null
      || pending !== null
    ) {
      throw new Error("Empty conversation cannot have a tail");
    }
    return;
  }
  if (tailId === null) {
    throw new Error("Conversation tail needs an id");
  }
  if (
    schemaVersion === CONVERSATION_SCHEMA_LEGACY
    && tailId !== conversationStateTurnId(turnCount)
  ) {
    throw new Error("Conversation tail does not match turn count");
  }
  if (tailVersion === null) {
    throw new Error("Conversation tail needs a version");
  }
  if (pending !== null && pending !== tailId) {
    throw new Error("Pending assistant must be the tail");
  }
  if ((tailVersion === 1) !== (pending !== null)) {
    throw new Error("Only a reserved assistant has version one");
  }
}

function conversationStatePage(
  raw, conversationId, selectedBranchId, pendingAssistantId
) {
  if (!raw || typeof raw !== "object") {
    throw new TypeError("Conversation page must be an object");
  }
  if (raw.conversation_id !== conversationId) {
    throw new Error(
      "Conversation page belongs to another conversation"
    );
  }
  if (!Array.isArray(raw.turns)) {
    throw new TypeError("Conversation page turns must be a list");
  }
  if (raw.turns.length > CONVERSATION_PAGE_SIZE) {
    throw new RangeError("Conversation page exceeds 50 turns");
  }
  var schemaVersion = conversationStateSchemaVersion(raw);
  var branchId = conversationStateSelectedBranchId(
    raw.branch_id,
    conversationId,
    schemaVersion,
    selectedBranchId
  );
  var branchRevision = conversationStateRevision(
    raw, schemaVersion, "page"
  );
  var turns = raw.turns.map(function (turn) {
    return conversationStateTurn(
      turn, branchId, schemaVersion, pendingAssistantId
    );
  });
  turns.sort(conversationStateTurnOrder);
  if (turns.length > 0 && turns[0].role !== "user") {
    throw new Error("Conversation page must start on a user turn");
  }
  for (var index = 1; index < turns.length; index++) {
    if (turns[index].index !== turns[index - 1].index + 1) {
      throw new Error("Conversation page has a turn gap");
    }
  }
  return {
    schemaVersion: schemaVersion,
    branchId: branchId,
    branchRevision: branchRevision,
    catalogRevision: conversationStateCatalogRevision(
      raw.catalog_revision, schemaVersion
    ),
    defaultBranchId: conversationStateDefaultBranchId(
      raw.default_branch_id,
      branchId,
      schemaVersion
    ),
    turns: turns,
    branchPoints: conversationStateBranchPoints(
      raw.branch_points, schemaVersion
    ),
    nextBefore: conversationStateOptionalString(raw.next_before),
    hasMore: raw.has_more === true,
  };
}

function conversationStatePageMatchesManifest(page, manifest) {
  if (page.schemaVersion !== manifest.schema_version) {
    throw new Error("Conversation page schema changed while loading");
  }
  if (page.branchId !== manifest.branch_id) {
    throw new Error("Conversation page belongs to another branch");
  }
  if (page.catalogRevision !== manifest.catalog_revision) {
    throw new Error(
      "Conversation catalog changed while loading a page"
    );
  }
  if (page.defaultBranchId !== manifest.default_branch_id) {
    throw new Error(
      "Conversation default branch changed while loading"
    );
  }
}

function conversationStateSameSelectedBranch(state, manifest) {
  if (state.conversation === null) {
    throw new Error("Conversation mutation has no active state");
  }
  if (state.conversation.id !== manifest.id) {
    throw new Error("Conversation mutation changed conversation");
  }
  if (state.selectedBranchId !== manifest.branch_id) {
    throw new Error("Conversation mutation changed selected branch");
  }
}

function conversationStateBranchPoints(raw, schemaVersion) {
  if (
    raw === undefined
    && schemaVersion === CONVERSATION_SCHEMA_LEGACY
  ) {
    return [];
  }
  if (!Array.isArray(raw)) {
    throw new TypeError("Conversation branch points must be a list");
  }
  if (raw.length > CONVERSATION_BRANCH_POINTS_MAX) {
    throw new RangeError("Conversation branch points exceed 256");
  }
  var points = raw.map(conversationStateBranchPoint);
  points.sort(conversationStateBranchPointOrder);
  for (var index = 1; index < points.length; index++) {
    if (
      points[index].turn_index === points[index - 1].turn_index
      && points[index].source_branch_id
        === points[index - 1].source_branch_id
    ) {
      throw new Error("Conversation branch point is duplicated");
    }
  }
  return points;
}

function conversationStateBranchPoint(raw) {
  if (!raw || typeof raw !== "object") {
    throw new TypeError(
      "Conversation branch point must be an object"
    );
  }
  var source = conversationStateBranchId(
    raw.source_branch_id, "source branch id"
  );
  var selected = conversationStateBranchId(
    raw.selected_branch_id, "selected branch id"
  );
  var branches = conversationStateBranchIdList(
    raw.branch_ids, "branch ids"
  );
  var deleted = conversationStateBranchIdList(
    raw.deleted_branch_ids, "deleted branch ids", true
  );
  if (branches.indexOf(source) === -1) {
    throw new Error("Branch point source is not an alternative");
  }
  if (branches.indexOf(selected) === -1) {
    throw new Error("Branch point selection is not an alternative");
  }
  for (var index = 0; index < deleted.length; index++) {
    if (branches.indexOf(deleted[index]) === -1) {
      throw new Error("Deleted branch is not an alternative");
    }
  }
  return Object.freeze({
    turn_index: conversationStatePositive(
      raw.turn_index, "branch point turn index"
    ),
    source_branch_id: source,
    selected_branch_id: selected,
    branch_ids: branches,
    deleted_branch_ids: deleted,
  });
}

function conversationStateBranchIdList(raw, name, emptyAllowed) {
  if (!Array.isArray(raw)) {
    throw new TypeError(name + " must be a list");
  }
  if (
    (!emptyAllowed && raw.length < 1)
    || raw.length > CONVERSATION_BRANCHES_MAX
  ) {
    throw new RangeError(name + " count is out of bounds");
  }
  var values = raw.map(function (value) {
    return conversationStateBranchId(value, name);
  });
  if (new Set(values).size !== values.length) {
    throw new Error(name + " contains a duplicate");
  }
  Object.freeze(values);
  return values;
}

function conversationStateContextPack(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return Object.freeze({});
  }
  var packed = {};
  var scalars = [
    "first_included_index",
    "omitted_turn_count",
    "prompt_token_count",
    "output_reserve",
    "requested_total_budget",
    "effective_total_budget",
  ];
  for (var index = 0; index < scalars.length; index++) {
    var name = scalars[index];
    if (Number.isInteger(raw[name]) && raw[name] >= 0) {
      packed[name] = raw[name];
    }
  }
  if (Array.isArray(raw.included_turn_ids)) {
    packed.included_turn_ids = raw.included_turn_ids
      .filter(function (value) {
        return typeof value === "string";
      })
      .slice(0, CONVERSATION_TURNS_MAX);
    Object.freeze(packed.included_turn_ids);
  }
  return Object.freeze(packed);
}

function conversationStateMetadata(raw, turn) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return Object.freeze({});
  }
  if (turn && turn.pendingReservation) {
    return conversationStatePendingMetadata(raw, turn);
  }
  var kept = {};
  var names = ["status", "finish_reason", "interrupted"];
  for (var index = 0; index < names.length; index++) {
    var value = raw[names[index]];
    if (
      typeof value === "string"
      || typeof value === "boolean"
    ) {
      kept[names[index]] = value;
    }
  }
  return Object.freeze(kept);
}

function conversationStatePendingMetadata(raw, turn) {
  var names = Object.keys(raw);
  if (names.length === 0) {
    return Object.freeze({});
  }
  if (
    names.length !== 1
    || names[0] !== CONVERSATION_PENDING_GENERATION_KEY
  ) {
    throw new Error(
      "Pending assistant metadata has unrelated fields"
    );
  }
  var kept = {};
  kept[CONVERSATION_PENDING_GENERATION_KEY] =
    conversationStateGenerationConfiguration(
      raw[CONVERSATION_PENDING_GENERATION_KEY],
      turn
    );
  return Object.freeze(kept);
}

function conversationStateGenerationConfiguration(raw, turn) {
  return conversationGenerationCreate().fromWire(
    raw, turn.modelId, turn.inputMode
  );
}

function conversationStateRunLink(raw) {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (!raw || typeof raw !== "object") {
    throw new TypeError("Run link must be an object or null");
  }
  return Object.freeze({
    run_id: conversationStateString(raw.run_id, "run id"),
    revision: conversationStateNonnegative(
      raw.revision, "run revision"
    ),
  });
}

function conversationStateCanEdit(state, identity) {
  conversationStateAssert(state);
  if (state.conversation === null) {
    return true;
  }
  if (!identity || typeof identity !== "object") {
    return false;
  }
  if (
    typeof identity.assistant_turn_id !== "string"
    || identity.assistant_turn_id === ""
  ) {
    return false;
  }
  var manifest = state.conversation;
  if (identity.conversation_id !== manifest.id) {
    return false;
  }
  if (identity.branch_id !== manifest.branch_id) {
    return false;
  }
  if (manifest.pending_assistant_id !== null) {
    return false;
  }
  if (manifest.tail_turn_id !== identity.assistant_turn_id) {
    return false;
  }
  if (
    !Number.isInteger(identity.branch_revision)
    || manifest.branch_revision !== identity.branch_revision
  ) {
    return false;
  }
  var assistant = conversationStateTailAssistant(state);
  if (assistant === null) {
    return false;
  }
  if (
    !Number.isInteger(identity.assistant_turn_index)
    || assistant.index !== identity.assistant_turn_index
  ) {
    return false;
  }
  if (
    !Number.isInteger(identity.assistant_turn_version)
    || assistant.version !== identity.assistant_turn_version
  ) {
    return false;
  }
  if (
    typeof identity.assistant_text !== "string"
    || assistant.text !== identity.assistant_text
  ) {
    return false;
  }
  return true;
}

function conversationStateTailAssistant(state) {
  conversationStateAssert(state);
  if (state.conversation === null) {
    return null;
  }
  var tailId = state.conversation.tail_turn_id;
  for (var index = state.turns.length - 1; index >= 0; index--) {
    if (state.turns[index].turn_id === tailId) {
      return state.turns[index];
    }
  }
  return null;
}

function conversationStatePendingGenerationConfiguration(state) {
  conversationStateAssert(state);
  if (
    state.conversation === null
    || state.conversation.pending_assistant_id === null
  ) {
    return null;
  }
  var assistant = conversationStateTailAssistant(state);
  if (
    assistant === null
    || assistant.turn_id
      !== state.conversation.pending_assistant_id
  ) {
    throw new Error(
      "Pending assistant is missing from the conversation cache"
    );
  }
  var configuration = assistant.metadata[
    CONVERSATION_PENDING_GENERATION_KEY
  ];
  return configuration || null;
}

function conversationStateActiveUser(state) {
  var assistant = conversationStateTailAssistant(state);
  if (assistant === null) {
    return null;
  }
  for (var index = state.turns.length - 1; index >= 0; index--) {
    var turn = state.turns[index];
    if (
      turn.role === "user"
      && turn.index === assistant.index - 1
    ) {
      return turn;
    }
  }
  return null;
}

function conversationStateMessages(state, draft) {
  conversationStateAssert(state);
  var turns = conversationStateCandidateTurns(state, draft);
  if (turns.length === 0) {
    return null;
  }
  while (turns.length > CONVERSATION_MESSAGE_TURNS_MAX) {
    turns.splice(0, 2);
  }
  var offset = turns[0].index - 1;
  var messages = turns.map(function (turn) {
    return {
      role: turn.role,
      content: turn.text,
      turn_id: turn.turn_id,
    };
  });
  var result = {
    messages: messages,
    candidate_turn_offset: offset,
  };
  conversationStateAddIdentity(state, result);
  return result;
}

function conversationStateCandidateTurns(state, draft) {
  var turns = state.turns.slice();
  var conversation = state.conversation;
  if (conversation === null) {
    return conversationStateDraftTurn(1, draft);
  }
  var pending = conversation.pending_assistant_id;
  if (pending !== null) {
    turns = turns.filter(function (turn) {
      return turn.turn_id !== pending;
    });
  } else {
    turns = turns.concat(
      conversationStateDraftTurn(
        conversation.turn_count + 1, draft
      )
    );
  }
  while (turns.length > 0 && turns[0].role !== "user") {
    turns.shift();
  }
  return turns;
}

function conversationStateDraftTurn(index, draft) {
  if (typeof draft !== "string" || draft.trim() === "") {
    return [];
  }
  return [{
    turn_id: conversationStateTurnId(index),
    index: index,
    role: "user",
    text: draft.trim(),
  }];
}

function conversationStateAddIdentity(state, payload) {
  var conversation = state.conversation;
  if (
    conversation === null
    || conversation.pending_assistant_id === null
  ) {
    return;
  }
  var assistant = conversationStateTailAssistant(state);
  if (
    assistant === null
    || assistant.turn_id !== conversation.pending_assistant_id
  ) {
    throw new Error("Pending assistant is missing from the cache");
  }
  payload.conversation_id = conversation.id;
  payload.branch_id = conversation.branch_id;
  payload.branch_revision = conversation.branch_revision;
  payload.assistant_turn_id =
    conversation.pending_assistant_id;
  payload.assistant_turn_index = assistant.index;
}

function conversationStateIdentity(state) {
  conversationStateAssert(state);
  if (state.conversation === null) {
    return null;
  }
  var tail = state.conversation.tail_turn_id;
  if (tail === null) {
    return null;
  }
  var assistant = conversationStateTailAssistant(state);
  if (assistant === null) {
    return null;
  }
  return {
    conversation_id: state.conversation.id,
    branch_id: state.conversation.branch_id,
    branch_revision: state.conversation.branch_revision,
    assistant_turn_id: tail,
    assistant_turn_index: assistant.index,
    assistant_turn_version: assistant.version,
    assistant_text: assistant.text,
  };
}

function conversationStateAssert(state) {
  if (!state || typeof state !== "object") {
    throw new TypeError("Conversation state must be an object");
  }
  if (!Array.isArray(state.turns)) {
    throw new TypeError("Conversation state turns must be a list");
  }
  if (!Array.isArray(state.branchPoints)) {
    throw new TypeError(
      "Conversation state branch points must be a list"
    );
  }
  if (state.conversation === null) {
    if (state.selectedBranchId !== null) {
      throw new Error("Empty state cannot select a branch");
    }
  } else if (
    state.selectedBranchId !== state.conversation.branch_id
  ) {
    throw new Error("Selected branch and manifest differ");
  }
  if (state.turns.length > CONVERSATION_TURNS_MAX) {
    throw new RangeError(
      "Conversation state exceeds its cache bound"
    );
  }
  if (
    state.branchPoints.length
    > CONVERSATION_BRANCH_POINTS_MAX
  ) {
    throw new RangeError(
      "Conversation branch points exceed their cache bound"
    );
  }
  for (var index = 1; index < state.turns.length; index++) {
    if (state.turns[index - 1].index >= state.turns[index].index) {
      throw new Error("Conversation turns must be chronological");
    }
  }
  if (
    state.pagesLoaded < 0
    || state.pagesLoaded > CONVERSATION_PAGES_MAX
  ) {
    throw new RangeError("Conversation page count is out of bounds");
  }
  return true;
}

function conversationStateSeal(state) {
  Object.freeze(state.turns);
  Object.freeze(state.branchPoints);
  return Object.freeze(state);
}

function conversationStateTurnOrder(left, right) {
  return left.index - right.index;
}

function conversationStateBranchPointOrder(left, right) {
  if (left.turn_index !== right.turn_index) {
    return left.turn_index - right.turn_index;
  }
  return left.source_branch_id.localeCompare(
    right.source_branch_id
  );
}

function conversationStateTurnId(index) {
  return String(index).padStart(8, "0");
}

function conversationStateSchemaVersion(raw) {
  var value = raw.schema_version;
  if (value === undefined || value === null) {
    return CONVERSATION_SCHEMA_LEGACY;
  }
  if (
    value !== CONVERSATION_SCHEMA_LEGACY
    && value !== CONVERSATION_SCHEMA_BRANCHES
  ) {
    throw new TypeError("Conversation schema version is unsupported");
  }
  return value;
}

function conversationStateSelectedBranchId(
  raw, conversationId, schemaVersion, selectedBranchId
) {
  var legacy = "b_" + conversationId;
  if (schemaVersion === CONVERSATION_SCHEMA_BRANCHES) {
    return conversationStateBranchId(raw, "branch id");
  }
  var value = raw;
  if (value === undefined || value === null) {
    value = selectedBranchId || legacy;
  }
  var branchId = conversationStateBranchId(value, "branch id");
  if (branchId !== legacy) {
    throw new Error("Legacy conversation has a foreign branch");
  }
  return branchId;
}

function conversationStateDefaultBranchId(
  raw, branchId, schemaVersion
) {
  if (raw === undefined || raw === null) {
    if (schemaVersion === CONVERSATION_SCHEMA_BRANCHES) {
      throw new TypeError("Conversation default branch is missing");
    }
    return branchId;
  }
  var defaultBranchId = conversationStateBranchId(
    raw, "default branch id"
  );
  if (
    schemaVersion === CONVERSATION_SCHEMA_LEGACY
    && defaultBranchId !== branchId
  ) {
    throw new Error("Legacy default branch must be selected");
  }
  return defaultBranchId;
}

function conversationStateRevision(raw, schemaVersion, name) {
  var value = raw.branch_revision;
  if (value === undefined || value === null) {
    if (schemaVersion === CONVERSATION_SCHEMA_BRANCHES) {
      throw new TypeError(name + " branch revision is missing");
    }
    value = raw.revision;
  }
  var revision = conversationStatePositive(
    value, name + " branch revision"
  );
  if (
    raw.revision !== undefined
    && raw.revision !== null
    && raw.revision !== revision
  ) {
    throw new Error(name + " revision aliases differ");
  }
  return revision;
}

function conversationStateCatalogRevision(raw, schemaVersion) {
  if (raw === undefined || raw === null) {
    if (schemaVersion === CONVERSATION_SCHEMA_BRANCHES) {
      throw new TypeError("Conversation catalog revision is missing");
    }
    return 0;
  }
  return conversationStateNonnegative(
    raw, "conversation catalog revision"
  );
}

function conversationStateBranchId(value, name) {
  var branchId = conversationStateString(value, name);
  if (!/^b_[0-9a-f]{32}$/.test(branchId)) {
    throw new TypeError(name + " has an invalid format");
  }
  return branchId;
}

function conversationStateTurnIdValue(value) {
  var turnId = conversationStateString(value, "turn id");
  if (turnId.length > CONVERSATION_IDENTIFIER_CHARS_MAX) {
    throw new RangeError("Turn id exceeds 128 characters");
  }
  var legacy = /^[0-9]{8}$/.test(turnId);
  var opaque = /^t_[0-9a-f]{32}_[0-9]{8}_[0-9a-f]{16}$/
    .test(turnId);
  if (!legacy && !opaque) {
    throw new TypeError("Turn id has an invalid format");
  }
  return turnId;
}

function conversationStateOptionalTurnId(value) {
  if (value === null || value === undefined) {
    return null;
  }
  return conversationStateTurnIdValue(value);
}

function conversationStateErrorText(error) {
  if (error && typeof error.message === "string") {
    return error.message;
  }
  return String(error || "Conversation request failed");
}

function conversationStateString(value, name) {
  if (typeof value !== "string" || value === "") {
    throw new TypeError(name + " must be a non-empty string");
  }
  return value;
}

function conversationStateOptionalString(value) {
  if (value === null || value === undefined) {
    return null;
  }
  return conversationStateString(value, "optional value");
}

function conversationStateText(value) {
  if (typeof value !== "string") {
    throw new TypeError("Conversation text must be a string");
  }
  return value;
}

function conversationStatePositive(value, name) {
  if (!Number.isInteger(value) || value < 1) {
    throw new TypeError(name + " must be a positive integer");
  }
  return value;
}

function conversationStateNonnegative(value, name) {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeError(name + " must be a non-negative integer");
  }
  return value;
}
