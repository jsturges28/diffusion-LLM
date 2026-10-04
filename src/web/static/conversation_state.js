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

if (CONVERSATION_TURNS_MAX !== 200) {
  throw new Error("Conversation cache must hold 200 turns");
}
if (CONVERSATION_MESSAGE_TURNS_MAX % 2 !== 1) {
  throw new Error("Conversation context must end on a user turn");
}

function conversationStateCreate() {
  return conversationStateSeal({
    conversation: null,
    turns: [],
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

function conversationStateCreated(conversation) {
  return {
    conversation: conversationStateManifest(conversation),
    turns: [],
    nextBefore: null,
    hasMore: false,
    pagesLoaded: 0,
    loadingOlder: false,
    error: "",
  };
}

function conversationStateLoaded(action) {
  var manifest = conversationStateManifest(action.conversation);
  var page = conversationStatePage(action.page, manifest.id);
  if (page.revision !== manifest.revision) {
    throw new Error(
      "Conversation page and manifest revisions differ"
    );
  }
  return {
    conversation: manifest,
    turns: conversationStateTrim(page.turns),
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
    rawPage, state.conversation.id
  );
  if (page.revision !== state.conversation.revision) {
    throw new Error("Older page has a stale conversation revision");
  }
  var turns = conversationStateMerge(page.turns, state.turns);
  var pages = Math.min(
    CONVERSATION_PAGES_MAX,
    state.pagesLoaded + (page.turns.length > 0 ? 1 : 0)
  );
  var canLoad = page.hasMore
    && pages < CONVERSATION_PAGES_MAX
    && turns.length < CONVERSATION_TURNS_MAX;
  return conversationStateCopy(state, {
    turns: conversationStateTrim(turns),
    nextBefore: canLoad ? page.nextBefore : null,
    hasMore: canLoad,
    pagesLoaded: pages,
    loadingOlder: false,
    error: "",
  });
}

function conversationStateAppended(state, action) {
  var manifest = conversationStateManifest(action.conversation);
  var turns = conversationStateMerge(state.turns, [
    conversationStateTurn(action.userTurn),
    conversationStateTurn(action.assistantTurn),
  ]);
  return conversationStateCopy(state, {
    conversation: manifest,
    turns: conversationStateTrim(turns),
    loadingOlder: false,
    error: "",
  });
}

function conversationStateTailChanged(state, action) {
  var manifest = conversationStateManifest(action.conversation);
  var changed = conversationStateTurn(action.turn);
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

function conversationStateCopy(state, changes) {
  return Object.assign({}, state, changes);
}

function conversationStateMerge(earlier, later) {
  var byId = {};
  var combined = earlier.concat(later);
  for (var index = 0; index < combined.length; index++) {
    var turn = conversationStateTurn(combined[index]);
    byId[turn.turn_id] = turn;
  }
  var turns = Object.keys(byId).map(function (turnId) {
    return byId[turnId];
  });
  turns.sort(conversationStateTurnOrder);
  return turns;
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
  var id = conversationStateString(raw.id, "conversation id");
  if (!/^[0-9a-f]{32}$/.test(id)) {
    throw new TypeError("Conversation id has an invalid format");
  }
  var revision = conversationStatePositive(
    raw.revision, "conversation revision"
  );
  var turnCount = conversationStateNonnegative(
    raw.turn_count, "conversation turn count"
  );
  var tailId = conversationStateOptionalString(raw.tail_turn_id);
  var pending = conversationStateOptionalString(
    raw.pending_assistant_id
  );
  var tailVersion = raw.tail_version === null
    ? null
    : conversationStatePositive(
      raw.tail_version, "tail version"
    );
  conversationStateManifestInvariants(
    turnCount, tailId, tailVersion, pending
  );
  return Object.freeze({
    id: id,
    title: typeof raw.title === "string" ? raw.title : "",
    revision: revision,
    turn_count: turnCount,
    tail_turn_id: tailId,
    tail_version: tailVersion,
    pending_assistant_id: pending,
  });
}

function conversationStateTurn(raw) {
  if (!raw || typeof raw !== "object") {
    throw new TypeError("Conversation turn must be an object");
  }
  var role = raw.role;
  if (role !== "user" && role !== "assistant") {
    throw new TypeError("Conversation turn has an invalid role");
  }
  var turn = {
    turn_id: conversationStateString(raw.turn_id, "turn id"),
    index: conversationStatePositive(raw.index, "turn index"),
    version: conversationStatePositive(raw.version, "turn version"),
    role: role,
    text: conversationStateText(raw.text),
    partial: raw.partial === true,
    model_id: conversationStateOptionalString(raw.model_id),
    input_mode: conversationStateOptionalString(raw.input_mode),
    context_pack: conversationStateContextPack(raw.context_pack),
    metadata: conversationStateMetadata(raw.metadata),
    run_link: conversationStateRunLink(raw.run_link),
  };
  var expectedId = conversationStateTurnId(turn.index);
  if (turn.turn_id !== expectedId) {
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
  turnCount, tailId, tailVersion, pending
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
  if (tailId !== conversationStateTurnId(turnCount)) {
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

function conversationStatePage(raw, conversationId) {
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
  var turns = raw.turns.map(conversationStateTurn);
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
    revision: conversationStatePositive(
      raw.revision, "page revision"
    ),
    turns: turns,
    nextBefore: conversationStateOptionalString(raw.next_before),
    hasMore: raw.has_more === true,
  };
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

function conversationStateMetadata(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return Object.freeze({});
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
  if (manifest.pending_assistant_id !== null) {
    return false;
  }
  if (manifest.tail_turn_id !== identity.assistant_turn_id) {
    return false;
  }
  if (
    !Number.isInteger(identity.conversation_revision)
    || manifest.revision !== identity.conversation_revision
  ) {
    return false;
  }
  var assistant = conversationStateTailAssistant(state);
  if (assistant === null) {
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
  payload.conversation_id = conversation.id;
  payload.conversation_revision = conversation.revision;
  payload.assistant_turn_id =
    conversation.pending_assistant_id;
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
    conversation_revision: state.conversation.revision,
    assistant_turn_id: tail,
    turn_index: parseInt(tail, 10),
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
  if (state.turns.length > CONVERSATION_TURNS_MAX) {
    throw new RangeError(
      "Conversation state exceeds its cache bound"
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
  return Object.freeze(state);
}

function conversationStateTurnOrder(left, right) {
  return left.index - right.index;
}

function conversationStateTurnId(index) {
  return String(index).padStart(8, "0");
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
