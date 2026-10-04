// Strict REST and compare-and-swap client for conversations.
//
// Mutations share one promise tail. Each reads the latest reducer
// revision only when it reaches the front of that queue, so a quick
// completion followed by a run link cannot send the same stale CAS.
// A conflict is reloaded but never retried blindly: retrying an
// append could duplicate the user's text.

"use strict";

function conversationClientCreate(options) {
  var owner = {
    request: conversationClientCallback(options, "request"),
    readState: conversationClientCallback(options, "readState"),
    applyAction: conversationClientCallback(
      options, "applyAction"
    ),
    onConflict: conversationClientOptional(options, "onConflict"),
    mutationTail: Promise.resolve(),
  };
  return Object.freeze({
    create: function (title) {
      return conversationClientCreateConversation(owner, title);
    },
    restore: function (conversationId) {
      return conversationClientRestore(owner, conversationId);
    },
    loadOlder: function () {
      return conversationClientLoadOlder(owner);
    },
    appendUser: function (input) {
      return conversationClientAppendUser(owner, input);
    },
    updateAssistant: function (input) {
      return conversationClientUpdateAssistant(owner, input);
    },
    linkRun: function (input) {
      return conversationClientLinkRun(owner, input);
    },
    flush: function () {
      return owner.mutationTail;
    },
  });
}

function conversationClientEnqueue(owner, operation) {
  var task = function () {
    return Promise.resolve().then(operation).catch(function (error) {
      if (!conversationClientShouldReload(error)) {
        throw error;
      }
      return conversationClientReloadConflict(owner, error)
        .then(function () {
          throw error;
        });
    });
  };
  var pending = owner.mutationTail.then(task, task);
  owner.mutationTail = pending.catch(function () {
    // A failed mutation retires without poisoning later operations.
  });
  return pending;
}

function conversationClientCreateConversation(owner, title) {
  return conversationClientEnqueue(owner, function () {
    return conversationClientJson(
      owner.request,
      "/api/conversations",
      {
        method: "POST",
        headers: conversationClientHeaders(),
        body: JSON.stringify({
          title: title || "New conversation",
        }),
      }
    ).then(function (body) {
      var conversation = conversationClientManifest(body);
      owner.applyAction({
        type: "created",
        conversation: conversation,
      });
      return conversation;
    });
  });
}

function conversationClientRestore(owner, conversationId) {
  conversationClientIdentifier(conversationId, "conversation id");
  return conversationClientLoadConsistent(
    owner.request, conversationId, 2
  ).then(function (loaded) {
    owner.applyAction({
      type: "loaded",
      conversation: loaded.conversation,
      page: loaded.page,
    });
    return owner.readState();
  });
}

function conversationClientLoadConsistent(
  request, conversationId, retries
) {
  return conversationClientMetadata(
    request, conversationId
  ).then(function (conversation) {
    return conversationClientPage(
      request, conversationId, null
    ).then(function (page) {
      if (page.revision === conversation.revision) {
        return { conversation: conversation, page: page };
      }
      if (retries > 0) {
        return conversationClientLoadConsistent(
          request, conversationId, retries - 1
        );
      }
      throw new Error(
        "Conversation kept changing while it was loaded"
      );
    });
  });
}

function conversationClientLoadOlder(owner) {
  var state = owner.readState();
  if (
    !state.conversation
    || !state.hasMore
    || !state.nextBefore
    || state.loadingOlder
  ) {
    return Promise.resolve(state);
  }
  owner.applyAction({ type: "older_started" });
  return conversationClientPage(
    owner.request, state.conversation.id, state.nextBefore
  ).then(function (page) {
    owner.applyAction({ type: "older_loaded", page: page });
    return owner.readState();
  }).catch(function (error) {
    owner.applyAction({ type: "failed", error: error });
    throw error;
  });
}

function conversationClientAppendUser(owner, input) {
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var body = {
      expected_revision: state.conversation.revision,
      text: conversationClientText(input.text, "user text"),
      model_id: conversationClientIdentifier(
        input.modelId, "model id"
      ),
      input_mode: conversationClientInputMode(input.inputMode),
      metadata: conversationClientObject(input.metadata),
    };
    return conversationClientMutation(
      owner.request,
      conversationClientTurnsUrl(state.conversation.id),
      "POST",
      body
    ).then(function (result) {
      conversationClientAppendResult(result);
      owner.applyAction({
        type: "appended",
        conversation: result.conversation,
        userTurn: result.user_turn,
        assistantTurn: result.assistant_turn,
      });
      return result;
    });
  });
}

function conversationClientUpdateAssistant(owner, input) {
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var assistantId = conversationClientIdentifier(
      input.assistantTurnId, "assistant turn id"
    );
    var body = conversationClientAssistantBody(state, input);
    var url = conversationClientTurnUrl(
      state.conversation.id, assistantId
    );
    return conversationClientMutation(
      owner.request, url, "PUT", body
    ).then(function (result) {
      conversationClientTailResult(result);
      owner.applyAction({
        type: "assistant_updated",
        conversation: result.conversation,
        turn: result.turn,
      });
      return result;
    });
  });
}

function conversationClientAssistantBody(state, input) {
  return {
    expected_revision: state.conversation.revision,
    text: conversationClientText(
      input.text, "assistant text", true
    ),
    partial: conversationClientBoolean(input.partial, "partial"),
    context_pack: conversationClientObject(input.contextPack),
    metadata: conversationClientObject(input.metadata),
  };
}

function conversationClientLinkRun(owner, input) {
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var assistantId = conversationClientIdentifier(
      input.assistantTurnId, "assistant turn id"
    );
    var body = {
      expected_revision: state.conversation.revision,
      run_id: conversationClientIdentifier(input.runId, "run id"),
      run_revision: conversationClientNonnegative(
        input.runRevision, "run revision"
      ),
    };
    var url = conversationClientTurnUrl(
      state.conversation.id, assistantId
    ) + "/run";
    return conversationClientMutation(
      owner.request, url, "PUT", body
    ).then(function (result) {
      conversationClientTailResult(result);
      owner.applyAction({
        type: "run_linked",
        conversation: result.conversation,
        turn: result.turn,
      });
      return result;
    });
  });
}

function conversationClientReloadConflict(owner, error) {
  var state = owner.readState();
  var conversationId = error.conversationId;
  if (!conversationId && state.conversation) {
    conversationId = state.conversation.id;
  }
  if (!conversationId) {
    return Promise.resolve();
  }
  return conversationClientRestore(owner, conversationId)
    .then(function () {
      error.conversationReloaded = true;
      if (conversationClientIsConflict(error) && owner.onConflict) {
        owner.onConflict(error);
      }
    });
}

function conversationClientMutation(request, url, method, body) {
  return conversationClientJson(request, url, {
    method: method,
    headers: conversationClientHeaders(),
    body: JSON.stringify(body),
  });
}

function conversationClientMetadata(request, conversationId) {
  var url = "/api/conversations/"
    + encodeURIComponent(conversationId) + "/metadata";
  return conversationClientJson(request, url, {
    method: "GET",
  }).then(conversationClientManifest);
}

function conversationClientPage(request, conversationId, before) {
  var url = conversationClientTurnsUrl(conversationId)
    + "?limit=" + CONVERSATION_PAGE_SIZE;
  if (before !== null) {
    url += "&before=" + encodeURIComponent(before);
  }
  return conversationClientJson(request, url, {
    method: "GET",
  }).then(function (body) {
    conversationClientPageShape(body, conversationId);
    return body;
  });
}

function conversationClientJson(request, url, init) {
  var responseStatus = 0;
  return Promise.resolve().then(function () {
    return request(url, init);
  }).then(function (response) {
    if (!response || typeof response.json !== "function") {
      throw new TypeError("Conversation response is not HTTP JSON");
    }
    responseStatus = Number(response.status) || 0;
    return response.json().then(function (body) {
      if (!response.ok) {
        throw conversationClientHttpError(
          body, responseStatus
        );
      }
      if (!body || typeof body !== "object") {
        throw new TypeError(
          "Conversation response body must be an object"
        );
      }
      return body;
    });
  }).catch(function (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(String(error));
  });
}

function conversationClientHttpError(body, status) {
  var message = body && typeof body.error === "string"
    ? body.error
    : "Conversation request failed (" + status + ")";
  var error = new Error(message);
  error.status = status;
  error.reason = body && typeof body.reason === "string"
    ? body.reason
    : "";
  error.conversationId =
    body && typeof body.conversation_id === "string"
      ? body.conversation_id
      : "";
  error.revision =
    body && Number.isInteger(body.revision)
      ? body.revision
      : null;
  return error;
}

function conversationClientIsConflict(error) {
  return Boolean(
    error
    && error.status === 409
    && error.reason === "revision_conflict"
  );
}

function conversationClientShouldReload(error) {
  if (conversationClientIsConflict(error)) {
    return true;
  }
  return Boolean(
    error
    && (
      error.status === undefined
      || error.status === 0
    )
  );
}

function conversationClientManifest(body) {
  if (!body || typeof body !== "object") {
    throw new TypeError("Conversation metadata body is invalid");
  }
  conversationStateReduce(conversationStateCreate(), {
    type: "created",
    conversation: body.conversation,
  });
  return body.conversation;
}

function conversationClientPageShape(body, conversationId) {
  var rawTurns = Array.isArray(body.turns) ? body.turns : [];
  var tail = rawTurns.length > 0
    ? rawTurns[rawTurns.length - 1]
    : null;
  var turnCount = tail && Number.isInteger(tail.index)
    ? tail.index
    : 0;
  var tailId = tail ? tail.turn_id : null;
  var tailVersion = tail ? tail.version : null;
  var pending = tail && tail.role === "assistant"
    && tail.version === 1
    ? tail.turn_id
    : null;
  var manifest = {
    id: conversationId,
    revision: conversationClientPositive(
      body.revision, "page revision"
    ),
    turn_count: turnCount,
    tail_turn_id: tailId,
    tail_version: tailVersion,
    pending_assistant_id: pending,
  };
  conversationStateReduce(conversationStateCreate(), {
    type: "loaded",
    conversation: manifest,
    page: body,
  });
}

function conversationClientAppendResult(result) {
  conversationClientManifest(result);
  conversationStateTurn(result.user_turn);
  conversationStateTurn(result.assistant_turn);
}

function conversationClientTailResult(result) {
  conversationClientManifest(result);
  conversationStateTurn(result.turn);
}

function conversationClientMutableState(state) {
  conversationStateAssert(state);
  if (!state.conversation) {
    throw new Error("No active conversation");
  }
  return state;
}

function conversationClientTurnsUrl(conversationId) {
  return "/api/conversations/"
    + encodeURIComponent(conversationId) + "/turns";
}

function conversationClientTurnUrl(conversationId, turnId) {
  return conversationClientTurnsUrl(conversationId)
    + "/" + encodeURIComponent(turnId);
}

function conversationClientHeaders() {
  return { "Content-Type": "application/json" };
}

function conversationClientCallback(options, name) {
  if (!options || typeof options[name] !== "function") {
    throw new TypeError(
      "conversationClientCreate needs options." + name
    );
  }
  return options[name];
}

function conversationClientOptional(options, name) {
  if (!options || options[name] === undefined) {
    return null;
  }
  if (typeof options[name] !== "function") {
    throw new TypeError(
      "conversationClientCreate options." + name
      + " must be a function"
    );
  }
  return options[name];
}

function conversationClientIdentifier(value, name) {
  if (typeof value !== "string" || value === "") {
    throw new TypeError(name + " must be a non-empty string");
  }
  if (value.length > 128) {
    throw new RangeError(name + " exceeds 128 characters");
  }
  return value;
}

function conversationClientText(value, name, emptyAllowed) {
  if (typeof value !== "string") {
    throw new TypeError(name + " must be a string");
  }
  if (!emptyAllowed && value.trim() === "") {
    throw new TypeError(name + " must not be blank");
  }
  return value;
}

function conversationClientInputMode(value) {
  if (value !== "chat" && value !== "completion") {
    throw new TypeError("input mode must be chat or completion");
  }
  return value;
}

function conversationClientObject(value) {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Conversation metadata must be an object");
  }
  return value;
}

function conversationClientBoolean(value, name) {
  if (typeof value !== "boolean") {
    throw new TypeError(name + " must be a boolean");
  }
  return value;
}

function conversationClientPositive(value, name) {
  if (!Number.isInteger(value) || value < 1) {
    throw new TypeError(name + " must be a positive integer");
  }
  return value;
}

function conversationClientNonnegative(value, name) {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeError(name + " must be a non-negative integer");
  }
  return value;
}
