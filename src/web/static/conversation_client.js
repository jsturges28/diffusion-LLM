// Strict REST and compare-and-swap client for conversations.
//
// Mutations share one promise tail. Each reads the latest reducer
// revision only when it reaches the front of that queue, so a quick
// completion followed by a run link cannot send the same stale CAS.
// A conflict is reloaded but never retried blindly: retrying an
// append could duplicate the user's text.

"use strict";

var CONVERSATION_SELECTION_EPOCH_MAX = 1000000;

function conversationClientCreate(options) {
  var owner = {
    request: conversationClientCallback(options, "request"),
    readState: conversationClientCallback(options, "readState"),
    applyAction: conversationClientCallback(
      options, "applyAction"
    ),
    onConflict: conversationClientOptional(options, "onConflict"),
    mutationTail: Promise.resolve(),
    selectionEpoch: 0,
  };
  return Object.freeze({
    create: function (title) {
      return conversationClientCreateConversation(owner, title);
    },
    restore: function (conversationId, branchId) {
      return conversationClientRestore(
        owner, conversationId, branchId
      );
    },
    listBranches: function () {
      return conversationClientListBranches(owner);
    },
    selectBranch: function (branchId) {
      return conversationClientSelectBranch(owner, branchId);
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
    editUserFork: function (input) {
      return conversationClientEditUserFork(owner, input);
    },
    deleteFromPathFork: function (input) {
      return conversationClientDeleteFromPathFork(owner, input);
    },
    retryAssistantFork: function (input) {
      return conversationClientRetryAssistantFork(owner, input);
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

function conversationClientRestore(
  owner, conversationId, selectedBranchId, actionType
) {
  conversationClientIdentifier(conversationId, "conversation id");
  if (selectedBranchId !== undefined && selectedBranchId !== null) {
    conversationClientBranchId(selectedBranchId);
  }
  return conversationClientLoadConsistent(
    owner.request, conversationId, selectedBranchId || null, 2
  ).then(function (loaded) {
    owner.applyAction({
      type: actionType || "loaded",
      conversation: loaded.conversation,
      page: loaded.page,
    });
    return owner.readState();
  });
}

function conversationClientLoadConsistent(
  request, conversationId, selectedBranchId, retries
) {
  return conversationClientMetadata(
    request, conversationId, selectedBranchId
  ).then(function (conversation) {
    var manifest = conversationStateManifest(conversation);
    return conversationClientPage(
      request, conversationId, manifest.branch_id, null
    ).then(function (page) {
      var parsed = conversationStatePage(
        page, conversationId, manifest.branch_id
      );
      if (conversationClientPageIsCurrent(parsed, manifest)) {
        return { conversation: conversation, page: page };
      }
      if (retries > 0) {
        return conversationClientLoadConsistent(
          request,
          conversationId,
          selectedBranchId,
          retries - 1
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
    owner.request,
    state.conversation.id,
    state.selectedBranchId,
    state.nextBefore
  ).then(function (page) {
    var parsed = conversationStatePage(
      page,
      state.conversation.id,
      state.selectedBranchId
    );
    if (
      !conversationClientPageIsCurrent(
        parsed, state.conversation
      )
    ) {
      return conversationClientRestore(
        owner,
        state.conversation.id,
        state.selectedBranchId,
        "loaded"
      );
    }
    owner.applyAction({ type: "older_loaded", page: page });
    return owner.readState();
  }).catch(function (error) {
    owner.applyAction({ type: "failed", error: error });
    throw error;
  });
}

function conversationClientListBranches(owner) {
  var state = conversationClientMutableState(owner.readState());
  var conversationId = state.conversation.id;
  var url = "/api/conversations/"
    + encodeURIComponent(conversationId) + "/branches";
  return conversationClientJson(owner.request, url, {
    method: "GET",
  }).then(function (body) {
    conversationClientBranchesShape(body, conversationId);
    return body;
  });
}

function conversationClientSelectBranch(owner, branchId) {
  var selected = conversationClientBranchId(branchId);
  var epoch = conversationClientNextSelectionEpoch(owner);
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    if (selected === state.selectedBranchId) {
      return state;
    }
    return conversationClientLoadConsistent(
      owner.request,
      state.conversation.id,
      selected,
      2
    ).then(function (loaded) {
      if (epoch !== owner.selectionEpoch) {
        return owner.readState();
      }
      owner.applyAction({
        type: "loaded",
        conversation: loaded.conversation,
        page: loaded.page,
      });
      return owner.readState();
    });
  });
}

function conversationClientNextSelectionEpoch(owner) {
  owner.selectionEpoch = (
    owner.selectionEpoch % CONVERSATION_SELECTION_EPOCH_MAX
  ) + 1;
  return owner.selectionEpoch;
}

function conversationClientAppendUser(owner, input) {
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var body = {
      branch_id: state.selectedBranchId,
      branch_revision: state.conversation.branch_revision,
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
      body,
      state
    ).then(function (result) {
      conversationClientAppendResult(result);
      return conversationClientApplyMutation(
        owner,
        state.conversation,
        result,
        {
          type: "appended",
          conversation: result.conversation,
          userTurn: result.user_turn,
          assistantTurn: result.assistant_turn,
        }
      );
    });
  });
}

function conversationClientUpdateAssistant(owner, input) {
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    conversationClientSelectedBranch(state, input);
    conversationClientSelectedRevision(state, input);
    var assistantId = conversationClientIdentifier(
      input.assistantTurnId, "assistant turn id"
    );
    var body = conversationClientAssistantBody(state, input);
    var url = conversationClientTurnUrl(
      state.conversation.id, assistantId
    );
    return conversationClientMutation(
      owner.request, url, "PUT", body, state
    ).then(function (result) {
      conversationClientTailResult(result);
      return conversationClientApplyMutation(
        owner,
        state.conversation,
        result,
        {
          type: "assistant_updated",
          conversation: result.conversation,
          turn: result.turn,
        }
      );
    });
  });
}

function conversationClientEditUserFork(owner, input) {
  var operationId = conversationClientOperationId(
    input && input.operationId
  );
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var turnId = conversationClientIdentifier(
      input.userTurnId, "user turn id"
    );
    var body = conversationClientForkBody(
      state, operationId
    );
    body.text = conversationClientText(input.text, "user text");
    body.model_id = conversationClientIdentifier(
      input.modelId, "model id"
    );
    body.input_mode = conversationClientInputMode(input.inputMode);
    body.metadata = conversationClientObject(input.metadata);
    var url = conversationClientBranchesUrl(
      state.conversation.id
    ) + "/edit-user/" + encodeURIComponent(turnId);
    return conversationClientForkMutation(
      owner, state, url, body, "edit"
    );
  });
}

function conversationClientDeleteFromPathFork(owner, input) {
  var operationId = conversationClientOperationId(
    input && input.operationId
  );
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var turnId = conversationClientIdentifier(
      input.userTurnId, "user turn id"
    );
    var url = conversationClientBranchesUrl(
      state.conversation.id
    ) + "/delete-from-path/" + encodeURIComponent(turnId);
    return conversationClientForkMutation(
      owner,
      state,
      url,
      conversationClientForkBody(state, operationId),
      "delete"
    );
  });
}

function conversationClientRetryAssistantFork(owner, input) {
  var operationId = conversationClientOperationId(
    input && input.operationId
  );
  return conversationClientEnqueue(owner, function () {
    var state = conversationClientMutableState(owner.readState());
    var turnId = conversationClientIdentifier(
      input.assistantTurnId, "assistant turn id"
    );
    var body = conversationClientForkBody(
      state, operationId
    );
    body.model_id = conversationClientIdentifier(
      input.modelId, "model id"
    );
    body.input_mode = conversationClientInputMode(input.inputMode);
    var url = conversationClientBranchesUrl(
      state.conversation.id
    ) + "/retry-assistant/" + encodeURIComponent(turnId);
    return conversationClientForkMutation(
      owner, state, url, body, "retry"
    );
  });
}

function conversationClientForkBody(state, operationId) {
  return {
    operation_id: conversationClientOperationId(operationId),
    branch_id: state.selectedBranchId,
    branch_revision: state.conversation.branch_revision,
    catalog_revision: state.conversation.catalog_revision,
  };
}

function conversationClientForkMutation(
  owner, state, url, body, kind
) {
  return conversationClientMutation(
    owner.request, url, "POST", body, state
  ).then(function (result) {
    var conversation = conversationClientForkResult(result, kind);
    var manifest = conversationStateManifest(conversation);
    return conversationClientRestore(
      owner,
      manifest.id,
      manifest.branch_id,
      "forked"
    ).then(function () {
      return result;
    }).catch(function (error) {
      if (error && typeof error === "object") {
        error.conversationId = manifest.id;
        error.branchId = manifest.branch_id;
      }
      throw error;
    });
  });
}

function conversationClientAssistantBody(state, input) {
  return {
    branch_id: state.selectedBranchId,
    branch_revision: conversationClientPositive(
      input.branchRevision, "captured branch revision"
    ),
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
    conversationClientSelectedBranch(state, input);
    var assistantId = conversationClientIdentifier(
      input.assistantTurnId, "assistant turn id"
    );
    var assistantIndex = conversationClientPositive(
      input.assistantTurnIndex, "assistant turn index"
    );
    var assistantVersion = conversationClientPositive(
      input.assistantTurnVersion, "assistant turn version"
    );
    conversationClientLinkTail({
      state: state,
      assistantId: assistantId,
      assistantIndex: assistantIndex,
      assistantVersion: assistantVersion,
    });
    var body = {
      branch_id: state.selectedBranchId,
      branch_revision: state.conversation.branch_revision,
      assistant_turn_index: assistantIndex,
      assistant_turn_version: assistantVersion,
      run_id: conversationClientIdentifier(input.runId, "run id"),
      run_revision: conversationClientNonnegative(
        input.runRevision, "run revision"
      ),
    };
    var url = conversationClientTurnUrl(
      state.conversation.id, assistantId
    ) + "/run";
    return conversationClientMutation(
      owner.request, url, "PUT", body, state
    ).then(function (result) {
      conversationClientTailResult(result);
      return conversationClientApplyMutation(
        owner,
        state.conversation,
        result,
        {
          type: "run_linked",
          conversation: result.conversation,
          turn: result.turn,
        }
      );
    });
  });
}

function conversationClientApplyMutation(
  owner, previous, result, action
) {
  owner.applyAction(action);
  var next = conversationStateManifest(result.conversation);
  if (!conversationClientCatalogChanged(previous, next)) {
    return Promise.resolve(result);
  }
  var state = conversationClientMutableState(owner.readState());
  var pageCount = Math.max(
    1,
    Math.ceil(state.turns.length / CONVERSATION_PAGE_SIZE)
  );
  return conversationClientCatalogPages(
    owner.request, next, pageCount
  ).then(function (pages) {
    owner.applyAction({
      type: "catalog_refreshed",
      conversation: result.conversation,
      pages: pages,
    });
    return result;
  });
}

function conversationClientCatalogChanged(previous, next) {
  return (
    previous.catalog_revision !== next.catalog_revision
    || previous.default_branch_id !== next.default_branch_id
  );
}

async function conversationClientCatalogPages(
  request, manifest, pageCount
) {
  if (
    !Number.isInteger(pageCount)
    || pageCount < 1
    || pageCount > CONVERSATION_PAGES_MAX
  ) {
    throw new RangeError("Catalog refresh page count is invalid");
  }
  var pages = [];
  var before = null;
  for (var index = 0; index < pageCount; index++) {
    var raw = await conversationClientPage(
      request, manifest.id, manifest.branch_id, before
    );
    var page = conversationStatePage(
      raw, manifest.id, manifest.branch_id
    );
    if (!conversationClientPageIsCurrent(page, manifest)) {
      throw new Error(
        "Conversation changed during catalog refresh"
      );
    }
    pages.push(raw);
    if (!page.hasMore) {
      break;
    }
    before = page.nextBefore;
  }
  return pages;
}

function conversationClientReloadConflict(owner, error) {
  var state = owner.readState();
  var conversationId = error.conversationId;
  var branchId = error.branchId;
  if (!conversationId && state.conversation) {
    conversationId = state.conversation.id;
  }
  if (!branchId && state.conversation) {
    if (state.conversation.id === conversationId) {
      branchId = state.selectedBranchId;
    }
  }
  if (!conversationId) {
    return Promise.resolve();
  }
  return conversationClientRestore(
    owner, conversationId, branchId || null
  )
    .then(function () {
      error.conversationReloaded = true;
      error.conversationReplayable =
        conversationClientIsTransportLoss(error);
      if (conversationClientIsConflict(error) && owner.onConflict) {
        error.conversationConflict = true;
        owner.onConflict(error);
      }
    });
}

function conversationClientMutation(
  request, url, method, body, state
) {
  return conversationClientJson(request, url, {
    method: method,
    headers: conversationClientHeaders(),
    body: JSON.stringify(body),
  }).catch(function (error) {
    conversationClientBindMutationError(error, state);
    throw error;
  });
}

function conversationClientMetadata(
  request, conversationId, selectedBranchId
) {
  var url = "/api/conversations/"
    + encodeURIComponent(conversationId) + "/metadata";
  if (selectedBranchId !== null) {
    url += "?branch_id=" + encodeURIComponent(selectedBranchId);
  }
  return conversationClientJson(request, url, {
    method: "GET",
  }).then(conversationClientManifest);
}

function conversationClientPage(
  request, conversationId, selectedBranchId, before
) {
  var url = conversationClientTurnsUrl(conversationId)
    + "?limit=" + CONVERSATION_PAGE_SIZE
    + "&branch_id=" + encodeURIComponent(selectedBranchId);
  if (before !== null) {
    url += "&before=" + encodeURIComponent(before);
  }
  return conversationClientJson(request, url, {
    method: "GET",
  }).then(function (body) {
    conversationClientPageShape(
      body, conversationId, selectedBranchId
    );
    return body;
  });
}

function conversationClientJson(request, url, init) {
  var responseStatus = 0;
  var responseReceived = false;
  return Promise.resolve().then(function () {
    return request(url, init);
  }).then(function (response) {
    responseReceived = true;
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
    if (
      error instanceof Error
      || (
        error
        && typeof error === "object"
        && typeof error.message === "string"
      )
    ) {
      if (!responseReceived && error.status === undefined) {
        error.conversationTransportLoss = true;
      }
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
  error.branchId =
    body && typeof body.branch_id === "string"
      ? body.branch_id
      : "";
  error.branchRevision =
    body && Number.isInteger(body.branch_revision)
      ? body.branch_revision
      : null;
  error.catalogRevision =
    body && Number.isInteger(body.catalog_revision)
      ? body.catalog_revision
      : null;
  return error;
}

function conversationClientIsConflict(error) {
  if (!error || error.status !== 409) {
    return false;
  }
  return [
    "revision_conflict",
    "branch_revision_conflict",
    "catalog_revision_conflict",
    "state_conflict",
  ].indexOf(error.reason) !== -1;
}

function conversationClientShouldReload(error) {
  if (conversationClientIsConflict(error)) {
    return true;
  }
  return conversationClientIsTransportLoss(error);
}

function conversationClientIsTransportLoss(error) {
  return Boolean(
    error
    && (
      error.conversationTransportLoss === true
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

function conversationClientPageShape(
  body, conversationId, selectedBranchId
) {
  conversationStatePage(
    body, conversationId, selectedBranchId
  );
}

function conversationClientAppendResult(result) {
  var raw = conversationClientManifest(result);
  var manifest = conversationStateManifest(raw);
  conversationStateTurn(
    result.user_turn,
    manifest.branch_id,
    manifest.schema_version
  );
  conversationStateTurn(
    result.assistant_turn,
    manifest.branch_id,
    manifest.schema_version
  );
}

function conversationClientTailResult(result) {
  var raw = conversationClientManifest(result);
  var manifest = conversationStateManifest(raw);
  conversationStateTurn(
    result.turn,
    manifest.branch_id,
    manifest.schema_version
  );
}

function conversationClientForkResult(result, kind) {
  var raw = conversationClientManifest(result);
  var manifest = conversationStateManifest(raw);
  if (!result.catalog || typeof result.catalog !== "object") {
    throw new TypeError("Conversation fork catalog is invalid");
  }
  var catalogRevision = conversationClientNonnegative(
    result.catalog.catalog_revision,
    "fork catalog revision"
  );
  if (catalogRevision !== manifest.catalog_revision) {
    throw new Error("Fork catalog and conversation revisions differ");
  }
  conversationClientForkBranch(result.branch, manifest);
  if (kind === "edit") {
    conversationStateTurn(
      result.user_turn,
      manifest.branch_id,
      manifest.schema_version
    );
    conversationStateTurn(
      result.assistant_turn,
      manifest.branch_id,
      manifest.schema_version
    );
  } else if (kind === "retry") {
    conversationStateTurn(
      result.assistant_turn,
      manifest.branch_id,
      manifest.schema_version
    );
  } else if (kind !== "delete") {
    throw new Error("Unknown conversation fork kind");
  }
  return raw;
}

function conversationClientForkBranch(branch, manifest) {
  if (!branch || typeof branch !== "object") {
    throw new TypeError("Conversation fork branch is invalid");
  }
  if (branch.branch_id !== manifest.branch_id) {
    throw new Error("Fork branch and conversation ids differ");
  }
  if (branch.branch_revision !== manifest.branch_revision) {
    throw new Error("Fork branch and conversation revisions differ");
  }
}

function conversationClientPageIsCurrent(page, manifest) {
  return (
    page.schemaVersion === manifest.schema_version
    && page.branchId === manifest.branch_id
    && page.branchRevision === manifest.branch_revision
    && page.catalogRevision === manifest.catalog_revision
    && page.defaultBranchId === manifest.default_branch_id
  );
}

function conversationClientBranchesShape(body, conversationId) {
  if (body.conversation_id !== conversationId) {
    throw new Error("Branch catalog belongs to another conversation");
  }
  conversationClientNonnegative(
    body.catalog_revision, "catalog revision"
  );
  conversationClientBranchId(body.default_branch_id);
  if (!Array.isArray(body.branch_ids)) {
    throw new TypeError("Branch catalog ids must be a list");
  }
  if (!Array.isArray(body.branches)) {
    throw new TypeError("Branch catalog entries must be a list");
  }
  if (
    body.branch_ids.length < 1
    || body.branch_ids.length > CONVERSATION_BRANCHES_MAX
    || body.branches.length !== body.branch_ids.length
  ) {
    throw new RangeError("Branch catalog count is out of bounds");
  }
  var seen = {};
  for (var index = 0; index < body.branch_ids.length; index++) {
    var branchId = conversationClientBranchId(
      body.branch_ids[index]
    );
    if (seen[branchId]) {
      throw new Error("Branch catalog contains a duplicate");
    }
    seen[branchId] = true;
    conversationClientBranchShape(body.branches[index], branchId);
  }
  if (!seen[body.default_branch_id]) {
    throw new Error("Branch catalog default is not a member");
  }
}

function conversationClientBranchShape(branch, expectedId) {
  if (!branch || typeof branch !== "object") {
    throw new TypeError("Branch catalog entry must be an object");
  }
  if (branch.branch_id !== expectedId) {
    throw new Error(
      "Branch catalog entry order differs from its ids"
    );
  }
  conversationClientPositive(
    branch.branch_revision, "branch revision"
  );
}

function conversationClientBindMutationError(error, state) {
  if (!error || typeof error !== "object") {
    return;
  }
  if (error.status === 409) {
    error.conversationConflict = true;
    error.conversationReplayable = false;
  }
  if (!error.conversationId) {
    error.conversationId = state.conversation.id;
  }
  if (!error.branchId) {
    error.branchId = state.selectedBranchId;
  }
}

function conversationClientMutableState(state) {
  conversationStateAssert(state);
  if (!state.conversation) {
    throw new Error("No active conversation");
  }
  return state;
}

function conversationClientSelectedBranch(state, input) {
  if (!input || input.branchId === undefined) {
    return state.selectedBranchId;
  }
  var branchId = conversationClientBranchId(input.branchId);
  if (branchId !== state.selectedBranchId) {
    throw new Error("Mutation does not target the selected branch");
  }
  return branchId;
}

function conversationClientSelectedRevision(state, input) {
  var revision = conversationClientPositive(
    input && input.branchRevision,
    "captured branch revision"
  );
  if (revision !== state.conversation.branch_revision) {
    throw new Error(
      "Assistant completion no longer targets the selected revision"
    );
  }
  return revision;
}

function conversationClientLinkTail(options) {
  var assistant = conversationStateTailAssistant(options.state);
  if (
    assistant === null
    || assistant.turn_id !== options.assistantId
    || assistant.index !== options.assistantIndex
    || assistant.version !== options.assistantVersion
  ) {
    throw new Error(
      "Run link no longer targets the exact selected assistant"
    );
  }
}

function conversationClientTurnsUrl(conversationId) {
  return "/api/conversations/"
    + encodeURIComponent(conversationId) + "/turns";
}

function conversationClientBranchesUrl(conversationId) {
  return "/api/conversations/"
    + encodeURIComponent(conversationId) + "/branches";
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

function conversationClientBranchId(value) {
  var branchId = conversationClientIdentifier(value, "branch id");
  if (!/^b_[0-9a-f]{32}$/.test(branchId)) {
    throw new TypeError("branch id has an invalid format");
  }
  return branchId;
}

function conversationClientOperationId(value) {
  if (
    typeof value !== "string"
    || !/^[0-9a-f]{32}$/.test(value)
  ) {
    throw new TypeError(
      "operation id must be 32 lowercase hex characters"
    );
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
