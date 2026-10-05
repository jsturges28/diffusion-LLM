// Accessible actions for durable conversation cards.
//
// The view owns card structure. This controller supplies bounded
// decorations and owns every interaction that those decorations
// start: clipboard fallback, inline edit state, native confirmations,
// branch dispatch, focus restoration, and live feedback.

"use strict";

var CONVERSATION_ACTION_TEXT_MAX = 1000000;
var CONVERSATION_ACTION_SUMMARY_MAX = 2000;
var CONVERSATION_CLIPBOARD_FALLBACK_MAX = 1000000;
var CONVERSATION_ACTION_FEEDBACK_MS = 4000;
var CONVERSATION_COPY_FEEDBACK_MS = 1200;
var CONVERSATION_ACTION_EPOCH_MAX = 1000000;

function conversationActionsCreate(options) {
  var view = conversationActionViewCreate();
  var owner = {
    view: view,
    root: view.root(),
    deleteDialog: view.dialog("delete"),
    retryDialog: view.dialog("retry"),
    readState: conversationActionsCallback(options, "readState"),
    readConfiguration: conversationActionsCallback(
      options, "readConfiguration"
    ),
    readBlockReason: conversationActionsCallback(
      options, "readBlockReason"
    ),
    requestRender: conversationActionsCallback(
      options, "requestRender"
    ),
    onStateChanged: conversationActionsCallback(
      options, "onStateChanged"
    ),
    selectBranch: conversationActionsCallback(
      options, "selectBranch"
    ),
    editUser: conversationActionsCallback(options, "editUser"),
    deleteUser: conversationActionsCallback(
      options, "deleteUser"
    ),
    retryAssistant: conversationActionsCallback(
      options, "retryAssistant"
    ),
    createOperationId: conversationActionsCallback(
      options, "createOperationId"
    ),
    edit: null,
    confirmation: null,
    pendingAction: false,
    feedbackTimer: null,
    copyFeedbackTimers: {},
    copyEpoch: 0,
    wired: false,
  };

  return Object.freeze({
    wire: function () {
      conversationActionsWire(owner);
    },
    decorateTurn: function (article, turn, branchPoints) {
      var edit = owner.edit !== null
        && owner.edit.turnId === turn.turn_id
        ? owner.edit
        : null;
      owner.view.decorateTurn({
        article: article,
        turn: turn,
        branchPoints: branchPoints,
        edit: edit,
        assistantComplete:
          conversationActionsAssistantComplete(owner, turn),
        textMax: CONVERSATION_ACTION_TEXT_MAX,
      });
    },
    decorateActive: function (
      turn, branchPoints, workspaceVisible
    ) {
      owner.view.decorateActive({
        turn: turn,
        branchPoints: branchPoints,
        workspaceVisible: workspaceVisible,
        assistantComplete: turn !== null
          && conversationActionsAssistantComplete(owner, turn),
      });
    },
    deletionMarker: function (point) {
      return owner.view.deletionMarker(point);
    },
    reconcile: function () {
      conversationActionsReconcile(owner);
    },
    blocking: function () {
      return conversationActionsLocallyBlocked(owner);
    },
    report: function (message, danger) {
      conversationActionsReport(owner, message, danger === true);
    },
    closeAll: function () {
      conversationActionsCloseAll(owner);
    },
  });
}

function conversationActionsWire(owner) {
  if (owner.wired) {
    return;
  }
  owner.wired = true;
  owner.root.addEventListener("click", function (event) {
    conversationActionsClick(owner, event);
  });
  owner.root.addEventListener("input", function (event) {
    conversationActionsInput(owner, event);
  });
  owner.root.addEventListener("keydown", function (event) {
    conversationActionsKeydown(owner, event);
  });
  conversationActionsWireDialog(owner, owner.deleteDialog);
  conversationActionsWireDialog(owner, owner.retryDialog);
}

function conversationActionsWireDialog(owner, dialog) {
  dialog.addEventListener("click", function (event) {
    conversationActionsDialogClick(owner, dialog, event);
  });
  dialog.addEventListener("cancel", function (event) {
    if (owner.pendingAction) {
      event.preventDefault();
    }
  });
  dialog.addEventListener("close", function () {
    conversationActionsDialogClosed(owner, dialog);
  });
}

function conversationActionsClick(owner, event) {
  var target = event.target;
  var button = target && target.closest
    ? target.closest("[data-conversation-action]")
    : null;
  if (!button || button.disabled) {
    return;
  }
  var action = button.getAttribute("data-conversation-action");
  var turnId = button.getAttribute("data-turn-id");
  if (action === "copy") {
    conversationActionsCopy(owner, turnId, button);
  } else if (action === "edit") {
    conversationActionsBeginEdit(owner, turnId);
  } else if (action === "delete") {
    conversationActionsOpenDelete(owner, turnId, button);
  } else if (action === "retry") {
    conversationActionsOpenRetry(owner, turnId, button);
  } else if (action === "edit-cancel") {
    conversationActionsCancelEdit(owner);
  } else if (action === "edit-save") {
    conversationActionsSaveEdit(owner);
  } else if (
    action === "branch-previous"
    || action === "branch-next"
  ) {
    conversationActionsSelectBranch(owner, button);
  }
}

function conversationActionsInput(owner, event) {
  if (
    owner.edit === null
    || owner.edit.saving
    || owner.edit.submittedDraft !== null
  ) {
    return;
  }
  var input = event.target;
  if (!input || !input.hasAttribute(
    "data-conversation-edit-input"
  )) {
    return;
  }
  if (
    input.getAttribute("data-conversation-edit-input")
    !== owner.edit.turnId
  ) {
    return;
  }
  owner.edit.draft = String(input.value).slice(
    0, CONVERSATION_ACTION_TEXT_MAX
  );
}

function conversationActionsKeydown(owner, event) {
  if (event.key !== "Escape" || owner.edit === null) {
    return;
  }
  var input = event.target;
  if (!input || !input.hasAttribute(
    "data-conversation-edit-input"
  )) {
    return;
  }
  event.preventDefault();
  conversationActionsCancelEdit(owner);
}

function conversationActionsDialogClick(owner, dialog, event) {
  if (event.target === dialog) {
    if (!owner.pendingAction) {
      dialog.close("cancel");
    }
    return;
  }
  var target = event.target;
  var button = target && target.closest
    ? target.closest("[data-conversation-confirmation]")
    : null;
  if (!button || button.disabled) {
    return;
  }
  var action = button.getAttribute(
    "data-conversation-confirmation"
  );
  if (action === "cancel") {
    dialog.close("cancel");
  } else if (action === "confirm-delete") {
    conversationActionsConfirmDelete(owner);
  } else if (action === "confirm-retry") {
    conversationActionsConfirmRetry(owner);
  }
}

function conversationActionsAssistantComplete(owner, turn) {
  var state = owner.readState();
  var conversation = state && state.conversation;
  if (!conversation || turn.role !== "assistant") {
    return false;
  }
  if (conversation.pending_assistant_id === turn.turn_id) {
    return false;
  }
  return turn.version > 1;
}

function conversationActionsBeginEdit(owner, turnId) {
  if (conversationActionsMutationBlocked(owner)) {
    return;
  }
  var turn = conversationActionsFindTurn(owner, turnId);
  if (turn === null || turn.role !== "user") {
    conversationActionsReport(
      owner, "That user message is no longer available.", true
    );
    return;
  }
  var operationId = conversationActionsNewOperation(
    owner, "Edit cannot start"
  );
  var configuration = conversationActionsReadConfiguration(
    owner, "Edit cannot start"
  );
  if (operationId === null || configuration === null) {
    return;
  }
  owner.edit = {
    turnId: turn.turn_id,
    original: turn.text,
    draft: turn.text,
    submittedDraft: null,
    saving: false,
    operationId: operationId,
    configuration: configuration,
    request: null,
    replayedAfterReload: false,
  };
  conversationActionsStateChanged(owner);
  owner.requestRender();
  owner.view.focusEdit(turn.turn_id);
}

function conversationActionsCancelEdit(owner) {
  if (owner.edit === null || owner.edit.saving) {
    return;
  }
  var turnId = owner.edit.turnId;
  owner.edit = null;
  conversationActionsStateChanged(owner);
  owner.requestRender();
  owner.view.focusEditTrigger(turnId);
}

function conversationActionsSaveEdit(owner) {
  if (owner.edit === null || owner.edit.saving) {
    return;
  }
  var draft = owner.edit.submittedDraft === null
    ? owner.edit.draft
    : owner.edit.submittedDraft;
  if (draft.trim() === "") {
    conversationActionsReport(
      owner, "The edited message cannot be blank.", true
    );
    owner.view.focusEdit(owner.edit.turnId);
    return;
  }
  if (draft.length > CONVERSATION_ACTION_TEXT_MAX) {
    conversationActionsReport(
      owner, "The edited message is too long.", true
    );
    owner.view.focusEdit(owner.edit.turnId);
    return;
  }
  if (conversationActionsExternalBlocked(owner)) {
    return;
  }
  if (owner.edit.request === null) {
    owner.edit.submittedDraft = draft;
    owner.edit.request = Object.freeze({
      operationId: owner.edit.operationId,
      userTurnId: owner.edit.turnId,
      text: draft,
      modelId: owner.edit.configuration.modelId,
      inputMode: owner.edit.configuration.inputMode,
      metadata: Object.freeze({}),
      configuration: owner.edit.configuration,
    });
  }
  conversationActionsRunEdit(owner, owner.edit);
}

function conversationActionsRunEdit(owner, edit) {
  if (edit.request === null) {
    throw new Error("Edit request was not frozen");
  }
  edit.saving = true;
  owner.pendingAction = true;
  conversationActionsStateChanged(owner);
  owner.requestRender();
  owner.view.focusEditPending(edit.turnId);
  Promise.resolve().then(function () {
    return owner.editUser(edit.request);
  }).then(function (outcome) {
    conversationActionsEditSucceeded(
      owner, outcome, edit.turnId
    );
  }).catch(function (error) {
    if (conversationActionsReplayAfterReload(edit, error)) {
      conversationActionsRunEdit(owner, edit);
      return;
    }
    conversationActionsEditFailed(owner, error, edit.turnId);
  });
}

function conversationActionsEditSucceeded(owner, outcome, turnId) {
  if (outcome === false) {
    conversationActionsEditFailed(
      owner, new Error("conversation action is unavailable"), turnId
    );
    return;
  }
  owner.pendingAction = false;
  owner.edit = null;
  conversationActionsStateChanged(owner);
  var result = conversationActionsOutcomeResult(outcome);
  conversationActionsOutcomeFeedback(
    owner,
    outcome,
    "Created an edited path and started regeneration."
  );
  var nextId = result && result.user_turn
    ? result.user_turn.turn_id
    : null;
  if (nextId && owner.view.focusTurn(nextId)) {
    return;
  }
  if (
    !conversationActionsOutcomeLaunched(outcome)
    || !owner.view.focusActive()
  ) {
    owner.view.focusComposer();
  }
}

function conversationActionsEditFailed(owner, error, turnId) {
  owner.pendingAction = false;
  if (conversationActionsExplicitConflict(error)) {
    owner.edit = null;
    conversationActionsStateChanged(owner);
    owner.requestRender();
    conversationActionsReportError(
      owner, "Edit and regenerate failed", error
    );
    if (!owner.view.focusEditTrigger(turnId)) {
      owner.view.focusBranchFallback("");
    }
    return;
  }
  if (owner.edit !== null && owner.edit.turnId === turnId) {
    owner.edit.saving = false;
  }
  conversationActionsStateChanged(owner);
  owner.requestRender();
  conversationActionsReportError(
    owner, "Edit and regenerate failed", error
  );
  if (
    !owner.view.focusEdit(turnId)
    && !owner.view.focusEditTrigger(turnId)
  ) {
    owner.view.focusBranchFallback("");
  }
}

function conversationActionsOpenDelete(owner, turnId, trigger) {
  if (conversationActionsMutationBlocked(owner)) {
    return;
  }
  var turn = conversationActionsFindTurn(owner, turnId);
  if (turn === null || turn.role !== "user") {
    conversationActionsReport(
      owner, "That user message is no longer available.", true
    );
    return;
  }
  var operationId = conversationActionsNewOperation(
    owner, "Delete cannot start"
  );
  if (operationId === null) {
    return;
  }
  var state = owner.readState();
  var count = state.conversation.turn_count - turn.index + 1;
  if (!Number.isInteger(count) || count < 1) {
    throw new Error("Delete count is outside the selected path");
  }
  owner.view.setDialogMessage(
    "delete",
    count + " selected and later dependent turn"
    + (count === 1 ? "" : "s")
    + " will leave this path. The original path remains available."
  );
  owner.view.clearDialogStatus("delete");
  owner.confirmation = {
    kind: "delete",
    turnId: turn.turn_id,
    turnIndex: turn.index,
    count: count,
    trigger: trigger,
    dialog: owner.deleteDialog,
    operationId: operationId,
    request: Object.freeze({
      operationId: operationId,
      userTurnId: turn.turn_id,
    }),
    replayedAfterReload: false,
    restoreFocus: true,
  };
  conversationActionsStateChanged(owner);
  owner.deleteDialog.showModal();
  owner.view.focusDialogCancel("delete");
}

function conversationActionsOpenRetry(owner, turnId, trigger) {
  if (conversationActionsMutationBlocked(owner)) {
    return;
  }
  var turn = conversationActionsFindTurn(owner, turnId);
  if (
    turn === null
    || !conversationActionsAssistantComplete(owner, turn)
  ) {
    conversationActionsReport(
      owner, "That assistant response cannot be retried.", true
    );
    return;
  }
  var operationId = conversationActionsNewOperation(
    owner, "Retry cannot start"
  );
  var configuration = conversationActionsReadConfiguration(
    owner, "Retry cannot start"
  );
  if (operationId === null || configuration === null) {
    return;
  }
  owner.view.setDialogMessage(
    "retry",
    "Retry with " + conversationActionsModelLabel(configuration)
    + ". Input mode: " + configuration.inputMode + ". "
    + "Current Run settings: "
    + configuration.settingsSummary
    + ". This creates an alternate path."
  );
  owner.view.clearDialogStatus("retry");
  owner.confirmation = {
    kind: "retry",
    turnId: turn.turn_id,
    turnIndex: turn.index,
    trigger: trigger,
    dialog: owner.retryDialog,
    operationId: operationId,
    configuration: configuration,
    request: Object.freeze({
      operationId: operationId,
      assistantTurnId: turn.turn_id,
      modelId: configuration.modelId,
      inputMode: configuration.inputMode,
      configuration: configuration,
    }),
    replayedAfterReload: false,
    restoreFocus: true,
  };
  conversationActionsStateChanged(owner);
  owner.retryDialog.showModal();
  owner.view.focusDialogCancel("retry");
}

function conversationActionsConfirmDelete(owner) {
  var confirmation = owner.confirmation;
  if (
    confirmation === null
    || confirmation.kind !== "delete"
    || owner.pendingAction
  ) {
    return;
  }
  owner.view.clearDialogStatus("delete");
  owner.pendingAction = true;
  conversationActionsSetDialogPending(
    owner, confirmation, true
  );
  conversationActionsStateChanged(owner);
  conversationActionsRunDelete(owner, confirmation);
}

function conversationActionsRunDelete(owner, confirmation) {
  Promise.resolve().then(function () {
    return owner.deleteUser(confirmation.request);
  }).then(function (outcome) {
    conversationActionsDeleteSucceeded(
      owner, confirmation, outcome
    );
  }).catch(function (error) {
    if (
      conversationActionsReplayAfterReload(confirmation, error)
    ) {
      conversationActionsRunDelete(owner, confirmation);
      return;
    }
    conversationActionsConfirmationFailed(
      owner, confirmation, "Delete from path failed", error
    );
  });
}

function conversationActionsDeleteSucceeded(
  owner, confirmation, outcome
) {
  if (outcome === false) {
    conversationActionsConfirmationFailed(
      owner,
      confirmation,
      "Delete from path failed",
      new Error("conversation action is unavailable")
    );
    return;
  }
  var result = conversationActionsOutcomeResult(outcome);
  owner.pendingAction = false;
  confirmation.restoreFocus = false;
  owner.confirmation = null;
  conversationActionsSetDialogPending(
    owner, confirmation, false
  );
  owner.view.clearDialogStatus("delete");
  confirmation.dialog.close("confirm");
  conversationActionsStateChanged(owner);
  var removed = result && result.removed_turn_count;
  var count = Number.isInteger(removed)
    ? removed
    : confirmation.count;
  conversationActionsReport(
    owner,
    "Deleted " + count + " turn"
      + (count === 1 ? "" : "s")
      + " from this path. The original path remains available.",
    false
  );
  var branchId = result && result.branch
    ? result.branch.branch_id
    : null;
  if (!owner.view.focusDeletion(branchId)) {
    owner.view.focusRoot();
  }
}

function conversationActionsConfirmRetry(owner) {
  var confirmation = owner.confirmation;
  if (
    confirmation === null
    || confirmation.kind !== "retry"
    || owner.pendingAction
  ) {
    return;
  }
  owner.view.clearDialogStatus("retry");
  owner.pendingAction = true;
  conversationActionsSetDialogPending(
    owner, confirmation, true
  );
  conversationActionsStateChanged(owner);
  conversationActionsRunRetry(owner, confirmation);
}

function conversationActionsRunRetry(owner, confirmation) {
  Promise.resolve().then(function () {
    return owner.retryAssistant(confirmation.request);
  }).then(function (outcome) {
    conversationActionsRetrySucceeded(
      owner, confirmation, outcome
    );
  }).catch(function (error) {
    if (
      conversationActionsReplayAfterReload(confirmation, error)
    ) {
      conversationActionsRunRetry(owner, confirmation);
      return;
    }
    conversationActionsConfirmationFailed(
      owner, confirmation, "Retry failed", error
    );
  });
}

function conversationActionsRetrySucceeded(
  owner, confirmation, outcome
) {
  if (outcome === false) {
    conversationActionsConfirmationFailed(
      owner,
      confirmation,
      "Retry failed",
      new Error("conversation action is unavailable")
    );
    return;
  }
  owner.pendingAction = false;
  confirmation.restoreFocus = false;
  owner.confirmation = null;
  conversationActionsSetDialogPending(
    owner, confirmation, false
  );
  owner.view.clearDialogStatus("retry");
  confirmation.dialog.close("confirm");
  conversationActionsStateChanged(owner);
  conversationActionsOutcomeFeedback(
    owner,
    outcome,
    "Created an alternate path and started retry generation."
  );
  if (conversationActionsOutcomeLaunched(outcome)) {
    if (!owner.view.focusActive()) {
      owner.view.focusComposer();
    }
  } else {
    owner.view.focusComposer();
  }
}

function conversationActionsConfirmationFailed(
  owner, confirmation, prefix, error
) {
  owner.pendingAction = false;
  conversationActionsSetDialogPending(
    owner, confirmation, false
  );
  if (conversationActionsExplicitConflict(error)) {
    confirmation.restoreFocus = false;
    owner.confirmation = null;
    if (confirmation.dialog.open) {
      confirmation.dialog.close("conflict");
    }
    conversationActionsStateChanged(owner);
    conversationActionsReportError(owner, prefix, error);
    if (!owner.view.focusTrigger(confirmation)) {
      owner.view.focusBranchFallback("");
    }
    return;
  }
  conversationActionsStateChanged(owner);
  conversationActionsReportError(owner, prefix, error);
  if (
    !confirmation.dialog.open
    && !owner.view.focusTrigger(confirmation)
  ) {
    owner.view.focusBranchFallback("");
  }
}

function conversationActionsSetDialogPending(
  owner, confirmation, pending
) {
  owner.view.setDialogPending(confirmation.kind, pending);
}

function conversationActionsDialogClosed(owner, dialog) {
  var confirmation = owner.confirmation;
  if (
    confirmation === null
    || confirmation.dialog !== dialog
  ) {
    return;
  }
  if (owner.pendingAction) {
    return;
  }
  owner.confirmation = null;
  owner.view.clearDialogStatus(confirmation.kind);
  conversationActionsStateChanged(owner);
  if (confirmation.restoreFocus) {
    owner.view.focusTrigger(confirmation);
  }
}

function conversationActionsSelectBranch(owner, button) {
  if (conversationActionsMutationBlocked(owner)) {
    return;
  }
  var branchId = button.getAttribute("data-branch-id");
  var focusKey = button.getAttribute(
    "data-conversation-focus-key"
  );
  if (!branchId || !focusKey) {
    throw new Error("Branch action is missing its target");
  }
  owner.pendingAction = true;
  conversationActionsStateChanged(owner);
  Promise.resolve().then(function () {
    return owner.selectBranch(branchId);
  }).then(function (selected) {
    owner.pendingAction = false;
    conversationActionsStateChanged(owner);
    if (selected === false) {
      conversationActionsReport(
        owner, "Path selection is currently unavailable.", true
      );
      owner.view.focusBranchFallback(focusKey);
      return;
    }
    conversationActionsReport(
      owner, "Selected alternate path.", false
    );
    owner.view.focusBranchFallback(focusKey);
  }).catch(function (error) {
    owner.pendingAction = false;
    conversationActionsStateChanged(owner);
    conversationActionsReportError(
      owner, "Path selection failed", error
    );
    owner.view.focusBranchFallback(focusKey);
  });
}

function conversationActionsCopy(owner, turnId, button) {
  var turn = conversationActionsFindTurn(owner, turnId);
  if (turn === null) {
    conversationActionsReport(
      owner, "That message is no longer available.", true
    );
    return;
  }
  conversationActionsSetCopyFeedback(
    owner, turnId, "idle", ""
  );
  owner.copyEpoch = (
    owner.copyEpoch % CONVERSATION_ACTION_EPOCH_MAX
  ) + 1;
  var epoch = owner.copyEpoch;
  var isCurrent = function () {
    return epoch === owner.copyEpoch;
  };
  if (conversationActionsNeedsSynchronousCopy()) {
    try {
      conversationActionsClipboardFallbackSync(
        turn.text, button, isCurrent
      );
      if (epoch === owner.copyEpoch) {
        conversationActionsSetCopyFeedback(
          owner, turnId, "copied", "Copied"
        );
      }
    } catch (_error) {
      if (epoch === owner.copyEpoch) {
        conversationActionsSetCopyFeedback(
          owner, turnId, "error", "Copy failed"
        );
      }
    }
    return;
  }
  conversationActionsWriteClipboard(
    turn.text, button, isCurrent
  )
    .then(function () {
      if (epoch === owner.copyEpoch) {
        conversationActionsSetCopyFeedback(
          owner, turnId, "copied", "Copied"
        );
      }
    })
    .catch(function (_error) {
      if (epoch === owner.copyEpoch) {
        conversationActionsSetCopyFeedback(
          owner, turnId, "error", "Copy failed"
        );
      }
    });
}

function conversationActionsSetCopyFeedback(
  owner, turnId, state, message
) {
  var timer = owner.copyFeedbackTimers[turnId];
  if (timer !== undefined) {
    clearTimeout(timer);
    delete owner.copyFeedbackTimers[turnId];
  }
  owner.view.setCopyFeedback(turnId, state, message);
  if (state === "idle") {
    return;
  }
  var delay = state === "error"
    ? CONVERSATION_ACTION_FEEDBACK_MS
    : CONVERSATION_COPY_FEEDBACK_MS;
  owner.copyFeedbackTimers[turnId] = setTimeout(function () {
    delete owner.copyFeedbackTimers[turnId];
    owner.view.setCopyFeedback(turnId, "idle", "");
  }, delay);
}

// QtWebEngine's Clipboard permission path is the likely native-only
// hazard, not a proven crash cause. Keep its fallback inside the
// original click gesture and never enter that asynchronous API.
function conversationActionsNeedsSynchronousCopy() {
  if (
    typeof window === "object"
    && window !== null
    && "pywebview" in window
  ) {
    return true;
  }
  var userAgent = (
    typeof navigator === "object"
    && navigator !== null
    && typeof navigator.userAgent === "string"
  )
    ? navigator.userAgent
    : "";
  return /\bQtWebEngine\b/i.test(userAgent);
}

function conversationActionsWriteClipboard(
  text, restoreFocus, isCurrent
) {
  var stillCurrent = typeof isCurrent === "function"
    ? isCurrent
    : function () { return true; };
  var clipboard = typeof navigator === "object"
    ? navigator.clipboard
    : null;
  if (clipboard && typeof clipboard.writeText === "function") {
    return Promise.resolve().then(function () {
      return clipboard.writeText(text);
    }).catch(function () {
      if (!stillCurrent()) {
        return false;
      }
      return conversationActionsClipboardFallback(
        text, restoreFocus, stillCurrent
      );
    });
  }
  return conversationActionsClipboardFallback(
    text, restoreFocus, stillCurrent
  );
}

function conversationActionsClipboardFallback(
  text, restoreFocus, isCurrent
) {
  return Promise.resolve().then(function () {
    return conversationActionsClipboardFallbackSync(
      text, restoreFocus, isCurrent
    );
  });
}

function conversationActionsClipboardFallbackSync(
  text, restoreFocus, isCurrent
) {
  if (text.length > CONVERSATION_CLIPBOARD_FALLBACK_MAX) {
    throw new Error("message exceeds the fallback copy bound");
  }
  if (
    typeof isCurrent === "function"
    && !isCurrent()
  ) {
    return false;
  }
  var activeElement = document.activeElement;
  var focusTarget = restoreFocus || activeElement;
  if (typeof document.execCommand !== "function") {
    throw new Error("clipboard access is unavailable");
  }
  var input = document.createElement("textarea");
  input.className = "conversation-clipboard-fallback";
  input.value = text;
  input.readOnly = true;
  input.tabIndex = -1;
  input.setAttribute("aria-hidden", "true");
  document.body.appendChild(input);
  try {
    input.select();
    input.setSelectionRange(0, text.length);
    if (!document.execCommand("copy")) {
      throw new Error("the browser refused the fallback copy");
    }
  } finally {
    input.remove();
    if (
      focusTarget
      && focusTarget.isConnected
      && typeof focusTarget.focus === "function"
    ) {
      focusTarget.focus();
    }
  }
  return true;
}

function conversationActionsMutationBlocked(owner) {
  if (conversationActionsLocallyBlocked(owner)) {
    conversationActionsReport(
      owner, "Finish the open conversation action first.", true
    );
    return true;
  }
  return conversationActionsExternalBlocked(owner);
}

function conversationActionsExternalBlocked(owner) {
  var reason = owner.readBlockReason();
  if (typeof reason !== "string") {
    throw new TypeError(
      "conversation action block reason must be a string"
    );
  }
  if (reason === "") {
    return false;
  }
  conversationActionsReport(owner, reason, true);
  return true;
}

function conversationActionsLocallyBlocked(owner) {
  return Boolean(
    owner.edit !== null
    || owner.confirmation !== null
    || owner.pendingAction
  );
}

function conversationActionsReconcile(owner) {
  var changed = false;
  if (
    owner.edit !== null
    && conversationActionsFindTurn(owner, owner.edit.turnId) === null
    && !owner.pendingAction
  ) {
    owner.edit = null;
    changed = true;
  }
  if (
    owner.confirmation !== null
    && conversationActionsFindTurn(
      owner, owner.confirmation.turnId
    ) === null
    && !owner.pendingAction
  ) {
    var confirmation = owner.confirmation;
    owner.confirmation = null;
    confirmation.restoreFocus = false;
    owner.view.clearDialogStatus(confirmation.kind);
    confirmation.dialog.close("stale");
    changed = true;
  }
  if (changed) {
    conversationActionsStateChanged(owner);
  }
}

function conversationActionsReplayAfterReload(action, error) {
  if (
    !error
    || error.conversationReloaded !== true
    || error.conversationReplayable !== true
    || action.replayedAfterReload
  ) {
    return false;
  }
  action.replayedAfterReload = true;
  return true;
}

function conversationActionsExplicitConflict(error) {
  return Boolean(
    error
    && (
      error.conversationConflict === true
      || error.status === 409
    )
  );
}

function conversationActionsCloseAll(owner) {
  var changed = owner.edit !== null || owner.confirmation !== null;
  owner.edit = null;
  if (owner.confirmation !== null) {
    owner.view.clearDialogStatus(owner.confirmation.kind);
    owner.confirmation.restoreFocus = false;
    owner.confirmation = null;
  }
  var dialogs = [owner.deleteDialog, owner.retryDialog];
  for (var index = 0; index < dialogs.length; index++) {
    if (dialogs[index].open) {
      dialogs[index].close("closed");
    }
  }
  if (changed) {
    conversationActionsStateChanged(owner);
    owner.requestRender();
  }
}

function conversationActionsConfiguration(owner) {
  var raw = owner.readConfiguration();
  if (!raw || typeof raw !== "object") {
    throw new TypeError(
      "conversation action configuration must be an object"
    );
  }
  var fields = [
    raw.modelId,
    raw.modelDisplay,
    raw.inputMode,
    raw.settingsSummary,
  ];
  for (var index = 0; index < fields.length; index++) {
    if (typeof fields[index] !== "string" || fields[index] === "") {
      throw new TypeError(
        "conversation action configuration is incomplete"
      );
    }
  }
  if (raw.settingsSummary.length > CONVERSATION_ACTION_SUMMARY_MAX) {
    throw new RangeError(
      "conversation action settings summary is too long"
    );
  }
  if (typeof raw.experimental !== "boolean") {
    throw new TypeError(
      "conversation action experimental flag is invalid"
    );
  }
  if (
    typeof raw.valid !== "boolean"
    || typeof raw.validationMessage !== "string"
  ) {
    throw new TypeError(
      "conversation action validation snapshot is invalid"
    );
  }
  var parameters = conversationActionsParameters(raw.parameters);
  return Object.freeze({
    modelId: raw.modelId,
    modelDisplay: raw.modelDisplay,
    inputMode: raw.inputMode,
    settingsSummary: raw.settingsSummary,
    parameters: parameters,
    experimental: raw.experimental,
    valid: raw.valid,
    validationMessage: raw.validationMessage,
  });
}

function conversationActionsParameters(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new TypeError(
      "conversation action parameters must be an object"
    );
  }
  var result = {};
  var names = Object.keys(raw);
  for (var index = 0; index < names.length; index++) {
    var value = raw[names[index]];
    var type = typeof value;
    if (
      type !== "string"
      && type !== "number"
      && type !== "boolean"
    ) {
      throw new TypeError(
        "conversation action parameter value is invalid"
      );
    }
    if (type === "number" && !isFinite(value)) {
      throw new TypeError(
        "conversation action parameter number is invalid"
      );
    }
    result[names[index]] = value;
  }
  return Object.freeze(result);
}

function conversationActionsReadConfiguration(owner, prefix) {
  try {
    return conversationActionsConfiguration(owner);
  } catch (error) {
    conversationActionsReportError(owner, prefix, error);
    return null;
  }
}

function conversationActionsNewOperation(owner, prefix) {
  try {
    return conversationActionsOperationId(
      owner.createOperationId()
    );
  } catch (error) {
    conversationActionsReportError(owner, prefix, error);
    return null;
  }
}

function conversationActionsOperationId(value) {
  if (
    typeof value !== "string"
    || !/^[0-9a-f]{32}$/.test(value)
  ) {
    throw new TypeError(
      "secure operation id must be 32 lowercase hex characters"
    );
  }
  return value;
}

function conversationActionsModelLabel(configuration) {
  if (configuration.modelDisplay === configuration.modelId) {
    return configuration.modelDisplay;
  }
  return configuration.modelDisplay
    + " (" + configuration.modelId + ")";
}

function conversationActionsFindTurn(owner, turnId) {
  if (typeof turnId !== "string" || turnId === "") {
    return null;
  }
  var state = owner.readState();
  if (!state || !Array.isArray(state.turns)) {
    throw new TypeError(
      "conversation action state must contain turns"
    );
  }
  for (var index = 0; index < state.turns.length; index++) {
    if (state.turns[index].turn_id === turnId) {
      return state.turns[index];
    }
  }
  return null;
}

function conversationActionsOutcomeResult(outcome) {
  if (
    outcome
    && typeof outcome === "object"
    && outcome.result
    && typeof outcome.result === "object"
  ) {
    return outcome.result;
  }
  return outcome;
}

function conversationActionsOutcomeLaunched(outcome) {
  return !(
    outcome
    && typeof outcome === "object"
    && outcome.launched === false
  );
}

function conversationActionsOutcomeFeedback(
  owner, outcome, fallback
) {
  var feedback =
    outcome
    && typeof outcome === "object"
    && typeof outcome.feedback === "string"
      ? outcome.feedback
      : fallback;
  var danger = Boolean(
    outcome
    && typeof outcome === "object"
    && outcome.launched === false
  );
  conversationActionsReport(owner, feedback, danger);
}

function conversationActionsReport(owner, message, danger) {
  if (typeof message !== "string") {
    throw new TypeError(
      "conversation action feedback must be a string"
    );
  }
  if (owner.feedbackTimer !== null) {
    clearTimeout(owner.feedbackTimer);
    owner.feedbackTimer = null;
  }
  owner.view.setFeedback(message, danger);
  if (message === "") {
    return;
  }
  owner.feedbackTimer = setTimeout(function () {
    owner.view.setFeedback("", false);
    owner.feedbackTimer = null;
  }, CONVERSATION_ACTION_FEEDBACK_MS);
}

function conversationActionsReportError(owner, prefix, error) {
  var detail = error && typeof error.message === "string"
    ? error.message
    : "unknown error";
  var message = prefix + ": " + detail;
  var confirmation = owner.confirmation;
  if (
    confirmation !== null
    && confirmation.dialog.open
  ) {
    owner.view.showDialogStatus(confirmation.kind, message);
    return;
  }
  conversationActionsReport(owner, message, true);
}

function conversationActionsStateChanged(owner) {
  owner.onStateChanged({
    blocking: conversationActionsLocallyBlocked(owner),
    pending: owner.pendingAction,
    editing: owner.edit !== null,
    confirming: owner.confirmation !== null,
  });
}

function conversationActionsCallback(options, name) {
  if (!options || typeof options[name] !== "function") {
    throw new TypeError(
      "conversationActionsCreate needs options." + name
    );
  }
  return options[name];
}
