// Conversation shell geometry and transcript scroll ownership.
//
// The transcript view draws compact turns. This controller owns the
// static rich assistant card, including legacy runs that predate
// conversations, and the few events allowed to move the transcript.

"use strict";

var CONVERSATION_SCROLL_NONE = "none";
var CONVERSATION_SCROLL_ANCHOR = "anchor";
var CONVERSATION_SCROLL_TAIL = "tail";

function conversationShellCreate() {
  var transcript = conversationShellElement(
    "conversation-transcript"
  );
  var activeAssistant = conversationShellElement(
    "active-assistant-card"
  );

  function render(options) {
    conversationShellOptions(options);
    var before = {
      height: transcript.scrollHeight,
      top: transcript.scrollTop,
    };
    var workspaceVisible =
      conversationShellWorkspaceVisible(options);
    activeAssistant.hidden = !workspaceVisible;
    transcript.classList.toggle(
      "has-active-assistant", workspaceVisible
    );
    options.renderTranscript(workspaceVisible);
    conversationShellApplyScroll(
      transcript, options.action, before
    );
  }

  return Object.freeze({
    render: render,
  });
}

function conversationShellWorkspaceVisible(options) {
  if (
    !Number.isInteger(options.runFrameCount)
    || options.runFrameCount < 0
  ) {
    throw new TypeError(
      "Conversation shell needs a valid run frame count"
    );
  }
  if (options.runIdentity === null) {
    return options.runFrameCount > 0;
  }
  return conversationShellIdentityMatches(
    options.runIdentity, options.conversationIdentity
  );
}

function conversationShellIdentityMatches(run, conversation) {
  if (!run || typeof run !== "object") {
    return false;
  }
  if (!conversation || typeof conversation !== "object") {
    return false;
  }
  return (
    run.conversation_id === conversation.conversation_id
    && run.branch_id === conversation.branch_id
    && run.branch_revision === conversation.branch_revision
    && run.assistant_turn_id === conversation.assistant_turn_id
    && run.assistant_turn_index
      === conversation.assistant_turn_index
    && run.assistant_turn_version
      === conversation.assistant_turn_version
    && run.assistant_text === conversation.assistant_text
  );
}

function conversationShellApplyScroll(root, action, before) {
  var intent = conversationShellScrollIntent(action);
  if (intent === CONVERSATION_SCROLL_NONE) {
    return;
  }
  if (intent === CONVERSATION_SCROLL_ANCHOR) {
    var added = root.scrollHeight - before.height;
    root.scrollTop = Math.max(0, before.top + added);
    return;
  }
  if (intent !== CONVERSATION_SCROLL_TAIL) {
    throw new Error("Unknown conversation scroll intent");
  }
  // Wait until the reserved assistant has mounted its rich card.
  // The append and launch complete in one promise turn.
  requestAnimationFrame(function () {
    root.scrollTop = root.scrollHeight;
  });
}

function conversationShellScrollIntent(action) {
  if (!action || typeof action.type !== "string") {
    return CONVERSATION_SCROLL_NONE;
  }
  if (action.type === "older_loaded") {
    return CONVERSATION_SCROLL_ANCHOR;
  }
  if (
    action.type === "appended"
    || action.type === "forked"
  ) {
    return CONVERSATION_SCROLL_TAIL;
  }
  return CONVERSATION_SCROLL_NONE;
}

function conversationShellOptions(options) {
  if (!options || typeof options !== "object") {
    throw new TypeError("Conversation shell needs options");
  }
  if (typeof options.renderTranscript !== "function") {
    throw new TypeError(
      "Conversation shell needs renderTranscript"
    );
  }
}

function conversationShellElement(id) {
  var element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing conversation shell element #" + id);
  }
  return element;
}
