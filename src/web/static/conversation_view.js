// Bounded transcript DOM for the generator.
//
// The active tail assistant is deliberately absent. Its frames,
// overlays, and edit tools occupy the shell's static rich card after
// the compact-turn mount. Frozen assistants are text records with
// honest saved or text-only badges and an Analytics link when one
// exists.

"use strict";

function conversationViewCreate(options) {
  var onLoadOlder = conversationViewCallback(
    options, "onLoadOlder"
  );
  var decorateTurn = conversationViewCallback(
    options, "decorateTurn"
  );
  var decorateActive = conversationViewCallback(
    options, "decorateActive"
  );
  var deletionMarker = conversationViewCallback(
    options, "deletionMarker"
  );
  var root = conversationViewElement("conversation-transcript");
  var empty = conversationViewElement("conversation-empty");
  var status = conversationViewElement("conversation-status");
  var loadOlder = conversationViewElement("btn-load-older");
  var turnsRoot = conversationViewElement("conversation-turns");
  var wired = false;

  function wire() {
    if (wired) {
      return;
    }
    wired = true;
    loadOlder.addEventListener("click", function () {
      if (!loadOlder.disabled) {
        onLoadOlder();
      }
    });
  }

  function render(state, options) {
    conversationStateAssert(state);
    var workspaceVisible = Boolean(
      options && options.workspaceVisible
    );
    var workspaceAssistantTurnId =
      options
      && typeof options.workspaceAssistantTurnId === "string"
        ? options.workspaceAssistantTurnId
        : null;
    var visible = conversationViewTurns(
      state, workspaceAssistantTurnId
    );
    var nodes = conversationViewNodes({
      visible: visible,
      state: state,
      decorateTurn: decorateTurn,
      deletionMarker: deletionMarker,
    });
    turnsRoot.replaceChildren.apply(turnsRoot, nodes);
    var activeTurn = conversationViewTurnById(
      state.turns, workspaceAssistantTurnId
    );
    decorateActive(
      activeTurn,
      activeTurn === null
        ? []
        : conversationViewPointsAt(
          state.branchPoints, activeTurn.index
        ),
      workspaceVisible
    );
    conversationViewEmpty(
      empty, state, visible.length, workspaceVisible
    );
    conversationViewLoadButton(loadOlder, state);
    status.textContent = state.error;
    status.hidden = state.error === "";
    root.classList.toggle(
      "has-turns", visible.length > 0 || workspaceVisible
    );
    if (
      nodes.length
      > CONVERSATION_TURNS_MAX + CONVERSATION_BRANCH_POINTS_MAX
    ) {
      throw new Error("Transcript DOM exceeded its turn bound");
    }
  }

  return Object.freeze({
    wire: wire,
    render: render,
  });
}

function conversationViewNodes(options) {
  var entries = [];
  var turnIndexes = {};
  for (var index = 0; index < options.visible.length; index++) {
    var turn = options.visible[index];
    turnIndexes[turn.index] = true;
    entries.push({
      index: turn.index,
      order: index,
      node: conversationViewTurn(
        turn,
        options.state,
        options.decorateTurn,
        conversationViewPointsAt(
          options.state.branchPoints, turn.index
        )
      ),
    });
  }
  conversationViewDeletionEntries(
    entries,
    turnIndexes,
    options.state.branchPoints,
    options.deletionMarker
  );
  entries.sort(conversationViewEntryOrder);
  return entries.map(function (entry) {
    return entry.node;
  });
}

function conversationViewDeletionEntries(
  entries, turnIndexes, points, deletionMarker
) {
  for (var index = 0; index < points.length; index++) {
    var point = points[index];
    var selectedIsDeleted =
      point.deleted_branch_ids.indexOf(
        point.selected_branch_id
      ) !== -1;
    if (!selectedIsDeleted || turnIndexes[point.turn_index]) {
      continue;
    }
    entries.push({
      index: point.turn_index,
      order: CONVERSATION_TURNS_MAX + index,
      node: deletionMarker(point),
    });
  }
}

function conversationViewEntryOrder(left, right) {
  if (left.index !== right.index) {
    return left.index - right.index;
  }
  return left.order - right.order;
}

function conversationViewPointsAt(points, turnIndex) {
  return points.filter(function (point) {
    return point.turn_index === turnIndex;
  });
}

function conversationViewTurnById(turns, turnId) {
  if (typeof turnId !== "string" || turnId === "") {
    return null;
  }
  for (var index = turns.length - 1; index >= 0; index--) {
    if (turns[index].turn_id === turnId) {
      return turns[index];
    }
  }
  return null;
}

function conversationViewTurns(state, workspaceAssistantTurnId) {
  var tailId = state.conversation
    ? state.conversation.tail_turn_id
    : null;
  var pendingId = state.conversation
    ? state.conversation.pending_assistant_id
    : null;
  return state.turns.filter(function (turn) {
    return !(
      turn.role === "assistant"
      && turn.turn_id === tailId
      && (
        turn.turn_id === pendingId
        || turn.turn_id === workspaceAssistantTurnId
      )
    );
  });
}

function conversationViewTurn(
  turn, state, decorateTurn, branchPoints
) {
  var article = document.createElement("article");
  article.className =
    "conversation-turn conversation-turn-" + turn.role;
  article.setAttribute("data-turn-id", turn.turn_id);
  article.setAttribute(
    "aria-label",
    turn.role === "user" ? "You" : "Assistant"
  );
  if (
    turn.role === "user"
    && state.conversation
    && state.conversation.tail_turn_id
    && turn.index === state.conversation.turn_count - 1
  ) {
    article.classList.add("is-active-user");
  }

  var header = document.createElement("header");
  header.className = "conversation-turn-header";
  var speaker = document.createElement("span");
  speaker.className = "conversation-turn-speaker";
  speaker.textContent = turn.role === "user" ? "You" : "Assistant";
  header.appendChild(speaker);
  conversationViewBadges(header, turn);

  var text = document.createElement("div");
  text.className = "conversation-turn-text";
  conversationViewTurnText(text, turn);
  article.appendChild(header);
  article.appendChild(text);
  decorateTurn(article, turn, branchPoints);
  return article;
}

function conversationViewTurnText(element, turn) {
  if (turn.role !== "assistant" || turn.text !== "") {
    element.textContent = turn.text;
    return;
  }
  element.classList.add("is-empty");
  element.textContent = turn.partial
    ? "The run stopped before producing any text."
    : "The model ended before producing any text."
      + " Retry to sample another continuation.";
}

function conversationViewBadges(header, turn) {
  if (turn.role !== "assistant") {
    return;
  }
  if (turn.model_id) {
    header.appendChild(
      conversationViewBadge(turn.model_id, "model")
    );
  }
  if (turn.partial) {
    header.appendChild(
      conversationViewBadge("Partial", "partial")
    );
  }
  if (turn.text === "") {
    var noOutput = conversationViewBadge("No output", "empty");
    noOutput.title =
      "The model completed without producing text or token data.";
    header.appendChild(noOutput);
    return;
  }
  if (turn.run_link) {
    header.appendChild(
      conversationViewBadge("Saved", "saved")
    );
    header.appendChild(conversationViewAnalyticsLink(turn.run_link));
    return;
  }
  var textOnly = conversationViewBadge("Text only", "text-only");
  textOnly.title =
    "This response has no saved XAI run. Its text remains,"
    + " but frames, candidates, and overlays are unavailable.";
  header.appendChild(textOnly);
}

function conversationViewBadge(text, kind) {
  var badge = document.createElement("span");
  badge.className =
    "conversation-turn-badge conversation-turn-badge-" + kind;
  badge.textContent = text;
  return badge;
}

function conversationViewAnalyticsLink(runLink) {
  var link = document.createElement("a");
  link.className = "conversation-run-link";
  link.href = "/analytics.html?run="
    + encodeURIComponent(runLink.run_id);
  link.textContent = "Open in Analytics";
  link.setAttribute(
    "aria-label",
    "Open saved run " + runLink.run_id + " in Analytics"
  );
  return link;
}

function conversationViewEmpty(
  empty, state, visibleCount, workspaceVisible
) {
  if (visibleCount > 0 || workspaceVisible) {
    empty.hidden = true;
    empty.textContent = "";
    return;
  }
  empty.hidden = false;
  if (state.conversation === null) {
    empty.textContent =
      "Send a message to start a durable conversation.";
    return;
  }
  if (state.conversation.turn_count === 0) {
    empty.textContent =
      "This conversation is empty. Send a message when ready.";
    return;
  }
  empty.textContent =
    "The active response is shown in the workspace below.";
}

function conversationViewLoadButton(button, state) {
  button.hidden = !state.hasMore && !state.loadingOlder;
  button.disabled = state.loadingOlder;
  button.textContent = state.loadingOlder
    ? "Loading older messages..."
    : "Load older messages";
}

function conversationViewElement(id) {
  var element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing conversation view element #" + id);
  }
  return element;
}

function conversationViewCallback(options, name) {
  if (!options || typeof options[name] !== "function") {
    throw new TypeError(
      "conversationViewCreate needs options." + name
    );
  }
  return options[name];
}
