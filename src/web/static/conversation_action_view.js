// DOM-only rendering and focus for durable conversation actions.
//
// Loaded as a classic script before conversation_actions.js. It
// exposes one factory; workflow state and asynchronous work remain in
// the action controller.

"use strict";

var conversationActionViewCreate = (function () {
  function create() {
    var owner = {
      root: requiredElement("conversation-transcript"),
      activeMount: requiredElement("active-assistant-actions"),
      activeCard: requiredElement("active-assistant-card"),
      feedback: requiredElement("conversation-action-status"),
      sendButton: requiredElement("btn-generate"),
      composer: requiredElement("prompt-input"),
      dialogs: {
        delete: dialogElements("delete"),
        retry: dialogElements("retry"),
      },
    };
    return Object.freeze({
      root: function () {
        return owner.root;
      },
      dialog: function (kind) {
        return dialogFor(owner, kind).dialog;
      },
      decorateTurn: function (settings) {
        decorateTurn(owner, settings);
      },
      decorateActive: function (settings) {
        decorateActive(owner, settings);
      },
      deletionMarker: function (point) {
        return deletionMarker(point);
      },
      setDialogMessage: function (kind, message) {
        setDialogMessage(owner, kind, message);
      },
      setDialogPending: function (kind, pending) {
        setDialogPending(owner, kind, pending);
      },
      clearDialogStatus: function (kind) {
        clearDialogStatus(owner, kind);
      },
      showDialogStatus: function (kind, message) {
        showDialogStatus(owner, kind, message);
      },
      focusDialogCancel: function (kind) {
        dialogFor(owner, kind).cancel.focus();
      },
      setFeedback: function (message, danger) {
        setFeedback(owner, message, danger);
      },
      focusEdit: function (turnId) {
        return focusAttribute(
          owner,
          "data-conversation-edit-input",
          turnId
        );
      },
      focusEditPending: function (turnId) {
        return focusAttribute(
          owner,
          "data-conversation-edit-status",
          turnId
        );
      },
      focusEditTrigger: function (turnId) {
        return focusAttribute(
          owner,
          "data-conversation-edit-for",
          turnId
        );
      },
      focusTrigger: function (confirmation) {
        return focusTrigger(owner, confirmation);
      },
      focusKey: function (key) {
        return focusAttribute(
          owner, "data-conversation-focus-key", key
        );
      },
      focusBranchFallback: function (key) {
        focusBranchFallback(owner, key);
      },
      focusTurn: function (turnId) {
        return focusTurn(owner, turnId);
      },
      focusDeletion: function (branchId) {
        return focusDeletion(owner, branchId);
      },
      focusActive: function () {
        return focusActive(owner);
      },
      focusComposer: function () {
        focusComposer(owner);
      },
      focusRoot: function () {
        focusRoot(owner);
      },
    });
  }

  function decorateTurn(owner, settings) {
    turnShape(settings.turn);
    branchPointsShape(settings.branchPoints);
    if (
      settings.turn.role === "user"
      && settings.edit !== null
    ) {
      renderEdit(settings);
    } else {
      var actions = turnButtons(settings);
      if (actions !== null) {
        settings.article.appendChild(actions);
      }
    }
    appendPagers(settings.article, settings.branchPoints);
  }

  function decorateActive(owner, settings) {
    owner.activeMount.replaceChildren();
    owner.activeMount.hidden = true;
    if (!settings.workspaceVisible || settings.turn === null) {
      return;
    }
    turnShape(settings.turn);
    branchPointsShape(settings.branchPoints);
    if (!settings.assistantComplete) {
      return;
    }
    owner.activeMount.appendChild(
      assistantButtons(settings.turn)
    );
    appendPagers(owner.activeMount, settings.branchPoints);
    owner.activeMount.hidden = false;
  }

  function turnButtons(settings) {
    if (settings.turn.role === "user") {
      return userButtons(settings.turn);
    }
    if (settings.assistantComplete) {
      return assistantButtons(settings.turn);
    }
    return null;
  }

  function userButtons(turn) {
    var actions = buttonBar();
    actions.appendChild(iconButton({
      action: "copy",
      icon: "copy",
      label: "Copy user message",
      turnId: turn.turn_id,
    }));
    actions.appendChild(iconButton({
      action: "edit",
      icon: "edit",
      label: "Edit user message",
      turnId: turn.turn_id,
      focusAttribute: "data-conversation-edit-for",
    }));
    actions.appendChild(iconButton({
      action: "delete",
      icon: "delete",
      label: "Delete from this path",
      turnId: turn.turn_id,
      focusAttribute: "data-conversation-delete-for",
    }));
    return actions;
  }

  function assistantButtons(turn) {
    var actions = buttonBar();
    actions.appendChild(iconButton({
      action: "copy",
      icon: "copy",
      label: "Copy assistant message",
      turnId: turn.turn_id,
    }));
    actions.appendChild(iconButton({
      action: "retry",
      icon: "retry",
      label: "Retry assistant response",
      turnId: turn.turn_id,
      focusAttribute: "data-conversation-retry-for",
    }));
    return actions;
  }

  function buttonBar() {
    var actions = document.createElement("div");
    actions.className = "conversation-turn-actions";
    actions.setAttribute("role", "group");
    actions.setAttribute("aria-label", "Message actions");
    return actions;
  }

  function iconButton(settings) {
    var button = document.createElement("button");
    button.type = "button";
    button.className =
      "conversation-action conversation-action-" + settings.action;
    button.title = settings.label;
    button.setAttribute("aria-label", settings.label);
    button.setAttribute(
      "data-conversation-action", settings.action
    );
    button.setAttribute("data-turn-id", settings.turnId);
    if (settings.focusAttribute) {
      button.setAttribute(
        settings.focusAttribute, settings.turnId
      );
    }
    button.appendChild(icon(settings.icon));
    return button;
  }

  function icon(kind) {
    var svg = svgElement();
    if (kind === "copy") {
      copyIcon(svg);
    } else if (kind === "edit") {
      editIcon(svg);
    } else if (kind === "delete") {
      deleteIcon(svg);
    } else if (kind === "retry") {
      retryIcon(svg);
    } else {
      throw new Error("Unknown conversation action icon");
    }
    return svg;
  }

  function svgElement() {
    var svg = document.createElementNS(
      "http://www.w3.org/2000/svg", "svg"
    );
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", "15");
    svg.setAttribute("height", "15");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "1.8");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    return svg;
  }

  function svgPath(svg, value) {
    var path = document.createElementNS(
      "http://www.w3.org/2000/svg", "path"
    );
    path.setAttribute("d", value);
    svg.appendChild(path);
  }

  function copyIcon(svg) {
    svgPath(
      svg,
      "M9 8h10a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9"
        + "a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Z"
    );
    svgPath(
      svg,
      "M17 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9"
        + "a2 2 0 0 0 2 2h2"
    );
  }

  function editIcon(svg) {
    svgPath(
      svg,
      "M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16v4Z"
    );
    svgPath(svg, "m13.5 6.5 4 4");
  }

  function deleteIcon(svg) {
    svgPath(svg, "M4 7h16");
    svgPath(svg, "M9 7V4h6v3");
    svgPath(svg, "m6 7 1 14h10l1-14");
    svgPath(svg, "M10 11v6M14 11v6");
  }

  function retryIcon(svg) {
    svgPath(svg, "M20 11a8 8 0 1 0 1 4M20 4v7h-7");
  }

  function renderEdit(settings) {
    var text = settings.article.querySelector(
      ".conversation-turn-text"
    );
    if (!text) {
      throw new Error("Editable conversation card has no text");
    }
    settings.article.classList.add("is-editing");
    settings.article.setAttribute(
      "data-conversation-editing", "true"
    );
    settings.article.setAttribute(
      "aria-label", "You, editing message"
    );
    appendEditBadge(settings.article);
    text.classList.add("is-editing");
    text.replaceChildren(editForm(settings));
  }

  function appendEditBadge(article) {
    var header = article.querySelector(
      ".conversation-turn-header"
    );
    if (!header) {
      return;
    }
    var badge = document.createElement("span");
    badge.className =
      "conversation-turn-badge conversation-turn-badge-editing";
    badge.textContent = "Editing";
    header.appendChild(badge);
  }

  function editForm(settings) {
    var form = document.createElement("div");
    form.className = "conversation-inline-edit";
    if (settings.edit.saving) {
      form.appendChild(editPendingStatus(settings.turn.turn_id));
    } else {
      form.appendChild(editInput(settings));
    }
    form.appendChild(editNote(settings.edit.configuration));
    form.appendChild(editButtons(settings.edit));
    return form;
  }

  function editInput(settings) {
    var input = document.createElement("textarea");
    input.className = "conversation-inline-edit-input";
    input.value = settings.edit.draft;
    input.rows = 4;
    input.maxLength = settings.textMax;
    input.readOnly = settings.edit.submittedDraft !== null;
    input.setAttribute("maxlength", String(settings.textMax));
    input.setAttribute("aria-label", "Edit user message");
    input.setAttribute(
      "aria-readonly",
      input.readOnly ? "true" : "false"
    );
    input.setAttribute(
      "data-conversation-edit-input",
      settings.turn.turn_id
    );
    return input;
  }

  function editPendingStatus(turnId) {
    var status = document.createElement("p");
    status.className = "conversation-inline-edit-status";
    status.tabIndex = -1;
    status.textContent =
      "Saving edited message and starting regeneration...";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.setAttribute("aria-atomic", "true");
    status.setAttribute("aria-busy", "true");
    status.setAttribute("data-conversation-edit-status", turnId);
    return status;
  }

  function editNote(configuration) {
    var note = document.createElement("p");
    note.className = "conversation-inline-edit-note";
    note.textContent =
      "Saving creates an alternate path and regenerates with "
      + modelLabel(configuration) + ". "
      + "Input mode: " + configuration.inputMode + ". "
      + "Current Run settings: "
      + configuration.settingsSummary + ".";
    return note;
  }

  function editButtons(edit) {
    var actions = document.createElement("div");
    actions.className = "conversation-inline-edit-actions";
    actions.appendChild(textButton({
      action: "edit-cancel",
      label: "Cancel",
      className: "btn-secondary",
      disabled: edit.saving,
    }));
    actions.appendChild(textButton({
      action: "edit-save",
      label: editButtonLabel(edit),
      className: "btn-accent",
      disabled: edit.saving,
    }));
    return actions;
  }

  function editButtonLabel(edit) {
    if (edit.saving) {
      return "Saving...";
    }
    if (edit.submittedDraft !== null) {
      return "Retry save and regenerate";
    }
    return "Save and regenerate";
  }

  function textButton(settings) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = settings.className;
    button.textContent = settings.label;
    button.disabled = settings.disabled === true;
    button.setAttribute(
      "data-conversation-action", settings.action
    );
    return button;
  }

  function appendPagers(mount, branchPoints) {
    for (
      var index = 0;
      index < branchPoints.length;
      index++
    ) {
      var pager = branchPager(branchPoints[index]);
      if (pager !== null) {
        mount.appendChild(pager);
      }
    }
  }

  function branchPager(point) {
    var total = point.branch_ids.length;
    if (total <= 1) {
      return null;
    }
    var selected = point.branch_ids.indexOf(
      point.selected_branch_id
    );
    if (selected < 0) {
      throw new Error("Selected branch is absent from its pager");
    }
    var key = branchKey(point);
    var pager = document.createElement("div");
    pager.className = "conversation-branch-pager";
    pager.tabIndex = -1;
    pager.setAttribute("role", "group");
    pager.setAttribute(
      "aria-label",
      "Alternate paths at turn " + point.turn_index
    );
    pager.setAttribute("data-conversation-focus-key", key);
    pager.appendChild(branchButton({
      action: "branch-previous",
      label: "Previous path at turn " + point.turn_index,
      branchId: selected > 0
        ? point.branch_ids[selected - 1]
        : point.selected_branch_id,
      disabled: selected === 0,
      focusKey: key,
      text: "\u2039",
    }));
    pager.appendChild(branchPosition(point, selected, total));
    pager.appendChild(branchButton({
      action: "branch-next",
      label: "Next path at turn " + point.turn_index,
      branchId: selected < total - 1
        ? point.branch_ids[selected + 1]
        : point.selected_branch_id,
      disabled: selected === total - 1,
      focusKey: key,
      text: "\u203a",
    }));
    return pager;
  }

  function branchPosition(point, selected, total) {
    var status = document.createElement("span");
    status.className = "conversation-branch-position";
    status.textContent =
      "Path " + (selected + 1) + " / " + total;
    status.setAttribute(
      "aria-label",
      "Selected path " + (selected + 1) + " of " + total
        + " at turn " + point.turn_index
    );
    return status;
  }

  function branchButton(settings) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-branch-button";
    button.textContent = settings.text;
    button.title = settings.label;
    button.disabled = settings.disabled;
    button.setAttribute("aria-label", settings.label);
    button.setAttribute(
      "data-conversation-action", settings.action
    );
    button.setAttribute("data-branch-id", settings.branchId);
    button.setAttribute(
      "data-conversation-focus-key", settings.focusKey
    );
    return button;
  }

  function deletionMarker(point) {
    var marker = document.createElement("div");
    marker.className = "conversation-deletion-marker";
    marker.tabIndex = -1;
    marker.setAttribute("role", "note");
    marker.setAttribute(
      "data-conversation-deletion-branch",
      point.selected_branch_id
    );
    var label = document.createElement("span");
    label.className = "conversation-deletion-label";
    label.textContent = "Messages removed from this path";
    marker.appendChild(label);
    var pager = branchPager(point);
    if (pager !== null) {
      marker.appendChild(pager);
    }
    return marker;
  }

  function setDialogMessage(owner, kind, message) {
    if (typeof message !== "string") {
      throw new TypeError("Dialog message must be a string");
    }
    dialogFor(owner, kind).message.textContent = message;
  }

  function setDialogPending(owner, kind, pending) {
    var elements = dialogFor(owner, kind);
    elements.cancel.disabled = pending;
    elements.confirm.disabled = pending;
    elements.dialog.classList.toggle("is-pending", pending);
    elements.dialog.setAttribute(
      "aria-busy", pending ? "true" : "false"
    );
    if (pending) {
      elements.status.setAttribute("role", "status");
      elements.status.setAttribute("aria-live", "polite");
      elements.status.setAttribute("aria-atomic", "true");
      elements.status.textContent = kind === "delete"
        ? "Deleting from this path..."
        : "Creating the retry path...";
      elements.status.hidden = false;
      elements.status.setAttribute("aria-busy", "true");
      elements.status.focus();
    } else {
      elements.status.removeAttribute("aria-busy");
    }
  }

  function clearDialogStatus(owner, kind) {
    var status = dialogFor(owner, kind).status;
    status.textContent = "";
    status.hidden = true;
    status.removeAttribute("aria-busy");
  }

  function showDialogStatus(owner, kind, message) {
    if (typeof message !== "string" || message === "") {
      throw new TypeError("Dialog status must be a non-empty string");
    }
    var status = dialogFor(owner, kind).status;
    status.setAttribute("role", "alert");
    status.setAttribute("aria-live", "assertive");
    status.setAttribute("aria-atomic", "true");
    status.textContent = message;
    status.hidden = false;
    status.removeAttribute("aria-busy");
    status.focus();
  }

  function setFeedback(owner, message, danger) {
    owner.feedback.textContent = message;
    owner.feedback.hidden = message === "";
    owner.feedback.classList.toggle("is-error", danger);
  }

  function focusTrigger(owner, confirmation) {
    if (
      confirmation.trigger
      && confirmation.trigger.isConnected
      && typeof confirmation.trigger.focus === "function"
    ) {
      confirmation.trigger.focus();
      return true;
    }
    return focusAttribute(
      owner,
      confirmation.kind === "delete"
        ? "data-conversation-delete-for"
        : "data-conversation-retry-for",
      confirmation.turnId
    );
  }

  function focusAttribute(owner, attribute, value) {
    var selector = "[" + attribute + "=\"" + value + "\"]";
    var element = owner.root.querySelector(selector);
    if (!element || typeof element.focus !== "function") {
      return false;
    }
    element.focus();
    return true;
  }

  function focusTurn(owner, turnId) {
    var element = owner.root.querySelector(
      "[data-turn-id=\"" + turnId + "\"]"
    );
    if (!element || typeof element.focus !== "function") {
      return false;
    }
    element.setAttribute("tabindex", "-1");
    element.focus();
    return true;
  }

  function focusDeletion(owner, branchId) {
    if (!branchId) {
      return false;
    }
    return focusAttribute(
      owner, "data-conversation-deletion-branch", branchId
    );
  }

  function focusActive(owner) {
    if (
      owner.activeCard.hidden
      || typeof owner.activeCard.focus !== "function"
    ) {
      return false;
    }
    owner.activeCard.setAttribute("tabindex", "-1");
    owner.activeCard.focus();
    return true;
  }

  function focusComposer(owner) {
    if (!owner.sendButton.hidden && !owner.sendButton.disabled) {
      owner.sendButton.focus();
      return;
    }
    owner.composer.focus();
  }

  function focusBranchFallback(owner, key) {
    if (key !== "" && focusAttribute(
      owner, "data-conversation-focus-key", key
    )) {
      return;
    }
    if (focusRoot(owner)) {
      return;
    }
    if (focusActive(owner)) {
      return;
    }
    focusComposer(owner);
  }

  function focusRoot(owner) {
    if (
      owner.root.hidden
      || owner.root.isConnected === false
      || typeof owner.root.focus !== "function"
    ) {
      return false;
    }
    owner.root.setAttribute("tabindex", "-1");
    owner.root.focus();
    return true;
  }

  function modelLabel(configuration) {
    if (
      configuration.modelDisplay === configuration.modelId
    ) {
      return configuration.modelDisplay;
    }
    return configuration.modelDisplay
      + " (" + configuration.modelId + ")";
  }

  function branchKey(point) {
    return point.turn_index + ":" + point.source_branch_id;
  }

  function turnShape(turn) {
    if (!turn || typeof turn !== "object") {
      throw new TypeError("Conversation action turn is invalid");
    }
    if (turn.role !== "user" && turn.role !== "assistant") {
      throw new TypeError("Conversation action role is invalid");
    }
    if (
      typeof turn.turn_id !== "string"
      || turn.turn_id === ""
    ) {
      throw new TypeError("Conversation action turn id is invalid");
    }
  }

  function branchPointsShape(points) {
    if (!Array.isArray(points)) {
      throw new TypeError(
        "Conversation action branch points must be a list"
      );
    }
  }

  function dialogElements(kind) {
    return {
      dialog: requiredElement(
        "conversation-" + kind + "-dialog"
      ),
      message: requiredElement(
        "conversation-" + kind + "-message"
      ),
      status: requiredElement(
        "conversation-" + kind + "-status"
      ),
      cancel: requiredElement(
        "btn-conversation-" + kind + "-cancel"
      ),
      confirm: requiredElement(
        "btn-conversation-" + kind + "-confirm"
      ),
    };
  }

  function dialogFor(owner, kind) {
    var elements = owner.dialogs[kind];
    if (!elements) {
      throw new Error("Unknown conversation action dialog");
    }
    return elements;
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error(
        "Missing conversation action view element #" + id
      );
    }
    return element;
  }

  return create;
}());
