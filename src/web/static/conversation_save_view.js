// DOM and focus ownership for the Save Conversation workflow.

"use strict";

function conversationSaveViewCreate() {
  var owner = {
    button: conversationSaveViewRequired("btn-save-conversation"),
    dialog: conversationSaveViewRequired("conversation-save-dialog"),
    title: conversationSaveViewRequired(
      "conversation-save-title-input"
    ),
    exchanges: conversationSaveViewRequired(
      "conversation-save-exchanges"
    ),
    xai: conversationSaveViewRequired("conversation-save-xai"),
    textOnly: conversationSaveViewRequired(
      "conversation-save-text-only"
    ),
    unavailable: conversationSaveViewRequired(
      "conversation-save-unavailable"
    ),
    active: conversationSaveViewRequired("conversation-save-active"),
    status: conversationSaveViewRequired("conversation-save-status"),
    cancel: conversationSaveViewRequired(
      "btn-conversation-save-cancel"
    ),
    confirm: conversationSaveViewRequired(
      "btn-conversation-save-confirm"
    ),
    trigger: null,
    wired: false,
  };
  return Object.freeze({
    wire: function (callbacks) {
      conversationSaveViewWire(owner, callbacks);
    },
    show: function (preview, savesActiveTail) {
      conversationSaveViewShow(owner, preview, savesActiveTail);
    },
    close: function () {
      conversationSaveViewClose(owner);
    },
    setDisabled: function (disabled) {
      owner.button.disabled = disabled === true;
    },
    setPending: function (pending) {
      conversationSaveViewPending(owner, pending);
    },
    showStatus: function (message) {
      conversationSaveViewStatus(owner, message);
    },
    clearStatus: function () {
      conversationSaveViewStatus(owner, "");
    },
    title: function () {
      return owner.title.value;
    },
    isOpen: function () {
      return owner.dialog.open === true;
    },
  });
}

function conversationSaveViewWire(owner, callbacks) {
  if (owner.wired) {
    return;
  }
  if (
    !callbacks
    || typeof callbacks.open !== "function"
    || typeof callbacks.confirm !== "function"
    || typeof callbacks.cancel !== "function"
  ) {
    throw new TypeError("conversation save view needs callbacks");
  }
  owner.wired = true;
  owner.button.addEventListener("click", function () {
    owner.trigger = owner.button;
    callbacks.open();
  });
  owner.cancel.addEventListener("click", callbacks.cancel);
  owner.confirm.addEventListener("click", callbacks.confirm);
  owner.dialog.addEventListener("cancel", function (event) {
    event.preventDefault();
    callbacks.cancel();
  });
}

function conversationSaveViewShow(owner, preview, savesActiveTail) {
  owner.title.value = preview.default_title;
  owner.exchanges.textContent = String(preview.exchange_count);
  owner.xai.textContent = String(preview.xai_count);
  owner.textOnly.textContent = String(preview.text_only_count);
  owner.unavailable.textContent = String(preview.unavailable_count);
  owner.active.hidden = !savesActiveTail;
  conversationSaveViewStatus(owner, "");
  conversationSaveViewPending(owner, false);
  if (!owner.dialog.open) {
    owner.dialog.showModal();
  }
  owner.title.focus();
  owner.title.select();
}

function conversationSaveViewClose(owner) {
  if (owner.dialog.open) {
    owner.dialog.close();
  }
  conversationSaveViewPending(owner, false);
  conversationSaveViewStatus(owner, "");
  if (owner.trigger && owner.trigger.isConnected !== false) {
    owner.trigger.focus();
  }
  owner.trigger = null;
}

function conversationSaveViewPending(owner, pending) {
  var active = pending === true;
  owner.title.disabled = active;
  owner.cancel.disabled = active;
  owner.confirm.disabled = active;
  owner.dialog.classList.toggle("is-pending", active);
  owner.dialog.setAttribute("aria-busy", active ? "true" : "false");
}

function conversationSaveViewStatus(owner, message) {
  owner.status.textContent = message;
  owner.status.hidden = message === "";
  if (message !== "") {
    owner.status.focus();
  }
}

function conversationSaveViewRequired(id) {
  var element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing conversation save element #" + id);
  }
  return element;
}
