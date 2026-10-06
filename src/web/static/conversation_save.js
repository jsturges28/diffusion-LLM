// Immutable selected-path snapshot workflow for the generator.

"use strict";

function conversationSaveCreate(options) {
  var owner = {
    client: conversationSaveRequired(options, "client"),
    view: conversationSaveRequired(options, "view"),
    readHead: conversationSaveCallback(options, "readHead"),
    readBlockReason:
      conversationSaveCallback(options, "readBlockReason"),
    shouldSaveActiveTail:
      conversationSaveCallback(options, "shouldSaveActiveTail"),
    saveActiveTail:
      conversationSaveCallback(options, "saveActiveTail"),
    createOperationId:
      conversationSaveCallback(options, "createOperationId"),
    onSaved: conversationSaveCallback(options, "onSaved"),
    onBlocked: conversationSaveCallback(options, "onBlocked"),
    onBusy: conversationSaveCallback(options, "onBusy"),
    openedHead: null,
    operationId: null,
    savesActiveTail: false,
    activeTailSaved: false,
    pending: false,
  };
  return Object.freeze({
    wire: function () {
      conversationSaveWire(owner);
    },
    refresh: function () {
      conversationSaveRefresh(owner);
    },
    close: function () {
      conversationSaveCancel(owner);
    },
  });
}

function conversationSaveWire(owner) {
  owner.view.wire({
    open: function () {
      conversationSaveOpen(owner);
    },
    confirm: function () {
      conversationSaveConfirm(owner);
    },
    cancel: function () {
      conversationSaveCancel(owner);
    },
  });
  conversationSaveRefresh(owner);
}

function conversationSaveOpen(owner) {
  var blocked = owner.readBlockReason();
  var head = owner.readHead();
  if (blocked !== "" || head === null) {
    owner.onBlocked(
      blocked || "Complete a response before saving this conversation."
    );
    return;
  }
  owner.pending = true;
  owner.onBusy(true);
  conversationSaveRefresh(owner);
  owner.client.preview(head).then(function (preview) {
    var current = owner.readHead();
    if (!conversationSaveHeadsMatch(head, current)) {
      throw new Error(
        "The selected path changed while its summary was prepared."
      );
    }
    owner.openedHead = head;
    owner.operationId = null;
    owner.activeTailSaved = false;
    owner.savesActiveTail = owner.shouldSaveActiveTail(head);
    owner.pending = false;
    owner.view.show(preview, owner.savesActiveTail);
    conversationSaveRefresh(owner);
  }).catch(function (error) {
    owner.pending = false;
    owner.onBusy(false);
    owner.onBlocked(conversationSaveErrorMessage(error, false));
    conversationSaveRefresh(owner);
  });
}

function conversationSaveConfirm(owner) {
  if (owner.pending || owner.openedHead === null) {
    return;
  }
  var current = owner.readHead();
  if (!conversationSaveHeadsMatch(owner.openedHead, current)) {
    owner.view.showStatus(
      "The selected path changed. Cancel and open Save Conversation"
      + " again."
    );
    return;
  }
  var title = owner.view.title().trim();
  if (title === "" || title.length > 200) {
    owner.view.showStatus(
      "Enter a title between 1 and 200 characters."
    );
    return;
  }
  if (owner.operationId === null) {
    owner.operationId = owner.createOperationId();
  }
  owner.pending = true;
  owner.view.clearStatus();
  owner.view.setPending(true);
  conversationSaveMaybeTail(owner).then(function () {
    return conversationSavePublish(owner, title);
  }).then(function (result) {
    owner.pending = false;
    owner.view.close();
    owner.onBusy(false);
    owner.onSaved(result);
    conversationSaveReset(owner);
    conversationSaveRefresh(owner);
  }).catch(function (error) {
    owner.pending = false;
    owner.view.setPending(false);
    owner.view.showStatus(
      conversationSaveErrorMessage(error, owner.activeTailSaved)
    );
    conversationSaveRefresh(owner);
  });
}

function conversationSaveMaybeTail(owner) {
  if (!owner.savesActiveTail) {
    return Promise.resolve();
  }
  return Promise.resolve(owner.saveActiveTail()).then(function (saved) {
    if (saved !== true) {
      throw new Error(
        "The active response could not be saved, so the conversation"
        + " snapshot was not created."
      );
    }
    owner.activeTailSaved = true;
    owner.savesActiveTail = false;
    var refreshed = owner.readHead();
    if (!conversationSaveSamePath(owner.openedHead, refreshed)) {
      throw new Error(
        "The selected path changed while its active run was saved."
      );
    }
    owner.openedHead = refreshed;
  });
}

function conversationSavePublish(owner, title) {
  var head = owner.readHead();
  if (head === null) {
    throw new Error("The selected conversation is no longer available.");
  }
  return owner.client.preview(head).then(function () {
    return owner.client.create(Object.assign({}, head, {
      operationId: owner.operationId,
      title: title,
    }));
  });
}

function conversationSaveCancel(owner) {
  if (owner.pending) {
    return;
  }
  owner.view.close();
  owner.onBusy(false);
  conversationSaveReset(owner);
  conversationSaveRefresh(owner);
}

function conversationSaveReset(owner) {
  owner.openedHead = null;
  owner.operationId = null;
  owner.savesActiveTail = false;
  owner.activeTailSaved = false;
}

function conversationSaveRefresh(owner) {
  var blocked = owner.readBlockReason();
  var disabled = (
    owner.pending
    || owner.view.isOpen()
    || owner.readHead() === null
    || blocked !== ""
  );
  owner.view.setDisabled(disabled);
}

function conversationSaveHeadsMatch(left, right) {
  if (left === null || right === null) {
    return left === right;
  }
  var names = [
    "conversation_id",
    "branch_id",
    "branch_revision",
    "turn_count",
    "tail_turn_id",
    "tail_version",
  ];
  for (var index = 0; index < names.length; index++) {
    if (left[names[index]] !== right[names[index]]) {
      return false;
    }
  }
  return true;
}

function conversationSaveSamePath(left, right) {
  if (left === null || right === null) {
    return false;
  }
  return (
    left.conversation_id === right.conversation_id
    && left.branch_id === right.branch_id
    && left.turn_count === right.turn_count
    && left.tail_turn_id === right.tail_turn_id
  );
}

function conversationSaveErrorMessage(error, activeTailSaved) {
  var message = error instanceof Error
    ? error.message
    : "Saved conversation request failed.";
  if (activeTailSaved) {
    return "The active run was saved, but the conversation snapshot"
      + " was not: " + message;
  }
  return message;
}

function conversationSaveRequired(options, name) {
  if (!options || !options[name]) {
    throw new TypeError(
      "conversationSaveCreate needs options." + name
    );
  }
  return options[name];
}

function conversationSaveCallback(options, name) {
  var callback = conversationSaveRequired(options, name);
  if (typeof callback !== "function") {
    throw new TypeError(
      "conversationSaveCreate needs callback options." + name
    );
  }
  return callback;
}
