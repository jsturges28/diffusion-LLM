// Saved-conversation catalog and bounded exchange viewer.

"use strict";

var ANALYTICS_CONVERSATION_TURNS_MAX = 200;
var ANALYTICS_CONVERSATION_PAGE_LIMIT = 50;

function analyticsConversationsCreate(options) {
  var owner = {
    client: analyticsConversationsRequired(options, "client"),
    openXai: analyticsConversationsCallback(options, "openXai"),
    showToast: analyticsConversationsCallback(options, "showToast"),
    focusFallback: analyticsConversationsRequired(
      options, "focusFallback"
    ),
    panel: analyticsConversationsElement("conversations-panel"),
    tbody: analyticsConversationsElement("conversations-tbody"),
    empty: analyticsConversationsElement("conversations-empty"),
    detail: analyticsConversationsElement(
      "conversation-detail-modal"
    ),
    closeDetail: analyticsConversationsElement(
      "btn-close-conversation-detail"
    ),
    doneDetail: analyticsConversationsElement(
      "btn-done-conversation"
    ),
    title: analyticsConversationsElement(
      "saved-conversation-title"
    ),
    rename: analyticsConversationsElement(
      "btn-rename-conversation"
    ),
    meta: analyticsConversationsElement(
      "saved-conversation-meta"
    ),
    status: analyticsConversationsElement(
      "saved-conversation-status"
    ),
    older: analyticsConversationsElement(
      "btn-load-older-exchanges"
    ),
    exchanges: analyticsConversationsElement(
      "saved-conversation-exchanges"
    ),
    remove: analyticsConversationsElement(
      "btn-delete-conversation"
    ),
    deleteDialog: analyticsConversationsElement(
      "modal-delete-conversation"
    ),
    deleteLabel: analyticsConversationsElement(
      "delete-conversation-label"
    ),
    deleteCancel: analyticsConversationsElement(
      "btn-delete-conversation-cancel"
    ),
    deleteClose: analyticsConversationsElement(
      "btn-delete-conversation-close"
    ),
    deleteConfirm: analyticsConversationsElement(
      "btn-delete-conversation-confirm"
    ),
    rows: [],
    current: null,
    turns: [],
    nextBefore: null,
    hasMore: false,
    opener: null,
    loaded: false,
    requestEpoch: 0,
    listEpoch: 0,
    deleteEpoch: 0,
    pendingDeleteId: null,
    wired: false,
  };
  return Object.freeze({
    wire: function () {
      analyticsConversationsWire(owner);
    },
    activate: function () {
      owner.panel.hidden = false;
      if (owner.loaded) {
        analyticsConversationsRenderTable(owner);
        return Promise.resolve(owner.rows);
      }
      return analyticsConversationsRefresh(owner);
    },
    deactivate: function () {
      owner.panel.hidden = true;
      analyticsConversationsClose(owner);
    },
    refresh: function () {
      return analyticsConversationsRefresh(owner);
    },
    adopt: function (rows) {
      analyticsConversationsAdvanceListEpoch(owner);
      owner.rows = Array.isArray(rows) ? rows : [];
      owner.loaded = true;
      analyticsConversationsRenderTable(owner);
    },
    openLinked: function (snapshotId) {
      return analyticsConversationsOpen(owner, snapshotId, null);
    },
  });
}

function analyticsConversationsWire(owner) {
  if (owner.wired) {
    return;
  }
  owner.wired = true;
  owner.tbody.addEventListener("click", function (event) {
    analyticsConversationsTableClick(owner, event);
  });
  owner.tbody.addEventListener("keydown", function (event) {
    if (event.key !== "Enter") {
      return;
    }
    if (event.target.closest("button")) {
      return;
    }
    var row = event.target.closest("tr[data-snapshot-id]");
    if (row) {
      event.preventDefault();
      analyticsConversationsOpen(
        owner,
        row.getAttribute("data-snapshot-id"),
        row.querySelector(".snapshot-open-button")
      );
    }
  });
  owner.closeDetail.addEventListener("click", function () {
    analyticsConversationsClose(owner);
  });
  owner.doneDetail.addEventListener("click", function () {
    analyticsConversationsClose(owner);
  });
  owner.detail.addEventListener("cancel", function () {
    analyticsConversationsClose(owner);
  });
  owner.detail.addEventListener("click", function (event) {
    if (event.target === owner.detail) {
      analyticsConversationsClose(owner);
    }
  });
  owner.rename.addEventListener("click", function () {
    analyticsConversationsRename(owner);
  });
  owner.older.addEventListener("click", function () {
    analyticsConversationsLoadOlder(owner);
  });
  owner.exchanges.addEventListener("click", function (event) {
    analyticsConversationsExchangeClick(owner, event);
  });
  owner.remove.addEventListener("click", function () {
    analyticsConversationsAskDelete(owner);
  });
  owner.deleteCancel.addEventListener("click", function () {
    analyticsConversationsCloseDelete(owner);
  });
  owner.deleteClose.addEventListener("click", function () {
    analyticsConversationsCloseDelete(owner);
  });
  owner.deleteConfirm.addEventListener("click", function () {
    analyticsConversationsDelete(owner);
  });
  owner.deleteDialog.addEventListener("click", function (event) {
    if (event.target === owner.deleteDialog) {
      analyticsConversationsCloseDelete(owner);
    }
  });
  owner.deleteDialog.addEventListener("cancel", function (event) {
    event.preventDefault();
    analyticsConversationsCloseDelete(owner);
  });
}

function analyticsConversationsRefresh(owner) {
  var epoch = analyticsConversationsAdvanceListEpoch(owner);
  owner.status.textContent = "";
  return owner.client.list().then(function (rows) {
    if (owner.listEpoch !== epoch) {
      return owner.rows;
    }
    owner.rows = Array.isArray(rows) ? rows : [];
    owner.loaded = true;
    analyticsConversationsRenderTable(owner);
    return owner.rows;
  }).catch(function (error) {
    if (owner.listEpoch !== epoch) {
      return owner.rows;
    }
    owner.rows = [];
    analyticsConversationsRenderTable(owner);
    owner.showToast(
      error instanceof Error
        ? error.message
        : "Saved conversations could not be loaded."
    );
    return [];
  });
}

function analyticsConversationsRenderTable(owner) {
  owner.tbody.replaceChildren();
  owner.empty.hidden = owner.rows.length !== 0;
  for (var index = 0; index < owner.rows.length; index++) {
    owner.tbody.appendChild(
      analyticsConversationsRow(owner.rows[index])
    );
  }
}

function analyticsConversationsRow(snapshot) {
  var row = document.createElement("tr");
  row.setAttribute("data-snapshot-id", snapshot.snapshot_id);
  if (snapshot.invalid) {
    row.classList.add("run-invalid");
  }
  row.appendChild(
    analyticsConversationsCell(
      analyticsConversationsDate(snapshot.created_at)
    )
  );
  row.appendChild(analyticsConversationsTitleCell(snapshot));
  row.appendChild(
    analyticsConversationsCell(snapshot.exchange_count || 0)
  );
  row.appendChild(
    analyticsConversationsCell(snapshot.xai_count || 0)
  );
  row.appendChild(
    analyticsConversationsCell(snapshot.text_only_count || 0)
  );
  row.appendChild(
    analyticsConversationsCell(snapshot.unavailable_count || 0)
  );
  var actions = document.createElement("td");
  var remove = document.createElement("button");
  remove.type = "button";
  remove.className = "snapshot-row-delete";
  remove.textContent = "\u00d7";
  remove.title = "Delete saved conversation";
  remove.setAttribute("aria-label", "Delete saved conversation");
  remove.setAttribute("data-delete-snapshot", snapshot.snapshot_id);
  actions.appendChild(remove);
  row.appendChild(actions);
  return row;
}

function analyticsConversationsTitleCell(snapshot) {
  var cell = document.createElement("td");
  var button = document.createElement("button");
  button.type = "button";
  button.className = "snapshot-open-button";
  button.textContent = snapshot.invalid
    ? "Unreadable snapshot"
    : String(snapshot.title || "Untitled");
  button.setAttribute(
    "aria-label", "Open saved conversation " + button.textContent
  );
  cell.appendChild(button);
  return cell;
}

function analyticsConversationsCell(value) {
  var cell = document.createElement("td");
  cell.textContent = String(value);
  return cell;
}

function analyticsConversationsDate(value) {
  if (typeof value !== "string" || value === "") {
    return "";
  }
  var date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function analyticsConversationsTableClick(owner, event) {
  var remove = event.target.closest("[data-delete-snapshot]");
  if (remove) {
    var removeId = remove.getAttribute("data-delete-snapshot");
    var removeRow = analyticsConversationsFind(owner, removeId);
    analyticsConversationsSetCurrent(owner, removeRow, remove);
    analyticsConversationsAskDelete(owner);
    return;
  }
  var row = event.target.closest("tr[data-snapshot-id]");
  if (!row) {
    return;
  }
  analyticsConversationsOpen(
    owner,
    row.getAttribute("data-snapshot-id"),
    row.querySelector(".snapshot-open-button")
  );
}

function analyticsConversationsOpen(owner, snapshotId, opener) {
  var summary = analyticsConversationsFind(owner, snapshotId);
  if (!summary) {
    return Promise.resolve(false);
  }
  if (summary.invalid) {
    owner.showToast(summary.error || "Snapshot is unreadable.");
    return Promise.resolve(false);
  }
  var epoch = analyticsConversationsAdvanceEpoch(owner);
  analyticsConversationsSetCurrent(owner, summary, opener);
  owner.rename.disabled = false;
  owner.status.textContent = "Loading exchanges...";
  owner.turns = [];
  owner.nextBefore = null;
  owner.hasMore = false;
  owner.older.disabled = false;
  owner.older.hidden = true;
  if (!owner.detail.open) {
    owner.detail.showModal();
  }
  return Promise.all([
    owner.client.metadata(snapshotId),
    owner.client.turns(
      snapshotId, null, ANALYTICS_CONVERSATION_PAGE_LIMIT
    ),
  ]).then(function (values) {
    if (!analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      return false;
    }
    analyticsConversationsAdoptMetadata(owner, values[0]);
    analyticsConversationsAdoptPage(owner, values[1], false);
    return true;
  }).catch(function (error) {
    if (!analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      return false;
    }
    owner.status.textContent = error instanceof Error
      ? error.message
      : "Saved conversation could not be opened.";
    return false;
  });
}

function analyticsConversationsSetCurrent(owner, summary, opener) {
  owner.current = summary;
  owner.opener = opener;
  owner.title.value = summary ? summary.title || "" : "";
}

function analyticsConversationsAdoptMetadata(owner, metadata) {
  owner.current = Object.assign({}, owner.current, metadata);
  owner.title.value = String(metadata.title || "");
  var source = metadata.source || {};
  owner.meta.textContent =
    metadata.exchange_count + " exchanges"
    + " \u00b7 " + metadata.xai_count + " XAI"
    + " \u00b7 branch " + String(source.branch_id || "unknown");
  owner.status.textContent = "";
}

function analyticsConversationsAdoptPage(owner, page, older) {
  var incoming = Array.isArray(page.turns) ? page.turns : [];
  owner.turns = older
    ? incoming.concat(owner.turns)
    : incoming;
  while (
    owner.turns.length > ANALYTICS_CONVERSATION_TURNS_MAX
  ) {
    owner.turns.pop();
  }
  owner.nextBefore = page.next_before || null;
  owner.hasMore = page.has_more === true;
  owner.older.hidden = !owner.hasMore;
  analyticsConversationsRenderExchanges(owner);
}

function analyticsConversationsRenderExchanges(owner) {
  owner.exchanges.replaceChildren();
  for (var index = 0; index + 1 < owner.turns.length; index += 2) {
    var user = owner.turns[index];
    var assistant = owner.turns[index + 1];
    if (user.role !== "user" || assistant.role !== "assistant") {
      owner.status.textContent =
        "This snapshot contains an invalid turn sequence.";
      return;
    }
    owner.exchanges.appendChild(
      analyticsConversationsExchange(owner, user, assistant)
    );
  }
}

function analyticsConversationsExchange(owner, user, assistant) {
  var article = document.createElement("article");
  article.className = "saved-conversation-exchange";
  var heading = document.createElement("h3");
  heading.textContent = "Exchange " + ((assistant.index || 2) / 2);
  article.appendChild(heading);
  article.appendChild(
    analyticsConversationsMessage("User", user.text)
  );
  article.appendChild(
    analyticsConversationsMessage("Assistant", assistant.text)
  );
  var footer = document.createElement("div");
  footer.className = "saved-conversation-exchange-footer";
  var status = analyticsConversationsXaiLabel(assistant.xai);
  var badge = document.createElement("span");
  badge.className = "saved-conversation-xai-status";
  badge.textContent = status;
  footer.appendChild(badge);
  if (assistant.partial === true) {
    var partial = document.createElement("span");
    partial.textContent = "Partial";
    footer.appendChild(partial);
  }
  if (assistant.xai && assistant.xai.status === "pinned") {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "saved-conversation-view-xai";
    button.textContent = "View XAI";
    button.setAttribute(
      "data-view-xai-turn", assistant.turn_id
    );
    footer.appendChild(button);
  }
  article.appendChild(footer);
  return article;
}

function analyticsConversationsMessage(role, text) {
  var wrap = document.createElement("div");
  wrap.className =
    "saved-conversation-message is-" + role.toLowerCase();
  var label = document.createElement("strong");
  label.textContent = role;
  var body = document.createElement("pre");
  body.textContent = String(text || "");
  wrap.appendChild(label);
  wrap.appendChild(body);
  return wrap;
}

function analyticsConversationsXaiLabel(xai) {
  if (!xai || xai.status === "text_only") {
    return "Text only";
  }
  if (xai.status === "unavailable") {
    return "XAI unavailable when saved";
  }
  return "XAI preserved";
}

function analyticsConversationsExchangeClick(owner, event) {
  var button = event.target.closest("[data-view-xai-turn]");
  if (!button || !owner.current) {
    return;
  }
  var turnId = button.getAttribute("data-view-xai-turn");
  var assistant = analyticsConversationsTurn(owner, turnId);
  if (!assistant || !assistant.xai) {
    return;
  }
  var snapshotId = owner.current.snapshot_id;
  owner.openXai({
    snapshotId: snapshotId,
    turnId: turnId,
    returnFocus: button,
    summary: {
      backend: assistant.xai.backend || assistant.model_id,
      model: assistant.xai.model || assistant.model_id,
      model_type: assistant.xai.model_type,
      processor: assistant.xai.processor,
      partial: assistant.partial === true,
      prompt: analyticsConversationsPromptFor(owner, assistant),
    },
    urls: {
      metadata: owner.client.pinnedUrl(
        snapshotId, turnId, "metadata"
      ),
      metrics: owner.client.pinnedUrl(
        snapshotId, turnId, "metrics"
      ),
      frames: owner.client.pinnedUrl(
        snapshotId, turnId, "frames"
      ),
    },
  });
}

function analyticsConversationsPromptFor(owner, assistant) {
  var target = Number(assistant.index) - 1;
  for (var index = 0; index < owner.turns.length; index++) {
    if (owner.turns[index].index === target) {
      return owner.turns[index].text;
    }
  }
  return "";
}

function analyticsConversationsTurn(owner, turnId) {
  for (var index = 0; index < owner.turns.length; index++) {
    if (owner.turns[index].turn_id === turnId) {
      return owner.turns[index];
    }
  }
  return null;
}

function analyticsConversationsLoadOlder(owner) {
  if (!owner.current || !owner.hasMore || !owner.nextBefore) {
    return;
  }
  var snapshotId = owner.current.snapshot_id;
  var epoch = owner.requestEpoch;
  owner.older.disabled = true;
  owner.client.turns(
    snapshotId,
    owner.nextBefore,
    ANALYTICS_CONVERSATION_PAGE_LIMIT
  ).then(function (page) {
    if (!analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      return;
    }
    analyticsConversationsAdoptPage(owner, page, true);
    owner.older.disabled = false;
  }).catch(function (error) {
    if (!analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      return;
    }
    owner.older.disabled = false;
    owner.status.textContent = error instanceof Error
      ? error.message
      : "Older exchanges could not be loaded.";
  });
}

function analyticsConversationsRename(owner) {
  if (!owner.current) {
    return;
  }
  var title = owner.title.value.trim();
  if (title === "" || title.length > 200) {
    owner.status.textContent =
      "Enter a title between 1 and 200 characters.";
    return;
  }
  var snapshotId = owner.current.snapshot_id;
  var revision = owner.current.title_revision;
  var epoch = owner.requestEpoch;
  owner.rename.disabled = true;
  owner.client.rename(
    snapshotId,
    title,
    revision
  ).then(function (metadata) {
    analyticsConversationsAdvanceListEpoch(owner);
    analyticsConversationsReplaceRow(owner, metadata);
    analyticsConversationsRenderTable(owner);
    if (analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      analyticsConversationsAdoptMetadata(owner, metadata);
      owner.rename.disabled = false;
      owner.showToast("Saved conversation renamed.");
    }
  }).catch(function (error) {
    if (!analyticsConversationsAccepts(
      owner, epoch, snapshotId
    )) {
      return;
    }
    owner.rename.disabled = false;
    owner.status.textContent = error instanceof Error
      ? error.message
      : "Saved conversation could not be renamed.";
  });
}

function analyticsConversationsReplaceRow(owner, metadata) {
  for (var index = 0; index < owner.rows.length; index++) {
    if (owner.rows[index].snapshot_id === metadata.snapshot_id) {
      var currentRevision = Number(
        owner.rows[index].title_revision || 0
      );
      var receivedRevision = Number(
        metadata.title_revision || 0
      );
      if (receivedRevision < currentRevision) {
        return;
      }
      owner.rows[index] = Object.assign(
        {}, owner.rows[index], metadata
      );
      return;
    }
  }
}

function analyticsConversationsAskDelete(owner) {
  if (!owner.current) {
    return;
  }
  owner.deleteEpoch += 1;
  owner.pendingDeleteId = owner.current.snapshot_id;
  owner.deleteLabel.textContent = owner.current.title;
  owner.deleteDialog.showModal();
}

function analyticsConversationsCloseDelete(owner) {
  if (owner.deleteDialog.open) {
    owner.deleteDialog.close();
  }
  owner.deleteEpoch += 1;
  owner.pendingDeleteId = null;
  owner.deleteConfirm.disabled = false;
}

function analyticsConversationsDelete(owner) {
  if (!owner.pendingDeleteId) {
    return;
  }
  var snapshotId = owner.pendingDeleteId;
  var deleteEpoch = owner.deleteEpoch;
  owner.deleteConfirm.disabled = true;
  owner.client.delete(snapshotId).then(function () {
    if (owner.deleteEpoch === deleteEpoch) {
      owner.deleteConfirm.disabled = false;
      analyticsConversationsCloseDelete(owner);
    }
    analyticsConversationsAdvanceListEpoch(owner);
    owner.rows = owner.rows.filter(function (row) {
      return row.snapshot_id !== snapshotId;
    });
    analyticsConversationsRenderTable(owner);
    if (
      owner.current
      && owner.current.snapshot_id === snapshotId
    ) {
      analyticsConversationsClose(owner);
    }
    owner.showToast("Saved conversation deleted.");
  }).catch(function (error) {
    if (owner.deleteEpoch !== deleteEpoch) {
      return;
    }
    owner.deleteConfirm.disabled = false;
    analyticsConversationsCloseDelete(owner);
    owner.status.textContent = error instanceof Error
      ? error.message
      : "Saved conversation could not be deleted.";
  });
}

function analyticsConversationsClose(owner) {
  analyticsConversationsAdvanceEpoch(owner);
  if (owner.detail.open) {
    owner.detail.close();
  }
  var snapshotId = owner.current
    ? owner.current.snapshot_id
    : null;
  owner.current = null;
  owner.turns = [];
  owner.nextBefore = null;
  owner.hasMore = false;
  owner.older.disabled = false;
  owner.older.hidden = true;
  owner.exchanges.replaceChildren();
  owner.status.textContent = "";
  var opener = owner.opener;
  owner.opener = null;
  if ((!opener || opener.isConnected === false) && snapshotId) {
    var row = owner.tbody.querySelector(
      'tr[data-snapshot-id="' + snapshotId + '"]'
    );
    opener = row
      ? row.querySelector(".snapshot-open-button")
      : null;
  }
  if (opener && opener.isConnected !== false) {
    opener.focus();
  } else {
    owner.focusFallback.focus();
  }
}

function analyticsConversationsAdvanceEpoch(owner) {
  owner.requestEpoch += 1;
  if (owner.requestEpoch > Number.MAX_SAFE_INTEGER) {
    owner.requestEpoch = 1;
  }
  return owner.requestEpoch;
}

function analyticsConversationsAdvanceListEpoch(owner) {
  owner.listEpoch += 1;
  if (owner.listEpoch > Number.MAX_SAFE_INTEGER) {
    owner.listEpoch = 1;
  }
  return owner.listEpoch;
}

function analyticsConversationsAccepts(
  owner, epoch, snapshotId
) {
  return Boolean(
    owner.requestEpoch === epoch
    && owner.current
    && owner.current.snapshot_id === snapshotId
  );
}

function analyticsConversationsFind(owner, snapshotId) {
  for (var index = 0; index < owner.rows.length; index++) {
    if (owner.rows[index].snapshot_id === snapshotId) {
      return owner.rows[index];
    }
  }
  return null;
}

function analyticsConversationsElement(id) {
  var element = document.getElementById(id);
  if (!element) {
    throw new Error("Missing Analytics conversation element #" + id);
  }
  return element;
}

function analyticsConversationsRequired(options, name) {
  if (!options || !options[name]) {
    throw new TypeError(
      "analyticsConversationsCreate needs options." + name
    );
  }
  return options[name];
}

function analyticsConversationsCallback(options, name) {
  var callback = analyticsConversationsRequired(options, name);
  if (typeof callback !== "function") {
    throw new TypeError(
      "analyticsConversationsCreate needs callback options." + name
    );
  }
  return callback;
}
