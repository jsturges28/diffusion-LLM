// Strict REST client for immutable conversation snapshots.

"use strict";

function savedConversationClientCreate(options) {
  if (!options || typeof options.request !== "function") {
    throw new TypeError(
      "savedConversationClientCreate needs options.request"
    );
  }
  var request = options.request;
  return Object.freeze({
    preview: function (head) {
      return savedConversationClientJson(
        request,
        "/api/analytics/conversations/preview",
        {
          method: "POST",
          headers: savedConversationClientHeaders(),
          body: JSON.stringify(
            savedConversationClientHead(head)
          ),
        }
      );
    },
    create: function (input) {
      var body = savedConversationClientHead(input);
      body.operation_id = savedConversationClientOperation(
        input.operationId
      );
      body.title = savedConversationClientTitle(input.title);
      return savedConversationClientJson(
        request,
        "/api/analytics/conversations",
        {
          method: "POST",
          headers: savedConversationClientHeaders(),
          body: JSON.stringify(body),
        }
      );
    },
  });
}

function savedConversationClientHead(input) {
  if (!input || typeof input !== "object") {
    throw new TypeError("saved conversation head must be an object");
  }
  var head = {
    conversation_id: savedConversationClientId(
      input.conversation_id, "conversation id"
    ),
    branch_id: savedConversationClientId(
      input.branch_id, "branch id"
    ),
    branch_revision: savedConversationClientPositive(
      input.branch_revision, "branch revision"
    ),
    turn_count: savedConversationClientPositive(
      input.turn_count, "turn count"
    ),
    tail_turn_id: savedConversationClientId(
      input.tail_turn_id, "tail turn id"
    ),
    tail_version: savedConversationClientPositive(
      input.tail_version, "tail version"
    ),
  };
  if (head.turn_count % 2 !== 0) {
    throw new TypeError("saved conversation must end on an assistant");
  }
  return head;
}

function savedConversationClientJson(request, url, init) {
  return Promise.resolve(request(url, init)).then(function (response) {
    return Promise.resolve(response.json()).then(function (body) {
      if (!response.ok) {
        var message = body && typeof body.error === "string"
          ? body.error
          : "Saved conversation request failed";
        var error = new Error(message);
        error.status = response.status;
        throw error;
      }
      return body;
    });
  });
}

function savedConversationClientId(value, name) {
  if (typeof value !== "string" || value === "") {
    throw new TypeError(name + " must be a non-empty string");
  }
  return value;
}

function savedConversationClientPositive(value, name) {
  if (!Number.isInteger(value) || value < 1) {
    throw new TypeError(name + " must be a positive integer");
  }
  return value;
}

function savedConversationClientOperation(value) {
  if (typeof value !== "string" || !/^[0-9a-f]{32}$/.test(value)) {
    throw new TypeError("operation id must be 32 lowercase hex digits");
  }
  return value;
}

function savedConversationClientTitle(value) {
  if (typeof value !== "string") {
    throw new TypeError("saved conversation title must be a string");
  }
  var title = value.trim();
  if (title === "" || title.length > 200) {
    throw new TypeError(
      "saved conversation title must contain 1 to 200 characters"
    );
  }
  return title;
}

function savedConversationClientHeaders() {
  return { "Content-Type": "application/json" };
}
