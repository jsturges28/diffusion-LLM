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
      body.require_tail_xai = input.requireTailXai === true;
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
    list: function () {
      return savedConversationClientJson(
        request,
        "/api/analytics/conversations",
        { method: "GET" }
      );
    },
    metadata: function (snapshotId) {
      return savedConversationClientGet(
        request,
        savedConversationClientBase(snapshotId) + "/metadata"
      );
    },
    turns: function (snapshotId, before, limit) {
      var url = savedConversationClientBase(snapshotId) + "/turns";
      var query = new URLSearchParams();
      if (before) {
        query.set("before", before);
      }
      if (limit) {
        query.set("limit", String(limit));
      }
      var suffix = query.toString();
      return savedConversationClientGet(
        request, suffix ? url + "?" + suffix : url
      );
    },
    rename: function (snapshotId, title, revision) {
      return savedConversationClientJson(
        request,
        savedConversationClientBase(snapshotId),
        {
          method: "PATCH",
          headers: savedConversationClientHeaders(),
          body: JSON.stringify({
            title: savedConversationClientTitle(title),
            expected_title_revision:
              savedConversationClientPositive(
                revision, "title revision"
              ),
          }),
        }
      );
    },
    delete: function (snapshotId) {
      return savedConversationClientJson(
        request,
        savedConversationClientBase(snapshotId),
        { method: "DELETE" }
      );
    },
    pinnedUrl: function (snapshotId, turnId, resource) {
      var allowed = ["metadata", "metrics", "frames"];
      if (allowed.indexOf(resource) === -1) {
        throw new TypeError("unknown pinned run resource");
      }
      return savedConversationClientBase(snapshotId)
        + "/turns/" + encodeURIComponent(
          savedConversationClientId(turnId, "turn id")
        )
        + "/run/" + resource;
    },
  });
}

function savedConversationClientGet(request, url) {
  return savedConversationClientJson(
    request, url, { method: "GET" }
  );
}

function savedConversationClientBase(snapshotId) {
  return "/api/analytics/conversations/"
    + encodeURIComponent(
      savedConversationClientSnapshotId(snapshotId)
    );
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

function savedConversationClientSnapshotId(value) {
  if (typeof value !== "string" || !/^[0-9a-f]{32}$/.test(value)) {
    throw new TypeError(
      "saved conversation id must be 32 lowercase hex digits"
    );
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
