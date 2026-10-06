// Strict request shapes for immutable conversation snapshots.
//
// Strategy: load the shipped client alone and capture every fetch.
// Passing proves preview and create share one exact-head contract,
// create adds only title/idempotency fields, and HTTP errors retain
// their status for conflict handling.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..", "..", "..",
  "src", "web", "static",
  "saved_conversation_client.js"
);

function load() {
  const context = vm.createContext({
    Promise,
    Error,
    TypeError,
    JSON,
    URLSearchParams,
    encodeURIComponent,
  });
  vm.runInContext(
    fs.readFileSync(SOURCE, "utf8"),
    context,
    { filename: SOURCE }
  );
  return context;
}

function head() {
  return {
    conversation_id: "a".repeat(32),
    branch_id: "b_" + "b".repeat(32),
    branch_revision: 4,
    turn_count: 2,
    tail_turn_id: "c".repeat(32),
    tail_version: 2,
  };
}

test("preview sends only the exact selected head", async () => {
  const context = load();
  const requests = [];
  const client = context.savedConversationClientCreate({
    request(url, init) {
      requests.push({ url, init });
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ exchange_count: 1 }),
      });
    },
  });

  await client.preview(head());

  assert.equal(
    requests[0].url,
    "/api/analytics/conversations/preview"
  );
  assert.deepEqual(
    JSON.parse(requests[0].init.body),
    head()
  );
});

test("create adds title and stable operation identity", async () => {
  const context = load();
  let body = null;
  const client = context.savedConversationClientCreate({
    request(url, init) {
      assert.equal(url, "/api/analytics/conversations");
      body = JSON.parse(init.body);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ snapshot_id: "d".repeat(32) }),
      });
    },
  });

  await client.create(Object.assign({}, head(), {
    operationId: "e".repeat(32),
    title: "  Investigation  ",
  }));

  assert.equal(body.operation_id, "e".repeat(32));
  assert.equal(body.title, "Investigation");
  assert.equal(body.branch_revision, 4);
});

test("HTTP conflicts retain their status", async () => {
  const context = load();
  const client = context.savedConversationClientCreate({
    request() {
      return Promise.resolve({
        ok: false,
        status: 409,
        json: () => Promise.resolve({ error: "head moved" }),
      });
    },
  });

  await assert.rejects(
    client.preview(head()),
    (error) => {
      assert.equal(error.message, "head moved");
      assert.equal(error.status, 409);
      return true;
    }
  );
});

test("malformed heads fail before a request", () => {
  const context = load();
  let requested = false;
  const client = context.savedConversationClientCreate({
    request() {
      requested = true;
    },
  });

  assert.throws(
    () => client.preview(Object.assign(head(), { turn_count: 3 })),
    { name: "TypeError" }
  );
  assert.equal(requested, false);
});

test("Analytics reads and pinned URLs stay under snapshot scope",
  async () => {
  const context = load();
  const urls = [];
  const client = context.savedConversationClientCreate({
    request(url) {
      urls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve([]),
      });
    },
  });
  const snapshotId = "f".repeat(32);

  await client.list();
  await client.turns(snapshotId, "51", 50);
  const pinned = client.pinnedUrl(
    snapshotId, "turn/id", "frames"
  );

  assert.equal(urls[0], "/api/analytics/conversations");
  assert.match(urls[1], /before=51/);
  assert.match(urls[1], /limit=50/);
  assert.equal(
    pinned,
    "/api/analytics/conversations/" + snapshotId
      + "/turns/turn%2Fid/run/frames"
  );
});
