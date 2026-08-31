// Tests for the collections API client in collections_client.js.
//
// Strategy: load the shipped file into a fresh vm context, the same
// pattern the other browser tests use, and hand it a fake fetch. The
// module takes its fetch as an option precisely so this is possible;
// it touches no DOM and holds no state of its own.
//
// What it is for: the Analytics page used to serialize its whole
// collections array and PUT it, so two windows hydrated from the same
// value each filed a different run and the later write erased the
// earlier one. These functions send the gesture instead, and the
// server applies it to whatever is stored under the lock.
//
// So what matters here is not much more than "the right request goes
// to the right place", but that is exactly the layer where a wrong
// method or a missing encode turns into a silent no-op in a browser.
// What passing proves: each gesture reaches its own endpoint with its
// own verb, the answer is unwrapped to the list, a refusal rejects
// with the server's reason rather than resolving to nothing, and ids
// with awkward characters survive the trip.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "src",
  "web",
  "static",
  "collections_client.js"
);

// A list shaped like the server's answer.
const LIST = [{ id: "papers", name: "Papers", runs: ["run-a"] }];

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "collections_client.js",
  });
  return sandbox;
}

// Records what was asked for and answers with whatever is staged.
function fakeFetch(reply) {
  const calls = [];
  const answer = reply || {
    ok: true,
    body: { success: true, collections: LIST },
  };
  const impl = (url, init) => {
    calls.push({ url: url, init: init || {} });
    return Promise.resolve({
      ok: answer.ok,
      json: () => Promise.resolve(answer.body),
    });
  };
  return { impl: impl, calls: calls };
}

function clientWith(reply) {
  const sandbox = load();
  const fetcher = fakeFetch(reply);
  const client = sandbox.collectionsClientCreate({
    fetchImpl: fetcher.impl,
  });
  return { client: client, calls: fetcher.calls };
}

function bodyOf(call) {
  return JSON.parse(call.init.body);
}

// ---- Each gesture reaches its own endpoint ----

test("listing asks for the collections", async () => {
  const { client, calls } = clientWith();

  const list = await client.list();

  assert.equal(calls[0].url, "/api/collections");
  assert.equal(calls[0].init.method, "GET");
  assert.deepEqual(Array.from(list), LIST);
});

test("creating posts the name", async () => {
  const { client, calls } = clientWith();

  await client.create("Papers");

  assert.equal(calls[0].url, "/api/collections");
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(bodyOf(calls[0]), { name: "Papers" });
});

test("creating with a run files it in the same request", async () => {
  // Two requests could leave the collection made and empty, which is
  // not what naming a collection from a run's dialog asked for.
  const { client, calls } = clientWith();

  await client.create("Papers", "run-a");

  assert.deepEqual(bodyOf(calls[0]), {
    name: "Papers",
    run_id: "run-a",
  });
});

test("creating without a run sends no run key", async () => {
  const { client, calls } = clientWith();

  await client.create("Papers");

  assert.equal("run_id" in bodyOf(calls[0]), false);
});

test("renaming posts to the collection's own path", async () => {
  const { client, calls } = clientWith();

  await client.rename("papers", "Read Later");

  assert.equal(calls[0].url, "/api/collections/papers/rename");
  assert.deepEqual(bodyOf(calls[0]), { name: "Read Later" });
});

test("deleting uses DELETE, not a post", async () => {
  const { client, calls } = clientWith();

  await client.destroy("papers");

  assert.equal(calls[0].url, "/api/collections/papers");
  assert.equal(calls[0].init.method, "DELETE");
});

test("filing a run posts to the collection's runs", async () => {
  const { client, calls } = clientWith();

  await client.addRun("papers", "run-a");

  assert.equal(calls[0].url, "/api/collections/papers/runs");
  assert.deepEqual(bodyOf(calls[0]), { run_id: "run-a" });
});

test("filing a selection is one request", async () => {
  // The whole reason the bulk path exists. Six sequential adds can
  // stop at four and leave a half-applied gesture; one request
  // either files all of them or none.
  const { client, calls } = clientWith();

  await client.addRuns("papers", ["run-a", "run-b", "run-c"]);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/collections/papers/runs");
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(bodyOf(calls[0]), {
    run_ids: ["run-a", "run-b", "run-c"],
  });
});

test("filing a selection sends no single run key", async () => {
  // The server reads run_ids when it is present, so a stray run_id
  // beside it would be silently ignored rather than refused.
  const { client, calls } = clientWith();

  await client.addRuns("papers", ["run-a"]);

  assert.equal("run_id" in bodyOf(calls[0]), false);
});

test("creating with a selection files all of it", async () => {
  const { client, calls } = clientWith();

  await client.createWithRuns("Papers", ["run-a", "run-b"]);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/collections");
  assert.deepEqual(bodyOf(calls[0]), {
    name: "Papers",
    run_ids: ["run-a", "run-b"],
  });
});

test("a bulk collection id is encoded into the path", async () => {
  const { client, calls } = clientWith();

  await client.addRuns("a/b", ["run-a"]);

  assert.equal(calls[0].url, "/api/collections/a%2Fb/runs");
});

test("a refused batch rejects with the reason", async () => {
  const { client } = clientWith({
    ok: false,
    status: 409,
    body: { success: false, reason: "unknown_run", error: "no" },
  });

  await assert.rejects(
    client.addRuns("papers", ["run-a", "ghost"]),
    (error) => error.reason === "unknown_run"
  );
});

test("unfiling a run deletes the membership", async () => {
  const { client, calls } = clientWith();

  await client.removeRun("papers", "run-a");

  assert.equal(
    calls[0].url, "/api/collections/papers/runs/run-a"
  );
  assert.equal(calls[0].init.method, "DELETE");
});

test("the star is one request", async () => {
  // Not a loop over collections. Clearing a filled star touches
  // every collection at once, and split across requests it could
  // stop half way.
  const { client, calls } = clientWith();

  await client.toggleFavorite("run-a");

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/collections/favorite");
  assert.deepEqual(bodyOf(calls[0]), { run_id: "run-a" });
});

// ---- Awkward ids ----

test("ids are encoded into the path", async () => {
  // A run id is a folder name and a collection id is slugged, so
  // neither should carry a slash today. Encoding anyway, because a
  // path built by concatenation is one naming rule away from
  // addressing the wrong resource.
  const { client, calls } = clientWith();

  await client.removeRun("a/b", "c d");

  assert.equal(
    calls[0].url, "/api/collections/a%2Fb/runs/c%20d"
  );
});

// ---- Failure ----

test("a refusal rejects with the server's reason", async () => {
  const { client } = clientWith({
    ok: false,
    body: {
      success: false,
      reason: "collection_limit",
      message: "at most 24 collections",
    },
  });

  await assert.rejects(client.create("One More"), (error) => {
    assert.equal(error.reason, "collection_limit");
    assert.equal(error.message, "at most 24 collections");
    return true;
  });
});

test("a 200 that is not a success is still a refusal", async () => {
  // The page adopts whatever resolves, so resolving on a body that
  // says success is false would replace the list with nothing.
  const { client } = clientWith({
    ok: true,
    body: { success: false, reason: "unknown_run" },
  });

  await assert.rejects(client.addRun("papers", "ghost"));
});

test("a refusal without a reason still carries a message", async () => {
  const { client } = clientWith({ ok: false, body: {} });

  await assert.rejects(client.destroy("papers"), (error) => {
    assert.equal(error.reason, "unavailable");
    assert.ok(error.message.length > 0);
    return true;
  });
});

test("an empty collections field resolves to a list", async () => {
  // Deleting the last collection answers with nothing left, and the
  // caller iterates whatever it gets.
  const { client } = clientWith({
    ok: true,
    body: { success: true },
  });

  const list = await client.destroy("papers");

  assert.deepEqual(Array.from(list), []);
});
