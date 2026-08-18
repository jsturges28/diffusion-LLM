// Tests for one reading of what the server says about models.
//
// Strategy: load the shipped file into a fresh vm context, inject a
// fake fetch, and read the payload through the accessors. No DOM and
// no state, so every case is one call.
//
// What passing proves is agreement rather than transport. Four pages
// fetched `/api/models` and each decided for itself what the answer
// meant: whether a model counts as resident, what an absent device
// defaults to, whether a missing `gpu_name` is false or undefined.
// Analytics and Settings implement the same rule for the same link,
// and the second said so in a comment rather than in code. A reading
// that drifts is worse than a fetch written twice, because the pages
// then disagree about one server response while both look right.
//
// The malformed-payload cases are not hypothetical politeness. This
// endpoint answers before a model is resident, when none can load,
// and on a host with no GPU, so absent fields are the normal case
// and every accessor has to have an opinion about them.
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
  "model_client.js"
);

function load() {
  const sandbox = {};
  vm.runInNewContext(fs.readFileSync(SOURCE, "utf8"), sandbox, {
    filename: "model_client.js",
  });
  return sandbox;
}

const LLADA = { id: "llada", capabilities: { model_type: "diffusion" } };
const SMOL = {
  id: "smollm3",
  capabilities: { model_type: "autoregressive" },
};

const RESIDENT = {
  active: "smollm3",
  active_device: "cpu",
  gpu_name: "NVIDIA GeForce RTX 4090",
  models: [LLADA, SMOL],
};

// Answered before anything is resident, which is the state every
// page meets on a cold start.
const IDLE = { active: null, models: [LLADA, SMOL] };

// -- the fetch --

test("the list is fetched from one endpoint", async () => {
  const api = load();
  const asked = [];
  const fake = (url) => {
    asked.push(url);
    return Promise.resolve({ json: () => RESIDENT });
  };

  await api.modelClientLoad(fake);

  assert.deepEqual(asked, ["/api/models"]);
});

test("the payload arrives parsed", async () => {
  const api = load();
  const fake = () => Promise.resolve({ json: () => RESIDENT });

  const info = await api.modelClientLoad(fake);

  assert.equal(api.modelClientActiveId(info), "smollm3");
});

test("a rejected fetch stays rejected", async () => {
  // Each page handles this differently, from a visible error on the
  // menu to a link that simply stays hidden, so the client must not
  // decide for them.
  const api = load();
  const fake = () => Promise.reject(new Error("offline"));

  await assert.rejects(
    () => api.modelClientLoad(fake),
    /offline/
  );
});

// -- is anything resident --

test("a resident model is named", () => {
  const api = load();

  assert.equal(api.modelClientActiveId(RESIDENT), "smollm3");
  assert.equal(api.modelClientHasActive(RESIDENT), true);
});

test("an idle server reports none", () => {
  const api = load();

  assert.equal(api.modelClientActiveId(IDLE), null);
  assert.equal(api.modelClientHasActive(IDLE), false);
});

test("an empty active id is not a model", () => {
  // The gate two pages use to reveal the Generation link, which is
  // honest only when there is something to generate with.
  const api = load();

  assert.equal(api.modelClientHasActive({ active: "" }), false);
});

test("a missing payload is not a model", () => {
  const api = load();

  assert.equal(api.modelClientHasActive(undefined), false);
  assert.equal(api.modelClientHasActive({}), false);
});

// -- the list --

test("the models come back as an array", () => {
  const api = load();

  assert.equal(api.modelClientList(RESIDENT).length, 2);
});

test("a payload with no models reads as empty", () => {
  const api = load();

  assert.equal(api.modelClientList({}).length, 0);
  assert.equal(api.modelClientList(undefined).length, 0);
});

test("a models field of the wrong shape reads as empty", () => {
  const api = load();

  assert.equal(api.modelClientList({ models: "llada" }).length, 0);
});

test("a model is found by id", () => {
  const api = load();

  assert.equal(api.modelClientFind(RESIDENT, "llada"), LLADA);
  assert.equal(api.modelClientFind(RESIDENT, "mamba"), null);
});

test("the resident model itself is one lookup", () => {
  const api = load();

  assert.equal(api.modelClientActiveModel(RESIDENT), SMOL);
});

test("nothing resident means no resident model", () => {
  const api = load();

  assert.equal(api.modelClientActiveModel(IDLE), null);
});

test("an active id naming an absent model reads as none", () => {
  // The server can name a model the list does not carry if the two
  // are computed apart; guessing the first entry would attribute a
  // run to the wrong model.
  const api = load();

  assert.equal(
    api.modelClientActiveModel({ active: "mamba", models: [LLADA] }),
    null
  );
});

// -- device and GPU --

test("the device is the one the model actually got", () => {
  const api = load();

  assert.equal(api.modelClientActiveDevice(RESIDENT), "cpu");
});

test("an absent device is null, not a guess", () => {
  // Defaulting to cuda here would label a run with hardware it never
  // touched, which is the failure DATA-04 exists to prevent.
  const api = load();

  assert.equal(api.modelClientActiveDevice(IDLE), null);
  assert.equal(api.modelClientActiveDevice({}), null);
});

test("a GPU is reported when the server names one", () => {
  const api = load();

  assert.equal(api.modelClientGpuPresent(RESIDENT), true);
  assert.match(api.modelClientGpuName(RESIDENT), /RTX/);
});

test("a GPU-less host reports false, not undefined", () => {
  // Passed straight into rendering as a flag, where undefined and
  // false read the same until something compares it.
  const api = load();

  assert.equal(api.modelClientGpuPresent({}), false);
  assert.equal(api.modelClientGpuName({}), null);
});

test("an empty GPU name is no GPU", () => {
  const api = load();

  assert.equal(api.modelClientGpuPresent({ gpu_name: "" }), false);
});
