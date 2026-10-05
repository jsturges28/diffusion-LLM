// Shared browser codec for pending generation configurations.
//
// Strategy: exercise the same JSON vectors as Python, then construct
// values JSON cannot represent and every fixed-size boundary. Passing
// proves wire and durable paths share one strict, bounded codec.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const ROOT = path.join(__dirname, "..", "..", "..");
const SOURCE = path.join(
  ROOT, "src", "web", "static", "conversation_generation.js"
);
const VECTORS = JSON.parse(fs.readFileSync(
  path.join(
    ROOT, "tests", "web",
    "generation_configuration_vectors.json"
  ),
  "utf8"
));

function codec() {
  const context = vm.createContext({
    Object,
    Array,
    Number,
    JSON,
    Error,
    TypeError,
    RangeError,
  });
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context);
  return context.conversationGenerationCreate();
}

function host(value) {
  return JSON.parse(JSON.stringify(value));
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test("shared valid vectors round trip exactly", () => {
  const owner = codec();
  for (const vector of VECTORS.valid) {
    const parsed = owner.fromWire(
      vector.wire,
      vector.wire.model_id,
      vector.wire.input_mode
    );
    assert.deepEqual(host(parsed), vector.action, vector.name);
    assert.deepEqual(
      host(owner.toWire(
        parsed,
        vector.wire.model_id,
        vector.wire.input_mode
      )),
      vector.wire,
      vector.name
    );
    assert.equal(Object.isFrozen(parsed), true);
    assert.equal(Object.isFrozen(parsed.parameters), true);
  }
});

test("shared malformed vectors are refused", () => {
  const owner = codec();
  for (const vector of VECTORS.invalid) {
    assert.throws(
      () => owner.fromWire(vector.wire),
      new RegExp(vector.error, "i"),
      vector.name
    );
  }
});

test("reservation identity is checked in both directions", () => {
  const owner = codec();
  const vector = VECTORS.valid[0];
  assert.throws(
    () => owner.fromWire(vector.wire, "other", "chat"),
    /reserved model/
  );
  assert.throws(
    () => owner.toWire(vector.action, "llada", "completion"),
    /reserved input mode/
  );
});

test("parameter count, names, strings, and bytes are bounded", () => {
  const owner = codec();
  const base = clone(VECTORS.valid[0].wire);
  base.parameters = {};
  for (let index = 0; index < 65; index += 1) {
    base.parameters["p" + index] = index;
  }
  assert.throws(() => owner.fromWire(base), /too many/);

  base.parameters = { ["x".repeat(129)]: 1 };
  assert.throws(() => owner.fromWire(base), /128 characters/);

  base.parameters = { text: "x".repeat(1025) };
  assert.throws(() => owner.fromWire(base), /too long/);

  base.parameters = {};
  for (let index = 0; index < 16; index += 1) {
    base.parameters["value_" + index] = "\u00e9".repeat(1024);
  }
  assert.throws(() => owner.fromWire(base), /16384 bytes/);
});

test("non-JSON numeric and value forms are refused", () => {
  const owner = codec();
  const invalid = [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    1e101,
    null,
    {},
    [],
    undefined,
  ];
  for (const value of invalid) {
    const raw = clone(VECTORS.valid[0].wire);
    raw.parameters.temperature = value;
    assert.throws(
      () => owner.fromWire(raw),
      /finite|numeric bound|invalid type/
    );
  }
});

test("schema ids are canonical lowercase SHA-256 forms", () => {
  const owner = codec();
  assert.equal(owner.isSchemaId("a".repeat(64)), true);
  for (const value of [
    "a".repeat(63),
    "a".repeat(65),
    "A".repeat(64),
    "g".repeat(64),
    null,
  ]) {
    assert.equal(owner.isSchemaId(value), false);
  }
});
