// Pending generation compatibility and retry selection.
//
// Strategy: load a durable pending turn through the real reducer,
// then vary only resident model metadata. Passing proves schema
// evolution is refused while matching cross-window recovery remains
// exact and legacy turns use a validated Draft snapshot.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);
const ID = "a".repeat(32);
const BRANCH = "b_" + "b".repeat(32);
const USER =
  "t_" + "b".repeat(32) + "_00000001_1111111111111111";
const ASSISTANT =
  "t_" + "b".repeat(32) + "_00000002_2222222222222222";
const SCHEMA_ID = "7".repeat(64);

function load() {
  const context = vm.createContext({});
  for (const name of [
    "conversation_generation.js",
    "conversation_state.js",
    "pending_generation_controller.js",
  ]) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context
    );
  }
  return context;
}

function wireConfiguration() {
  return {
    codec_version: 1,
    model_id: "llada",
    input_mode: "chat",
    device: "cuda",
    schema_id: SCHEMA_ID,
    experimental: false,
    parameters: { temperature: 0.25 },
  };
}

function pendingState(api, metadata) {
  const conversation = {
    schema_version: 2,
    id: ID,
    title: "Pending",
    branch_id: BRANCH,
    branch_revision: 2,
    revision: 2,
    catalog_revision: 1,
    default_branch_id: BRANCH,
    turn_count: 2,
    tail_turn_id: ASSISTANT,
    tail_version: 1,
    pending_assistant_id: ASSISTANT,
  };
  return api.conversationStateReduce(
    api.conversationStateCreate(),
    {
      type: "loaded",
      conversation,
      page: {
        schema_version: 2,
        conversation_id: ID,
        branch_id: BRANCH,
        branch_revision: 2,
        revision: 2,
        catalog_revision: 1,
        default_branch_id: BRANCH,
        turns: [
          {
            turn_id: USER,
            branch_id: BRANCH,
            index: 1,
            version: 1,
            role: "user",
            text: "question",
            partial: false,
            model_id: null,
            input_mode: null,
            context_pack: {},
            metadata: {},
            run_link: null,
          },
          {
            turn_id: ASSISTANT,
            branch_id: BRANCH,
            index: 2,
            version: 1,
            role: "assistant",
            text: "",
            partial: true,
            model_id: "llada",
            input_mode: "chat",
            context_pack: {},
            metadata,
            run_link: null,
          },
        ],
        next_before: null,
        has_more: false,
        branch_points: [],
      },
    }
  );
}

function harness(settings) {
  const api = load();
  let state = pendingState(
    api,
    settings.legacy
      ? {}
      : { pending_generation_v1: wireConfiguration() }
  );
  let model = {
    id: "llada",
    display_name: "LLaDA",
    generation_schema_ids: {
      cuda: settings.schemaId || SCHEMA_ID,
    },
  };
  const draft = {
    modelId: "llada",
    inputMode: "chat",
    device: "cuda",
    schemaId: settings.schemaId || SCHEMA_ID,
    experimental: false,
    parameters: { temperature: 0.75 },
  };
  const controller = api.pendingGenerationControllerCreate({
    readState: () => state,
    readActiveModel: () => model,
    readActiveModelId: () => settings.modelId || "llada",
    readActiveDevice: () => settings.device || "cuda",
    readInputMode: () => "chat",
    readDraftConfiguration: () => draft,
    readDraftValidation: () => ({
      valid: settings.draftValid !== false,
    }),
    modelDisplayName: () => "LLaDA",
  });
  return {
    controller,
    setState(value) { state = value; },
    setModel(value) { model = value; },
  };
}

test("matching durable settings recover exactly", () => {
  const h = harness({});
  const configuration = h.controller.retryConfiguration();

  assert.equal(configuration.parameters.temperature, 0.25);
  assert.equal(h.controller.settingsValid(), true);
  assert.equal(
    h.controller.launchBlockReason(configuration),
    ""
  );
});

test("schema evolution refuses a durable pending launch", () => {
  const h = harness({ schemaId: "8".repeat(64) });
  const configuration = h.controller.retryConfiguration();

  assert.match(
    h.controller.launchBlockReason(configuration),
    /different Run settings schema.*cannot launch.*changed/
  );
});

test("legacy pending turns select only a valid Draft", () => {
  const valid = harness({ legacy: true });
  const invalid = harness({ legacy: true, draftValid: false });

  assert.equal(
    valid.controller.retryConfiguration().parameters.temperature,
    0.75
  );
  assert.equal(invalid.controller.retryConfiguration(), null);
  assert.equal(invalid.controller.settingsValid(), false);
});

test("model and device mismatches keep switch guidance", () => {
  const model = harness({ modelId: "other" });
  const device = harness({ device: "cpu" });

  assert.match(
    model.controller.launchBlockReason(
      model.controller.retryConfiguration()
    ),
    /requires LLaDA.*Switch back/
  );
  assert.match(
    device.controller.launchBlockReason(
      device.controller.retryConfiguration()
    ),
    /requires LLaDA on GPU.*Switch back/
  );
});
