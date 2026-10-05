// Compatibility gate for durable pending assistant reservations.
//
// A saved generation configuration is historical data. It can be
// structurally readable while no longer matching the resident
// registry, so this controller selects retry settings and refuses
// unsafe launches without changing or clamping that data.

"use strict";

function pendingGenerationControllerCreate(options) {
  function callback(name) {
    if (!options || typeof options[name] !== "function") {
      throw new TypeError(
        "pendingGenerationControllerCreate needs options." + name
      );
    }
    return options[name];
  }

  var readState = callback("readState");
  var readActiveModel = callback("readActiveModel");
  var readActiveModelId = callback("readActiveModelId");
  var readActiveDevice = callback("readActiveDevice");
  var readInputMode = callback("readInputMode");
  var readDraftConfiguration = callback(
    "readDraftConfiguration"
  );
  var readDraftValidation = callback("readDraftValidation");
  var modelDisplayName = callback("modelDisplayName");

  function durableConfiguration() {
    return conversationStatePendingGenerationConfiguration(
      readState()
    );
  }

  function settingsValid() {
    var state = readState();
    var conversation = state && state.conversation;
    if (
      conversation
      && conversation.pending_assistant_id !== null
      && durableConfiguration() !== null
    ) {
      return true;
    }
    return readDraftValidation().valid === true;
  }

  function retryConfiguration() {
    var durable = durableConfiguration();
    if (durable !== null) {
      return durable;
    }
    var validation = readDraftValidation();
    if (!validation || validation.valid !== true) {
      return null;
    }
    return readDraftConfiguration();
  }

  function launchBlockReason(configuration) {
    var state = readState();
    var conversation = state && state.conversation;
    if (
      !conversation
      || conversation.pending_assistant_id === null
    ) {
      return "";
    }
    var assistant = conversationStateTailAssistant(state);
    if (assistant === null) {
      return (
        "This pending response is missing its reserved assistant."
      );
    }
    var requiredModel = configuration
      ? configuration.modelId
      : assistant.model_id;
    var requiredInput = configuration
      ? configuration.inputMode
      : assistant.input_mode;
    if (
      assistant.model_id !== requiredModel
      || assistant.input_mode !== requiredInput
    ) {
      return (
        "This pending response no longer matches its reserved"
        + " generation configuration."
      );
    }
    var label = modelDisplayName(requiredModel);
    if (readActiveModelId() !== requiredModel) {
      return (
        "This pending response requires " + label + ". Switch"
        + " back to that model, then press Send."
      );
    }
    if (readInputMode() !== requiredInput) {
      return (
        "This pending response requires " + requiredInput
        + " input mode. Switch back to " + label
        + ", then press Send."
      );
    }
    if (configuration === null) {
      return "";
    }
    if (readActiveDevice() !== configuration.device) {
      return (
        "This pending response requires " + label + " on "
        + deviceLabel(configuration.device)
        + ". Switch back to that device, then press Send."
      );
    }
    return schemaBlockReason(configuration);
  }

  function schemaBlockReason(configuration) {
    var model = readActiveModel();
    var identifiers = model && model.generation_schema_ids;
    var current = identifiers
      && identifiers[configuration.device];
    if (
      typeof current !== "string"
      || !/^[0-9a-f]{64}$/.test(current)
    ) {
      return (
        "The resident model does not expose the Run settings schema"
        + " required by this pending response. Reload the app."
      );
    }
    if (current === configuration.schemaId) {
      return "";
    }
    return (
      "This pending response was saved with a different Run settings"
      + " schema and cannot launch after the model configuration"
      + " changed. Start a new path with the current settings."
    );
  }

  function deviceLabel(device) {
    if (device === "cuda") {
      return "GPU";
    }
    if (device === "cpu") {
      return "CPU";
    }
    return String(device).toUpperCase();
  }

  return Object.freeze({
    durableConfiguration: durableConfiguration,
    launchBlockReason: launchBlockReason,
    retryConfiguration: retryConfiguration,
    settingsValid: settingsValid,
  });
}
