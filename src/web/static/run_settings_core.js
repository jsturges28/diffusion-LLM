// Pure schema semantics shared by every Run settings panel.
//
// The registry describes controls, defaults, bounds and presentation.
// This factory interprets that data without touching the DOM or
// browser storage. A caller supplies raw control values and receives
// parsed parameters, validation, summary copy or a frozen snapshot.

"use strict";

function runSettingsCoreCreate() {
  function requireObject(value, name) {
    if (!value || typeof value !== "object") {
      throw new TypeError(name + " must be an object");
    }
    return value;
  }

  function requireSpecs(specs) {
    if (!Array.isArray(specs)) {
      throw new TypeError("Run settings specs must be an array");
    }
    return specs;
  }

  function overrideFor(spec, device) {
    if (!spec.overrides || !device) {
      return null;
    }
    return spec.overrides[device] || null;
  }

  function defaultValue(spec, device) {
    requireObject(spec, "Run settings spec");
    var override = overrideFor(spec, device);
    if (
      override
      && override.default !== null
      && override.default !== undefined
    ) {
      return override.default;
    }
    return spec.default;
  }

  function bounds(spec, device, experimental) {
    requireObject(spec, "Run settings spec");
    var override = overrideFor(spec, device);
    var selected = null;
    if (override) {
      selected = experimental
        ? override.experimental
        : override.recommended;
    }
    if (!selected) {
      selected = experimental
        ? spec.experimental
        : spec.recommended;
    }
    if (!selected) {
      return null;
    }
    return Object.freeze({
      min: selected[0],
      max: selected[1],
    });
  }

  function defaults(specs, device) {
    requireSpecs(specs);
    var result = {};
    for (var index = 0; index < specs.length; index++) {
      result[specs[index].name] = defaultValue(
        specs[index], device
      );
    }
    return Object.freeze(result);
  }

  function included(spec, experimental) {
    return !spec.experimental_only || experimental;
  }

  function parseValue(spec, raw) {
    requireObject(spec, "Run settings spec");
    if (spec.type === "int" || spec.type === "float") {
      return Number(rawText(raw));
    }
    if (spec.type === "bool") {
      return Boolean(raw);
    }
    return raw;
  }

  function parameterValues(specs, rawValues, experimental) {
    requireSpecs(specs);
    requireObject(rawValues, "Run settings raw values");
    var result = {};
    for (var index = 0; index < specs.length; index++) {
      var spec = specs[index];
      if (included(spec, experimental)) {
        result[spec.name] = parseValue(
          spec, rawValues[spec.name]
        );
      }
    }
    return result;
  }

  function numeric(spec) {
    return spec.type === "int" || spec.type === "float";
  }

  function rawText(raw) {
    if (raw === null || raw === undefined) {
      return "";
    }
    return String(raw).trim();
  }

  function numericError(spec, raw, range) {
    var text = rawText(raw);
    var value = Number(text);
    if (text === "" || !Number.isFinite(value)) {
      return spec.label + " is empty or invalid.";
    }
    if (spec.type === "int" && !Number.isInteger(value)) {
      return spec.label + " must be a whole number.";
    }
    if (range && value < range.min) {
      if (value < 0) {
        return spec.label + " cannot be negative.";
      }
      return spec.label + " must be at least "
        + range.min + ".";
    }
    if (range && value > range.max) {
      return spec.label + " must be at most "
        + range.max + ".";
    }
    return "";
  }

  function addInvalid(invalidNames, name) {
    if (invalidNames.indexOf(name) === -1) {
      invalidNames.push(name);
    }
  }

  function validateNumeric(settings) {
    for (
      var index = 0;
      index < settings.specs.length;
      index++
    ) {
      var spec = settings.specs[index];
      if (!included(spec, settings.experimental)) {
        continue;
      }
      if (!numeric(spec)) {
        continue;
      }
      var error = numericError(
        spec,
        settings.rawValues[spec.name],
        bounds(spec, settings.device, settings.experimental)
      );
      if (error) {
        settings.errors.push(error);
        addInvalid(settings.invalidNames, spec.name);
      }
    }
  }

  function hasSpec(specs, name) {
    for (var index = 0; index < specs.length; index++) {
      if (specs[index].name === name) {
        return true;
      }
    }
    return false;
  }

  function invalid(invalidNames, name) {
    return invalidNames.indexOf(name) !== -1;
  }

  function validateDivisibility(settings) {
    if (!hasSpec(settings.specs, "gen_length")) {
      return;
    }
    if (!hasSpec(settings.specs, "block_length")) {
      return;
    }
    if (!hasSpec(settings.specs, "steps")) {
      return;
    }
    var genLength = Number(
      rawText(settings.rawValues.gen_length)
    );
    var blockLength = Number(
      rawText(settings.rawValues.block_length)
    );
    var steps = Number(rawText(settings.rawValues.steps));
    if (
      !invalid(settings.invalidNames, "gen_length")
      && !invalid(settings.invalidNames, "block_length")
      && blockLength > 0
      && genLength % blockLength !== 0
    ) {
      addInvalid(settings.invalidNames, "gen_length");
      addInvalid(settings.invalidNames, "block_length");
      settings.errors.push(
        "Gen Length (" + genLength
        + ") must be divisible by Block Length ("
        + blockLength + ")."
      );
      return;
    }
    validateStepsDivisibility({
      genLength: genLength,
      blockLength: blockLength,
      steps: steps,
      errors: settings.errors,
      invalidNames: settings.invalidNames,
    });
  }

  function validateStepsDivisibility(settings) {
    if (invalid(settings.invalidNames, "gen_length")) {
      return;
    }
    if (invalid(settings.invalidNames, "block_length")) {
      return;
    }
    if (invalid(settings.invalidNames, "steps")) {
      return;
    }
    if (settings.blockLength <= 0) {
      return;
    }
    if (settings.genLength % settings.blockLength !== 0) {
      return;
    }
    var blockCount =
      settings.genLength / settings.blockLength;
    if (
      blockCount > 0
      && settings.steps % blockCount !== 0
    ) {
      addInvalid(settings.invalidNames, "steps");
      settings.errors.push(
        "Steps (" + settings.steps
        + ") must be divisible by num_blocks ("
        + blockCount + ")."
      );
    }
  }

  function validate(options) {
    var settings = requireObject(
      options, "Run settings validation"
    );
    var specs = requireSpecs(settings.specs);
    var rawValues = requireObject(
      settings.rawValues, "Run settings raw values"
    );
    var errors = [];
    var invalidNames = [];
    var validation = {
      specs: specs,
      rawValues: rawValues,
      device: settings.device || null,
      experimental: settings.experimental === true,
      errors: errors,
      invalidNames: invalidNames,
    };
    validateNumeric(validation);
    validateDivisibility(validation);
    Object.freeze(errors);
    Object.freeze(invalidNames);
    return Object.freeze({
      valid: errors.length === 0,
      message: errors.length === 0 ? "" : errors[0],
      errors: errors,
      invalidNames: invalidNames,
    });
  }

  function clampRawValue(spec, raw, device, experimental) {
    if (!numeric(spec)) {
      return raw;
    }
    var range = bounds(spec, device, experimental);
    if (!range) {
      return raw;
    }
    var value = Number(rawText(raw));
    if (!Number.isFinite(value)) {
      return raw;
    }
    if (value < range.min) {
      return String(range.min);
    }
    if (value > range.max) {
      return String(range.max);
    }
    return raw;
  }

  function summaryValue(spec, raw) {
    if (spec.type === "bool") {
      return raw ? "On" : "Off";
    }
    if (spec.type === "select") {
      return prettifyOption(raw);
    }
    return String(raw);
  }

  function prettifyOption(value) {
    var text = String(value).replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  function primarySummary(specs, rawValues) {
    requireSpecs(specs);
    requireObject(rawValues, "Run settings raw values");
    var parts = [];
    for (var index = 0; index < specs.length; index++) {
      var spec = specs[index];
      if (spec.prominence === "primary") {
        parts.push(
          spec.label + " "
          + summaryValue(spec, rawValues[spec.name])
        );
      }
    }
    return parts.length > 0
      ? parts.join(", ")
      : "Model defaults";
  }

  function rawAtDefault(spec, raw, device) {
    var expected = defaultValue(spec, device);
    if (spec.type === "bool") {
      return raw === Boolean(expected);
    }
    return String(raw) === String(expected);
  }

  function valuesAtDefaults(options) {
    var settings = requireObject(
      options, "Run settings defaults comparison"
    );
    if (settings.experimental === true) {
      return false;
    }
    var specs = requireSpecs(settings.specs);
    var rawValues = requireObject(
      settings.rawValues, "Run settings raw values"
    );
    for (var index = 0; index < specs.length; index++) {
      if (
        !rawAtDefault(
          specs[index],
          rawValues[specs[index].name],
          settings.device || null
        )
      ) {
        return false;
      }
    }
    return true;
  }

  function configurationSnapshot(options) {
    var settings = requireObject(
      options, "Run settings snapshot"
    );
    if (
      typeof settings.modelId !== "string"
      || settings.modelId === ""
    ) {
      throw new Error(
        "Conversation actions need an active model"
      );
    }
    if (
      settings.inputMode !== "chat"
      && settings.inputMode !== "completion"
    ) {
      throw new Error(
        "Conversation actions need the model input mode"
      );
    }
    if (
      typeof settings.device !== "string"
      || settings.device === ""
    ) {
      throw new Error(
        "Conversation actions need the active device"
      );
    }
    if (
      typeof settings.schemaId !== "string"
      || !/^[0-9a-f]{64}$/.test(settings.schemaId)
    ) {
      throw new Error(
        "Conversation actions need the generation schema id"
      );
    }
    var specs = requireSpecs(settings.specs);
    var rawValues = requireObject(
      settings.rawValues, "Run settings raw values"
    );
    var experimental = settings.experimental === true;
    var validation = validate({
      specs: specs,
      rawValues: rawValues,
      device: settings.device || null,
      experimental: experimental,
    });
    var parameters = Object.freeze(
      parameterValues(specs, rawValues, experimental)
    );
    return Object.freeze({
      modelId: settings.modelId,
      modelDisplay: String(settings.modelDisplay || ""),
      device: settings.device,
      schemaId: settings.schemaId,
      inputMode: settings.inputMode,
      settingsSummary: primarySummary(specs, rawValues),
      parameters: parameters,
      experimental: experimental,
      valid: validation.valid,
      validationMessage: validation.message,
    });
  }

  return Object.freeze({
    defaultValue: defaultValue,
    defaults: defaults,
    bounds: bounds,
    included: included,
    parseValue: parseValue,
    parameterValues: parameterValues,
    validate: validate,
    clampRawValue: clampRawValue,
    optionLabel: prettifyOption,
    summaryValue: summaryValue,
    primarySummary: primarySummary,
    valuesAtDefaults: valuesAtDefaults,
    configurationSnapshot: configurationSnapshot,
  });
}
