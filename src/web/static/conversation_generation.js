// Pure codec for durable conversation generation configurations.
//
// The server persists snake_case wire data. Run settings and launch
// controllers use the existing camelCase action shape. This factory
// is the only browser owner of either representation's bounds.

"use strict";

function conversationGenerationCreate() {
  var CODEC_VERSION = 1;
  var PARAMETERS_MAX = 64;
  var PARAMETER_NAME_MAX = 128;
  var PARAMETER_STRING_MAX = 1024;
  var NUMBER_ABS_MAX = 1e100;
  var DEVICE_CHARS_MAX = 32;
  var IDENTIFIER_CHARS_MAX = 128;
  var CONFIGURATION_BYTES_MAX = 16 * 1024;
  var SCHEMA_ID_PATTERN = /^[0-9a-f]{64}$/;
  var WIRE_FIELDS = [
    "codec_version",
    "device",
    "experimental",
    "input_mode",
    "model_id",
    "parameters",
    "schema_id",
  ];

  function object(value, name) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new TypeError(name + " must be an object");
    }
    return value;
  }

  function boundedString(value, name, maximum) {
    if (typeof value !== "string" || value === "") {
      throw new TypeError(name + " must be a non-empty string");
    }
    if (value.length > maximum) {
      throw new RangeError(
        name + " exceeds " + maximum + " characters"
      );
    }
    return value;
  }

  function inputMode(value) {
    if (value !== "chat" && value !== "completion") {
      throw new TypeError("input mode must be chat or completion");
    }
    return value;
  }

  function device(value) {
    var selected = boundedString(
      value, "configuration device", DEVICE_CHARS_MAX
    );
    if (selected !== "cpu" && selected !== "cuda") {
      throw new TypeError("configuration device must be cpu or cuda");
    }
    return selected;
  }

  function schemaId(value) {
    if (
      typeof value !== "string"
      || !SCHEMA_ID_PATTERN.test(value)
    ) {
      throw new TypeError(
        "schema id must be 64 lowercase hexadecimal characters"
      );
    }
    return value;
  }

  function boolean(value, name) {
    if (typeof value !== "boolean") {
      throw new TypeError(name + " must be a boolean");
    }
    return value;
  }

  function exactFields(raw) {
    var actual = Object.keys(raw).sort();
    if (
      actual.length !== WIRE_FIELDS.length
      || actual.some(function (field, index) {
        return field !== WIRE_FIELDS[index];
      })
    ) {
      throw new TypeError(
        "generation configuration fields are invalid"
      );
    }
  }

  function parameterValue(value, name) {
    var type = typeof value;
    if (type === "boolean") {
      return value;
    }
    if (type === "string") {
      if (value.length > PARAMETER_STRING_MAX) {
        throw new RangeError(
          "generation parameter " + name + " is too long"
        );
      }
      return value;
    }
    if (type === "number") {
      if (!Number.isFinite(value)) {
        throw new TypeError(
          "generation parameter " + name + " must be finite"
        );
      }
      if (Math.abs(value) > NUMBER_ABS_MAX) {
        throw new RangeError(
          "generation parameter " + name
          + " exceeds its numeric bound"
        );
      }
      return value;
    }
    throw new TypeError(
      "generation parameter " + name + " has an invalid type"
    );
  }

  function parameters(raw) {
    object(raw, "generation configuration parameters");
    var names = Object.keys(raw);
    if (names.length > PARAMETERS_MAX) {
      throw new RangeError(
        "generation configuration has too many parameters"
      );
    }
    var result = {};
    for (var index = 0; index < names.length; index++) {
      var name = boundedString(
        names[index],
        "generation parameter name",
        PARAMETER_NAME_MAX
      );
      result[name] = parameterValue(raw[name], name);
    }
    return Object.freeze(result);
  }

  function identity(
    modelId, configuredInput, expectedModelId, expectedInput
  ) {
    if (
      expectedModelId !== undefined
      && modelId !== expectedModelId
    ) {
      throw new Error(
        "generation configuration changed its reserved model"
      );
    }
    if (
      expectedInput !== undefined
      && configuredInput !== expectedInput
    ) {
      throw new Error(
        "generation configuration changed its reserved input mode"
      );
    }
  }

  function actionConfiguration(settings) {
    var raw = object(settings, "generation configuration");
    var modelId = boundedString(
      raw.modelId,
      "configuration model id",
      IDENTIFIER_CHARS_MAX
    );
    var configuredInput = inputMode(raw.inputMode);
    var result = {
      codecVersion: CODEC_VERSION,
      modelId: modelId,
      inputMode: configuredInput,
      device: device(raw.device),
      schemaId: schemaId(raw.schemaId),
      experimental: boolean(
        raw.experimental, "configuration experimental"
      ),
      parameters: parameters(raw.parameters),
    };
    return Object.freeze(result);
  }

  function toWire(
    settings, expectedModelId, expectedInputMode
  ) {
    var action = actionConfiguration(settings);
    identity(
      action.modelId,
      action.inputMode,
      expectedModelId,
      expectedInputMode
    );
    var result = {
      codec_version: action.codecVersion,
      model_id: action.modelId,
      input_mode: action.inputMode,
      device: action.device,
      schema_id: action.schemaId,
      experimental: action.experimental,
      parameters: action.parameters,
    };
    serializedBound(result);
    return Object.freeze(result);
  }

  function fromWire(
    value, expectedModelId, expectedInputMode
  ) {
    var raw = object(value, "generation configuration");
    exactFields(raw);
    if (
      !Number.isInteger(raw.codec_version)
      || raw.codec_version !== CODEC_VERSION
    ) {
      throw new TypeError(
        "generation configuration codec version is unsupported"
      );
    }
    var modelId = boundedString(
      raw.model_id,
      "configuration model id",
      IDENTIFIER_CHARS_MAX
    );
    var configuredInput = inputMode(raw.input_mode);
    identity(
      modelId,
      configuredInput,
      expectedModelId,
      expectedInputMode
    );
    var result = {
      codecVersion: CODEC_VERSION,
      modelId: modelId,
      inputMode: configuredInput,
      device: device(raw.device),
      schemaId: schemaId(raw.schema_id),
      experimental: boolean(
        raw.experimental, "configuration experimental"
      ),
      parameters: parameters(raw.parameters),
    };
    serializedBound(raw);
    return Object.freeze(result);
  }

  function serializedBound(configuration) {
    var encoded = JSON.stringify(configuration);
    if (utf8Length(encoded) > CONFIGURATION_BYTES_MAX) {
      throw new RangeError(
        "generation configuration exceeds 16384 bytes"
      );
    }
  }

  function utf8Length(value) {
    var length = 0;
    for (var index = 0; index < value.length; index++) {
      var code = value.charCodeAt(index);
      if (code <= 0x7f) {
        length += 1;
      } else if (code <= 0x7ff) {
        length += 2;
      } else if (
        code >= 0xd800
        && code <= 0xdbff
        && index + 1 < value.length
        && value.charCodeAt(index + 1) >= 0xdc00
        && value.charCodeAt(index + 1) <= 0xdfff
      ) {
        length += 4;
        index += 1;
      } else {
        length += 3;
      }
      if (length > CONFIGURATION_BYTES_MAX) {
        return length;
      }
    }
    return length;
  }

  return Object.freeze({
    fromWire: fromWire,
    toWire: toWire,
    actionConfiguration: actionConfiguration,
    isSchemaId: function (value) {
      return typeof value === "string"
        && SCHEMA_ID_PATTERN.test(value);
    },
  });
}
