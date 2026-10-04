// The generator's WebSocket transport.
//
// Loaded as a classic script before app.js. The returned controller
// owns the live socket, reconnect policy and wire encoding in its
// closure. The page supplies lifecycle and message callbacks, so no
// model, run or rendering semantics live here.

"use strict";

var GENERATOR_SOCKET_RECONNECT_DELAY_MS = 2000;
var GENERATOR_SOCKET_RECONNECT_DELAY_MS_MAX = 16000;
// A browser hands message events over as complete strings. Keep a
// corrupt peer from making JSON parsing and page dispatch unbounded;
// valid run frames remain far below this ceiling.
var GENERATOR_SOCKET_MESSAGE_CHARS_MAX = 64 * 1024 * 1024;

if (!(GENERATOR_SOCKET_RECONNECT_DELAY_MS > 0)) {
  throw new Error(
    "Generator socket reconnect delay must be positive"
  );
}
if (
  GENERATOR_SOCKET_RECONNECT_DELAY_MS_MAX
  < GENERATOR_SOCKET_RECONNECT_DELAY_MS
) {
  throw new Error("Generator socket reconnect cap is too small");
}
if (!(GENERATOR_SOCKET_MESSAGE_CHARS_MAX > 0)) {
  throw new Error("Generator socket message bound must be positive");
}

function generatorSocketCreate(options) {
  var onOpen = generatorSocketRequiredCallback(
    options, "onOpen"
  );
  var onClose = generatorSocketRequiredCallback(
    options, "onClose"
  );
  var onMessage = generatorSocketRequiredCallback(
    options, "onMessage"
  );
  var onMalformed = generatorSocketRequiredCallback(
    options, "onMalformed"
  );
  var onFatal = generatorSocketRequiredCallback(
    options, "onFatal"
  );

  var socket = null;
  var reconnectTimer = null;
  var reconnectDelayMs =
    GENERATOR_SOCKET_RECONNECT_DELAY_MS;
  var reconnectSuppressed = false;

  function connect() {
    if (generatorSocketIsLive(socket)) {
      return false;
    }
    cancelReconnect();
    var candidate;
    try {
      candidate = new WebSocket(generatorSocketUrl());
    } catch (error) {
      reportFatal(error);
      scheduleReconnect();
      return false;
    }
    socket = candidate;
    bind(candidate);
    return true;
  }

  function bind(candidate) {
    candidate.onopen = function (event) {
      opened(candidate, event);
    };
    candidate.onclose = function (event) {
      closed(candidate, event);
    };
    candidate.onerror = function (event) {
      errored(candidate, event);
    };
    candidate.onmessage = function (event) {
      received(candidate, event);
    };
  }

  function opened(candidate, event) {
    if (candidate !== socket) {
      return;
    }
    reconnectDelayMs =
      GENERATOR_SOCKET_RECONNECT_DELAY_MS;
    callLifecycle(onOpen, event);
  }

  function closed(candidate, event) {
    if (candidate !== socket) {
      return;
    }
    socket = null;
    callLifecycle(onClose, event);
    scheduleReconnect();
  }

  function errored(candidate, event) {
    if (candidate !== socket) {
      return;
    }
    try {
      candidate.close();
    } catch (error) {
      socket = null;
      reportFatal(error);
      callLifecycle(onClose, event);
      scheduleReconnect();
    }
  }

  function received(candidate, event) {
    if (candidate !== socket) {
      return;
    }
    if (!event || typeof event.data !== "string") {
      reportMalformed(
        new TypeError("Generator socket frame must be text")
      );
      return;
    }
    if (event.data.length > GENERATOR_SOCKET_MESSAGE_CHARS_MAX) {
      reportMalformed(
        new RangeError("Generator socket frame exceeds its bound")
      );
      return;
    }
    var message;
    try {
      message = JSON.parse(event.data);
    } catch (error) {
      reportMalformed(error);
      return;
    }
    if (!generatorSocketMessageValid(message)) {
      reportMalformed(
        new TypeError("Generator socket frame needs a message type")
      );
      return;
    }
    try {
      onMessage(message);
    } catch (error) {
      reportFatal(error);
    }
  }

  function callLifecycle(callback, event) {
    try {
      callback(event);
    } catch (error) {
      reportFatal(error);
    }
  }

  function reportMalformed(error) {
    try {
      onMalformed(generatorSocketError(error));
    } catch (callbackError) {
      reportFatal(callbackError);
    }
  }

  function reportFatal(error) {
    onFatal(generatorSocketError(error));
  }

  function scheduleReconnect() {
    if (reconnectSuppressed || reconnectTimer !== null) {
      return;
    }
    var delayMs = reconnectDelayMs;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, delayMs);
    reconnectDelayMs = Math.min(
      reconnectDelayMs * 2,
      GENERATOR_SOCKET_RECONNECT_DELAY_MS_MAX
    );
  }

  function cancelReconnect() {
    if (reconnectTimer === null) {
      return;
    }
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function setReconnectSuppressed(suppressed) {
    if (typeof suppressed !== "boolean") {
      throw new TypeError(
        "Generator socket suppression must be boolean"
      );
    }
    var wasSuppressed = reconnectSuppressed;
    reconnectSuppressed = suppressed;
    if (suppressed) {
      cancelReconnect();
      return;
    }
    if (wasSuppressed && socket === null) {
      scheduleReconnect();
    }
  }

  function isReady() {
    return Boolean(
      socket && socket.readyState === WebSocket.OPEN
    );
  }

  function send(payload) {
    if (!isReady()) {
      return false;
    }
    if (
      !payload
      || typeof payload !== "object"
      || Array.isArray(payload)
    ) {
      throw new TypeError(
        "Generator socket payload must be an object"
      );
    }
    var encoded;
    try {
      encoded = JSON.stringify(payload);
    } catch (error) {
      reportFatal(error);
      return false;
    }
    try {
      socket.send(encoded);
    } catch (error) {
      reportFatal(error);
      return false;
    }
    return true;
  }

  function close() {
    if (socket === null) {
      return false;
    }
    var candidate = socket;
    try {
      candidate.close();
    } catch (error) {
      if (candidate === socket) {
        socket = null;
      }
      reportFatal(error);
      callLifecycle(onClose, null);
      scheduleReconnect();
      return false;
    }
    return true;
  }

  return Object.freeze({
    connect: connect,
    close: close,
    isReady: isReady,
    send: send,
    setReconnectSuppressed: setReconnectSuppressed,
  });
}

function generatorSocketRequiredCallback(options, name) {
  if (!options || typeof options[name] !== "function") {
    throw new TypeError(
      "generatorSocketCreate needs options." + name
    );
  }
  return options[name];
}

function generatorSocketUrl() {
  var protocol =
    location.protocol === "https:" ? "wss:" : "ws:";
  return protocol + "//" + location.host + "/ws";
}

function generatorSocketIsLive(socket) {
  if (socket === null) {
    return false;
  }
  return socket.readyState === WebSocket.CONNECTING
    || socket.readyState === WebSocket.OPEN
    || socket.readyState === WebSocket.CLOSING;
}

function generatorSocketMessageValid(message) {
  return Boolean(
    message
    && typeof message === "object"
    && !Array.isArray(message)
    && typeof message.type === "string"
    && message.type !== ""
  );
}

function generatorSocketError(error) {
  if (error instanceof Error) {
    return error;
  }
  return new Error(String(error));
}
