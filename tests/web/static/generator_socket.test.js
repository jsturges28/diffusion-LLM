// The generator socket controller, driven without app.js.
//
// Strategy: load the classic-script factory into a vm with a
// controllable WebSocket and timer queue. Each test drives the same
// callbacks a browser would, so passing proves the controller owns
// mutable transport state privately while preserving URL selection,
// reconnect timing, wire JSON, suppression and failure routing.

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(
  path.join(
    __dirname,
    "..",
    "..",
    "..",
    "src",
    "web",
    "static",
    "generator_socket.js"
  ),
  "utf8"
);

function socketClass() {
  class TestSocket {
    constructor(url) {
      this.url = url;
      this.readyState = TestSocket.CONNECTING;
      this.sent = [];
      this.closeCalls = 0;
      this.onopen = null;
      this.onclose = null;
      this.onerror = null;
      this.onmessage = null;
      TestSocket.opened.push(this);
    }

    open() {
      this.readyState = TestSocket.OPEN;
      this.onopen({ type: "open" });
    }

    close() {
      this.closeCalls += 1;
      this.readyState = TestSocket.CLOSED;
      this.onclose({ type: "close", code: 1000 });
    }

    fail() {
      this.onerror({ type: "error" });
    }

    deliver(raw) {
      this.onmessage({ data: raw });
    }

    send(raw) {
      this.sent.push(raw);
    }
  }

  TestSocket.CONNECTING = 0;
  TestSocket.OPEN = 1;
  TestSocket.CLOSING = 2;
  TestSocket.CLOSED = 3;
  TestSocket.opened = [];
  return TestSocket;
}

function timerQueue() {
  const timers = [];

  function setTimer(callback, delay) {
    const timer = {
      callback: callback,
      delay: delay,
      cancelled: false,
    };
    timers.push(timer);
    return timer;
  }

  function clearTimer(timer) {
    timer.cancelled = true;
  }

  function pending() {
    return timers.filter((timer) => !timer.cancelled);
  }

  function runNext() {
    const timer = pending()[0];
    assert.ok(timer, "expected a pending reconnect");
    timer.cancelled = true;
    timer.callback();
  }

  return {
    setTimer,
    clearTimer,
    pending,
    runNext,
  };
}

function loadFactory(settings) {
  const options = settings || {};
  const Socket = socketClass();
  const timers = timerQueue();
  const context = vm.createContext({
    Error,
    TypeError,
    RangeError,
    JSON,
    Math,
    Object,
    Boolean,
    String,
    WebSocket: Socket,
    location: {
      protocol: options.protocol || "http:",
      host: options.host || "example.test",
    },
    setTimeout: timers.setTimer,
    clearTimeout: timers.clearTimer,
  });
  vm.runInContext(SOURCE, context, {
    filename: "generator_socket.js",
  });
  return { context, Socket, timers };
}

function callbacks(events, overrides) {
  return Object.assign({
    onOpen: () => events.push("open"),
    onClose: () => events.push("close"),
    onMessage: (message) => {
      events.push("message:" + message.type);
    },
    onMalformed: (error) => {
      events.push("malformed:" + error.name);
    },
    onFatal: (error) => {
      events.push("fatal:" + error.message);
    },
  }, overrides || {});
}

function harness(settings, overrides) {
  const loaded = loadFactory(settings);
  const events = [];
  const controller = loaded.context.generatorSocketCreate(
    callbacks(events, overrides)
  );
  return Object.assign(loaded, { controller, events });
}

test("every lifecycle callback is required", () => {
  const { context } = loadFactory();
  const names = [
    "onOpen",
    "onClose",
    "onMessage",
    "onMalformed",
    "onFatal",
  ];

  for (const name of names) {
    const options = callbacks([]);
    delete options[name];
    assert.throws(
      () => context.generatorSocketCreate(options),
      new RegExp("options\\." + name)
    );
  }
});

test("transport state stays private in a frozen controller", () => {
  const { controller, Socket } = harness({
    protocol: "https:",
    host: "generator.test",
  });

  assert.equal(controller.connect(), true);
  assert.equal(Socket.opened[0].url, "wss://generator.test/ws");
  assert.equal(controller.socket, undefined);
  assert.equal(controller.reconnectTimer, undefined);
  assert.equal(controller.reconnectDelayMs, undefined);
  assert.equal(controller.reconnectSuppressed, undefined);
  assert.equal(Object.isFrozen(controller), true);
});

test("reconnects back off to the cap and an open resets them", () => {
  const { controller, Socket, timers, events } = harness();
  const expected = [2000, 4000, 8000, 16000, 16000];

  controller.connect();
  for (const delay of expected) {
    Socket.opened[Socket.opened.length - 1].close();
    assert.equal(timers.pending()[0].delay, delay);
    timers.runNext();
  }

  const current = Socket.opened[Socket.opened.length - 1];
  current.open();
  current.close();

  assert.equal(timers.pending()[0].delay, 2000);
  assert.equal(events.filter((event) => event === "open").length, 1);
});

test("malformed frames are reported and never dispatched", () => {
  const { controller, Socket, events } = harness();
  controller.connect();
  const socket = Socket.opened[0];

  socket.deliver("{");
  socket.deliver("null");
  socket.deliver('{"value": 1}');
  socket.deliver('{"type": "resident", "worker": "one"}');

  assert.deepEqual(events, [
    "malformed:SyntaxError",
    "malformed:TypeError",
    "malformed:TypeError",
    "message:resident",
  ]);
});

test("resident and worker frames keep their arrival order", () => {
  const { controller, Socket, events } = harness();
  controller.connect();
  const socket = Socket.opened[0];

  socket.deliver('{"type": "resident"}');
  socket.deliver('{"type": "model_status"}');

  assert.deepEqual(events, [
    "message:resident",
    "message:model_status",
  ]);
});

test("oversized frames stop before parsing or dispatch", () => {
  const {
    context,
    controller,
    Socket,
    events,
  } = harness();
  context.GENERATOR_SOCKET_MESSAGE_CHARS_MAX = 8;
  controller.connect();

  Socket.opened[0].deliver('{"type": "resident"}');

  assert.deepEqual(events, ["malformed:RangeError"]);
});

test("send checks readiness and serializes one payload", () => {
  const { controller, Socket, events } = harness();
  const payload = { type: "cancel", nested: { active: true } };

  assert.equal(controller.send(payload), false);
  controller.connect();
  assert.equal(controller.send(payload), false);
  Socket.opened[0].open();
  assert.equal(controller.send(payload), true);
  assert.equal(
    Socket.opened[0].sent[0],
    '{"type":"cancel","nested":{"active":true}}'
  );

  const circular = {};
  circular.self = circular;
  assert.equal(controller.send(circular), false);
  assert.match(events[events.length - 1], /^fatal:/);
  assert.throws(() => controller.send(null), /must be an object/);
});

test("suppression cancels retries and release starts one", () => {
  const { controller, timers, events } = harness();
  controller.connect();
  controller.setReconnectSuppressed(true);

  assert.equal(controller.close(), true);
  assert.deepEqual(events, ["close"]);
  assert.equal(timers.pending().length, 0);
  assert.equal(controller.close(), false);

  controller.setReconnectSuppressed(false);
  assert.equal(timers.pending().length, 1);
  assert.equal(timers.pending()[0].delay, 2000);

  controller.setReconnectSuppressed(true);
  assert.equal(timers.pending().length, 0);
});

test("error closes once and lifecycle dispatch stays bounded", () => {
  const { controller, Socket, timers, events } = harness();

  assert.equal(controller.connect(), true);
  assert.equal(controller.connect(), false);
  const socket = Socket.opened[0];
  socket.open();
  socket.deliver('{"type": "model_status"}');
  socket.fail();

  assert.equal(socket.closeCalls, 1);
  assert.deepEqual(events, [
    "open",
    "message:model_status",
    "close",
  ]);
  assert.equal(timers.pending().length, 1);
});

test("a page callback failure reaches the fatal callback", () => {
  const { controller, Socket, events } = harness({}, {
    onMessage: () => {
      throw new Error("page dispatch failed");
    },
  });
  controller.connect();

  Socket.opened[0].deliver('{"type": "frame"}');

  assert.deepEqual(events, ["fatal:page dispatch failed"]);
});
