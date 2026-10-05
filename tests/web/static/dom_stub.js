// A DOM small enough to load a page script, and nothing more.
//
// The browser modules with tests beside them are all dependency-light
// by design: they touch no DOM, so a bare `vm` context runs them. The
// page scripts are the opposite, and that gap is not academic. Three
// defects in one session shipped with green suites because the tests
// exercised a convenient stand-in rather than the thing: a chooser
// whose rows rendered and could not be clicked, a status decoder fed
// ints where Qt sends an enum, and a poll cadence whose injected
// intervals proved nothing about its defaults.
//
// Source inspection cannot close that gap. It reads what a file
// *contains*, never what the browser hands to it, so anything about
// which value reaches which function is invisible to it.
//
// This is the smallest thing that can: enough of `document` and
// `window` for a page script to finish loading, plus a record of the
// listeners it registered so a test can fire one. It is deliberately
// not a browser. Layout, styling, and event propagation are all
// absent, so a test that needs any of those wants real hardware and
// a manual item instead.
//
// The cost is that it breaks when a page reaches for a global this
// does not have, which is why `loadPage` fails loudly and names the
// global rather than returning a half-built context. A stub that
// quietly loads two thirds of a file would be worse than none.

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC_DIR = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);

// The generator page's own script order, minus the charting vendors,
// which are stubbed below. Kept in this order because these are
// classic scripts sharing one scope: loaded out of order, a later
// file's globals are undefined when an earlier one runs.
const GENERATOR_SCRIPTS = [
  "custom_select.js",
  "reduced_motion.js",
  "persist.js",
  "overlays.js",
  "activation_progress.js",
  "activation_client.js",
  "wire_errors.js",
  "model_client.js",
  "run_frames.js",
  "run_candidates.js",
  "run_snapshot.js",
  "conversation_state.js",
  "conversation_client.js",
  "conversation_view.js",
  "generator_run.js",
  "generator_socket.js",
  "candidate_flicker.js",
  "run_phases.js",
  "download_client.js",
  "download_toast.js",
  "generator_composer.js",
  "generator_model_panel.js",
  "generator_chrome.js",
  "generator_canvas.js",
  "generator_readouts.js",
  "generator_candidates.js",
  "generator_edit.js",
  "generator_watermark.js",
  "generator_modals.js",
  "app.js",
];

// The other pages' own orders, kept here for the same reason and so
// that each exists once: a test copying its own would go on passing
// against a page that had moved on. Analytics loads the charting
// vendors first, and those are stubbed below rather than listed.
// tests/web/test_page_script_lists.py holds each list to its page.
const ANALYTICS_SCRIPTS = [
  "custom_select.js",
  "reduced_motion.js",
  "persist.js",
  "overlays.js",
  "overlay_series.js",
  "chart_support.js",
  "line_charts.js",
  "run_candidates.js",
  "candidate_flicker.js",
  "token_viewer.js",
  "detail_requests.js",
  "collections_client.js",
  "download_client.js",
  "download_toast.js",
  "analytics.js",
];

const MENU_SCRIPTS = [
  "reduced_motion.js",
  "persist.js",
  "activation_progress.js",
  "model_client.js",
  "activation_client.js",
  "download_client.js",
  "download_toast.js",
  "menu.js",
];

const SETTINGS_SCRIPTS = [
  "custom_select.js",
  "reduced_motion.js",
  "persist.js",
  "overlays.js",
  "download_client.js",
  "download_toast.js",
  "settings.js",
];

const VISION_SCRIPTS = [
  "vision.js",
];

// Chart's config assignment walks arbitrary nested paths and none of
// it reaches the behaviour under test, so the stub says yes to
// everything rather than enumerating what the page happens to set.
function permissive() {
  return new Proxy({}, {
    get(target, key) {
      if (key === Symbol.toPrimitive || key === "toString") {
        return () => "";
      }
      if (!(key in target)) {
        target[key] = permissive();
      }
      return target[key];
    },
    set(target, key, value) {
      target[key] = value;
      return true;
    },
  });
}

function makeElement(id) {
  const element = {
    id,
    tag: null,
    parent: null,
    children: [],
    listeners: {},
    attributes: {},
    dataset: {},
    // A plain bag plus the two methods CSS custom properties need.
    // Page code sets `--name` values through these, which a bare
    // object silently is not able to accept.
    style: {
      setProperty(name, value) { this[name] = String(value); },
      removeProperty(name) { delete this[name]; },
      getPropertyValue(name) {
        return name in this ? this[name] : "";
      },
    },
    value: "",
    files: [],
    checked: false,
    disabled: false,
    hidden: false,
    textContent: "",
    classes: new Set(),
    scrollTop: 0,
    scrollHeight: 0,
    offsetWidth: 0,
    clientWidth: 0,
    isConnected: true,
  };

  // Backed by the same set as `classList`, because the two are one
  // thing in a real DOM and code freely mixes them: a widget that
  // builds a row with `className = "a b"` and a test that asks
  // `classList.contains("b")` must agree, and when they did not, the
  // test saw an element with no classes at all.
  Object.defineProperty(element, "className", {
    get: () => Array.from(element.classes).join(" "),
    set: (value) => {
      // Mutated rather than replaced, so nothing holding the set
      // ends up looking at an orphan.
      element.classes.clear();
      for (const name of String(value).split(/\s+/)) {
        if (name) {
          element.classes.add(name);
        }
      }
    },
    enumerable: true,
  });

  element.classList = {
    add: (...names) => names.forEach((n) => element.classes.add(n)),
    remove: (...names) =>
      names.forEach((n) => element.classes.delete(n)),
    contains: (name) => element.classes.has(name),
    toggle: (name, force) => {
      const on = force === undefined
        ? !element.classes.has(name)
        : !!force;
      if (on) {
        element.classes.add(name);
      } else {
        element.classes.delete(name);
      }
      return on;
    },
  };

  element.addEventListener = (type, fn) => {
    (element.listeners[type] = element.listeners[type] || []).push(fn);
  };
  element.removeEventListener = (type, fn) => {
    const list = element.listeners[type] || [];
    const at = list.indexOf(fn);
    if (at !== -1) {
      list.splice(at, 1);
    }
  };
  element.dispatch = (type, event) => {
    for (const fn of (element.listeners[type] || []).slice()) {
      fn(Object.assign({ target: element }, event || {}));
    }
  };
  // The name the page code uses, for a widget telling its caller
  // something changed. `dispatch` above is the test-facing spelling
  // that takes a type and a plain object; this takes an Event.
  element.dispatchEvent = (event) => {
    element.dispatch(event && event.type, event);
    return true;
  };

  element.appendChild = (child) => {
    element.children.push(child);
    child.parent = element;
    return child;
  };
  element.removeChild = (child) => {
    const at = element.children.indexOf(child);
    if (at !== -1) {
      element.children.splice(at, 1);
    }
    return child;
  };
  element.remove = () => {
    if (element.parent) {
      element.parent.removeChild(element);
    }
  };
  element.replaceChildren = (...kids) => {
    element.children = kids.slice();
  };
  element.insertBefore = (child, before) => {
    const at = element.children.indexOf(before);
    element.children.splice(
      at === -1 ? element.children.length : at, 0, child
    );
    child.parent = element;
    return child;
  };
  Object.defineProperty(element, "firstChild", {
    get: () => element.children[0] || null,
  });
  Object.defineProperty(element, "lastChild", {
    get: () =>
      element.children[element.children.length - 1] || null,
  });
  Object.defineProperty(element, "parentNode", {
    get: () => element.parent,
  });

  element.setAttribute = (key, value) => {
    element.attributes[key] = String(value);
  };
  element.getAttribute = (key) =>
    key in element.attributes ? element.attributes[key] : null;
  element.removeAttribute = (key) => {
    delete element.attributes[key];
  };
  element.hasAttribute = (key) => key in element.attributes;

  // Walks real parents, which is what makes delegated handlers
  // behave: a click lands on a child and the handler asks the
  // ancestor chain for the thing it cares about.
  element.closest = (selector) => {
    let node = element;
    while (node) {
      if (matches(node, selector)) {
        return node;
      }
      node = node.parent;
    }
    return null;
  };
  element.matches = (selector) => matches(element, selector);
  // Self or descendant, which is how outside-click handlers ask
  // "did this land on me".
  element.contains = (node) => {
    let walk = node;
    while (walk) {
      if (walk === element) {
        return true;
      }
      walk = walk.parent;
    }
    return false;
  };
  // A real depth-first search over children, because returning null
  // unconditionally is not neutral: page code reads `querySelector`
  // to find a reference element to measure against, and a null makes
  // it skip the measuring entirely. A test then proves nothing about
  // sizing while looking like it does.
  element.querySelector = (selector) =>
    element.querySelectorAll(selector)[0] || null;
  element.querySelectorAll = (selector) => {
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (matches(child, selector)) {
          found.push(child);
        }
        walk(child);
      }
    };
    walk(element);
    return found;
  };
  element.focus = () => {};
  element.blur = () => {};
  element.setSelectionRange = () => {};
  element.scrollIntoView = () => {};
  element.click = () => { element.dispatch("click"); };
  // The menu's background video is autoplayed and paused from script.
  // A resolved promise rather than a bare function because `play`
  // returns one and callers may attach a catch for autoplay refusal.
  element.play = () => Promise.resolve();
  element.pause = () => {};
  element.load = () => {};

  // Enough <dialog> for the modals. `open` is a property in the real
  // thing too, so page code reads it the same way. `close` fires its
  // event synchronously here where a browser queues a task: tests
  // asserting cleanup would otherwise need a tick, and the ordering
  // that difference could hide is called out where it matters in
  // analytics.js rather than being something a test can catch.
  element.open = false;
  // Not a real DOM property. It records which of the two open methods
  // was used, because they differ in exactly the way this migration
  // is about: only `showModal` traps focus, makes the rest of the
  // page inert and answers Escape. Without recording it, swapping one
  // for the other is invisible to a test and removes the whole point.
  element.openedModally = false;
  element.showModal = () => {
    if (element.open) {
      throw new Error("showModal on an already-open dialog");
    }
    element.open = true;
    element.openedModally = true;
  };
  element.show = () => {
    element.open = true;
    element.openedModally = false;
  };
  element.close = (returnValue) => {
    if (!element.open) {
      return;
    }
    element.open = false;
    element.openedModally = false;
    element.returnValue = returnValue || "";
    element.dispatch("close");
  };
  element.getBoundingClientRect = () => ({
    top: 0, left: 0, right: 0, bottom: 0, width: 0, height: 0,
  });
  // A real object rather than the permissive proxy: canvas code
  // calls these, and a proxy hands back an object where a function
  // was wanted. `measureText` returns a width proportional to the
  // string so a caller that sizes something gets a monotonic answer
  // instead of a constant.
  element.getContext = () => ({
    canvas: element,
    font: "",
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    globalAlpha: 1,
    measureText: (text) => ({ width: String(text || "").length * 7 }),
    fillText: () => {},
    strokeText: () => {},
    fillRect: () => {},
    clearRect: () => {},
    strokeRect: () => {},
    beginPath: () => {},
    closePath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    // The Vision page dashes the outline of an image's source shape.
    setLineDash: () => {},
    arc: () => {},
    fill: () => {},
    stroke: () => {},
    save: () => {},
    restore: () => {},
    translate: () => {},
    scale: () => {},
    rotate: () => {},
    setTransform: () => {},
    drawImage: () => {},
    getImageData: () => ({ data: [] }),
    putImageData: () => {},
    createLinearGradient: () => ({ addColorStop: () => {} }),
  });

  Object.defineProperty(element, "innerHTML", {
    get: () => serializeTextChildren(element),
    set: () => { element.children = []; },
  });

  return element;
}

// Enough of innerHTML to serve the one page script that reads it.
//
// `escHtml` in analytics.js is the sole reader: it appends a text
// node to a throwaway div and reads the escaped result back. A getter
// returning "" made that function return "" for every input, which
// silently emptied every label and value the panel renders and made
// them untestable, so tests could only assert that a row was absent.
//
// Elements are not serialized, only text children, and that is the
// whole of what a stub should claim here: rendering nested markup
// would be a half-built engine whose gaps are harder to notice than
// its absence. The escaping matches a browser's, because tests about
// escaping are worthless against an escaper that is merely similar.
function serializeTextChildren(element) {
  let html = "";
  for (let at = 0; at < element.children.length; at++) {
    const child = element.children[at];
    if (child.isTextNode) {
      html += escapeText(String(child.textContent));
    }
  }
  return html;
}

function escapeText(text) {
  return text
    .split("&").join("&amp;")
    .split("<").join("&lt;")
    .split(">").join("&gt;");
}

// Enough selector support for `closest`, which is the only place the
// page scripts depend on matching. Anything richer would be a
// half-built engine whose gaps are harder to notice than its absence.
function matches(node, selector) {
  if (!selector) {
    return false;
  }
  if (selector.startsWith("#")) {
    return node.id === selector.slice(1);
  }
  if (selector.startsWith(".")) {
    return node.classes.has(selector.slice(1));
  }
  const attribute = selector.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
  if (attribute) {
    const key = attribute[1];
    if (!(key in node.attributes)) {
      return false;
    }
    return attribute[2] === undefined
      || node.attributes[key] === attribute[2];
  }
  // A tag with an attribute filter, with or without a value:
  // `input[type="checkbox"]` and `tr[data-run-id]` both land here.
  // The value used to be mandatory, so a bare presence filter fell
  // through to the tag comparison below and matched nothing, which is
  // silent: `closest` returns null and the caller looks like it chose
  // not to act.
  const typed = selector.match(
    /^(\w+)\[([\w-]+)(?:="([^"]*)")?\]$/
  );
  if (typed) {
    if (node.tag !== typed[1]) {
      return false;
    }
    if (!(typed[2] in node.attributes)) {
      return false;
    }
    return typed[3] === undefined
      || node.attributes[typed[2]] === typed[3];
  }
  return node.tag === selector;
}

function makeDocument(registry, fontsReady) {
  // Recorded, not discarded. A widget that installs one document
  // listener per instance and removes none is a leak whose only
  // observable symptom is this count, so a stub that throws the
  // argument away cannot see the defect it is meant to catch.
  const documentListeners = {};
  const document = {
    getElementById(id) {
      if (!registry.has(id)) {
        registry.set(id, makeElement(id));
      }
      return registry.get(id);
    },
    createElement(tag) {
      const element = makeElement(null);
      element.tag = tag;
      return element;
    },
    createDocumentFragment: () => makeElement(null),
    createTextNode: (text) => {
      const node = makeElement(null);
      node.textContent = text;
      // Flagged rather than inferred from the absence of a tag, so
      // the innerHTML getter below can serialize text without having
      // to guess which appended children are text.
      node.isTextNode = true;
      return node;
    },
    // A stub rather than null: page scripts wire listeners onto
    // whatever this returns at load, and null would abort the load
    // for a selector that has nothing to do with the test.
    querySelector: (selector) => makeElement(selector),
    querySelectorAll: () => [],
    contains: (node) => !!node && node.isConnected !== false,
    addEventListener: (type, fn) => {
      (documentListeners[type] = documentListeners[type] || [])
        .push(fn);
    },
    removeEventListener: (type, fn) => {
      const list = documentListeners[type] || [];
      const at = list.indexOf(fn);
      if (at !== -1) {
        list.splice(at, 1);
      }
    },
    hidden: false,
    visibilityState: "visible",
  };
  document.listenerCount = (type) =>
    (documentListeners[type] || []).length;
  // Fire a document-level event the way a click outside every widget
  // would. `target` is whatever the click landed on, which is what
  // an outside-click handler tests against.
  document.dispatch = (type, event) => {
    for (const fn of (documentListeners[type] || []).slice()) {
      fn(Object.assign({ target: document.body }, event || {}));
    }
  };
  // Held open rather than pre-resolved, so a test can decide when
  // the webfont "arrives" and observe what the page does then. A
  // test that never resolves it is looking at the page as it is
  // before the font lands, which is a real state and the one that
  // mismeasures text.
  document.fonts = { ready: fontsReady };
  document.body = makeElement("body");
  document.documentElement = makeElement("html");
  document.head = makeElement("head");
  return document;
}

// A socket that connects to nothing and remembers what it was told.
// `deliver` plays a message back the way a worker would.
class FakeSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    FakeSocket.opened.push(this);
  }

  send(data) {
    this.sent.push(data);
  }

  close() {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose({ code: 1000, reason: "" });
    }
  }

  deliver(payload) {
    if (this.onmessage) {
      this.onmessage({ data: JSON.stringify(payload) });
    }
  }

  addEventListener(type, fn) {
    this["on" + type] = fn;
  }

  removeEventListener(type) {
    this["on" + type] = null;
  }
}
FakeSocket.CONNECTING = 0;
FakeSocket.OPEN = 1;
FakeSocket.CLOSING = 2;
FakeSocket.CLOSED = 3;
FakeSocket.opened = [];

// Answers every request with an empty success. Enough for a boot
// sequence to finish without the test having to describe endpoints
// it is not testing.
function inertFetch(calls) {
  return function (url, init) {
    calls.push({ url: String(url), init: init || {} });
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve({}),
      text: () => Promise.resolve("{}"),
    });
  };
}

// A complete-enough durable conversation API for generator tests
// whose subject is not HTTP. The real client is strict, so returning
// `{}` for these new boot and Send requests would make every older
// full-page test fail before it reached the behavior it owns.
function conversationFetch(baseFetch) {
  let serial = 0;
  let conversation = null;
  let turns = [];

  function response(body, status) {
    const code = status || 200;
    return Promise.resolve({
      ok: code >= 200 && code < 300,
      status: code,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  }

  function manifest() {
    const tail = turns[turns.length - 1] || null;
    const pending = tail
      && tail.role === "assistant"
      && tail.version === 1
      ? tail.turn_id
      : null;
    return {
      schema_version: 1,
      id: conversation.id,
      title: "New conversation",
      revision: conversation.revision,
      created_at: conversation.created_at,
      updated_at: conversation.created_at,
      turn_count: turns.length,
      tail_role: tail ? tail.role : null,
      tail_turn_id: tail ? tail.turn_id : null,
      tail_version: tail ? tail.version : null,
      pending_assistant_id: pending,
    };
  }

  function makeTurn(index, role, text, body) {
    const assistant = role === "assistant";
    return {
      schema_version: 1,
      conversation_id: conversation.id,
      conversation_revision: conversation.revision,
      turn_id: String(index).padStart(8, "0"),
      index,
      version: 1,
      role,
      created_at: conversation.created_at,
      updated_at: conversation.created_at,
      text,
      partial: assistant,
      model_id: assistant ? body.model_id : null,
      input_mode: assistant ? body.input_mode : null,
      context_pack: {},
      metadata: {},
      run_link: null,
    };
  }

  function createConversation() {
    serial += 1;
    conversation = {
      id: serial.toString(16).padStart(32, "0"),
      revision: 1,
      created_at: "2026-10-04T00:00:00Z",
    };
    turns = [];
    return response({ conversation: manifest() }, 201);
  }

  function appendTurn(body) {
    const index = turns.length + 1;
    conversation.revision += 1;
    const user = makeTurn(index, "user", body.text, body);
    const assistant = makeTurn(index + 1, "assistant", "", body);
    user.conversation_revision = conversation.revision;
    assistant.conversation_revision = conversation.revision;
    turns.push(user, assistant);
    return response({
      conversation: manifest(),
      user_turn: user,
      assistant_turn: assistant,
    }, 201);
  }

  function updateAssistant(body, runLink) {
    const assistant = turns[turns.length - 1];
    conversation.revision += 1;
    assistant.version += 1;
    assistant.conversation_revision = conversation.revision;
    if (runLink) {
      assistant.run_link = {
        run_id: body.run_id,
        revision: body.run_revision,
      };
    } else {
      assistant.text = body.text;
      assistant.partial = body.partial;
      assistant.context_pack = body.context_pack;
      assistant.metadata = body.metadata;
    }
    return response({
      conversation: manifest(),
      turn: assistant,
    });
  }

  return function (url, init) {
    const text = String(url);
    const path = text.split("?")[0];
    const method = (init && init.method) || "GET";
    if (!path.startsWith("/api/conversations")) {
      return baseFetch(url, init);
    }
    const body = init && init.body ? JSON.parse(init.body) : {};
    if (path === "/api/conversations" && method === "POST") {
      return createConversation();
    }
    if (conversation === null) {
      return response({
        error: "not found",
        reason: "not_found",
      }, 404);
    }
    if (path.endsWith("/metadata") && method === "GET") {
      return response({ conversation: manifest() });
    }
    if (path.endsWith("/turns") && method === "GET") {
      return response({
        conversation_id: conversation.id,
        revision: conversation.revision,
        turns: turns.slice(-50),
        next_before: null,
        has_more: false,
      });
    }
    if (path.endsWith("/turns") && method === "POST") {
      return appendTurn(body);
    }
    if (path.endsWith("/run") && method === "PUT") {
      return updateAssistant(body, true);
    }
    if (path.includes("/turns/") && method === "PUT") {
      return updateAssistant(body, false);
    }
    return response({});
  };
}

function unref(handle) {
  if (handle && typeof handle.unref === "function") {
    handle.unref();
  }
  return handle;
}

// `initial` holds entries already stored when the page loads, keyed
// as the page reads them.
function makeStorage(initial) {
  const store = new Map(Object.entries(initial || {}));
  return {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => { store.set(key, String(value)); },
    removeItem: (key) => { store.delete(key); },
    clear: () => { store.clear(); },
    get size() { return store.size; },
  };
}

/**
 * Load a page script and everything it depends on into one context.
 *
 * `options.fetchImpl` replaces `fetch`; `options.scripts` replaces
 * the generator's script list; `options.bootState` becomes
 * `window.__BOOT__`, which is how the server hands a page its opening
 * state. Omitting it is the meaningful other case, not merely the
 * default: it is what a page served without that state sees, and the
 * fetch fallback has to keep working for exactly that reason.
 *
 * `options.storage` seeds `localStorage` before any script runs, the
 * way a browser that saved settings presents them to the next page.
 * A page that reads a setting once at load sees it only this way.
 *
 * Returns the context plus the element registry, so a test can reach
 * an element by id and fire a listener the page registered on it.
 */
function loadPage(options) {
  const settings = options || {};
  const registry = new Map();
  const fetched = [];
  let announceFontsLoaded = () => {};
  const fontsReady = new Promise((resolve) => {
    announceFontsLoaded = resolve;
  });
  const document = makeDocument(registry, fontsReady);
  const sandbox = {
    console,
    document,
    JSON,
    Math,
    Date,
    Promise,
    URLSearchParams,
    AbortController,
    TextEncoder,
    TextDecoder,
    // Unreferenced, so a page's tickers and pollers do not hold the
    // test runner's event loop open after the assertions are done.
    // The page still gets working timers; they just stop counting
    // toward "is there anything left to do".
    setTimeout: (fn, ms) => unref(setTimeout(fn, ms)),
    clearTimeout,
    setInterval: (fn, ms) => unref(setInterval(fn, ms)),
    clearInterval,
    queueMicrotask,
    requestAnimationFrame: (fn) => unref(setTimeout(fn, 0)),
    cancelAnimationFrame: (handle) => clearTimeout(handle),
    localStorage: makeStorage(settings.storage),
    sessionStorage: makeStorage(),
    // Inert by default, for the same reason the socket is: a page
    // fetches during boot, and a default that rejected would fail
    // every test over a request none of them made. Records what was
    // asked for, so a test that does care can read it back or pass
    // its own `fetchImpl`.
    fetch: settings.conversationApi === false
      ? (settings.fetchImpl || inertFetch(fetched))
      : conversationFetch(
        settings.fetchImpl || inertFetch(fetched)
      ),
    // Inert by default, and inert rather than absent on purpose: a
    // page opens its socket during boot, so throwing here would fail
    // every test for a connection none of them drive. Records what
    // was sent, so a test that does care can read it back.
    WebSocket: settings.WebSocket || FakeSocket,
    // Enough of Event for `dispatchEvent(new Event("change"))`,
    // which is how the widgets tell their callers something changed.
    Event: class {
      constructor(type) {
        this.type = type;
      }
    },
    getComputedStyle: () => ({ getPropertyValue: () => "" }),
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    location: {
      protocol: "http:",
      host: "test",
      search: settings.locationSearch || "",
      href: "",
      pathname: "/",
      reload() {},
    },
    navigator: { userAgent: "node", clipboard: { writeText() {} } },
    alert: () => {},
    confirm: () => true,
  };
  // Window-level listeners, recorded like an element's so a test can
  // fire focus or beforeunload the way the browser would.
  const windowListeners = {};
  sandbox.addEventListener = (type, fn) => {
    (windowListeners[type] = windowListeners[type] || []).push(fn);
  };
  sandbox.removeEventListener = (type, fn) => {
    const list = windowListeners[type] || [];
    const at = list.indexOf(fn);
    if (at !== -1) {
      list.splice(at, 1);
    }
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.window.document = document;
  if (settings.bootState !== undefined) {
    sandbox.__BOOT__ = settings.bootState;
  }
  sandbox.Chart = function () {
    return { destroy() {}, update() {}, resize() {} };
  };
  sandbox.Chart.register = () => {};
  sandbox.Chart.defaults = permissive();
  sandbox.Chart.overrides = permissive();
  sandbox.Chart.helpers = permissive();
  sandbox.Chart.Tooltip = { positioners: {} };

  const context = vm.createContext(sandbox);
  const scripts = settings.scripts || GENERATOR_SCRIPTS;
  for (const name of scripts) {
    const file = path.join(STATIC_DIR, name);
    try {
      vm.runInContext(fs.readFileSync(file, "utf8"), context, {
        filename: name,
      });
    } catch (error) {
      // Loudly, and naming the file. A stub that silently loaded two
      // thirds of a page would let a test assert against a context
      // missing the half it cares about.
      throw new Error(
        `dom_stub could not load ${name}: ${error.message}.`
        + " Add what it reached for to the sandbox above, or narrow"
        + " the script list for this test."
      );
    }
  }
  return {
    context,
    registry,
    document,
    sandbox,
    // Every request the page made, in order.
    fetched,
    // The controller is the supported full-page socket seam. Its
    // mutable WebSocket and reconnect state remain private; tests can
    // ask it to connect or read readiness without replacing a page
    // global that production no longer has.
    generatorSocketController() {
      const controller = context.generatorSocket;
      if (
        !controller
        || typeof controller.connect !== "function"
        || typeof controller.isReady !== "function"
      ) {
        throw new Error("The loaded page has no generator socket");
      }
      return controller;
    },
    // Settle `document.fonts.ready`. Anything the page defers until
    // its webfont has loaded runs on the microtask after this.
    announceFontsLoaded() {
      announceFontsLoaded();
      return Promise.resolve();
    },
    // Fire a window-level listener, for the handful of page
    // behaviours that hang off focus or visibility rather than off
    // an element.
    fireWindow(type, event) {
      for (const fn of (windowListeners[type] || []).slice()) {
        fn(event || {});
      }
    },
  };
}

module.exports = {
  loadPage,
  makeElement,
  FakeSocket,
  GENERATOR_SCRIPTS,
  ANALYTICS_SCRIPTS,
  MENU_SCRIPTS,
  SETTINGS_SCRIPTS,
  VISION_SCRIPTS,
};
