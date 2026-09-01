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
  "overlays.js",
  "activation_client.js",
  "wire_errors.js",
  "model_client.js",
  "run_frames.js",
  "run_phases.js",
  "download_client.js",
  "download_toast.js",
  "app.js",
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
    style: {},
    value: "",
    checked: false,
    disabled: false,
    hidden: false,
    textContent: "",
    className: "",
    classes: new Set(),
    scrollTop: 0,
    scrollHeight: 0,
    offsetWidth: 0,
    clientWidth: 0,
  };

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
  element.querySelector = () => null;
  element.querySelectorAll = () => [];
  element.focus = () => {};
  element.blur = () => {};
  element.scrollIntoView = () => {};
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
    get: () => "",
    set: () => { element.children = []; },
  });

  return element;
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
  const typed = selector.match(/^(\w+)\[type="([^"]+)"\]$/);
  if (typed) {
    return node.tag === typed[1]
      && node.attributes.type === typed[2];
  }
  return node.tag === selector;
}

function makeDocument(registry) {
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
      return node;
    },
    // A stub rather than null: page scripts wire listeners onto
    // whatever this returns at load, and null would abort the load
    // for a selector that has nothing to do with the test.
    querySelector: (selector) => makeElement(selector),
    querySelectorAll: () => [],
    addEventListener: () => {},
    removeEventListener: () => {},
    hidden: false,
    visibilityState: "visible",
  };
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
FakeSocket.opened = [];

function unref(handle) {
  if (handle && typeof handle.unref === "function") {
    handle.unref();
  }
  return handle;
}

function makeStorage() {
  const store = new Map();
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
 * the generator's script list. Returns the context plus the element
 * registry, so a test can reach a element by id and fire a listener
 * the page registered on it.
 */
function loadPage(options) {
  const settings = options || {};
  const registry = new Map();
  const document = makeDocument(registry);
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
    localStorage: makeStorage(),
    sessionStorage: makeStorage(),
    fetch: settings.fetchImpl
      || (() => Promise.reject(new Error("no fetch in this test"))),
    // Inert by default, and inert rather than absent on purpose: a
    // page opens its socket during boot, so throwing here would fail
    // every test for a connection none of them drive. Records what
    // was sent, so a test that does care can read it back.
    WebSocket: settings.WebSocket || FakeSocket,
    getComputedStyle: () => ({ getPropertyValue: () => "" }),
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    location: { search: "", href: "", pathname: "/", reload() {} },
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
};
