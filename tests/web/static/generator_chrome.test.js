// The generator chrome, driven without app.js.
//
// Strategy: load its persistence and progress dependencies, then
// compose the factory with recording callbacks. Interactions use the
// public controller and the DOM controls it owns. The sparkline
// context records its latest path so the private history can be
// measured without exposing it.
//
// Passing proves the extraction owns real closure state, requires its
// page callbacks, bounds the live resource series, retires status
// operations and preserves the durable new-run cue.

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const { loadPage } = require("./dom_stub.js");

const SCRIPTS = [
  "persist.js",
  "activation_progress.js",
  "generator_chrome.js",
];
const NEW_RUNS_KEY = "diffusion_new_runs";
const RESOURCE_HISTORY_MAX = 120;

function completeOptions(state) {
  return {
    onTpsToggle() {
      state.tpsToggles += 1;
    },
    readReducedMotion() {
      return state.reducedMotion;
    },
    readDiffusionEffect() {
      return state.diffusionEffect;
    },
    readDiffusionTextMode() {
      return state.diffusionTextMode;
    },
    revealText(element, text, onDone) {
      element.textContent = text;
      if (onDone) {
        onDone();
      }
    },
    cancelReveal() {},
  };
}

function loadChrome(settings) {
  const config = settings || {};
  const page = loadPage({
    scripts: SCRIPTS,
    storage: config.storage,
  });
  const state = {
    tpsToggles: 0,
    reducedMotion: Boolean(config.reducedMotion),
    diffusionEffect: false,
    diffusionTextMode: "once",
  };
  const chrome = page.context.generatorChromeCreate(
    completeOptions(state)
  );
  chrome.wire();
  return { page, chrome, state };
}

function resourceSample(fraction) {
  return {
    kind: "vram",
    fraction: fraction,
    used_bytes: fraction * 24 * 1024 * 1024 * 1024,
    total_bytes: 24 * 1024 * 1024 * 1024,
  };
}

function recordSparkPaths(canvas) {
  const paths = [];
  let path = [];
  canvas.clientWidth = RESOURCE_HISTORY_MAX - 1;
  canvas.clientHeight = 100;
  canvas.getContext = () => ({
    setTransform() {},
    clearRect() {},
    beginPath() {
      path = [];
    },
    moveTo(x, y) {
      path.push({ x: x, y: y });
    },
    lineTo(x, y) {
      path.push({ x: x, y: y });
    },
    stroke() {
      paths.push(path.slice());
    },
  });
  return paths;
}

function wait(milliseconds) {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

test("every page callback is required", () => {
  const page = loadPage({ scripts: SCRIPTS });
  const state = {
    tpsToggles: 0,
    reducedMotion: false,
    diffusionEffect: false,
    diffusionTextMode: "once",
  };
  const names = Object.keys(completeOptions(state));

  for (const name of names) {
    const options = completeOptions(state);
    delete options[name];

    assert.throws(
      () => page.context.generatorChromeCreate(options),
      new RegExp(name)
    );
  }
});

test("DOM references and mutable state stay in the closure", () => {
  const { page } = loadChrome({});

  for (const name of [
    "statusMessage",
    "loadingOverlay",
    "resourceHistory",
    "elapsedTimer",
    "statusChips",
    "runStatusHandle",
    "RESOURCE_HISTORY_MAX",
  ]) {
    assert.equal(page.context[name], undefined, name);
  }
});

test("placeholder and loading chrome preserve their page IDs", () => {
  const { page, chrome } = loadChrome({});
  const output = page.registry.get("output-area");
  const overlay = page.registry.get("loading-overlay");

  chrome.showOutputPlaceholder("Mamba-3");
  chrome.setConnection("ready");
  chrome.setLoadingText("Loading Mamba-3\u2026");
  chrome.showLoading();

  assert.equal(output.children[0].id, "output-placeholder");
  assert.equal(
    output.children[0].textContent,
    "Mamba-3 output will appear here..."
  );
  assert.equal(
    page.registry.get("connection-badge").textContent, "ready"
  );
  assert.equal(
    page.registry.get("loading-text").textContent,
    "Loading Mamba-3\u2026"
  );
  assert.equal(overlay.classList.contains("hidden"), false);

  chrome.hideLoading();
  assert.equal(overlay.classList.contains("hidden"), true);
});

test("an empty terminal result is named in the output area", () => {
  const { page, chrome } = loadChrome({});
  const output = page.registry.get("output-area");

  chrome.showNoOutput(false);
  assert.match(
    output.children[0].textContent,
    /model ended before producing any text/
  );

  chrome.showNoOutput(true);
  assert.match(
    output.children[0].textContent,
    /run stopped before producing any text/
  );
});

test("TPS controls call the page callback", () => {
  const { page, state } = loadChrome({});
  const tps = page.registry.get("status-tps");
  let prevented = 0;

  tps.click();
  tps.dispatch("keydown", {
    key: "Enter",
    preventDefault() {
      prevented += 1;
    },
  });
  tps.dispatch("keydown", {
    key: "x",
    preventDefault() {
      prevented += 1;
    },
  });

  assert.equal(state.tpsToggles, 2);
  assert.equal(prevented, 1);
});

test("resource history is private and bounded to one minute", () => {
  const { page, chrome } = loadChrome({});
  const canvas = page.registry.get("status-resource-spark");
  const paths = recordSparkPaths(canvas);

  for (let index = 0; index < RESOURCE_HISTORY_MAX; index++) {
    chrome.handleResourceSample(resourceSample(0));
  }
  chrome.handleResourceSample(resourceSample(1));

  const latest = paths[paths.length - 1];
  assert.equal(latest.length, RESOURCE_HISTORY_MAX);
  assert.deepEqual(latest[0], { x: 0, y: 100 });
  assert.deepEqual(
    latest[latest.length - 1],
    { x: RESOURCE_HISTORY_MAX - 1, y: 0 }
  );
  assert.equal(page.context.resourceHistory, undefined);
});

test("run status replacement and ending retire every timer",
  async () => {
    const { page, chrome } = loadChrome({
      reducedMotion: true,
    });
    const stack = page.registry.get("status-stack");
    const message = page.registry.get("status-message");
    stack.appendChild(message);

    chrome.startRunStatus("Running");
    const first = stack.children[0];
    chrome.startRunStatus("Running edit");
    const second = stack.children[1];

    assert.equal(first.classList.contains("is-leaving"), true);
    assert.equal(first._dotsTimer, null);
    assert.equal(second._textEl.textContent, "Running edit");
    assert.equal(second.classList.contains("is-visible"), true);

    chrome.endRunStatus();
    assert.equal(second.classList.contains("is-leaving"), true);
    assert.equal(second._dotsTimer, null);

    await wait(180);
    assert.deepEqual(stack.children, [message]);
  }
);

test("new-run cue keeps the persisted key and de-duplicates flashes",
  () => {
    const { page, chrome } = loadChrome({
      storage: {
        [NEW_RUNS_KEY]: JSON.stringify(["old-run"]),
      },
    });
    const dot = page.registry.get("analytics-new-dot");
    const link = page.registry.get("link-analytics");

    chrome.boot();
    assert.equal(dot.textContent, "1");
    assert.equal(dot.classList.contains("is-empty"), false);

    chrome.showAnalyticsCue("new-run");
    chrome.showAnalyticsCue("new-run");

    assert.deepEqual(
      JSON.parse(page.sandbox.localStorage.getItem(NEW_RUNS_KEY)),
      ["old-run", "new-run"]
    );
    assert.equal(dot.textContent, "2");
    assert.equal(link.children.length, 1);
    assert.equal(link.children[0].textContent, "+1");
  }
);
