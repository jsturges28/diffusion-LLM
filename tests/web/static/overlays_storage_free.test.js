// The visual module draws without touching storage.
//
// Strategy: load overlays.js into a vm context whose localStorage,
// sessionStorage and fetch record every touch and refuse it, then run
// the token math the two pages draw with: the color ramps, commit
// steps in both shapes, the entropy frame, and revisions with their
// counts. A control asks the settings reader, which does read
// storage, so a trap that recorded nothing cannot pass for a module
// that touched nothing.
//
// Passing proves the other half of moving durable state into
// persist.js: the visual module's token math reaches for no storage
// at all, so a test of how tokens are drawn cannot pass or fail on
// what a page happened to persist. The settings reader and the
// overlay drawer still read storage here; they are not token math.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname, "..", "..", "..", "src", "web", "static", "overlays.js"
);

const GUARDED = ["localStorage", "sessionStorage", "fetch"];

// Two positions settling one per frame, as a per-frame run sends
// them, each frame's entropy its own.
const FRAMES = [
  [
    { t: "\u2591", m: true, id: 0, c: 0.2, e: 1.0 },
    { t: "\u2591", m: true, id: 0, c: 0.2, e: 1.1 },
  ],
  [
    { t: "a", m: false, id: 5, c: 0.6, e: 0.6 },
    { t: "\u2591", m: true, id: 0, c: 0.3, e: 0.9 },
  ],
  [
    { t: "a", m: false, id: 5, c: 0.7, e: 0.4 },
    { t: "b", m: false, id: 6, c: 0.8, e: 0.5 },
  ],
];

// The same two tokens from a run that only grows.
const POSITIONS = [
  { t: "a", m: false, id: 5, c: 0.7, e: 0.4 },
  { t: "b", m: false, id: 6, c: 0.8, e: 0.5 },
];

function load() {
  const touched = [];
  const sandbox = {};
  for (const name of GUARDED) {
    Object.defineProperty(sandbox, name, {
      get() {
        touched.push(name);
        throw new Error(name + " is off limits here");
      },
    });
  }
  const context = vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context, {
    filename: "overlays.js",
  });
  return { api: context, touched: touched };
}

function oneCanvas() {
  return 0;
}

// Compared as JSON: arrays built in the vm have that realm's
// prototypes, so a strict deepEqual against a host value fails on
// identity even when the contents match.
function same(actual, expected) {
  assert.equal(JSON.stringify(actual), JSON.stringify(expected));
}

test("the trap records a read: the settings reader touches storage", () => {
  const { api, touched } = load();

  api.overlaysLoadSettings();

  assert.ok(touched.includes("localStorage"), touched.join(", "));
});

test("loading the module touches no storage", () => {
  const { touched } = load();

  same(touched, []);
});

test("the color ramps touch no storage", () => {
  const { api, touched } = load();

  for (const color of [
    api.heatColor(0.5),
    api.commitColor(1, 3),
    api.diffColor(true),
    api.entropyColor(1.2),
    api.forgettingColor(0.3),
    api.revisionColor(2),
  ]) {
    assert.equal(typeof color, "string");
  }
  assert.equal(typeof api.overlaysMaskOpacity(0.5), "number");
  same(touched, []);
});

test("commit steps touch no storage, in either shape", () => {
  const { api, touched } = load();
  const reader = api.overlaysFrameReader(FRAMES);

  same(api.overlaysComputeCommitSteps(reader, FRAMES.length), [1, 2]);
  same(api.overlaysAppendCommitSteps(POSITIONS), [0, 0]);
  same(touched, []);
});

test("the entropy frame and revisions touch no storage", () => {
  const { api, touched } = load();
  const reader = api.overlaysFrameReader(FRAMES);

  assert.equal(api.overlaysEntropyFrame(reader, oneCanvas, 2, false), 2);
  const revisions = api.overlaysComputeRevisions(
    reader, FRAMES.length, oneCanvas, []
  );
  assert.equal(revisions.length, FRAMES.length);
  assert.ok(Array.isArray(
    api.overlaysRevisionCounts(revisions, 2, oneCanvas)
  ));
  same(touched, []);
});
