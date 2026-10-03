// Analytics' frame and signal adapter, driven without a page.
//
// Strategy: load overlays.js and overlay_series.js into one vm
// context, in the order Analytics loads them, and hand the adapter the
// two payloads the server really sends, kept in fixtures/ and held
// equal to the frames endpoint by tests/web/test_frames_fixtures.py.
// One is a per-frame LLaDA run edited at frame 2, with the run it
// branched from; the other is a SmolLM3 run that only grows, sent as
// flat positions. Every variant below is made from one of those
// rather than from scratch, so a key the server never sends cannot
// creep in.
//
// Passing proves the adapter answers the viewer's questions the same
// way of either shape: how long a series is, what a frame holds,
// where it ends, when each position settled, what the stopping
// track reads, which frame a channel is read at, and whether entropy
// can be drawn at all. The same
// questions asked through the whole page stay in
// analytics_frames.test.js and analytics_signal_axes.test.js.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const STATIC = path.join(
  __dirname, "..", "..", "..", "src", "web", "static"
);
const FIXTURES = path.join(__dirname, "fixtures");

// Analytics' own order: the adapter reads overlays.js.
const SOURCES = ["overlays.js", "overlay_series.js"];

const EDITED = "frames_llada_edited.json";
const APPEND = "frames_smollm3_append.json";

function load() {
  const context = vm.createContext({});
  for (const name of SOURCES) {
    vm.runInContext(
      fs.readFileSync(path.join(STATIC, name), "utf8"),
      context,
      { filename: name }
    );
  }
  return context;
}

// A fresh copy each time, so a test that varies one cannot leak into
// the next.
function fixture(name) {
  return JSON.parse(
    fs.readFileSync(path.join(FIXTURES, name), "utf8")
  );
}

function varied(name, change) {
  const payload = fixture(name);
  change(payload);
  return payload;
}

// A frame's tokens as words, a masked position as "_".
function words(tokens) {
  return tokens
    .map((token) => (token.m ? "_" : token.t.trim()))
    .join(" ");
}

// Compared as JSON: arrays built in the vm have that realm's
// prototypes, so a strict deepEqual against a host value fails on
// identity even when the contents match.
function same(actual, expected) {
  assert.equal(JSON.stringify(actual), JSON.stringify(expected));
}

function oneCanvas() {
  return 0;
}

// -- a series, in either shape --

test("a per-frame run is as long as its frames", () => {
  const api = load();
  const series = api.overlaySeriesOf(fixture(EDITED), false);

  assert.equal(api.overlaySeriesLength(series), 4);
  assert.equal(api.overlaySeriesPresent(series), true);
});

test("a run that only grows is as long as its positions", () => {
  const api = load();
  const series = api.overlaySeriesOf(fixture(APPEND), false);

  assert.equal(api.overlaySeriesLength(series), 3);
  assert.equal(api.overlaySeriesPresent(series), true);
});

test("a per-frame run's frame is the frame the server sent", () => {
  const api = load();
  const series = api.overlaySeriesOf(fixture(EDITED), false);

  assert.equal(words(api.overlaySeriesAt(series, 2)), "Yeast _ _");
  assert.equal(api.overlaySeriesAt(series, 4), null);
  assert.equal(api.overlaySeriesAt(series, -1), null);
});

test("a growing run's frame is its first positions", () => {
  const api = load();
  const series = api.overlaySeriesOf(fixture(APPEND), false);

  assert.equal(words(api.overlaySeriesAt(series, 0)), "Yeast");
  assert.equal(
    words(api.overlaySeriesAt(series, 2)), "Yeast eats sugar"
  );
  assert.equal(api.overlaySeriesAt(series, 3), null);
});

test("each shape ends on its last frame", () => {
  const api = load();
  const edited = api.overlaySeriesOf(fixture(EDITED), false);
  const append = api.overlaySeriesOf(fixture(APPEND), false);

  assert.equal(api.overlaySeriesFinalIndex(edited), 3);
  assert.equal(api.overlaySeriesFinalIndex(append), 2);
  assert.equal(
    words(api.overlaySeriesFinal(edited)), "Yeast ate sugar"
  );
  assert.equal(
    words(api.overlaySeriesFinal(append)), "Yeast eats sugar"
  );
});

test("the baseline is the run as it was before the edit", () => {
  const api = load();
  const baseline = api.overlaySeriesOf(fixture(EDITED), true);

  assert.equal(api.overlaySeriesLength(baseline), 4);
  assert.equal(
    words(api.overlaySeriesFinal(baseline)), "Yeast eats sugar"
  );
});

test("no payload is an empty series", () => {
  const api = load();
  const series = api.overlaySeriesOf(null, false);

  assert.equal(api.overlaySeriesLength(series), 0);
  assert.equal(api.overlaySeriesPresent(series), false);
  assert.equal(api.overlaySeriesFinal(series), null);
});

// -- when each position settled --

test("commit steps are when each position took its final value", () => {
  // The edit remasked position 1 at frame 2, so it settles a step
  // later in the edited run than in the one it branched from.
  const api = load();
  const payload = fixture(EDITED);

  same(
    api.overlaySeriesCommitSteps(api.overlaySeriesOf(payload, false)),
    [1, 3, 3]
  );
  same(
    api.overlaySeriesCommitSteps(api.overlaySeriesOf(payload, true)),
    [1, 2, 3]
  );
});

test("a growing run settles every position as it appears", () => {
  const api = load();
  const series = api.overlaySeriesOf(fixture(APPEND), false);

  same(api.overlaySeriesCommitSteps(series), [0, 0, 0]);
  same(api.overlaySeriesRevisions(series, oneCanvas, []), []);
});

// -- what the stopping track reads --

// The edited run as DiffusionGemma would send it: two canvases.
function twoCanvases(payload) {
  payload.canvas_index = [0, 0, 1, 1];
}

test("a branch's stop source carries its canvases and edits", () => {
  const api = load();
  const data = varied(EDITED, twoCanvases);
  const series = api.overlaySeriesOf(data, false);

  const source = api.overlaySeriesStopSource(data, series, false);

  assert.equal(source.count, 4);
  same([0, 1, 2, 3].map(source.canvasAt), [0, 0, 1, 1]);
  same(source.segmentStarts, [2]);
  assert.equal(
    words(source.readFrame(3)), words(api.overlaySeriesAt(series, 3))
  );
});

test("the run it forked from is one canvas, never resumed", () => {
  const api = load();
  const data = varied(EDITED, twoCanvases);
  const baseline = api.overlaySeriesOf(data, true);

  const source = api.overlaySeriesStopSource(data, baseline, true);

  assert.equal(source.count, 4);
  same([0, 1, 2, 3].map(source.canvasAt), [0, 0, 0, 0]);
  same(source.segmentStarts, []);
  assert.equal(
    words(source.readFrame(0)),
    words(api.overlaySeriesAt(baseline, 0))
  );
});

test("a payload without canvas indices reads as one canvas", () => {
  const api = load();
  const data = fixture(EDITED);

  const source = api.overlaySeriesStopSource(
    data, api.overlaySeriesOf(data, false), false
  );

  same([0, 1, 2, 3].map(source.canvasAt), [0, 0, 0, 0]);
});

test("a growing run's stop source reads its positions", () => {
  const api = load();
  const data = fixture(APPEND);
  const series = api.overlaySeriesOf(data, false);

  const source = api.overlaySeriesStopSource(data, series, false);

  assert.equal(source.count, 3);
  assert.equal(source.readFrame(1).length, 2);
  same(source.segmentStarts, []);
});

// -- the signal manifest --

test("a run's channels are found by name, with their shape", () => {
  const api = load();
  const edited = fixture(EDITED);
  const append = fixture(APPEND);

  assert.equal(
    api.overlaySeriesChannelShape(
      api.overlaySeriesChannel(edited, "entropy")
    ),
    "frame|position"
  );
  assert.equal(
    api.overlaySeriesChannelShape(
      api.overlaySeriesChannel(append, "entropy")
    ),
    "position"
  );
  assert.equal(api.overlaySeriesChannel(edited, "forgetting"), null);
  assert.equal(api.overlaySeriesChannelShape(null), "");
});

test("a channel that varies by frame follows the given frame", () => {
  // Clamped to the run, since a scrub left past the end of a longer
  // run would otherwise read a frame this one never had.
  const api = load();
  const payload = fixture(EDITED);
  const series = api.overlaySeriesOf(payload, false);
  const channel = api.overlaySeriesChannel(payload, "entropy");

  assert.equal(api.overlaySeriesChannelFrame(channel, series, 1), 1);
  assert.equal(api.overlaySeriesChannelFrame(channel, series, 9), 3);
});

test("a channel read once per position is read at the end", () => {
  // As is a run saved before manifests existed, which declares none.
  const api = load();
  const payload = fixture(APPEND);
  const series = api.overlaySeriesOf(payload, false);
  const channel = api.overlaySeriesChannel(payload, "entropy");

  assert.equal(api.overlaySeriesChannelFrame(channel, series, 0), 2);
  assert.equal(api.overlaySeriesChannelFrame(null, series, 0), 2);
});

test("entropy is drawable, absent, or a shape this build cannot draw", () => {
  const api = load();
  const bare = varied(EDITED, (payload) => {
    for (const frame of payload.frames) {
      for (const token of frame) {
        delete token.e;
      }
    }
  });
  const canvas = varied(EDITED, (payload) => {
    api.overlaySeriesChannel(payload, "entropy").axes = ["canvas"];
  });
  const unmanifested = varied(EDITED, (payload) => {
    payload.signals = null;
  });

  assert.equal(api.overlaySeriesEntropyAvailability(fixture(EDITED)), "ok");
  assert.equal(api.overlaySeriesEntropyAvailability(fixture(APPEND)), "ok");
  assert.equal(api.overlaySeriesEntropyAvailability(bare), "absent");
  assert.equal(
    api.overlaySeriesEntropyAvailability(canvas), "unsupported"
  );
  assert.equal(api.overlaySeriesEntropyAvailability(unmanifested), "ok");
});

test("a frame without entropy reads its canvas's last draft", () => {
  // A DiffusionGemma commit carries none of its own: the model
  // accepted the canvas rather than drawing it from a distribution.
  const api = load();
  const committed = varied(EDITED, (payload) => {
    for (const token of payload.frames[3]) {
      delete token.e;
    }
  });
  const series = api.overlaySeriesOf(committed, false);

  assert.equal(api.overlaySeriesEntropyFrame(series, 3, oneCanvas), 2);
  assert.equal(api.overlaySeriesEntropyFrame(series, 1, oneCanvas), 1);
  assert.equal(api.overlaySeriesHasEntropy(series), true);
});

test("per-token values are looked for on the final frame", () => {
  const api = load();
  for (const name of [EDITED, APPEND]) {
    const series = api.overlaySeriesOf(fixture(name), false);

    assert.equal(api.overlaySeriesHasTokenValue(series, "e"), true);
    assert.equal(api.overlaySeriesHasTokenValue(series, "f"), false);
  }
});
