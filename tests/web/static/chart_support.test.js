// The chart helpers more than one Analytics chart reads.
//
// Strategy: load chart_support.js alone into a vm context, once with
// a Chart global standing in for the vendored library and once
// without, and ask each helper what a chart asks it: whether charts
// can be drawn at all, the teardown, the zoom and gutter options, the
// tooltip's title and its series swatch.
//
// Passing proves the helpers work with nothing of a page around
// them, which is what lets the line charts, the entropy chart and
// the comparison view share them, and that a missing library reads
// as unavailable rather than failing at load.
//
// Run with: node --test tests/web/static/

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const SOURCE = path.join(
  __dirname, "..", "..", "..", "src", "web", "static",
  "chart_support.js"
);

function load(sandbox) {
  const context = vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context, {
    filename: "chart_support.js",
  });
  return context;
}

function withChart() {
  return load({ Chart: function () {} });
}

// Compared as JSON: objects built in the vm have that realm's
// prototypes, so a strict deepEqual against a host value fails on
// identity even when the contents match.
function same(actual, expected) {
  assert.equal(JSON.stringify(actual), JSON.stringify(expected));
}

test("charts are available when the library loaded", () => {
  assert.equal(withChart().chartSupportAvailable, true);
});

test("without the library, charts read as unavailable", () => {
  assert.equal(load({}).chartSupportAvailable, false);
});

test("destroying a chart tears it down and empties its slot", () => {
  const api = withChart();
  let destroyed = 0;
  const chart = {
    destroy() {
      destroyed += 1;
    },
  };

  assert.equal(api.chartSupportDestroy(chart), null);
  assert.equal(destroyed, 1);
});

test("destroying an empty slot leaves it empty", () => {
  assert.equal(withChart().chartSupportDestroy(null), null);
});

test("the tooltip's title names the frame", () => {
  const api = withChart();

  assert.equal(
    api.chartSupportTooltipTitle([{ label: "112" }]), "Frame 112"
  );
  assert.equal(api.chartSupportTooltipTitle([]), "");
});

test("a series swatch is the line's own color, unbordered", () => {
  const api = withChart();
  const line = { dataset: { borderColor: "#00aaff" } };

  same(api.chartSupportLineLabelColor(line), {
    borderColor: "transparent",
    backgroundColor: "#00aaff",
  });
});

test("a scripted border color gives a white swatch", () => {
  const api = withChart();
  const scripted = { dataset: { borderColor: () => "#00aaff" } };

  same(api.chartSupportLineLabelColor(scripted), {
    borderColor: "transparent",
    backgroundColor: "#ffffff",
  });
});

test("zoom and pan follow the x axis, by wheel and by pinch", () => {
  same(withChart().chartSupportZoomOptions(), {
    zoom: {
      wheel: { enabled: true },
      pinch: { enabled: true },
      mode: "x",
    },
    pan: { enabled: true, mode: "x" },
  });
});

test("the gutter makes room under the axis for the zoom dock", () => {
  same(withChart().chartSupportGutterLayout(), {
    padding: { bottom: 16 },
  });
});
