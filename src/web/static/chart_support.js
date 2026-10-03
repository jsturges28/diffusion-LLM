// Chart helpers that more than one of Analytics' charts reads: the
// line charts, the entropy chart and the comparison view. Loaded as
// a classic global script after the vendored Chart.js and before
// every script that draws a chart, so it must not depend on any
// page's state. It touches no DOM either, which is what makes it
// testable away from a browser.
//
// The page's own Chart.js setup, its defaults and the smart tooltip
// positioner, stays in analytics.js. It runs once as that page
// loads, and a chart reaches the positioner by the name it was
// registered under rather than by anything this file could hand it.

"use strict";

// Whether the charting library loaded at all. The charts used to
// assume it had, and the assumption was made at the top level of
// analytics.js: one missing script and the whole of that file failed
// to parse, taking the run table, the metadata, the overlays and
// deletion down with the charts. The library is vendored now so this
// should always be true, but the page should degrade rather than
// disappear if it ever is not.
var chartSupportAvailable = typeof Chart !== "undefined";

// Answers null so a caller empties its slot in the same statement
// that tears the chart down.
function chartSupportDestroy(chart) {
  if (chart) {
    chart.destroy();
  }
  return null;
}

// Room under the x axis for the zoom controls docked in the chart's
// bottom-left corner. The y-axis gutter alone is narrower than the
// three buttons, so without this the pill would overlap the first
// tick label. Not used by the compare panel, which has no dock.
function chartSupportGutterLayout() {
  return { padding: { bottom: 16 } };
}

// Shared zoom plugin options for scroll + pinch.
function chartSupportZoomOptions() {
  return {
    zoom: {
      wheel: { enabled: true },
      pinch: { enabled: true },
      mode: "x",
    },
    pan: {
      enabled: true,
      mode: "x",
    },
  };
}

// Shared tooltip title callback that prefixes
// the frame number so it reads "Frame 112"
// on its own line rather than just "112".
function chartSupportTooltipTitle(items) {
  if (items.length === 0) { return ""; }
  return "Frame " + items[0].label;
}

// Shared tooltip swatch color for the line charts. A line's
// backgroundColor is an area wash at around 0.1 alpha, so a swatch
// filled with it reads as almost nothing; the line's own color is
// what tells one series from another in a two-row tooltip.
//
// The border is deliberately invisible rather than absent. Chart.js
// resolves the swatch stroke as ``borderWidth || 1``, so asking for
// zero still strokes a pixel, and that ring plus the white backing
// underneath (the multiKeyBackground default in analytics.js) is
// what made the chip read as a colored frame around a lighter square.
// With both suppressed the swatch is exactly the inset fill.
//
// The entropy chart deliberately does not use this: its bars carry
// solid per-bar colors, so its swatches already read correctly and
// showing the hovered bar's own ramp color says more than the series
// color would.
function chartSupportLineLabelColor(ctx) {
  var color = ctx.dataset.borderColor;
  if (typeof color !== "string") {
    color = "#ffffff";
  }
  return {
    borderColor: "transparent",
    backgroundColor: color,
  };
}
