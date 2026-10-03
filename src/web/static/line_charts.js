// The Analytics detail panel's line charts: convergence, timing,
// tokens per second, confidence and stopping, with the two-page
// slots they share, the run pins, and the run crossfade's borrowing
// of them while its slider is dragged.
//
// Loaded as a classic global script after overlays.js,
// overlay_series.js and chart_support.js, which it reads, and before
// analytics.js, which creates it once. It defines one global name,
// lineChartsCreate, and keeps the rest inside that factory, so the
// page changes a chart only through the object it returns. The page
// owns the run crossfade and mediates between the charts and the
// token viewer: it hands the charts a way to read the crossfade, and
// calls in when a run opens, its frames land or the slider moves.
//
// The factory is long because it is this file's scope. The functions
// inside it are each small, and its own statements are only
// declarations and the object it returns.

"use strict";

// Create the panel's line charts. ``options.readBlend`` answers the
// run crossfade's position, 0 for the run an edit branched from and
// 1 for the branch, which the charts follow while the slider is
// dragged. Elements are found by id, so the page's markup is the rest
// of the contract.
function lineChartsCreate(options) {
  if (!options || typeof options.readBlend !== "function") {
    throw new TypeError("lineChartsCreate needs options.readBlend");
  }
  var readBlend = options.readBlend;

  var timingSection =
    document.getElementById("timing-section");

  // ---- The tooltip's burn-through ----

  // Inline plugin: once the tooltip box is drawn, "burn" the data
  // through it. Any trendline segment the box covers is redrawn,
  // clipped to the box rect, with a glow (so a box trapped over the
  // line stays legible), and the active point(s) are re-drawn glowing
  // on top.
  var burnThroughPlugin = {
    id: "burnThrough",
    afterDraw: function (chart) {
      var tt = chart.tooltip;
      if (!tt || tt.opacity === 0) { return; }
      var ctx = chart.ctx;
      var bx = tt.x;
      var by = tt.y;
      var bw = tt.width;
      var bh = tt.height;

      if (bw > 0 && bh > 0) {
        for (var di = 0; di < chart.data.datasets.length; di++) {
          var meta = chart.getDatasetMeta(di);
          if (
            meta.hidden || !meta.data || meta.data.length === 0
          ) {
            continue;
          }
          var alpha = chartSeriesAlpha(chart, di);
          if (alpha <= 0.02) { continue; }
          var ds = chart.data.datasets[di];
          var color = (typeof ds.borderColor === "string")
            ? ds.borderColor : "#ffffff";
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.rect(bx, by, bw, bh);
          ctx.clip();
          ctx.beginPath();
          burnThroughTrace(ctx, meta.data, ds.spanGaps === true);
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.shadowColor = color;
          ctx.shadowBlur = 10;
          ctx.stroke();
          ctx.restore();
        }
      }

      var active = chart.getActiveElements();
      if (active && active.length) {
        for (var i = 0; i < active.length; i++) {
          var ael = active[i].element;
          if (!ael) { continue; }
          var aalpha = chartSeriesAlpha(
            chart, active[i].datasetIndex
          );
          if (aalpha <= 0.02) { continue; }
          var ads = chart.data.datasets[active[i].datasetIndex];
          var acolor = (ads && typeof ads.borderColor === "string")
            ? ads.borderColor : "#ffffff";
          ctx.save();
          ctx.globalAlpha = aalpha;
          ctx.beginPath();
          ctx.arc(ael.x, ael.y, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = acolor;
          ctx.shadowColor = acolor;
          ctx.shadowBlur = 10;
          ctx.fill();
          ctx.restore();
        }
      }
    },
  };

  // A line's path through its points, broken at a gap exactly where
  // Chart.js breaks the line itself: always, unless the dataset spans
  // gaps. The glow is the line redrawn, so it may not add a segment
  // the chart does not draw.
  function burnThroughTrace(ctx, points, bridges) {
    var started = false;
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (!p || p.skip) {
        started = started && bridges;
        continue;
      }
      if (started) {
        ctx.lineTo(p.x, p.y);
      } else {
        ctx.moveTo(p.x, p.y);
        started = true;
      }
    }
  }

  // ---- State ----

  var chartConvergence = null;
  var chartTiming = null;
  // Shares the Timing slot with chartTiming; slotPage.timing says
  // which of the two is on screen.
  var chartTps = null;
  var chartConfidence = null;
  // Shares the Confidence slot, and like chartEntropy is built from
  // the frames payload in renderRunOverlays, because it needs every
  // position's entropy rather than the metrics payload's means.
  var chartStopping = null;

  // Map of chart name to Chart instance for zoom.
  var chartInstances = {};

  // Colors for resumed timing segments.
  var TIMING_COLOR = "#00aaff";
  var TIMING_RESUMED = "#66ccff";
  var CONFIDENCE_COLOR = "#ffb400";
  // Tokens per second shares Timing's slot and is derived from the
  // same series, so it keeps the same hue rather than claiming a new
  // one for what is the same measurement read a second way.
  var TPS_COLOR = TIMING_COLOR;

  // The pre-edit run's line on the timing and confidence charts.
  // Neutral grey rather than a second hue: timing already spends blue
  // on the branch, lighter blue on its resumed stretch, and amber on
  // canvas boundaries, so another color would read as a fourth
  // category instead of as the baseline both runs share.
  var COMPARE_ORIGINAL_COLOR = "#8b93a1";

  // Solid for the run that happened, dashed for the branch. The
  // counterfactual is the one drawn provisionally.
  var COMPARE_EDITED_DASH = [5, 3];

  // Wash strength for the band between the two runs. The grey side
  // runs stronger because it is desaturated and vanishes at the alpha
  // the saturated hues sit comfortably at.
  var BAND_ALPHA_EDITED = 0.16;
  var BAND_ALPHA_ORIGINAL = 0.24;

  // "#00aaff" -> "rgba(0, 170, 255, 0.16)". The band washes are
  // derived from the line colors rather than written out beside them,
  // so a wash cannot drift away from the line it belongs to, and
  // because their alpha has to be computed per draw anyway.
  function withAlpha(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return "rgba(" + r + ", " + g + ", " + b + ", "
      + alpha + ")";
  }

  // ---- Timing helpers ----

  function buildRemaskFrameSet(remaskEdits) {
    var set = {};
    if (!remaskEdits) { return set; }
    for (var i = 0; i < remaskEdits.length; i++) {
      var fi = remaskEdits[i].frame_index;
      set[fi] = remaskEdits[i].token_positions;
    }
    return set;
  }

  // Build cumulative elapsed values so the timing
  // line never drops to 0 after a resume. Returns
  // {values, resumeStartSet} where resumeStartSet
  // maps frame indices of each resume's first frame
  // to true.
  //
  // Runs saved since the client began carrying the
  // elapsed offset across a splice are already
  // cumulative, so this is a pass-through for them
  // and resumeStartSet comes back empty; older runs
  // still drop at each branch and get stitched here.
  function buildCumulativeTiming(raw, remaskSet) {
    var values = [];
    var resumeStartSet = {};
    var offset = 0;

    for (var i = 0; i < raw.length; i++) {
      if (i > 0 && raw[i] < raw[i - 1]) {
        offset = values[i - 1];
        resumeStartSet[i] = true;
      }
      values.push(
        +(raw[i] + offset).toFixed(3)
      );
    }
    return {
      values: values,
      resumeStartSet: resumeStartSet,
    };
  }

  // Where the resumed part of an edited run begins, as a set of frame
  // indices. Two sources, deliberately not merged: an elapsed drop is
  // the only trustworthy marker in an older run, whose timing array
  // still holds the pre-edit run's frames in full and so does not
  // line up with remask_edits at all. Once the array is aligned there
  // is no drop left to find, and the edit's own frame index is exact.
  function resumeBoundarySet(resumeStartSet, remaskSet) {
    if (Object.keys(resumeStartSet).length > 0) {
      return resumeStartSet;
    }
    var set = {};
    var keys = Object.keys(remaskSet);
    for (var i = 0; i < keys.length; i++) {
      set[keys[i]] = true;
    }
    return set;
  }

  // ---- Chart rendering ----

  // Inline Chart.js plugin: dashed vertical markers at the frame
  // indices where a new canvas (block) begins. Empty list is a
  // no-op, so single-canvas (LLaDA) runs draw nothing.
  function canvasBoundaryPlugin(boundaries) {
    return {
      id: "canvasBoundaries",
      afterDatasetsDraw: function (chart) {
        if (!boundaries || boundaries.length === 0) { return; }
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        var ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = "rgba(255,180,0,0.45)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        for (var i = 0; i < boundaries.length; i++) {
          var x = xScale.getPixelForValue(boundaries[i]);
          ctx.beginPath();
          ctx.moveTo(x, yScale.top);
          ctx.lineTo(x, yScale.bottom);
          ctx.stroke();
        }
        ctx.restore();
      },
    };
  }

  // How much the run crossfade is currently borrowing the line
  // charts, from 0 (the pins decide) to 1 (the slider decides).
  // Raised while the slider is being dragged and eased back on
  // release, so the slider never permanently governs a chart it does
  // not sit next to. Armed on press but engaged only once the thumb
  // actually moves, so a press that never becomes a drag leaves the
  // charts alone.
  var scrubWeight = 0;
  var scrubEaseHandle = null;
  var scrubArmed = false;
  var scrubEngaged = false;

  // Long enough to read as the charts handing control back, short
  // enough not to sit between the release and the answer.
  var SCRUB_EASE_MS = 180;

  // What a line chart's dataset draws at, blending the resting answer
  // (its pins) with the drag answer (the crossfade). At rest this is
  // exactly the pin state; mid-drag it is exactly the slider.
  function seriesBlendAlpha(name, index) {
    var state = linePinState[name];
    var pinned = state
      ? !!state[index === 0 ? "original" : "edited"]
      : true;
    var pinAlpha = pinned ? 1 : 0;
    if (scrubWeight === 0) {
      return pinAlpha;
    }
    var blendAlpha = (index === 0)
      ? 1 - readBlend()
      : readBlend();
    return pinAlpha
      + (blendAlpha - pinAlpha) * scrubWeight;
  }

  // The line-chart counterpart to compareBlendPlugin. Alpha at draw
  // time for the same reason: one number per dataset instead of
  // rewriting colors, and the segment coloring underneath stays as it
  // is. No-op on a run with only one series to show.
  function seriesBlendPlugin(name) {
    return {
      id: "seriesBlend-" + name,
      beforeDatasetDraw: function (chart, args) {
        if (chart.data.datasets.length < 2) { return; }
        chart.ctx.save();
        chart.ctx.globalAlpha = seriesBlendAlpha(
          name, args.index
        );
      },
      // Guarded identically to the save above, so the pair can never
      // come apart and leak canvas state into the next dataset.
      afterDatasetDraw: function (chart) {
        if (chart.data.datasets.length < 2) { return; }
        chart.ctx.restore();
      },
    };
  }

  // Alpha for the difference band. It describes a relationship
  // between the two runs rather than either one of them, so it
  // follows whichever is closer to invisible: a band bounded by a
  // line that is not there is a smear with no reading in it.
  function bandAlpha(name) {
    return Math.min(
      seriesBlendAlpha(name, 0),
      seriesBlendAlpha(name, 1)
    );
  }

  // The area between the two runs, colored by whichever bounds it
  // from above: the branch's own hue where the branch leads, the
  // original's grey where it does not. That rule needs no legend and
  // calls neither direction good nor bad, which matters because
  // "higher" means slower on the timing chart and better on
  // confidence. The runs share their prefix exactly, so the band is
  // empty until the edit and opens up only where the intervention
  // actually reached.
  //
  // Scriptable because its alpha tracks the pins and the crossfade,
  // and every path that moves either already calls chart.update,
  // which re-resolves this. Note the alpha lives in the color and not
  // in canvas state: the Filler plugin is registered globally, so it
  // draws on beforeDatasetDraw ahead of seriesBlendPlugin's inline
  // hook and would never see a globalAlpha set there.
  function compareBandFill(name, hue) {
    return function () {
      var alpha = bandAlpha(name);
      return {
        target: 0,
        above: withAlpha(hue, BAND_ALPHA_EDITED * alpha),
        below: withAlpha(
          COMPARE_ORIGINAL_COLOR, BAND_ALPHA_ORIGINAL * alpha
        ),
      };
    };
  }

  // The alpha a dataset is actually drawn at, resolved from the
  // canvas so the shared burn-through plugin can honor it without
  // knowing which chart it is decorating. Charts with a single
  // series, and the ones outside the run comparison, resolve to fully
  // opaque.
  function chartSeriesAlpha(chart, index) {
    if (chart.data.datasets.length < 2) { return 1; }
    var id = chart.canvas ? chart.canvas.id : "";
    if (id === "chart-timing") {
      return seriesBlendAlpha("timing", index);
    }
    if (id === "chart-tps") {
      return seriesBlendAlpha("tps", index);
    }
    if (id === "chart-confidence") {
      return seriesBlendAlpha("confidence", index);
    }
    if (id === "chart-stopping") {
      return seriesBlendAlpha("stopping", index);
    }
    return 1;
  }

  // A run faded out to nothing still reports values to the tooltip,
  // so a row is dropped once its series is effectively invisible. The
  // floor is above zero to also catch a crossfade parked at an end.
  function seriesRowVisible(name, item) {
    return seriesBlendAlpha(
      name, item.datasetIndex
    ) > 0.02;
  }

  // The branch is always the last dataset: an original series, when
  // there is one, is inserted ahead of it.
  function isEditedDataset(ctx) {
    var last = ctx.chart.data.datasets.length - 1;
    return ctx.datasetIndex === last;
  }

  // Redraw both line charts without animating: only alpha changed,
  // and the drag needs every frame to land immediately.
  function updateLineCharts() {
    if (chartTiming) {
      chartTiming.update("none");
    }
    if (chartTps) {
      chartTps.update("none");
    }
    if (chartConfidence) {
      chartConfidence.update("none");
    }
    if (chartStopping) {
      chartStopping.update("none");
    }
  }

  // Dim both charts' pins while the slider is driving them, so the
  // override is visible without changing what the pins hold.
  function setPinsPreviewing(previewing) {
    var buttons = document.querySelectorAll(
      ".compare-pin-btn"
    );
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle(
        "is-previewing", previewing
      );
    }
  }

  // Ease rather than snap, in both directions. An instant handover
  // reads as a glitch where a short settle reads as the charts
  // lending themselves out and taking themselves back.
  function easeScrubWeight(target) {
    cancelScrubEase();
    var from = scrubWeight;
    if (from === target) { return; }
    var start = performance.now();
    var step = function (now) {
      var t = (now - start) / SCRUB_EASE_MS;
      if (t < 1) {
        var eased = 1 - Math.pow(1 - t, 3);
        scrubWeight = from + (target - from) * eased;
        scrubEaseHandle = requestAnimationFrame(step);
      } else {
        scrubWeight = target;
        scrubEaseHandle = null;
      }
      updateLineCharts();
    };
    scrubEaseHandle = requestAnimationFrame(step);
  }

  function cancelScrubEase() {
    if (scrubEaseHandle !== null) {
      cancelAnimationFrame(scrubEaseHandle);
      scrubEaseHandle = null;
    }
  }

  // Pressing the slider only arms the preview. Engaging here instead
  // would fade the charts the instant the thumb is touched, before
  // the user has asked for anything.
  function armBlendScrub() {
    scrubArmed = true;
  }

  // Called from the slider's input handler, so the charts are
  // borrowed on the first actual movement of a press. Pointer drags
  // only: arrow keys on a focused slider produce input events with no
  // press to arm them, so keyboard adjustments move the tokens and
  // the entropy bars while the line charts stay on their pins.
  function engageBlendScrub() {
    if (!scrubArmed || scrubEngaged) { return; }
    scrubEngaged = true;
    setPinsPreviewing(true);
    easeScrubWeight(1);
  }

  function endBlendScrub() {
    scrubArmed = false;
    if (!scrubEngaged) { return; }
    scrubEngaged = false;
    setPinsPreviewing(false);
    easeScrubWeight(0);
  }

  // Which of the two runs each line chart draws. Both on by default,
  // so an edited run opens with the comparison already visible.
  // Exactly one of three states (original, edited, both) holds at any
  // time: a chart drawing neither has nothing to read, so the last
  // lit pin cannot be turned off.
  var linePinState = {
    timing: { original: true, edited: true },
    tps: { original: true, edited: true },
    confidence: { original: true, edited: true },
    stopping: { original: true, edited: true },
  };

  // Each newly-opened run starts with both runs pinned on, and with
  // the pins hidden until a chart actually renders a pre-edit series.
  // Hiding here rather than only in the render functions covers the
  // runs where a chart bails early for want of data, which would
  // otherwise leave the previous run's pins standing.
  function resetComparePins() {
    var names = Object.keys(linePinState);
    for (var i = 0; i < names.length; i++) {
      linePinState[names[i]].original = true;
      linePinState[names[i]].edited = true;
      updateComparePins(names[i], false);
    }
  }

  function comparePinsBothOn(state) {
    return state.original && state.edited;
  }

  // Reflect a chart's pin state onto its two buttons. The pin that is
  // the only one lit is marked locked, so the dead click reads as
  // unavailable before it is made rather than being swallowed.
  function refreshComparePins(name) {
    var state = linePinState[name];
    if (!state) { return; }
    var buttons = document.querySelectorAll(
      '.compare-pin-btn[data-chart="' + name + '"]'
    );
    for (var i = 0; i < buttons.length; i++) {
      var btn = buttons[i];
      var on = !!state[btn.getAttribute("data-series")];
      btn.classList.toggle("is-on", on);
      btn.classList.toggle(
        "is-locked", on && !comparePinsBothOn(state)
      );
      btn.setAttribute(
        "aria-pressed", on ? "true" : "false"
      );
    }
  }

  // With only one run there is nothing to pin, so the pair is hidden
  // for unedited runs and for those saved without the pre-edit
  // signal.
  function updateComparePins(name, hasOriginal) {
    var group = document.querySelector(
      '.compare-pins[data-chart="' + name + '"]'
    );
    if (group) {
      group.hidden = !hasOriginal;
    }
    refreshComparePins(name);
  }

  // ---- Two-page chart slots ----
  //
  // Two charts read together share one section's worth of vertical
  // space and a pager rather than each claiming a slot of their own.
  // Elapsed time and tokens per second are the same measurement read
  // two ways; confidence and the stopping chart are the model's
  // certainty and how far it had left to fall before a canvas could
  // stop. Each slot lists its pages in order, with the section and
  // the chart behind each; its buttons carry `data-<slot>-page`.
  var SLOT_PAGES = {
    timing: [
      { page: "elapsed", section: "timing-section", chart: "timing" },
      { page: "tps", section: "tps-section", chart: "tps" },
    ],
    confidence: [
      {
        page: "confidence",
        section: "confidence-section",
        chart: "confidence",
      },
      {
        page: "stopping",
        section: "stopping-section",
        chart: "stopping",
      },
    ],
  };
  // The page each slot last chose, kept from one run to the next.
  var slotPage = { timing: "elapsed", confidence: "confidence" };
  // Which pages the open run can actually draw. A run saved before a
  // signal existed may have one and not the other, and flipping to a
  // blank panel would read as a bug rather than as an absence.
  var slotReady = {
    timing: { elapsed: false, tps: false },
    confidence: { confidence: false, stopping: false },
  };

  function setSlotPage(slot, page) {
    var ready = slotReady[slot];
    if (
      !ready || !Object.prototype.hasOwnProperty.call(ready, page)
    ) {
      return;
    }
    slotPage[slot] = page;
    applySlotPage(slot);
  }

  function slotPageActive(slot) {
    var ready = slotReady[slot];
    if (ready[slotPage[slot]]) {
      return slotPage[slot];
    }
    var pages = SLOT_PAGES[slot];
    for (var i = 0; i < pages.length; i++) {
      if (ready[pages[i].page]) {
        return pages[i].page;
      }
    }
    return null;
  }

  // Each chart is built while its section is visible and may be
  // hidden again here: Chart.js sizes itself off the canvas it is
  // handed, and a canvas in a hidden section measures zero. Hiding
  // one changes the height the survivor has to fill, hence the
  // resize.
  function applySlotPage(slot) {
    var active = slotPageActive(slot);
    var pages = SLOT_PAGES[slot];
    var shown = null;
    for (var i = 0; i < pages.length; i++) {
      var section = document.getElementById(pages[i].section);
      if (section) {
        section.hidden = active !== pages[i].page;
      }
      if (active === pages[i].page) {
        shown = chartInstances[pages[i].chart];
      }
    }
    // After every section has its final visibility, so the survivor
    // measures the height it is actually left with.
    if (shown) {
      shown.resize();
    }
    refreshSlotPagers(slot, active);
  }

  // Scoped to the slot's own buttons. A page-wide query for every
  // pager would let one slot's readiness hide the other's.
  function refreshSlotPagers(slot, active) {
    var ready = slotReady[slot];
    var pages = SLOT_PAGES[slot];
    var all = true;
    for (var i = 0; i < pages.length; i++) {
      all = all && ready[pages[i].page];
    }
    var attribute = "data-" + slot + "-page";
    var buttons = document.querySelectorAll("[" + attribute + "]");
    for (var j = 0; j < buttons.length; j++) {
      buttons[j].disabled =
        buttons[j].getAttribute(attribute) === active;
      var pager = buttons[j].closest(".alt-pager");
      if (pager) {
        pager.hidden = !all;
      }
    }
  }

  function wireSlotPagers() {
    var slots = Object.keys(SLOT_PAGES);
    for (var s = 0; s < slots.length; s++) {
      wireSlotPager(slots[s]);
    }
  }

  function wireSlotPager(slot) {
    var attribute = "data-" + slot + "-page";
    var buttons = document.querySelectorAll("[" + attribute + "]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function (event) {
        setSlotPage(
          slot, event.currentTarget.getAttribute(attribute)
        );
      });
    }
  }

  // ---- Convergence and timing ----

  // Three measures can produce this curve, and which one a run got is
  // a property of the run. Two are exact and one is not, so the icon
  // beside the heading says which rather than letting a reader assume
  // the best of them.
  //
  // Only where there is something to say. A run whose mask is a real
  // token is measured the way the axis has always described, so it
  // gets no icon and the heading stays clean.
  function convergenceBasisNote(basis, modelLabel) {
    if (basis === "settlement") {
      // Exact, but measuring a different thing, and worth explaining
      // because the same chart on a LLaDA run does not mean this.
      return modelLabel
        + " has no mask token, so the curve counts positions already"
        + " holding what their canvas committed. Reading it as"
        + " \u201chow much is decided\u201d is right; the model's own"
        + " sense of certainty is a separate question.";
    }
    if (basis === "characters") {
      // Not exact, and the one a reader must not mistake for the
      // others, which is why its icon is tinted as well.
      return "Approximate: this run saved no per-token records, so"
        + " the curve counts mask characters rather than token"
        + " positions. A position resolving into a long token moves"
        + " it further than one resolving into a short token.";
    }
    // Includes an unknown basis, which means a newer server talking
    // to an older page. Silence beats guessing which it meant.
    return "";
  }

  function renderConvergenceBasis(data) {
    var icon = document.getElementById(
      "convergence-basis-info"
    );
    var tip = document.getElementById("convergence-basis-tip");
    if (!icon || !tip) { return; }

    var basis = data.convergence_basis;
    var text = convergenceBasisNote(
      basis, data.model_label || "This model"
    );
    tip.textContent = text;
    icon.hidden = !text;
    icon.classList.toggle(
      "is-approximate", basis === "characters"
    );
    icon.setAttribute(
      "aria-label", "How this convergence curve was measured"
    );
  }

  function renderConvergenceChart(data, remaskSet) {
    var canvas = document.getElementById(
      "chart-convergence"
    );
    renderConvergenceBasis(data);

    var labels = [];
    var values = [];
    for (
      var i = 0;
      i < data.convergence.length;
      i++
    ) {
      labels.push(data.convergence[i].frame);
      values.push(
        +(data.convergence[i].resolved_ratio
          * 100).toFixed(2)
      );
    }

    chartConvergence = new Chart(
      canvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: labels,
          datasets: [{
            label: "% Resolved",
            data: values,
            borderColor: "#00ff41",
            backgroundColor: "rgba(0,255,65,0.1)",
            fill: true,
            tension: 0.2,
            pointRadius: 0,
            borderWidth: 1.5,
            segment: {
              borderColor: function (ctx) {
                if (remaskSet[ctx.p1DataIndex]) {
                  return "#00aaff";
                }
                return undefined;
              },
              borderWidth: function (ctx) {
                if (remaskSet[ctx.p1DataIndex]) {
                  return 2.5;
                }
                return undefined;
              },
            },
          }],
        },
        options: convergenceOptions(remaskSet),
        plugins: [
          canvasBoundaryPlugin(data.canvas_boundaries || []),
          burnThroughPlugin,
        ],
      }
    );
    chartInstances.convergence = chartConvergence;
  }

  function convergenceOptions(remaskSet) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          callbacks: {
            title: chartSupportTooltipTitle,
            labelColor: chartSupportLineLabelColor,
            label: function (ctx) {
              return ctx.dataset.label + ": "
                + ctx.formattedValue;
            },
            afterLabel: function (ctx) {
              var pos = remaskSet[ctx.dataIndex];
              if (!pos) { return ""; }
              return overlaysRemaskSummary(pos);
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Frame",
          },
          ticks: { maxTicksLimit: 12 },
        },
        y: {
          title: {
            display: true,
            text: "% Resolved",
          },
          beginAtZero: true,
        },
      },
    };
  }

  function renderTimingChart(data, remaskSet) {
    if (
      !data.per_frame_elapsed
      || data.per_frame_elapsed.length === 0
    ) {
      timingSection.hidden = true;
      return;
    }
    // Shown before the chart is constructed, and possibly hidden
    // again by applySlotPage once its sibling has been built too:
    // Chart.js sizes itself off the canvas it is handed, and a canvas
    // in a hidden section measures zero.
    timingSection.hidden = false;
    slotReady.timing.elapsed = true;

    var canvas = document.getElementById(
      "chart-timing"
    );

    var cumResult = buildCumulativeTiming(
      data.per_frame_elapsed, remaskSet
    );
    var values = cumResult.values;
    var resumeSet = resumeBoundarySet(
      cumResult.resumeStartSet, remaskSet
    );

    var original = timingOriginalValues(data);
    var labels = compareFrameLabels(values, original);

    // Original first, so it draws beneath the branch it produced and
    // so dataset index 0 is the one the pins and crossfade fade out.
    var datasets = [];
    if (original) {
      datasets.push(compareOriginalDataset(original));
    }
    datasets.push(timingEditedDataset(
      values, remaskSet, resumeSet, !!original
    ));

    chartTiming = new Chart(
      canvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: labels,
          datasets: datasets,
        },
        options: timingOptions(remaskSet),
        plugins: [
          canvasBoundaryPlugin(data.canvas_boundaries || []),
          burnThroughPlugin,
          seriesBlendPlugin("timing"),
        ],
      }
    );
    chartInstances.timing = chartTiming;
    updateComparePins("timing", !!original);
  }

  // ---- Tokens per second ----
  //
  // The running average, not the instantaneous rate: tokens produced
  // so far over seconds spent so far. It shares the Timing slot
  // because it is the same two numbers read as a ratio, and reading
  // it as a running total keeps it level with the elapsed line beside
  // it. A per-step rate on a diffusion run is mostly the sampler's
  // reveal schedule sawtoothing, which says more about the schedule
  // than about throughput.
  //
  // Nothing new is stored for this. Every run already has its frame
  // timings, and mask counts fall out of the convergence series the
  // endpoint computes from history.txt, so it works on runs saved
  // long before the metric existed.
  function renderTpsChart(data, remaskSet, isAutoregressive) {
    var section = document.getElementById("tps-section");
    var elapsed = tpsElapsedValues(data, remaskSet);
    var produced = tokensProducedSeries(
      data, elapsed.length, isAutoregressive
    );
    if (!elapsed.length || !produced) {
      if (section) { section.hidden = true; }
      return;
    }
    // See renderTimingChart on why this is shown before building.
    if (section) { section.hidden = false; }
    slotReady.timing.tps = true;

    var values = tokenRateSeries(produced, elapsed);
    var original = tpsOriginalValues(data, isAutoregressive);
    var labels = compareFrameLabels(values, original);

    // Original first, so it draws beneath the branch it produced and
    // so dataset index 0 is the one the pins and crossfade fade out.
    var datasets = [];
    if (original) {
      datasets.push(compareOriginalDataset(original));
    }
    datasets.push(tpsEditedDataset(values, !!original));

    chartTps = new Chart(
      document.getElementById("chart-tps").getContext("2d"),
      {
        type: "line",
        data: { labels: labels, datasets: datasets },
        options: tpsOptions(remaskSet),
        plugins: [
          canvasBoundaryPlugin(data.canvas_boundaries || []),
          burnThroughPlugin,
          seriesBlendPlugin("tps"),
        ],
      }
    );
    chartInstances.tps = chartTps;
    updateComparePins("tps", !!original);
  }

  // The same stitched cumulative series the elapsed chart draws, so
  // the two charts in this slot cannot disagree about when a frame
  // landed.
  function tpsElapsedValues(data, remaskSet) {
    var raw = data.per_frame_elapsed;
    if (!raw || raw.length === 0) {
      return [];
    }
    return buildCumulativeTiming(raw, remaskSet).values;
  }

  // Tokens resolved by frame i, counted from the start of the run.
  //
  // Autoregressive runs emit exactly one token per frame, so the
  // frame index is the count and no data is needed. Diffusion runs
  // read the series the endpoint computes, because it needs the
  // canvas each frame belongs to: this used to subtract every frame's
  // mask count from the first frame's, which is right only while
  // there is one canvas, and undercounted a whole committed canvas on
  // multi-canvas DiffusionGemma runs.
  //
  // Returns null when the run carries nothing to count from, which
  // hides the chart rather than drawing a flat zero.
  function tokensProducedSeries(data, frames, isAutoregressive) {
    if (frames === 0) {
      return null;
    }
    var produced = [];
    var i;
    if (isAutoregressive) {
      for (i = 0; i < frames; i++) {
        produced.push(i + 1);
      }
      return produced;
    }
    var series = data.tokens_produced;
    if (!series || series.length === 0) {
      return null;
    }
    for (i = 0; i < frames; i++) {
      // Frames beyond the served series hold their last value rather
      // than dropping to zero, which the stitched elapsed axis can
      // reach on an edited run.
      produced.push(
        i < series.length
          ? series[i]
          : series[series.length - 1]
      );
    }
    return produced;
  }

  function tokenRateSeries(produced, elapsed) {
    var values = [];
    for (var i = 0; i < produced.length; i++) {
      var seconds = elapsed[i];
      // A frame that shares a timestamp with the run's start has no
      // window to average over. null rather than zero, so the line
      // skips the point instead of diving to the axis.
      if (!(seconds > 0)) {
        values.push(null);
      } else {
        values.push(+(produced[i] / seconds).toFixed(2));
      }
    }
    return values;
  }

  // Only autoregressive runs can show the pre-edit run here. A saved
  // run keeps the original's frame timings but not its canvas
  // history, and a rate needs both; an autoregressive run needs no
  // history, because one token per frame is structural. So a
  // diffusion comparison would have to invent the numerator.
  function tpsOriginalValues(data, isAutoregressive) {
    if (!isAutoregressive) {
      return null;
    }
    var raw = data.original_per_frame_elapsed;
    if (!raw || raw.length === 0) {
      return null;
    }
    var elapsed = buildCumulativeTiming(raw, {}).values;
    var produced = [];
    for (var i = 0; i < elapsed.length; i++) {
      produced.push(i + 1);
    }
    return tokenRateSeries(produced, elapsed);
  }

  // See timingEditedDataset for what ``paired`` switches and why.
  function tpsEditedDataset(values, paired) {
    return {
      label: paired ? "Edited" : "Tokens/s",
      data: values,
      borderColor: TPS_COLOR,
      backgroundColor: withAlpha(TPS_COLOR, 0.08),
      fill: paired
        ? compareBandFill("tps", TPS_COLOR)
        : true,
      borderDash: paired ? COMPARE_EDITED_DASH : [],
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 1.5,
      spanGaps: true,
    };
  }

  function tpsOptions(remaskSet) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          filter: function (item) {
            return seriesRowVisible("tps", item);
          },
          callbacks: {
            title: chartSupportTooltipTitle,
            labelColor: chartSupportLineLabelColor,
            label: function (ctx) {
              return ctx.dataset.label + ": "
                + ctx.formattedValue + " T/s";
            },
            afterLabel: function (ctx) {
              if (!isEditedDataset(ctx)) { return ""; }
              var pos = remaskSet[ctx.dataIndex];
              if (!pos) { return ""; }
              return "Resume point ("
                + pos.length
                + " tokens remasked)";
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: {
        x: {
          title: { display: true, text: "Frame" },
          ticks: { maxTicksLimit: 12 },
        },
        y: {
          title: { display: true, text: "Tokens/second" },
          beginAtZero: true,
        },
      },
    };
  }

  // Cumulative elapsed for the pre-edit run, or null when this run
  // carries none: an unedited run, or one saved before the signal
  // existed. That array is a single unbranched segment and so needs
  // no stitching; it goes through buildCumulativeTiming only so both
  // series are produced the same way.
  function timingOriginalValues(data) {
    var raw = data.original_per_frame_elapsed;
    if (!raw || raw.length === 0) {
      return null;
    }
    return buildCumulativeTiming(raw, {}).values;
  }

  // Frame labels spanning the longer run: a branch can outlive or
  // fall short of the run it forked from.
  function compareFrameLabels(values, original) {
    var count = values.length;
    if (original && original.length > count) {
      count = original.length;
    }
    var labels = [];
    for (var i = 0; i < count; i++) {
      labels.push(i);
    }
    return labels;
  }

  // The pre-edit run's line, shared by the timing and confidence
  // charts. Solid, neutral, and deliberately without the branch's
  // segment coloring: no remask or resume happened in this run.
  function compareOriginalDataset(values) {
    return {
      label: "Original",
      data: values,
      borderColor: COMPARE_ORIGINAL_COLOR,
      fill: false,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 1.5,
      spanGaps: true,
    };
  }

  // ``paired`` is true once there is an original series to compare
  // against, which switches the label to name its run and turns the
  // area fill into a band between the two runs. Filling both to the
  // axis instead would stack two translucent washes over the prefix
  // the runs share and read as a third color rather than as two runs.
  function timingEditedDataset(
    values, remaskSet, resumeSet, paired
  ) {
    return {
      label: paired ? "Edited" : "Elapsed",
      data: values,
      borderColor: TIMING_COLOR,
      backgroundColor: withAlpha(TIMING_COLOR, 0.08),
      fill: paired
        ? compareBandFill("timing", TIMING_COLOR)
        : true,
      borderDash: paired ? COMPARE_EDITED_DASH : [],
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 1.5,
      spanGaps: true,
      segment: {
        borderColor: function (ctx) {
          var fi = ctx.p1DataIndex;
          if (remaskSet[fi]) {
            return "#00ff41";
          }
          if (isInResumedRange(fi, resumeSet)) {
            return TIMING_RESUMED;
          }
          return undefined;
        },
        borderWidth: function (ctx) {
          if (remaskSet[ctx.p1DataIndex]) {
            return 2.5;
          }
          return undefined;
        },
      },
    };
  }

  // Check whether a frame index falls within a
  // resumed range (after a resume boundary but
  // not the remask point itself).
  function isInResumedRange(fi, resumeSet) {
    var keys = Object.keys(resumeSet);
    for (var k = 0; k < keys.length; k++) {
      if (fi >= parseInt(keys[k], 10)) {
        return true;
      }
    }
    return false;
  }

  function timingOptions(remaskSet) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          filter: function (item) {
            return seriesRowVisible("timing", item);
          },
          callbacks: {
            title: chartSupportTooltipTitle,
            labelColor: chartSupportLineLabelColor,
            label: function (ctx) {
              return ctx.dataset.label + ": "
                + ctx.formattedValue + "s";
            },
            afterLabel: function (ctx) {
              // The branch is the last dataset, and the resume is an
              // event in it alone, so the note is not repeated under
              // the original run's row.
              if (!isEditedDataset(ctx)) { return ""; }
              var pos = remaskSet[ctx.dataIndex];
              if (!pos) { return ""; }
              return "Resume point ("
                + pos.length
                + " tokens remasked)";
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Frame",
          },
          ticks: { maxTicksLimit: 12 },
        },
        y: {
          title: {
            display: true,
            text: "Seconds",
          },
          beginAtZero: true,
        },
      },
    };
  }

  // Mean per-frame confidence. Rises toward 100% as the canvas
  // converges; canvas boundaries mark each adaptive stop. Hidden
  // for legacy runs saved before confidence was recorded.
  function renderConfidenceChart(data) {
    var section = document.getElementById(
      "confidence-section"
    );
    var meanConf = data.mean_conf;
    if (!meanConf || meanConf.length === 0) {
      if (section) { section.hidden = true; }
      // The slot may still have its stopping page to show.
      applySlotPage("confidence");
      return;
    }
    // See renderTimingChart on why this is shown before building.
    if (section) { section.hidden = false; }
    slotReady.confidence.confidence = true;

    var canvas = document.getElementById(
      "chart-confidence"
    );

    var values = confidencePercentValues(meanConf);
    var original = confidenceOriginalValues(data);
    var labels = compareFrameLabels(values, original);

    var datasets = [];
    if (original) {
      datasets.push(compareOriginalDataset(original));
    }
    datasets.push(
      confidenceEditedDataset(values, !!original)
    );

    chartConfidence = new Chart(
      canvas.getContext("2d"),
      {
        type: "line",
        data: {
          labels: labels,
          datasets: datasets,
        },
        options: confidenceOptions(),
        plugins: [
          canvasBoundaryPlugin(data.canvas_boundaries || []),
          burnThroughPlugin,
          seriesBlendPlugin("confidence"),
        ],
      }
    );
    chartInstances.confidence = chartConfidence;
    updateComparePins("confidence", !!original);
    // The stopping page arrives with the frames payload, in either
    // order with this one, so each settles the slot as it lands.
    applySlotPage("confidence");
  }

  // Fractions to whole percents, preserving nulls so a frame that
  // recorded no confidence stays a gap rather than reading as zero.
  function confidencePercentValues(raw) {
    var out = [];
    for (var i = 0; i < raw.length; i++) {
      var v = raw[i];
      out.push(
        v === null || v === undefined
          ? null
          : +(v * 100).toFixed(2)
      );
    }
    return out;
  }

  // The pre-edit run's confidence, or null when this run carries
  // none.
  function confidenceOriginalValues(data) {
    var raw = data.original_mean_conf;
    if (!raw || raw.length === 0) {
      return null;
    }
    return confidencePercentValues(raw);
  }

  // See timingEditedDataset for what ``paired`` switches and why.
  function confidenceEditedDataset(values, paired) {
    return {
      label: paired ? "Edited" : "Mean confidence",
      data: values,
      borderColor: CONFIDENCE_COLOR,
      backgroundColor: withAlpha(CONFIDENCE_COLOR, 0.08),
      fill: paired
        ? compareBandFill("confidence", CONFIDENCE_COLOR)
        : true,
      borderDash: paired ? COMPARE_EDITED_DASH : [],
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 1.5,
      spanGaps: true,
    };
  }

  function confidenceOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          filter: function (item) {
            return seriesRowVisible("confidence", item);
          },
          callbacks: {
            title: chartSupportTooltipTitle,
            labelColor: chartSupportLineLabelColor,
            label: function (ctx) {
              return ctx.dataset.label + ": "
                + ctx.formattedValue + "%";
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Frame",
          },
          ticks: { maxTicksLimit: 12 },
        },
        y: {
          title: {
            display: true,
            text: "Mean confidence (%)",
          },
          beginAtZero: true,
          max: 100,
        },
      },
    };
  }

  // ---- Stopping ----
  //
  // Each frame's mean entropy against the stop threshold the run ran
  // under, for a model that stops adaptively. A log axis, because a
  // canvas falls three or four orders of magnitude on its way to the
  // line and a linear one would put all but its first frames on the
  // floor. The track is the readout's own (overlays.js), so the chart
  // and the readout above the canvas cannot disagree about a frame. A
  // committed canvas carries no entropy, so the line breaks there,
  // which is also where one canvas ends and the next begins.

  var STOPPING_COLOR = "#a98bff";
  // Where a canvas stopped, in the readout's green for a met rule.
  var STOPPING_MET_COLOR = "#00ff41";
  var STOPPING_THRESHOLD_COLOR = "rgba(0, 255, 65, 0.55)";
  // Chart.defaults' own face and size, set at the top of this file.
  var STOPPING_LABEL_FONT = "10px 'JetBrains Mono', monospace";
  // A ring on a frame where nothing changed, a dot where it stopped.
  var STOPPING_STEADY_RADIUS = 2.5;
  var STOPPING_STOP_RADIUS = 4;

  // Tear the chart down and take its page out of the slot. Called
  // before a new run's frames are fetched, beside clearEntropyChart.
  function clearStoppingChart() {
    chartStopping = chartSupportDestroy(chartStopping);
    chartInstances.stopping = null;
    slotReady.confidence.stopping = false;
    var section = document.getElementById("stopping-section");
    if (section) {
      section.hidden = true;
    }
    updateComparePins("stopping", false);
  }

  // ``data.series`` and ``data.baseline`` are the series that
  // renderRunOverlays reads off the payload before it calls this.
  function renderStoppingChart(data) {
    var rule = overlaysStopRuleFrom(data.stop_rule, null);
    var edited = rule
      ? stoppingSeries(data, data.series, false, rule)
      : null;
    if (!edited) {
      applySlotPage("confidence");
      return;
    }
    var original = stoppingSeries(data, data.baseline, true, rule);
    var section = document.getElementById("stopping-section");
    // See renderTimingChart on why this is shown before building.
    if (section) {
      section.hidden = false;
    }
    slotReady.confidence.stopping = true;
    var datasets = [];
    if (original) {
      datasets.push(stoppingOriginalDataset(original));
    }
    datasets.push(stoppingEditedDataset(edited, !!original));
    var canvas = document.getElementById("chart-stopping");
    chartStopping = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: compareFrameLabels(
          edited.values, original ? original.values : null
        ),
        datasets: datasets,
      },
      options: stoppingOptions(rule, edited),
      plugins: [
        stopThresholdPlugin(rule.threshold),
        canvasBoundaryPlugin(stoppingBoundaries(data.canvas_index)),
        burnThroughPlugin,
        seriesBlendPlugin("stopping"),
      ],
    });
    chartInstances.stopping = chartStopping;
    updateComparePins("stopping", !!original);
    applySlotPage("confidence");
  }

  // One run's series: each frame's mean entropy, null where the frame
  // is a commit or was never measured, with the track behind it and
  // the frames where a canvas stopped. Null when no frame carries
  // entropy, which is every run saved before entropy was recorded
  // everywhere.
  function stoppingSeries(data, series, original, rule) {
    var count = overlaySeriesLength(series);
    if (count === 0) {
      return null;
    }
    var track = overlaysStopTrack(
      overlaySeriesStopSource(data, series, original)
    );
    var values = [];
    var drafts = 0;
    for (var f = 0; f < count; f++) {
      var draft = track[f].kind === OVERLAYS_STOP_DRAFT;
      values.push(draft ? track[f].entropy : null);
      drafts += draft ? 1 : 0;
    }
    if (drafts === 0) {
      return null;
    }
    return {
      values: values,
      track: track,
      stops: stoppingStops(track, rule),
    };
  }

  // The last draft of every canvas that ended by the rule, by frame,
  // judged as the readout judges a commit: by its length against the
  // budget.
  function stoppingStops(track, rule) {
    var stops = {};
    for (var f = 1; f < track.length; f++) {
      if (track[f].kind !== OVERLAYS_STOP_COMMIT) {
        continue;
      }
      var reading = overlaysStopReadingAt(track, f, rule);
      var last = track[f - 1].kind === OVERLAYS_STOP_DRAFT;
      if (reading && reading.stopped && last) {
        stops[f - 1] = true;
      }
    }
    return stops;
  }

  // Which mark a frame wears: where its canvas stopped, a frame on
  // which nothing changed, or none.
  function stoppingMark(series, frame) {
    if (series.stops[frame]) {
      return "stop";
    }
    var entry = series.track[frame];
    if (entry.kind === OVERLAYS_STOP_DRAFT && entry.changed === 0) {
      return "steady";
    }
    return "none";
  }

  function stoppingMarkStyles(series) {
    var radius = { stop: STOPPING_STOP_RADIUS,
      steady: STOPPING_STEADY_RADIUS, none: 0 };
    var fill = { stop: STOPPING_MET_COLOR, steady: "transparent",
      none: STOPPING_COLOR };
    var border = { stop: STOPPING_MET_COLOR, steady: STOPPING_COLOR,
      none: STOPPING_COLOR };
    var styles = { radius: [], fill: [], border: [] };
    for (var f = 0; f < series.values.length; f++) {
      var mark = stoppingMark(series, f);
      styles.radius.push(radius[mark]);
      styles.fill.push(fill[mark]);
      styles.border.push(border[mark]);
    }
    return styles;
  }

  // See timingEditedDataset for what ``paired`` switches and why.
  // Gaps stay gaps: a line drawn across a commit would join two
  // canvases that have nothing to do with each other.
  function stoppingEditedDataset(series, paired) {
    var styles = stoppingMarkStyles(series);
    return {
      label: paired ? "Edited" : "Mean entropy",
      data: series.values,
      borderColor: STOPPING_COLOR,
      borderDash: paired ? COMPARE_EDITED_DASH : [],
      fill: false,
      tension: 0.2,
      borderWidth: 1.5,
      spanGaps: false,
      pointRadius: styles.radius,
      pointBackgroundColor: styles.fill,
      pointBorderColor: styles.border,
      pointBorderWidth: 1,
    };
  }

  function stoppingOriginalDataset(series) {
    var dataset = compareOriginalDataset(series.values);
    dataset.spanGaps = false;
    return dataset;
  }

  function stoppingOptions(rule, edited) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          filter: function (item) {
            return seriesRowVisible("stopping", item);
          },
          callbacks: {
            title: chartSupportTooltipTitle,
            labelColor: chartSupportLineLabelColor,
            label: function (ctx) {
              return ctx.dataset.label + ": "
                + overlaysStopEntropyText(ctx.parsed.y) + " nats";
            },
            afterLabel: function (ctx) {
              if (!isEditedDataset(ctx)) { return ""; }
              return stoppingNote(edited, ctx.dataIndex, rule);
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: stoppingScales(rule),
    };
  }

  function stoppingScales(rule) {
    return {
      x: {
        title: {
          display: true,
          text: "Frame",
        },
        ticks: { maxTicksLimit: 12 },
      },
      y: {
        type: "logarithmic",
        title: {
          display: true,
          text: "Mean entropy (nats)",
        },
        // The readout's own range, so the threshold always sits a
        // decade above the floor whatever the run set it to.
        suggestedMin:
          rule.threshold * OVERLAYS_STOP_TRACE_FLOOR_RATIO,
        suggestedMax: OVERLAYS_STOP_TRACE_TOP_NATS,
        ticks: { callback: stoppingTickLabel },
      },
    };
  }

  // The branch's line in the tooltip: the readout's words for the
  // frame, and where a canvas stopped, that it stopped there.
  function stoppingNote(series, frame, rule) {
    var reading = overlaysStopReadingAt(series.track, frame, rule);
    if (!reading) {
      return "";
    }
    var words = overlaysStopWords(reading);
    return series.stops[frame]
      ? words + "; the canvas stopped here"
      : words;
  }

  // Labels only the powers of ten, which is where a log axis keeps
  // its meaning; the ticks between stay as unlabelled gridlines.
  function stoppingTickLabel(value) {
    var power = Math.log10(value);
    if (Math.abs(power - Math.round(power)) > 1e-9) {
      return "";
    }
    return String(Number(value.toPrecision(1)));
  }

  // The frames where a new canvas begins: canvas_boundaries in
  // src/analytics/metrics.py, for a payload that carries the indices.
  function stoppingBoundaries(canvasIndex) {
    var boundaries = [];
    if (!Array.isArray(canvasIndex)) {
      return boundaries;
    }
    for (var i = 1; i < canvasIndex.length; i++) {
      if (canvasIndex[i] !== canvasIndex[i - 1]) {
        boundaries.push(i);
      }
    }
    return boundaries;
  }

  // Inline Chart.js plugin: the threshold as a dashed line across the
  // plot, labelled so "below the line" reads without a legend. Drawn
  // before the datasets, so a line or a mark always sits on top of
  // the annotation rather than under it. The label hangs under the
  // line at its left end: every canvas approaches the threshold from
  // above and only its last draft or two dip below, and a run opens
  // far above it.
  function stopThresholdPlugin(threshold) {
    return {
      id: "stopThreshold",
      beforeDatasetsDraw: function (chart) {
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        var y = yScale.getPixelForValue(threshold);
        if (!(y >= yScale.top && y <= yScale.bottom)) {
          return;
        }
        var ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = STOPPING_THRESHOLD_COLOR;
        ctx.fillStyle = STOPPING_THRESHOLD_COLOR;
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(xScale.left, y);
        ctx.lineTo(xScale.right, y);
        ctx.stroke();
        ctx.font = STOPPING_LABEL_FONT;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(
          "stops below " + threshold, xScale.left + 4, y + 3
        );
        ctx.restore();
      },
    };
  }

  // ---- Per-chart run pins ----

  // The pins own which runs a line chart draws at rest. They are
  // deliberately independent of the run crossfade, which only borrows
  // the charts for the duration of a drag (see engageBlendScrub).
  function handleComparePinClick(e) {
    var btn = e.target.closest(".compare-pin-btn");
    if (!btn) { return; }
    var name = btn.getAttribute("data-chart");
    var state = linePinState[name];
    if (!state) { return; }
    var series = btn.getAttribute("data-series");
    // Turning off the only lit pin would blank the chart, which is
    // the one state with nothing in it to read.
    if (state[series] && !comparePinsBothOn(state)) {
      return;
    }
    state[series] = !state[series];
    refreshComparePins(name);
    var chart = chartInstances[name];
    if (chart) {
      chart.update("none");
    }
  }

  // ---- What the page calls ----

  // The four charts built from the metrics payload, and the Timing
  // slot they settle on. The page checks for the library and clears
  // its error note first.
  function renderMetricsCharts(data, autoregressive) {
    var remaskEdits = data.remask_edits || [];
    var remaskSet = buildRemaskFrameSet(remaskEdits);

    var convergenceSection = document.getElementById(
      "convergence-section"
    );
    if (autoregressive) {
      if (convergenceSection) {
        convergenceSection.hidden = true;
      }
    } else {
      if (convergenceSection) {
        convergenceSection.hidden = false;
      }
      renderConvergenceChart(data, remaskSet);
    }
    renderTimingChart(data, remaskSet);
    renderTpsChart(data, remaskSet, autoregressive);
    // Last, so both charts have been sized while visible and the
    // slot settles on one page in the same paint.
    applySlotPage("timing");
    renderConfidenceChart(data);
    // The fourth chart, Entropy by Position, is built in
    // loadRunOverlays instead: it needs per-token records from the
    // frames payload, which that function already fetches.
  }

  // Every metrics chart back to empty, before a load and when one
  // fails. The page resets the tooltip eyes beside this, since they
  // span every chart it draws.
  function clearMetricsCharts() {
    resetComparePins();
    chartConvergence = chartSupportDestroy(chartConvergence);
    chartTiming = chartSupportDestroy(chartTiming);
    chartTps = chartSupportDestroy(chartTps);
    chartConfidence = chartSupportDestroy(chartConfidence);
    slotReady.timing.elapsed = false;
    slotReady.timing.tps = false;
    // The stopping page is the overlay load's to reset, since that is
    // the payload it is built from; see clearStoppingChart.
    slotReady.confidence.confidence = false;
  }

  // Hand the charts back to their pins at once, for a run opened
  // while an earlier drag is still easing out.
  function resetBlendScrub() {
    cancelScrubEase();
    scrubWeight = 0;
    scrubArmed = false;
    scrubEngaged = false;
    setPinsPreviewing(false);
  }

  // Follow the crossfade once a press has become a drag. The page
  // calls this after moving everything else the slider drives.
  function followBlendScrub() {
    engageBlendScrub();
    if (scrubEngaged) {
      updateLineCharts();
    }
  }

  // A line chart by the name its header buttons carry, or null.
  function lineChart(name) {
    if (!Object.prototype.hasOwnProperty.call(chartInstances, name)) {
      return null;
    }
    return chartInstances[name] || null;
  }

  // Once, at boot: the slot pagers and the pins.
  function wireLineCharts() {
    wireSlotPagers();
    document.addEventListener("click", handleComparePinClick);
  }

  return Object.freeze({
    renderMetrics: renderMetricsCharts,
    clearMetrics: clearMetricsCharts,
    renderStopping: renderStoppingChart,
    clearStopping: clearStoppingChart,
    armScrub: armBlendScrub,
    followBlend: followBlendScrub,
    endScrub: endBlendScrub,
    resetScrub: resetBlendScrub,
    chart: lineChart,
    wire: wireLineCharts,
  });
}
