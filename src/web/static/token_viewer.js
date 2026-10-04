// The Analytics detail panel's token viewer: a saved run's tokens at
// the scrubbed frame under the chosen overlay, its candidate popover,
// the token metrics strip and the stopping readout above it, the run
// crossfade between an edited run and the run it branched from, and
// the entropy chart, which follows the scrubber and lights the token
// under its bar.
//
// Loaded as a classic global script after custom_select.js,
// overlays.js, overlay_series.js, run_candidates.js,
// candidate_flicker.js and chart_support.js, which it reads, and
// before analytics.js, which creates it once. It defines one global
// name, tokenViewerCreate, and keeps the rest inside that factory, so
// the page reaches the viewer only through the object it returns.
// The crossfade is the viewer's: the page passes its moments on to
// the line charts, which read its position back, and fetches each
// run's frames for it.
//
// The factory is long because it is this file's scope. The functions
// inside it are each small, and its own statements are only
// declarations and the object it returns.

"use strict";

// Create the panel's token viewer. ``options.readTokenizer`` answers
// the open run's tokenizer, read off the catalog for the popover's
// footer and the strip's rank. ``options.onShown`` hears a run's
// frames drawn, with their payload. The crossfade's slider reports
// ``onBlendReset`` when a run resets it, ``onBlendInput`` as it
// moves, and ``onBlendPress`` and ``onBlendRelease`` around a drag.
// Elements are found by id, so the page's markup is the rest of the
// contract.
function tokenViewerCreate(options) {
  var settings = options || {};
  var needed = [
    "readTokenizer", "onShown", "onBlendReset",
    "onBlendInput", "onBlendPress", "onBlendRelease",
  ];
  for (var o = 0; o < needed.length; o++) {
    if (typeof settings[needed[o]] !== "function") {
      throw new TypeError(
        "tokenViewerCreate needs options." + needed[o]
      );
    }
  }
  var readTokenizer = settings.readTokenizer;
  var onShown = settings.onShown;
  var onBlendReset = settings.onBlendReset;
  var onBlendInput = settings.onBlendInput;
  var onBlendPress = settings.onBlendPress;
  var onBlendRelease = settings.onBlendRelease;

  // ---- Elements and state ----

  var overlayViewer =
    document.getElementById("overlay-viewer");
  var overlaySelectGroup =
    document.getElementById("overlay-select-group");
  var overlayDrawerHandle =
    document.getElementById("overlay-drawer-handle");
  var overlaySelectMount =
    document.getElementById("overlay-select-mount");
  var overlayHighlightCheckbox =
    document.getElementById("overlay-highlight-tokens");
  var overlayOutput =
    document.getElementById("overlay-output");
  var tokenMetricsStrip =
    document.getElementById("token-metrics");
  var stopReadout = document.getElementById("stop-readout");
  var overlayReadout =
    document.getElementById("overlay-readout");
  var overlayLegend =
    document.getElementById("overlay-legend");
  var overlayRevisionLegend =
    document.getElementById("overlay-revision-legend");
  var overlayEmpty =
    document.getElementById("overlay-empty");
  var overlayScrubber =
    document.getElementById("overlay-scrubber");
  var overlayScrubSlider =
    document.getElementById("overlay-scrubber-slider");
  var overlayScrubPrev =
    document.getElementById("overlay-scrub-prev");
  var overlayScrubNext =
    document.getElementById("overlay-scrub-next");
  var overlayScrubLabel =
    document.getElementById("overlay-scrubber-label");
  var overlaySelect = null;
  // The durable preferences, read once here as the generator reads
  // them once at boot: the Settings page takes effect on the next
  // load of either page. This is what makes the mask reveal
  // retroactive, since a saved diffusion run carries the same
  // per-position guess a live one does.
  var analyticsSettings = overlaysLoadSettings();
  // Cached frames payload and current overlay mode for the open run.
  var overlayData = null;
  var overlayMode = "none";
  // Frame shown by the scrubber. Defaults to the final frame so the
  // viewer opens exactly as before; scrubbing back replays earlier
  // frames through the active overlay.
  var overlayFrameIndex = 0;
  // Per-run derived data, memoized and invalidated when a new run
  // loads: the Commit Order gradient's per-position steps and the
  // diff change set (neither depends on the scrubber frame). The
  // pre-edit run needs steps of its own, since a commit step is a
  // property of a frame stream rather than of a token.
  var overlayCommitSteps = null;
  var overlayOriginalCommitSteps = null;
  var overlayDiffData = null;
  // Every frame's revised positions, for the open run and its
  // pre-edit snapshot. What is memoized is the walk; the counts
  // depend on the scrubbed frame, so they are taken at render time.
  var overlayRevisions = null;
  var overlayOriginalRevisions = null;
  // Whether the open run is autoregressive; gates Commit Order off
  // (diffusion-only for now), keeping None + Heatmap + Entropy.
  var overlayIsAutoregressive = false;
  // Candidate popover for the token overlay (mirrors the generator).
  // The page names the run it reads, "original" or "edited", and is
  // null where there is only one set to show, as on an unedited run.
  var altsPopover =
    document.getElementById("token-alts-popover");
  var altsPopoverPos = null;
  var altsPopoverPage = null;

  // Layered "Diff vs Original" controls (mirror the generator): two
  // opacity sliders plus a difference-blend toggle. State is kept
  // here so re-rendering the diff (on slider input) is cheap.
  var overlayDiffControls =
    document.getElementById("overlay-diff-controls");
  var overlayDiffOrigInput =
    document.getElementById("overlay-diff-original-opacity");
  var overlayDiffEditInput =
    document.getElementById("overlay-diff-edited-opacity");
  var overlayDiffBlendInput =
    document.getElementById("overlay-diff-blend");
  var overlayDiffOrigOpacity = 50;
  var overlayDiffEditOpacity = 100;
  var overlayDiffBlendOn = false;

  // Run-level crossfade between the pre-edit run and the branch: 1 is
  // the edited run alone, 0 the snapshot. One slider rather than the
  // diff overlay's two, because superimposed bars and tokens at
  // matching opacity just occlude each other, so the useful axis is
  // the mix. It governs every comparison surface at once (the token
  // layers and the entropy chart), which is why it lives in the modal
  // header rather than under either of them.
  var runBlendRow =
    document.getElementById("run-blend-row");
  var runBlendInput =
    document.getElementById("run-blend");
  var compareBlend = 1;

  // Per-position, so it is built from the frames payload in
  // loadRunOverlays rather than the metrics payload in loadRunCharts.
  var chartEntropy = null;

  // ---- Entropy chart plugins ----

  // Inline Chart.js plugin for the position-indexed entropy chart:
  // a dashed vertical marker at each edited position, so a What If
  // branch shows where the intervention happened and therefore where
  // the shared prefix ends. Empty list is a no-op, so unedited runs
  // draw nothing.
  //
  // Each marker carries the hue of the frame its edit was made at, on
  // the same ramp as Commit Order, so a run edited in several rounds
  // shows the order of its interventions rather than one flat colour.
  // ``colors`` is parallel to ``positions``, resolved by the caller
  // because the frame count is a property of the run and the plugin
  // only knows about geometry.
  //
  // Two hooks: the tint goes behind the bars (an edited column reads
  // as touched even where its bar is short), the dashed line goes
  // over them (a one-pixel bar would otherwise hide it).
  function substitutionMarkerPlugin(positions, colors) {
    return {
      id: "substitutionMarkers",
      beforeDatasetsDraw: function (chart) {
        if (!positions || positions.length === 0) { return; }
        var yScale = chart.scales.y;
        var ctx = chart.ctx;
        ctx.save();
        clipToChartArea(ctx, chart.chartArea);
        ctx.globalAlpha = OVERLAYS_EDIT_TINT_ALPHA;
        for (var i = 0; i < positions.length; i++) {
          var span = entropyColumnSpan(chart, positions[i]);
          if (span) {
            ctx.fillStyle = colors[i];
            ctx.fillRect(
              span.left,
              yScale.top,
              span.width,
              yScale.bottom - yScale.top
            );
          }
        }
        ctx.restore();
      },
      afterDatasetsDraw: function (chart) {
        if (!positions || positions.length === 0) { return; }
        var xScale = chart.scales.x;
        var yScale = chart.scales.y;
        var ctx = chart.ctx;
        ctx.save();
        clipToChartArea(ctx, chart.chartArea);
        ctx.globalAlpha = OVERLAYS_EDIT_LINE_ALPHA;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        for (var i = 0; i < positions.length; i++) {
          ctx.strokeStyle = colors[i];
          var x = xScale.getPixelForValue(positions[i]);
          ctx.beginPath();
          ctx.moveTo(x, yScale.top);
          ctx.lineTo(x, yScale.bottom);
          ctx.stroke();
        }
        ctx.restore();
      },
    };
  }

  // Confine drawing to the plotting area. Chart.js clips each dataset
  // for us, but the dataset-level hooks run outside that clip, so a
  // marker for a position zoom or pan has pushed off screen would
  // otherwise paint over the axes.
  function clipToChartArea(ctx, area) {
    ctx.beginPath();
    ctx.rect(
      area.left,
      area.top,
      area.right - area.left,
      area.bottom - area.top
    );
    ctx.clip();
  }

  // Pixel span of one position's column on the entropy chart. A long
  // run puts about a pixel per bar, so a highlight drawn at the true
  // bar width would be invisible; the floor mirrors the generator's
  // profile. Reads the laid-out element rather than the scale so it
  // stays correct under zoom and pan. Falls through the datasets
  // because a shorter original run has no element at a high index.
  function entropyColumnSpan(chart, index) {
    var bar = null;
    for (var di = 0; di < chart.data.datasets.length; di++) {
      var meta = chart.getDatasetMeta(di);
      if (meta && meta.data && meta.data[index]) {
        bar = meta.data[index];
        break;
      }
    }
    if (!bar) {
      return null;
    }
    var props = bar.getProps(["x", "width"], true);
    var width = Math.max(2, props.width);
    return { left: props.x - width / 2, width: width };
  }

  // Inline Chart.js plugin: a faint full-height guide behind the bar
  // under the pointer, so a one-pixel column is findable at a glance.
  // The bar itself brightens via the dataset's hoverBackgroundColor,
  // which (unlike a hand-drawn bar) still honors the crossfade alpha.
  var entropyHoverPlugin = {
    id: "entropyHover",
    beforeDatasetsDraw: function (chart) {
      var active = chart.getActiveElements();
      if (!active || active.length === 0) { return; }
      var span = entropyColumnSpan(chart, active[0].index);
      if (!span) { return; }
      var yScale = chart.scales.y;
      var ctx = chart.ctx;
      ctx.save();
      clipToChartArea(ctx, chart.chartArea);
      ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
      ctx.fillRect(
        span.left,
        yScale.top,
        span.width,
        yScale.bottom - yScale.top
      );
      ctx.restore();
    },
  };

  // Inline Chart.js plugin: the chart-to-token half of the
  // cross-highlight, so a tall warm bar can be read back to the word
  // the model was torn over.
  //
  // This has to be a plugin rather than the options.onHover callback,
  // which Chart.js only fires while the pointer is inside chartArea.
  // Leaving through the axis gutter or off the canvas therefore never
  // delivered the empty-elements call that clears the token, and the
  // last position stayed lit. afterEvent is notified for every event
  // in options.events (mouseout included) and runs after the active
  // set is recomputed, so getActiveElements is authoritative here.
  var tokenLinkPlugin = {
    id: "tokenLink",
    afterEvent: function (chart) {
      var active = chart.getActiveElements();
      var pos = active.length > 0 ? active[0].index : null;
      setTokenHighlight(pos);
      // The bar has no span behind it, so the strip takes the layer
      // the crossfade favors: the one a token hover would land on.
      setTokenMetricsHover(pos, null);
    },
  };

  // Inline Chart.js plugin: crossfade the entropy chart's two layers
  // from the run-level compareBlend. Applied as canvas alpha at draw
  // time rather than by rewriting several hundred color strings per
  // slider step, which also keeps the entropy ramp itself untouched.
  // No-op on an unedited run, where there is only one layer to show.
  var compareBlendPlugin = {
    id: "compareBlend",
    beforeDatasetDraw: function (chart, args) {
      if (chart.data.datasets.length < 2) { return; }
      var alpha = (args.index === 0)
        ? 1 - compareBlend
        : compareBlend;
      chart.ctx.save();
      chart.ctx.globalAlpha = alpha;
    },
    // Guarded identically to the save above, so the pair can never
    // come apart and leak canvas state into the next dataset.
    afterDatasetDraw: function (chart) {
      if (chart.data.datasets.length < 2) { return; }
      chart.ctx.restore();
    },
  };

  // ---- Token overlay viewer (durable commit-order / diff) ----

  // Diff needs the pre-edit snapshot and at least one remask edit.
  function overlayDiffAvailable(data) {
    return !!(
      data.records_available
      && overlaySeriesPresent(overlaySeriesOf(data, true))
      && data.remask_edits
      && data.remask_edits.length > 0
    );
  }

  // The run the viewer is showing, and the run it branched from.
  function overlayPrimary() {
    return overlayData ? overlayData.series : null;
  }

  function overlayBaseline() {
    return overlayData ? overlayData.baseline : null;
  }

  // The token array at scrubber frame ``index`` (guarded).
  function overlayFrameAt(index) {
    return overlaySeriesAt(overlayPrimary(), index);
  }

  // Revisions for whichever run a layer draws, memoized per run. The
  // pre-edit snapshot carries no edits of its own, and an edited run
  // never spans more than one canvas.
  function overlayRevisionsFor(isOriginal) {
    if (isOriginal) {
      if (overlayOriginalRevisions === null) {
        overlayOriginalRevisions = overlaySeriesRevisions(
          overlayBaseline(), singleCanvas, []
        );
      }
      return overlayOriginalRevisions;
    }
    if (overlayRevisions === null) {
      overlayRevisions = overlaySeriesRevisions(
        overlayPrimary(),
        overlayCanvasOf,
        overlayData ? overlayData.remask_edits : []
      );
    }
    return overlayRevisions;
  }

  // How many times each position of a layer had been revised by the
  // frame that layer shows: the scrubbed frame for the run, and that
  // frame clamped to the snapshot's length for the pre-edit layer.
  function overlayRevisionCountsFor(isOriginal) {
    if (!isOriginal) {
      return overlaysRevisionCounts(
        overlayRevisionsFor(false), overlayFrameIndex, overlayCanvasOf
      );
    }
    var index = overlayClampedIndex(overlayBaseline());
    if (index === null) {
      return [];
    }
    return overlaysRevisionCounts(
      overlayRevisionsFor(true), index, singleCanvas
    );
  }

  // Whether the Revisions overlay would paint anything: a diffusion
  // run that revised at least one position, asked of its saved
  // frames.
  function overlayRevisionsAvailable() {
    if (overlayIsAutoregressive || overlayData === null) {
      return false;
    }
    return overlaysHasRevisions(overlayRevisionsFor(false));
  }

  // Per-position candidate sets for the open run, or an empty list.
  function overlayAlternatives() {
    if (!overlayData || !overlayData.alternatives) {
      return [];
    }
    return overlayData.alternatives;
  }

  // The same for the pre-edit run. Present only for branches saved
  // since the snapshot began carrying its candidates, so an older
  // edited run pages through nothing.
  function overlayOriginalAlternatives() {
    if (!overlayData || !overlayData.original_alternatives) {
      return [];
    }
    return overlayData.original_alternatives;
  }

  // Whether this position has a candidate set from each run to page
  // between. Both runs record the same set left of the divergence
  // point, where the branch copies its prefix verbatim, so a pager
  // there would flip between two identical lists.
  function altsPageable(pos) {
    var divergence = overlayData
      ? divergencePosition(overlayData) : null;
    if (divergence === null || pos < divergence) {
      return false;
    }
    var original = overlayOriginalAlternatives()[pos];
    var edited = overlayAlternatives()[pos];
    return !!(
      original && original.length > 0
      && edited && edited.length > 0
    );
  }

  // ---- Candidate popover (read-only mirror of the generator's) ----

  function hideAltsPopover() {
    if (!altsPopover) {
      return;
    }
    altsPopover.hidden = true;
    altsPopover.textContent = "";
    altsPopoverPos = null;
    altsPopoverPage = null;
    // Same reason as in renderAltsPopover: the rows go without firing
    // the mouseleave that would have cleared their readout. Needed
    // here too, because scroll and resize close the popover on their
    // own rather than through a pointer leaving it.
    setCandidateMetricsHover(null);
  }

  // Which run's candidates a pageable position opens on: the one the
  // crossfade is currently favoring, so the popover agrees with what
  // the tokens and bars are showing. Both pages stay reachable
  // through the arrows either way, so the midpoint chooses a default
  // rather than gating access.
  function defaultAltsPage() {
    return compareBlend < 0.5 ? "original" : "edited";
  }

  // An autoregressive run pages by position, a diffusion run by
  // frame.
  function showAltsPopover(pos, span) {
    if (overlayIsAutoregressive) {
      altsPopoverPage = altsPageable(pos) ? defaultAltsPage() : null;
    } else {
      altsPopoverPage = candidatesPage();
    }
    renderAltsPopover(pos, span);
  }

  // Flip pages in place. Rendered without an anchor deliberately: the
  // two pages can differ in height, and re-placing the box under the
  // pointer that just clicked an arrow can slide it out from under
  // that pointer, firing the mouseleave that closes it.
  function setAltsPage(page) {
    if (altsPopoverPos === null) {
      return;
    }
    altsPopoverPage = page;
    renderAltsPopover(altsPopoverPos, null);
  }

  // With an anchor span, placed above the token (or below when that
  // would overflow). Without one, left where it already sits.
  function renderAltsPopover(pos, span) {
    if (!altsPopover) {
      return;
    }
    if (!overlayIsAutoregressive) {
      renderCandidatesPopover(pos, span);
      return;
    }
    var original = altsPopoverPage === "original";
    var alts = original
      ? overlayOriginalAlternatives()[pos]
      : overlayAlternatives()[pos];
    if (!alts || alts.length === 0) {
      hideAltsPopover();
      return;
    }
    // Each page marks the token its own run drew, so the Original
    // page does not mark the branch's substitution as chosen.
    var frame = overlayClampedFrame(
      original ? overlayBaseline() : overlayPrimary()
    );
    var chosen = frame && frame[pos] ? frame[pos].id : null;

    // Discarding the rows discards their pending mouseleave: a
    // removed node never fires one, so a readout for a row that no
    // longer exists would sit in the strip until the next hover.
    setCandidateMetricsHover(null);
    altsPopover.textContent = "";
    altsPopover.appendChild(
      overlaysBuildAltHeading(pos, altsPopoverPage, setAltsPage)
    );
    for (var i = 0; i < alts.length; i++) {
      altsPopover.appendChild(
        overlaysBuildAltRow(
          alts[i], chosen, setCandidateMetricsHover, i
        )
      );
    }
    var tokenizer = overlaysBuildAltTokenizer(readTokenizer());
    if (tokenizer) {
      altsPopover.appendChild(tokenizer);
    }
    altsPopover.classList.remove("alt-pickable");
    placeAltsPopover(span);
    altsPopoverPos = pos;
  }

  // A diffusion run's candidates at the scrubbed frame, as the
  // generator shows them: the latest captured frame at or before it,
  // on the same canvas and not from before an edit began. From the
  // frame an edit branched at on it pages between the two runs,
  // opening on the one the crossfade favours, and stays closed when
  // that run has nothing there.
  function renderCandidatesPopover(pos, span) {
    var page = altsPopoverPage;
    var reading = candidatesReading(page, pos);
    if (reading === null) {
      hideAltsPopover();
      return;
    }
    var other = page === null
      ? null
      : candidatesReading(otherAltsPage(page), pos);
    setCandidateMetricsHover(null);
    altsPopover.textContent = "";
    altsPopover.appendChild(
      overlaysBuildStepHeading(
        pos, reading.frame, reading.shown, page,
        other === null ? null : setAltsPage
      )
    );
    for (var i = 0; i < reading.set.c.length; i++) {
      altsPopover.appendChild(
        overlaysBuildAltRow(
          reading.set.c[i], reading.set.h, setCandidateMetricsHover, i
        )
      );
    }
    var tokenizer = overlaysBuildAltTokenizer(readTokenizer());
    if (tokenizer) {
      altsPopover.appendChild(tokenizer);
    }
    altsPopover.classList.remove("alt-pickable");
    placeAltsPopover(span);
    altsPopoverPos = pos;
  }

  // The canvas a saved frame belongs to, 0 for a model with only one.
  function overlayCanvasOf(frame) {
    var canvases = overlayData ? overlayData.canvas_index : null;
    if (!canvases || typeof canvases[frame] !== "number") {
      return 0;
    }
    return canvases[frame];
  }

  // The earliest frame any edit branched at, or null on an unedited
  // run. Before it both runs hold the same frames.
  function overlayDivergenceFrame() {
    var edits = overlayData ? overlayData.remask_edits || [] : [];
    var earliest = null;
    for (var e = 0; e < edits.length; e++) {
      var frame = edits[e].frame_index;
      if (earliest === null || frame < earliest) {
        earliest = frame;
      }
    }
    return earliest;
  }

  // The page a diffusion run's popover opens on: the run the
  // crossfade favours, from the frame an edit branched at on, where a
  // baseline was saved to compare against; null otherwise.
  function candidatesPage() {
    var divergence = overlayDivergenceFrame();
    var baseline = overlaySeriesLength(overlayBaseline());
    if (divergence === null || baseline === 0) {
      return null;
    }
    return overlayFrameIndex >= divergence ? defaultAltsPage() : null;
  }

  function otherAltsPage(page) {
    return page === "original" ? "edited" : "original";
  }

  // One run's set for a position, at the frame that run is showing,
  // as {frame, set, shown}, or null when it has none there. The
  // original clamps to its own last frame, as its layer does. An
  // edited run is single-canvas, so the pre-edit run's lookups are on
  // canvas 0.
  function candidatesReading(page, pos) {
    if (!overlayData) {
      return null;
    }
    if (page === "original") {
      var shown = overlayClampedIndex(overlayBaseline());
      if (shown === null) {
        return null;
      }
      return candidatesReadingOf(
        runCandidatesSetAt(
          overlayData.originalCandidateStore, shown, pos, singleCanvas
        ),
        shown
      );
    }
    return candidatesReadingOf(
      runCandidatesSetAt(
        overlayData.candidateStore, overlayFrameIndex, pos,
        overlayCanvasOf
      ),
      overlayFrameIndex
    );
  }

  function candidatesReadingOf(found, shown) {
    if (found === null) {
      return null;
    }
    return { frame: found.frame, set: found.set, shown: shown };
  }

  function singleCanvas() {
    return 0;
  }

  // Unhide before measuring: the height is unknown while hidden.
  // Without a span it stays where it already sits.
  function placeAltsPopover(span) {
    altsPopover.hidden = false;
    if (!span) {
      return;
    }
    var rect = span.getBoundingClientRect();
    var box = altsPopover.getBoundingClientRect();
    altsPopover.style.left =
      overlaysPopoverLeft(rect, box) + "px";
    altsPopover.style.top =
      overlaysPopoverTop(
        rect, box, overlayOutput.getBoundingClientRect().top
      ) + "px";
  }

  // A frame series at the scrubber's index, clamped to its end. The
  // two runs can differ in length, so the snapshot may stop short. A
  // series at the scrub position, clamped to its own end. The two
  // runs can differ in length, so a branch that outlives the one it
  // forked from must not read past the baseline's last frame.
  function overlayClampedFrame(series) {
    var index = overlayClampedIndex(series);
    if (index === null) {
      return null;
    }
    return overlaySeriesAt(series, index);
  }

  // The scrub position clamped to a series' own end, or null for an
  // empty series.
  function overlayClampedIndex(series) {
    var count = overlaySeriesLength(series);
    if (count === 0) {
      return null;
    }
    return Math.min(overlayFrameIndex, count - 1);
  }

  // Candidate rows are built by overlaysBuildAltRow in overlays.js;
  // this page's own copy was identical to the generator's and both
  // needed the same hover wiring, so they share one.

  function renderRunOverlays(data) {
    var hasRecords = !!data.records_available;
    var hasDiff = overlayDiffAvailable(data);
    if (!hasRecords && !hasDiff) {
      showOverlayUnavailable();
      return;
    }
    overlayData = data;
    // Built once here rather than at every read, so the shape the
    // server chose is resolved in one place and the rest of the page
    // only ever asks a series for a frame.
    overlayData.series = overlaySeriesOf(data, false);
    overlayData.baseline = overlaySeriesOf(data, true);
    overlayData.candidateStore = runCandidatesFromJson(
      data.candidates
    );
    overlayData.originalCandidateStore = runCandidatesFromJson(
      data.original_candidates
    );
    overlayViewer.hidden = false;
    overlayEmpty.hidden = true;
    overlayOutput.hidden = false;
    overlaySelectGroup.hidden = false;
    setOverlayDrawerOpen(false);
    resetRunBlend(hasDiff);
    // Mirror the generator: default to None; the drawer offers the
    // durable overlays (Heatmap for record runs, plus Commit Order
    // and Diff vs Original for diffusion runs with the required
    // data).
    buildOverlaySelect(data);
    setupOverlayScrubber(data);
    setOverlayMode("none");
    renderEntropyChart(data);
    onShown(data);
    refreshStopReadout();
  }

  // Configure the per-frame scrubber for the loaded run. Opens on the
  // final frame carrying records (the viewer's prior behavior); a run
  // with a single usable frame keeps the scrubber hidden and
  // disabled.
  function setupOverlayScrubber(data) {
    var series = overlaySeriesOf(data, false);
    var count = overlaySeriesLength(series);
    var maxIndex = count > 0 ? count - 1 : 0;
    overlayFrameIndex = overlaySeriesFinalIndex(series);
    if (!overlayScrubber) {
      return;
    }
    var hasMultiple = count > 1;
    overlayScrubber.hidden = !hasMultiple;
    overlayScrubSlider.min = "0";
    overlayScrubSlider.max = String(maxIndex);
    overlayScrubSlider.value = String(overlayFrameIndex);
    overlayScrubSlider.disabled = !hasMultiple;
    updateOverlayScrubLabel();
  }

  // Clamp to range, sync the slider + label, and re-render the active
  // overlay at the new frame.
  function setOverlayFrame(index) {
    if (!overlayData) {
      return;
    }
    var count = overlaySeriesLength(overlayPrimary());
    var maxIndex = count > 0 ? count - 1 : 0;
    var clamped = Math.max(0, Math.min(index, maxIndex));
    overlayFrameIndex = clamped;
    if (overlayScrubSlider) {
      overlayScrubSlider.value = String(clamped);
    }
    updateOverlayScrubLabel();
    refreshEntropyChart();
    // The spans are about to be replaced, so an open popover would be
    // anchored to a detached element.
    hideAltsPopover();
    renderCurrentOverlay();
    refreshStopReadout();
  }

  // Bring the entropy bars to the frame the scrubber now sits on. A
  // channel the run declares to vary by frame is read again there;
  // one decided once per position, or a run saved without a manifest,
  // keeps the values it opened with. Either way the fills follow the
  // frame. The datasets are edited in place and the chart is updated
  // with animation off: the slider fires continuously while dragged,
  // and an animated transition per tick would lag behind the pointer.
  function refreshEntropyChart() {
    if (!chartEntropy || !overlayData) {
      return;
    }
    var sets = chartEntropy.data.datasets;
    var channel = overlaySeriesChannel(overlayData, "entropy");
    if (overlaySeriesChannelShape(channel) === "frame|position") {
      refreshEntropyLayers(sets, channel);
    }
    entropyRecolor(sets, entropyDimsFuture(overlayData));
    chartEntropy.update("none");
  }

  // Each layer's bars, the tokens they name and their hover glow,
  // read again at the scrubbed frame. Each layer is clamped to its
  // own run, which a branch can outlive, and the labels span the
  // longer one.
  function refreshEntropyLayers(sets, channel) {
    var count = 0;
    for (var i = 0; i < sets.length; i++) {
      var source = overlayData[sets[i].seriesKey];
      if (!source) {
        throw new Error(
          "entropy layer reads no series: " + sets[i].seriesKey
        );
      }
      var layer = entropyLayerAt(
        source, channel, entropyLayerCanvasOf(sets[i].seriesKey)
      );
      sets[i].data = layer.values;
      sets[i].texts = layer.texts;
      sets[i].asOfStep = layer.asOfStep;
      sets[i].hoverBackgroundColor = entropyGlowColors(layer.values);
      count = Math.max(count, layer.values.length);
    }
    chartEntropy.data.labels = entropyLabels(count);
  }

  function updateOverlayScrubLabel() {
    if (!overlayScrubLabel) {
      return;
    }
    var count = overlaySeriesLength(overlayPrimary());
    var maxIndex = count > 0 ? count - 1 : 0;
    overlayScrubLabel.textContent =
      "Frame " + overlayFrameIndex + " / " + maxIndex;
  }

  function showOverlayUnavailable() {
    overlayViewer.hidden = false;
    overlaySelectGroup.hidden = true;
    overlayOutput.textContent = "";
    overlayOutput.classList.remove("token-layers");
    overlayOutput.hidden = true;
    overlayReadout.textContent = "";
    overlayReadout.hidden = true;
    overlayLegend.hidden = true;
    overlayRevisionLegend.hidden = true;
    if (overlayDiffControls) {
      overlayDiffControls.hidden = true;
    }
    if (overlayScrubber) {
      overlayScrubber.hidden = true;
    }
    resetRunBlend(false);
    clearTokenMetrics();
    refreshStopReadout();
    overlayEmpty.hidden = false;
  }

  function clearOverlay() {
    flickerStop();
    overlayData = null;
    overlayCommitSteps = null;
    overlayOriginalCommitSteps = null;
    overlayDiffData = null;
    overlayRevisions = null;
    overlayOriginalRevisions = null;
    overlayViewer.hidden = true;
    overlaySelectGroup.hidden = true;
    overlayOutput.textContent = "";
    overlayOutput.classList.remove("token-layers");
    overlayOutput.hidden = false;
    overlayReadout.textContent = "";
    overlayReadout.hidden = true;
    overlayLegend.hidden = true;
    overlayRevisionLegend.hidden = true;
    if (overlayDiffControls) {
      overlayDiffControls.hidden = true;
    }
    if (overlayScrubber) {
      overlayScrubber.hidden = true;
    }
    resetRunBlend(false);
    clearTokenMetrics();
    overlayEmpty.hidden = true;
  }

  // Slide the corner drawer open/closed and flip its handle glyph
  // (matches the generator's overlay drawer behavior).
  function setOverlayDrawerOpen(open) {
    if (!overlaySelectGroup) {
      return;
    }
    overlaySelectGroup.classList.toggle("open", open);
    if (overlayDrawerHandle) {
      overlayDrawerHandle.innerHTML =
        open ? "\u203A" : "\u2039";
    }
  }

  // Build the overlay custom-select mirroring the generator: None /
  // Heatmap for every record run, Entropy for runs that saved it,
  // plus Commit Order (diffusion only) and Diff vs Original (any
  // model with an edited run and its original snapshot), each gated
  // on data availability.
  function buildOverlaySelect(data) {
    var canDiff = overlayDiffAvailable(data);
    var options = [
      { value: "none", label: "None" },
      {
        value: "heatmap",
        label: "Heatmap",
        disabled: !data.records_available,
      },
    ];
    // Entropy is gated on the saved data, not the model type: it
    // shows how undecided the model was over the whole vocabulary,
    // which is a different question than the confidence Heatmap
    // answers.
    if (overlaySeriesCarriesEntropy(data)) {
      options.push({ value: "entropy", label: "Entropy" });
    }
    // What reading each token erased from a state-space model's
    // state, for the runs that recorded it.
    if (overlaySeriesCarriesForgetting(data)) {
      options.push({ value: "forgetting", label: "Forgetting" });
    }
    // Commit Order tints by resolution step, which a left-to-right
    // run does not have (its commit order is just position order).
    if (!overlayIsAutoregressive) {
      options.push({
        value: "commit",
        label: "Commit Order",
        disabled: !data.records_available,
      });
    }
    // How often each position changed its mind, for a run that
    // revised something: DiffusionGemma, today.
    if (overlayRevisionsAvailable()) {
      options.push({ value: "revisions", label: "Revisions" });
    }
    // A What If substitution gives autoregressive runs a real branch
    // to diff, so list it for them too once the data is there.
    if (!overlayIsAutoregressive || canDiff) {
      options.push({
        value: "diff",
        label: "Diff vs Original",
        disabled: !canDiff,
        title: canDiff
          ? undefined
          : "Only available for an edited run saved with"
            + " its original snapshot.",
      });
    }
    overlaySelectMount.innerHTML = "";
    overlaySelect = createCustomSelect(options, "none");
    overlaySelectMount.appendChild(overlaySelect);
    sizeCustomSelect(overlaySelect);
    overlaySelect.addEventListener("change", function () {
      setOverlayMode(overlaySelect.value);
    });
  }

  function setOverlayMode(mode) {
    overlayMode = mode;
    if (overlaySelect && overlaySelect.value !== mode) {
      overlaySelect.value = mode;
    }
    overlayLegend.hidden = mode !== "commit";
    overlayRevisionLegend.hidden = mode !== "revisions";
    if (overlayDiffControls) {
      overlayDiffControls.hidden = mode !== "diff";
    }
    hideAltsPopover();
    renderCurrentOverlay();
  }

  // Re-render the active overlay mode at the current scrubber frame.
  // Called on a mode change and on every scrubber move. Both are
  // reasons a stationary pointer now reads something else, so the
  // metrics strip is refreshed here rather than at each call site.
  function renderCurrentOverlay() {
    if (overlayMode === "diff") {
      renderDiffOverlay();
    } else if (overlayMode === "commit") {
      renderCommitOverlay();
    } else if (overlayMode === "revisions") {
      renderRevisionsOverlay();
    } else if (overlayMode === "heatmap") {
      renderHeatmapOverlay();
    } else if (overlayMode === "entropy") {
      renderEntropyOverlay();
    } else if (overlayMode === "forgetting") {
      renderForgettingOverlay();
    } else {
      renderNoneOverlay();
    }
    refreshTokenMetrics();
  }

  // Plain tokens at the current scrubber frame, no coloring (None).
  function renderNoneOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: function () { return null; },
    });
  }

  // Confidence heatmap: recolor resolved tokens at the current frame
  // by their persisted per-token confidence, using the shared
  // heatColor scale. Kept for autoregressive runs too (the natural
  // per-token confidence view). Masked positions render as the mask
  // glyph.
  function renderHeatmapOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: function (index, tok) {
        if (typeof tok.c === "number") {
          return heatColor(tok.c);
        }
        return null;
      },
    });
  }

  // Entropy: recolor resolved tokens at the current frame by the
  // entropy recorded with each, on a decisive (cool) to torn (hot)
  // ramp. That is the frame's own reading, since a diffusion position
  // is re-decided at every step; a commit, which carries none, is
  // colored from its canvas's last draft. Each layer borrows from its
  // own run.
  function renderEntropyOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: entropyColorFor(overlayPrimary(), overlayCanvasOf),
      originalColorFor: entropyColorFor(
        overlayBaseline(), singleCanvas
      ),
    });
  }

  // One layer's token colors under the Entropy overlay. What it
  // borrows is found once per render rather than once per token.
  function entropyColorFor(series, canvasOf) {
    var borrow = entropyBorrowFor(series, canvasOf);
    var borrowed = borrow ? borrow.tokens : null;
    return function (index, tok) {
      var value = entropyOfToken(tok, borrowed, index);
      return value === null ? null : entropyColor(value);
    };
  }

  // What a layer's frame on screen borrows its entropy from, as
  // {step, tokens}, or null when that frame carries its own or there
  // is none to borrow.
  function entropyBorrowFor(series, canvasOf) {
    var at = overlayClampedIndex(series);
    if (at === null) {
      return null;
    }
    var source = overlaySeriesEntropyFrame(series, at, canvasOf);
    if (source < 0 || source === at) {
      return null;
    }
    return { step: source, tokens: overlaySeriesAt(series, source) };
  }

  // A position's entropy: its token's own, else the borrowed frame's
  // at the same position, else null.
  function entropyOfToken(tok, borrowed, index) {
    if (tok && typeof tok.e === "number") {
      return tok.e;
    }
    var other = borrowed ? borrowed[index] : null;
    if (other && typeof other.e === "number") {
      return other.e;
    }
    return null;
  }

  // Forgetting: recolor tokens by what reading each one erased from a
  // state-space model's recurrent state, dim for little and bright
  // for much. Read once per token as it went in, so like entropy a
  // position's value never changes across frames.
  function renderForgettingOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: function (index, tok) {
        if (typeof tok.f === "number") {
          return forgettingColor(tok.f);
        }
        return null;
      },
    });
  }

  // Commit Order. Unlike the per-token modes above, its colors come
  // from the frame stream rather than from fields on the token, so
  // the pre-edit layer cannot be described by the same callback: it
  // needs its own steps, computed from the snapshot's frames.
  function renderCommitOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    // Commit steps come from the full frame stream (final frame is
    // ground truth), so they are memoized per run and applied to
    // whichever frame the scrubber shows (mirrors the generator).
    if (overlayCommitSteps === null) {
      overlayCommitSteps = overlaySeriesCommitSteps(overlayPrimary());
    }
    var baseline = overlayBaseline();
    if (overlayOriginalCommitSteps === null
      && overlaySeriesPresent(baseline)) {
      overlayOriginalCommitSteps =
        overlaySeriesCommitSteps(baseline);
    }
    // The ramp's denominator is the last frame index each series can
    // reach, asked of the series rather than of a frame array: an
    // append-only run has no array to measure, and the two shapes
    // have to normalize the same way or the same run would read
    // differently depending on how it was stored.
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: commitColorFor(
        overlayCommitSteps,
        overlaySeriesLength(overlayPrimary()) - 1
      ),
      originalColorFor: commitColorFor(
        overlayOriginalCommitSteps || [],
        overlaySeriesLength(baseline) - 1
      ),
    });
  }

  function commitColorFor(steps, maxStep) {
    return function (index) {
      var step = steps[index];
      if (typeof step === "number" && step >= 0) {
        return commitColor(step, maxStep);
      }
      return null;
    };
  }

  // Revisions: tint each settled token by how many times its position
  // had changed its mind by the scrubbed frame. Its colors come from
  // the frame stream, as Commit Order's do, so the pre-edit layer
  // counts its own run at the frame that layer shows.
  function renderRevisionsOverlay() {
    overlayReadout.hidden = true;
    overlayReadout.textContent = "";
    renderOverlayTokens({
      frame: overlayFrameAt(overlayFrameIndex),
      colorFor: revisionColorFor(overlayRevisionCountsFor(false)),
      originalColorFor: revisionColorFor(
        overlayRevisionCountsFor(true)
      ),
    });
  }

  function revisionColorFor(counts) {
    return function (index) {
      return revisionColor(counts[index]);
    };
  }

  // Render the active mode's tokens: one layer normally, two stacked
  // and crossfaded when the run carries a pre-edit snapshot. ``opts``
  // carries the edited ``frame``, a colorFor(index, token) that never
  // sees a masked position, and an optional originalColorFor for
  // modes whose colors are not a function of the token alone.
  //
  // The default is to reuse the same callback for both layers, which
  // is what makes the comparison mean anything: the pre-edit layer is
  // colored by its own confidence or entropy, not the branch's.
  function renderOverlayTokens(opts) {
    flickerStop();
    overlayOutput.textContent = "";
    tokenHighlightPos = null;
    var edited = {
      colorFor: overlayColorFn(opts.colorFor),
      classFor: editedClassFn,
      revealMask: overlaysDrawsGuess(analyticsSettings),
      opacityFor: overlayOpacityFn,
    };
    var original = overlayComparisonFrame();
    if (original !== null) {
      renderOverlayLayers(original, opts.frame || [], edited, {
        colorFor: overlayColorFn(
          opts.originalColorFor || opts.colorFor
        ),
        revealMask: overlaysDrawsGuess(analyticsSettings),
        opacityFor: overlayOpacityFn,
      });
      return;
    }
    overlayOutput.classList.remove("token-layers");
    if (!opts.frame) {
      return;
    }
    var fragment = document.createDocumentFragment();
    var spans = [];
    for (var i = 0; i < opts.frame.length; i++) {
      spans.push(
        overlaysBuildTokenSpan(
          i, opts.frame[i], OVERLAYS_MASK_CHAR, edited
        )
      );
      fragment.appendChild(spans[i]);
    }
    overlayOutput.appendChild(fragment);
    if (overlayCandidatesCycle()) {
      flickerStart([flickerLayer(
        spans, opts.frame, overlayData.candidateStore,
        overlayFrameIndex, overlayCanvasOf
      )], OVERLAYS_MASK_CHAR);
    }
  }

  // Whether a saved run's unsettled positions cycle through their
  // candidates. A saved run has always finished, so the choice is the
  // question; reduced motion is answered where the cycling starts.
  function overlayCandidatesCycle() {
    return analyticsSettings.unsettledShows === "candidates"
      && overlayData !== null;
  }

  // Cycle both stacked layers, each with its own run's candidates at
  // the frame it shows, as the generator does: the pre-edit run's for
  // the original layer, clamped as that layer is.
  function overlayStackedFlicker(layers, origTokens, editedTokens) {
    if (!overlayCandidatesCycle()) {
      return;
    }
    var index = overlayClampedIndex(overlayBaseline());
    flickerStart([
      flickerLayer(
        layers[0].children, origTokens,
        overlayData.originalCandidateStore,
        index === null ? -1 : index, singleCanvas
      ),
      flickerLayer(
        layers[1].children, editedTokens, overlayData.candidateStore,
        overlayFrameIndex, overlayCanvasOf
      ),
    ], OVERLAYS_MASK_CHAR);
  }

  // Fade an unsettled position by the confidence the run recorded for
  // it, the same curve the generator draws live. There is no remask
  // selection to hold solid here, so unlike the generator's version
  // this is the curve and nothing else.
  function overlayOpacityFn(index, tok, masked) {
    if (!masked) {
      return null;
    }
    if (!tok) {
      return MASK_OPACITY_FLOOR;
    }
    return overlaysMaskOpacity(tok.c);
  }

  // Mark the positions a saved edit touched, so a run's interventions
  // can be found by looking rather than by remembering. Only ever
  // given to the branch layer: the pre-edit run below it is what the
  // model did on its own, and marking it would claim otherwise.
  function editedClassFn(index) {
    if (positionWasEdited(editedPositionMarks(overlayData), index)) {
      return "token-edited";
    }
    return "";
  }

  // Stack the pre-edit run under the branch at the crossfade's mix.
  function renderOverlayLayers(
    origTokens, editedTokens, edited, original
  ) {
    overlayOutput.classList.add("token-layers");
    var editedTakes = overlaysEditedOwnsPointer(
      1 - compareBlend, compareBlend
    );
    var originalLayer = overlaysBuildTokenLayer(origTokens, {
      layerClass: "token-layer-original",
      opacity: 1 - compareBlend,
      interactive: !editedTakes,
      colorFor: original.colorFor,
      revealMask: original.revealMask,
      opacityFor: original.opacityFor,
    });
    var editedLayer = overlaysBuildTokenLayer(editedTokens, {
      layerClass: "token-layer-edited",
      opacity: compareBlend,
      interactive: editedTakes,
      colorFor: edited.colorFor,
      classFor: edited.classFor,
      revealMask: edited.revealMask,
      opacityFor: edited.opacityFor,
    });
    overlayOutput.appendChild(originalLayer);
    overlayOutput.appendChild(editedLayer);
    overlayStackedFlicker(
      [originalLayer, editedLayer], origTokens, editedTokens
    );
  }

  // ---- Token hover highlight ----

  // The pointer-driven half of the highlight, which is pure CSS once
  // the class is on the container (see .token-hover-highlight in
  // style.css). Analytics never applied it at all before, which left
  // the chart-to-token direction lighting tokens that a direct hover
  // could not.
  //
  // The preference is shared with the generator through the settings
  // blob, so the drawer checkbox here and the one there mean the same
  // thing rather than each page keeping its own idea.
  function updateOverlayHoverHighlight() {
    var on = overlaysReadHighlightTokens();
    if (overlayHighlightCheckbox) {
      overlayHighlightCheckbox.checked = on;
    }
    if (!overlayOutput) {
      return;
    }
    overlayOutput.classList.toggle("token-hover-highlight", on);
  }

  function onOverlayHighlightToggle() {
    overlaysWriteHighlightTokens(overlayHighlightCheckbox.checked);
    updateOverlayHoverHighlight();
  }

  // ---- Cross-highlighting: token overlay <-> entropy chart ----

  // Position currently lit from the chart side, so a pointer sweeping
  // the bars does not re-query the DOM on every mousemove. Reset
  // whenever the overlay re-renders, since that drops the class.
  var tokenHighlightPos = null;

  // Light the token(s) at a position. There are two when the run is
  // layered, and lighting both keeps the mark visible wherever the
  // crossfade happens to sit. A token that renders to nothing, a line
  // break, gets the extra class that stands a marker in its place,
  // since the tint alone would have no box to fill.
  function setTokenHighlight(pos) {
    if (tokenHighlightPos === pos) {
      return;
    }
    clearTokenHighlight();
    tokenHighlightPos = pos;
    if (pos === null || !overlayOutput) {
      return;
    }
    var spans = overlayOutput.querySelectorAll(
      "[data-pos=\"" + pos + "\"]"
    );
    for (var i = 0; i < spans.length; i++) {
      spans[i].classList.add("token-cross-highlight");
      if (overlaysTokenIsZeroWidth(spans[i].textContent)) {
        spans[i].classList.add("token-zero-width");
      }
    }
  }

  function clearTokenHighlight() {
    tokenHighlightPos = null;
    if (!overlayOutput) {
      return;
    }
    var lit = overlayOutput.querySelectorAll(
      ".token-cross-highlight"
    );
    for (var i = 0; i < lit.length; i++) {
      lit[i].classList.remove("token-cross-highlight");
      lit[i].classList.remove("token-zero-width");
    }
  }

  // The reverse: light the entropy bar at a token position. Chart.js
  // already owns this highlight (the hover plugin's column guide and
  // the bars' own hoverBackgroundColor both key off active elements),
  // so driving it from a token hover is a matter of setting those.
  // A no-op for runs without the chart.
  function setEntropyBarHighlight(pos) {
    if (!chartEntropy) {
      return;
    }
    var elements = [];
    var datasets = chartEntropy.data.datasets;
    for (var i = 0; pos !== null && i < datasets.length; i++) {
      if (pos < datasets[i].data.length) {
        elements.push({ datasetIndex: i, index: pos });
      }
    }
    chartEntropy.setActiveElements(elements);
    chartEntropy.update("none");
  }

  // The pre-edit run's tokens at the scrubber's frame, or null when
  // there is nothing to compare against. Clamped to the snapshot's
  // final frame once it ends, since a branch can outlive or fall
  // short of the run it forked from (mirrors renderDiffOverlay).
  function overlayComparisonFrame() {
    if (!overlayData || !overlayDiffAvailable(overlayData)) {
      return null;
    }
    return overlayClampedFrame(overlayBaseline());
  }

  // Spare every mode from repeating the masked-position check: a mask
  // takes its color from .token-mask, not from the overlay.
  function overlayColorFn(colorFor) {
    return function (index, tok) {
      if (!tok || tok.m) {
        return null;
      }
      return colorFor(index, tok);
    };
  }

  // ---- Token metrics strip ----
  //
  // The readout above the canvas, rendered by the same shared
  // function the generator uses so a token reads identically on both
  // pages. Two sources feed it here: a direct token hover, and the
  // entropy chart (through tokenLinkPlugin), which is what makes a
  // tall bar readable as a word without moving the pointer to the
  // text.
  var metricsHoverPos = null;

  // Which stacked run the reading came from, taken from the hovered
  // span's own layer so the strip reports what is on screen.
  var metricsHoverOriginal = false;

  // The candidate under the pointer in the popover, or null. A
  // second, independent hover source: the left group answers "what is
  // at this position" and this answers "what about the one I am
  // reading".
  var metricsCandidate = null;

  function setTokenMetricsHover(pos, target) {
    metricsHoverPos = pos;
    metricsHoverOriginal =
      pos === null ? false : metricsLayerIsOriginal(target);
    refreshTokenMetrics();
  }

  // Fed by every candidate row. The reading carries the rank; the
  // width it is measured against comes from the page, since a row
  // cannot know it.
  function setCandidateMetricsHover(reading) {
    metricsCandidate = reading === null ? null : {
      text: reading.t,
      probability: reading.p,
      rank: reading.rank || null,
      vocabSize: metricsVocabSize(),
    };
    refreshTokenMetrics();
  }

  // The output width of the model that produced the run on screen,
  // read from the run itself and never from a resident worker: this
  // page is routinely looking at a run whose checkpoint is not
  // loaded. Runs saved before this was recorded report no width, and
  // the rank then shows without a denominator rather than with a
  // wrong one.
  function metricsVocabSize() {
    return readTokenizer().model_vocab_size || null;
  }

  // Re-read the held position, for anything that changes what a
  // stationary pointer is pointing at: a new frame, a new overlay, a
  // different run.
  function refreshTokenMetrics() {
    overlaysRenderTokenMetrics(
      tokenMetricsStrip, buildTokenMetricsReading()
    );
    // The strip's width changes with what it reads, the readout's
    // words with nothing a hover does, so only the fit is redone.
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  function clearTokenMetrics() {
    metricsHoverPos = null;
    metricsHoverOriginal = false;
    metricsCandidate = null;
    overlaysRenderTokenMetrics(tokenMetricsStrip, null);
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  // A crossfade hands the pointer to the other layer at the midpoint.
  // A stationary reading has no new span to ask, so it re-derives
  // from ownership, which is what the next hover would report anyway.
  function refreshTokenMetricsLayer() {
    if (metricsHoverPos !== null) {
      metricsHoverOriginal = metricsLayerIsOriginal(null);
    }
    refreshTokenMetrics();
    // The readout follows the same layer, so a crossfade past the
    // midpoint moves it to the other run as well.
    refreshStopReadout();
  }

  // ---- The stopping readout ----
  //
  // The generator's readout for a saved run; overlays.js holds the
  // rule and its drawing. It reads the scrubbed frame of whichever
  // run takes the pointer, which is the rule the strip names its run
  // by, and shows only for a run whose model stops adaptively: the
  // frames payload says so by carrying the rule that run stopped by.

  function refreshStopReadout() {
    overlaysRenderStopReadout(stopReadout, stopReadoutReading());
    overlaysFitStopReadout(tokenMetricsStrip, stopReadout);
  }

  function stopReadoutReading() {
    if (!overlayData) {
      return null;
    }
    var rule = overlaysStopRuleFrom(overlayData.stop_rule, null);
    if (rule === null) {
      return null;
    }
    var original = metricsLayered() && metricsLayerIsOriginal(null);
    var series = original ? overlayBaseline() : overlayPrimary();
    var index = overlayClampedIndex(series);
    if (index === null) {
      return null;
    }
    var track = overlaysStopTrack(
      overlaySeriesStopSource(overlayData, series, original)
    );
    return overlaysStopReadingAt(track, index, rule);
  }

  // Chart hover has no span, so it falls back to whichever layer
  // takes the pointer, which is the one the user could have hovered
  // instead.
  function metricsLayerIsOriginal(target) {
    if (target && target.closest) {
      var layer = target.closest(".token-layer");
      if (layer) {
        return layer.classList.contains("token-layer-original");
      }
    }
    if (!metricsLayered()) {
      return false;
    }
    if (overlayMode === "diff") {
      return !overlaysEditedOwnsPointer(
        overlayDiffOrigOpacity, overlayDiffEditOpacity
      );
    }
    return !overlaysEditedOwnsPointer(
      1 - compareBlend, compareBlend
    );
  }

  // Whether both runs are on the canvas together. Every layered mode
  // gates on the same thing the crossfade does.
  function metricsLayered() {
    return !!(overlayData && overlayDiffAvailable(overlayData));
  }

  // The tokens the canvas is drawing for the hovered layer, both
  // clamped to their own final frame the way the render paths clamp
  // them.
  function metricsFrameTokens() {
    if (metricsHoverOriginal) {
      return overlayClampedFrame(overlayBaseline());
    }
    return overlayFrameAt(overlayFrameIndex);
  }

  // Assemble one reading, or null when the held position no longer
  // names a token in the frame now on screen.
  function buildTokenMetricsReading() {
    if (metricsHoverPos === null || !overlayData) {
      return null;
    }
    var tokens = metricsFrameTokens();
    if (!tokens || metricsHoverPos >= tokens.length) {
      return null;
    }
    var index = metricsHoverPos;
    var tok = tokens[index];
    var masked = !tok || !!tok.m;
    var entropy = metricsEntropyReading(index, tok);
    return {
      position: index,
      total: tokens.length,
      tokenText: tok ? tok.t : "",
      masked: masked,
      maskChar: OVERLAYS_MASK_CHAR,
      confidence: metricsConfidence(tok, masked),
      entropy: entropy.value,
      extra: overlaysEntropyNote(
        metricsExtra(index, tok), entropy.asOfStep
      ),
      candidate: metricsCandidate,
      runLabel: metricsRunLabel(),
    };
  }

  // The hovered position's entropy: its token's own, or on a commit
  // the value its canvas's last draft held there, with that draft's
  // step so the strip can say whose reading it is.
  function metricsEntropyReading(index, tok) {
    if (tok && typeof tok.e === "number") {
      return { value: tok.e, asOfStep: null };
    }
    var borrow = metricsHoverOriginal
      ? entropyBorrowFor(overlayBaseline(), singleCanvas)
      : entropyBorrowFor(overlayPrimary(), overlayCanvasOf);
    var value = borrow
      ? entropyOfToken(null, borrow.tokens, index)
      : null;
    if (value === null) {
      return { value: null, asOfStep: null };
    }
    return { value: value, asOfStep: borrow.step };
  }

  // A resolved token from a run saved before confidence was recorded
  // reads as a dash rather than as zero, which would have claimed the
  // model was certain of nothing. A mask keeps the zero it reported.
  function metricsConfidence(tok, masked) {
    if (!tok) {
      return 0;
    }
    if (typeof tok.c === "number") {
      return tok.c;
    }
    return masked ? 0 : null;
  }

  // The overlay-specific line. Computed at hover time from the same
  // memoized state the coloring uses, so no per-token callback has to
  // be threaded through the render paths to carry it.
  function metricsExtra(index, tok) {
    if (overlayMode === "forgetting") {
      return overlaysForgettingReading(tok);
    }
    if (overlayMode === "commit") {
      var steps = metricsHoverOriginal
        ? overlayOriginalCommitSteps
        : overlayCommitSteps;
      var step = steps ? steps[index] : null;
      if (typeof step !== "number" || step < 0) {
        return "";
      }
      return "Resolved at step: " + step;
    }
    if (overlayMode === "revisions") {
      return overlaysRevisionReading(
        overlayRevisionCountsFor(metricsHoverOriginal)[index]
      );
    }
    if (overlayMode === "diff" && overlayDiffData) {
      if (overlayDiffData.origins[index]) {
        return "(remasked here)";
      }
      if (overlayDiffData.changed[index]) {
        return "was: " + overlayDiffData.origText[index];
      }
    }
    return "";
  }

  // Named only while both runs are on the canvas together. With one
  // run drawn there is nothing to disambiguate.
  function metricsRunLabel() {
    if (!metricsLayered()) {
      return "";
    }
    return metricsHoverOriginal ? "Original" : "Edited";
  }

  // Layered diff (mirrors the generator): the original and edited
  // final frames are stacked with independent opacity and an optional
  // difference blend, driven by the control row. The shared builder
  // in overlays.js owns the layer construction.
  function renderDiffOverlay() {
    flickerStop();
    // The change set is computed from the two runs' final frames (so
    // it is stable across the scrub) and memoized; only the rendered
    // layers vary per frame.
    if (overlayDiffData === null) {
      var curFinal = overlaySeriesFinal(overlayPrimary());
      var origFinal = overlaySeriesFinal(overlayBaseline());
      overlayDiffData = overlaysComputeDiff(
        curFinal, origFinal, overlayData.remask_edits
      );
    }
    var diff = overlayDiffData;
    overlayReadout.hidden = false;
    overlayReadout.textContent =
      "Diverged " + diff.changedCount
      + "/" + diff.totalCount;

    // Edited layer at the current frame; original layer clamped to
    // its final frame once it ends (the runs can differ in length /
    // resume boundaries), matching the generator (app.js
    // renderDiffOverlay).
    var editedTokens = overlayFrameAt(overlayFrameIndex) || [];
    var origTokens = overlayClampedFrame(overlayBaseline()) || [];

    overlayOutput.textContent = "";
    tokenHighlightPos = null;
    overlayOutput.classList.add("token-layers");
    var layered = overlaysBuildDiffLayers(
      origTokens,
      editedTokens,
      diff,
      {
        originalOpacity: overlayDiffOrigOpacity,
        editedOpacity: overlayDiffEditOpacity,
        blend: overlayDiffBlendOn,
        revealMask: overlaysDrawsGuess(analyticsSettings),
        opacityFor: overlayOpacityFn,
      }
    );
    // Taken before the append, which empties the fragment.
    var stacked = [layered.children[0], layered.children[1]];
    overlayOutput.appendChild(layered);
    overlayStackedFlicker(stacked, origTokens, editedTokens);
  }

  // Wire the diff control row once: sliders and the blend toggle
  // update the retained state and re-render only while the diff
  // overlay is the active mode.
  function wireOverlayDiffControls() {
    if (overlayDiffOrigInput) {
      overlayDiffOrigInput.addEventListener("input", function () {
        overlayDiffOrigOpacity = Number(overlayDiffOrigInput.value);
        rerenderDiffOverlay();
      });
    }
    if (overlayDiffEditInput) {
      overlayDiffEditInput.addEventListener("input", function () {
        overlayDiffEditOpacity = Number(overlayDiffEditInput.value);
        rerenderDiffOverlay();
      });
    }
    if (overlayDiffBlendInput) {
      overlayDiffBlendInput.addEventListener("change", function () {
        overlayDiffBlendOn = !!overlayDiffBlendInput.checked;
        rerenderDiffOverlay();
      });
    }
  }

  // The three controls above share one response: redraw the layers
  // and re-read the strip, which the opacity sliders can flip between
  // runs.
  function rerenderDiffOverlay() {
    if (overlayMode !== "diff") {
      return;
    }
    renderDiffOverlay();
    refreshTokenMetricsLayer();
  }

  // Wire the per-frame scrubber once: the slider and the prev/next
  // arrows all route through setOverlayFrame, which clamps, syncs the
  // controls, and re-renders the active overlay at the chosen frame.
  function wireOverlayScrubber() {
    if (overlayScrubSlider) {
      overlayScrubSlider.addEventListener("input", function () {
        setOverlayFrame(Number(overlayScrubSlider.value));
      });
    }
    if (overlayScrubPrev) {
      overlayScrubPrev.addEventListener("click", function () {
        setOverlayFrame(overlayFrameIndex - 1);
      });
    }
    if (overlayScrubNext) {
      overlayScrubNext.addEventListener("click", function () {
        setOverlayFrame(overlayFrameIndex + 1);
      });
    }

    // Candidate popover on token hover, for runs saved with the
    // Alternatives capture. Read-only here (substitution lives on the
    // generator, which still holds the worker's run state). The same
    // hover lights the matching entropy bar.
    if (overlayOutput) {
      overlayOutput.addEventListener(
        "mouseover",
        function (e) {
          var target = e.target;
          if (!target.classList.contains("token-span")) {
            return;
          }
          var raw = target.getAttribute("data-pos");
          if (raw === null) {
            return;
          }
          var pos = parseInt(raw, 10);
          setEntropyBarHighlight(pos);
          setTokenMetricsHover(pos, target);
          if (pos === altsPopoverPos) {
            return;
          }
          showAltsPopover(pos, target);
        }
      );
      overlayOutput.addEventListener(
        "mouseleave",
        function () {
          setEntropyBarHighlight(null);
          // Reaching into the popover keeps it open, so its
          // pagination arrows are clickable (mirrors the generator).
          // The strip holds its reading for the same reason: it
          // describes the position whose candidates are being read.
          if (altsPopover && altsPopover.matches(":hover")) {
            return;
          }
          clearTokenMetrics();
          hideAltsPopover();
        }
      );
    }
    if (altsPopover) {
      altsPopover.addEventListener("mouseleave", function () {
        clearTokenMetrics();
        hideAltsPopover();
      });
    }
    window.addEventListener(
      "scroll",
      function () {
        if (altsPopoverPos !== null) {
          hideAltsPopover();
        }
      },
      true
    );
  }

  // ---- Entropy by position ----

  // Tear the chart down and hide its section. Called before a new
  // run's frames are fetched, and on the paths where a run turns out
  // to carry no usable records at all.
  function clearEntropyChart() {
    chartEntropy = chartSupportDestroy(chartEntropy);
    // The chart owns one direction of the cross-highlight, so tearing
    // it down while a bar is hovered would otherwise strand the class
    // on whichever token was last lit.
    clearTokenHighlight();
    var section = document.getElementById("entropy-section");
    if (section) {
      section.hidden = true;
    }
    var notice = document.getElementById("entropy-unavailable");
    if (notice) {
      notice.hidden = true;
    }
  }

  // The run captured entropy, and declared it in a shape no view here
  // understands. Name the shape: the reader is either looking at a
  // run from a newer build, or at a channel whose declaration is
  // wrong, and either way the axes are the useful thing to show.
  function showEntropyUnavailable(data) {
    chartEntropy = chartSupportDestroy(chartEntropy);
    clearTokenHighlight();
    var section = document.getElementById("entropy-section");
    var notice = document.getElementById("entropy-unavailable");
    var wrap = section
      ? section.querySelector(".chart-wrap")
      : null;
    if (!section || !notice) {
      return;
    }
    section.hidden = false;
    if (wrap) {
      wrap.hidden = true;
    }
    var channel = overlaySeriesChannel(data, "entropy");
    notice.textContent =
      "This run records entropy over "
      + overlaySeriesChannelShape(channel).replace("|", " and ")
      + ", which this version has no chart for.";
    notice.hidden = false;
  }

  // Every position touched by a saved edit. For an autoregressive
  // What If branch that is the single substituted position; for a
  // diffusion run it is the remasked set, so the marker generalizes.
  function editedPositions(data) {
    var marks = editedPositionMarks(data);
    var positions = [];
    for (var key in marks) {
      if (positionWasEdited(marks, key)) {
        positions.push(Number(key));
      }
    }
    return positions;
  }

  // Whether a position was touched by an edit. A separate predicate
  // because frame 0 is a real answer and a falsy one, so asking the
  // map directly would silently drop an edit made at the very first
  // frame.
  function positionWasEdited(marks, position) {
    return typeof marks[position] === "number";
  }

  // One colour per entry of ``editedPositions(data)``, in the same
  // order. The denominator is the run's last frame index, asked of
  // the series so the append and per-frame shapes normalize alike.
  function editedPositionColors(data, positions) {
    var marks = editedPositionMarks(data);
    var series = overlaySeriesOf(data, false);
    var maxFrame = overlaySeriesLength(series) - 1;
    var colors = [];
    for (var i = 0; i < positions.length; i++) {
      colors.push(overlaysEditColor(marks[positions[i]], maxFrame));
    }
    return colors;
  }

  // Every touched position mapped to the frame its edit was made at,
  // as a lookup for the token layer, which asks about every position
  // it draws. The frame is the value rather than a bare true because
  // the markers colour themselves by it on the Commit Order ramp; a
  // position remasked twice keeps the later frame, which falls out of
  // the log being appended chronologically.
  //
  // Keyed on the run payload itself, which is replaced wholesale when
  // a run is selected and never mutated in place, so switching runs
  // rebuilds this and staying on one does not.
  var editedMarksCache = { data: null, marks: {} };

  function editedPositionMarks(data) {
    if (editedMarksCache.data === data) {
      return editedMarksCache.marks;
    }
    var edits = (data && data.remask_edits) || [];
    var marks = {};
    for (var i = 0; i < edits.length; i++) {
      var group = edits[i].token_positions || [];
      for (var j = 0; j < group.length; j++) {
        marks[group[j]] = edits[i].frame_index;
      }
    }
    editedMarksCache = { data: data, marks: marks };
    return marks;
  }

  // The position where the two runs part ways, or null when the run
  // was never edited. A What If branch copies the original trace's
  // prefix verbatim, so everything left of this index is identical in
  // both series and only the right side is worth comparing.
  //
  // This single-boundary reading is autoregressive-shaped. Diffusion
  // remasks are scattered rather than a prefix cut, so once those
  // runs carry entropy the comparison will want per-position
  // divergence instead of one index.
  function divergencePosition(data) {
    var positions = editedPositions(data);
    if (positions.length === 0) {
      return null;
    }
    var earliest = positions[0];
    for (var i = 1; i < positions.length; i++) {
      if (positions[i] < earliest) {
        earliest = positions[i];
      }
    }
    return earliest;
  }

  // One entropy layer. grouped:false is load-bearing: left grouped,
  // Chart.js sits the two runs side by side and halves every bar,
  // where the whole point is to superimpose them and crossfade.
  //
  // `texts` are the tokens the bars stand for, and `seriesKey` names
  // the overlayData series the layer was read from, "series" or
  // "baseline", so a scrub can read both again in place. `asOfStep`
  // is the earlier draft a commit's bars were borrowed from, or null.
  // The fills are left to entropyRecolor, which the scrub shares.
  function entropyDataset(label, series, seriesKey) {
    return {
      label: label,
      data: series.values,
      texts: series.texts,
      seriesKey: seriesKey,
      asOfStep: series.asOfStep,
      hoverBackgroundColor: entropyGlowColors(series.values),
      borderWidth: 0,
      barPercentage: 1,
      categoryPercentage: 1,
      grouped: false,
    };
  }

  // Each bar's hover color, on the overlay's glow ramp.
  function entropyGlowColors(values) {
    var colors = [];
    for (var i = 0; i < values.length; i++) {
      colors.push(entropyGlowColor(values[i]));
    }
    return colors;
  }

  // The chart's x labels: one per position, from 0.
  function entropyLabels(count) {
    var labels = [];
    for (var i = 0; i < count; i++) {
      labels.push(i);
    }
    return labels;
  }

  // Whether a bar past the scrubbed frame stands for a position that
  // does not exist yet. True only on an append stream, where frame k
  // is the frame that introduced position k, and never for a channel
  // the run declares to vary by frame: a diffusion canvas holds every
  // position at every frame. A run saved without a manifest is
  // decided by its stream alone, so an old autoregressive run still
  // fades its tail and an old diffusion run no longer does.
  function entropyDimsFuture(data) {
    var channel = overlaySeriesChannel(data, "entropy");
    if (overlaySeriesChannelShape(channel) === "frame|position") {
      return false;
    }
    return overlaySeriesOf(data, false).positions !== null;
  }

  // Every layer's fills for the scrubbed frame, at open and on scrub.
  function entropyRecolor(sets, dimsFuture) {
    for (var i = 0; i < sets.length; i++) {
      sets[i].backgroundColor = entropyFillColors(
        sets[i].data, dimsFuture
      );
    }
  }

  // Per-bar fills. When `dimsFuture` holds, bars past the scrubbed
  // frame fade, so the chart and the canvas above it agree about
  // which tokens exist at this frame.
  //
  // Baked into the color because Chart.js has no per-bar opacity. It
  // multiplies with the whole-dataset globalAlpha the crossfade sets
  // in compareBlendPlugin, which is the wanted composition: a dim bar
  // in the receding run is dimmer still.
  function entropyFillColors(values, dimsFuture) {
    var colors = [];
    for (var i = 0; i < values.length; i++) {
      if (dimsFuture && i > overlayFrameIndex) {
        colors.push(entropyDimColor(values[i]));
      } else {
        colors.push(entropyColor(values[i]));
      }
    }
    return colors;
  }

  // The pre-edit run's entropy, or null when there is nothing to
  // compare against. Needs three things at once: a divergence point,
  // a saved snapshot, and entropy inside it. The snapshot exists for
  // any edited run but predates the entropy signal on older ones, so
  // an older branch falls back to the single layer.
  function entropyOriginalSeries(data, divergence) {
    if (divergence === null) {
      return null;
    }
    var baseline = overlaySeriesOf(data, true);
    if (!overlaySeriesPresent(baseline)) {
      return null;
    }
    if (!overlaySeriesHasEntropy(baseline)) {
      return null;
    }
    var channel = overlaySeriesChannel(data, "entropy");
    return entropyLayerAt(baseline, channel, singleCanvas);
  }

  // One layer's bars at the frame its channel's axes name, read
  // through the frame whose entropy describes it, so a commit shows
  // its canvas's last draft. `asOfStep` is that draft, or null when
  // the layer read the frame itself.
  function entropyLayerAt(series, channel, canvasOf) {
    var at = overlaySeriesChannelFrame(
      channel, series, overlayFrameIndex
    );
    var source = overlaySeriesEntropyFrame(series, at, canvasOf);
    var layer = overlaySeriesEntropyValues(
      series, source < 0 ? at : source
    );
    layer.asOfStep = source >= 0 && source !== at ? source : null;
    return layer;
  }

  // Which canvases a layer's frames sit on. The pre-edit run of an
  // edited one is single-canvas, since only those can be edited.
  function entropyLayerCanvasOf(seriesKey) {
    if (seriesKey === "baseline") {
      return singleCanvas;
    }
    return overlayCanvasOf;
  }

  // One bar per generated position, tall and hot where the model was
  // torn. Unlike the three charts above it is indexed by position
  // rather than frame, which is also why it is drawn as bars: an
  // autoregressive position is an independent decision, not a point
  // in a time series. Hidden for runs saved without the entropy
  // signal.
  function renderEntropyChart(data) {
    var section = document.getElementById("entropy-section");
    if (!chartSupportAvailable) {
      clearEntropyChart();
      return;
    }
    var availability = overlaySeriesEntropyAvailability(data);
    if (availability === "absent") {
      clearEntropyChart();
      return;
    }
    if (availability === "unsupported") {
      // Said rather than hidden. The run captured entropy in a shape
      // this build cannot draw, and an empty space is how a channel
      // dropped by accident would also look.
      showEntropyUnavailable(data);
      return;
    }
    if (section) {
      section.hidden = false;
      // Undo whatever the unsupported branch may have hidden, so
      // switching between two runs does not leave the canvas dark.
      var wrap = section.querySelector(".chart-wrap");
      if (wrap) {
        wrap.hidden = false;
      }
      var notice = document.getElementById("entropy-unavailable");
      if (notice) {
        notice.hidden = true;
      }
    }

    var divergence = divergencePosition(data);
    var series = overlaySeriesOf(data, false);
    var channel = overlaySeriesChannel(data, "entropy");
    var edited = entropyLayerAt(series, channel, overlayCanvasOf);
    var original = entropyOriginalSeries(data, divergence);

    // Labels span the longer run: a branch can outlive or fall short
    // of the run it forked from.
    var count = edited.values.length;
    if (original && original.values.length > count) {
      count = original.values.length;
    }

    // Original first, so it draws beneath the branch it produced and
    // so dataset index 0 is the one the crossfade fades out.
    var datasets = [];
    if (original) {
      datasets.push(entropyDataset("Original", original, "baseline"));
    }
    datasets.push(entropyDataset("Edited", edited, "series"));
    entropyRecolor(datasets, entropyDimsFuture(data));

    var markerPositions = editedPositions(data);

    var canvas = document.getElementById("chart-entropy");
    chartEntropy = new Chart(
      canvas.getContext("2d"),
      {
        type: "bar",
        data: {
          labels: entropyLabels(count),
          datasets: datasets,
        },
        options: entropyChartOptions(original ? divergence : null),
        // Deliberately without the line charts' burn-through plugin:
        // it redraws a dataset's *line* through the tooltip box,
        // which a bar chart has none of, and would stroke a stray
        // polyline across the bar tops instead. The eye toggle covers
        // hiding the box.
        //
        // Marker before hover so the pointer's white guide lays over
        // the edit tint rather than under it.
        plugins: [
          substitutionMarkerPlugin(
            markerPositions,
            editedPositionColors(data, markerPositions)
          ),
          entropyHoverPlugin,
          compareBlendPlugin,
          tokenLinkPlugin,
        ],
      }
    );
  }

  // Back to the edited run at full opacity, so each run opens on the
  // branch it is a record of rather than on the previous run's mix.
  // The control shows for any edited run saved with its snapshot,
  // which is a wider gate than the entropy chart's: the token layers
  // only need the snapshot, while a second bar series also needs it
  // to carry per-token entropy.
  function resetRunBlend(visible) {
    compareBlend = 1;
    // A run can be opened while a previous drag is still easing out.
    onBlendReset();
    if (runBlendInput) {
      runBlendInput.value = "100";
    }
    if (runBlendRow) {
      runBlendRow.hidden = !visible;
    }
  }

  // Only layer alpha changes, so nothing is reparsed; "none" skips
  // the animation that would otherwise lag the drag. The line charts
  // are only touched once a press has become a drag, and are left
  // entirely alone for a keyboard adjustment.
  function onRunBlendInput() {
    compareBlend = Number(runBlendInput.value) / 100;
    if (chartEntropy) {
      chartEntropy.update("none");
    }
    applyTokenLayerBlend();
    refreshTokenMetricsLayer();
    onBlendInput();
  }

  // Restyle the stacked token layers in place. Rebuilding them would
  // mean several hundred spans per slider step, and would also drop
  // the popover mid-drag. Diff mode is left alone: its two sliders
  // own the layers there.
  function applyTokenLayerBlend() {
    if (overlayMode === "diff") {
      return;
    }
    var original =
      overlayOutput.querySelector(".token-layer-original");
    var edited =
      overlayOutput.querySelector(".token-layer-edited");
    if (!original || !edited) {
      return;
    }
    original.style.opacity = String(1 - compareBlend);
    edited.style.opacity = String(compareBlend);
    overlaysApplyLayerPointers(
      overlayOutput, 1 - compareBlend, compareBlend
    );
  }

  // ``texts`` holds one token-text array per dataset, so the tooltip
  // can name the token each layer chose. ``divergence`` is null on a
  // run with nothing to compare against, which collapses the tooltip
  // back to the single unlabeled row.
  function entropyChartOptions(divergence) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      layout: chartSupportGutterLayout(),
      interaction: {
        mode: "index",
        intersect: false,
      },
      // The bar-to-token half of the cross-highlight lives in
      // tokenLinkPlugin rather than onHover; see its comment.
      plugins: {
        legend: { display: false },
        tooltip: {
          position: "smart",
          caretSize: 0,
          xAlign: "left",
          yAlign: "top",
          filter: function (item) {
            return entropyTooltipFilter(item, divergence);
          },
          callbacks: {
            title: positionTooltipTitle,
            label: function (ctx) {
              return entropyTooltipLabel(ctx, divergence);
            },
          },
        },
        zoom: chartSupportZoomOptions(),
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Position",
          },
          ticks: { maxTicksLimit: 12 },
          grid: { display: false },
        },
        y: {
          title: {
            display: true,
            text: "Entropy (nats)",
          },
          beginAtZero: true,
          // Suggested, not fixed: keeps the scale comparable across
          // runs at the overlay's reference maximum while still
          // letting an unusually torn position through instead of
          // clipping it.
          suggestedMax: OVERLAYS_ENTROPY_REF_NATS,
        },
      },
    };
  }

  // The shared chartSupportTooltipTitle prefixes "Frame", which would
  // misread this chart's x axis.
  function positionTooltipTitle(items) {
    if (items.length === 0) {
      return "";
    }
    return "Position " + items[0].label;
  }

  // Which rows the hovered position is worth showing. A null value is
  // the tail of a run that stopped short of its counterpart. The
  // original layer is dropped left of the divergence point because
  // the branch copies its prefix verbatim there, so a second row
  // would only ever restate the first.
  function entropyTooltipFilter(item, divergence) {
    if (item.parsed.y === null) {
      return false;
    }
    if (divergence === null) {
      return true;
    }
    if (item.datasetIndex > 0) {
      return true;
    }
    return item.dataIndex >= divergence;
  }

  // Naming the token is the thing the generator's compact profile
  // cannot do, so the tooltip carries it alongside the value.
  //
  // From the divergence point on, each row is named for its run. Note
  // that at the marked position itself the two rows carry the same
  // nats and different tokens, which is the intervention in one line:
  // entropy describes the distribution the prefix produced, and
  // forcing a token changes which one was drawn, not the distribution
  // it was drawn from.
  function entropyTooltipLabel(ctx, divergence) {
    var value = ctx.formattedValue + " nats";
    var series = ctx.dataset.texts || [];
    var text = series[ctx.dataIndex];
    var row = text ? value + "  \u2022  " + text : value;
    if (typeof ctx.dataset.asOfStep === "number") {
      row += ", " + overlaysEntropyAsOf(ctx.dataset.asOfStep);
    }
    if (divergence === null || ctx.dataIndex < divergence) {
      return row;
    }
    return ctx.dataset.label + ": " + row;
  }

  // ---- What the page calls ----

  // Clear the viewer for a run whose frames are on their way: tokens,
  // popover, entropy chart and readout, and whether the run is
  // autoregressive, which the catalog knows and its frames do not.
  function beginRunOverlays(autoregressive) {
    clearOverlay();
    hideAltsPopover();
    clearEntropyChart();
    // clearOverlay forgot the run, so this hides the readout until
    // the new run's frames say whether it has one.
    refreshStopReadout();
    overlayIsAutoregressive = autoregressive;
  }

  // Once, at boot: the drawer, the diff controls, the scrubber, the
  // highlight checkbox, the crossfade's slider, and the strip and
  // readout the viewer writes into.
  function wireTokenViewer() {
    // The shared helper owns the handle click as well as the drag, so
    // this binds none of its own (see overlaysMakeDrawerDraggable).
    overlaysMakeDrawerDraggable({
      group: overlaySelectGroup,
      handle: overlayDrawerHandle,
      container: document.getElementById("overlay-output-wrap"),
      storageKey: "diffusion_overlay_drawer_top_analytics",
      onToggle: setOverlayDrawerOpen,
    });
    wireOverlayDiffControls();
    wireOverlayScrubber();
    if (overlayHighlightCheckbox) {
      overlayHighlightCheckbox.addEventListener(
        "change", onOverlayHighlightToggle
      );
    }
    if (runBlendInput) {
      runBlendInput.addEventListener("input", onRunBlendInput);
      runBlendInput.addEventListener("pointerdown", onBlendPress);
      // On window rather than the slider: a drag frequently releases
      // with the pointer well outside the track.
      window.addEventListener("pointerup", onBlendRelease);
      window.addEventListener("pointercancel", onBlendRelease);
    }
    overlaysBuildTokenMetrics(tokenMetricsStrip);
    overlaysBuildStopReadout(stopReadout);
  }

  // The crossfade's position: 0 for the run an edit branched from, 1
  // for the branch. The line charts read it while a drag borrows
  // them.
  function currentBlend() {
    return compareBlend;
  }

  // The entropy chart, or null while none is drawn.
  function currentEntropyChart() {
    return chartEntropy;
  }

  return Object.freeze({
    beginRun: beginRunOverlays,
    show: renderRunOverlays,
    showUnavailable: showOverlayUnavailable,
    clear: clearOverlay,
    setFrame: setOverlayFrame,
    setMode: setOverlayMode,
    blend: currentBlend,
    entropyChart: currentEntropyChart,
    refreshHoverHighlight: updateOverlayHoverHighlight,
    wire: wireTokenViewer,
  });
}
