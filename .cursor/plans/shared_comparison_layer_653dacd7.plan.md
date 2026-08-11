---
name: shared comparison layer
overview: Make the pre-edit run a first-class token layer in Analytics, driven by one crossfade promoted to the detail modal header, and add bidirectional entropy cross-highlighting plus Original/Edited popover pagination on both pages.
todos:
  - id: span-refactor
    content: Give layered token spans token-span, data-pos, and titles in overlaysBuildDiffLayerSpans; rename .diff-layer to .token-layer across overlays.js, style.css, and analytics.css; set pointer-events so the more opaque layer owns the pointer.
    status: completed
  - id: crossfade-promote
    content: "Move #entropy-blend-row into a new #detail-header row beside #detail-title in analytics.html, rename it and entropyBlend to run-level names, widen its visibility gate from the entropy series to overlayDiffAvailable, and keep the per-run reset."
    status: completed
  - id: layered-overlays
    content: Change renderOverlayTokens' colorFn signature to receive the token, memoize a second commit-steps array for the original run, and emit two crossfaded layers in None, Heatmap, Entropy, and Commit Order when a snapshot exists.
    status: completed
  - id: cross-highlight-analytics
    content: Add setTokenHighlight(pos) in analytics.js, drive chartEntropy.setActiveElements from token hover, and add onHover on the entropy chart to light the token from the bar.
    status: completed
  - id: cross-highlight-generator
    content: "Add mousemove and mouseleave on #entropy-profile inverting drawEntropyProfile's step math into a position, route it through setEntropyHoverPosition, and add setTokenHighlight(pos) on the generator."
    status: completed
  - id: popover-pagination
    content: Port the generator's keep-open guard to the analytics popover, turn .alt-heading into a flex row with Original/Edited arrows for positions at or past divergence, page between the two candidate sets with the crossfade choosing the opening page, and mark the chosen token per page.
    status: completed
  - id: docs-verify
    content: Update README, ROADMAP, HANDOFF, and the About/Help modals; run node --check, ReadLints, pytest as a regression check, and the 70-column audit; write the manual-verification checklist including the degrade paths for older runs.
    status: completed
isProject: false
---

# Shared comparison layer

The pre-edit run already exists as a layer in two places: the diff overlay's token stack and the entropy chart's crossfade. This pass makes it a layer *everywhere* in Analytics under one shared control, and closes the entropy cross-highlighting gap on both pages.

## Settled

- Analytics gets the full treatment. The generator gets bidirectional cross-highlighting and popover pagination only; its crossfade and two-layer stack are deferred, and when scheduled the crossfade goes in its own row below the frame scrubber.
- The crossfade lives in the detail modal header and governs the token view and the entropy chart from one place. The entropy chart's own slider moves rather than duplicating.
- Diff mode keeps its two independent opacity sliders, because Difference blend needs both layers up at once. The header crossfade stays visible and keeps driving the entropy chart; diff mode simply overrides it for the token view.
- Interaction handoff generalizes the agreed midpoint: **the more opaque layer owns the pointer**. For a crossfade that is exactly the 50 rule; for diff's two sliders it still has a defined answer.
- Cross-highlighting is not gated on the `highlightTokens` comfort setting.

## 1. Fix the layered spans (the blocker)

`overlaysBuildDiffLayerSpans` in [src/web/static/overlays.js](src/web/static/overlays.js) builds bare spans:

```javascript
var span = document.createElement("span");
if (tok.m) {
  span.textContent = mask;
  span.style.color = "var(--mask-color)";
} else {
  span.textContent = tok.t;
  span.style.color = overlaysDiffLayerColor(diff, i, isOriginal, blend);
}
```

No class, no `data-pos`. Both pages' hover paths require both (`hoveredTokenPosition` at [app.js:4520](src/web/static/app.js), the `mouseover` guard at [analytics.js:1865](src/web/static/analytics.js)), which is the single reason hover, the popover, and entropy highlighting are dead in diff mode. Give these spans `token-span` plus `token-mask`/`token-resolved`, `data-pos`, and the same `title` the normal path builds.

Rename `.diff-layer` to `.token-layer` (with `-original`/`-edited` variants) across [overlays.js](src/web/static/overlays.js), [style.css:2353](src/web/static/style.css), and [analytics.css:791](src/web/static/analytics.css). Layering is no longer diff-specific and the old noun would mislead. The grid stacking (`grid-area: 1 / 1`) is unchanged.

Set `pointer-events` inline next to opacity: the more opaque layer gets `auto`, the other `none`. Today the edited layer wins hit-testing purely by being the later sibling, with no pass-through.

## 2. Promote the crossfade to the modal header

`#entropy-blend-row` currently sits inside `#entropy-section` in [analytics.html](src/web/static/analytics.html). Move it into a new `#detail-header` row wrapping `#detail-title` (line 136). [analytics.css:221](src/web/static/analytics.css) already carries an unused `#detail-header` rule, so the styling hook exists.

Rename the control and its state to something run-level rather than entropy-level (`#run-blend-row` / `#run-blend`, `entropyBlend` to `compareBlend`), and widen its gate: today `setEntropyBlendVisible(original !== null)` keys on the entropy series, but it must now appear whenever the run is edited and has a snapshot, matching `overlayDiffAvailable` at [analytics.js:1242](src/web/static/analytics.js). A snapshot missing `e` still gets the token crossfade; the chart just stays single-dataset.

Keep the existing reset in `clearEntropyChart` ([analytics.js:2270](src/web/static/analytics.js)) so switching runs reopens on Edited.

## 3. Generalize the two-layer stack to every overlay mode

`renderOverlayTokens(frame, colorFn, titleFn)` at [analytics.js:1677](src/web/static/analytics.js) is the single builder behind None, Heatmap, Entropy, and Commit Order. Each mode's `colorFn` closes over the edited `frame` and indexes it by position:

```javascript
function (i) {
  var tok = frame ? frame[i] : null;
  if (tok && typeof tok.e === "number") { return entropyColor(tok.e); }
  return null;
}
```

Change the callback signature to receive the token (`colorFn(i, tok)`) so the same function colors either layer by *its own* values. That is what makes the comparison honest: the original layer shows the original run's entropy, not the branch's.

Commit Order needs a second memoized steps array from `overlaysComputeCommitSteps(overlayData.original_frames)`, since its colors come from the frame stream rather than per-token fields.

When the crossfade is active and a snapshot exists, emit two stacked layers through the shared builder from step 1; otherwise emit one, exactly as today.

## 4. Cross-highlighting, both directions, both pages

Neither page can highlight a token by position today; highlighting is pure CSS `:hover` ([style.css:2233](src/web/static/style.css)). Add a small `setTokenHighlight(pos)` on each page that toggles a `token-cross-highlight` class on `[data-pos="N"]` (plural, since there may be two layers) and tracks the lit position so clearing is cheap.

**Analytics.** Token hover already resolves a position at [analytics.js:1865](src/web/static/analytics.js); from there call `chartEntropy.setActiveElements([{ datasetIndex, index: pos }])` plus `update("none")`. `setActiveElements` is currently used nowhere in the repo. For the reverse, add `onHover` to the entropy chart options; `entropyHoverPlugin` at [analytics.js:1112](src/web/static/analytics.js) already reads `getActiveElements()` for its white guide, so the chart side needs no new hit-testing.

**Generator.** `#entropy-profile` has no mouse handlers at all. Add `mousemove` and `mouseleave`, inverting the layout math in `drawEntropyProfile` ([app.js:1929](src/web/static/app.js)):

```javascript
var step = cssWidth / values.length;          // as drawn
var index = Math.floor(x / step);             // as read back
```

Feed that into the existing `setEntropyHoverPosition` ([app.js:2032](src/web/static/app.js)), which already owns the column glow and the nats readout, and add the token highlight alongside it.

## 5. Popover pagination

Arrows appear only for positions at or past `divergencePosition` ([analytics.js:2338](src/web/static/analytics.js)) where both candidate sets exist. `original_alternatives` already reaches the client in `overlayData` and is unused; the generator has `originalPositionAlts` in memory.

`.alt-heading` ([style.css:2422](src/web/static/style.css)) is a plain text div on both pages, so it becomes a flex row with the title left and the arrows right. The crossfade picks the opening page (below 50 opens Original), but both pages stay reachable, so dragging past the midpoint never costs access to a candidate set.

Mark the chosen token per page: the Original page must read the original run's id at that position, not the branch's.

**Prerequisite:** Analytics hides the popover immediately on `mouseleave` ([analytics.js:1883](src/web/static/analytics.js)) with no keep-open guard, so the arrows would be unclickable. Port the generator's `altsPopover.matches(":hover")` check from [app.js:4534](src/web/static/app.js) along with the popover's own `mouseleave`.

## What degrades

Runs saved before the previous pass have no `original_alternatives`, `original_per_frame_elapsed`, or `original_mean_conf`. Every surface here needs its fallback exercised: no snapshot means no crossfade and a single layer, a snapshot without `e` means crossfade but no second chart dataset, and no original candidates means no arrows with the popover otherwise unchanged.

## Verification

No Python changes, so `pytest` is a regression check only. Run `node --check` on each changed JS file, ReadLints on everything touched, and the 70-column audit. Update README, ROADMAP, HANDOFF, and the in-app About/Help copy, then hand back a manual checklist since none of this can be exercised without a display.