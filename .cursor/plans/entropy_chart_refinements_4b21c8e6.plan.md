---
name: Entropy chart refinements
overview: Add a hover highlight, an original-vs-edited crossfade slider, a divergence-aware two-row tooltip, and edit-orange marker styling to the Analytics Entropy by Position chart, plus a collision-aware tooltip positioner shared by all four charts.
todos:
  - id: hover-highlight
    content: Add entropyHoverPlugin (full-height guide column at the active index, min 2px, drawn in beforeDatasetsDraw) and a glowColors array passed as hoverBackgroundColor in renderEntropyChart.
    status: completed
  - id: marker-orange
    content: "Recolor substitutionMarkerPlugin to the #ff9f1c edit color and add a rgba(255,159,28,0.15) full-height column tint in beforeDatasetsDraw, hoisting both to module constants."
    status: completed
  - id: original-series
    content: Refactor entropyChartValues into entropySeriesFrom(frames), extract framesHaveEntropy from overlayEntropyAvailable, and add divergencePosition(data).
    status: completed
  - id: crossfade
    content: "Add the second Original dataset with grouped:false, the entropyAlphaPlugin driven by globalAlpha, and the #entropy-blend slider markup, CSS, wiring, and reset."
    status: completed
  - id: tooltip-rows
    content: "Make the entropy tooltip divergence-aware: filter null and pre-divergence Original rows, label Original/Edited at and after the marker."
    status: completed
  - id: smart-positioner
    content: Rewrite the smart positioner to score four corners against the cursor and the drawn data (bar rects, line segments) with TL/TR/BL/BR tie-break and hysteresis.
    status: completed
  - id: docs
    content: Update the Help modal entropy entry, README, ROADMAP, and HANDOFF.
    status: completed
  - id: verify
    content: Run node --check, ReadLints, the 70-column audit, and pytest, then hand back the manual verification checklist.
    status: completed
isProject: false
---

# Entropy chart refinements and smart tooltip placement

All changes are frontend. Python is untouched.

Files: [analytics.js](src/web/static/analytics.js), [analytics.html](src/web/static/analytics.html), [analytics.css](src/web/static/analytics.css), plus docs in [index.html](src/web/static/index.html), [README.md](README.md), [ROADMAP.md](ROADMAP.md), [HANDOFF.md](HANDOFF.md).

`overlays.js` needs no change: `entropyGlowColor` already exists, and layer opacity is done with canvas `globalAlpha` rather than by rewriting color strings.

## 1. Hover highlight

Two pieces in `renderEntropyChart`:

- Build a `glowColors` array alongside the existing `colors` loop and pass it as `hoverBackgroundColor`. Chart.js repaints the active bar natively, which correctly covers both datasets and respects the crossfade alpha below.
- A new `entropyHoverPlugin` drawing the faint full-height guide in `beforeDatasetsDraw` so bars sit on top. Read `chart.getActiveElements()[0].index`, take `x` and `width` from that bar element (`getProps(["x", "width"], true)`, which stays correct under zoom), and fill `rgba(255, 255, 255, 0.1)` from `yScale.top` to `yScale.bottom` at `Math.max(2, width)`.

The `Math.max(2, ...)` matters here: a 256-position run puts roughly 1px per bar in a 200px-tall chart, so the guide column is what makes the hover findable. This mirrors `drawEntropyProfileGlow` in [app.js](src/web/static/app.js):

```1969:1972:src/web/static/app.js
  ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
  ctx.fillRect(
    left, 0, Math.max(2, layout.barWidth), layout.cssHeight
  );
```

Deliberate deviation from the generator: no manual bright-bar-plus-shadow redraw. The generator can afford it with one dataset and no blending; here a hand-drawn bar would ignore the crossfade alpha and paint a fully opaque bar over a faded layer. Native `hoverBackgroundColor` avoids that.

## 2. Edit-orange marker and column tint

`substitutionMarkerPlugin` gains a `beforeDatasetsDraw` hook for the tint and keeps `afterDatasetsDraw` for the line. Register it before `entropyHoverPlugin` so a hover on the marked column layers the white guide over the orange tint.

- Dashed line: `rgba(0,255,65,0.55)` becomes `#ff9f1c`, the `.token-remasked` color from [style.css](src/web/static/style.css).
- Column tint: `rgba(255, 159, 28, 0.15)`, the exact `.token-remasked` background, full height at `Math.max(2, barWidth)`.

Hoist both to module constants near `TIMING_COLOR`.

## 3. Original series and divergence point

Refactor the data helpers so both runs go through one path:

- Generalize `entropyChartValues(data)` into `entropySeriesFrom(frames)` returning `{ values, texts }`, called for `data.frames` and for `data.original_frames`.
- Extract `framesHaveEntropy(frames)` from the body of `overlayEntropyAvailable`, and have `overlayEntropyAvailable` call it. Use it to gate the original series: `original_frames` exists for any edited run, but `e` only for runs saved after this session, so a pre-Phase-C edited run must fall back to a single dataset with no slider.
- `divergencePosition(data)`: `Math.min` over `editedPositions(data)`, or `null` when unedited.

## 4. Crossfade slider

Markup in `#entropy-section` below `.chart-wrap`, hidden by default:

```html
<div id="entropy-blend-row" hidden>
  <span class="diff-slider-label diff-orig-label">Original</span>
  <input type="range" id="entropy-blend" min="0" max="100" value="100" />
  <span class="diff-slider-label diff-edit-label">Edited</span>
</div>
```

Reuses `.diff-slider-label` / `.diff-orig-label` / `.diff-edit-label` from style.css; `analytics.css` gets a small flex rule for the row and an accent-color for the input, next to the existing `#overlay-diff-controls` block.

Chart side:

- Dataset 0 is Original (drawn underneath), dataset 1 is Edited. **Both need `grouped: false`**, otherwise Chart.js side-by-sides them and halves every bar's width.
- `labels` runs to `Math.max(edited.length, original.length)`, since a branch can be shorter or longer than its parent.
- `entropyAlphaPlugin` sets `ctx.save(); ctx.globalAlpha = ...` in `beforeDatasetDraw(chart, args)` keyed on `args.index`, and restores in `afterDatasetDraw`. Alphas are `edited = t`, `original = 1 - t` for slider fraction `t`.
- The `input` handler stores `t` in a module var and calls `chartEntropy.render()`. No data mutation means no `update()`, so this stays cheap on a 512-bar chart.
- `clearEntropyChart` hides the row and resets `t` to 1.

Expect the visible effect to be confined to the right of the dashed line: the branch copies the original trace's prefix verbatim, so the two series are identical left of the marker.

## 5. Divergence-aware tooltip

In `entropyChartOptions`, the tooltip takes the divergence position and both text arrays:

- `filter`: drop any item whose `parsed.y` is null (positions past a shorter run's end), and drop the Original dataset's item when `dataIndex < divergence`. Left of the marker the two runs agree, so one unlabeled row is the honest rendering.
- `label`: below divergence, unchanged (`"1.678 nats  •  ·Get"`). At and above it, prefix `"Original "` / `"Edited "` and pick the text array by `ctx.datasetIndex`.

The two rows at the marker itself will show the same nats with different tokens. That is correct and is the point of the row: entropy is a property of the distribution given the prefix, and the substitution changes only which token was drawn from it. See the `_substitute_loop` docstring in [ar_sampler.py](src/inference/ar_sampler.py) lines 525-532. The labels are what keep it from reading as a duplicate.

Note for later: the "one boundary, shared prefix" rule is autoregressive-shaped. When diffusion entropy lands, scattered remasks will likely want per-position divergence instead. Leaving a comment, not building for it.

## 6. Collision-aware `smart` positioner

Rewrite `Chart.Tooltip.positioners.smart` in [analytics.js](src/web/static/analytics.js) lines 150-178. Today it only knows the hovered point:

```165:170:src/web/static/analytics.js
    var x = (el.x > midX)
      ? area.left + pad
      : area.right - pad - w;
    var y = (el.y > midY)
      ? area.top + pad
      : area.bottom - pad - h;
```

On a monotonically rising timing line, "diagonally opposite the point" is precisely where the line is going, which is the screenshot-2 behavior.

New algorithm, using `this.width` / `this.height` from the previous frame as today:

1. Build four candidate rects from `chartArea` at pad 10.
2. Reject any containing `eventPosition` inflated by a margin.
3. Reject any colliding with drawn data. For bar elements test the full bar rect via `getProps(["x", "y", "base", "width"])`; testing only the bar top would report the bottom corners as free when they are solid bars. For line elements test each consecutive pair as a segment against the rect (bounding-box reject, then a slab clip), not just vertices: a 20-frame DiffusionGemma canvas has segments long enough to cross a corner box with no point inside it. Early-exit on first hit.
4. Take the first survivor in the order top-left, top-right, bottom-left, bottom-right.
5. Hysteresis: stash the chosen corner on the chart instance and keep it while it stays collision-free, so the box does not hop between two equally valid corners as the cursor moves.
6. No survivor: fall back to the first cursor-avoiding corner and let `burnThroughPlugin` keep the line legible, which preserves today's behavior as the designed fallback.

Shared by all four charts. On the entropy chart the bars fill the bottom, so it will settle in a top corner.

## 7. Docs and verification

Help modal entropy entry in [index.html](src/web/static/index.html) gains the hover highlight, the orange marker, and the crossfade slider; README Analytics bullet, ROADMAP shipped list, and HANDOFF recently-shipped plus manual checklist follow.

Verify with `node --check src/web/static/analytics.js`, ReadLints on changed files, a 70-column audit of new lines, and `.venv/bin/python -m pytest` as an unchanged-baseline sanity run. Hover, crossfade, and tooltip placement all need a browser and a GPU-saved run, so the handback carries a manual checklist covering: an unedited AR run (no slider, no marker, hover works), an edited AR run saved this session (slider crossfades, tooltip splits at the marker, orange tint), a pre-Phase-C edited run (slider absent, single dataset), and the timing tooltip settling top-left.