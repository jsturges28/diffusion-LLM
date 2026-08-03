# HANDOFF: next session

Living, per-session handoff. The agent updates this at the end of each session so
the next one can pick up cold (see `AGENTS.md`). Read `README.md` + `ROADMAP.md`
first, then deliberate the work below with the maintainer before Plan mode
(deliberate in Ask mode → Plan → Agent).

## What it is

A local FastAPI + WebSocket visual playground and analytics suite for discrete
diffusion LLMs, oriented toward explainability (xAI). Runs in the browser
(localhost) and as an optional native desktop app. Watch models denoise live:
scrub frame history, remask tokens and resume, color tokens by confidence,
entropy, or commit order, inspect the candidates a model nearly chose, diff an
edited run (or an autoregressive What If branch) against the original, and
compare runs in an analytics suite.

## Models (one resident at a time)

- **LLaDA-8B-Instruct**: masked discrete diffusion, bf16 (~17GB). Interactive
  remask/resume + guided multi-frame editing.
- **DiffusionGemma-26B-A4B**: block-autoregressive encoder-decoder MoE,
  self-quantized 4-bit NF4 (~18GB), 256-token canvases, adaptive stopping,
  optional "thinking" channel. Single-canvas remask/resume works; **multi-canvas
  resume is NOT done** (Edit Frames disabled for multi-canvas runs, deferred).
- **SmolLM3-3B**: autoregressive baseline, decoder-only, bf16 (~6GB), in
  `.venv-ar`. Streams token-by-token (one full-snapshot frame per token) with
  per-token sampling confidence and always-on entropy; optional thinking channel
  and an opt-in top-5 **Alternatives** capture. Runs on GPU or CPU
  (per-activation toggle on the menu), so it is the model a GPU-less host can
  run. No diffusion remask/resume; its counterfactual is **What If?**
  substitution instead (`supports_substitution`).

## Architecture (process isolation; incompatible transformers versions)

- **Supervisor**: `src/web/server.py` (runs in `.venv`). Serves the **Main Menu**
  at `/` and the generator at `/generate` (gated: redirects to `/` when no model
  is active; `/index.html` 307s to `/generate`). Model Manager spawns ONE worker
  at a time with a pre-flight VRAM check; proxies `/ws` (no auto-boot: it errors
  and closes if no worker is active); serves analytics + save + run-delete;
  auto-stamps HTML asset URLs. `/api/models` also returns `gpu_name` +
  `free_vram_gib` + per-model `fits` for the menu. Durable UI state
  (`src/web/ui_state.py`, stored in `results/ui_state.json`) is served via
  `GET`/`PUT /api/ui-state`; the GET reconciles the "new run" cue against
  existing run folders so deleted runs cannot inflate the count.
- **Workers**: `src/backends/{llada_worker,dgemma_worker,smollm3_worker}.py` via
  `run_worker.py`; contract in `protocol.py` / `registry.py` / `worker_base.py`.
  LLaDA → `.venv` (transformers 4.38.2); DiffusionGemma → `.venv-dgemma`
  (transformers 5.13); SmolLM3 → `.venv-ar` (transformers 4.53). `run_worker.py`
  takes `--device`, forwarded via `create_worker_app(device=...)` into
  `Backend.load(device=...)` (kw-only, default "cuda").
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler,ar_sampler}.py`;
  NF4 in `dgemma_nf4.py`. Analytics metrics: `src/analytics/metrics.py`.
- **Frontend** (shared, schema-driven): `src/web/static/{menu.html, menu.js,
  index.html, app.js, overlays.js, analytics.html, analytics.js, analytics.css,
  style.css, custom_select.js}` + `assets/title-screen.{webm,mp4}`. `overlays.js`
  holds the shared color ramps (`heatColor`, `commitColor`, `diffColor`,
  `entropyColor`), the layered-diff builder (`overlaysBuildDiffLayers`), the "new
  run" registry, and the durable-UI-state layer (`persistHydrate` on boot +
  `persistSet` write-through to `/api/ui-state`); the old `ascii_scene.js` idle
  animation was removed.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle: uvicorn on a
  stable localhost port `DESKTOP_PORT=8760` with an ephemeral fallback, on a
  daemon thread, graceful shutdown frees worker VRAM on close; runs a persistent
  (non-private) web-storage profile; prefers Qt/QtWebEngine via `_select_gui`,
  falls back to GTK; `_set_app_identity` sets GTK prgname + Qt desktopFileName for
  dock integration). Note: durable UI state now lives server-side, so it no longer
  depends on the window origin/port.
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop`; `assets/icon.svg`
  is the app icon.

## Recently shipped (this session)

**Latest pass: the line-chart comparison layer.** Frontend only, so `pytest`
(79/79) is a regression check; `node --check`, ReadLints, and the 70-column
audit are clean. Nothing needing a display was exercised; checklist items 21 to
27 at the bottom cover them.

The finding that set the scope: **the timing and confidence charts had no
original series at all.** `addOriginalRunSignals` (`app.js`) has been saving
`original_per_frame_elapsed` and `original_mean_conf` and the metrics route has
been serving them, but neither string appeared anywhere in `analytics.js`, so
both charts were still single-dataset and an edited run could only ever show
its branch. Building that second series was the bulk of the work; the controls
sit on top of it.

- **Both runs, distinguished by dash rather than by hue.** Original solid in a
  neutral grey (`COMPARE_ORIGINAL_COLOR`), branch dashed
  (`COMPARE_EDITED_DASH`) in the chart's existing color, original at dataset
  index 0 to match the entropy chart. Grey and not a second hue because timing
  already spends blue on the branch, lighter blue on its resumed stretch, and
  amber on canvas boundaries; a fourth color would read as a fourth category
  rather than as the baseline. The two share a prefix and separate at the edit,
  so the dashes leaving the solid line *is* the reading.
  - `fill` drops to false whenever there are two series (the `paired` flag on
    `timingEditedDataset` / `confidenceEditedDataset`). Two translucent fills
    stacked over the shared prefix blend into a third color and stop reading as
    two runs at all. A single-series run keeps its filled area and looks
    exactly as it did.
  - The branch keeps its `segment` callbacks; the original gets none, because
    no remask or resume happened in it. The "Resume point" `afterLabel` is
    gated on `isEditedDataset` for the same reason, or it prints under both
    rows.
  - Labels span the longer run (`compareFrameLabels`): a branch can outlive or
    fall short of the run it forked from.
- **Pins are a three-state control, not two checkboxes.** `linePinState` per
  chart, both on at open, rendered by `refreshComparePins`. Turning off the
  only lit pin is refused and the button carries `is-locked` so the dead click
  reads as unavailable beforehand. Every state still reaches every other in at
  most two clicks. The alternative (clicking the sole lit pin swaps to the
  other) was rejected: pressing 1 and watching 2 light up is a surprise, and it
  buys nothing over two clicks.
  - `resetComparePins` hides both groups as well as resetting them, because a
    chart that bails early for want of data never reaches its
    `updateComparePins` call and would otherwise leave the previous run's pins
    standing.
- **The crossfade borrows the line charts; it does not own them.** This was the
  design argument of the session. The slider governs the token view and the
  entropy bars, where "both at full opacity" is not expressible because marks
  occlude. On strokes it is, so the pins own the resting state and the slider
  only takes over for the length of a pointer drag: `scrubWeight` eases to 1
  and back over `SCRUB_EASE_MS`, with `seriesBlendAlpha` lerping between the
  pin answer and the blend answer at draw time. The
  whole modal moves together during the drag, which was the point, without
  permanently tying two frame-indexed charts to a slider four hundred pixels
  away in another column.
  - Alpha at draw time (`seriesBlendPlugin`), mirroring `compareBlendPlugin`,
    with the same paired `save`/`restore` guard and the same `< 2 datasets`
    no-op.
  - The ease matters, and it runs in **both** directions. An instant revert on
    release reads as a glitch; a short settle reads as the charts handing
    control back. The pins dim (`is-previewing`) while it is happening.
  - `pointerdown` only **arms** (`scrubArmed`); the borrow engages on the first
    `input` of that press (`scrubEngaged`). Engaging on the press itself would
    fade the charts the instant the thumb is touched, before the user has asked
    for anything, and would make a click that never becomes a drag a visible
    event. This split is also what keeps keyboard out: arrow keys fire `input`
    with nothing armed.
  - `pointerup` and `pointercancel` are on **window**, not the slider: a drag
    frequently releases with the pointer well off the track.
  - **Keyboard is deliberately excluded.** Arrow keys on a focused range input
    fire `input` with no press to end a preview, so there is no natural release
    event to hand control back on. Arrow keys move the tokens and the bars; the
    line charts stay pinned.
  - `burnThroughPlugin` needed fixing for this. It redraws every dataset's line
    inside the tooltip box at full strength, so a pinned-off run came back to
    life there. It now resolves the same alpha via `chartSeriesAlpha`, which
    reads the canvas id so the shared plugin does not have to know which chart
    it is decorating. Tooltip rows are filtered on the same value.
- **Zoom moved into the chart, and the processor out of it.** The `+` / `-` /
  reset trio left the header for a segmented pill docked in each chart's
  bottom-left axis gutter (`.chart-zoom-dock`, all four charts), dimmed to 0.5
  until `.chart-wrap:hover`. The gutter corner is only about 40 to 45px wide
  against a ~56px pill, so `chartGutterLayout` adds 16px of bottom padding to
  reserve a clean strip rather than overlapping the first tick label; the
  compare panel is excluded since it has no dock. No handler changes were
  needed: `handleZoomClick` is a document-level delegate on `data-chart` /
  `data-action`. That frees the header for the pins. `#gpu-label` is gone
  entirely, replaced by a `processorMetaRow` line in the run summary that reads
  `run.processor` (already normalized to GPU / CPU / Unknown at save time) so a
  CPU run is finally labelled correctly.
- **Tooltip swatches were white** on every line chart, a long-standing bug
  spotted while reviewing this pass. Chart.js paints `multiKeyBackground`
  (`'#fff'` by default) behind each swatch and then fills it with the dataset's
  `backgroundColor`; ours are area washes at 0.08 to 0.1 alpha, and the compare
  panel's is literally `"transparent"`, so the white showed through and the
  series color survived only as a rim. A shared `lineLabelColor` callback now
  paints the swatch with the line's own color on convergence, timing,
  confidence, and the compare charts. The entropy chart is deliberately left
  alone: its bars carry solid per-bar colors, so its swatch already read
  correctly and showing the hovered bar's ramp color says more.

**Rejected along the way, so it does not come back:** a discontinuous slider
with a middle "park" dock and a translucent ghost thumb. It was coherent as a
state machine, but the dock is a slider position with no meaning for the token
view or the entropy bars, which the same slider governs; it would have relocated
the honesty problem rather than solved it. It also breaks the drag gesture and
would have meant replacing `<input type="range">` with a hand-rolled widget.

**Previous pass: the shared comparison layer, then two persistence changes.**
Three commits. No Python changed in any of them, so `pytest` (79/79) is a
regression check only; `node --check`, ReadLints, and the 70-column audit are
clean. Nothing needing a display was exercised; checklist items 16 to 20 at the
bottom cover them.

The first commit's idea: **the pre-edit run is a layer, not a mode.** It used to
be one in exactly two places (the diff overlay's token stack, the entropy
chart's crossfade), each with its own control. Four refinements to that same
surface are folded in, since three of them amend code the layer work introduced
and splitting would have committed code the next commit rewrote.

- **The layered spans were inert, which was the blocker.**
  `overlaysBuildDiffLayerSpans` emitted bare `<span>`s with no class and no
  `data-pos`, and both pages' hover paths require both, so hover, the popover,
  and entropy highlighting were all dead in diff mode. Replaced by
  `overlaysBuildTokenSpan` / `overlaysBuildTokenLayer` in `overlays.js`, now the
  single span builder behind every path on both pages: `token-span` plus
  `token-mask` / `token-resolved`, `data-pos`, and a caller-supplied
  `titleFor(index, token)`. A null token renders as the mask glyph instead of
  being skipped, because two layers only line up if both emit a span per
  position. `.diff-layer` became `.token-layer` and the container class
  `.diff-overlay-mode` became `.token-layers`, since layering is no longer
  diff-specific.
- **Pointer ownership is explicit.** Stacked layers share a grid cell, so the
  later sibling used to win every hit test even at zero opacity.
  `overlaysEditedOwnsPointer` states the rule once (the more opaque layer takes
  it, ties to edited) and `overlaysApplyLayerPointers` re-applies it during a
  slider drag, where the generator restyles in place rather than re-rendering.
- **One run-level crossfade** on the token overlay's own heading row
  (`#run-blend-row` / `#run-blend`, state `compareBlend`), grouped with the
  commit-order legend inside `#overlay-header-controls`. That wrapper exists
  because two `margin-left: auto` siblings split the free space between them
  instead of clustering; one auto margin on a wrapper keeps the pair adjacent
  at the right whether or not the legend is showing. Living inside
  `#overlay-viewer` also means the control inherits the viewer's hidden state.
  Its gate widened from the entropy series to `overlayDiffAvailable`, because
  the token layers only need the snapshot while a second bar series also needs
  it to carry `e`. The per-run reset moved out of `clearEntropyChart` into
  `resetRunBlend`, called from `loadRunOverlays`, `clearOverlay`, and
  `showOverlayUnavailable`.
- **Every overlay mode is layered now**, not just Diff. `renderOverlayTokens`
  takes an options object and its `colorFor(index, token)` reads the token, so
  the same callback colors either layer **by its own values**: the original
  layer shows the original run's confidence and entropy, not the branch's. That
  is the whole point of the comparison. Commit Order is the exception, since a
  commit step is a property of a frame stream rather than of a token, so it
  memoizes a second `overlayOriginalCommitSteps` from `original_frames`.
- **Cross-highlighting, both directions, both pages**, via a
  `token-cross-highlight` class and a `setTokenHighlight(pos)` on each page
  (there can be two spans per position now, so both get lit). Analytics drives
  the chart with `setActiveElements` from token hover and the tokens from
  `tokenLinkPlugin`; the generator adds the missing `mousemove` on
  `#entropy-profile`, inverting `drawEntropyProfile`'s `cssWidth / values.length`
  step. The bar-to-token direction is a plugin `afterEvent` hook rather than the
  `onHover` option **on purpose**: Chart.js only fires `onHover` while the
  pointer is inside `chartArea`, so leaving through the axis gutter or off the
  canvas never delivered the empty-elements call and the last token stayed lit.
  `afterEvent` fires for every event in `options.events`, `mouseout` included,
  after the active set is recomputed.
- **One highlight look, not two.** The pointer-driven hover and the
  entropy-driven cross-highlight now share a single rule in `style.css`
  (neutral white plus a soft glow). Neutral because the overlays paint
  arbitrary backgrounds underneath, and the old orange wash vanished on top of
  an orange `.token-remasked` or the Heatmap's warm end. Analytics applies
  `token-hover-highlight` to `#overlay-output` via `updateOverlayHoverHighlight`,
  which it never did before; the chart could light a token that a direct hover
  could not.
- **Highlight tokens left the Settings page** for a checkbox in each page's
  `#overlay-drawer-content` (`#overlay-highlight-tokens`, `.overlay-check`),
  next to the tokens it acts on, applying immediately instead of through the
  Settings stage/Save cycle. It defaults **on** now, via the `gpuTicker`-style
  `parsed.highlightTokens !== false` in `parseSettings`, since a visible
  control no longer needs discovering. Three things worth knowing:
  - The value still lives in the shared `diffusion_settings` blob, read and
    written through `overlaysReadHighlightTokens` /
    `overlaysWriteHighlightTokens` in `overlays.js`, so both pages agree and
    the value survives a restart.
  - `settings.js` therefore has to keep carrying a field it no longer shows.
    `cloneSettings` and `parseSettings` still include `highlightTokens`
    because Save writes the whole blob; drop it there and saving Settings
    silently clobbers the checkbox. `resetStaged` also preserves it, so
    Reset cannot flip a switch that is not on that page.
  - The status-bar `Highlighted Tokens: On/Off` readout is gone, along with
    `updateStatusPrefs`. A visible checkbox makes a mirror redundant.
  - The drawer is hidden until a run exists on both pages, so the checkbox
    appears with the token view. That is also the only time it does anything.
- **Popover pagination.** `.alt-heading` is now a flex row carrying an
  Original/Edited pager (`overlaysBuildAltHeading`) for positions at or past
  divergence where both runs captured candidates. Each page marks the token its
  own run drew. Two details worth keeping: paging re-renders **without** an
  anchor span, because re-placing a box of a different height under the pointer
  that just clicked an arrow slides it away and fires the close; and the
  generator's substitution click is gated on `altsPopoverPage !== "original"`,
  since the worker only holds the live run's state. Analytics also gained the
  generator's `matches(":hover")` keep-open guard, without which the arrows
  would be unclickable.

- **Hyperparameters and the prompt draft survive navigation now.** `boot()`
  rebuilt every control from `specDefault` and `restoreSessionState` never
  touched `paramInputs`, so a trip to Analytics and back reset the form. New
  `PARAM_STATE_KEY = "diffusion_param_state"` in sessionStorage, shaped
  `{ "<modelId>": { experimental, params: { name: rawValue }, prompt } }`.
  Points worth not relitigating:
  - **It could not ride `SESSION_KEY`.** `saveSessionState` bails unless a run
    completed, and `clearSessionState` fires at the *start* of every generate,
    so params would be wiped by Generate and lost outright if you navigated
    mid-run. Form state and run artifacts have different lifetimes.
  - **Keyed by model id** because `param_specs` differ per model and a model
    switch ends in `location.reload()`, which sessionStorage survives, so each
    model keeps its own values across a switch.
  - **Raw `input.value` / `input.checked`, not `getParamValues()`**, so a
    half-typed entry round-trips instead of being rewritten by a `parseFloat`.
  - **Deliberately not in `PERSIST_KEYS`.** It is supposed to die with the app;
    a fresh launch should open on the recommended defaults.
  - Restore order is Experimental, then values, then `applyLimits()` (which
    clamps, refreshes the range tooltips, and validates in one call). It runs
    before `restoreSessionState`, so a completed run's prompt still wins over
    the draft with no special casing. Restoring by spec name means a changed
    spec set degrades: unknown names are dropped, missing ones keep their
    default, and a select value no longer among `spec.options` is refused
    rather than forwarded to the server.
  - Every mutation path goes through `onParamFormChanged`, so the store and the
    Reset button's enabled state cannot drift. Reset re-saves rather than
    clearing the record, because the prompt draft shares it.

**Deferred deliberately:** the generator's own crossfade and two-layer stack.
When scheduled, the crossfade goes in its own row below the frame scrubber.

**Previous pass: What If lifecycle fixes plus the edited-run timing foundation.**
Two commits, both in-sandbox verified (`pytest` 79/79, `py_compile`,
`node --check`, ReadLints clean, 70-column audit). Nothing needing CUDA or a
display was exercised; checklist items 11 to 15 at the bottom cover them.

**What If lifecycle (two reported bugs).**

- **The button stayed live after Confirm.** `confirmGuidedEdit` fires an async
  `saveRun()` and then immediately calls `activateScrubber()`, which re-shows
  `#btn-what-if` and runs `updateEditFramesLock()` while `editedRunSaved` is
  still false, so a second edit could be started during the save. The lock
  condition is now `editedRunSaved || (isSaving && remaskEdits.length > 0)`
  (edits already exist at confirm time, so it reads synchronously), and
  `updateEditFramesLock` is called when the save starts and from both terminal
  handlers, so the lock spans the save and releases if it fails. `setButtonLocked`
  / `setButtonUnlocked` also set `aria-disabled`.
  Note the CSS deliberately does **not** use `pointer-events: none`: that would
  suppress the native `title` tooltip explaining *why* the button is locked. The
  `is-locked` guards in `enterSubstitutionMode` / `enterRemaskMode` are what
  actually block activation.
- **Retry then picking a later token errored.** The client's Retry restores its
  arrays to the pre-substitution run (`restoreEditSnapshot`), but the worker was
  doing `state.update(branch)`, so `last_run_state` had moved to the branch. The
  popover offered the original's candidates while `_validate_substitute` checked
  the branch's, and every position at or after the edit failed with "token X was
  not among the captured candidates at position Y". Positions *before* the edit
  worked, which is what made it look intermittent (a branch copies its prefix
  verbatim, alternatives included). Fixed by discarding the branch trace
  (`state_sink=None`). Chaining was already unreachable from the UI, so pinning
  to the recorded run matches what Retry does. `tests/backends/test_smollm3_substitute.py`
  covers it against a stub, including the negative case that a branch-only
  candidate is still rejected.

**Edited-run timing was misaligned, not just mislabeled.** Both splice sites
(`doSubstitute`, `doGuidedResume`) truncated the frame arrays but left
`perFrameElapsed` whole, so a branch's samples appended to the original's full
array. For an edited run `per_frame_elapsed` was therefore **longer than**
`frames`, and the Timing chart's x axis stopped meaning the same thing as every
other chart's (a 256-token run showed Timing running to ~374 while Confidence
ran to 242). Because `worker_base.py` restarts its clock per segment,
`elapsed_seconds` (the array's last value) was the branch duration alone: the
summary read 2.08s while the chart's own cumulative total showed ~7s.

- Both sites now call `truncateRunArraysAt(offset)`, which cuts `perFrameElapsed`
  with its siblings and records `resumeElapsedOffset` (the elapsed at the last
  kept frame). `handleFrame` adds that offset and re-rounds to the worker's two
  decimals, so the series stays cumulative and frame-aligned. Snapshotted in
  `captureEditSnapshot` / `restoreEditSnapshot` and cleared in `resetRunState`.
- `handleDone`'s first-completion block now also captures
  `originalPerFrameElapsed`, `originalMeanConf`, and `originalPositionAlts`.
  The candidate sets are the load-bearing one: `doSubstitute` truncates
  `positionAlts` at the edit and the branch overwrites the rest, so this is the
  only chance to keep the pre-edit run's. All three are session-persisted in the
  `full` payload, or an Analytics round trip loses them.
- Persisted for edited runs as `original_per_frame_elapsed`,
  `original_elapsed_seconds`, `original_mean_conf` (metadata) and
  `original_alternatives.json`, all optional on `SaveRunRequest` (strict
  pydantic, so `tests/web/test_save_signals.py` pins them). Served by
  `/metrics` and `/frames` respectively. Nothing consumes them yet; they are the
  foundation for the confidence and timing comparison work below.
- `alternativeRecordsFrom` used to gate on the module-level `positionAlts` via
  `alternativesAvailable()`, which would have answered for the wrong run. Split
  out `hasAnyAlternatives(positions)` so it gates on its argument.
- **Legacy runs are repaired at read time.** `total_elapsed_seconds` in
  `metrics.py` sums the segments an elapsed drop delimits, which is the same
  arithmetic `buildCumulativeTiming` already does for the chart. Applied in
  `list_runs` and `_compute_run_metrics` so the two endpoints cannot disagree.
  It is idempotent: a monotonic series has no drop, so it returns the final
  sample unchanged, which is why it can run over every run unconditionally.
- **Watch this when adding the dashed edit marker.** `buildCumulativeTiming`
  finds resume boundaries by looking for elapsed drops, and new runs no longer
  have any, so the light-blue post-resume coloring would have silently vanished.
  `resumeBoundarySet` picks the source explicitly: drops when present (a legacy
  array still holds the pre-edit frames in full, so `remask_edits` does not line
  up with it at all), otherwise the edit's own `frame_index`.

**Earlier in this session.** Landed the `results/` rename, all of **AR Phase C**
(entropy, top-k alternatives, What If substitution), an Analytics **Entropy by
Position** chart on top of it, and then a counterfactual layer on that chart
(hover column, edit-orange markers, a divergence-aware two-row tooltip, an
Original/Edited crossfade slider) plus a collision-aware tooltip positioner
shared by every chart. The Entropy chart and the tooltip positioner have since
been eyeballed by the maintainer and look right; everything else below is still
unvalidated. The sampler and substitution path are covered by unit tests against
a stub model, and the tooltip geometry and tooltip-row logic were exercised
against the real source in a throwaway node harness (corner choice for rising /
falling / flat / bar-dense charts, Liang-Barsky edge cases, hysteresis,
divergence filtering). **Nothing needing CUDA or a display has been exercised in
any pass; see the manual-verification checklist at the bottom.**

**One thing to know before testing:** a SmolLM3 run needs the new **Alternatives**
toggle on for the popover and What If to appear. The `Results/` to `results/`
folder move is already done, history included (see step 0 under "Where to pick
up" for why it was a merge rather than a rename).

**`results/` rename.** One functional line, `RESULTS_DIR = Path("results")`
(`src/web/server.py:74`); everything else routes through that constant. Plus
`.gitignore` and copy in `analytics.js` / `analytics.html` / `index.html` (Help) /
`metrics.py` / `ui_state.py` / `desktop.py` / the docs.

**AR entropy (always on).** `_entropy_nats` in `src/inference/ar_sampler.py`
computes `-(p log p)` over the untempered softmax `_sample_next` already builds,
via `torch.special.entr` so a zero probability contributes 0 instead of a NaN.
Stored raw in nats: normalizing by `log(vocab)` was rejected because a ~128k
vocabulary collapses every realistic value into the bottom of a [0,1] scale, so
the display side normalizes against `OVERLAYS_ENTROPY_REF_NATS = 5.0`
(`overlays.js`) instead. It reaches the UI as a new token field `e` and needed an
explicit `TokenRecord.e` in `server.py`, since that model is strict pydantic and
would otherwise have dropped it silently (a test pins this).

- **Entropy overlay**: `entropyColor` (`overlays.js`) on a cool-blue-to-hot-amber
  ramp, deliberately off the green Heatmap and magenta diff. Offered by both
  `buildOverlaySelect` sites gated on data presence (`entropyAvailable()` /
  `overlayEntropyAvailable(data)`), not on `model_type`, so it lights up for any
  future model that emits `e`.
- **Entropy profile**: `#entropy-profile` canvas in `#scrubber-section`
  (`drawEntropyProfile` in `app.js`), one column per position with the current
  frame's column at full opacity against the others' 0.68, plus a nats readout.
  That opacity is the *only* marking of the scrubber's position, deliberately:
  a drawn marker standing there at rest reads as a rendering artifact (it did,
  and was reported as one), and `drawEntropyProfileGlow` already owns the
  guide-under-the-pointer language. Do not re-add a standing marker.
  Frame index equals position here, since the AR worker emits no leading empty
  frame; an earlier `currentScrubFrame - 1` put the marker, the bright bar, and
  the resting readout one column short. This is the **sequence
  profile**, not a per-position trajectory: an AR model decides each position
  once. (Earlier roadmap notes said "trajectory"; corrected in `ROADMAP.md`.)
  Hovering a token lights its column (`entropyHoverPos` /
  `setEntropyHoverPosition` / `drawEntropyProfileGlow`, with `entropyGlowColor`
  in `overlays.js`) and swings the readout to that position. Hover is tracked for
  every token, not just ones with captured alternatives, and is cleared whenever
  the profile is off screen so it cannot leave a stale column lit.
- **Entropy by Position chart** (Analytics): `#entropy-section` /
  `#chart-entropy` in `analytics.html` with `renderEntropyChart` and friends in
  `analytics.js`. The suite's first chart indexed by **position** rather than
  frame, and drawn as **bars** for that reason. The per-frame axis is a dead end
  for AR: `mean_conf` is a cumulative mean (`conf_sum / count` in
  `ar_sampler.py`), so it flattens by construction and a per-frame mean-entropy
  chart would just be a second flat curve. Also restores a third chart for AR
  runs, which hide Convergence (a fully resolved frame flatlines at 100%).
  Details worth keeping in mind:
  - It is built in `loadRunOverlays`, not `loadRunCharts`, because the
    per-position `e` lives in the **frames** payload rather than the metrics
    payload, and that function already fetches it. `loadRunCharts` carries a
    comment pointing at this. Torn down by `clearEntropyChart` before the fetch,
    so switching runs cannot leave a stale chart up mid-flight.
  - Gated on `overlayEntropyAvailable(data)`, matching the overlay: data
    presence, not `model_type`.
  - y axis uses `suggestedMax: OVERLAYS_ENTROPY_REF_NATS`, not a hard `max`, so
    runs stay comparable without clipping an unusually torn position.
  - `substitutionMarkerPlugin` (modeled on `canvasBoundaryPlugin`) marks the
    union of `token_positions` across `remask_edits`, which for a What If
    branch is the substituted position and therefore the end of the shared
    prefix. Reading `token_positions` rather than `frame_index` is what makes
    it generalize to diffusion remasks. Two hooks: a faint tint behind the bars
    (`beforeDatasetsDraw`) and the dashed line over them
    (`afterDatasetsDraw`), both in `EDIT_COLOR` / `EDIT_TINT` (`#ff9f1c`, the
    `.token-remasked` color). It was accent green at first, which read as "this
    is the app's highlight" rather than "this is an edit".
  - Deliberately **not** given `burnThroughPlugin`: that plugin redraws a
    dataset's line through the tooltip box, which a bar chart has none of, and
    would stroke a stray polyline across the bar tops.
  - `entropyHoverPlugin` lights the hovered column the way the generator's
    profile does, using `entropyColumnSpan` with a 2px floor because a
    256-position run gives each bar about a pixel. The bar itself brightens via
    the dataset's `hoverBackgroundColor` (built from `entropyGlowColor`) rather
    than a hand-drawn bar, so the highlight still honors the crossfade alpha
    below; that is the one place it deviates from `drawEntropyProfileGlow`.

**Counterfactual entropy (Analytics, edited runs).** The chart carries the
pre-edit run as a second layer, so a What If branch can be read against the run
it forked from.

- Two datasets, `Original` at index 0 and `Edited` at index 1, both with
  **`grouped: false`**. That flag is load-bearing: left grouped, Chart.js sits
  the runs side by side and halves every bar instead of superimposing them.
  Labels span the longer of the two, since a branch can outlive or fall short of
  its parent.
- The `#entropy-blend` slider crossfades them through `entropyBlendPlugin`,
  which sets canvas `globalAlpha` per dataset at draw time. Chosen over
  regenerating several hundred `hsl()` strings per slider step, and it leaves
  the entropy ramp itself untouched. A single slider rather than the diff
  overlay's two, because superimposed bars at matching opacity just occlude each
  other. `clearEntropyChart` resets it to Edited so a new run never opens on a
  stale mix.
- The original layer needs three things at once (`entropyOriginalSeries`): a
  divergence point, a saved `original_frames`, and `e` inside it. The snapshot
  exists for any edited run but predates the entropy signal on older ones, so a
  pre-Phase-C branch degrades to the single layer with no slider.
  `framesHaveEntropy` was split out of `overlayEntropyAvailable` for this.
- `divergencePosition` is the earliest edited position.
  `entropyTooltipFilter` drops the Original row left of it, because a branch
  copies its prefix verbatim and the row would only restate the first;
  `entropyTooltipLabel` names the rows from there rightward. The same fact is
  why the crossfade only visibly moves right of the marker. Note the reasoning
  is AR-shaped (one prefix cut); diffusion remasks are scattered, so once those
  runs carry entropy this will want per-position divergence instead.
- **Expect the two rows at the marker to show the same nats.** That is the
  intervention in one line, not a bug: `_substitute_loop` keeps the forced
  position's originally captured entropy because the distribution there is a
  function of the prefix, and forcing a token changes only which one was drawn
  from it. The row labels are what keep it from reading as a duplicate.

**Collision-aware chart tooltips.** `Chart.Tooltip.positioners.smart` used to
park the box in the corner diagonally opposite the hovered point, which knows
where the cursor is but not where the line goes: on a monotonically rising
timing curve "diagonally opposite" aims the box straight at the trendline. It
now scores the four plot-area corners (`smartTooltipCorner`), rejecting any that
holds the pointer or collides with drawn data, and takes the first survivor in
top-left, top-right, bottom-left, bottom-right order.

- Bars are tested as their whole body (`barsHitRect`), not their top edge: the
  bottom corners of a bar chart are solid even where no bar top reaches them.
- Trendlines are tested segment by segment (`lineHitsRect` +
  `segmentHitsRect`, Liang-Barsky), not vertex by vertex. An AR run's points sit
  about a pixel apart so vertices would do, but a 20-frame DiffusionGemma canvas
  has segments long enough to stride across a corner box without landing a
  vertex in it. A zero-length segment degenerates into a point-in-rect test,
  which covers a lone point.
- `pickTooltipCorner` adds hysteresis via `chart.$smartCorner`: the standing
  corner wins while it stays clear, so the box settles instead of hopping
  between two equally good corners on every twitch.
- Shared by all five tooltips (all force `xAlign: "left"` / `yAlign: "top"`, so
  the positioner returns the box's top-left origin). `burnThroughPlugin` is now
  the genuine last resort it was meant to be: it only fires when every corner is
  occupied.

**AR top-k alternatives (opt-in).** `alternatives` BOOL `ParamSpec` on `SMOLLM3`
(`registry.py`), k fixed at `TOP_K_ALTERNATIVES = 5`. Key payload decision: a
position's candidate set is fixed the moment it is sampled, so `_build_frame`
attaches `alts` only to the frame that introduces that position (its last token)
and `handleFrame` accumulates into `positionAlts` by position. O(n·k) on the wire
instead of O(n²·k). Persisted as `alternatives.json`, position-indexed with
`null` gaps, read back by `load_run_frames` (`alternatives_available`) and served
by `/api/analytics/runs/{id}/frames`. Candidate text is the **raw** decode
(control tokens intact); `overlaysAltDisplay` makes whitespace visible at render
time. Hover popover: `#token-alts-popover` (body-level fixed, so the output
area's scroll box and the analytics modal cannot clip it), mirrored read-only in
`analytics.js`. It is placed **above** the token by
`overlaysPopoverLeft` / `overlaysPopoverTop` (shared by both pages): the browser
draws the native `title` tooltip below the cursor and that cannot be moved, so a
popover below would land underneath it. Flips below only for tokens near the top
of the viewport.

**AR What If substitution.**

- **Protocol**: `supports_substitution` on `ModelCapabilities` + `MSG_SUBSTITUTE`
  (`protocol.py`), `Backend.handle_substitute` + a dispatch branch
  (`worker_base.py`). Deliberately NOT `supports_resume`: that flag unhides Edit
  Frames with its frame-selection phase and randomize-remasks slider, neither of
  which means anything for a left-to-right model.
- **Sampler**: `_decode_loop` was refactored onto a shared `_stream_tokens` +
  `_Trace` core, so `streaming_substitute` reuses it. It prefills prompt + kept
  prefix + the forced token in one pass, emits a **seed frame** covering
  positions `0..position`, then continues. **Greedy** (temperature 0) so the
  downstream divergence is the intervention rather than fresh RNG in a shifted
  context. The forced position keeps its originally captured confidence,
  entropy, and candidate set.
- **Worker**: `Smollm3Backend.last_run_state` holds the per-position trace,
  published through a `state_sink` dict rather than the `done` frame so it never
  hits the wire. `_validate_substitute` rejects an out-of-range position and any
  token that was not in that position's captured top-k, keeping the branch a real
  counterfactual. Substitutions **do not** chain: `handle_substitute` passes
  `state_sink=None` so `last_run_state` stays pinned to the recorded run (see
  the Retry fix below). Note the trace key `alternatives` vs the param
  `alternatives_enabled`: they collide if you ever `state.update(params)`.
- **Frontend**: `#btn-what-if` (shown when `supports_substitution &&
  alternativesAvailable()`), `beginSubstitutionSession` / `doSubstitute` reusing
  the diffusion `resumeFrameOffset` / `isResuming` splice path and the existing
  Confirm/Retry review (`retryGuidedEdit` branches back into substitution). The
  branch is recorded as an ordinary `RemaskEdit` (`frame_index == position`), so
  the Analytics Edited column, saved metadata, and the durable diff all work with
  **no server schema change**.
- **Diff un-gated**: both `buildOverlaySelect` sites now list Diff vs Original
  for AR runs once `hasDiff` / `canDiff` (diffusion still lists it up front,
  disabled). `overlaysComputeDiff` already clamps to `min(cur, orig)`, so the
  differing lengths a substitution can produce are handled.
- **Rollback hardening**: `handleError` now calls `restoreEditSnapshot()` before
  `resetGuidedMode()`. A resume or substitution truncates the run before the
  worker answers, so a rejected request (e.g. a worker that lost
  `last_run_state`) previously stranded the user with a half-run. `positionAlts`
  joined the edit snapshot and the session-state payload.

**Earlier session: Settings page + Commit Order overlay.**

- **Commit Order is now an overlay-picker option on the generator** (`app.js`),
  matching analytics: added to `buildOverlaySelect` (diffusion-only, like Diff),
  `effectiveColorMode` returns `overlayMode`, the legend follows the selection,
  and the persistent "Show Commit Order" setting is gone (a stale commit
  selection is reset for AR runs).
- **Shared Settings page** (`settings.html` / `settings.js` / `settings.css`,
  new; route in `server.py`): the generator's Settings modal is replaced by a
  `/settings.html` page with a left tab rail (Appearance / Interface) over the
  four remaining settings (highlight tokens, diffusion-style text + Mode, device
  ticker). The schema/parse/equality moved to `overlays.js` (`SETTINGS_DEFAULTS`
  / `parseSettings` / `settingsEqual`), shared by `app.js` and `settings.js`;
  hydrate-on-navigate (edits persist server-side, the generator applies on next
  load). A gear icon (shared `.header-link-icon` / `.header-link-active`, moved
  from `analytics.css` to `style.css`) links to it from the generator, Main
  Menu, and Analytics.

**Menu model pagination.** The Main Menu model list is paginated
(`MODELS_PER_PAGE = 3`, `menu.js`) with a `< i/N >` pager (styled like the
prompt-history control) pinned to the panel's bottom-left and the Settings gear
bottom-right; the pager hides during confirm/selecting.

**Cross-page download navigation + draggable toast.**

- A model download is a global server task, so the user can now freely navigate
  (pagination, Analytics, Settings, Generation) while it runs. A shared
  `download_toast.js` (on every page) polls `/api/models/download-status` and
  shows a toast whenever the inline veneer is not visible; clicking it returns
  to the download. The menu re-attaches the veneer on load / page-back
  (`reattachDownload` / `syncDownloadBinding`), and `menu-busy` fences model-row
  activation for the download's duration (concurrent activation deferred).
- **Ack + label**: `POST /api/models/download/ack` clears `done`/`error` -> idle
  so completion fires once; `download-status` returns `target_name`.
- **Toast is draggable** with snap-to-quadrant, persisting the corner via the
  UI-state layer (`diffusion_download_toast_corner`); default lower-left.
- **Partial-cache fix**: `is_repo_cached` / `_has_incomplete` (`hf_download.py`)
  treat a cache with `*.incomplete` parts as not-downloaded (used in the fetch
  fast path and `_is_downloaded`), so an interrupted download shows the veneer
  again and resumes instead of bricking on load. The non-functional Cancel
  button + all its wiring (endpoint, method, `_download_cancelled`) were removed.

**In-app docs.** The About / Help modals (`index.html`, `#modal-about` /
`#modal-help`) gained an **Entropy &amp; alternatives** section (what entropy
means next to confidence, the nats scale, the sequence-profile framing) and a
**What If?** section (the one-step flow, why it is greedy, why only captured
candidates are allowed), plus the new Alternatives hyperparameter, the Entropy
overlay, the re-scoped Diff copy, an **Entropy chart** entry in the Analytics
chart list (why it is per position and drawn as bars, and what the dashed marker
means), and `tokens.json` / `alternatives.json` in the saved-files list. Earlier
session: SmolLM3-3B added throughout, the
overlays/settings section rewritten for the Commit Order overlay + shared
Settings page, and the analytics token-overlay section updated for the per-frame
scrubber + Heatmap. `AGENTS.md` asks each session to check About/Help whenever
`HANDOFF.md` is touched.

**Analytics per-frame scrubber + durable Heatmap** (frontend-only; the frames
endpoint `/api/analytics/runs/{id}/frames` already shipped every frame, so no
server / `metrics.py` / `tokens.json` change).

- **Scrubber** (`analytics.html`, `analytics.css`, `analytics.js`): the detail
  modal's token overlay gained a prev/slider/next `Frame i / N` scrubber that
  replays every saved frame through the active overlay, opening on the final
  frame. `renderNoneOverlay` / `renderCommitOverlay` / `renderDiffOverlay` and
  the new `renderHeatmapOverlay` render `overlayFrameAt(overlayFrameIndex)`, with
  `renderCurrentOverlay()` as the shared dispatch (mode change + every scrub).
  Commit steps and the diff change-set are memoized per run (cleared on load);
  the layered Diff clamps the original to its final frame past its end, mirroring
  the generator.
- **Durable Heatmap** (`analytics.js` + shared `overlays.js` `heatColor`): a new
  drawer option recolors resolved tokens by persisted confidence.
- **AR gating**: `loadRunOverlays(runId, run)` threads `model_type`;
  `buildOverlaySelect` offers only None + Heatmap for autoregressive runs (Commit
  Order + Diff stay diffusion-only), reusing the existing `runIsAutoregressive`.
- **Edge cases**: single-frame / legacy runs keep the scrubber hidden + disabled
  (the "unavailable" path is unchanged); empty or out-of-range frames render as a
  blank canvas.

**App icon redesign** (`assets/icon.svg`, `assets/icon.png`,
`scripts/render_icon.py`, `desktop.py`, `README.md`).

- The token-grid icon is replaced by **three CP437 diffusion shade blocks
  (`░ ▒ ▓`)** as vector dither `<pattern>`s (25% / 50% / 75% coverage), drawn as
  vertical-rectangle cells (taller than wide, matching a monospace glyph) under a
  **corner-to-corner dark-to-bright green denoise gradient** (`#245a34` ->
  `#00ff41`) revealed through an SVG `<mask>`. A finer 4px dither plus the
  diagonal keep the dense block's solid columns from reading as flat color walls.
- **PNG**: no SVG rasterizer exists on the hosts, so `scripts/render_icon.py`
  (Pillow, already a dep) redraws the same geometry to `assets/icon.png` (512px,
  4x supersampled). `desktop.py` `ICON_PATH` prefers the PNG and falls back to the
  SVG; the app-menu launcher keeps the SVG. Regenerate with
  `.venv/bin/python scripts/render_icon.py` (keep it in sync with the SVG).

**Earlier session: dropdown, ticker, glyphs, and download (final polish pass).**

- **Download** (`hf_download.py`, `server.py`): the "Click to Download" veneer
  now reports a smooth bar via a **disk-size poller** (repo total from
  `HfApi().repo_info(files_metadata=True)`; polls the cache `blobs/` dir incl.
  `*.incomplete`), replacing the old tqdm hook, which `snapshot_download` never
  routes to the per-file byte downloads (only the outer file-count bar). Xet is
  disabled (`HF_HUB_DISABLE_XET=1`) before the first `huggingface_hub` import in
  `server.py` + `run_worker.py` so the classic downloader is used.
- **Family glyphs** (final, `menu.js`): diffusion = a faint "D" + faint "F"
  with a crisp full-opacity reversed epsilon over their overlap (the D bowl +
  F mid bar), reading as a superposition resolving; autoregressive = an "@"
  that becomes an "R" (inner "a" counter, a tail that loops over the head back
  to a start node, two matched legs). Tooltip stays on the wrapper span.
- **Dropdown polish**: the collapsed device pill is fixed-width (the wider
  `+/-X` face is shrunk to fit, so the border never grows) and its ticker is
  gated off on CPU (headroom is a GPU-only figure); the option list matches the
  collapsed width with ellipsized names (`title` = full name); the hover
  side-popup tints only the trailing signed `+/-X` green/red (body stays grey),
  matching the border. Ticker gated by the "Device tag ticker" Setting
  (`gpuTicker`, default on) + reduced motion.
- **Loaded-model affordance** (`app.js`/`style.css`): the resident model's
  dropdown row is tinted like the Generate button and inert (re-selecting it is
  redundant); its loaded device button is locked (`is-current`) while the other
  device stays switchable (GPU <-> CPU).
- **AR "Step" fix** (`ar_sampler.py`): frames emit a 1-based `index`, so a full
  N-token run ends "Step N/N" (matching the diffusion convention) instead of
  "N-1/N".
- **Veneer + confirm**: a veneered (uncached) model hides its description and
  shows "Click to Download"; on success **Ok** removes the veneer and
  denoise-reveals the description; the menu contracts to a check/X confirm
  before loading (the confirm box no longer inherits the row's pointer cursor).

**Earlier: Menu VRAM headroom, download, and UX polish** (device-aware
activation + that pass).

- **Non-blocking activation**: `manager.activate` spawns and returns; a
  background `_monitor_startup` task tracks `starting|downloading|loading|ready|
  error` (`server.py`), polled via `GET /api/models/activation`; `POST
  /api/models/activate/cancel` stops it. The menu shows a progress bar + Cancel;
  `switchModel` polls the same way. This also fixed the VRAM-on-close leak (the
  load no longer holds the manager lock).
- **VRAM fits fix + headroom**: reclaimable VRAM is counted only when the
  resident model is on `cuda` (`_models_snapshot`), fixing GPU models wrongly
  reading "Available" while a CPU-resident model was loaded. Each model now
  carries a signed `vram_headroom_gib = (free + reclaimable) - required`, shown
  as a green/red pill left of the device tag (menu + dropdown), replacing the
  old "~X GiB Available/Insufficient" text.
- **Orphaned-worker guards**: startup sweep kills stray `run_worker` procs with
  ppid 1 (`_sweep_orphan_workers`), plus the `PR_SET_PDEATHSIG` spawn guard.
- **"Click to Download" veneer**: uncached HF-repo models show a veneer that
  pre-fetches weights (no VRAM) via `POST /api/models/{id}/download` +
  `_run_download` (supervisor thread, `download_with_progress`) with a progress
  bar, then restores the row. `downloaded`/`downloadable` are in the snapshot.
- **Select-to-confirm**: the menu contracts to the chosen row with a check/X
  confirm before loading; the dropdown confirm was reworked (cursor/centering,
  "Unload the current model X and load Y on GPU/CPU?", max-content width).
- **Rename to "LLM Visualizer"** across menu/index/analytics/about/titles +
  desktop `WINDOW_TITLE` (kept `APP_ID`). Menu GPU:/CPU: readout, model-family
  glyphs (diffusion/AR) beside each name, and Analytics **Processor** column +
  per-run timing header (GPU/CPU name).

**Earlier: Phase A, first autoregressive model (SmolLM3-3B).** The AR baseline
end to end, reusing the frame/token contract.

- **New env + deps**: `.venv-ar` with `requirements-ar.txt` (torch 2.8.0 CUDA
  wheel + `transformers==4.53.0` + worker server stack); built and validated
  (SmolLM3 runs on GPU and CPU). `requirements-ar.txt` still lists direct deps
  only, so a `.venv-ar/bin/pip freeze > requirements-ar.txt` to capture the full
  transitive pins is a remaining nice-to-have. README has a CPU-only install
  note. Note `.venv-ar` is now in `.gitignore`.
- **`model_type` capability flag** (`src/backends/protocol.py`):
  `Literal["diffusion","autoregressive"]` on `ModelCapabilities`, default
  "diffusion"; reaches the client via `/api/models`.
- **Device threading**: `Backend.load(*, device="cuda")` +
  `create_worker_app(backend, *, device=...)` (`worker_base.py`), `--device` in
  `run_worker.py`; LLaDA/DiffusionGemma `load` updated (dgemma refuses non-cuda;
  LLaDA falls back to CPU). Supervisor `manager.activate(*, device=None)`
  resolves None → cuda-if-GPU-else-cpu, appends `--device`, and skips
  `_preflight_vram` on CPU. `activate_model` accepts an optional `{device}` body.
- **Registry**: `SMOLLM3` (`src/backends/registry.py`), id `smollm3`,
  `.venv-ar`, `HuggingFaceTB/SmolLM3-3B`, `min_vram_gib=8.0`,
  `model_type="autoregressive"`, `supports_resume=False`; params
  max_new_tokens/temperature(0.6)/top_p(0.95)/seed/thinking.
- **AR sampler** (`src/inference/ar_sampler.py`): manual token-by-token KV-cache
  loop in a thread, full-snapshot frames `{t, m:false, id, c}` + `done`
  {final_text, thinking}; `c` = untempered softmax prob of the sampled token;
  temperature/top_p sampling; `<think>` split; cancel between tokens.
- **AR worker** (`src/backends/smollm3_worker.py`): mirrors dgemma, `load(*,
  device)` bf16 on GPU/CPU, clamps params (CPU caps max_new_tokens to 128), no
  resume.
- **Menu CPU/GPU toggle** (`menu.js`/`style.css`): AR row is always selectable
  (CPU fallback) and carries a `.menu-model-device` toggle; GPU option greys out
  when no GPU or won't fit; `selectModel` posts `{device}`.
- **Generator gating** (`app.js`): `isAutoregressive()`; Diff option omitted
  from the overlay picker for AR; Commit Order setting disabled + forced Off +
  dimmed (`updateCommitSettingAvailability`), `effectiveColorMode` and the
  status bar guarded; Edit Frames already `supports_resume`-gated.
- **Analytics gating**: `model_type` persisted in `metadata.json`
  (`_save_run_blocking`); `analytics.js` hides the convergence chart
  (`#convergence-section`) for AR runs and skips their series in the compare
  view; timing + confidence kept.

## Previously shipped (recent sessions; now in git + README)

- Durable server-side UI state (`results/ui_state.json` via `/api/ui-state`,
  `src/web/ui_state.py`): Settings, the "new run" cue, prompt history, and the
  generate teaser survive restarts and are shared across browser + desktop,
  independent of window origin; the cue self-heals against deleted runs. Plus an
  analytics table rework (reordered columns, diffusion-textured Edited
  checkmark, checkbox row highlight, multi-select bulk delete) and a stable
  desktop launcher port with a persistent web-storage profile.

- Main Menu + title video + generation gated behind model selection; consistent
  Menu/Generation/Analytics nav; analytics layered Diff vs Original; removed the
  idle-animation feature; opt-in diffusion-style text + button micro-interactions;
  confidence-driven mask opacity (LLaDA); in-place edited-run save; randomize
  remasks; GPU/desktop robustness (`nvidia-smi` resolution, `libxcb-cursor0`).
- Earlier: durable overlays + the analytics token viewer, the wide detail modal,
  guided-edit Confirm/Retry, prompt history + New Run, the optional desktop app,
  and automatic asset cache-busting. See git log / `README.md` for detail.

## Conventions

- Use `.venv/bin/python`, `.venv-dgemma/bin/python`, and `.venv-ar/bin/python`
  explicitly; never system Python. Dependency files: `requirements.txt`,
  `requirements-dgemma.txt`, `requirements-ar.txt`, `requirements-desktop.txt`.
  Follow TigerStyle and `.cursor/rules/`.
- Verify: `.venv/bin/python -m pytest`; `node --check` on changed JS; `py_compile`
  on changed Python; ReadLints. GUI/GPU can't be tested in-sandbox; hand back with
  a manual-verification checklist.
- See `AGENTS.md` for the full workflow, commit discipline, and this handoff habit.

## Where to pick up

This session shipped the `results/` rename and all of **AR Phase C** (see
"Recently shipped"), but **none of it has been validated on hardware**, so the
first task is the checklist below. After that, the agreed next candidates are
Mamba-3, then extending entropy / top-k to the diffusion models. Deliberate each
in Ask mode before Plan.

**0. Validate AR Phase C on hardware (do this first).** Nothing here could be
exercised in-sandbox (no CUDA, no display).

The folder move is **already done**: `results/` now holds all 103 runs (184 MB)
and `Results/` is gone. Worth knowing why it was not a plain rename. `RESULTS_DIR`
is relative to the process CWD, so the server created a fresh lowercase `results/`
on its next run and both directories coexisted; `mv Results results` would then
have nested the history as `results/Results/`, invisible to Analytics. The two
`ui_state.json` files were merged in favor of the older one, since the new one was
written from a partial in-memory state and had lost `diffusion_settings` and
`diffusion_download_toast_corner`. Only the lowercase name is gitignored, so if a
stray `Results/` ever reappears it will show up as untracked and is easy to stage
by accident. Then:

1. **Alternatives off** (the default): a SmolLM3 run still streams normally, the
   **Entropy** overlay appears and recolors tokens, the entropy profile draws
   under the scrubber and tracks the scrubber's frame, tooltips show an `Entropy`
   line, hovering a token makes its profile column glow and swings the nats
   readout to it, and **no** What If button or hover popover appears.
2. **Alternatives on**: hovering a token opens the candidate popover with five
   rows, sane probabilities, the chosen token highlighted, and readable
   whitespace/control candidates. It should sit **above** the token, clear of the
   native tooltip below the cursor, and flip below only for tokens near the top of
   the viewport. Reaching into the popover to click a candidate should keep it
   open and keep the column glowing.
3. **What If**: the button appears; clicking it underlines the captured positions;
   clicking a candidate truncates and regenerates from there; the run lands in
   the Confirm/Retry review; **Retry** re-enters substitution (not the diffusion
   edit session); **Confirm** saves.
4. **Diff vs Original** is offered for the confirmed AR branch and renders both
   layers (worth checking with a substitution near the end, where the branch and
   original differ in length).
5. **Analytics**: open the saved branch. Entropy overlay, the hover popover, the
   Edited column, and Diff vs Original should all work post-hoc. Confirm
   `alternatives.json` is present and position-indexed.
6. **Entropy chart**: the detail modal shows a third chart for AR runs (Timing,
   Confidence, Entropy by Position) whose bars match the shape of the generator's
   profile for the same run. Hover lights the column and names the token with its
   nats; zoom, pan, Reset, and the eye toggle all behave. Then open an AR run
   saved **before** this session: the section should stay hidden rather than
   drawing an empty chart. Switch between two runs without closing the modal to
   confirm no stale chart survives.
7. **Counterfactual layer** (open the confirmed What If branch): the substituted
   position carries an **orange** dashed marker and a faint orange column tint,
   matching the generator's remask color rather than the old green. Hovering at
   or right of that position gives two rows labeled Original and Edited;
   hovering left of it gives one unlabeled row. **The two rows at the marker
   should show the same nats and different tokens**, which is correct (see
   "Counterfactual entropy" above), so treat matching numbers there as a pass,
   not a bug. The Original/Edited slider appears; dragging it left fades the
   branch out and the pre-edit run in, and the change should be visible only
   right of the marker. Reopening a different run should reset the slider to
   Edited. An AR run edited **before** this session (snapshot without `e`) should
   show the marker but no slider and no second row.
8. **Tooltip placement**: on the Timing chart, hover anywhere along the rising
   line; the box should settle in the **top-left** rather than on the trendline,
   and should not flicker between corners as you sweep the pointer. On
   Confidence (a curve hugging the top) expect a bottom corner. Park the cursor
   in the box's chosen corner and confirm it moves out of the way. Burn-through
   should now be rare: it is the fallback for when all four corners are
   occupied, most likely on a heavily zoomed view.
9. **Error path**: substitute after switching models and back (which clears
   `last_run_state`). Expect a clean error message and the **original run
   restored**, not a truncated one.
10. **Regressions**: one LLaDA Edit Frames session still works end to end
    (`_stream_tokens` refactor and the `handleError` rollback are the shared
    surfaces), both models still save/load normally, and a diffusion run's
    detail modal still shows exactly its three charts with no Entropy section.
    The tooltip positioner is shared, so give the Convergence chart a hover too.

The rest cover the What If lifecycle and timing pass:

11. **Confirm locks What If**: SmolLM3 run, What If, pick a candidate, let it
    finish, click the green check. The button must grey out the instant you
    click (not when the save lands) and stay unclickable, and hovering it should
    still show the "already has a saved edit" tooltip. Try clicking it during
    the save specifically, since that was the open window.
12. **Retry then pick a later token**: same setup but click the blue Retry
    icon, then hover a token **after** the edit position and pick a candidate.
    It should run, with no "not among the captured candidates" error and no
    need to hit Generate first. Repeat picking a token *before* the edit
    position, which worked even with the bug, to confirm nothing regressed.
13. **Timing alignment on a fresh edited run**: save one, open Analytics. The
    Timing chart's last frame index should now match Confidence and Entropy
    (before, a 256-token run showed ~374 against 242), and the **Elapsed** row
    should agree with the chart's final y value rather than showing just the
    branch (2.08s against a ~7s chart was the symptom). The post-edit segment
    should still be light blue, now driven by `remask_edits` rather than an
    elapsed drop.
14. **Legacy runs still read correctly**: open an edited run saved *before* this
    pass. Its Time column and Elapsed row should now show the repaired total,
    its Timing chart should be unchanged (still misaligned, since the stored
    array cannot be re-cut), and its post-edit segment should still be light
    blue via the drop heuristic. Also confirm an unedited legacy run's elapsed
    is untouched.
15. **Diffusion parity and session round trip**: run a LLaDA Edit Frames resume
    and confirm the same alignment holds there. Then generate, edit, navigate to
    Analytics and back, and save: the restored session should still produce a
    complete edited run (the pre-edit signals ride in the `full` sessionStorage
    payload, which falls back to a lighter one when the quota is hit).

The rest cover the shared comparison layer. Every item has a degrade path,
because runs saved before the previous pass carry no `original_alternatives`.

16. **Diff mode is interactive again** (the regression this pass was blocked
    on). Open an edited run in Analytics, pick **Diff vs Original**, and hover a
    token: the tooltip, the candidate popover, and the entropy bar highlight
    should all now work, where before nothing happened. Drag the two opacity
    sliders past each other and confirm the popover starts reading the layer
    that is now more opaque. Then check the same in the generator's diff
    overlay. Nulls in a frame now draw as `░` rather than vanishing, so watch a
    substitution near the end where the two runs differ in length.
17. **The crossfade governs everything.** Open a What If branch saved **this
    session**. The Original/Edited slider should sit at the right of the
    **Token overlay** heading row, directly above the text it blends, and
    appear for any edited run with a snapshot (not only ones carrying
    entropy). Switch to **Commit Order** and confirm the legend and the
    slider sit together at the right rather than spreading across the row,
    and narrow the window until the header wraps. Drag it in **None**,
    then **Heatmap**, then **Entropy**, then **Commit Order**: the tokens should
    crossfade between the two runs in every mode, and the entropy chart's bars
    should follow the same slider. The key correctness check is Heatmap and
    Entropy: at full Original the colors must be the **pre-edit run's own**
    confidence and entropy, not the branch's colors under the original text.
    Commit Order has its own second steps array, so its original layer should
    show the pre-edit run's gradient. Switch runs and confirm it reopens on
    Edited. Then open an **unedited** run: no slider, one layer, everything
    exactly as before.
18. **Cross-highlighting, and that it lets go.** In Analytics, hover a bar and
    watch the matching token light up white in the overlay above; hover a token
    and watch its bar light. Then sweep off the chart three ways: down into the
    x-axis label strip, sideways off the canvas, and by closing the modal
    outright. The token must go dark every time (this was the sticky-highlight
    bug, and the axis-gutter exit is the one the old `onHover` wiring missed).
    On the generator, sweep the entropy profile and confirm the token lights and
    tracks correctly at both ends of the sequence (that is the math inversion;
    an off-by-one would show as a consistent one-token drift). With the
    crossfade mid-way, both layers should light at once.
18b. **One highlight look.** A direct token hover and an entropy-driven
    highlight should now be visually identical (white with a soft glow) on both
    pages, where the direct hover used to be a separate orange. Check it over a
    remasked token and at the Heatmap's warm end, which is where the old orange
    disappeared. Analytics tokens should highlight on direct hover at all,
    which they previously did not.
18c. **Entropy profile: no standing marker.** The strip should carry **no**
    drawn marker at rest, only the hover guide under the pointer. The 2px
    white stub that used to float at the top near the last column is gone.
    The scrubber's frame is still marked, but only by its bar sitting at full
    opacity against the others' 0.68, and the nats readout at the right
    should name that frame's own position: scrub to the **last** frame and
    confirm the readout reports the final position, not the one before it.
18d. **The highlight checkbox, and that nothing clobbers it.** The **Overlay**
    drawer on both pages should carry a **Highlight tokens** checkbox, ticked
    by default, taking effect the instant you toggle it. Untick it on the
    generator, open Analytics, and confirm its drawer opens unticked too, then
    toggle it there and confirm the generator agrees on return. Reload and
    confirm it holds. The load-bearing regression check: with it **unticked**,
    open the Settings page, change something (or just hit **Reset**), hit
    **Save**, then go back and confirm the checkbox is still unticked.
    Settings saves the whole blob and no longer shows this field, so a
    clobber here is the failure mode to watch. The status bar should no
    longer carry `Highlighted Tokens: On/Off`, and Settings' Appearance tab
    should be down to the diffusion-text rows.
18e. **Form state across a visit, and only a visit.** This is the reported bug:
    turn on **Alternatives**, change a number or two, tick **Experimental**,
    type into the prompt, then go to Analytics and come back. Everything
    should be exactly as you left it. Repeat the round trip **before** running
    anything and again **after** a completed run; both paths were broken, and
    the second is where a completed run's prompt should override the draft
    rather than the other way round. Then hit **Generate** and confirm the
    params are still there afterward (they used to be cleared at the *start*
    of a run). Switch models and back, confirming each model keeps its own
    values across the reload. Finally close the app and relaunch: everything
    should be back to the recommended defaults, which is the intended
    boundary. Worth one look at a half-typed value (leave a trailing decimal
    point), which should come back as typed rather than rounded.
18f. **Reset to Defaults.** The **Reset** button at the right of the
    Experimental row should start greyed out, light up the moment anything
    differs, and grey out again once you undo the change by hand. Click it
    with Experimental on and several params changed: everything returns to
    the defaults, Experimental clears, and the range tooltips follow the
    narrower bounds. It should leave the prompt alone, and it should be
    unavailable while a generation is running.
19. **Popover pagination.** On a branch saved this session, hover a token
    **past** the substitution: the heading should read "Position N: Edited" with
    a `‹ ›` pager, the current side disabled. Click `‹` for the pre-edit
    candidates, and confirm the marked "chosen" row changes to the token the
    original run actually drew. The box should stay put rather than jumping when
    the two pages differ in row count, and reaching into it must not close it.
    Hover a token **before** the substitution: no pager, since both runs
    recorded the same set there. In Analytics, drag the crossfade below halfway
    and confirm a freshly hovered token opens on **Original**. On the generator
    with What If armed, the Edited page should be clickable and the Original
    page read-only (no hint line, no substitution on click).
20. **Degrade paths.** Open an edited run saved **before** the previous pass:
    the crossfade should be absent (no snapshot at all) or present with a
    single-dataset chart (snapshot without `e`), and the popover should show no
    pager (no `original_alternatives`) while otherwise working. Nothing here
    should throw; the failure mode to watch for is a blank overlay.
21. **Both runs draw on the line charts.** Open a What If branch (or a LLaDA
    Edit Frames resume) saved with its pre-edit snapshot. Timing and Confidence
    should each show two lines: grey solid for the original, colored dashed for
    the branch, overlapping exactly until the edit and separating after it.
    Neither chart should be filled. An **unedited** run should still show one
    filled line and **no pins** at all.
22. **The pins, all three states.** Both are lit green on open. Click 2: only
    the grey original remains and 1 becomes unclickable (cursor does not change
    to a pointer, clicking it does nothing). Click 2 again for both. Click 1:
    only the dashed branch remains. Confirm the two charts are independent of
    each other, and that tooltips drop the row for whichever run is hidden.
23. **The drag preview and the ease back.** With both pins lit, drag the
    Original / Edited crossfade above the token view. Timing and Confidence
    should crossfade along with the tokens and the entropy bars, and the pins
    should dim while you hold. Release: the two line charts should settle back
    to showing both over roughly a fifth of a second, and the pins should
    brighten. Release with the pointer dragged well outside the slider (and
    outside the modal) and confirm it still hands back.
24. **Keyboard is intentionally not previewing.** Click the slider thumb, then
    use the arrow keys. The tokens and entropy bars should move; Timing and
    Confidence should not. This is by design, not a bug.
25. **Zoom dock.** All four charts should carry the `+` / `-` / reset pill in
    their bottom-left corner, faint until you hover the chart. Check it does
    not overlap the first x-axis tick label or the axis title on any of them,
    and that zoom, pan, and reset still work. Scroll-wheel zoom over the plot
    should be unaffected.
26. **Tooltip swatches.** Hover any of Convergence, Timing, Confidence, and a
    compare-panel chart. Each swatch should be a solid chip of that series'
    line color, not a white box with a colored rim. On an edited run the two
    chips should be visibly different (grey for the original).
27. **Processor row.** The run summary above the charts should carry a
    `GPU: <name>` line (or `CPU: <name>` for a SmolLM3 CPU run, which is the
    case the old timing header got wrong), and the Timing header should now
    read just "Timing" plus its controls. An older run saved without the field
    should fall back to the detected GPU rather than showing an empty row.

**0. The comparison surfaces (agreed with the maintainer, partly shipped).** The
timing foundation exists to serve these, and the unifying idea is settled: **the
pre-edit run is a first-class layer everywhere**, driven by one shared
Original/Edited state rather than a bespoke control per surface. The shared
layer, cross-highlighting, popover pagination, and now the line charts have all
landed (see "Recently shipped"); what remains is below.

One correction to the framing above, learned from the line-chart pass: **one
shared state does not mean one shared control.** "Both runs at once" is only
expressible on stroke marks, so the line charts needed their own pins, with the
crossfade reduced to a momentary borrow during a drag. Expect the same split
anywhere the mark type cannot show two runs at full opacity.

- **The generator's crossfade and two-layer stack**, deliberately deferred. It
  has the bidirectional cross-highlighting and the popover pager already, but
  its token view is still single-layer outside diff mode. The shared primitives
  are in place (`overlaysBuildTokenLayer`, `overlaysEditedOwnsPointer`,
  `overlaysApplyLayerPointers`), so the work is the control and routing
  `renderFrameWithTokens` through the layer builder. **Settled:** the crossfade
  goes in its own row below the frame scrubber, not in the toolbar. Unlike
  Analytics the generator has live state to respect, so gate it on
  `diffAvailable()` and keep it out of the way during an edit session
  (`remaskMode !== null`).
- **Confidence chart**: cumulative versus per-position toggle. Note the default is
  **modality-aware**, not per-position everywhere: the AR `mean_conf` is a
  cumulative running mean so its curve is degenerate, but LLaDA's and
  DiffusionGemma's are per-frame canvas means and that curve is what makes
  adaptive stopping visible. Per-position is the entropy chart again (read `c` off
  the final frame), so it comes from the **frames** payload while the line comes
  from **metrics**, and the toggle straddles two independent fetches.
- **Timing chart**, the remainder after this pass. The two-line overlay shipped,
  so what is left is: a dashed edit marker (reuse `substitutionMarkerPlugin`
  with frame indices via `resumeBoundarySet`; its orange tint is sized from bar
  geometry, so a line chart needs either a width strategy or no tint), "E204"
  style tooltip labels for the branch's frames, and both elapsed totals in the
  summary rather than only the combined one. The marker is the valuable one:
  the two lines separate visibly, but nothing yet names the frame where they
  do.
- **Status message stack** (the one piece with no dependency on any of this):
  `#status-message` is a single overwritten span, and both existing toast systems
  (the draggable download toast, the analytics delete toast) are single-slot. The
  split that makes it tractable: leave the steady-state readouts (`Step`,
  `Elapsed`, prefs) in the footer and lift only the **event** messages into a
  stack, which is what makes "Saving run" and "Resuming" coexist. Watch for
  collision with the download toast, which is fixed-position and drag-positioned.

**1. State-space models: Mamba-3 (new model class).** A genuinely new xAI
direction: SSMs compress all context into a fixed-size recurrent state, so they
unlock overlays no other model here can show. Target the 1.5B SISO / MIMO
checkpoints on the `state-spaces` HF org (Mamba-3, arXiv 2603.15569). **Key open
decision, settle first: base vs instruct.** The `state-spaces` weights are base
LMs (Pile / SlimPajama), not instruction-tuned, so chat prompting reads as raw
completion unless we find an instruct-tuned variant or frame it as a
base-completion model. Constraints: its own `.venv-ssm`; native `mamba-ssm` +
`causal-conv1d` CUDA / Triton kernels, so GPU-only (no CPU fallback like SmolLM3)
and likely a custom decode loop rather than transformers `.generate`. VRAM is
trivial (~3 GB weights + constant state). The **streaming baseline reuses the AR
frame / token contract** and `model_type` gating (no Edit Frames; keep timing,
confidence, Heatmap); reserve a distinct `ssm` capability flag for later.
**Phase-2 xAI payoff:** SSM-native overlays on the recurrent state, per-token
Δ / state-write intensity, a state-norm evolution sparkline, and fixed-state
forgetting / retrieval probes, all gated on capturing kernel intermediates
(forward hooks or the reference path). Its streaming baseline can lift the AR
signal capture wholesale, since `ar_sampler.py` now separates the numeric core
(`_sample_next`, `_entropy_nats`, `_top_alternatives`) from the decode loop, and
an SSM is left-to-right too, so the position-indexed views (the profile and the
Analytics Entropy chart) apply to it unchanged.

**2. Entropy and top-k for the diffusion models.** The signals generalize but the
shape does not: a diffusion position is re-decided every step, so entropy becomes
a per-position **trajectory** over steps rather than the single value the AR case
yields, and the same is true of its candidate set. Settle in Plan: the payload (a
trajectory per position is O(n·steps), against the O(n) the AR profile costs),
whether it rides DiffusionGemma's existing `entropy_signal` toggle, and what the
position-indexed views become when a position has a history (both the profile and
the Analytics Entropy chart read the final frame today, which for diffusion would
show only each position's last value). The overlay plumbing is already
model-agnostic: both `buildOverlaySelect` sites and the chart gate Entropy on the
data (`entropyAvailable()` / `overlayEntropyAvailable(data)`), not on
`model_type`, so a diffusion sampler that emits `e` lights them up with no
frontend change, which is exactly why the shape question needs settling first.

**Standing backlog (unchanged; see `ROADMAP.md`):** multi-canvas DiffusionGemma
resume (the remaining Phase 2 milestone); mask confidence-opacity for
DiffusionGemma; integrate Phi-4-mini-instruct / Gemma-3n-E2B-it (share
`ar_sampler.py` + `.venv-ar`); download-from-menu for a curated candidate
allowlist (needs a dynamic registry layer); cross-model comparison on identical
prompt/seed; aggregate analytics across runs; the deferred Randomize-Prompt
utility worker; **real download cancellation** (run the fetch in a killable
subprocess, then terminate it and `rmtree` the model's HF cache dir on Cancel,
deferred because the current thread-based fetch cannot be killed, the
`.incomplete`/resume fix makes an interrupted download recoverable in the
meantime).

**Remaining AR nice-to-haves (from the SmolLM3 session):** confirm SmolLM3's
thinking delimiter on a real thinking-on run (`ar_sampler.py` splits on
`<think>` / `</think>`; adjust `_split_thinking` / `_STRIP_TOKENS` if the tags
differ), and `.venv-ar/bin/pip freeze > requirements-ar.txt` to capture the full
transitive pins.

## North star & backlog

Host open-source models of every generation paradigm (diffusion, autoregressive,
and next state-space) behind one contract, reusing the diff/overlay tooling as a
cross-model explainability lens. Standing backlog (`ROADMAP.md`): multi-canvas
DiffusionGemma resume; entropy + top-k for the diffusion models (a trajectory per
position, not a single value); latent-space probes; a random-prompt generator (a
small CPU-only "utility" model in its own concurrent worker, since the supervisor
stays torch-free and the main manager runs one worker at a time); in-app
camera/screenshot-to-Downloads button; aggregate analytics across saved runs.
