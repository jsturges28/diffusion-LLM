# HANDOFF: next session

Living, per-session handoff. The agent updates this at the end of each session so
the next one can pick up cold (see `AGENTS.md`). Read `README.md` + `ROADMAP.md`
first, then deliberate the work below with the maintainer before Plan mode
(deliberate in Ask mode → Plan → Agent).

## What it is

A local FastAPI + WebSocket visual playground and analytics suite for LLMs,
deepest on discrete diffusion and built to take more model classes over time,
oriented toward explainability (XAI). Runs in the browser
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

**Latest pass: the token metrics strip.** One always-present readout above each
token canvas, on the generator and in the Analytics detail modal, replacing the
native `title` tooltip on tokens. Entirely frontend; no Python module was
touched. `pytest`, `node --check`, ReadLints, and the 70-column audit are clean.
Checklist items 74 to 80 carry the display work.

- **Why the tooltip had to go, beyond looks.** Three problems, and only one of
  them was cosmetic. The browser delays a `title` by around half a second and
  offers no way to configure that. It cannot be styled or positioned; the
  candidate popover's above-the-token preference existed *to dodge it*
  (`overlaysPopoverTop`), a workaround that is now just a choice. And it is
  bound to one element, so the entropy chart could never feed it however
  obviously it should have. The strip is fed by **both** hover sources on both
  pages, which is the feature the tooltip structurally could not have had.
- **It is a net deletion.** The tooltip text was written in exactly one place,
  `overlaysSyncTokenSpan`, and everything upstream existed only to feed it:
  `tokenTitleFn`, `tokenTitleExtra`, `tokenExtraLabel`, `tokenLabel` and
  `confLabel` on the generator plus an inline `titleFor` in its diff overlay;
  `overlayTitleFn`, `commitExtraFor`, `overlayConfText`, `overlayEntropyText`
  and the `extraFor` / `originalExtraFor` parameter chain on Analytics. The
  strip computes the same values at hover time from the same memoized state
  (`currentDiffData`, `commitStepsFor`, `overlayDiffData`), so none of that was
  rerouted, and `tokenLayerOptions` lost the `total` argument it only ever
  passed to a title.
- **One renderer, two callers.** `overlaysBuildTokenMetrics` builds the row's
  children once at boot and caches them on the element; `overlaysRenderTokenMetrics`
  writes a plain reading object into them. Each page owns every decision about
  *what* is under the pointer (which frame, which overlay, which layer) and
  hands over a flat object, which is what keeps the formatting from forking
  into two dialects. Both bars reuse the existing ramps (`heatColor`,
  `entropyColor` / `overlaysEntropyFraction`), so the strip speaks the same
  colors as the canvas above it.
- **Always present, ~28px reserved.** Anything that appears on hover pushes the
  canvas down at the moment the pointer enters it, moving the tokens out from
  under the cursor that summoned it. Idle it keeps every label and shows a dash
  per value, dimmed as a whole, so it reads as a key to what it will say.
- **`metricsHoverPos` is deliberately not `entropyHoverPos`.**
  `setEntropyHoverPosition` forces its position to null whenever the profile row
  is hidden, which is exactly the live-generation case where the strip has
  something to say. They answer different questions with different lifetimes:
  which column is lit, versus which position is being read. Both are set from
  the same three call sites.
- **Live generation gained a readout it never had, for free.**
  `LIVE_TOKEN_OPTIONS = {}` means streaming tokens carry no hooks at all, but
  they have always carried `data-pos`, so the strip works there once
  `renderLiveFrame` refreshes it per frame.
- **Absent is not zero.** The tooltip printed `Confidence: 0` when a run had
  never recorded the signal, which is a claim about the model rather than about
  the record. The strip shows a dash when the field is missing and a real number
  when it is genuinely zero. Masks keep their zero, including positions queued
  for remasking, where whatever the old token scored says nothing about it.
- **Which stacked layer a reading came from** is taken from the hovered span's
  own `.token-layer-original` / `.token-layer-edited` ancestor, so the strip
  reports what is on screen. Chart hover has no span and falls back to
  `overlaysEditedOwnsPointer`, which is the same answer by construction: exactly
  one layer is interactive, so the hovered span is always in it. A crossfade
  moved without the pointer moving has no new span to ask, so
  `refreshTokenMetricsLayer` re-derives from ownership on a blend change.
- **Where the refresh hooks live.** Both pages funnel it: the generator's
  `renderFrameWithTokens` became a two-line wrapper over
  `renderFrameWithTokensDraw` so every one of its two dozen callers refreshes,
  and Analytics refreshes at the end of `renderCurrentOverlay`. That is the
  point to extend, not the call sites.
- **The candidate popover now clears the strip.** `overlaysPopoverTop` took a
  third argument, the viewport y of the token canvas's own top edge, and
  prefers "above" only while the box stays inside the canvas. It had tested
  against the *viewport*, which the canvas starts far below, so a token in the
  first line or two pushed the popover up over the strip (and, on the
  generator, over the hyperparameter row). Overlapping the tokens is the whole
  point of placing it above; overlapping their readout is not. The fallback is
  safe by construction: the only tokens that reach it are near the top of the
  canvas, which is the case with the most room underneath.
- **No `aria-live`.** A region that re-announces on every hover would be
  unusable with a screen reader, so this stays a visual readout.

**Previous pass: per-class glow tuning, sub-setting grouping, and the load
sweep.** Entirely frontend; no Python module was touched. `pytest` (157/157),
`node --check`, ReadLints, and the 70-column audit are clean. Checklist items
67 to 73 carry the GPU and display work.

- **The glow is tuned per model class.** **Brightness** (50-200%) and **Fade
  time** (200-2000ms) are stored per `model_type` behind a **Tune for** picker,
  so only two sliders are on screen at a time. The reason it has to be per
  class is that the trail an eye can follow is roughly *rate times fade*: a GPU
  autoregressive run outpaces a diffusion step by an order of magnitude, so the
  default that reads perfectly on LLaDA is gone before it registers on SmolLM3.
  Both defaults are the old fixed look (100%, 500ms), so an existing profile
  sees no change. Three things are load-bearing:
  - The values reach the keyframes as **whole shadow lists** in custom
    properties (`--token-birth-shadow`, `--token-birth-shadow-off`,
    `--token-birth-duration`), not as numbers nested inside `rgba()`. That
    keeps each keyframe a plain substitution, which is the portable form; the
    properties also carry the old literals as `var()` fallbacks, so the effect
    still looks right if it runs before JS sets them.
  - Brightness scales the **blur radii as well as the alphas**. Alpha alone
    tops out barely above the default 0.9, which is not enough headroom to
    make a fast run legible, so "brighter" has to mean bloomier too.
  - The concurrency cap is **derived from the fade**, not fixed:
    `clamp(round(fadeSeconds * 96), 48, 192)` in
    [app.js](src/web/static/app.js). The 96 tokens/second ceiling is chosen so
    the 500ms default still lands on the 48 the cap was hardcoded at, making
    default behavior provably unchanged. Left fixed, a 2s fade at
    autoregressive speeds would have the FIFO rather than the timer decide when
    a flash ends: the trail would stop growing exactly when the user lengthened
    it, and its tail would look cut rather than faded. This is the subtlety to
    remember if the bounds ever widen.
  - The Settings page has no generation on it, so the two sliders would
    otherwise be set blind. The preview is **a block of copy whose words are
    the tokens**, driven through the same `overlaysApplyGlowVars` and the same
    keyframes as the live canvas, which is what stops it drifting from the real
    thing. Under reduced motion it holds a fixed handful lit instead of
    animating: the constraint is on movement, not on the glow, and brightness
    is still legible where a fade is not.
  - **Why the preview is copy and not one token.** Brightness is judgeable on a
    single token; **fade is not**, because what a fade produces is a *trail*,
    and a trail needs enough simultaneously lit tokens to have a head and a
    tail. The count is roughly fade over tick, so the geometry and the pacing
    are one decision, and both are pinned by the worst case (the 2000ms
    maximum): 38 words of copy at 90ms per word autoregressive / 380ms per
    burst diffusion peaks at 23 and 28 lit at that maximum, and 3 and 6 at the
    200ms minimum. Change any one of the copy length, `GLOW_PREVIEW_TICK_MS`,
    the burst range, or `max-width` and the others need rechecking, or the top
    of the range saturates into a solid block and stops showing anything.
  - The two classes are **paced to the same 3420ms total** so switching **Tune
    for** compares the glow rather than the tempo, but they spend it in the
    shape of the real thing: one word per tick left to right for
    autoregressive, a scattered burst for diffusion. Note that 380ms is
    *derived*, not chosen: the burst sizes are drawn, they come to nine ticks,
    and 3420 / 9 is what equalises the totals. Reseeding moves the tick count,
    so the interval has to be recomputed with it.
  - **Burst sizes vary** (`GLOW_PREVIEW_BURST_MIN` to `_MAX`, uniform, mean 4),
    because a real denoising step does not resolve the same count every time
    and a metronome reads as synthetic. A guard absorbs a would-be final burst
    of one so the sequence never ends on a lone flicker.
  - The scatter order is a **seeded Fisher-Yates shuffle**
    (`glowPreviewShuffled` over `glowPreviewRandom`, a mulberry32). One stream
    drives both the order and the burst sizes, so the entire schedule follows
    from `GLOW_PREVIEW_SEED` and is identical on every replay, reload, and
    machine. Repeatability is the requirement, so that two settings are
    compared against each other rather than against whichever draw looked
    better; randomness is only how the shape is reached.
  - **This replaced a golden-ratio stride, and the reason is worth keeping.**
    `order[i] = (i * stride) % count` covers the block evenly, which is what a
    low-discrepancy sequence is *for*, but it is an arithmetic progression: the
    shipped 38-word/stride-23 version had **exactly one distinct step size
    across the whole sequence**. Wrapped onto four rendered lines that read as a
    rigid motif marching across the rows (line 1 lit as 0,8,1,9,2,10,3,4,5,6,7;
    68% of within-line steps moved rightward), which is the opposite of
    scattered. No stride fixes it, because every stride is an arithmetic
    progression. **Even coverage and looking unordered are different
    properties**, and the stride optimised for the wrong one.
  - `GLOW_PREVIEW_SEED` is **chosen by measurement, so do not treat it as
    arbitrary and do not "tidy" it to 1.** Across 800 candidates, seed 662 is
    one of eleven that put no two adjacent words in the same burst while
    landing on nine ticks, and it has the widest spread of those: 0.83 of the
    words in a burst fall on different rendered lines, with a rightward bias of
    exactly 50%, meaning none. Editing the copy changes the word count and
    invalidates all of it, so rescan rather than assuming it carried over.
  - Sliders replay on a **180ms debounce** (`replayGlowPreviewSoon`), not on
    `input`. A 3.4s sequence restarted on every event of a drag would never get
    past its first word. Discrete actions (click, class change, Reset) play
    immediately.
- **Sub-settings are grouped, not hidden.** A sub-row is indented and, when its
  parent preference is off, **dimmed and inert** rather than hidden. That is
  what makes the indent mean anything, and it also converted the existing
  **Mode** row (under Render diffusion-style text) from `hidden` to the same
  treatment. The group's closing hairline moved to the `border-top` of whatever
  preference follows the last sub-row
  ([settings.css](src/web/static/settings.css)). That form was chosen to avoid
  both `:has()` (thin support in the WebKitGTK desktop window) and an "I am
  last" class in the markup that would rot as rows are added; a group that ends
  the panel correctly gets no line at all. `.settings-row-disabled` already
  existed in `style.css` but was dead: it only covered `.toggle-switch`, so it
  grew to cover the custom select and range inputs, plus an override cancelling
  `.custom-select.disabled`'s own 0.5 opacity, which would otherwise compound
  with the row's 0.4 down to 0.2 and read as broken.
- **The load bar sweeps before it fills.** The gap the maintainer noticed
  between the loading UI appearing and the bar starting is real work, not a
  rendering delay: the worker process spawning, importing torch and
  transformers in its own virtualenv, uvicorn coming up so `/health` answers at
  all, and only then `load_target_bytes` reading the shard headers. None of it
  is measurable, so the fix is **not** a bar parked at 0%, which reads as hung
  and is exactly what `load_progress.py` refuses to draw. It is a sweeping
  track plus the honest label **Starting worker**. No backend change was
  needed: `starting` was already set in `activate()`
  ([server.py](src/web/server.py)) and already returned as `state` by
  `/api/models/activation`; the shared reducer simply fell through to its
  generic branch.
  - `overlaysActivationProgress` went from a boolean `determinate` to a
    three-way `mode`: `hidden`, `sweep`, `fill`. `idle`, `error`, and any
    unrecognised state map to `hidden`, which preserves today's behavior for
    them and keeps the sweep off a page that is not loading anything.
  - **Know this before touching the finish path.** Both
    `finishLoadingProgress` ([app.js](src/web/static/app.js)) and
    `finishActivationProgress` ([menu.js](src/web/static/menu.js)) read
    `container.hidden` to mean "a bar was never on screen", which is how an
    unmeasurable checkpoint used to finish with no bar at all. With a track
    present for every activation, that check now only guards an overlay that
    was never raised, and every activation closes on a brief full bar. The
    menu gains the most: it used to hide the entire row, label included, so it
    showed *nothing* during the gap.
  - While sweeping, the width belongs to the CSS class and the inline width is
    removed, because the sweep animates `transform` while a measured bar
    animates `width`. Handing the property back on the way out is also what
    makes the switch to a real reading one eased move instead of a jump.
  - **`switchFailed` now tears the track down** rather than leaving the next
    switch to re-sync it. Worth knowing why, because the reasoning is not
    local: `#loading-overlay.hidden` is `opacity: 0`, **not** `display: none`,
    so the element stays in the layout tree and a sweep left on it would keep
    animating unseen for the rest of the session. Nothing was visible and it
    self-healed on the next switch, but a permanently running compositor
    animation is not something to leave lying around. It resets by calling
    `setLoadingProgress("idle", null)`, which routes through the reducer's
    `hidden` mode, so that mode is now load-bearing instead of purely
    defensive. The menu needs no equivalent: it hides via the `hidden`
    *attribute*, and `display: none` does stop animations.
  - `switchModel` seeds the track with `starting` rather than `loading`, so the
    opening frame already reads "Starting worker" instead of saying "Loading"
    for one poll interval and then correcting itself.
- **`prefersReducedMotion` is shared now.** It lived twice, identically, in
  `app.js` and inside `menu.js`'s IIFE, and the preview needed a third. It
  moved to [overlays.js](src/web/static/overlays.js) and both copies were
  deleted; every page loads `overlays.js` ahead of its own script, so all
  eleven call sites kept working untouched. Unprefixed, unlike its neighbours,
  because it is a predicate about the environment rather than part of the
  overlay model.
- **Two CSS corrections.** The thin bar above the Analytics **Original /
  Edited** crossfade was a **cascade leak**, not an Analytics style at all: the
  row kept `margin-top`, `padding-top`, and `border-top` from `style.css` after
  it moved onto the Token overlay heading row, and those are still correct on
  the generator, where it stacks on its own line. So they are zeroed in
  [analytics.css](src/web/static/analytics.css) rather than removed at the
  source. And the **pager arrows** now read by brightness rather than hue
  (`--text-primary` actionable, `--text-dim` not, hover deepening the wash
  instead of changing colour). The accent green was on the *disabled* arrow,
  which had it backwards twice over: the dead control was both the brightest
  thing in the row and camouflaged against the green chart title beside it.
  Nothing is lost by dropping it as a position indicator, since both pagers are
  two-page and the page is always named next to them.

**Previous pass: the reveal signal, the birth glow, and Tokens per Second.**
One missing piece of data gated both features, so it landed first and
everything else hangs off it. Checklist items 58 to 66 carry its GPU and
display work.

- **`ruff` is installed and configured.** There was no `pyproject.toml`,
  `ruff.toml`, or `setup.cfg` anywhere in the repo, and `black` was pinned with
  no config, so it defaulted to 88 columns against code hand-wrapped to 70: a
  single `black .` would have reflowed every Python file in the tree. The new
  config-only [pyproject.toml](pyproject.toml) pins both tools to 70 and selects
  `C901` and `PLR1702`, the two gates the style rules name. `PLR1702` is still a
  preview rule, so `preview` is paired with `explicit-preview-rules` to enable
  only the preview rules actually named in `select` rather than every unstable
  rule ruff ships. **Baseline: 159 findings**, all pre-existing and none fixed
  (129 `E501`, mostly 71 to 73 column overruns concentrated in
  `llada_sampler.py`; 10 `PLR1702`; 4 `C901`, the worst being
  `create_worker_app` at 23 and `_save_run_blocking` at 21). The count after
  this session's work is also 159, so nothing here added to it.
- **Frames now say which positions were born.** `revealed` lists the positions
  that became resolved in that frame and had not been resolved earlier in the
  same canvas. The monotonicity is the load-bearing part and is documented once
  in [reveal.py](src/inference/reveal.py): a signal that said "resolved right
  now" would re-fire every frame for every settled token, and one that said
  "changed since last frame" would flicker on DiffusionGemma, whose draft tokens
  churn until they stabilize. The diff is a pure helper the caller folds back
  into its own set, so the samplers keep a single place where that state moves,
  and it unit tests without a model or a GPU.
  - LLaDA ([streaming_sampler.py](src/inference/streaming_sampler.py)) seeds an
    empty set in `streaming_generate`, but `streaming_resume` seeds from the
    canvas it inherited. Starting it empty there would have reported the whole
    surviving prefix as newborn on frame 0.
  - DiffusionGemma ([dgemma_sampler.py](src/inference/dgemma_sampler.py)) keeps
    the set per canvas and clears it beside `self._prev` in `put`, since the
    next canvas restarts from fresh noise.
  - SmolLM3 ([ar_sampler.py](src/inference/ar_sampler.py)) needs no state:
    decoding is strictly left to right, so the frame that grows the sequence to
    *n* tokens is the one that produced position *n-1*.
- **Live generation renders tokens, not characters, and reuses the nodes.**
  This is a performance win that happens to unblock the glow. `renderFrame`
  built one span per *character* and tore the whole output down every frame,
  which at LLaDA's default `gen_length` of 160 is roughly 640 inline boxes laid
  out from scratch per step; `renderLiveFrame` keeps a constant ~160 spans and
  writes only where something differs. `overlaysSyncTokenSpan`
  ([overlays.js](src/web/static/overlays.js)) holds the property application the
  builder used to inline, so both paths agree on what a token span looks like,
  and every write there is guarded by a read because an unconditional
  `textContent` assignment relayouts the block even when the text is identical.
  Reuse is validated by reading the first span's `parentNode` back rather than
  by trusting every other render path to announce itself. `renderFrame` stays as
  the fallback for models that send no token metadata.
- **Newly denoised tokens flash.** `#output-area.live-tokens .token-span
  [data-born]` fades a constant-blur shadow's alpha over 0.5s with no fade in.
  Constant blur because animating the radius re-rasterizes a different-sized
  blur every tick, which is the shape the `.token-mask` scroll comment warns
  about. It is keyed off an **attribute** rather than a class specifically
  because `overlaysSyncTokenSpan` owns `className` and would otherwise cut a
  glow short the moment its position changed. Concurrency is capped at 48 (LLaDA
  at `steps=8` reveals about 20 at once) and `animationend` is delegated to the
  container. Setting `tokenBirthGlow`, default on, under Settings > Appearance.
- **Tokens per Second, in the footer and in Analytics.** `#status-tps` is the
  one interactive footer readout: click or press Enter to swap the run average
  for the last step, persisted in the settings blob as `tpsMode`. **Elapsed was
  wrong and is fixed in the same place**: the footer printed the raw
  `data.elapsed`, which is segment-local, so it jumped backwards after an edit;
  it now reads the cumulative `perFrameElapsed` tail. In Analytics the Timing
  slot became two pages behind a pager, Elapsed Time and Tokens per Second,
  because they are the same two numbers read two ways and did not each deserve a
  chart slot. **No new storage and no backfill gap**: a masked token renders as
  exactly one mask glyph, so `compute_convergence`'s `mask_count` already is a
  token count, and every run ever saved has its frame timings. The rate is
  clamped at zero for DiffusionGemma, whose mask count can rise between drafts.
  The pre-edit comparison is offered for autoregressive runs only, since a saved
  run keeps the original's timings but not its canvas history, and a rate needs
  both.
- **Analytics shows both elapsed totals.** Closes the outstanding item: an
  edited run now reports the original's end-to-end time and its own separately,
  rather than one combined figure that hid whether the intervention cost time.

**Previous pass: resident navigation, picker flip, and load-bar corrections.**
The four items the maintainer left open after the polish pass below. `pytest`,
`node --check`, ReadLints, and the 70-column audit are clean; two throwaway
Node harnesses (7 checks on the drop-up geometry, 9 on the activation reducer)
ran green and were deleted. Checklist items 48 to 57 carry the GPU and display
work.

- **Selecting the resident model no longer pretends to load it.** The server
  has always treated that activation as a no-op
  ([server.py](src/web/server.py), `activate`), so the only real damage was the
  clear the polish pass had just added: `selectModel` wiped the run snapshot
  unconditionally, including on the path that spawns nothing. The menu now
  reads `active_device` off `/api/models` (it was being discarded) and asks
  *Go back to the Generation page?* instead. Only an AR row can name a
  placement the resident worker is not using, so that is the only case
  `isResidentSelection` compares a device on; everything else offers one
  placement, which sidesteps mirroring the server's resolution of a
  device-less request. The POST still goes out, and its `state` is the
  discriminator: `ready` means the no-op fired and this is navigation,
  anything else means the worker had died since the menu was drawn and the
  POST respawned it, so the loading UI still comes up.
- **Dropdowns flip up when they would be clipped.** The rows the maintainer
  saw occluded were the Overlay picker's option list, not the drawer.
  `#output-section` hides its overflow and `.custom-select-list` was pinned to
  `top: calc(100% + 4px)`, so a drawer dragged low had nowhere to put its
  choices. `open()` now measures after unhiding (a hidden list has no height)
  against the nearest ancestor that would clip it, and flips only when the
  list does not fit below **and** there is more room above, so it never trades
  one clipped list for a worse one. Shared by every dropdown in the app, and
  Analytics loads `style.css`, so one rule covers both pages.
- **DiffusionGemma's bar reserves a tail for the copy.** It is the only worker
  that unpickles the whole state dict into anonymous RAM before copying any of
  it, so RSS hit the target exactly while the copy had not started, leaving the
  bar at 100% for the entire second half of the wait. See "The reserved tail"
  below for why clamping was not enough and why the parameter is opt-in.
- **The bar climbs and finishes.** Two causes, neither of them a fast load.
  The sampler's closing 100% never reaches the browser, because the worker
  reaches ready in the same breath and `_apply_health` drops progress on that
  transition; the reducer now names the `ready` state and both pages hold the
  full bar for `OVERLAYS_LOAD_COMPLETE_HOLD_MS` before navigating. And three
  polls were stacked at 500ms, so the browser was up to a second stale at a
  500ms refresh; the supervisor and both clients now read at 250ms. The
  supervisor keeps 500ms until the worker first answers, since every poll
  before that is a refused connection during its torch import.

**Earlier this session: polish, plus a determinate model-load bar.** `pytest` (133/133,
49 of them new), `node --check`, ReadLints, and the 70-column audit are clean.
Two throwaway Node harnesses covered the parts pytest cannot reach, both run
green and then deleted: 26 checks on the drawer drag (clamping, the swallowed
click, the threshold boundary, the open-drawer lockout, resize rescue, corrupt
storage) and 19 on the activation reducer. GPU and display work is on the
checklist.

Seven small items and one large one.

- **A model switch now starts on a blank canvas.** Two independent holes.
  `saveSessionState` recorded only the model id, so GPU-to-CPU on the same
  model matched on restore and the old run came back; `device` is now part of
  that identity (a snapshot predating the key is read as matching, so nobody
  loses an in-flight run to the upgrade). That alone cannot fix switching away
  and back, which lands on a genuinely matching pair, so both activation paths
  now clear the snapshot on their way out. The key and its clear moved to
  `overlaysClearLastRun` in [overlays.js](src/web/static/overlays.js) because
  `menu.js` is an IIFE and could not see `app.js`'s copy.
- **The placeholder names the model.** "Diffusion output will appear here" was
  simply wrong under SmolLM3. The static string in `index.html` stays generic
  since it paints before `/api/models` resolves, which is also the fallback for
  boot's failure path.
- **Cosmetics.** The Analytics "Edited" check lost its dot-pattern stroke (the
  texture only muddied a 16px glyph) along with the now-orphaned `<pattern>`,
  `.svg-defs` wrapper, and three CSS rules. `#prompt-row label` gained 3px of
  `margin-bottom`, which is what sets the clearance under the
  absolutely-positioned `#prompt-history`. The loading overlay's "~30 seconds"
  guess is gone, replaced by the real bar below.
- **Docs read as an LLM visualizer.** `xAI` to `XAI` in all 14 spots (the
  desktop entry already said "LLM XAI Visualizer"), and the framing in
  `README.md`, `AGENTS.md`, this file, and the About modal now leads with
  language models generally while still saying the depth is in discrete
  diffusion.
- **The collapsed drawer drags vertically.** One shared helper,
  `overlaysMakeDrawerDraggable`, wired from both pages. It moves `top`, not
  `transform`, because the group already animates `transform` for its slide and
  the box is absolutely positioned. It owns the handle's **click** as well as
  its drag: at the target node, listeners fire in registration order regardless
  of the capture flag, so a separate swallow-the-click listener would have been
  a silent ordering dependency. Persisted per page.
- **AR Alternatives defaults on.** It gates the hover popover and What If?
  entirely, so off by default hid the model's two most interesting affordances
  behind a toggle. `smollm3_worker` now reads every absent-key fallback from the
  registry spec (`_spec_default`) instead of keeping a second copy of each
  default, which is what would have let the two disagree. The sampler's own
  Python defaults stay `False`; only the UI default moved.
- **A real bar for the model load.** See "What the load bar samples" below.

**Earlier this session: the status message row.** `pytest`, `node --check`,
ReadLints, and the 70-column audit were clean at the time (84/84, five of them
new for `_display_run_path`). The chip lifecycle was exercised head-on with a throwaway Node harness
that located the real source region in `app.js` by content markers, evaluated
it against a stub DOM and a fake clock, and asserted the coexistence case,
independent dot timers, retirement, the cap, and timer leaks. The fourth pass
re-ran it with the motion states added, checking that a dismissed chip takes
`is-leaving` and not merely the loss of `is-visible`, and that `statusRowReflow`
emits its writes in the order FLIP requires (`transition: none`, the inverted
offset, a forced reflow, then release) with the offset equal to the width the
arriving chip consumed. All passed and the harness was deleted. What the
harness cannot see is layout, easing, and copy, so checklist items 38 to 47
carry the weight.

`#status-message` was a single overwritten span. Two operations at once lost
one of them: entering **What If?** on an unsaved run auto-saves the original,
and picking a candidate before that POST lands used to replace "Saving run"
with "Resuming", so the save became invisible while still holding the confirm
and retry buttons disabled.

Three more passes followed once it could be seen on screen, all folded into the
notes below. The second made chips quiet, gave messages a subject, restored
the ellipsis in cycle mode, and gave the box real width. The third replaced
the upward column with a leftward row, and split each chip's word and
ellipsis into separate spans, which is what properly fixed cycle mode. The
fourth was motion: chips rise in and step aside going out, neighbours ease
rather than snap, and saved paths stopped depending on which branch wrote
them. See the three "What the Nth pass changed" sections after the two traps.

- **The split is by lifetime, not by category.** Work in flight gets a
  transient chip; the run's *resting state* stays in the footer. That is the
  part worth carrying forward, because the resting line is also what
  `saveSessionState` captures ([app.js](src/web/static/app.js),
  `statusMessage.textContent`), so drawing the line there meant session
  persistence needed **no changes at all**. Chips can expire without taking
  any record with them, and an error that scrolls away is still on screen
  underneath. The second pass sharpened this into a rule: a chip says only
  what is *happening* and never what *happened*, so every outcome has exactly
  one home.
- **The enabling refactor had its precedent one function above it.**
  `denoiseReveal` already stored its timer on the element, with a comment
  saying it did so "so independent targets can animate simultaneously without
  one cancelling the other". But `startStatusDots` / `startStatusCycle` /
  `stopStatusDots` kept module-level `statusDotsTimer` / `statusCycleTimer` /
  `statusDotsCount`, so two chips could not animate their own dots. Moving
  those onto the element and giving the three functions an `el` parameter is
  what the rest composes on.
- **Chips live on the footer's own row.** `#status-message` sits inside
  `#status-stack`, a right-anchored flex row, and chips are inserted before
  it, so they extend leftward from the resting line without ever displacing
  it. (It began as a bottom-anchored absolute column growing upward; see
  "What the third pass changed".)
- **Bounded at four rather than scrollable.** The real ceiling is two (one
  run; `saveRun` guards itself with `isSaving`), so a scroll region would
  have been dead code. The cap exists so an unforeseen caller cannot push an
  unbounded run of chips out under the fade, costing layout while unreadable.

### What the load bar samples, and why it is shaped that way

[load_progress.py](src/inference/load_progress.py) is the companion to
`hf_download.py`: one covers getting weights onto disk, the other covers
reading them into memory, which is often the longer wait and had no readout at
all. There is no progress hook to borrow (`from_pretrained`'s shard counter is
coarse, differs across the three pinned `transformers` versions, and says
nothing about the host-to-device copy), so it samples memory counters the same
way `hf_download` samples the cache directory. Four decisions carry the design.

**One fraction, not two phases.** LLaDA loads with `device_map="auto"` on CUDA,
so accelerate streams shards straight to the GPU and RSS barely moves, while
SmolLM3 memory-maps its shards, so its two counters climb over the same stretch
rather than one after the other. A sequential CPU-then-GPU bar would sit at
zero through half of one of them. The reading is
`max(rss_delta, device_delta)` against a single target, and the **stage label
follows whichever counter is being reported**: any device bytes at all mean the
copy has started. (That rule was originally `on_device >= resident`, which kept
saying "weights" through nearly all of a copy that follows a full read into
RAM; see the reserved tail below.) Both counters are baselined, so a fresh
worker starts at zero on each.

**Except one shape, which needs a reserved tail.** DiffusionGemma is a pickled
state dict, so `torch.load` materializes every byte in anonymous RAM before one
is copied across. Its RSS reaches the target *exactly* while the copy has not
begun, which parked the bar at 100% for the whole second half of the wait.
Such a load passes a `host_stage_ceiling`, which compresses the read into
`[0, ceiling]` and scales the copy into what is left. **Clamping the combined
reading is not enough**, and this is the trap: the monotonic floor would then
jump the reserved tail in a single step the moment the copy started, since RSS
already accounts for the whole target. The split is by expected *time*, not by
bytes, because reading ~17 GiB off disk dominates the copy across PCIe and an
even split would park the bar at half and then sprint. The parameter is opt-in
and defaults to `1.0`, at which the arithmetic reduces to the single-counter
one above, so the two loads that already tracked their wait are provably
untouched by it: only [dgemma_worker.py](src/backends/dgemma_worker.py) passes
a value.

**The target comes from the requested dtype, not the disk dtype.** LLaDA on CPU
passes `torch_dtype=None`, which means torch's default, which is fp32, so a
BF16 checkpoint takes twice its on-disk size in RAM. The worker passes
`torch.get_default_dtype()` in that case rather than `None`, or the bar would
stall at 50%. Targets come from `metadata.total_size` in the shard index (LLaDA
14.93 GiB / 6 shards, SmolLM3 5.73 GiB / 2, both uniform BF16), scaled by the
dtype ratio; DiffusionGemma's single packed NF4 `model_nf4.pt` is its own size
with no scaling.

**Unmeasurable means no bar, never a guessed one.** A mixed-dtype checkpoint,
an unreadable header, an unrecognized layout: all return target `0`, and the
client renders the phase label with a spinner. `_index_dtype` reads *every*
shard rather than sampling one, so a checkpoint keeping fp32 embeddings in a
separate shard is caught instead of scaled by the wrong ratio.

**The reading only goes up.** The CPU allocator hands pages back mid-load, and
a bar that walks backwards reads as broken even when nothing is wrong, so each
sample is floored at the previous peak. The floor is on the **fraction**, not
the byte count, which is what the two-segment case needs and is also the thing
the user actually sees; `loaded_bytes` stays honest about what is resident and
is diagnostics only (no client reads it).

Two smaller notes. The sampler runs on the helper thread and the **load stays on
the caller's**, the opposite of `download_with_progress`: moving a heavyweight
library-driven load between threads to gain a progress bar would trade real risk
for a cosmetic one, and reading two counters is the safe thing to relocate. And
the whole thing is a context manager, so the thread is joined and the bar
completed even when the load raises. That completing frame carries the stage
the load actually reached rather than naming one, or it would put "Moving to
GPU" on the last frame of a load that never touched a GPU.

The wiring is four layers: `sample_load_progress` writes the backend's
`load_progress`, `resolve_load_status` in
[worker_base.py](src/backends/worker_base.py) reads the dict's new `phase` key
to tell a load from a download (**absent means download**, since `hf_download`
predates the key), `_apply_health` in [server.py](src/web/server.py) now keeps
progress for `loading` instead of nulling it, and
`overlaysActivationProgress` reduces one poll to `{determinate, percent, label}`
for both the generator's overlay and the menu's inline bar. The generator also
polls on the **boot** path now (`startLoadProgressPoll`, driven from
`handleModelStatus`): that path raises the same overlay without going through
`switchModel`, so the first load of a session, reliably the slowest, was the one
load with no bar.

One thing that layering costs: the closing 100% the sampler emits **never
reaches the browser**, because the worker sets ready in the same breath and
`_apply_health` nulls progress on that transition. So the reducer names the
`ready` state itself, and both pages hold a full bar briefly before navigating.
Both gate that hold on whether a bar was on screen, read off the element rather
than tracked beside it, so an unmeasurable checkpoint that ran indeterminate
does not have one flash into existence at the end.

### Two things a future change here will trip over

**`resetStatus()` must stay footer-only.** It is called immediately before
`startRunStatus(editRunLabel(...))` in both `doSubstitute` and
`doGuidedResume`, which is precisely the moment a save may be in flight. If it
ever clears the stack, the exact bug this pass fixed comes straight back.
There is a comment on it saying so.

**A chip's slot is held until it fades, not until it is retired.**
`statusChips` holds chips that own a slot, and a departing chip still owns one
because it is still on screen; it leaves the list when dismissal *starts*,
which is what makes a late `statusRetire` a no-op rather than a resurrection.
A live chip persists indefinitely by design, since it stands for work still
running, so any new caller of `statusPush` owns the obligation to retire its
handle on every exit path, including failures. `startRunStatus` retires the
previous handle for exactly this reason.

### What the second pass changed

- **Chips went quiet.** Having them report outcomes put "Done" directly above
  "Done." and "Saved" above "Saved to results/...", which read as stutter and
  was the only thing that ever put a second line in that corner during
  ordinary use. A chip now just leaves when its work ends, and the footer
  filling in *is* the handoff. `statusResolve`, `STATUS_CHIP_HOLD_MS`,
  `_holdTimer`, and `.status-chip.is-error` all went with it; the five
  terminal sites call `statusRetire`. `endRunStatus` takes no arguments now.
- **Messages name their subject.** `saveRun` already computed `wasEdited` for
  the Edit Frames lock, so hoisting it above the push yields "Saving original
  run" / "Saving edited run" and a footer of "Saved <label> run to ...",
  which matters because entering What If or Edit Frames saves on your behalf.
  Resumes read "Running edit from frame X to Y" via `editRunLabel`; in
  `doGuidedResume` the endpoint is a single `resumeTarget` local shared with
  the request's `max_frames`, so the text and the wire cannot disagree.
- **Cycle mode gained the ellipsis.** It had been suppressed on the theory
  that re-diffusing the word was indicator enough, but that left the ellipsis
  present on two text settings and absent on the third with nothing on screen
  to explain it. First attempt made one pass reveal-then-tick with cycle mode
  repeating the pass; the third pass replaced that with two genuinely
  independent animations (below), which is what it should have been.
- **The stack got real width.** `#status-stack` was sized by
  `margin-left: auto`, but its only child is absolutely positioned, so the
  box was zero-wide: `max-width: 100%` resolved against nothing and a long
  saved path ran left underneath "Elapsed:". `flex: 1; min-width: 0` gives it
  the leftover footer width, and the message clamps with an ellipsis at the
  footer's own gap. Worth remembering if anything else is ever added to that
  row.

### What the third pass changed

- **The column became a row.** Chips now extend leftward from the resting
  message instead of stacking above it, which is what the maintainer wanted
  from the start; the vertical version was the one place the design and the
  build had diverged. `#status-stack-items` is gone: `#status-stack` is
  itself the flex row (`justify-content: flex-end`, `overflow: hidden`), so
  the absolute positioning, the reserved `min-height`, and the `z-index` that
  existed only to let a column escape the footer all went with it.
- **The separators need no JavaScript.** Two messages abutting in the same
  accent green read as one sentence, so each item takes a middle dot from
  whatever precedes it. Chips are inserted *directly before* the message, so
  `.status-chip + #status-message:not(:empty)` matches exactly when a chip is
  up, and `.status-chip + .status-chip` covers the rest. The subtlety worth
  keeping: a separator belongs to the item on the *right* of the gap, so it
  has to track the chip on its left or it holds full strength through that
  chip's fade and then blinks out. Keying it on
  `.status-chip:not(.is-visible) + ...` handles both directions at once,
  fading it in with an arriving chip too.
- **The message never yields, but it still truncates.** Both it and the chips
  are `flex-shrink: 0`, so overflow spills off the *left* edge, where the
  clip and the mask fade are, making the oldest chip the thing that gives
  way. `max-width: 100%` on the message then caps it against the row, so it
  ellipsizes only when it alone is too long. Do not give it `flex-shrink: 1`
  to "make room": with the chips fixed it would become the only shrinkable
  item and get squeezed first, which is backwards. All of this is cosmetic to
  the data, since `saveSessionState` persists `textContent`.
- **The fade is free of JavaScript too.** A left-edge `mask-image` gradient
  can stay on permanently because the row is right-anchored, so content only
  reaches the faded band when it genuinely overflows. Shipped with the
  `-webkit-` prefix as well, for the WebKitGTK desktop window.
- **The dots got their own span, and their own timer.** The chip is now
  `.status-chip-text` plus `.status-chip-dots`, the latter `width: calc(3ch +
  0.12em)`: three monospace characters plus the footer's `letter-spacing`
  applied to each. Two things follow. The chip is one fixed width for life,
  which a right-anchored row needs (any width change shoves every chip to its
  left), and the word and the ellipsis can animate independently, so the dots
  tick continuously in *all* three text modes. That is what actually fixed
  cycle mode: the old collapse was never the dot count, it was
  `denoiseReveal` writing the bare word into the shared text node and
  dropping the padding for the length of each re-diffusion.

### What the fourth pass changed

The row was correct by then and still read badly, because every motion in it
was either wrong-way or missing. Three fixes, one of them in the backend.

- **A chip has three states, not two.** Sharing one rule between "not yet
  entered" and "leaving" meant a dismissed chip transitioned *back* to its
  entrance offset. With the entrance coming from the right, that walked the
  fading chip into the resting line and printed the two over each other, which
  nothing could catch at runtime: transforms move no layout, so the message
  never knew. `.is-leaving` is now its own class, drifting left, away from the
  line taking over from it, and shortened to 150ms because by then the footer
  already carries the outcome. `STATUS_CHIP_FADE_MS` matches it, as always.
- **The entrance rises from the window edge.** `--status-rise: 24px` on
  `#status-stack` is the footer's 8px bottom padding plus `#app`'s 16px, and it
  feeds both the chip's starting offset and the clip, so the two cannot drift
  apart. Getting the rise required dropping `overflow: hidden` for
  `clip-path: inset(0 0 calc(-1 * var(--status-rise)) 0)`: overflow can only
  clip both axes, and the left/right clamp is the half that has to stay.
  (It is also why the row's baseline alignment survived; an `overflow` value
  makes the box a scroll container, whose baseline is its margin edge rather
  than its text.) Negative insets are well-supported and are the standard way
  to clip on three sides. The offsets live on the `translate` longhand so
  `transform` stays free for the FLIP below; the two compose rather than
  overwrite.
- **`statusRowReflow` eases every sideways jump.** Flex has no transition for
  "the item beside me changed width", so a chip arriving, a chip's node
  finally leaving, and the resting line filling in all snapped their
  neighbours across in one frame. It is a plain first-last-invert-play:
  measure, mutate, hand each moved chip its former position as a `transform`,
  force a reflow, release. Two details are load-bearing. It reads its set from
  the **DOM**, not `statusChips`, because a chip midway through its fade has
  already left that list but still holds row width and is the likeliest thing
  to be shoved; and `.is-leaving` therefore has to keep `transform` in its
  transition list, or that chip's slide would snap. It no-ops under
  `prefersReducedMotion`. Wired at six sites: the insert, the deferred
  removal, and the five terminal messages.
- **Saved paths agree between branches.** The save endpoint reaches its folder
  two ways, and a fresh save stayed relative while an in-place update went
  through `_existing_run_dir`, which must `resolve()` for its traversal guard.
  So the same message read `results/...` after one save and
  `/home/you/.../results/...` after the next, purely as an artifact of which
  branch ran. `_display_run_path` normalizes at the one point both branches
  meet, leaving the guard untouched, and falls back to the full path when the
  run is genuinely outside the repo (a symlinked `results`, or a server
  started elsewhere), which is an operating condition rather than a broken
  invariant. Covered by `tests/web/test_display_run_path.py`, including the
  case that asserts the two branches produce one string.

**Previous pass: three finishing touches on the comparison surfaces.** Frontend
only; `pytest` (79/79), `node --check`, ReadLints, and the 70-column audit are
clean. `withAlpha` was checked head-on in Node against all four hues plus the
zero-alpha case a dark pin produces, since a bad hex parse would fail silently
as `rgba(NaN, ...)`. All three items are visual and need eyes; checklist items
34 to 37 cover them.

- **The generator's entropy profile now marks its edits.** Analytics had the
  marker since the Entropy by Position chart landed; the generator's strip did
  not, so a branch's forcing point was invisible there. Ported as two passes
  around the existing series draw: `drawEntropyProfileEditTint` under the bars
  and `drawEntropyProfileEditLines` over them, with the hover glow last, which
  reproduces the stacking Analytics gets for free from plugin order (the
  pointer's guide lays over the edit tint, not under it). Positions come from a
  flattened `editedProfilePositions()` rather than a single index, so
  sequential What If rounds each stay marked; the dash is centered at
  `pos * step + barWidth / 2` to match Analytics, and the tint is floored at
  2px because at a few hundred tokens a bar-width column is too thin to find.
  Worth knowing why this does not contradict the previous session: the *scrub*
  marker was removed from this strip on purpose because a standing neutral
  guide reads as an artifact when the bar opacity already says where you are.
  This one is a different statement. It names a position the run was
  intervened at, which is true wherever the cursor is, so it should persist.
  The comment above `drawEntropyProfile` was reworded to keep that distinction
  legible to whoever reads it next.
- **The line charts have their area fill back, as a band between the curves.**
  `fill: !paired` had disabled it for two-series charts, correctly: two washes
  to the axis stack over the shared prefix and read as a third color. A band
  (`fill: {target: 0}` on the branch) has neither problem and is strictly more
  informative, since the runs share their prefix exactly and the band is
  therefore empty until the edit. Colored by whichever run bounds the region
  from above (`above` the branch's hue, `below` the original's grey), which is
  a rule that needs no legend and, importantly, stays neutral across two
  charts that disagree about whether higher is good: higher is *slower* on
  timing and *better* on confidence, so a good/bad palette would have to be
  inverted between them.
- **The tooltip swatch fix from the previous session was only half done.**
  `lineLabelColor` fixed the fill but the swatch kept a colored ring with a
  white sliver inside it. Two causes: Chart.js resolves the swatch stroke as
  `borderWidth || 1`, so asking for zero still strokes a pixel, and the white
  `multiKeyBackground` shows through because the stroke is centered on a
  one-pixel inset and covers only half of it. Transparent `borderColor` plus
  a global transparent `multiKeyBackground` leaves exactly the inset fill.
  The global default also cleans up the entropy chart, whose swatches only
  looked right because their default border is a near-black invisible against
  the tooltip.

### Where the band's alpha lives, and why it is not canvas state

This is the one non-obvious thing in the pass. `seriesBlendPlugin` dims a line
by setting `globalAlpha` on `beforeDatasetDraw`, and the natural instinct is to
let the fill ride along. It cannot: **Filler is a globally registered plugin**,
and `notifyPlugins` walks registered plugins before a chart's inline ones, so
Filler's `beforeDatasetDraw` always runs *first* and would draw the band before
any alpha is set. So the alpha is baked into the color instead, by a scriptable
`fill` (`compareBandFill`) returning `min` of the two series alphas times the
base wash. `min` because the band describes a relationship: bounded by a line
you cannot see, it is a smear with no reading in it.

Scriptables are re-resolved on `chart.update`, and every path that moves a pin
(`handleComparePinClick`) or the scrub (`updateLineCharts`) already calls
`chart.update("none")`, so there is no separate invalidation to keep in sync.
If a future change moves blend state somewhere that does *not* update the
chart, the band will stick at its last alpha; that is the failure mode to look
for.

**Previous pass: the generator crossfade and two-layer token stack.** Frontend
only, so `pytest` (79/79) is a regression check; `node --check`, ReadLints, and
the 70-column audit are clean. The shared span builder was additionally
exercised head-on with a throwaway Node harness (a stub `document`,
`vm.runInThisContext` on `overlays.js`, 20 assertions covering the no-hook
Analytics path, each generator hook, and the null-token guard); all passed and
the harness was deleted. Nothing needing a display was exercised; checklist
items 28 to 33 at the bottom cover them.

This closes the last structural gap between the two pages. The generator has
had the layered diff since the counterfactual overlay landed, but its other
four overlays stayed single-layer, so a branch could only be compared against
its original from *inside* Diff. It also still built its token spans inline,
the one place in the app not going through `overlays.js`.

- **A `#run-blend-row` below the scrubber**, sibling to `#diff-overlay-controls`
  and mutually exclusive with it: whichever row governs the layers currently on
  screen is the one that shows. Placed there rather than above the output area
  (where Analytics puts its copy) because the generator's output top-right is
  already occupied by the overlay drawer, and the diff sliders set the
  precedent that blend controls live under the scrubber.
- **`runBlendActive()` is `diffAvailable() && remaskMode === null`**, and that
  gate is what makes the stack *safe* rather than merely hidden. The obvious
  hazard with two stacked runs is clicking a token in the one you cannot edit.
  It cannot happen: `token-clickable` requires `remaskMode === "edit"` and
  `token-substitutable` requires `substitutionMode`, which implies
  `remaskMode === "substitute"`. Both are excluded by the same gate, so no
  interactive affordance can ever render on a stacked layer. No extra guard
  was needed, which is why the gate is worth keeping stated in one predicate.
- **Layer opacity is restyled in place on drag** (`applyRunBlendToLayers`),
  never rebuilt. Several hundred spans per slider step would also drop the
  candidate popover mid-drag. Same reasoning, same shape, as Analytics'
  `applyTokenLayerBlend`.
- **One span builder for the whole app.** `overlaysBuildTokenSpan` gained three
  optional callbacks, all defaulting to today's behavior so Analytics passes
  none of them: `maskedFor` (draw the glyph over a resolved token, for a remask
  selection), `classFor` (`token-remasked` / `token-clickable` /
  `token-substitutable`), and `opacityFor` (masks graded by the model's live
  predicted confidence, which Analytics never needed because its masks are
  flat). `maskedFor` is deliberately consulted *only* for a token that exists,
  so a hook can add masking but never strip it off a hole and leave `tok.t`
  read from null.
  - `applyTokenColor` both tinted a span and appended to its tooltip, which the
    two-callback contract cannot express. Split into a pure `tokenColorAt` and
    `tokenTitleExtra`, mirroring `overlayColorFn` / `overlayTitleFn`. (The
    tooltip half is gone; the metrics-strip pass replaced `tokenTitleExtra`
    with `metricsExtra`, read at hover time.)
  - `.token-remasked` (`style.css`) is declared after `.token-mask` at equal
    specificity, so a span now carrying both still renders orange. No CSS
    change was needed.
- **Commit steps are memoized per run.** `originalCommitSteps` sits beside
  `commitSteps`, and both plus `diffData` are cleared by a single
  `invalidateRunMemos()` replacing five duplicated two-line resets. Without
  this the ghost layer would have been painted from the *branch's* settle
  schedule and misreported every position past the edit. Only Commit Order
  needed it: Heatmap and Entropy read the token's own fields and are per-layer
  correct for free.
- **The entropy profile carries both runs**, mixed by the same slider, with the
  pre-edit columns underneath. It steps off the longer of the two
  (`entropyProfileColumns`), and that helper is shared with
  `entropyProfilePosition` so the drawing and the pointer-to-position inverse
  cannot disagree about the step. The glow and the nats readout follow
  `runBlendFavorsOriginal()`, as does the candidate popover's opening page
  (`defaultAltsPage`, ported from Analytics).
  - `onRunBlendInput` calls `updateEntropyProfileVisibility()` rather than
    `drawEntropyProfile()` directly. The latter would un-hide the strip for a
    run with no entropy at all, since `entropyProfileValues` returns a 0 per
    token rather than an empty array.
- **Deliberately not persisted:** the crossfade resets to Edited on every
  `activateScrubber`, so a resumed branch opens on itself and a session restore
  starts fresh. Matches how Analytics resets per run open. Change
  `resetRunBlend` if that turns out to be the wrong call.

**Previous pass: the line-chart comparison layer.** Frontend only, so `pytest`
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
  `token-mask` / `token-resolved`, `data-pos`, and (at the time) a
  caller-supplied `titleFor(index, token)`, since removed with the native
  tooltip. A null token renders as the mask glyph instead of
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

**Deferred at the time:** the generator's own crossfade and two-layer stack,
shipped in the latest pass above.

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

This session shipped the `results/` rename, all of **AR Phase C**, and the
passes listed under "Recently shipped". Everything through checklist item 66 has
been validated on hardware by the maintainer; **items 67 to 79 have not**, so
they are the first task. After that, the next candidate is the **What If? typed
token** (deliberated already, decisions recorded below), then Mamba-3, then
extending entropy / top-k to the diffusion models. Deliberate each in Ask mode
before Plan.

**Deferred with its decisions already settled: the What If? typed token and an
AR Top-K knob.** This was deliberated with the maintainer alongside the metrics
strip and split off so the strip could ship on its own. Nothing below is open;
it is ready to plan.

- **The shape.** Under the popover's five candidates, a text box reading *Enter
  your own*. Clicking it drops a green check and a red X out from behind its
  right edge (the status chips' motion, without the fade); clicking outside is
  a cancel. Confirming *solidifies* the entry into a row you then click to run,
  exactly like a candidate, with a small retry icon at its right to redo it.
- **Tokenization is a vocabulary lookup, not re-tokenization, and this is the
  key insight.** `streaming_substitute` keeps `prefix_ids[:position]` verbatim
  and continues decoding, so the sequence is never re-encoded and the boundary
  effects that make BPE context-sensitive cannot arise here. Resolving the
  typed text means encoding that string standalone and asking whether it is
  exactly one vocabulary entry. The tokenizer is already loaded
  (`self.tokenizer`, already used by `_capture_candidates`), and encoding a
  short string is microseconds, so a live preview as the user types is
  practical over the existing websocket.
- **Ship exactly one token first.** The blocker for multi-token is not
  validation, it is alignment: a substitution of length *n* shifts every
  downstream index, and `overlaysComputeDiff`, the position-indexed entropy
  chart, and the dashed edit marker are all index-based. Confirm stays disabled
  until the resolved count is exactly 1 and re-enables live as the text
  changes. When multi-token lands, cap on **resolved token count** (4) rather
  than characters, so the live preview is already computing the limit and the
  limit explains itself.
- **Show the pieces, not the why.** The split visualizer is endorsed: render
  each piece with its id and the count, hoverable, reusing the glow keyframes.
  Do not promise causation. BPE reaches a split through a merge sequence that
  fast tokenizers do not expose cheaply, so there is no short interpretable
  reason available; the maintainer agreed to leave that rabbit hole alone.
- **Name the tokenizer.** Surface which tokenizer produced the split, since it
  is the one honest answer to "why these pieces" that is available for free,
  and it is what makes the pieces comparable across models later. The worker
  already holds the object, so `type(self.tokenizer).__name__` plus
  `tokenizer.name_or_path` (and `is_fast`) costs nothing; the open question is
  only whether it rides the existing model capabilities payload, which every
  page already has, or the typed-token preview response, which is the only
  place it is needed today. Prefer capabilities: it is a property of the model,
  not of one keystroke, and putting it there is what lets the About or Help
  copy and any later tokenizer-specific view read it without a new round trip.
- **Color.** Single token in the app's normal green; multiple pieces in
  alternating tints; the warning orange `#ff9f1c` **only** when the count
  exceeds what is allowed, since that color already means edit/remask
  everywhere else and would misread on the ordinary case.
- **Leading space, decided by the token being replaced** rather than by a
  sentence-position heuristic: if the original token at that position began
  with a space, pre-seed one, and let a single backspace remove it. That is
  exact where a rule about "mid-sentence" would only be usually right.
- **`forced_conf` gets strictly better.** It currently comes from the captured
  candidate's stored probability, which a typed token has none of, and logits
  are not persisted. But `streaming_substitute` re-runs the prefix anyway, so
  it can read the typed token's **true** probability from that distribution.
  That is the most interesting number the feature produces: it can honestly
  report a wild choice as 0.003. `_validate_substitute`'s current rejection
  ("not among the captured candidates") is deliberate and needs a second,
  explicitly typed path rather than loosening.
- **Top-K for AR, with one correction to the framing.** Hugging Face applies
  top-k *before* top-p, so they compose as a truncation followed by a nucleus
  cut rather than as alternatives. The knob is still worth having. The
  configurable *capture* count (how many candidates the popover lists) is
  separately deferred until the typed token ships.

**Held deliberately, pending the sweep.** The ambitious version of the load
gap fix was to *measure* the unmeasurable phase: time the worker startup on
first run, persist the figure per model, and drive a real bar from the previous
run's timing. It was held on the maintainer's call unless the sweep proves
unsatisfying in practice. Two things to weigh if it ever comes back. First,
a cold start and a warm one differ by more than the estimate would tolerate
(page cache, and whether the venv's imports are resident), so the bar would
routinely stall at 90% or finish early, which is worse than admitting there is
no number. Second, it would put per-model timing state in
`results/ui_state.json`, which currently holds only user preferences. The
cheaper middle ground, if the sweep alone reads as too vague, is to name the
sub-phase rather than to measure it: the worker already knows when it has
finished importing and when uvicorn is answering, so `starting` could split
into two or three labeled sweeps without inventing a percentage.

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
   under the scrubber and tracks the scrubber's frame, the metrics strip shows
   an `Entropy` value, hovering a token makes its profile column glow and swings
   the nats readout to it, and **no** What If button or hover popover appears.
2. **Alternatives on**: hovering a token opens the candidate popover with five
   rows, sane probabilities, the chosen token highlighted, and readable
   whitespace/control candidates. It should sit **above** the token (a
   preference kept for readability now that the native tooltip it originally
   dodged is gone) and flip **below** for a token near the top of the canvas,
   where sitting above would put it over the metrics strip. Reaching into the
   popover to click a candidate should keep it open and keep the column
   glowing.
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

The generator crossfade pass (all on the **generation** page, not Analytics):

28. **Nothing changed before a branch exists.** Generate a plain run of each
    kind (LLaDA, SmolLM3) and scrub through it in every overlay. No
    Original / Edited row should appear below the scrubber, and the tokens
    should look exactly as they did: masks fading up with confidence as you
    scrub a mid-run frame, Heatmap and Entropy and Commit Order colors
    unchanged, tooltips carrying the same lines. This is the regression check
    that matters most, since every span on the page now comes from the shared
    builder rather than the old inline one.
29. **The crossfade appears and blends.** Run SmolLM3 with Alternatives, use
    **What If?**, and confirm. The Original / Edited row should appear below
    the scrubber. Drag it: the text should fade between the two runs, and the
    entropy strip's columns should fade with it. Switch to **Heatmap** and
    drag to full Original: the colors should be the *original* run's
    confidence, not the branch's colors sitting under the original words.
    Repeat with a LLaDA Edit Frames resume.
30. **Pointer follows the opaque side.** With the slider left of center, hover
    a token past the divergence. The tooltip should describe the original
    run's token, and the candidate popover should open on its **Original**
    page. Drag right of center and hover the same position: both should flip
    to the branch. Then scrub to a mid-run frame and confirm the masks still
    grade by confidence in *both* layers.
31. **Edit modes still render one layer.** With a branch in place, click
    **Edit Frames** (LLaDA). The crossfade row must disappear, the tokens must
    go back to a single layer, and clicking one must still select it for
    remasking (orange, fully opaque). Exit and confirm the crossfade returns.
    Do the same with **What If?** armed on SmolLM3: the dotted underlines must
    be present and clickable, with no ghost layer behind them.
32. **Diff still owns its own controls.** Pick **Diff vs Original**: the
    crossfade row must hide and the two opacity sliders plus Difference blend
    must take its place, working as before. Switch back to any other overlay
    and confirm the crossfade returns at the value you left it.
33. **Commit Order colors each run by its own schedule.** On an edited LLaDA
    run, pick **Commit Order** and drag the crossfade fully left. The ghost
    layer's tints and its "Resolved at step" tooltips should describe the
    original run, which for positions past the edit means different values
    than the branch shows at full right. If the two look identical past the
    edit, `originalCommitSteps` is not being reached.
34. **The generator's entropy strip marks the edit.** Run SmolLM3 with
    **Alternatives** on, use **What If?** on a mid-sequence token, and let the
    branch finish. A dashed orange line over a faint tint must appear on the
    substituted position in the profile below the scrubber, matching the
    Analytics entropy chart's marker. Scrub the frame slider and drag the
    Original / Edited crossfade end to end: the marker must stay put through
    both. Hover the marked column and confirm the white hover glow draws *over*
    the orange tint rather than under it. Then run a second What If on a
    different position and confirm **both** markers show. On an unedited run
    the strip must have no marker at all.
35. **The band shows the cost of the edit.** Open an edited SmolLM3 or LLaDA
    run in Analytics. The timing and confidence charts should show a wash
    filling the gap between the grey solid line and the dashed branch, empty
    to the left of the edit and opening up to the right of it. Where the
    branch is higher the wash takes the chart's color (blue on timing, amber
    on confidence); where the original is higher it should be grey. Check both
    directions if you can produce them: a branch that is slower than the
    original gives blue, one that finishes faster gives grey.
36. **The band respects the pins and the scrub.** Unpin **1**, then **2** (one
    at a time): the band must vanish entirely with either, since it needs both
    edges. Re-pin both and drag the token view's Original / Edited crossfade:
    the band should fade with whichever line is fading out and come back as
    the charts ease home on release. It must not double-dim, that is, it
    should never look noticeably more transparent than the fainter of the two
    lines bounding it.
37. **Tooltip swatches are solid chips.** Hover any point on the timing,
    confidence, or convergence chart, and on the compare panel's charts. Each
    tooltip row's swatch must be a flat square of the series color with no
    colored ring around it and no white edge inside it. Check the entropy
    chart too: its swatches show the hovered bar's own ramp color and should
    be unchanged apart from losing the same white edge.
38. **The original bug is gone.** Generate a run and do **not** save it. Enter
    **What If?**, which auto-saves the original in the background, and pick a
    candidate immediately. Both *Saving original run* and *Running edit from
    frame N to end* must be up at the same time, side by side on one row
    separated by a faint dot, the resume nearest the resting message, each
    animating its own dots independently. Before this pass the save's message
    vanished. Let both finish: each should drift left and fade as its work
    ends, taking its separator with it and leaving *Done.* alone at the
    right.
39. **The single-message case looks unchanged.** Generate a plain run with
    nothing else happening. The *Running* text must sit on the same baseline
    as *Step* and *Elapsed*, at the far right where the status message has
    always sat, with nothing above or below it. This is the detail most
    likely to be a pixel off, since the row's alignment is no longer pinned
    by hand the way the old absolute column was.
40. **The footer keeps the record, and only the footer.** Save a completed
    run. The chip reads *Saving original run* and then disappears as the
    footer settles on *Saved original run to results/...*; at no point should
    two lines say the same thing. Navigate to Analytics and back: the footer
    line must still be there (it is what `saveSessionState` persists) and
    there should be no leftover chip. Then force a failure if you can (stop
    the server mid-save): the chip just goes, and the footer carries the full
    error text in red. Save an edited run too and confirm both the chip and
    the footer say *edited* rather than *original*, **and that both saves
    report the same shape of path**: `results/<run>`, never an absolute
    `/home/...` one. That asymmetry was the bug this pass fixed, and the
    edited save is the branch that used to show the long form.
41. **Nothing lingers or leaks.** Run several generate/save cycles in a row
    and confirm chips always drain, never pile up permanently, and never
    leave a half-faded ghost or an orphaned separator dot. Retry a run while
    one is going (Generate again after an error) and confirm the old chip
    goes rather than sitting there animating forever.
42. **The ellipsis animates in all three text modes, at a fixed width.** In
    Settings, cycle through the diffusion text effect (off, default, cycle)
    and start a run each time. The dots must tick in all three. In *Cycle*
    watch the word specifically: it re-diffuses every second or so, and the
    dots must keep ticking straight through that, with the message's left
    edge dead still. Any horizontal twitch as the word re-diffuses, or as the
    dot count passes through zero, means the fixed slot has been lost.
43. **Long messages clamp instead of colliding.** Save a run whose results
    path is long (a deep output directory, or just narrow the window). The
    footer line must truncate with an ellipsis at a small gap to the right of
    *Elapsed:*, never overlapping or sliding under it. Widen the window again
    and it should return to the full path. Note the boundary sits further
    right on diffusion runs, where the commit legend occupies that space.
44. **The row gives way at the left, not the right.** Narrow the window with
    two messages up (easiest during a What If auto-save) until they no longer
    fit. The oldest must fade out against the left edge rather than being cut
    with a hard vertical edge, and the resting message on the right must keep
    its full width throughout: it should never be the thing squeezed to make
    room for a chip.
45. **Chips rise in and step aside going out.** Start a run and watch the
    message appear: it should rise from below the row, out of the window's
    bottom edge, fading as it comes, not slide in from the right. Then watch
    one end. It must drift *left*, away from the resting line, and be gone
    quickly. The specific failure to look for is the fading text and the
    footer line printing over each other for a moment, which is what the
    old shared entrance/exit rule did. Check the rise clears the footer's
    padding cleanly rather than appearing to start mid-air; if it looks
    wrong, `--status-rise` in `style.css` is the single knob, and it must
    stay equal to the footer's bottom padding plus `#app`'s.
46. **Neighbours slide, they do not jump.** With one chip up, start a second
    operation (the What If auto-save overlap is easiest). The first chip must
    *glide* left as the new one arrives, not teleport. Then let a chip finish
    and watch the survivors glide back right. Hardest case, and the one worth
    doing deliberately: save a run with a long path while a chip is still
    up, so the resting line grows from empty to its full width in one go;
    the chip should ease across rather than being flung. Then turn on the
    OS "reduce motion" setting and confirm all of this degrades to plain
    fades with no sliding at all.
47. **The row does not fight the other floating surfaces.** Trigger a model
    download so the draggable download toast appears: at its default
    bottom-left it should not touch the row at bottom-right. Dragging the
    toast onto the row is expected to overlap and is not a bug.

**Resident navigation, picker flip, and load-bar corrections (needs a GPU
and a display).** The polish pass before this one was verified clean apart
from the three items it left open, which are 50, 51, and 54 below; item 55
is carried over untouched because it was never reported on.

48. **Re-selecting the resident model is navigation.** Generate with a model,
    go to the Main Menu, and pick that same model again. The confirm must read
    *Go back to the Generation page?* rather than offering to load it, there
    should be no loading animation or bar, and **the run must still be on the
    canvas when you arrive**, hyperparameters included. Do the same from the
    header selector, which should be a no-op there too. Then check the case
    that must still clear: with SmolLM3 resident on GPU, pick its **CPU** row
    from the menu. That is a real switch, so it must load properly and land on
    an empty canvas.
49. **A genuine switch still clears, from both paths.** Generate with SmolLM3,
    switch to a diffusion model and back, from the header selector and then
    from the menu. Only the run output should go; hyperparameters are keyed per
    model on purpose and must survive. Then generate, visit Analytics, and come
    back: that run must still be there, which is the case an over-eager clear
    would break.
50. **The overlay picker opens upward near the bottom.** Drag the collapsed
    drawer handle to the bottom of the output area, open it, and open the
    Overlay picker. The choices must appear **above** the drawer, fully inside
    the output area, rather than being cut off by its border. Drag the handle
    back to the top and confirm the picker opens downward again. Repeat on the
    Analytics page and in its run-detail modal. Then check the drawer drag
    itself still behaves: a drag must not open it, a plain click must, the
    position must survive a reload, and the two pages must remember separate
    positions.
51. **DiffusionGemma reserves a tail for the copy.** Activate DiffusionGemma
    and watch the sub-line. *Loading weights* should climb to roughly 90% and
    stop there, then *Moving to GPU* should carry the last stretch to 100%.
    The specific failure this replaced was the bar reaching 100% while still
    reading *Loading weights*, with *Moving to GPU* flashing past at full.
    Neither phase should jump: a leap from 90 to 100 in one step means the copy
    is not being tracked.
52. **LLaDA and SmolLM3 bars are unchanged.** These two take a different code
    path from DiffusionGemma and are meant to be untouched by that fix, so this
    is the regression check. Activate each on GPU: the bar should track the
    wait the way it did before, with no stall at 90% and no jump.
53. **"Moving to GPU" now arrives earlier.** This is an intended change. On
    SmolLM3 the sub-line used to stay on *Loading weights* until VRAM overtook
    RAM; it should now flip to *Moving to GPU* as soon as the copy starts, and
    stay there. LLaDA streams straight to the GPU and should read *Moving to
    GPU* almost immediately, as before.
54. **The bar climbs and finishes rather than cutting off.** Activate a model
    from the Main Menu, which is the fast path where this was worst, and watch
    the inline bar: it should step up several times and be seen at **100%**
    before the page moves to the generator, not vanish somewhere partway. Do
    the same for a header switch (which reloads) and for the very first load
    after a full restart (which comes up through the boot path). All three
    should complete visibly.
55. **A CPU load never claims a GPU.** Activate SmolLM3 with the device toggle
    on CPU. The sub-line must stay on *Loading weights* for the whole load
    **including the final frame**, and never say *Moving to GPU*. The bar
    should still be roughly linear: this is the fp32-vs-BF16 target case, so a
    bar that finishes at half is the scale factor being wrong.
56. **The download bar still works.** Delete a model's cache (or use a model
    you have never loaded) and activate it. The download phase must still read
    "Downloading NN%" both in the menu's inline bar and in the generator
    overlay, and must hand over cleanly to the load phase afterwards rather
    than resetting or double-counting.
57. **A failed load does not leave a bar behind.** Force an activation failure
    (see the note below this list for how). The overlay must close, the error
    must surface, and a subsequent successful switch must start from an empty
    track rather than the failed run's fill.

**How to force an activation failure.** Earlier revisions of items 57 and 73
said to request DiffusionGemma on CPU, "which is refused". That is wrong twice
over and cost the maintainer a session's worth of hunting, so it is written out
here once. There is no CPU option for a diffusion model in the UI at all
(`isAutoregressive` gates the device toggle in
[menu.js](src/web/static/menu.js); diffusion rows get a static GPU tag), and
reaching past the UI would not be refused either: `_resolve_device` accepts
`cpu` for any model and `activate()` **skips** the VRAM pre-flight for it
([server.py](src/web/server.py)), so the app would earnestly try to load LLaDA
into host RAM. Three things do work, and the first two fail through different
paths, which matters depending on what you are testing:

- **Rename a venv** (fails *inside* `activate()`, so the POST returns non-ok
  and the poll never runs). `mv .venv-dgemma .venv-dgemma.bak`, then activate
  DiffusionGemma: it raises "venv python not found" before spawning anything.
  Instant, no GPU, no weights touched, and a second `mv` puts it back. This is
  the one for item 57.
- **Kill the worker mid-load** (fails through the *poll*, which is the path
  that matters for anything about the progress track, since the bar is on
  screen when the error lands). `pkill -f "run_worker.*llada"` while it loads;
  `_monitor_startup` sees the dead process and reports "worker exited during
  startup". This is the one for item 73.
- **Occupy the VRAM** from another process and activate LLaDA, which trips
  `_preflight_vram`. The most realistic and the slowest to set up.

**Reveal signal, birth glow, and Tokens per Second (needs a GPU and a
display).** The case to press hardest is 58: it is the peak concurrent-glow
scenario on the weakest renderer, and it exercises the rendering rewrite and the
animation at the same time.

58. **LLaDA at `steps=8`, `gen_length=160`, in the `desktop.py` window.** This
    is the stress case: eight steps over 160 positions means roughly twenty
    tokens are born at once, each drawing a blurred halo, on WebKitGTK. Watch
    for stutter during the run and for glows that linger past half a second.
    Then run the same prompt at `steps=64`, where reveals are sparse and each
    flash should be individually legible.
59. **The streaming view looks the way it always did.** This is the regression
    check for the character-to-token rewrite, so compare against memory
    carefully. Masked positions must still carry their soft glow (that is a
    `live-tokens` rule restoring what `.token-mask` deliberately drops for the
    scrubber), mask opacity must still track confidence, and line wrapping must
    be unchanged. One difference is expected and is arguably a fix: the live
    text is now the concatenation of per-token decodes rather than one decode of
    the whole sequence, which is exactly what the scrubber has always shown, so
    the text should no longer shift subtly the moment a run finishes.
60. **Scrubbing does not retrigger the glow.** Finish a run, then drag the
    scrubber back and forth. Nothing should flash: the glow fires only from the
    live path. Then confirm the scrubber's own masked tokens do **not** glow,
    which is the check that `live-tokens` was taken off the container when the
    run ended.
61. **The glow toggle.** Turn **Token birth glow** off in Settings, save, and
    run: no flashes, and no stutter either. Turn it back on. Then set the OS to
    prefer reduced motion and run again with the setting on; there should be no
    animation at all.
62. **DiffusionGemma does not strobe.** Its draft tokens churn before settling,
    which is the case the monotone reveal set exists for. Each position should
    flash **once**. Watch a multi-canvas run in particular: a new canvas is
    fresh noise, so its positions are entitled to flash again, but positions
    within one canvas are not.
63. **Elapsed no longer jumps backwards.** Generate, note the footer Elapsed,
    then run an Edit Frames resume (or a What If substitution on SmolLM3). The
    number must keep climbing from where it was, not restart near zero. This is
    the bug the fix targets, so it is worth doing before anything else on the
    footer.
64. **Tokens per Second in the footer.** It should climb and settle during a
    run. Click it: the label swaps to the last step and the number gets noisier,
    which is expected. Reload the page and confirm the mode stuck. Then check
    that a Settings **Reset to defaults** does *not* flip it back, since its
    control is the footer and not that page. Generate, go to Analytics, come
    back: the readout should still be there and should honor the current mode.
65. **The Timing pager in Analytics.** Open a run: the section reads **Elapsed
    Time** with two small arrows beside the heading. The right arrow swaps to
    **Tokens per Second**, which should be a smooth curve that settles rather
    than a sawtooth, and the chart must be **correctly sized** rather than
    squashed, since it was built while its section was briefly visible. Zoom,
    pan, Reset, and the eye toggle should all work on it. Switch between two
    runs without closing the modal to confirm no stale chart survives. On an
    edited **SmolLM3** run the compare pins appear and the pre-edit line draws;
    on an edited **diffusion** run they must stay hidden, which is deliberate.
66. **Both elapsed totals.** Open an edited run's detail: the summary lists
    *Elapsed (original)* and *Elapsed (edited)* rather than one figure. An
    unedited run, and an edited run saved before the pre-edit signal existed,
    should both still show the single *Elapsed* row.

**Per-class glow tuning, sub-setting grouping, and the load sweep (needs a GPU
and a display).** Press 68 hardest: maximum brightness against the longest fade
is the worst case for paint cost, and it is the only combination that reaches
the raised concurrency cap.

67. **Defaults are unchanged.** Before touching a slider, run LLaDA and
    SmolLM3 and confirm the glow looks exactly as it did last session. Both
    classes default to 100% and 500ms, and the cap arithmetic is chosen to land
    on the old fixed 48 at that fade, so any visible difference here means the
    scaling or the fallbacks are wrong rather than merely mistuned.
68. **The worst case: SmolLM3 on GPU at 200% and 2000ms.** This is the stress
    test. Set **Tune for** to Autoregressive, push both sliders to maximum,
    save, and run a long generation in the `desktop.py` window. Expect a long
    bright meteor trail; watch for stutter, since doubled radii quadruple the
    blurred area per token and up to 192 tokens can be glowing at once. Then
    confirm the tail **fades** rather than being cut off partway, which is the
    check that the cap followed the fade instead of staying at 48.
69. **The two classes are independent.** With the autoregressive pair still at
    maximum, switch **Tune for** to Diffusion: its sliders must snap back to
    100% / 500ms, not inherit what you just set. Run LLaDA to confirm it still
    uses its own pair, then run SmolLM3 again to confirm it kept the loud one.
    Reload the page between the two to check both survived the save.
70. **The preview matches the real thing.** Drag each slider and watch the copy
    block replay a second or so after you let go. What it shows should be what
    a run then does, since both go through the same function and the same
    keyframes. Check the shape against the class: **Autoregressive** sweeps
    left to right, **Diffusion** scatters with no marching or directional
    pattern to it, and in bursts of visibly differing size. Push **Fade time**
    to 2000ms and confirm the trail still has a **dark head ahead of it and a
    faded tail behind it** rather than filling solid; this is the tightest
    case, peaking at 28 of 38 lit, so it is the one to actually look at. Drop
    it to 200ms and confirm only about six words are lit at a time. Replay
    twice at one setting and confirm the scatter lights the **same words in the
    same order** both times, which is what makes two settings comparable. Check
    the block's **right edge lines up with the slider readouts** above it.
    Click the block to replay
    without moving a slider. Turn **Token birth glow** off: all four sub-rows
    dim, become unclickable, and any sequence in flight stops immediately
    rather than finishing into a dimmed row.
71. **Sub-setting grouping.** In Settings, confirm there is no hairline between
    a preference and its sub-settings, that the sub-rows are indented under it,
    and that a single line closes each group off from the next preference (with
    no line at all after the last group, since it ends the panel). Toggle
    **Render diffusion-style text** off: **Mode** must now dim in place rather
    than disappear, and must not be clickable or keyboard-reachable while dim.
    Check the dim level reads as inactive rather than broken, which is the
    override cancelling the custom select's own opacity.
72. **The load sweep, on both surfaces.** Activate a model from the Main Menu
    and watch the inline bar: a sweeping track labeled *Starting worker* should
    appear immediately, where previously the row showed nothing for several
    seconds. It must hand over to *Loading weights* with a real percentage in
    one eased move rather than a jump, then to *Moving to GPU*, then hold a
    brief full bar before the page moves on. Repeat as a **switch** from the
    generator header, which uses the other renderer. DiffusionGemma is the one
    to check for the handoff, since its reserved tail makes the fill behave
    differently from the other two.
73. **The sweep's edge cases.** Force a failed activation, using the **kill the
    worker mid-load** recipe from the note under item 57 rather than either of
    the others: it is the only one that fails while the sweep is on screen,
    which is the transition being tested. The overlay must still close and
    surface the error, and a later successful switch must start from a clean
    track rather than the failed run's state. Then set the OS to prefer
    reduced motion and activate
    again: the track should show a dim, still bar rather than a moving one, and
    the labels must still change phase. Finally confirm the pager arrows on the
    Analytics charts and in the SmolLM3 candidate popover now read bright when
    they act and dim when they do not, and that the thin bar above the
    Analytics **Original / Edited** slider is gone while the generator's own
    crossfade **keeps** its separator.

*Items 74 to 80 cover the token metrics strip. None could be exercised
in-sandbox (no display).*

74. **Both pages idle correctly.** Open the generator before generating
    anything: the strip sits directly above the output canvas, holds its
    labels, shows a dash for each value, and the canvas below it is not
    clipped or scrolled. Open an Analytics run detail: the same row sits
    between the **Token overlay** heading and the bordered canvas, and the
    modal is still `90vh` with no scrollbar it did not have before. Confirm
    the reserved height does not shift when you hover in and out.
75. **The two hover sources agree.** On an AR run, hover a token and note the
    values, then move to the same column in the entropy profile below the
    scrubber: the strip should read the identical position and numbers.
    Leaving either surface returns it to idle, except that reaching *into* the
    candidate popover must keep the reading, since that popover is about the
    position you are still reading. Repeat both directions in the Analytics
    modal against its Entropy chart.
76. **It follows the frame, not just the pointer.** Park the pointer on one
    token and drive the scrubber with the arrow buttons: position stays put
    while the token, confidence and entropy change under it, and the strip
    goes idle on a frame where that position does not exist (the target
    placeholder during guided editing is the clearest case). Then hover a
    token during a **live** run: the values update as the run streams, which
    is new (streaming tokens never had a tooltip).
77. **The crossfade names its run.** On a confirmed What If branch, drag the
    **Original / Edited** slider: past the midpoint the tag at the right end
    of the strip flips, and the confidence and entropy change to the other
    run's values for positions right of the substitution. Do the same with the
    Diff overlay's two opacity sliders. With no branch, the tag is absent
    entirely rather than reading "Edited".
78. **The overlay extras.** Under **Commit Order** on a diffusion run, hovering
    a resolved token adds `Resolved at step: N`, and a position that never
    settled adds nothing. Under **Diff vs Original**, a changed position reads
    `was: X` and a remask origin reads `(remasked here)`. Switching overlays
    with the pointer held still should swap the extra without moving anything
    else in the row.
79. **Dashes, whitespace, and no tooltips anywhere.** On a LLaDA or
    DiffusionGemma run, entropy reads as a dash rather than 0, since diffusion
    tokens carry no `e`. Hover a token that is a plain space: it shows as a
    middle dot, not an empty box. Then rest the pointer on any token, on both
    pages, for a couple of seconds: **no native tooltip should appear** over
    any token, in any overlay, in either layer. (Buttons and table cells still
    have their own titles; those are meant to stay.)
80. **The popover clears the strip.** On an AR run with **Alternatives**,
    hover a token in the **first line or two** of the output: the candidate
    popover should open *below* the token rather than above it, leaving the
    metrics strip and the hyperparameter row uncovered. Hover a token further
    down and it should go back to opening above. Scroll the canvas so a
    mid-run token sits at the very top and confirm it flips there too, since
    the test is the token's position on screen and not its position in the
    run. Repeat both in the Analytics detail modal, where the strip is the
    thing being protected. Reaching down into a below-placed popover to click
    a candidate should still keep it open.

**0. The comparison surfaces (agreed with the maintainer, partly shipped).** The
timing foundation exists to serve these, and the unifying idea is settled: **the
pre-edit run is a first-class layer everywhere**, driven by one shared
Original/Edited state rather than a bespoke control per surface. The shared
layer, cross-highlighting, popover pagination, the line charts, and now the
generator's own crossfade have all landed (see "Recently shipped"); what
remains is below.

One correction to the framing above, learned from the line-chart pass: **one
shared state does not mean one shared control.** "Both runs at once" is only
expressible on stroke marks, so the line charts needed their own pins, with the
crossfade reduced to a momentary borrow during a drag. Expect the same split
anywhere the mark type cannot show two runs at full opacity.

- ~~The generator's crossfade and two-layer stack~~ **shipped this session**
  (see "Recently shipped"): `#run-blend-row` below the scrubber, gated on
  `runBlendActive()`, with `renderFrameWithTokens` routed through the shared
  span builder. The one thing worth carrying forward from it: that gate turned
  out to be load-bearing for *safety*, not just for tidiness, since it is what
  makes an interactive affordance on the un-editable layer structurally
  impossible. Any future surface that stacks the two runs should establish the
  same property rather than guarding each click site.
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
- ~~Status message stack~~ **shipped this session** (see "Recently shipped").
  One correction to the framing this entry had: the split that mattered was
  not footer readouts versus event messages, since `Step` and `Elapsed` were
  already their own elements. It was *inside* `#status-message`, between work
  in flight and the run's resting state. The download toast turned out not to
  collide, since it defaults to bottom-left and the stack is bottom-right,
  though a user who drags it there can still overlap it; that was judged not
  worth constraining a drag surface over.

**1. State-space models: Mamba-3 (new model class).** A genuinely new XAI
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
**Phase-2 XAI payoff:** SSM-native overlays on the recurrent state, per-token
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
