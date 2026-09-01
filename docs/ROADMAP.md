# Roadmap and Feature Backlog

A single, living reference for planned and candidate features, with enough
implementation detail to pick any of them up cold. This complements the short
Roadmap section in `README.md` (the public-facing summary) and the build plans
in `.cursor/plans/` (the historical, per-milestone plans). The deepest design
rationale still lives in the chat transcripts; this document is meant to capture
the parts worth keeping out of the chats.

## How to use this document

- Treat it as a living doc: update it whenever scope, priorities, or decisions
  change (that is explicitly allowed here regardless of any plan-execution
  boilerplate).
- Each planned feature lists: goal, current state, technical approach and hooks,
  files likely touched, open questions and risks, and a definition of done.
- The experimental section is a backlog of candidate directions to deliberate on
  before committing, not a queue of accepted work.

## Current status (orientation)

Phase 1 is complete: both models run locally with live visualization and the
analytics suite.

- Shipped: multi-model supervisor/worker architecture (process isolation),
  LLaDA (bf16) and DiffusionGemma (self-quantized NF4), per-token confidence and
  the confidence heatmap, analytics (convergence, timing, confidence, plus
  canvas-boundary markers), reproducibility metadata, graceful VRAM handling, and
  the DiffusionGemma thinking-mode split view.
- Shipped (latest session): DiffusionGemma single-canvas remask/resume (Phase 2
  below, via seed-canvas re-entry); two XAI overlays: commit-order
  (resolution-step) token coloring and the counterfactual "Diff vs Original"
  comparison (opacity sliders + difference blend); a grouped overlay picker
  (None / Heatmap / Diff) with persistent per-browser settings (highlight tokens,
  commit order) behind staged Save/Reset; and analytics run deletion (confirm
  modal + toast) with contained, toggleable chart tooltips (line burn-through).
- Shipped (this session): durable XAI overlays. Per-token records (display
  text, mask flag, vocab id, confidence) and the pre-edit snapshot are now
  persisted per run (`tokens.json` / `original_tokens.json`), the overlay math
  is shared between pages (`src/web/static/overlays.js`), and the Analytics
  Suite gained a static commit-order / Diff-vs-Original token viewer gated on
  data availability. Persisting confidence also makes a future durable Heatmap
  render a data-free follow-up.
- Shipped (analytics + edit-flow refinements): the run detail is now a wide
  fade-in modal (X or click-outside to close) with a corner overlay drawer
  (None / Commit Order / Diff vs Original) mirroring the generator; a sortable
  `Diff vs Original?` column marks runs that carry a pre-edit snapshot; the
  Group By options were pruned to the shared columns; and the guided editor now
  ends on a Confirm/Retry review step, locking Edit Frames once an edited run is
  saved (until the next Generate).
- Shipped (desktop wrapper): an optional pywebview launcher (`desktop.py`) runs
  the UI in a native window and owns the server lifecycle (starts uvicorn on an
  ephemeral localhost port, graceful shutdown frees worker VRAM on close), plus
  a Linux app-menu entry generator (`scripts/install_desktop_entry.sh`) and an
  app icon (`assets/icon.svg`). The browser path (`main.py`) is unchanged, so
  there is no dual maintenance. Cross-platform packaging (AppImage / Windows /
  macOS) remains deferred and is gated more by the CUDA/torch stack than by the
  webview layer.
- Shipped (menu + shell): a **Main Menu** landing page at `/` (a looping
  title-screen video, WebM with an MP4 fallback, over a GPU/VRAM-aware model
  picker that greys out models that will not fit). Generation moved to
  `/generate` and is now **gated behind model selection**: a direct hit with no
  active model redirects to the menu, and the `/ws` proxy no longer auto-boots a
  default worker. Consistent header nav across pages (Menu / Generation /
  Analytics; the Generation link surfaces only when a model is resident), the
  analytics layered **Diff vs Original** overlay (Original/Edited opacity sliders
  + difference blend, from the #1 backlog item), and removal of the old
  idle-animation feature (ASCII scene + donut + the Idle Display setting) in
  favor of a plain output placeholder.
- Shipped (this session, continued): the remaining backlog polish plus a round
  of refinements. An opt-in **diffusion-style text** effect (status messages
  resolve from block-glyph noise; a Default/Cycle Mode sub-setting; honors
  reduced motion) reused for button micro-interactions (Shuffle press, the
  Generate/New Run idle cycle with a one-time discovery teaser, and Lock In
  dissolving into mask glyphs). **Confidence-driven mask rendering**: masks are
  the accent green and their opacity tracks the model's live predicted confidence
  for LLaDA (`streaming_sampler.py` emits per-masked-position confidence), rising
  from a solid floor to full as a token nears its reveal. A **randomize-remasks**
  control (slider + N-of-M + Shuffle) in Edit Frames, which now opens on Frame 1.
  An Analytics **"new run" cue**: a persisted set of unseen runs drives a count
  badge on the generator's Analytics link and a per-row green dot cleared on open.
  **In-place edited-run save** so an edited run replaces its pre-edit original
  (one Analytics row, not two), with the session persisting canvas/confidence
  arrays + the last run id. **GPU/desktop robustness**: robust `nvidia-smi`
  resolution with logging, a driver/library-mismatch message on the menu, and a
  documented `libxcb-cursor0` (Qt/X11) dependency.
- Shipped (persistence + analytics polish): **durable server-side UI state**.
  Settings, the analytics "new run" cue, prompt history, and the generate
  teaser now persist in `results/ui_state.json` via `GET`/`PUT /api/ui-state`
  (`src/web/ui_state.py`), hydrated into localStorage on boot (`persistHydrate`
  / `persistSet` in `overlays.js`). This fixes desktop-app persistence, which
  the QtWebEngine profile keyed by the launcher's varying window origin/port; it
  also unifies state across the browser and desktop entry points. The cue is
  reconciled against existing runs on read, so a deleted run cannot inflate the
  count. Plus an **analytics table rework**: reordered columns (Date, Model,
  Prompt, Time, Edited), the renamed **Edited** marker as a diffusion-textured
  SVG checkmark (blank when unedited), the "new run" dot moved to the leading
  column, checkbox row highlighting, and multi-select **bulk delete**. Desktop
  launcher now uses a stable port (ephemeral fallback) and a persistent
  web-storage profile.
- Shipped (this session): the **first autoregressive model**, SmolLM3-3B, in a
  dedicated `.venv-ar` (Phase A below). Token-by-token streaming with per-token
  sampling confidence (`src/inference/ar_sampler.py`,
  `src/backends/smollm3_worker.py`); a `model_type` capability flag
  (`protocol.py`) that gates diffusion-only UI (Edit Frames, Diff overlay,
  Commit Order, convergence) off while keeping timing, confidence, and the
  Heatmap; per-activation CPU/GPU device selection threaded from the Main Menu
  through `run_worker.py` into `Backend.load(device=...)`, with the GPU
  pre-flight skipped on CPU and a CPU-capable torch wheel so GPU-less hosts can
  run it.
- Shipped (this session): the **menu + model-switch UX pass** on top of Phase A,
  validated on hardware. Non-blocking activation with a menu progress bar +
  Cancel; signed VRAM-headroom pills (accounting for the reclaimable resident
  model); a "Click to Download" veneer that pre-fetches uncached weights with a
  smooth **disk-size progress poller** (`hf_download.py`, replacing the tqdm hook
  that `snapshot_download` never routes to per-file byte downloads; Xet disabled
  before the first Hub import); select-to-confirm on the menu and dropdown; the
  rename to **LLM Visualizer**; model-family glyphs (diffusion D+F superposition
  with a crisp reversed epsilon; autoregressive @-to-R with a feedback loop); an
  Analytics **Processor** column + per-run timing device name; dropdown polish
  (fixed-width device pill, collapsed-width list with ellipsized names, green/red
  headroom tint, CPU-gated ticker, loaded-model highlight with a locked device);
  a 1-based AR step counter; and orphaned-worker guards (startup sweep +
  `PR_SET_PDEATHSIG`).
- Shipped (this session): the **analytics per-frame scrubber + durable
  Heatmap**, and the **app icon redesign**. The detail modal's token overlay
  gained a frame scrubber that replays every saved frame through the overlays
  (None / Heatmap / Commit Order / Diff), reusing the shared overlay math
  (`src/web/static/overlays.js`) and mirroring the generator; the new
  **Heatmap** recolors by persisted confidence, while Commit Order and Diff
  stay gated to diffusion runs (autoregressive runs get None + Heatmap). This
  was frontend-only, since the frames endpoint already shipped every frame
  (`analytics.js` / `analytics.html` / `analytics.css`). The app icon is now
  three CP437 diffusion shade blocks (`░ ▒ ▓`) as vector dither patterns under
  a corner-to-corner dark-to-bright green denoise gradient (`assets/icon.svg`,
  plus `assets/icon.png` via `scripts/render_icon.py`; `desktop.py` prefers the
  PNG, the launcher keeps the SVG).
- Shipped (this session): the **Settings page + Commit Order overlay**, **menu
  pagination**, and **cross-page download navigation**. Commit Order moved from a
  persistent Settings toggle to a generator overlay-picker option (diffusion-only,
  `app.js`), matching analytics. The generator's Settings modal became a shared
  **`/settings.html`** page (left tab rail: Appearance / Interface) with a gear
  icon in the generator, Main Menu, and Analytics headers; the settings schema
  now lives in `overlays.js` (`SETTINGS_DEFAULTS` / `parseSettings` /
  `settingsEqual`), shared by `app.js` and `settings.js`. The Main Menu model
  list is paginated (`i/N` pager, `menu.js`). A model download now runs as a
  global task the user can navigate away from: a shared draggable, corner-snapping
  toast (`download_toast.js`, persisted via `diffusion_download_toast_corner`)
  surfaces progress/completion when the inline veneer is off-screen; the menu
  re-attaches the veneer on return, `POST /api/models/download/ack` clears the
  terminal state, and `is_repo_cached` / `_has_incomplete` (`hf_download.py`)
  make a partial (`*.incomplete`) cache resume instead of bricking. The
  non-functional Cancel button was removed (real cancellation deferred).
- Shipped (this session): the **`results/` rename** and **autoregressive Phase
  C** in full. Saved runs now live in lowercase `results/` (one functional line,
  `RESULTS_DIR` in `src/web/server.py`, plus copy). Per-token **entropy** is
  captured on every AR run (`_entropy_nats` in `src/inference/ar_sampler.py`,
  off the untempered softmax the sampler already computes), persisted as
  `TokenRecord.e`, and drawn by a new **Entropy** overlay (`entropyColor` in
  `overlays.js`, on a cool-to-hot ramp normalized against
  `OVERLAYS_ENTROPY_REF_NATS = 5.0` rather than `log(vocab)`), plus a
  per-position **entropy profile** canvas under the scrubber. An opt-in
  **Alternatives** capture (top 5 per position) feeds a hover popover in both
  the generator and Analytics; each candidate set travels once, on the frame
  that introduces its position, so the wire cost is O(n·k) instead of O(n²·k),
  and it persists to `alternatives.json` indexed by position. **What If?**
  substitution closes the loop: a `supports_substitution` capability plus a
  `substitute` message (deliberately separate from `supports_resume`, which
  unlocks the diffusion remask UI), `streaming_substitute` +
  `Smollm3Backend.last_run_state` for a greedy re-decode from a forced position,
  and the branch recorded as an ordinary `RemaskEdit` so the Analytics Edited
  column and the durable **Diff vs Original** (now un-gated for edited AR runs)
  work with no schema change.
- Shipped (this session): the Analytics **Entropy by Position** chart, a
  follow-on to Phase C. The first chart in the suite indexed by token position
  rather than by frame, which is also why it is bars rather than a line: an AR
  model decides each position once, so its entropy is a property of the position
  and not a point in a time series. This is the chart the per-frame axis could
  not give us, since AR `mean_conf` is a cumulative mean (`conf_sum / count` in
  `ar_sampler.py`) and therefore flat by construction. Frontend only, off the
  frames payload `loadRunOverlays` already fetches; per-bar color from
  `entropyColor`, hover naming the token, and dashed markers at edited positions
  so a What If branch shows where its shared prefix ends. It also restores a
  third chart for AR runs, which hide Convergence.
- Shipped (this session): the counterfactual layer on that chart, plus a
  collision-aware tooltip positioner. The chart now carries a hover column
  matching the generator's profile, edit-orange markers and tint (`#ff9f1c`,
  the `.token-remasked` color) rather than accent green, a tooltip that splits
  into labeled **Original** / **Edited** rows from the divergence point
  rightward, and an **Original** / **Edited** crossfade slider over two
  superimposed `grouped: false` bar datasets, blended with canvas
  `globalAlpha` rather than by rewriting several hundred color strings per
  slider step. The original layer reads the pre-edit snapshot that
  `original_tokens.json` already carries, gated on the snapshot actually
  holding `e` so pre-Phase-C branches degrade to the single layer. Because a
  branch copies its prefix verbatim, both the second tooltip row and the
  visible crossfade start at the marker, which makes the divergence point
  legible without drawing anything extra. `Chart.Tooltip.positioners.smart`
  now scores the four plot-area corners against the pointer and the drawn
  data (bar bodies as rects, trendlines segment by segment via Liang-Barsky,
  so a sparse run's long segment cannot slip across a corner box unnoticed)
  and keeps its standing corner while it stays clear; `burnThroughPlugin`
  becomes the genuine last resort it was meant to be.
- Shipped (this session): the **shared comparison layer**, which turns the
  pre-edit run from a mode into a layer. `overlays.js` now owns one token-span
  builder (`overlaysBuildTokenSpan` / `overlaysBuildTokenLayer`) behind every
  path on both pages, so a stacked layer finally carries `token-span` and
  `data-pos` and is interactive; that alone repaired hover, the popover, and
  entropy highlighting in Diff mode, where they had never worked. Pointer
  ownership between exactly overlapping layers is now stated once
  (`overlaysEditedOwnsPointer`: the more opaque layer takes it, ties to
  edited) rather than falling out of sibling order. In Analytics the entropy
  chart's slider was promoted to a run-level crossfade on the token overlay's
  heading row, gated on the snapshot rather than on the entropy series, and
  `renderOverlayTokens` takes a `colorFor(index, token)` so **every** overlay
  mode stacks and blends the two runs, each layer colored by its own values.
  Commit Order needs a second memoized steps array, since a commit step is a
  property of a frame stream rather than of a token. Entropy bars and tokens
  cross-highlight in both directions on both pages (`setActiveElements` one
  way, a `token-cross-highlight` class the other; on the generator the missing
  half was a `mousemove` on `#entropy-profile` inverting its own layout math).
  The bar-to-token direction had to become a plugin `afterEvent` hook rather
  than `options.onHover`, which Chart.js only fires inside `chartArea`, so
  exiting through the axis gutter left the last token lit. The candidate
  popover pages between the two runs' top-k sets from the divergence point
  rightward, each page marking the token its own run drew. The generator's own
  crossfade and two-layer stack are deferred.
- Shipped (this session): **two persistence changes on opposite tiers.** The
  pointer hover and the entropy cross-highlight collapsed into one neutral
  white look (an accent tint disappears on an orange remask or the Heatmap's
  warm end), and `highlightTokens` moved out of the Settings page into a
  checkbox in each page's Overlay drawer: on by default, applied on tick, still
  in the shared `diffusion_settings` blob so both pages agree across a restart.
  `settings.js` keeps round-tripping the field it no longer shows, since Save
  writes the blob wholesale. Separately, hyperparameters, the Experimental
  toggle, and the prompt draft became **session**-scoped in a new
  `diffusion_param_state` sessionStorage key keyed by model id, deliberately
  outside `PERSIST_KEYS` so a fresh launch still starts from the recommended
  defaults, with a `#btn-param-defaults` Reset on the Experimental row that
  disables itself while everything already matches.
- Shipped (this session): the **line-chart comparison layer**, which finally
  consumes `original_per_frame_elapsed` / `original_mean_conf`. Both were
  already saved by `addOriginalRunSignals` and served by the metrics route,
  but the timing and confidence charts had stayed single-series, so an edited
  run could only ever show its branch. They now draw both runs, the original
  solid in grey and the branch dashed in the chart's own hue, sharing a prefix
  and separating at the edit. Which runs are drawn is owned by two **pins**
  per chart header (1 / 2, lit accent green, both on at open) as a three-state
  control rather than two independent checkboxes: the last lit pin is locked,
  because a chart drawing neither run has no reading. The run crossfade stays
  the token view's control and only *borrows* these two for the length of a
  pointer drag, easing back over 180ms on release (`scrubWeight` lerped
  against the pin answer in `seriesBlendPlugin`), which keeps the modal moving
  together without tying two frame-indexed charts to a slider that lives four
  hundred pixels away. Keyboard adjustments are deliberately excluded: arrow
  keys produce input events with no press to end them. Also moved the zoom
  controls into a segmented pill docked in each chart's bottom-left axis
  gutter (freeing the header for the pins, `layout.padding.bottom` reserving
  the strip), and the processor name from the timing header into its own run
  summary row, correctly labelled GPU or CPU from the run's own metadata.
  Fixed a long-standing bug the two-series tooltips made obvious: Chart.js
  paints a white backing behind each tooltip swatch and fills it with the
  dataset's `backgroundColor`, which on the line charts is an area wash at
  0.08 alpha (`"transparent"` on the compare panel), so every swatch read
  white with a colored rim. A shared `lineLabelColor` now paints them with
  the line's own color, which is what tells Original from Edited in a
  two-row tooltip.
- Shipped (this session): the **generator crossfade and two-layer token
  stack**, closing the last gap between the two pages. The generator had the
  layered diff since the counterfactual overlay landed, but its other four
  overlays stayed single-layer, so a branch could only be compared against its
  original inside Diff. A `#run-blend-row` below the scrubber (beside the diff
  sliders it is mutually exclusive with) now stacks the pre-edit run under the
  branch in every non-diff overlay, gated on `runBlendActive()`
  (`diffAvailable() && remaskMode === null`). That gate is what makes the
  stack safe rather than merely hidden: `token-clickable` needs
  `remaskMode === "edit"` and `token-substitutable` implies
  `remaskMode === "substitute"`, so a clickable affordance can never appear on
  a layer belonging to the run you cannot edit. Layer opacity is restyled in
  place on drag rather than rebuilt, since several hundred spans per slider
  step would also drop the candidate popover mid-drag.
- Also this session: **one span builder for the whole app**. The generator had
  built its spans inline since before `overlays.js` existed, because the
  shared builder could not express a remask selection, the edit-mode classes,
  or a mask graded by live predicted confidence. Three optional callbacks
  (`maskedFor`, `classFor`, `opacityFor`, all defaulting to today's behavior
  so Analytics passes none) closed that gap, and `applyTokenColor`, which both
  tinted a span and appended to its tooltip, split into a pure `tokenColorAt`
  and `tokenTitleExtra`. `maskedFor` is deliberately consulted only for a
  token that exists, so a hook can add masking but never strip it off a hole
  and leave `tok.t` read from null. Commit steps are now memoized per run
  (`originalCommitSteps` beside `commitSteps`, both cleared by one
  `invalidateRunMemos`), because a ghost layer painted from the branch's
  settle schedule would have misreported every position past the edit. The
  entropy profile gained the same treatment, stepping off the longer of the
  two runs so the drawing and the pointer-to-position inverse agree.
- Also this session: three finishing passes on the comparison surfaces. The
  generator's entropy profile gained the **edit marker** Analytics already
  had (tint under the bars, dashed orange line over them, hover glow last, so
  the pointer's guide lays over the tint rather than under it), drawn from a
  flattened `editedProfilePositions()` rather than a single index so
  sequential What If rounds each stay marked. It is deliberately the one
  standing mark on a strip whose scrub position is carried by bar opacity: it
  names a semantic fact about the run, not the cursor. The line charts got
  their **area fill back as a band between the two curves** (`fill: {target:
  0}` on the branch) instead of two washes to the axis, colored by whichever
  run bounds the region from above, which needs no legend and stays neutral
  across two charts that disagree about whether higher is good. Its alpha is
  `min` of the two series alphas and is baked into the color by a scriptable
  `fill`, not set as canvas state: Filler is registered globally, so it draws
  on `beforeDatasetDraw` ahead of `seriesBlendPlugin`'s inline hook and would
  never see a `globalAlpha` set there. Every path that moves a pin or the
  scrub already calls `chart.update`, which re-resolves the scriptable.
  Finally the tooltip swatch fix from the previous session was completed:
  `lineLabelColor` had painted the fill correctly but Chart.js resolves the
  swatch stroke as `borderWidth || 1`, so a colored ring survived, and the
  white backing showed as a half-pixel band inside it because the stroke is
  centered on a one-pixel inset. Transparent `borderColor` plus a global
  transparent `multiKeyBackground` leaves just the fill.
- Shipped (this session): the **status message stack**, the last item that
  had no dependency on the comparison-surface work. `#status-message` was a
  single overwritten span, so two operations at once lost one of them: the
  auto-save of the pre-edit run on entering What If, then picking a candidate,
  left only "Resuming". The split is by lifetime rather than by category.
  Work in flight raises a transient chip; the run's resting state (Done, the
  saved path, an error) stays in the footer, which is also what
  `saveSessionState` persists, so chips are free to expire without taking a
  record with them. That split is why session persistence needed no changes
  at all. The enabling refactor was small and had its precedent one function
  above it: `denoiseReveal` already kept its timer on the element so
  independent targets could animate at once, but `startStatusDots` kept
  module-level singletons, so two chips could not animate their own dots.
  Chips render inside the footer's own slot (a bottom-anchored column whose
  last row is the resting message, which collapses when empty), so a single
  chip lands exactly where the message alone used to and the common case
  looks unchanged. Bounded at four rather than made scrollable, since the
  real ceiling is two (one run, and `saveRun` guards itself with `isSaving`).
  The one trap: `resetStatus()` runs immediately before every resume, which
  is exactly when a save may be in flight, so it clears the footer only.
- And a third pass, after seeing it on screen a second time: **the column
  became a row.** Chips now extend leftward from the resting message rather
  than stacking above it, separated by a faint middle dot, clipped and faded
  against the gutter the footer's own gap already leaves before the readouts.
  Two details carried the change. The separators need no JavaScript: chips are
  inserted directly before the message, so `.status-chip + #status-message`
  matches exactly when a chip is up, and keying their opacity on `is-visible`
  makes them fade in and out with the neighbor they belong to. And the message
  keeps `flex-shrink: 0` with `max-width: 100%`, so it truncates only against
  the row itself and never to make room for a chip; overflow spills off the
  left, where the fade is, so the oldest chip is always what gives way. The
  clamp is cosmetic regardless: `saveSessionState` persists `textContent`. The
  same pass split each chip into a word span and a fixed-width dots span
  (`3ch` plus the footer's letter-spacing), which is what finally let the
  ellipsis tick continuously in every text mode, cycle included, since
  re-diffusing the word no longer rewrites the dots.
- Also this session, after seeing the stack rendered: **chips went quiet, and
  the messages got specific.** Letting a chip report its own outcome put
  "Done" on top of "Done." and "Saved" on top of "Saved to results/...",
  which read as stutter and was the only thing that ever pushed a second line
  into an already crowded corner. Chips now say only what is happening and
  simply leave when it is over, with the footer filling in as the handoff, so
  the whole `statusResolve` path, the hold timer, and the chip error style
  deleted themselves. Messages also name their subject rather than just their
  verb: a save reads "Saving original run" or "Saving edited run" off the
  `wasEdited` flag it already computed, and a resume reads "Running edit from
  frame X to Y" (or "to end"), which for `doGuidedResume` comes from a single
  `resumeTarget` shared with the request's `max_frames` so the text and the
  wire cannot drift. A layout fix rode along: `#status-stack` had been sized
  by `margin-left: auto` while its only child was absolutely positioned,
  leaving it zero-wide, so a long saved path ran left across "Elapsed:" and
  `max-width` had nothing to resolve against; `flex: 1; min-width: 0` gives it
  real width and the text now ellipsizes at the footer's own gutter.
- And a fourth pass on the same row, this one about **motion, which was the
  last thing still wrong.** A chip now rises in from the window's bottom edge
  and steps *left* on the way out, rather than sharing one rule with its
  entrance and so drifting back into the resting line it was handing off to.
  The exit is shortened to 150ms, since the footer already carries the outcome
  by then. Getting the rise meant trading `overflow: hidden` for a negative
  `clip-path` inset, because only the left and right clamps are wanted, with
  the rise distance held in one custom property that both the clip and the
  offset read. Flex offers no transition for a neighbour changing width, so
  `statusRowReflow` wraps every mutation that reshapes the row (a chip
  arriving, a chip's node leaving, the resting line filling in) in a
  first-last-invert-play, gated on `prefers-reduced-motion`. The row's
  entrances use the `translate` longhand precisely so that FLIP can own
  `transform` and the two compose. One backend fix rode along: the save
  endpoint reaches its folder two ways, and only one of them resolves, so the
  same message read `results/...` after a fresh save and an absolute path
  after an in-place update. `_display_run_path` normalizes where the branches
  meet, leaving the traversal guard alone.
- Shipped (this session): a **polish pass** plus the **model-load progress
  bar**. The polish, briefly: a model switch now clears the run snapshot
  (keyed by device as well as model, and dropped by both activation paths,
  since switching away and back lands on a matching pair that no identity
  check can reject); the output placeholder names the resident model; the
  Analytics "Edited" check lost its dot-pattern stroke and the three orphaned
  rules behind it; the prompt label gained 3px, which is what sets the
  clearance under the absolutely-positioned history control; the docs read as
  an LLM visualizer with the depth in discrete diffusion, and `xAI` is `XAI`
  throughout; the collapsed overlay drawer drags vertically via one shared
  helper in `overlays.js`, moving `top` because the group already animates
  `transform`, and owning the handle's click as well as its drag because at
  the target node listeners fire in registration order regardless of the
  capture flag; and AR **Alternatives** defaults on, with `smollm3_worker`
  now reading every absent-key fallback from the registry spec instead of
  keeping a second copy of each default.
- The bar itself is `src/inference/load_progress.py`, the companion to
  `hf_download.py`: getting weights onto disk had a readout, reading them into
  memory did not, and it is often the longer wait. There is no hook to borrow,
  so it samples memory counters the way `hf_download` samples the cache
  directory. Two findings shaped it. LLaDA loads with `device_map="auto"`, so
  accelerate streams shards straight to the GPU and RSS barely moves, while
  SmolLM3 fills RAM and copies after: sequential CPU-then-GPU phases would
  leave the bar at zero through half of one of them, so it reports
  `max(rss_delta, cuda_allocated)` over one target and names whichever counter
  it is reading. And LLaDA on CPU passes `torch_dtype=None`, which means
  fp32 from a BF16 checkpoint, so the target is scaled by the **requested**
  dtype, not the on-disk one. Anything unmeasurable (mixed dtypes, an
  unreadable header, an unfamiliar layout) returns a zero target and renders
  as the phase label with a spinner, because a confidently wrong bar is worse
  than none. The reading is floored at its previous peak, since the CPU
  allocator returns pages mid-load. The sampler runs on the helper thread and
  the load stays on the caller's, the opposite of `download_with_progress`:
  moving a heavyweight library-driven load between threads for a progress bar
  would trade real risk for a cosmetic one. The boot path polls too, which is
  what finally gave the slowest load of a session a bar.
- Shipped (this session): the four items left open by the pass above.
  **Re-selecting the resident model is navigation**: the server always treated
  that activation as a no-op, so the only damage was the clear the pass above
  had just added, which wiped the run on a path that spawns nothing. The menu
  now reads `active_device` (it was being discarded), asks *Go back to the
  Generation page?*, and uses the activate response's `state` as the
  discriminator, so a worker that died since the menu was drawn still gets the
  loading UI. **Dropdowns flip up when clipped**: the occluded rows were the
  Overlay picker's list, not the drawer, and the flip lives in the shared
  factory so every dropdown inherits it, flipping only when the list does not
  fit below and there is more room above. **A reserved tail for the pickled
  checkpoint**: DiffusionGemma unpickles the whole state dict into RAM before
  copying, so its read filled the bar and left the copy nowhere to go. The
  trap is that clamping is not enough, since the monotonic floor would jump
  the tail in one step; the read is compressed into `[0, ceiling]` and the copy
  scaled into the rest. Opt-in, defaulting to 1.0, so the two loads that
  already tracked their wait are provably untouched. **The bar finishes**: the
  closing 100% never reached the browser (the worker goes ready in the same
  breath and `_apply_health` drops progress), so the reducer names `ready` and
  both pages hold a full bar briefly, and three stacked 500ms polls came down
  to 250ms.
- Shipped (this session): the **reveal signal**, the **token birth glow**, and
  **Tokens per Second**. One missing piece of data gated both features, so it
  landed first: every sampler now stamps `revealed` on each frame, the
  positions that became resolved in that frame and had not been resolved
  earlier in the same canvas. The monotonicity is the whole design (see
  `src/inference/reveal.py`): "resolved right now" would re-fire on every
  settled token every frame, and "changed since last frame" would flicker on
  DiffusionGemma, whose drafts churn before they settle. Each sampler owns that
  differently: a **resume** seeds the set from the canvas it inherited, or the
  entire surviving prefix reports as newborn on frame 0; DiffusionGemma clears
  it per canvas, since the next one is fresh noise; and the autoregressive
  sampler needs no state at all, because left-to-right decoding means the frame
  that reaches *n* tokens produced position *n-1*.
  The rendering change that consumed it is a **net performance win**: the live
  path built one span per *character* and tore the whole output down every
  frame, roughly 640 inline boxes at LLaDA's default length, where the token
  view keeps a constant ~160 and writes only where something differs. That is
  also what made the glow possible, since an animation needs a node that
  survives the next frame. The glow itself animates a **constant-blur** shadow's
  alpha (animating the radius re-rasterizes a different-sized blur every tick,
  which is what the `.token-mask` scroll note warns about) and is keyed off a
  **data attribute** rather than a class, because the span-sync function owns
  `className` and would otherwise cut a glow short the moment its position
  changed.
  **Tokens per Second** needed no new storage and has no backfill gap: a masked
  token renders as exactly one mask glyph, so `compute_convergence`'s
  `mask_count` already is a token count, and every run ever saved carries its
  frame timings. In Analytics it shares the Timing slot behind a pager rather
  than claiming a chart of its own, since it is the same two numbers read as a
  ratio. The footer's **Elapsed** was wrong and was fixed alongside it: it
  printed the raw segment-local `data.elapsed`, so it jumped backwards after an
  edit. `ruff` was pinned and configured in the same pass (config-only
  `pyproject.toml`, 70 columns for both ruff and black, `C901` and `PLR1702`
  selected), establishing a **159-finding baseline** that this session's work
  did not add to; the findings were deliberately left unfixed.
- Shipped (this session): **per-class glow tuning, sub-setting grouping, the
  load sweep**, and two CSS corrections. Frontend-only, no Python touched.
  The glow's **Brightness** and **Fade time** are stored per `model_type`
  behind a class picker, because the trail an eye can follow is roughly rate
  times fade and an autoregressive GPU run outpaces a diffusion step by an
  order of magnitude: the default that reads perfectly on LLaDA is gone before
  it registers on SmolLM3. Three details carry the design. The values reach the
  keyframes as **whole shadow lists** in custom properties rather than as
  numbers nested inside `rgba()`, which keeps each keyframe a plain
  substitution. Brightness scales the **blur radii as well as the alphas**,
  because alpha alone tops out barely above the default 0.9 and there is no
  headroom in that. And the concurrency cap is now **derived from the fade**
  (`clamp(round(fadeSeconds * 96), 48, 192)`, with 96 chosen so the 500ms
  default still lands on the 48 it was fixed at): left fixed, a long fade at
  autoregressive speeds would have the FIFO rather than the timer decide when a
  flash ends, so the trail would stop growing exactly when the user lengthened
  it and its tail would look cut rather than faded.
  **Sub-settings** are indented and dimmed-when-inactive rather than hidden,
  which is what makes the indent mean anything, and the group's closing
  hairline moved to the `border-top` of the next preference. That avoids both
  `:has()` (thin support on WebKitGTK) and an "I am last" class in the markup
  that would rot; a group ending the panel correctly gets no line at all.
  **The load sweep** closes the dead time the maintainer noticed between the
  loading UI appearing and the bar starting. The gap is real work, not a
  rendering delay: a worker process spawning, importing torch and transformers
  in its own virtualenv, uvicorn coming up so `/health` answers at all, and
  only then `load_target_bytes` reading the shard headers. Nothing can measure
  it, so the fix is not a bar parked at 0% (which reads as hung, and is the
  thing `load_progress.py` refuses to draw); it is a sweeping track plus the
  honest label **Starting worker**. No backend change was needed: `starting`
  was already set in `activate()` and already returned by
  `/api/models/activation`; the shared reducer just fell through to its generic
  branch. That reducer went from a boolean `determinate` to a three-way `mode`
  (`hidden` / `sweep` / `fill`), since there are now three outcomes rather than
  two. One consequence worth knowing: both `finish*Progress` functions used
  `container.hidden` to mean "a bar was never shown", so an unmeasurable
  checkpoint used to end with no bar at all; with a track always present, every
  activation now closes on a brief full bar, and the menu in particular went
  from showing *nothing* during the gap to showing the sweep.
  The two corrections: the **Analytics crossfade separator** was a cascade
  leak, not an Analytics style at all (the row kept `margin-top`,
  `padding-top`, and `border-top` from `style.css`, where the generator still
  stacks it on its own line and still wants them), and the **pager arrows** now
  read by brightness instead of hue, since the accent green was on the
  *disabled* arrow: backwards twice over, being both the brightest thing in the
  row and camouflaged against the green chart title beside it.
- Shipped (this session): the **token metrics strip**, one always-present
  readout above each token canvas on both pages, replacing the native `title`
  tooltip. Frontend-only, no Python touched. Three things were wrong with the
  tooltip and only one of them was cosmetic: the browser delays it by around
  half a second with no way to configure that, it cannot be styled or placed
  (`overlaysPopoverTop` preferred above the token purely to dodge it, and now
  takes the canvas's top edge as a ceiling so it clears the strip instead), and
  it is
  bound to one element, so the entropy chart could never feed it however
  obviously it should have. The strip is fed by both hover sources on both
  pages, which is the feature the tooltip structurally could not have.
  It is a net deletion. The tooltip text was written in exactly one place
  (`overlaysSyncTokenSpan`), and everything upstream existed only to feed it:
  `tokenTitleFn`, `tokenTitleExtra`, `tokenExtraLabel`, `tokenLabel`, `confLabel`
  and an inline `titleFor` on the generator; `overlayTitleFn`, `commitExtraFor`,
  `overlayConfText`, `overlayEntropyText` and the `extraFor` / `originalExtraFor`
  parameter chain on Analytics. The strip computes the same values at hover time
  from the same memoized state, so none of that was rerouted.
  Three decisions carry it. **Always present**, because anything that appears on
  hover moves the canvas out from under the pointer that summoned it. **Its own
  hover variable** (`metricsHoverPos`), because `setEntropyHoverPosition` forces
  `entropyHoverPos` to null whenever the profile row is hidden, which is exactly
  the live-generation case where the strip has something to say; they answer
  different questions with different lifetimes. And **absent is not zero**: the
  tooltip printed `Confidence: 0` for a run that never recorded the signal, which
  is a claim about the model rather than about the record. A dash says the run
  does not carry it. Live generation gained a readout it never had for free:
  `LIVE_TOKEN_OPTIONS = {}` meant streaming tokens carried no title, but they
  always carried `data-pos`.
- Shipped (this session): **tokenizer identity, the typed token, and an AR
  top-k knob**, in that order, as three commits.

  The identity is read off the loaded object in `worker_base._health`
  (`describe_tokenizer`: class, `name_or_path`, `is_fast`, `vocab_size`),
  cached by the supervisor into `manager.active_tokenizer` on the same
  ready transition that caches `active_versions`, and written into
  `metadata["reproducibility"]["tokenizer"]` at save time. It deliberately
  does **not** ride `ModelCapabilities`: that is static registry data,
  served with no worker running, so a name there would be a hand-maintained
  string free to drift from whatever the checkpoint loads, which is exactly
  the failure a pedagogical readout cannot afford. `vocab_size` rather than
  `len(tokenizer)` because the base figure, not the one inflated by added
  special tokens, is what the entropy ceiling of ln(vocab) refers to.
  `tokenizerMetaRow` returns `""` when the key is absent, so the runs saved
  before it existed render unchanged; no endpoint changed, because
  `list_runs` returns raw metadata dicts and a new key flows through on its
  own.

  The typed token's real cost was **making the popover pinnable**, which was
  a change to what the popover *is* rather than a detail of the text field.
  It was hover-scoped and destructive: `hideAltsPopover` blanks
  `textContent` and fired on the popover's `mouseleave`, the output area's
  `mouseleave`, any capture-phase `scroll`, and `resize`, while
  `renderAltsPopover` rebuilds every child on each hover and page flip. Four
  separate ways to erase a half-typed word. The fix has two halves that
  matter independently: the closers stand down while `altsPopoverPinned()`,
  and the entry's state lives *outside* the DOM so `buildTypedRow` rehydrates
  it after any rebuild, which is what makes a re-render harmless rather than
  merely rare. Two traps found while building it. The pin cannot test "the
  draft is non-empty", because the field arrives pre-seeded with a leading
  space and that would pin the popover the instant the pointer crossed a
  mid-sentence token; it tests an `active` flag set on focus and cleared only
  by a deliberate exit, since blur is not an exit (clicking confirm blurs the
  input). And a cancel has to decide what to leave behind by where the
  pointer is: over the popover it re-renders to the candidates, anywhere else
  it closes, because nothing will ever come along to close a box the pointer
  has already left.

  The preview is a new `tokenize` / `tokenize_result` pair dispatched
  **outside** `gen_lock`, since it is a microsecond vocabulary lookup and the
  lock exists to serialize generation; taking it would stall typing behind a
  running model. The client carries a monotonic `request_id` and also
  compares the echoed text, because debouncing does not guarantee ordering.
  `Backend.handle_tokenize` is a default on the base class using
  `getattr(self, "tokenizer", None)`, so diffusion What If inherits a working
  preview whenever it arrives. Server-side re-resolution in
  `_check_typed_token` is the contract behind the client's disabled confirm
  button, and requiring the id to match the text is what stops a preview that
  went stale mid-keystroke from forcing a token the user never saw. The
  captured-candidate branch was left strict rather than loosened, so an
  unflagged request still cannot smuggle in an arbitrary id.

  The **true confidence** turned out to need no extra compute, only a moved
  boundary. `_substitute_loop` used to prefill prompt + prefix + the forced
  token in one pass; it now stops just short of the forced token, so the last
  position's logits *are* the distribution that position was sampled from and
  `probs[forced_id]` is the honest answer. The forced token is then forwarded
  against that cache via a new optional `past` on `_stream_tokens`. Same
  total work, two calls instead of one, and it survives `budget == 0` where
  `_stream_tokens` never runs. The seed frame had to move after the probe to
  carry the measured value. `forced_conf` is `None` only for a typed token; a
  captured candidate keeps the probability its own run recorded.
  `forced_entropy` stays `state["entropies"][position]` in both branches,
  because entropy describes the distribution at that position and does not
  change with the token forced into it. That is the line someone would
  plausibly "fix" by mistake.

  Top-k is distinct from `TOP_K_ALTERNATIVES = 5`, which is the capture count
  and stays fixed. It is applied before top-p, matching Hugging Face, so the
  two compose as a truncation with a nucleus taken inside it; the order is
  observable, since top-k renormalizes over what it kept and a nucleus
  measured against that inflated distribution bites harder. The default is
  `-1`, not `0`: both disable the filter, but `0` reads as "no candidates at
  all", which is the one thing a sampling truncation cannot mean. `0` still
  disables it, so runs saved under the older default replay unchanged.
- Shipped (this session): **the probe, the rank, and the strip's candidate
  readout**, closing the gap left by the typed row having no figure to show.

  A new `probe` / `probe_result` pair, dispatched **inside** `gen_lock`,
  unlike the `tokenize` pair beside it: a probe is a real forward pass, so
  admitting one alongside a generation would put two passes on the same
  device. `Backend.handle_probe` raises by default rather than being
  implemented on the base class the way `handle_tokenize` is, because
  answering needs a committed prefix to prefill up to, and a diffusion run
  reveals positions out of order and has no such thing.

  `probe_token` shares its prefill with the substitution path through
  `_position_distribution`, which is what makes the promise keepable: the
  strip quotes a figure before you run, the branch reports one after, and
  they cannot diverge because they are the same read of the same
  distribution. `test_probe_agrees_with_a_typed_substitution` pins that
  directly. Rank comes from `(probs > p).sum() + 1`, a comparison and a sum
  on a distribution already in hand, so it is free next to the pass that
  produced it; its denominator is `probs.numel()`, the model's output width,
  deliberately not the tokenizer's `vocab_size`, since a padded embedding
  makes those differ (128,256 against 128,000 for SmolLM3) and what was
  ranked is what could have been ranked.

  The **precision problem** is why the readout landed in the strip. A typed
  token is most interesting where it is improbable, and the popover row is
  320px wide: it can hold `<0.1%` and no more. The strip had half its length
  idle, so hovering any candidate row now fills its right half with the
  probability to three significant figures plus, for a typed token, the rank.
  The left group keeps reporting the committed token throughout rather than
  going idle, since the longest left readings barely reach the midpoint and
  holding both makes the two chips a legend: grey for what the run
  committed, green for what it merely weighed. The right group hides
  entirely when nothing is hovered, unlike the left, which stays visible as
  a key to what the strip reports.

  Two subtleties in the wiring. Rows bind `mouseenter` / `mouseleave` rather
  than `mouseover`, so crossing the bar and the percentage inside one row
  does not retrigger the readout. And `renderAltsPopover` clears the
  candidate before discarding its rows, because a removed node never fires
  the `mouseleave` that would have cleared it, which would leave a readout
  for a row that no longer exists.

  `overlaysBuildAltRow` was lifted into `overlays.js` in the same pass: both
  pages had a copy identical but for returning a row against a fragment, and
  both needed the same hover wiring. While ruff was to hand,
  `create_worker_app` came down from complexity 23 to 20 and `_ws` below the
  gate entirely, by lifting the load gate into `_await_model_ready` and
  collapsing the three byte-identical streaming branches into one dispatch
  through a dict. Adding the probe branch had pushed both further over a gate
  they were already past.
- Shipped (this session): **rank everywhere, the chosen row, the edit tint,
  scrub dimming, and the retained KV cache.**

  The trigger was a discrepancy worth recording, because the wrong fix was
  available and cheap. A typed token that the position *had* captured
  measured at 38.3% against a recorded 39.8%. Neither number was wrong.
  A run samples position *n* from one decode step against a cache built
  incrementally; a probe rebuilt the same prefix as a single prefill. Those
  are different orders of accumulation over the same values, and in bf16
  (8 mantissa bits) they part company by roughly an ulp, which is a
  percentage point down here. Rounding the display to hide it would have
  been a lie about a real arithmetic difference, so the two paths were made
  the same call instead.

  Two fixes, in that order. First, `requestTypedProbe` consults
  `positionAlts` before sending anything: if the token is one of the five
  the position recorded, the stored probability *is* the answer, and it is
  better information than a measurement as well as free. Second, the run's
  KV cache is retained on `last_run_state` and handed to both the probe and
  the substitution, so a measurement makes the same call the run made rather
  than a reconstruction of it.

  The cache work has four traps in it. `DynamicCache.crop` mutates in place,
  so slicing with it would consume the cache that a later probe needs; the
  slice is built as fresh views (`_sliced_cache`) and the record is never
  handed out directly. Reuse is gated on the prefix ids matching what the
  cache was built from, and *any* disagreement falls back to a fresh
  prefill: answering confidently from the wrong sequence is the failure that
  still returns a plausible number. Position 0 has no cached token to decode
  against and prefills unconditionally. And residency is bounded
  (`AR_CACHE_BYTES_MAX`, 512 MiB): a cache large enough to pass the ceiling
  is dropped rather than trimmed, since a run that big was never worth
  holding for the session. Invalidation rides `last_run_state = None` in
  `handle_generate`, which is the only place the pinned run changes.

  Rank turned out to need the *model's* output width, not the tokenizer's
  vocabulary: 128,256 against 128,000 for SmolLM3, because the embedding is
  padded for alignment. A rank is a place among the tokens that could have
  been ranked, so `describe_tokenizer` now also reports `model_vocab_size`
  from `model.config.vocab_size`, riding the plumbing the tokenizer identity
  already had rather than growing a second path. The captured five need no
  stored rank at all, since `torch.topk` returns them in order and a row's
  index is its rank; `rank` is set on exactly one entry, the sixth, and
  `_dump_alternatives` uses `exclude_none` so the other five do not each
  carry a null into a file already running to tens of kilobytes.

  The sixth row is appended, never substituted for the fifth: the five are a
  statement about what the model preferred, and dropping one to make room
  would quietly break it. It is excluded from being a substitution target
  (`alt-row-outside`), because forcing the token already sitting there would
  spend a full regeneration to arrive where it started.

  The edit tint and the scrub dimming are both about agreement between what
  two parts of the page say. `.token-edited` is a background rather than a
  color, so it composes under the Heatmap and Entropy overlays instead of
  fighting them, and it is softer than `.token-remasked` because the two
  mean different things: one is "selected, about to be redrawn", the other
  is "this run was intervened here", which stays true afterwards. Both pages
  memoize the edit positions into a lookup map keyed on the edit log's
  identity, since the class function runs per token per render and a
  diffusion run can carry many edits. Dimming past the scrubber needed two
  different mechanisms: the generator's canvas profile takes a third
  emphasis tier, while Chart.js has no per-bar opacity, so the Analytics
  chart bakes alpha into each bar's fill color (`entropyDimColor`, returning
  `hsla`) and multiplies with the crossfade's whole-dataset alpha.
- Shipped (this session): **context-window metrics, prompt import, and
  Analytics collections.** Three independent arcs, three commits, in that
  order, because the import control's "this fits" promise is only honest
  once the counting exists.

  The context readout under the textarea (`1,240 / 65,536`, amber when the
  prompt plus the output budget would overflow) rests on two decisions. The
  window is **read off the loaded object**, not declared in the registry:
  `describe_context_length` in `worker_base.py` prefers
  `model.config.max_position_embeddings` and falls back to
  `tokenizer.model_max_length`, which is frequently a sentinel of `int(1e30)`
  and therefore needs an upper bound (`CONTEXT_LENGTH_SANE_MAX`); neither
  being sane returns `None` rather than a guess. Same reasoning as
  `describe_tokenizer`: registry data is static and served with no worker
  running, so a declared number is free to drift from the checkpoint. And the
  count is of the **templated** sequence, produced by the code that builds
  the real inputs: `Backend.prompt_token_count` mirrors `_build_inputs`
  (SmolLM3 and DiffusionGemma inherit it), and the LLaDA worker overrides it
  through a newly extracted `build_llada_inputs` that its sampler and
  `_store_state` now share, so the counted tokens are provably the generated
  tokens. Counting raw text would understate: the chat template adds role
  markers, and `enable_thinking` changes them.

  `MSG_COUNT_PROMPT` is a separate message rather than a flag on `tokenize`,
  dispatched outside `gen_lock` the same way, because `tokenize` caps at 200
  characters and answers with one object per token; a 40 KB import would be
  tens of thousands of objects to answer with a single integer. The client's
  request has its own id counter rather than sharing `requestTypedPreview`'s,
  which is bound to What If state, and it re-requests when the `thinking`
  flag changes, since that changes the template. The authoritative count
  rides `prompt_len` on the `done` frame from all three samplers into a
  `context` block in `metadata.json`, so a saved run records what the model
  received rather than what the client estimated; the two Analytics detail
  rows stay absent for older runs, as the tokenizer rows already do.

  Prompt import is client-side end to end (`file.text()`), by button or by
  drop on the textarea, with a byte cap checked **before** reading and a
  character cap on what is inserted. `#prompt-history` is absolutely
  positioned and hidden when history is empty, so the import button could not
  live inside it: both now sit in a `#prompt-actions` flex container that is
  always present. Markdown goes in raw, since the model reads it fine and
  stripping it would misrepresent the file. A non-empty box confirms first,
  through the generator's first confirm modal, following the Analytics
  `#modal-delete` pattern.

  Collections **reuse `ui_state.py`** rather than a new file: one
  `diffusion_collections` key holding `[{id, name, runs: [run_id]}]` at the
  same 262,144 cap as `diffusion_new_runs`, which widened that module's
  docstring, because this is the first key that is durable user intent rather
  than a cache. Membership is a **set**, so a run sits in several collections
  without a later migration. Ids are folder names, and `_reconcile_collections`
  prunes ones whose folder is gone on every `GET /api/ui-state`, following
  `_reconcile_new_runs`: a collection is a list the user reads, so a stale id
  would show as a row that cannot be opened and a tab count that overstates.
  Malformed entries pass through untouched, since repairing a shape the client
  wrote is not this endpoint's job. "All" is a view rather than a stored
  collection; Favorites is created on the first star. Filtering happens before
  `sortRuns` in `renderTable`, and three things had to follow the active tab
  or they would act on invisible rows: `onSelectAll`, `checkedRunIds`, and
  `applyDeletions`, which also drops deleted ids from every collection so the
  table does not wait for the next hydrate to agree with itself. Storage
  eviction stayed out of scope: 175 runs occupy 440 MB against 189 GB free,
  so the pressure it would relieve is roughly 75,000 runs away, and bulk
  delete already exists.
- For the feature overview and architecture, see `README.md`. For the build
  history, see `.cursor/plans/`.

---

## Next session (accepted directions)

Agreed with the maintainer (deliberate each in Ask mode before Plan). (The
`results/` rename and all of AR Phase C shipped this session; see above.)

1. **State-space models: Mamba-3 (new model class).** Integrate a 1.5B Mamba-3
   SISO / MIMO checkpoint (`state-spaces` HF org, arXiv 2603.15569) as the first
   SSM, opening a distinct XAI lens: the fixed-size recurrent state. Key open
   decision, base vs instruct (the `state-spaces` weights are base LMs, not
   instruction-tuned). Own `.venv-ssm`; native `mamba-ssm` / `causal-conv1d`
   CUDA kernels (GPU-only, custom decode loop); ~3 GB VRAM. The streaming
   baseline reuses the AR frame / token contract and `model_type` gating; the
   phase-2 payoff is SSM-native state overlays (per-token Δ / state-write
   intensity, state-norm sparkline, fixed-state forgetting probes), which need
   kernel-intermediate capture. Now unblocked, since the AR tools have shipped.
2. **Entropy and top-k for the diffusion models.** The AR signals generalize, but
   the shape does not: a diffusion position is re-decided every step, so entropy
   becomes a per-position trajectory over steps rather than the single value the
   AR case yields. Needs a decision on payload (a trajectory per position is
   O(n·steps)) and on whether it rides DiffusionGemma's existing
   `entropy_signal` toggle. Also on what the position-indexed views become when
   a position has a history: both the profile under the scrubber and the new
   Analytics **Entropy by Position** chart read the final frame today, which for
   diffusion would show only each position's last value, so they would want a
   frame selector or a different shape entirely.
3. **An elapsed readout that ticks on a clock.** Accepted on 2026-08-28
   and **shipped the same day**, in `52b0968`. Kept here rather than
   deleted because the wrinkle below is the reason it works the way it
   does, and a future change to the footer needs to know it.

   The readout used to advance only when a frame landed:
   `updateRunRateFooter` had exactly one caller, inside `handleFrame`, and
   it printed `runFrames.elapsed[last]`. Between frames the number was
   frozen, so a wedged run and a merely slow one looked identical, which
   was the case worth fixing rather than the smoothness.

   The wrinkle, and the reason this is not a one-liner: that value is the
   *worker's*, stamped as `time.monotonic() - start_time` in
   `worker_base.py`. It measures inference and excludes the socket hop, the
   proxy and any render backlog, and it is the same figure that reaches the
   saved run and the Analytics duration. Replacing it with a browser clock
   would make the live number drift above the recorded one, which is the
   same class of defect as the live-versus-Analytics throughput disagreement
   `ANALYTICS-02` fixed.

   So it interpolates rather than substituting. Each frame's worker
   elapsed is stamped alongside `Date.now()`, the ticker shows that value
   plus the local delta since the stamp, and every frame re-syncs. Drift is
   bounded by one frame interval and the terminal frame lands on the
   worker's exact figure. The ticker is cleared on the terminal frame, on a
   disconnect, and on leaving the page, or it would outlive the run it
   describes.

   The open question was settled by leaving it alone: only the elapsed line
   moves between frames. Throughput still never consults a clock, because
   run-average sums `runFrames.revealed` over the worker's elapsed and
   last-step divides one frame's reveals by the delta between two worker
   stamps, and both read the array Analytics reads, which is exactly what
   makes the two pages agree. A rate that decayed during a stall would be
   useful for the few seconds a run is stuck and restless for the rest of
   the time.

4. **Stop making DiffusionGemma's confidence optional.** Accepted on
   2026-08-29 and **shipped on 2026-08-30**, kept here rather than moved
   because the reasoning is the useful part and the shipped notes below
   record what changed against it. Sequenced immediately after the
   mask-opacity retune so that a changed capture and a changed curve could
   not confound each other on hardware.

   **What shipped.** The reduction first: `_from_logits` computes
   `exp(max - logsumexp)` over chunks of 32 positions, which is the same
   number from two reductions with a transient of roughly 33 MiB instead of
   a canvas-sized softmax, pinned by a test against the old formulation.
   Then the gate, at all eight sites, and the stability-derived confidence
   with it. One thing turned out slightly larger than scoped: `_stable` was
   carried in the resume checkpoint, so `DgemmaFrame` lost a field too, and
   the entry below on the revision glow is why that was the right call
   rather than a loss.

   One consequence worth stating plainly, because it crosses the boundary:
   `mean_conf` is now a mean over real probabilities on every frame, where
   before it averaged a probability or a stability count depending on the
   toggle. Runs saved on either side of 2026-08-30 are plotted on the same
   Analytics axis and are not strictly comparable.

   The toggle is misnamed, which is most of the case for removing it. It
   emits argmax confidence, not entropy: DiffusionGemma never writes an `e`
   field at all, only `ar_sampler.py` does, and a saved DiffusionGemma run
   carries exactly `t`, `m`, `id`, `c`. So the switch does not choose
   between two signals. It chooses whether the *primary confidence channel*
   exists, the one that mask opacity, the Heatmap, and the candidate
   reveal's fade all read, and a run without it is not a cheaper run but a
   run with a hole where the measurement goes. The audit reached the same
   place from the other direction (`AUDIT_REPORT.md`, `ROADMAP-03`):
   "reusing the existing toggle would also conflate confidence with entropy
   while paying a large temporary-memory cost".

   **Answer the cost before deleting the gate.** The gate exists for a real
   number: `_from_logits` builds a float32 softmax over 256 positions by
   roughly 262K vocabulary entries, about 256 MiB, and `tensor.float()`
   copies before it, so peak transient is around half a gigabyte per
   denoising step. Invisible on a 24 GB card holding an 18 GB model, which
   is why it never surfaced, but it is what the toggle protects. It is also
   avoidable: max probability is `exp(max_logit - logsumexp(logits))`, two
   reductions with no probability tensor, and chunking the reduction over
   positions bounds the transient to tens of megabytes. That is `ROADMAP-03`'s
   own direction ("numerically stable reductions over logits ... without
   retaining a full probability tensor longer than required") and it
   generalizes to entropy and top-k. Once it is cheap, removing the gate is
   a deletion rather than a bet on the maintainer's hardware being typical.

   The surface is small: the spec in `registry.py`, four sites in
   `dgemma_worker.py` (parse, generate, `last_run_state`, resume), two
   assignments in `dgemma_sampler.py`, and a test fixture knob. No frontend,
   because the parameter panel is generated from `param_specs`. No
   migration, because the Analytics detail panel renders params
   model-agnostically and `_compare_label` skips specs it cannot find, so
   runs already saved with the signal off keep displaying it.

   What it buys beyond the copy cleanup: the stability-*confidence* branch
   in `_emit` and the `self._stable` counter that feeds it become dead and
   can go. What it does **not** buy, so nobody over-claims: the absent-`c`
   guard stays, because LLaDA's frame 0 and every already-saved run keep
   their shape forever; and `masks_are_real` with the settlement convergence
   basis stays, because DiffusionGemma infers `m` from a token changing
   between steps, which is a property of renoising and not of this switch.

---

## Autoregressive model support (Phases A and C shipped)

The north-star direction: host open-source **autoregressive (AR) LLMs** alongside
the diffusion models, reusing the frame/token contract, overlays, and analytics.
The backend contract already generalizes (a model is a `ModelInfo` in
`registry.py` plus a worker implementing `Backend.load` / `handle_generate`), and
the streaming "frame" (a full snapshot plus a `{t, m, id, c}` token list) maps
cleanly onto AR: frame N is the sequence after N generated tokens, every token
`m: false`, `c` = the sampling softmax confidence.

**Phase A (shipped this session).** First AR model end to end:
- **SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B`) in a dedicated **`.venv-ar`**
  (`transformers==4.53.0`, incompatible with LLaDA's 4.38.2), with
  `requirements-ar.txt`. Chosen over Phi-4-mini (standard but less novel) and
  Gemma-3n-E2B (novel MatFormer/effective-2B, but the trickiest first
  integration) for a clean integration whose think/no-think mode reuses the
  existing thinking UI. The torch wheel is the standard CUDA build, so it runs
  on GPU or CPU (not a CPU-only wheel), the code picks the device per
  activation.
- **AR streaming**: a manual token-by-token sampling loop (not HF's text
  streamer) capturing per-token confidence, one full-snapshot frame per new
  token (`ar_sampler.py`). Reuses save, token records, and the scrubber
  (left-to-right replay). Note: full-snapshot frames make the payload O(n^2) in
  tokens; the registry caps the recommended `max_new_tokens` at 256 and the
  worker clamps harder (~128) on CPU.
- **Model-type gate**: `model_type` ("diffusion" | "autoregressive") on
  `ModelCapabilities`; hides diffusion-only UI (Edit Frames, Diff overlay,
  Commit Order, convergence chart), keeps run + timing + confidence + Heatmap
  (the natural per-token AR confidence view).
- **Per-activation CPU/GPU device**: `device` on the activate request threaded
  through `run_worker.py` -> `create_worker_app` -> `Backend.load(device=...)`,
  skipping the VRAM preflight on CPU; a CPU/GPU toggle on the AR menu row that
  greys the GPU option when a GPU is absent or the model will not fit. A
  body-less activate resolves to GPU-if-present-else-CPU server-side.
- **Analytics**: convergence chart dropped for AR (would flatline); timing and
  confidence kept. `model_type` is persisted in each run's `metadata.json`.

**Phase C (shipped this session).** The AR XAI trio, built in dependency order
(read-only signals first, the intervention last):
- **Entropy**, always on. `-(p log p)` over the untempered softmax
  (`_entropy_nats`), in nats, stored raw. Normalizing by `log(vocab)` was
  rejected: on a ~128k vocabulary every realistic value collapses into the
  bottom tenth of a [0,1] scale, so the display side normalizes against
  `OVERLAYS_ENTROPY_REF_NATS = 5.0` instead. Surfaced as an **Entropy** overlay
  and a **per-position entropy profile** under the scrubber. Note the framing
  correction: because an AR model samples each position exactly once, this is a
  profile across the sequence, not the per-position trajectory the earlier notes
  described (that shape belongs to diffusion, where a position is re-decided
  each step).
- **Top-k alternatives**, opt-in (`alternatives` BOOL `ParamSpec` on `SMOLLM3`,
  mirroring DiffusionGemma's `entropy_signal`), k fixed at 5. A position's
  candidate set is fixed the moment it is sampled, so it rides only the frame
  that introduces that position and the client accumulates by position: O(n·k)
  on the wire against the O(n²·k) of repeating it per snapshot. Shown in a hover
  popover with probability bars, mirrored read-only in Analytics.
- **What If? substitution.** `supports_substitution` + a `substitute` message
  rather than reusing `supports_resume`, which would unhide Edit Frames with its
  diffusion-only frame selection and randomize-remasks slider.
  `streaming_substitute` prefills prompt + kept prefix + the forced token in one
  pass, then continues **greedily** so the divergence is the intervention's
  effect rather than fresh RNG in a shifted context. The worker's
  `last_run_state` holds the per-position trace (published via a `state_sink`,
  off the wire) and validates that the requested id was actually among that
  position's captured candidates. The frontend reuses the diffusion resume
  splice path (`resumeFrameOffset` / `isResuming`) and the Confirm/Retry review,
  and records the branch as an ordinary `RemaskEdit`, so no server schema
  changed.
- **Entropy by Position chart** in the Analytics detail modal, the same signal
  promoted from the generator's compact profile into an inspectable chart: axes,
  zoom/pan, a tooltip naming the token, a hover column, and an edit-orange
  marker plus tint at each edited position. Built from the frames payload in
  `loadRunOverlays` rather than the metrics payload in `loadRunCharts`, so it
  costs no extra request. Gated on the data (`overlayEntropyAvailable`), not on
  `model_type`.
- **Counterfactual entropy comparison** on that chart for edited runs: the
  pre-edit snapshot as a second superimposed bar layer, an Original/Edited
  crossfade slider, and a tooltip that grows a labeled second row exactly where
  the branch stops sharing its prefix. At the substituted position the two rows
  carry the same nats and different tokens, which is the intervention stated in
  one line: `_substitute_loop` keeps the forced position's originally captured
  entropy because the distribution there is a function of the prefix, and the
  substitution changes only which token was drawn from it.

**Remaining AR follow-ups.**
- Integrate **Phi-4-mini-instruct** and **Gemma-3n-E2B-it**.
- **Download-from-menu**: a curated allowlist of the candidate models; needs a
  dynamic registry layer (registry is a static dict today) + disk-space checks.

**Deferred (future session): Randomize Prompt.** A dice button (left of the
prompt-history button) that fills the prompt via a small always-on CPU model, with
a Settings model dropdown. Deferred because the randomizer must coexist with the
resident model, which the one-worker-at-a-time manager forbids; it needs a
separate concurrent CPU "utility" worker with its own lifecycle. High effort for
the near-term payoff.

---

## Phase 2: DiffusionGemma interactive remask and resume

Status: **shipped for single-canvas runs**; multi-canvas resume remaining.

**Goal.** Bring LLaDA's scrubber remask/resume experience to DiffusionGemma:
pick a frame, remask (renoise) tokens, and resume to a later frame or to the
end, with the same guided multi-frame editing flow.

**What shipped (single-canvas).**
- `streaming_resume` in `src/inference/dgemma_sampler.py` re-enters a chosen
  frame's canvas via `decoder_input_ids`, renoises the remasked positions to
  random tokens, and continues denoising under a reduced `max_denoising_steps`
  budget, reusing the frame streamer for capture.
- `DgemmaBackend.handle_resume` in `src/backends/dgemma_worker.py` reconstructs
  the seed canvas from the stored `frame_history`, validates the request, and
  splices the new frames onto the history. `_validate_resume` enforces the
  single-canvas scope (it raises if any frame has `canvas_index > 0`).
- `capabilities.supports_resume=True` for DiffusionGemma in
  `src/backends/registry.py`, so the guided editing UI unlocks. The frontend
  gates **Edit Frames** with `runIsMultiCanvas()` so multi-canvas runs keep it
  disabled, and surfaces the renoise semantics ("nearby tokens may also change")
  in the guided status text.

**Remaining: multi-canvas resume.** This is the hard part and stays open.
- Multi-canvas chaining: a run can span several 256-token canvases; a resume must
  target the correct canvas while preserving already-committed prior canvases.
- Encoder-decoder plus KV cache: re-entering canvas N mid-run means reconstructing
  the decoder state relative to committed canvases, not just seeding one canvas.
- Adaptive stopping: step counts vary per canvas, so a frame index does not map
  to a fixed step schedule.

**Files (for the multi-canvas follow-up).**
- `src/inference/dgemma_sampler.py` (extend the resume generator to seed a
  specific canvas while replaying committed ones)
- `src/backends/dgemma_worker.py` (lift the single-canvas guard in
  `_validate_resume`; track per-canvas frame provenance)
- Frontend is already generic; re-enable **Edit Frames** for multi-canvas once the
  backend supports it.

**Open questions and risks.**
- Cross-canvas resume semantics: resuming within canvas N while canvases before N
  stay committed, without recomputing them.
- Whether the installed `generate()` can seed a non-first canvas cleanly or needs
  a patched entry point for that case.

**Definition of done (single-canvas): met.** Select a single-canvas DiffusionGemma
frame, remask tokens, resume to a later frame or to the end; frames stream
correctly; save and analytics are unaffected; the UI unlocks. Multi-canvas resume
is the remaining milestone.

---

## Phase 3: Multimodal image input

Priority: after Phase 2.

**Goal.** Accept image input to DiffusionGemma (a multimodal-capable base) for
image-grounded text generation.

**Current state.** Text only. `src/backends/dgemma_worker.py` deliberately loads
`AutoTokenizer` (not `AutoProcessor`) to avoid pulling in torchvision.

**Requirements and considerations.**
- `AutoProcessor` plus torchvision for image preprocessing; a vision tower adds
  VRAM on top of the roughly 18 GB NF4 text stack, so the 24 GB budget must be
  re-validated (the vision components may need lower precision or careful
  offload).
- Protocol and UI: extend the `generate` message and add an image-upload control;
  output frames remain text canvases.
- Keep the text-only path intact; image input should be optional.

**Files likely touched.**
- `src/backends/dgemma_worker.py` (AutoProcessor path + image inputs)
- `src/inference/dgemma_sampler.py` (thread pixel inputs through `generate`)
- `src/backends/protocol.py` and the frontend (image-upload control + payload)
- `requirements-dgemma.txt` (pin torchvision)

**Open questions and risks.**
- VRAM headroom for the vision tower alongside the NF4 experts on a 24 GB card.
- Whether the self-quantized NF4 checkpoint retains or needs the vision
  components, or whether they load separately.

**Definition of done.** Upload an image plus a prompt, DiffusionGemma generates
grounded text, VRAM stays within budget, and the text-only path is unaffected.

---

## Settled decisions and deliberate stopping points

Places where work stopped on purpose, with the reasoning that would be
expensive to rediscover. These are not backlog items: each one is either a
line drawn deliberately, or a trap a future change will otherwise walk into.
They moved here from `docs/HANDOFF.md` when `META-01` reduced it to a cold-start
page.

**The desktop window recovers from a dead renderer rather than
preventing one.** Recorded 2026-08-31, after the window was reported
turning white after fifteen or more minutes and needing a restart,
which had been happening for a while.

The page was not at fault, and it is worth writing down that this was
checked: all six `new Chart(` calls in `analytics.js` route through
`destroyChart` before recreating, and every `setInterval` is paired
with a `clearInterval`, so the two usual suspects for a slow leak in
this app are both clean.

The mechanism is one level down. Chromium runs the page in its own
process, and when that process dies QtWebEngine leaves the view
blank: no error, no event the page can see, nothing in any log.
pywebview does not connect `renderProcessTerminated` at all (checked
against 6.2.1, the string does not appear in its Qt backend), so
nothing notices and nothing recovers. That is the entire reason the
symptom was a silent white window rather than an error.

The reported trigger is the machine idling until the screen blanks or
locks, which fits a GPU context lost to suspend that the renderer
does not survive. If that is right, the app cannot prevent it: it is
the compositor and the driver, one layer below anything this codebase
owns. So the window recovers instead, bounded to three automatic
reloads, because a death on resume is a fact of the host and reviving
it is invisible, while a death that repeats is a bug and a window
that reloads forever would hide it.

Worth resisting the temptation to call this fixed. Recovery is a
treatment for a cause still unconfirmed, which is why every
termination is appended to `renderer-crashes.log` under the user data
directory: the app is normally launched from a desktop entry, where
stderr goes nowhere, and without that file the only evidence is a
white window and a memory of when. The **negative result is the
valuable one**: if the window still goes white and that log is empty,
the renderer did not die, this diagnosis was wrong, and the
investigation starts again somewhere else.

Two things deliberately not done. `--no-sandbox`, which pywebview
applies on arch, manjaro, nixos, rhel and pop but not here, is the
standard fix for a window that is white *from launch*, a different
failure from one that renders correctly and dies later; adding it
would be cargo-culting a remedy for a symptom this is not. And
software rendering would sidestep GPU suspend entirely at a cost to
the smooth scrolling and compositing the Qt backend was chosen for.

**Where the watch is installed is not a detail, and reasoning about
Qt threading is not a substitute for running it.** The first version
of this shipped broken: it installed the watch from pywebview's
documented post-start callback, on the stated theory that connecting
a signal across threads is harmless. It is not. Qt objects belong to
the thread that created them, and `view.page()` from another thread
parents a `QWebEnginePage` to a view the GUI thread owns, which Qt
refuses with "Cannot create children for a parent that is in a
different thread" and the process does not survive. The app aborted
at launch with a core dump, so a rare white window had been traded
for one that never opened. The watch now goes on inside a wrapper
around `BrowserView.__init__`, which pywebview builds on the GUI
thread and which has set its page by the time it returns, and
`LLM_VISUALIZER_NO_RENDERER_WATCH` exists so a future pywebview that
breaks the hook cannot make the app unstartable.

Two things made that avoidable in hindsight, both now available.
`QT_QPA_PLATFORM=offscreen` runs the whole desktop app headlessly in
the agent sandbox: the launcher comes up, QtWebEngine loads the real
pages, and the renderer even crashes on its own for lack of a GPU,
which exercised the recovery path end to end without hardware. The
belief that GUI work here is unverifiable was too broad; it is the
*display* that cannot be checked, not the process. And the status
decoder was tested only with plain ints while PyQt6 sends an enum
member that `int()` raises on, so every test passed while the one
line a user reads said
`RenderProcessTerminationStatus.CrashedTerminationStatus`. A fake
whose shape is more convenient than the real type is a fake that
tests the fake.

**Source inspection cannot see event wiring, and a DOM harness is
cheaper than assumed.** Recorded 2026-08-31, after bulk filing shipped
with a dead dialog: the chooser's target rows rendered correctly,
counted overlaps correctly, and could not be clicked, because the list
listened only for `change` and a target row is a `<button>`, which
never fires it.

Two source-inspection tests bracketed the defect without touching it.
One proved the rows are built by `buildCollectionTarget`; the other
proved they carry no checkbox, which is the very property that breaks
a `change` listener. Both were right, and the bug lived in the seam.
That is the general shape of the limit: source inspection reads what
the page *contains*, never what the browser *delivers to it*, so any
defect that is purely about which event reaches which handler is
invisible to every frontend test this repo currently has.

The standing assumption was that closing that gap needs jsdom, a
dependency plus a harness. It does not. `analytics.js` loads into a
`vm` against a hand-rolled DOM shim of roughly ninety lines, in the
page's real script order, and a synthetic click on a row's child
element produces the actual `POST`. That was built and run during this
session: it passes on the fix and fails on the bug, which is the only
property that makes such a harness worth anything.

Not built now, deliberately. Getting it to load took five rounds of
stubbing (`overlaysLoadSettings`, Chart's nested config assignment,
`querySelector` returning null, `getComputedStyle`, `getContext`), and
each of those is a way it breaks later for reasons that have nothing to
do with the behaviour under test. A harness that fails on unrelated
changes is a tax paid every session, and one that quietly passes on
broken code is worse: the first version of this one did exactly that,
reporting success against the shipped bug because its request filter
matched the boot's `GET /api/analytics/runs`.

The trigger to build it is a **second** event-wiring bug. One is a
mistake; two is a class, and at that point the tax is worth paying.
Until then the cheap substitute is the pattern this bug produced,
under `tests/web/` in the collections polish suite: grep for
`test_the_chooser_hears_the_event_a_target_can_emit`. It pairs the two
facts that bracket the seam, asserting a button row and a `click`
listener together rather than separately.

**What a flat entropy profile actually means: domain first, repetition
second.** Settled by the project's first real experiment, on 2026-08-18,
after a 2048-token SmolLM3 code run came back with a profile near zero for
most of its length. Two prompts under identical settings and seed, both
keeping the "occupies 2048 tokens" demand so the length pressure stayed a
constant rather than becoming a second variable:

- **A**, a Python cookbook required to use a different standard-library
  module per recipe and never reuse a structure.
- **B**, unrelated prose paragraphs with no shared theme.

A produced genuinely varied content, 41-plus distinct modules with no
repeated function, and flatlined anyway. B, under the same length pressure,
stayed dense and high throughout. What that settles:

- **Domain sets the floor.** Code is intrinsically low-entropy even when the
  content is varied. Mean confidence was roughly 90 percent for A against 65
  to 70 for B.
- **Repetition is a slope on that floor, not the floor.** A's profile
  *declines* rather than sitting flat, because the recipe scaffolding stayed
  constant while the payload changed. The model learns the frame and copies
  it even while the filling varies.
- **The original run was both effects stacked**, code domain plus verbatim
  body repetition, which is how it reached near zero.
- **The spikes land on real decision points**, which module to reach for and
  what problem to pose, with the troughs on forced syntax. This is the
  closest thing to a validation the entropy feature has: it is measuring
  uncertainty rather than tracking token frequency.
- **Models pad code far more readily than prose.** B stalled near 670 tokens
  across several seeds and only filled the budget once the prompt explicitly
  licensed it to keep inventing subjects. Boilerplate is cheap to emit.

The Help copy in `src/web/static/index.html` led with repetition before this
and was corrected to lead with domain. If that paragraph is ever rewritten
again, this is the evidence it has to agree with.

**A mask's confidence distribution is a fingerprint of the remasking
strategy.** Measured on 2026-08-30, again from a prediction written before
the run, which is the only reason it is worth keeping.

Two LLaDA runs, same prompt, same 128 steps, same 160-token canvas, with
`remasking` the only deliberate difference. Taking the median top-1
probability among masked positions per frame, then across the run:

- `low_confidence`: **0.145**. At frame 1, 96 percent of masked positions sit
  below 0.2, and at frame 90 still 64 percent do. It never climbs.
- `random`: **0.992**. Frame 1 is identical at 0.112, since both start from
  the same all-masked canvas, but by frame 10 the median is 0.353, by frame
  30 it is 0.945, and from frame 90 it is pinned at 1.000.

The mechanism is that `low_confidence` retains exactly the positions the
model is least sure of, so the masked set *is* the low-confidence tail by
construction and each step shaves its top off. `random` retains an arbitrary
sample, which is mostly positions whose context now determines them
completely, so the masked set fills with tokens the model is certain of and
merely has not been allowed to write. This also explains, from a second
direction, why the mask-opacity retune found a median between 0.11 and 0.21
across a whole run: that measurement was taken on a `low_confidence` run.

Not controlled, and worth saying: one run per arm, both with seed -1, both
carrying one edit. At a sevenfold separation in the median the direction is
not in doubt, but the exact figures are one draw each.

The consequence for the flicker scoped under layer three is that the strategy
would become directly visible as how much the canvas moves. Counting
positions whose top candidate does not dominate the cycle, the median per
frame is 63 under `low_confidence` against 15 under `random`: one churns, the
other sits nearly still while producing the same kind of text.

**A DiffusionGemma canvas brightens together as it nears its stop.**
Observed on 2026-08-28, and recorded because it was written down as a
prediction first and could have come back the other way.

The reasoning behind the prediction was that adaptive stopping fires on two
conditions read over the whole canvas, so confidence should rise across
positions in concert rather than position by position, and reset when the
next canvas starts from fresh noise. That is what a 512-token two-canvas run
with the Entropy Signal on showed. It is weak evidence in the sense that it
is one run judged by eye rather than a measurement, but it is the right
shape, and it means the mask-opacity overlay is reading the same quantity the
halting criterion does.

Two things fell out of looking that were not what the item asked:

- **The grading was only ever visible on rewind.** The live streaming view
  passed no callbacks to the span builder, so the canvas was flat while it
  was being written and graded only when scrubbed back over. Fixed the same
  day; the entry under Shipped records why the gap existed.
- **A revision is invisible, and a birth is not.** On DiffusionGemma a
  position can settle, be revised, and settle again, and only the first is
  marked. Watching a canvas replace its placeholder runs with real content,
  which is the most interesting thing it does, currently produces no visual
  at all. Scoped in the XAI backlog as a second glow rather than a wider
  first one, because the suppression that hides it is the same mechanism
  that stops a churning position from strobing.

**Collections ship without storage eviction.** The original framing paired
favorites with storage-pressure relief; the measurement killed that half. 175
runs occupied 440 MB against 189 GB free, so the pressure eviction would
relieve is roughly 75,000 runs away, and the table already has multi-select
bulk delete. If it is reopened, the honest trigger is a disk figure, not a run
count.

**Three collections loose ends, each a chosen stopping point.**

- Collection order is creation order, with Favorites `unshift`ed to the front.
  Drag-to-reorder was not built; the stored array *is* the order, so it is a
  UI change rather than a storage one.
- The `+` tab reuses the rename editor (`beginCollectionNameEdit` with a null
  collection) so creating and renaming cannot drift apart. The cost is that
  the tab strip reflows while you type a long name.
- The chooser applies each checkbox immediately, so **Done** only closes the
  dialog. A footer button that commits invites closing and wondering whether
  anything happened. A cancel would need a staged copy of the membership, not
  a change to the checkbox handler.

**The context count is per template, so it is per `thinking` flag.**
`promptCountThinkingSent` keeps a count taken under one template from being
displayed against the other. Any future parameter that changes the template, a
system prompt field for instance, needs the same treatment or the readout goes
quietly stale. Relatedly, `prompt_token_count` is one base implementation plus
one override: SmolLM3 and DiffusionGemma inherit the `_build_inputs` mirror
and LLaDA overrides it. **A new model class that builds its inputs differently
must override it too**, or it reports a count from a template it does not use,
which is worse than reporting nothing.

**The KV cache is retained but its benefit is unmeasured.** The cache lives on
`last_run_state` and is reused by both the probe and the substitution
(`_reuse_cache`, `_sliced_cache`, `_cache_record` in `ar_sampler.py`). The
claim that prefill dominates a substitution's cost is well founded and was
never timed, because the sandbox has no GPU. To get the number: time a
substitution at a late position in a several-hundred-token run, once with the
cache present and once with `state.get("cache")` forced to None.

**Follow-ons left open by the typed token.** None of these blocks anything.

- **Multi-token substitution.** Shipped at exactly one token, and the blocker
  was never validation, it is **alignment**: a substitution of length *n*
  shifts every downstream index, and `overlaysComputeDiff`, the
  position-indexed entropy chart, and the dashed edit marker are all
  index-based. When it lands, cap on resolved token count (4 was the agreed
  figure) rather than on characters, so the live preview already computes the
  limit and the limit explains itself. `_check_typed_token` is the one place
  that hard-codes `len(pieces) != 1`.
- **The "why" behind a BPE split stays out of scope**, deliberately. A split
  is reached through a merge sequence that fast tokenizers do not expose
  cheaply, so there is no short interpretable reason to show. Naming the
  tokenizer is the honest answer available for free, which is what the
  Analytics row does. Do not reopen without a concrete plan for getting merge
  ranks out of the fast tokenizers.
- **A configurable capture count** (currently `TOP_K_ALTERNATIVES = 5`) is
  deferred. It is distinct from the sampling `top_k` that shipped, and needs a
  decision about already-saved runs whose `alternatives.json` holds five.
- **Diffusion What If** would inherit the typed preview for free, since
  `Backend.handle_tokenize` is a base-class default reading
  `getattr(self, "tokenizer", None)` and all three workers answer it. What is
  missing is the substitution path, not the tokenization.
- **The popover pin is generator-only.** Analytics needs none of it because
  its popover is read-only. If Analytics ever gains an interactive one, both
  halves have to come with it: the closer suppression *and* the
  state-outside-the-DOM rehydration.
- **Two probabilities for the same position disagree unless they are the same
  call.** In `bf16` a prefill and an incremental decode over identical values
  differ by about an ulp, which is a visible percentage point at these
  magnitudes. Any readout that recomputes something a run already measured has
  this problem. In order: prefer the recorded figure, then reuse the run's own
  cache, then measure afresh. Do not round the display to hide it.

**Comparison surfaces, the remainder.** The unifying idea is settled, that the
pre-edit run is a first-class layer everywhere driven by one shared
Original/Edited state. One correction learned from the line-chart pass: one
shared state does not mean one shared control. "Both runs at once" is only
expressible on stroke marks, so the line charts needed their own pins with the
crossfade reduced to a momentary borrow during a drag. Expect that split
anywhere the mark type cannot show two runs at full opacity. What is left:

- **Confidence chart**, a cumulative versus per-position toggle. The default is
  **modality-aware**, not per-position everywhere: the AR `mean_conf` is a
  cumulative running mean so its curve is degenerate, while LLaDA's and
  DiffusionGemma's are per-frame canvas means and that curve is what makes
  adaptive stopping visible. Per-position comes from the **frames** payload
  while the line comes from **metrics**, so the toggle straddles two
  independent fetches.
- **Timing chart**: a dashed edit marker (reuse `substitutionMarkerPlugin`
  with frame indices via `resumeBoundarySet`; its orange tint is sized from
  bar geometry, so a line chart needs either a width strategy or no tint), and
  "E204" style tooltip labels for the branch's frames. The marker is the
  valuable one: the two lines separate visibly, but nothing yet names the
  frame where they do.

**The load bar's unmeasurable phase stays unmeasured.** The ambitious fix was
to time worker startup on first run, persist it per model, and drive a real
bar from the previous run's timing. Held on the maintainer's call. Two things
to weigh if it returns: a cold start and a warm one differ by more than the
estimate would tolerate, so the bar would routinely stall at 90% or finish
early, which is worse than admitting there is no number; and it would put
per-model timing state in `results/ui_state.json`, which currently holds only
user preferences. The cheaper middle ground is to name the sub-phase rather
than measure it, since the worker already knows when it has finished importing
and when uvicorn is answering.

## Experimental / XAI feature backlog (to deliberate)

The suite is shaping up as an explainability playground, so these are candidate
directions to scope together before building. None are committed. For any that
capture heavy signals (logits, entropy), prefer the established pattern:
cheap-by-default with an opt-in for the expensive path (as with the Entropy
signal toggle).

**Per-run notes, explicitly post-audit.** Raised on 2026-08-18, straight after
the entropy ablation above, and by the same route: the findings that made that
experiment worth keeping were written into this file by hand, because the runs
themselves had nowhere to hold them. A note attached to a run, with some way to
reference or collate a group of runs, would let a session's reasoning live
beside the evidence it came from.

Two things to settle before building it, both of which are why this is not
scoped yet:

- **It overlaps Analytics collections**, which already group runs. The overlap
  is soft: a collection answers "which runs belong together", a note answers
  "what did I conclude". They may still want to be one feature rather than two,
  and deciding that after building both would be the expensive order.
- **It probably does not stay a list.** Notes that reference other notes and
  groups of runs are a graph, and the shape it grows into depends on how far
  the project goes. Building the flat version first is defensible, but only if
  the storage does not assume flatness.

The framing to aim at is a researcher's working notebook over the run history
rather than a comment field: sectioned, cross-referenced, and traversable in
the order the questions actually arrived rather than the order the runs did.

**Candidate reveal for DiffusionGemma, in place of the mask glyph.** Raised on
2026-08-18 while reading a multi-canvas run's convergence curve, and worth
scoping properly because most of the data already exists. Three layers, in
increasing cost.

*Layer one is free and retroactive.* DiffusionGemma has no mask token; the `░`
is substituted into the frame text at render time while the token record keeps
the real guess, so `{'t': ' properties', 'm': True, 'id': 5082}` is a typical
"masked" position. Showing the model's current best guess instead of a block
needs no new capture and works on every DiffusionGemma run already saved. It
would also make a finding self-evident that currently needs a paragraph to
explain: early in a canvas the display would fill with `" the"` and then be
eaten by real content, which is precisely why the convergence curve overstates
(see the ledger).

**Shipped on 2026-08-28 as the Settings toggle "Reveal the mask candidate",
for both diffusion models rather than only this one.** Off by default. The
client owns the glyph now: the shared span builder substitutes `tok.t` for
the mask character when the preference is on, keeping `token-mask` and the
confidence fade, so a revealed guess reads as a dim tinted word and never as
a settled one. See the correction below on what LLaDA needed.

The metrics strip was deliberately left alone, and the reasoning is worth
keeping because the plan for this work assumed the opposite. It looked as
though the strip already named the candidate for DiffusionGemma, since both
pages hand it `tokenText: tok.t`. They do, but `overlaysMetricTokenText`
returns the mask character for any position flagged masked, so the strip has
always shown `░` there for both models and still does. That reads as a
distinction rather than a contradiction: the canvas is a reading of the
model's draft, and the strip is a statement about the position, which is
also why it reports a remask selection as masked. If hardware says the
mismatch is jarring rather than clarifying, the fix is one branch in that
function; the manual item asks the question.

*Layer two is opacity, and it needed one small change.* Fade each unsettled
token by the model's confidence in it. Confidence was already computed with
Entropy Signal on and then discarded for exactly the positions that would be
faded, because `_emit` wrote `c` only `if not unresolved`.

**Shipped with `XAI-01` on 2026-08-18, gated rather than unconditional.**
This entry used to say that writing `c` unconditionally was the whole
capture change. That was very slightly wrong, and the correction is worth
keeping. An unresolved position is by definition one that just changed, so
`_emit` reset its stability count to zero on the same pass and the stability
branch could only ever write `0.0`. Since `maskOpacity` returns the floor for
zero and for absent alike, writing it would have cost payload on every frame
of every run for a pixel-identical canvas. So `c` is written for an unsettled
position only when `conf_override` supplied a real softmax confidence, which
is exactly when the Entropy Signal is on. The overlay itself needed no work:
`tokenOpacityFn` was already asking every masked token for its confidence and
receiving nothing.

Layer one, and the stopping readout below, no longer wait on a capture
change. Layer one shipped on 2026-08-28; the stopping readout remains
unbuilt, and has the number it needs on any run recorded with the signal on.

An earlier worry, recorded because it was wrong and the correction is the
useful part: the streamed logits are temperature-scaled, and the schedule
anneals `t` from `t_max` to `t_min` across a canvas, so the concern was that
opacity would trace the sampler's schedule rather than per-position certainty.
The measurement says otherwise. Across one canvas the temperature moved 0.800
to 0.642, a 1.25x sharpening, while mean confidence moved 0.165 to 1.000. A
1.25x rescale cannot produce a sixfold rise, so the climb is overwhelmingly
genuine settling and the temperature contributes a few percent. **Do not undo
the temperature**; the signal is trustworthy as it stands.

*Layer three is the top-k stack*, unsettled tokens rendered as their top five
candidates with opacity by probability share, expanding on click into
something like SmolLM3's alternatives popover. This is the expensive one and
it is already sequenced: `ROADMAP-03`'s axis-aware signal manifest "precedes
its native XAI phase and diffusion entropy and top-k", and `ORG-03`
consolidates the LLaDA sampling kernel before top-k goes into it. The shape is
also new, per-frame *and* per-position, where `alternatives.json` today is
per-position only, which suffices for an autoregressive run because a position
is decided once and does not for a diffusion draft that is re-decided every
step. Budget roughly frames x canvas width x k records.

**Two renderings, one capture.** Scoped 2026-08-30. The stack above encodes
probability share as stacked opacity, which is spatial. The alternative is to
encode it as a *duty cycle*, cycling a position through its candidates so
each is shown for its share of the time. Same data, same semantics, different
channel, so both are display choices over this one capture rather than
separate features, and the dependencies above cover them equally. The
time-average of the flicker is literally the distribution, which makes it a
temporal dither, and it degrades correctly at both ends: a dominant candidate
holds the position almost still with ghosting below fusion, a flat one
churns.

What was worked out before anyone builds it:

- **The two models are 24x apart and only one has a problem.** Measured:
  DiffusionGemma runs at about 960ms per step, so splitting one step five
  ways gives 192ms per candidate and a per-step cycle simply works. LLaDA
  runs at about 40ms, giving 8ms, which is half a frame at 60Hz. Build it
  for DiffusionGemma first: no timing work, no accumulation, a tenth of the
  effort, and it answers the perception question for both.
- **Separate the display clock from the data clock.** Driving the animation
  off frame arrival restarts the cycle 12 times per 500ms on LLaDA. The
  animation needs its own timer, fed either by *latching* the most recent
  real distribution for a cycle, or by *smoothing* across steps with an
  exponential moving average. Build both and compare: latching only ever
  shows distributions that actually occurred, smoothing evolves continuously
  instead of jumping, and the capture is identical either way.
- **Tie the capture rate to the display rate, not the step rate.** If the
  display latches every 500ms the worker need only send top-k that often,
  which is every twelfth frame on LLaDA. Per step it would be roughly 24KB
  and 20,000 candidate objects a second for the browser to parse; tied to
  the display it is a fiftieth of that. This also takes `RUNTIME-01` off the
  critical path for a live-only version, since nothing has to be persisted
  to animate. It does not clear `ROADMAP-03`, because a per-frame top-k
  field on the wire is exactly the bespoke sidecar that finding exists to
  prevent.
- **The duty cycle selects for itself.** An earlier sketch had a rule
  choosing which positions deserve to flicker. Unnecessary: a position at
  c=0.99 gives its top candidate 495ms of a 500ms cycle and renders static,
  while one at c=0.11 churns. Confidence *is* the amount of motion, so calm
  positions cost no attention and no rule has to exclude them. A cap is a
  motion budget, not a selection mechanism, and it binds only in the opening
  frames where a canvas is genuinely undetermined.
- **Reduced motion is a hard requirement here**, not the courtesy it is for
  the birth glow. A canvas of 158 positions each changing ten times a second
  is well past any flashing guidance in aggregate, so the effect must be
  inert under `prefersReducedMotion` and capped even when motion is allowed.
- **It is an alternative to the mask reveal, not a layer on it.** Cycling
  changes a position's width, so each needs reserving to its widest
  candidate, which pads the canvas and ruins the natural text flow that the
  reveal exists to give. The Settings copy should present them as two ways
  to read the same thing rather than as stackable.

Note one asymmetry if this is ever mapped onto LLaDA as a display preference,
which it could be. On DiffusionGemma layer one reveals something currently
thrown away; on LLaDA a mask is a real token with nothing behind it, so the
same display would require newly computing what the model was considering.
Identical on screen, quite different underneath.

**That last sentence was wrong, and the correction is the useful part.**
Found on 2026-08-28 while scoping layer one for both models. LLaDA's
`_diffusion_step` takes an argmax over every position on every step, uses it
to fill the few positions it reveals, and drops the rest one line later. The
softmax over that same argmax is `true_conf`, which the sampler has always
returned and which drives LLaDA's mask opacity. So the app has been reporting
how confident the model is about a guess it could not name, and nothing new
had to be computed: `_diffusion_step` returns `x0` alongside what it already
returned, and `_build_token_list` decodes it into `t` for masked positions.
The asymmetry that does survive is one of recording rather than of
computation. DiffusionGemma keeps the guess in the frame already, so layer
one is retroactive there; LLaDA wrote the glyph into `t`, so its reveal is
forward-only and a run saved before this shows blocks, correctly, because
nothing else was written down.

**A live adaptive-stopping readout, which pairs with the above.** Same signal
at a canvas scale rather than a token scale. DiffusionGemma halts a canvas
when two conditions hold, both from the checkpoint's `generation_config.json`:
the accepted canvas is unchanged for `stability_threshold` steps (1), and the
mean entropy of the processed logits falls below `confidence_threshold`
(0.005 nats).

It is a threshold, not a probability, so the honest readout is a distance
rather than a percent chance of stopping; inventing the latter would be the
plausible-but-false number this project keeps trying not to draw.

The cheap version is already in hand. A mean entropy under 0.005 nats forces
max-probability to about 0.9995, which is the quantity `c` already holds. On
the 2026-08-18 entropy-signal run the implied entropy crossed 0.005 one to two
frames before each canvas actually halted, the lag being the stability
condition plus the fact that a binary-entropy floor understates a 262k-way
distribution. The exact version is one more reduction over a softmax
`_from_logits` already materialises; the heavy part, moving the logits to the
CPU, is paid already whenever Entropy Signal is on.

One measurement caveat, now historical but worth keeping because it dates
the data: the figures above average over the positions carrying `c`, which
at the time was the stable subset rather than the whole canvas the criterion
uses. Near a halt the two coincide because everything is stable, so the tail
those numbers describe is trustworthy while the early ones sit on a biased
sample. The `c` change removing that bias shipped with `XAI-01` on
2026-08-28, so a run recorded with the Entropy Signal on since then carries
the unsettled positions too and needs no such allowance. Anything measured
before that date does.

**A second glow for a revision, distinct from a birth.** Raised on
2026-08-28 while confirming that mask opacity had started grading live. The
observation: on LLaDA and the autoregressive models the birth glow is
complete, because a position is decided once and the first settle is the
whole story. On DiffusionGemma a position can settle, be revised, and settle
again, and only the first of those is marked. Watching a canvas replace its
placeholder `" the"` runs with real content, which is the most interesting
thing it does, currently produces no visual at all.

The suppression is deliberate and should stay. `_emit` keeps a
`_seen_revealed` set precisely so a churning position does not strobe every
few steps, and without it the glow would stop carrying information. Which
means the fix is not to loosen the guard: the anti-flicker rule and the
missing revision signal are the same mechanism, so the answer has to be a
*second* signal rather than a wider first one.

A revision and a birth are different events and deserve different marks. A
birth is a hole being filled; a revision is the model changing its mind,
which is arguably the more informative of the two and the one this project
exists to surface. The data is already in hand, and needs no new capture or
payload.

**Corrected on 2026-08-30: it is in hand in the browser, not the sampler.**
This paragraph used to say `_prev` and `_stable` sit in `_emit` and that the
detection belongs there. That sent an implementer to the wrong layer, and it
mattered, because `_stable` was removed with the stability-derived confidence
it was the only reader of. Nothing was lost. `overlaysComputeCommitSteps`
already derives a per-position temporal quantity, the step at which a
position settled, out of nothing but the frame stream, and drives the Commit
Order overlay on both pages from it. Every token record carries `id` on every
frame, live and saved, so "this position changed after having settled" is a
comparison of consecutive frames and "how long had the old value held" is a
backwards scan, both computable at frame *f* from the frames the client
already holds. A resumed branch is no exception, since the pre-edit run is
held too, in `originalRun.tokens`, which is the same thing
`overlayOriginalCommitSteps` already handles for Commit Order.

Building it client-side also gets Analytics for free, where a sampler-side
flag would have needed a new persisted field to be reviewable at all.

What needs deliberating is what it should look like, since a mark that fires
on every re-settle would reintroduce the flicker under a new name. Worth
considering a decay rather than a flash, or a threshold on how long the
previous value had held, so that a position thrashing between two candidates
reads differently from one that committed, sat, and then genuinely changed.
Scoped per canvas either way: `put` clears the seen set at each commit, so
every position is legitimately newborn on the next canvas.

Revisited on 2026-08-28, the same day, once the mask reveal shipped, because
the reveal weakens the premise without removing it. This entry opens by
saying that watching a canvas replace its placeholder `" the"` with real
content "currently produces no visual at all". With the reveal on you now
watch exactly that, in text, which was the specific example given. What
survives is narrower and still worth building: the reveal shows the *state*
of a position on each frame, and a viewer has to notice a difference between
two frames to see a change, which is precisely what is hard to catch while
scrubbing or at speed. A mark says a change *happened* at this position, on
this step, and stays visible for a moment after. So the entry is not
subsumed, but it should be built against a canvas that already names its
candidates: with the reveal off, a revision is a block that stays a block,
and a mark is the only evidence there is. That, rather than the `" the"`
example, is the case to design for.

Shipped from this backlog (see `README.md`):
- Token commit-order coloring: tokens are tinted by the step at which they
  resolved (light green early to red-orange late), as a persistent overlay. Now
  durable: reviewable post-hoc in the Analytics Suite from saved per-token data.
- Counterfactual / intervention diff: the "Diff vs Original" overlay compares an
  edited run against the original, with opacity sliders and a difference blend.
  Now durable: the pre-edit snapshot is saved, so the diff is reviewable in the
  Analytics Suite. A propagation heatmap remains a possible extension.
- Confidence-driven mask opacity (LLaDA): a still-masked token's opacity tracks
  the model's live predicted confidence for that position, rising from a solid
  floor to full as it nears the reveal, so the "heating up" before a commit is
  visible live and while scrubbing. DiffusionGemma feeds the same overlay on
  runs with the Entropy Signal on, since `XAI-01` landed the capture change
  scoped under "Candidate reveal for DiffusionGemma" above. Its canvas does
  brighten roughly together as it nears each adaptive stop; see the settled
  decision recorded from that observation.

  The live half arrived later than this entry did, and the gap is worth
  recording because it made a working feature look broken. When per-token
  spans replaced the character renderer in the streaming view, the new path
  passed no callbacks, deliberately, so that refactor would not change what
  the page looked like. The `opacityFor` hook it added was therefore used
  only by the scrubbed path: a mask graded itself when you scrubbed back over
  a finished run and stayed flat while the run was being written. It surfaced
  on 2026-08-28 on DiffusionGemma, where the number exists only with the
  Entropy Signal on, so the flat live canvas read as the capture change
  failing rather than as a renderer never asking. Only `opacityFor` is wired
  live: `colorFor` has nothing to do while the overlay drawer is hidden, and
  `maskedFor` and `classFor` serve a remask selection that cannot be made
  mid-run.

  **The curve was retuned on 2026-08-29, and it was measured rather than
  eyeballed.** The complaint that started it was that the canvas looked like
  one shade of green with no grading at all. It was grading. Running a saved
  128-step LLaDA run through the shipped curve, a frame's masked positions
  came out at p10 0.48, p50 0.58, p90 0.65 at frame 20, and 0.46 / 0.54 /
  1.00 at frame 78. A 1.35x spread on 14px monospace is not a gradient
  anyone can see.

  Two causes, and the second is the interesting one. A 0.35 floor spent a
  third of the channel before confidence said anything. And the map was
  linear over `[0, 0.4]`, applied to a quantity whose median across that
  whole run sits between 0.11 and 0.21, so nearly every position landed in
  the bottom of what was left. Four candidates were measured on the same two
  frames:

  - Linear, floor 0.35, cap 0.40 (shipped): 0.48 / 0.58 / 0.65.
  - Linear, floor 0.05, cap 1.0: 0.13 / 0.19 / 0.23, with 31 of 120
    positions under 0.15. Not "a few ghost tokens", a canvas that is mostly
    a whisper.
  - Square root, floor 0.05: 0.32 / 0.41 / 0.46, nothing crushed.
  - Linear, floor 0.05, cap 0.60: 0.18 / 0.28 / 0.34.

  The square root shipped. The general point outlives the constants: a
  linear map onto a left-skewed quantity wastes the channel, and any future
  signal drawn this way (entropy, top-k share) should be checked against its
  own distribution before a ramp is chosen for it.

  Two decisions came with it. `overlaysMaskOpacity` moved to `overlays.js`,
  because Analytics had never graded its masks at all: the confidence was in
  the saved file and no render path on that page asked for it, and the same
  was true of the Diff overlay on *both* pages, whose `overlaysDiffColorFor`
  returns null for a masked position specifically to keep masks identical to
  the single-layer views. And absent confidence now draws solid instead of
  falling to the floor. The two used to be conflated, which was harmless at
  0.35 and would have been catastrophic at 0.05: LLaDA's opening frame, every
  run saved before the capture, and every DiffusionGemma run without the
  Entropy Signal carry no number at all, and would have rendered as blank
  canvases. Unmeasured is not the same claim as measured and hopeless.
- Randomize remasks in Edit Frames (slider + N-of-M + Shuffle): seeds the
  meta-explainability question of whether a remask pattern shapes convergence.
- "New run saved" analytics cue: a persisted count badge (generator + Main Menu)
  + per-row dots pointing users to freshly saved runs, cleared per run on open.
  Now backed by durable server-side UI state, decremented on delete, and
  reconciled against existing runs so orphaned ids cannot inflate the count.

Shipped from the durable-overlay data model:
- Analytics per-frame scrubber: the detail modal's token overlay now replays
  every saved frame through None / Heatmap / Commit Order / Diff (the layered
  Diff clamps the original to its final frame past its end), reusing the shared
  overlay math with no re-save. Commit Order and Diff stay diffusion-only.
- Durable Heatmap render: the persisted per-token confidence now drives a
  Heatmap overlay in the analytics viewer (kept for autoregressive runs too).

Still candidate directions:

- Top-k alternatives on hover: show competing candidate tokens and their
  probabilities at a given frame (needs logit capture; heavier, opt-in).
- Per-position confidence/entropy trajectory: a small sparkline of a single
  position's certainty across steps.
- Cross-model comparison on identical prompt and seed: run LLaDA and
  DiffusionGemma (sequentially, since VRAM is exclusive) on the same input and
  compare convergence and confidence side by side in analytics.
- Autoregressive baseline comparison: contrast diffusion against a small
  autoregressive model on the same prompt (also noted in the README extensions).
- Aggregate analytics across saved runs: confidence calibration,
  steps-to-converge distributions, and per-canvas statistics.
- Canvas seeding experiments: pre-fill part of the canvas and watch the model
  complete around it.
- Desktop UX: an in-app "camera" button to capture the current view (diffusion
  output / analytics) to a PNG in ~/Downloads, via the pywebview JS->Python
  bridge (with a browser-mode `<a download>` fallback). Cross-platform and
  avoids depending on per-OS screenshot tools; fidelity is perfect for the token
  output and charts (DOM rasterizers do not render `backdrop-filter`).
- Random prompt generator (deferred; see "Next up" above): a "surprise me" dice
  control that fills the prompt box with a generated prompt. A small
  autoregressive model on CPU (model-agnostic, no contention with the resident
  diffusion model on the 24 GB GPU, fast for a one-liner), but it must run in its
  own concurrent "utility" worker: the supervisor stays torch-free and the main
  manager runs one worker at a time, so the randomizer cannot share the resident
  worker or load in-process. That lifecycle work is why it is deferred behind the
  core AR generation feature.

---

## Where things live (quick map)

- Backend contract: `src/backends/{protocol,registry,worker_base,run_worker}.py`
- Model backends: `src/backends/{llada_worker,dgemma_worker}.py`
- Samplers: `src/inference/{streaming_sampler,dgemma_sampler,dgemma_nf4}.py`
- Supervisor and API: `src/web/server.py`
- Frontend: `src/web/static/{index.html,app.js,analytics.html,analytics.js,style.css}`
- Analytics metrics: `src/analytics/metrics.py`
- Adding a model: add a `ModelInfo` to `registry.py` plus a worker module; the
  frontend and analytics are schema-driven, so most UI follows automatically.
- Environments: LLaDA and the supervisor run in `.venv` (transformers 4.38.2);
  DiffusionGemma runs in `.venv-dgemma` (transformers 5.13). Always use each
  environment's own Python explicitly.
- Dependency files: `requirements.txt` (core `.venv`), `requirements-dgemma.txt`
  (the `.venv-dgemma` env), and `requirements-desktop.txt` (optional pywebview
  desktop add-on for `.venv`). These are flat, fully-pinned freezes, which is
  idiomatic for an ML repo and encodes real structure (two incompatible
  `transformers` universes plus an optional feature layer).

## Dependency management: potential pyproject.toml consolidation

The flat `requirements-*.txt` layout is intentional for now. Consider migrating
to a single `pyproject.toml` with `[project.optional-dependencies]` extras (e.g.
`dgemma`, `desktop`, a future `ar` for autoregressive models) plus a lockfile
(uv / pip-tools) for the transitive pinning the flat freezes currently provide.
That collapses everything into one authoritative file and scales to new groups
without adding files. Triggers to make the switch: (a) the file count would
exceed ~4-5, (b) a new incompatible environment is added (e.g. an autoregressive
model class), or (c) the project is ever packaged/distributed. Until one of
those, the flat files are simpler and reproducibility is already covered.

A `pyproject.toml` now exists, but it is **tool configuration only**: no
`[build-system]` and no `[project]` table, since the app runs from source. It
holds ruff and black, both pinned to 70 columns so a stray `black .` cannot
reflow the whole tree. If the consolidation above ever happens, that file is
where the dependency groups would go, and the tool tables already there stay
as they are.

## References

- `README.md`: feature overview, architecture, and the short public roadmap.
- `.cursor/plans/`: the per-milestone build plans (multi-model architecture,
  interactive remasking, Milestone 4 visualization + VRAM handling).
- Chat transcripts: the detailed design rationale behind the decisions above.
