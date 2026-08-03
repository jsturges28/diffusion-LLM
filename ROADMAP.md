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
  below, via seed-canvas re-entry); two xAI overlays: commit-order
  (resolution-step) token coloring and the counterfactual "Diff vs Original"
  comparison (opacity sliders + difference blend); a grouped overlay picker
  (None / Heatmap / Diff) with persistent per-browser settings (highlight tokens,
  commit order) behind staged Save/Reset; and analytics run deletion (confirm
  modal + toast) with contained, toggleable chart tooltips (line burn-through).
- Shipped (this session): durable xAI overlays. Per-token records (display
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
- For the feature overview and architecture, see `README.md`. For the build
  history, see `.cursor/plans/`.

---

## Next session (accepted directions)

Agreed with the maintainer (deliberate each in Ask mode before Plan). (The
`results/` rename and all of AR Phase C shipped this session; see above.)

1. **State-space models: Mamba-3 (new model class).** Integrate a 1.5B Mamba-3
   SISO / MIMO checkpoint (`state-spaces` HF org, arXiv 2603.15569) as the first
   SSM, opening a distinct xAI lens: the fixed-size recurrent state. Key open
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

**Phase C (shipped this session).** The AR xAI trio, built in dependency order
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

## Experimental / xAI feature backlog (to deliberate)

The suite is shaping up as an explainability playground, so these are candidate
directions to scope together before building. None are committed. For any that
capture heavy signals (logits, entropy), prefer the established pattern:
cheap-by-default with an opt-in for the expensive path (as with the Entropy
signal toggle).

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
  visible live and while scrubbing. DiffusionGemma is a separate follow-up.
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

## References

- `README.md`: feature overview, architecture, and the short public roadmap.
- `.cursor/plans/`: the per-milestone build plans (multi-model architecture,
  interactive remasking, Milestone 4 visualization + VRAM handling).
- Chat transcripts: the detailed design rationale behind the decisions above.
