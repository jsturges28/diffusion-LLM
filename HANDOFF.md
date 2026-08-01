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

**NOT yet validated on hardware.** This session landed the `results/` rename,
all of **AR Phase C** (entropy, top-k alternatives, What If substitution), and an
Analytics **Entropy by Position** chart on top of it. In-sandbox checks pass
(`pytest` 54/54, `py_compile` under both `.venv` and `.venv-ar`, `node --check`,
ReadLints clean), and the sampler plus substitution path are covered by new unit
tests against a stub model, but nothing that needs CUDA or a display was
exercised. **See the manual-verification checklist at the bottom.**

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
  frame's column highlighted and a nats readout. This is the **sequence
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
  - `substitutionMarkerPlugin` (modeled on `canvasBoundaryPlugin`) draws dashed
    markers at the union of `token_positions` across `remask_edits`, which for a
    What If branch is the substituted position and therefore the end of the
    shared prefix. Reading `token_positions` rather than `frame_index` is what
    makes it generalize to diffusion remasks.
  - Deliberately **not** given `burnThroughPlugin`: that plugin redraws a
    dataset's line through the tooltip box, which a bar chart has none of, and
    would stroke a stray polyline across the bar tops.

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
  counterfactual. Substitutions chain (the branch becomes the next re-entry
  point). Note the trace key `alternatives` vs the param `alternatives_enabled`:
  they collide if you ever `state.update(params)`.
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
   profile for the same run. Hover names the token and reads nats; zoom, pan,
   Reset, and the eye toggle all behave; the What If branch shows a dashed green
   marker at the substituted position. Then open an AR run saved **before** this
   session: the section should stay hidden rather than drawing an empty chart.
   Switch between two runs without closing the modal to confirm no stale chart
   survives.
7. **Error path**: substitute after switching models and back (which clears
   `last_run_state`). Expect a clean error message and the **original run
   restored**, not a truncated one.
8. **Regressions**: one LLaDA Edit Frames session still works end to end
   (`_stream_tokens` refactor and the `handleError` rollback are the shared
   surfaces), both models still save/load normally, and a diffusion run's detail
   modal still shows exactly its three charts with no Entropy section.

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
