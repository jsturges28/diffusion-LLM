# HANDOFF — next session

Living, per-session handoff. The agent updates this at the end of each session so
the next one can pick up cold (see `AGENTS.md`). Read `README.md` + `ROADMAP.md`
first, then deliberate the work below with the maintainer before Plan mode
(deliberate → Plan → Agent).

## What it is

A local FastAPI + WebSocket visual playground and analytics suite for discrete
diffusion LLMs, oriented toward explainability (xAI). Runs in the browser
(localhost) and as an optional native desktop app. Watch models denoise live:
scrub frame history, remask tokens and resume, color tokens by confidence or
commit order, diff an edited run against the original, and compare runs in an
analytics suite.

## Models (one resident in the 24GB GPU at a time)

- **LLaDA-8B-Instruct** — masked discrete diffusion, bf16 (~17GB). Interactive
  remask/resume + guided multi-frame editing.
- **DiffusionGemma-26B-A4B** — block-autoregressive encoder-decoder MoE,
  self-quantized 4-bit NF4 (~18GB), 256-token canvases, adaptive stopping,
  optional "thinking" channel. Single-canvas remask/resume works; **multi-canvas
  resume is NOT done** (Edit Frames disabled for multi-canvas runs — deferred).

## Architecture (process isolation; incompatible transformers versions)

- **Supervisor**: `src/web/server.py` (runs in `.venv`). Model Manager spawns ONE
  worker at a time with a pre-flight VRAM check; proxies `/ws`; serves analytics +
  save + run-delete; auto-stamps HTML asset URLs.
- **Workers**: `src/backends/{llada_worker,dgemma_worker}.py` via `run_worker.py`;
  contract in `protocol.py` / `registry.py` / `worker_base.py`. LLaDA → `.venv`
  (transformers 4.38.2); DiffusionGemma → `.venv-dgemma` (transformers 5.13).
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler}.py`; NF4 in
  `dgemma_nf4.py`. Analytics metrics: `src/analytics/metrics.py`.
- **Frontend** (shared, schema-driven): `src/web/static/{index.html, app.js,
  overlays.js, analytics.html, analytics.js, analytics.css, style.css,
  custom_select.js, ascii_scene.js}`.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle — uvicorn on an
  ephemeral localhost port on a daemon thread, graceful shutdown frees worker VRAM
  on close; prefers Qt/QtWebEngine via `_select_gui`, falls back to GTK;
  `_set_app_identity` sets GTK prgname + Qt desktopFileName for dock integration).
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop`; `assets/icon.svg`
  is the app icon.

## Recently shipped

- **Durable overlays**: per-token records `{t,m,id,c}` + the pre-edit snapshot
  persisted per run (`tokens.json` / `original_tokens.json`); overlay math shared
  in `overlays.js`; a static commit-order / Diff-vs-Original viewer in the
  Analytics run modal.
- **Analytics run detail** is now a wide fade-in modal (X / click-outside) with
  the token-overlay canvas as the centerpiece + a corner overlay drawer (None /
  Commit Order / Diff vs Original), a sortable "Diff vs Original?" column, pruned
  Group By, red trashcans, charts on the right.
- **Guided editing**: Confirm(✓)/Retry(↺) review step after Resume to End; Edit
  Frames auto-saves the original run on entry; Edit Frames locks after an edited
  run is saved (until next Generate); Generate / New Run / Edit Frames all freeze
  during a save; "Saving run…" status + dimmed scrubber during save.
- **Prompt history** (localStorage) with a browse control (‹/› + `i/N` + ✓/✗) at
  the prompt box top-right; **New Run** flow (Generate becomes "New Run" once a run
  is finalized, clearing the canvas/prompt for a fresh start).
- **Optional desktop app** (pywebview), named "LLM XAI Visualizer".
- **Automatic asset cache-busting**: server stamps `?v=<mtime>` on local CSS/JS at
  serve time. No manual `?v` bumps — edit and reload.

## Conventions

- Use `.venv/bin/python` and `.venv-dgemma/bin/python` explicitly; never system
  Python. Dependency files: `requirements.txt`, `requirements-dgemma.txt`,
  `requirements-desktop.txt`. Follow TigerStyle and `.cursor/rules/`.
- Verify: `.venv/bin/python -m pytest`; `node --check` on changed JS; `py_compile`
  on changed Python; ReadLints. GUI/GPU can't be tested in-sandbox — hand back with
  a manual-verification checklist.
- See `AGENTS.md` for the full workflow, commit discipline, and this handoff habit.

## Where to pick up (six items; decisions settled — confirm, then deliberate scope)

**1. Analytics "Diff vs Original" sliders.** Bring the layered diff (Original /
Edited opacity sliders + Difference blend) from the generator into the Analytics
run modal — analytics currently shows only the STATIC final-frame diff. The
`/frames` endpoint already serves both edited + original token streams, so this is
porting the layered render (`app.js` `renderDiffOverlay` / `buildDiffLayerSpans` /
`diffLayerColor` / `#diff-overlay-controls`) — ideally EXTRACTED into shared
`overlays.js` — plus the control row into analytics. Default: sliders on the final
frame; a per-frame scrubber in the modal is an optional stretch (saved data
supports it).

**2. Dedicated Main Menu page (DECIDED: separate page/route, Option B).** A landing
screen at `/` with the running title screen + a centered model-selection modal
(green lettering). Each row shows selectability from GPU/VRAM (`free_vram >=`
registry `min_vram_gib`), a "Checking for GPU and availability…" state while
checking, and the current GPU + free space. Selecting a model enters the
Generation page (blank canvas / light "Output will appear here." text). Move
generation off `/` to its own route; update the server's auto-stamped HTML routes
(`serve_index` etc.) and `desktop.py` (window opens at `/`, now the menu). Extend
`/api/models` (or add `/api/system`) with `gpu_name` + `free_vram_gib` + per-model
`fits` — the supervisor already has `_free_vram_gib()` and `_gpu_name()`. This is
the architectural core; tackle first.

**2b. Per-model silos + agnostic analytics (design principle for #2).** Let the
GENERATION UI specialize per model (already schema-driven via registry
capabilities/param_specs) rather than forcing one model's tooling on another; keep
ANALYTICS model-agnostic for comparison. Clarified: static analytics comparison
reads saved token streams and needs NO model loading, so it always works across
all models' past runs regardless of what's resident; only LIVE cross-model runs
(future) are VRAM-gated (grey out + tooltip). A dedicated global-comparison page is
a plausible later home for the static side.

**3. Title screen video.** The maintainer provided an **MP4** (converted from a
~50MB GIF) at **`src/web/static/assets/title-screen.mp4`** (served at
`/assets/title-screen.mp4`; the `static/` dir is mounted at `/` in
`src/web/server.py`). Wire it via `<video autoplay loop muted playsinline>` so it
**plays on a continuous loop** (`muted` + `playsinline` are required for autoplay to
start in browsers and WebKitGTK), with a graceful fallback (keep/offer the current
`ascii_scene.js` / donut idle animation), and use it as the Main Menu backdrop (#2).
It is committed to the repo (Git LFS if it can't get under ~10MB). **Open decision
(maintainer):** overlay app title text on the video or leave it as-is; lean on
whether the MP4 already carries its own title/branding (if it does, skip the overlay
to avoid clashing; if it is an ambient backdrop, add the app title in the green
terminal palette). Confirm the asset before finishing #2.

**4. "Render diffusion-style text" Settings toggle.** Persist per-browser like the
other settings (`appSettings` / localStorage). A scramble→resolve ("denoising")
effect on dynamic status texts to start (the `startStatusDots` messages:
"Running…", "Saving run…", "Done."), same green palette. Small shared utility;
honor `prefers-reduced-motion`.

**5. Randomize remasks in Edit Frames.** In the remask "edit" state, add a control
to randomly select a subset of resolved token positions into `remaskedPositions`
(then Lock In as usual). DECIDED UI: a **slider plus an "N of M" count input**.
Grounding: `app.js` `remaskedPositions`, `enterRemaskMode` / `beginEditSession`,
`updateGuidedUI` "edit" case, `lockInEdits`. Seeds a meta-explainability direction
(does a remask pattern shape convergence paths?) — feature now, study later.

**6. "New run saved" Analytics cue.** On save-success (`saveRun` `.then`), light up
the header "Analytics" link with a transient rising "+1" and a glowing green "new"
dot badge, to point users to where saved runs live. Clear the dot when Analytics is
opened (session-scoped is fine).

**Suggested sequencing:** settle #2 structure + #3 asset first (they shape the
shell), then #2/#2b, then #1, then #4/#5/#6 (independent polish) in any order.

## North star & backlog

Generalize the backend contract to host open-source **autoregressive LLMs** and
latent-space probes, reusing the diff/overlay tooling as a cross-model comparison
lens. Standing backlog (`ROADMAP.md`): multi-canvas DiffusionGemma resume; top-k
alternatives on hover; per-position entropy sparkline; autoregressive baseline /
random-prompt generator (preferably a tiny CPU-side AR model in the supervisor —
model-agnostic, no GPU contention); in-app camera/screenshot-to-Downloads button;
aggregate analytics across saved runs.
