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

- **Supervisor**: `src/web/server.py` (runs in `.venv`). Serves the **Main Menu**
  at `/` and the generator at `/generate` (gated: redirects to `/` when no model
  is active; `/index.html` 307s to `/generate`). Model Manager spawns ONE worker
  at a time with a pre-flight VRAM check; proxies `/ws` (no auto-boot: it errors
  and closes if no worker is active); serves analytics + save + run-delete;
  auto-stamps HTML asset URLs. `/api/models` also returns `gpu_name` +
  `free_vram_gib` + per-model `fits` for the menu.
- **Workers**: `src/backends/{llada_worker,dgemma_worker}.py` via `run_worker.py`;
  contract in `protocol.py` / `registry.py` / `worker_base.py`. LLaDA → `.venv`
  (transformers 4.38.2); DiffusionGemma → `.venv-dgemma` (transformers 5.13).
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler}.py`; NF4 in
  `dgemma_nf4.py`. Analytics metrics: `src/analytics/metrics.py`.
- **Frontend** (shared, schema-driven): `src/web/static/{menu.html, menu.js,
  index.html, app.js, overlays.js, analytics.html, analytics.js, analytics.css,
  style.css, custom_select.js}` + `assets/title-screen.{webm,mp4}`. `overlays.js`
  holds the shared layered-diff builder (`overlaysBuildDiffLayers`); the old
  `ascii_scene.js` idle animation was removed.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle — uvicorn on an
  ephemeral localhost port on a daemon thread, graceful shutdown frees worker VRAM
  on close; prefers Qt/QtWebEngine via `_select_gui`, falls back to GTK;
  `_set_app_identity` sets GTK prgname + Qt desktopFileName for dock integration).
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop`; `assets/icon.svg`
  is the app icon.

## Recently shipped (this session)

- **Main Menu** at `/` (`menu.html` / `menu.js`): a looping title-screen video
  (`assets/title-screen.webm` with an `.mp4` fallback; WebM decodes in webviews
  that lack H.264) over a model picker showing the detected GPU + free VRAM and
  greying out models that will not fit. Selecting a model activates its worker and
  navigates to `/generate`. Fallback to the animated grid if the video can't play.
- **Generation gated behind model selection**: generator moved to `/generate`
  (`/index.html` 307s there); `serve_generate` redirects to `/` when no model is
  active; the `/ws` proxy no longer auto-boots a default worker. `/api/models`
  gained `gpu_name` + `free_vram_gib` + per-model `fits` (effective-VRAM: a
  resident model's VRAM counts as reclaimable).
- **Consistent header nav**: Menu / Generation / Analytics across pages. On
  Analytics the **Generation** link (`#link-generation`) appears only when a model
  is resident (checked via `/api/models` `active`); stale back-icon and
  empty-state links fixed.
- **Analytics layered Diff vs Original** (#1 shipped): the run modal's Diff
  overlay now uses the shared `overlaysBuildDiffLayers` with Original/Edited
  opacity sliders + a Difference-blend toggle (final frame), matching the
  generator; `app.js` refactored onto the same helper.
- **Removed the idle-animation feature**: deleted `ascii_scene.js`, the donut,
  and the Idle Display setting; the output area's resting state is now a plain
  `showOutputPlaceholder()`.
- Prior sessions (context): durable overlays + analytics token viewer, the wide
  analytics detail modal, guided-edit Confirm/Retry, prompt history + New Run,
  the optional desktop app, and automatic asset cache-busting. See git log /
  `README.md` for detail.

## Conventions

- Use `.venv/bin/python` and `.venv-dgemma/bin/python` explicitly; never system
  Python. Dependency files: `requirements.txt`, `requirements-dgemma.txt`,
  `requirements-desktop.txt`. Follow TigerStyle and `.cursor/rules/`.
- Verify: `.venv/bin/python -m pytest`; `node --check` on changed JS; `py_compile`
  on changed Python; ReadLints. GUI/GPU can't be tested in-sandbox — hand back with
  a manual-verification checklist.
- See `AGENTS.md` for the full workflow, commit discipline, and this handoff habit.

## Where to pick up (items 4-6; items 1-3 shipped this session)

Items 1 (analytics diff sliders), 2/2b (Main Menu route + per-model silos), and 3
(title video) shipped this session. The remaining three are independent polish and
can be tackled in any order. Decisions were settled in earlier deliberation;
confirm scope, then deliberate → Plan → Agent.

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

**Suggested sequencing:** #4/#5/#6 are independent; do them in any order. #5
(randomize remasks) seeds the meta-explainability direction; #4 and #6 are small
UI polish that can land quickly.

## North star & backlog

Generalize the backend contract to host open-source **autoregressive LLMs** and
latent-space probes, reusing the diff/overlay tooling as a cross-model comparison
lens. Standing backlog (`ROADMAP.md`): multi-canvas DiffusionGemma resume; a
per-frame scrubber for the analytics overlays (the saved token stream already
carries every frame); top-k alternatives on hover; per-position entropy sparkline;
autoregressive baseline / random-prompt generator (preferably a tiny CPU-side AR
model in the supervisor — model-agnostic, no GPU contention); in-app
camera/screenshot-to-Downloads button; aggregate analytics across saved runs.
