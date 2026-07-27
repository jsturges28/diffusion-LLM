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
  `free_vram_gib` + per-model `fits` for the menu. Durable UI state
  (`src/web/ui_state.py`, stored in `Results/ui_state.json`) is served via
  `GET`/`PUT /api/ui-state`; the GET reconciles the "new run" cue against
  existing run folders so deleted runs cannot inflate the count.
- **Workers**: `src/backends/{llada_worker,dgemma_worker}.py` via `run_worker.py`;
  contract in `protocol.py` / `registry.py` / `worker_base.py`. LLaDA → `.venv`
  (transformers 4.38.2); DiffusionGemma → `.venv-dgemma` (transformers 5.13).
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler}.py`; NF4 in
  `dgemma_nf4.py`. Analytics metrics: `src/analytics/metrics.py`.
- **Frontend** (shared, schema-driven): `src/web/static/{menu.html, menu.js,
  index.html, app.js, overlays.js, analytics.html, analytics.js, analytics.css,
  style.css, custom_select.js}` + `assets/title-screen.{webm,mp4}`. `overlays.js`
  holds the shared layered-diff builder (`overlaysBuildDiffLayers`), the "new run"
  registry, and the durable-UI-state layer (`persistHydrate` on boot +
  `persistSet` write-through to `/api/ui-state`); the old `ascii_scene.js` idle
  animation was removed.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle — uvicorn on a
  stable localhost port `DESKTOP_PORT=8760` with an ephemeral fallback, on a
  daemon thread, graceful shutdown frees worker VRAM on close; runs a persistent
  (non-private) web-storage profile; prefers Qt/QtWebEngine via `_select_gui`,
  falls back to GTK; `_set_app_identity` sets GTK prgname + Qt desktopFileName for
  dock integration). Note: durable UI state now lives server-side, so it no longer
  depends on the window origin/port.
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop`; `assets/icon.svg`
  is the app icon.

## Recently shipped (this session)

- **Durable server-side UI state** (fixes desktop persistence). QtWebEngine keys
  localStorage by window origin, and the launcher's port varied per run, so
  Settings / the "new run" cue / prompt history / the generate teaser reset every
  restart. They now persist in `Results/ui_state.json` via `GET`/`PUT /api/ui-state`
  (`src/web/ui_state.py`: whitelisted keys, bounded sizes, atomic writes, a lock).
  The frontend hydrates localStorage from the server once on boot
  (`persistHydrate` wraps each page's boot) and write-throughs on change
  (`persistSet`), so the existing synchronous localStorage reads are unchanged.
  State is now shared across the browser and desktop app. Tests in `tests/web/`.
- **New-run cue hardening**: deleting a run now clears its id
  (`overlaysClearNewRun` in `applyDeletions`), and `GET /api/ui-state` reconciles
  the cue against existing run folders (`_reconcile_new_runs`), so an orphaned id
  (a run deleted outside the app, or before this fix) can no longer inflate the
  count. The Main Menu badge now matches the generator's green count pill.
- **Analytics table rework**: columns reordered to Date, Model, Prompt, Time,
  **Edited**, actions (`TABLE_KEYS` + `<thead>`). "Diff vs Original?" renamed to
  **Edited**, rendered as an SVG checkmark filled with a diffusion dot pattern
  (`#edited-dots`, medium-shade density), centered over its column, blank for
  unedited runs. The "new run" dot moved from Prompt to the leading Date column.
- **Bulk delete + row highlight**: checking rows shades them (`row-checked`,
  applied inline on toggle) and shows a trashcan with the selected count in the
  actions header; it deletes all selected via the shared confirm modal
  ("Delete N runs?"), reporting partial failures.
- **Desktop launcher persistence**: `desktop.py` now binds a stable port
  (`DESKTOP_PORT=8760`, with an ephemeral fallback when it is busy) and runs a
  persistent (non-private) web-storage profile. Secondary now that UI state is
  server-side, but it keeps a stable window origin.

## Previously shipped (recent sessions; now in git + README)

- Main Menu + title video + generation gated behind model selection; consistent
  Menu/Generation/Analytics nav; analytics layered Diff vs Original; removed the
  idle-animation feature; opt-in diffusion-style text + button micro-interactions;
  confidence-driven mask opacity (LLaDA); in-place edited-run save; randomize
  remasks; GPU/desktop robustness (`nvidia-smi` resolution, `libxcb-cursor0`).
- Earlier: durable overlays + the analytics token viewer, the wide detail modal,
  guided-edit Confirm/Retry, prompt history + New Run, the optional desktop app,
  and automatic asset cache-busting. See git log / `README.md` for detail.

## Conventions

- Use `.venv/bin/python` and `.venv-dgemma/bin/python` explicitly; never system
  Python. Dependency files: `requirements.txt`, `requirements-dgemma.txt`,
  `requirements-desktop.txt`. Follow TigerStyle and `.cursor/rules/`.
- Verify: `.venv/bin/python -m pytest`; `node --check` on changed JS; `py_compile`
  on changed Python; ReadLints. GUI/GPU can't be tested in-sandbox — hand back with
  a manual-verification checklist.
- See `AGENTS.md` for the full workflow, commit discipline, and this handoff habit.

## Where to pick up

This session shipped durable server-side UI state (fixing desktop persistence
once and for all), the analytics table rework (Edited column + diffusion-textured
checkmark, reordered columns, checkbox row highlight, multi-select bulk delete),
and new-run-cue hardening (decrement on delete + server-side reconcile). All
maintainer-validated; the tree is at a clean stopping point.

Immediate next: the maintainer will hand over **next-session ideas**, to be
deliberated and then recorded here (and in `ROADMAP.md`) before Plan mode
(deliberate → Plan → Agent). Update this section once the plan is settled.

Standing candidate directions (from the backlog below), none committed:
- Multi-canvas DiffusionGemma resume (the remaining Phase 2 milestone).
- Per-frame scrubber for the analytics overlays (saved token stream carries every
  frame; the layered diff currently renders the final frame only).
- Mask confidence-opacity for DiffusionGemma (LLaDA shipped; dgemma uses a
  different signal, so it is a separate, heavier effort).
- xAI: top-k alternatives on hover; per-position entropy sparkline; cross-model
  comparison on identical prompt/seed; autoregressive baseline.
- Random-prompt generator (tiny CPU-side AR model in the supervisor) and an in-app
  camera/screenshot-to-Downloads button.

## North star & backlog

Generalize the backend contract to host open-source **autoregressive LLMs** and
latent-space probes, reusing the diff/overlay tooling as a cross-model comparison
lens. Standing backlog (`ROADMAP.md`): multi-canvas DiffusionGemma resume; a
per-frame scrubber for the analytics overlays (the saved token stream already
carries every frame); top-k alternatives on hover; per-position entropy sparkline;
autoregressive baseline / random-prompt generator (preferably a tiny CPU-side AR
model in the supervisor — model-agnostic, no GPU contention); in-app
camera/screenshot-to-Downloads button; aggregate analytics across saved runs.
