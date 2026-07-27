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
  that lack H.264) over a model picker. Each row is left-stacked (name, then
  `~X GiB` + status, then description) and shows "Available" + green check or
  "Insufficient VRAM" + red X (greyed out); selecting a model shows a "Loading..."
  diffusion cycle, activates the worker, and navigates to `/generate`. Falls back
  to the animated grid if the video can't play.
- **Generation gated behind model selection**: generator moved to `/generate`
  (`/index.html` 307s there); `serve_generate` redirects to `/` when no model is
  active; the `/ws` proxy no longer auto-boots a default worker. `/api/models`
  returns `gpu_name` + `free_vram_gib` + per-model `fits` (effective-VRAM: a
  resident model's VRAM counts as reclaimable) + `gpu_status`.
- **Consistent header nav**: Menu / Generation / Analytics across pages; the
  Analytics **Generation** link appears only when a model is resident.
- **Analytics layered Diff vs Original**: the run modal's Diff overlay uses the
  shared `overlaysBuildDiffLayers` (Original/Edited opacity sliders + Difference
  blend); `app.js` refactored onto the same helper.
- **Removed the idle-animation feature**: deleted `ascii_scene.js` + the donut +
  the Idle Display setting; the resting state is a plain `showOutputPlaceholder()`.
- **Diffusion-style text effect** (opt-in Setting, `appSettings.diffusionText`):
  status messages resolve from block-glyph noise (`denoiseReveal`, per-element
  timers), honoring `prefers-reduced-motion`. A **Mode** sub-setting
  (`diffusionTextMode`) adds "Cycle" (re-diffuse on a loop). Button
  micro-interactions reuse it: Shuffle press (reveal + lagging glow), the
  Generate/New Run **idle cycle** (2s hold; always runs once before the first-ever
  run as a discovery teaser, then follows the setting), and **Lock In** dissolving
  into mask glyphs before it commits.
- **Mask rendering**: `--mask-color` is now the accent green, and mask **opacity
  tracks the model's live predicted confidence** (LLaDA): `streaming_sampler.py`
  emits per-masked-position `true_conf`, and `renderFrameWithTokens` maps it from
  a solid floor up to full as a token nears its reveal (Mapping A). DiffusionGemma
  masks stay solid for now.
- **Analytics "new run" cue**: a shared localStorage set (`overlays.js`) of unseen
  run ids drives a count badge on the generator's Analytics link and a green dot
  per new row in the table; opening a run clears just that one.
- **In-place edited-run save**: an edited/bundled run reuses its pre-edit folder
  (`SaveRunRequest.run_id` -> `_save_run_blocking` updates in place, path-guarded)
  so it is one Analytics row, not two. `saveRun` sends `run_id` + a clean
  `canvas_index`; the session now persists `frameCanvasIndex`/`frameMeanConf` +
  `lastSavedRunId`, and save-success persists LAST (so the round-trip keeps the
  final run id + "Saved to..." status). Guided-edit buttons freeze during any
  save; the standalone Save is disabled during a guided session.
- **Randomize remasks**: Edit Frames "edit" phase has a slider + "N of M" + Shuffle
  (min 1); Edit Frames now opens on Frame 1 (Frame 0 is all-masked).
- **GPU/desktop robustness**: `nvidia-smi` resolved via `shutil.which` + common
  fallbacks with logging (fixes desktop-launch PATH gaps); `_gpu_status` detects a
  driver/library mismatch and the menu says so. `README.md` +
  `install_desktop_entry.sh` note the Qt/X11 `libxcb-cursor0` dependency.
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

## Where to pick up

All six previously-queued items shipped this session (Main Menu + title video +
gate + per-model silos; analytics diff sliders; diffusion-style text; randomize
remasks; the "new run" analytics cue), plus a round of refinements (mask
confidence-opacity, button micro-interactions, in-place edited-run save, GPU/desktop
robustness, and the save-flow fixes). The tree is at a clean, validated stopping
point.

Immediate next: the maintainer has **a couple of small changes** to make before the
next session, and will then hand over **next-session ideas** (to be recorded here
once described). Until then, deliberate any new work with the maintainer first
(deliberate → Plan → Agent).

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
