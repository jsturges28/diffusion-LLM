# HANDOFF — next session

Living, per-session handoff. The agent updates this at the end of each session so
the next one can pick up cold (see `AGENTS.md`). Read `README.md` + `ROADMAP.md`
first, then deliberate the work below with the maintainer before Plan mode
(deliberate in Ask mode → Plan → Agent).

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

**Next session (settled): Phase A, core autoregressive support.** Bring the first
autoregressive LLM into the app, reusing the frame/token contract. Decisions are
locked; start with a plan-time check of SmolLM3's required `transformers` version
to pin the new env, then implement.

- **New backend: SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B`) in a dedicated
  **`.venv-ar`** with a CPU-first torch wheel (the model is CPU-primary). Add a
  `ModelInfo` to `src/backends/registry.py` and a worker module implementing
  `Backend.load` / `handle_generate` (`src/backends/worker_base.py:67-94`),
  mirroring the DiffusionGemma worker. New `requirements-ar.txt`.
- **AR streaming as growing-sequence frames**: a manual token-by-token sampling
  loop (not HF's text streamer) so we capture per-token confidence, emitting one
  frame per new token in the existing `{t, m, id, c}` shape (`m` always false,
  `c` = sampling softmax confidence). Reuses streaming, save, token records, and
  the scrubber (as a left-to-right replay) unchanged. Note: full-snapshot frames
  make payload O(n^2) in tokens; fine for a few hundred, revisit for long runs.
- **Model-type gate**: add `model_type` ("diffusion" | "autoregressive") or an
  `is_autoregressive` flag to `ModelInfo`/`ModelCapabilities` (`protocol.py:45-70`)
  and branch the diffusion-only UI off it in `app.js`: hide Edit Frames
  (already `supports_resume`-gated, `app.js:1970-1975`), the Heatmap/Diff
  overlays, and the analytics convergence chart; keep run + timing + confidence.
- **Per-activation CPU/GPU device selection**: add `device` to the activate
  request (`POST /api/models/{id}/activate`, currently body-less, `server.py`
  ~495-515), thread it through `run_worker.py` into `Backend.load(device=...)`,
  and skip `_preflight_vram` when `device="cpu"`. Menu rows are left-stacked with
  no top-right slot (`style.css:2176-2276`), so add the toggle by making the AR
  row `position: relative` and absolutely positioning the control (like
  `#prompt-history`), greyed out (reuse the `fits` logic) when GPU will not fit.
- **Analytics accommodation**: gate the convergence chart off for AR runs
  (`compute_convergence`, `metrics.py:59-98`, would otherwise flatline); keep
  timing (tokens/sec is a natural AR metric) and confidence.

**Deferred (future session): Randomize Prompt.** A "randomize a short prompt"
dice button left of the prompt-history button (mirror `#btn-prompt-history`,
`index.html:49-81`, confirm/cancel without cycling) driven by an always-on CPU
model, with a Settings model dropdown. Deferred because the randomizer must
coexist with the resident model, which the one-worker-at-a-time manager
(`_stop_locked` on every activate, `server.py:263-314`) forbids: it needs a
separate concurrent CPU "utility" worker with its own lifecycle. High effort for
the payoff right now, so build the bigger AR feature first.

**Phase C (only if time this session):**
- Top-k "change the last token" resume: the standout AR xAI feature. AR resume is
  truncate-force-continue (easier than diffusion resume), slotting into the
  existing `supports_resume` + resume-message path; top-k capture is opt-in like
  DiffusionGemma's `entropy_signal`. Keep the scrubber for this.
- Integrate the other candidates: Phi-4-mini-instruct (standard/reliable) and
  Gemma-3n-E2B-it (novel MatFormer/effective-2B, trickiest).
- Download-from-menu: a curated allowlist of the candidate models; needs a
  dynamic registry layer (registry is a static dict today) + disk-space checks.

Standing candidate directions (from the backlog below), none committed:
- Multi-canvas DiffusionGemma resume (the remaining Phase 2 milestone).
- Per-frame scrubber for the analytics overlays (saved token stream carries every
  frame; the layered diff currently renders the final frame only).
- Mask confidence-opacity for DiffusionGemma (LLaDA shipped; dgemma uses a
  different signal, so it is a separate, heavier effort).
- xAI: per-position entropy sparkline; cross-model comparison on identical
  prompt/seed.

## North star & backlog

Generalize the backend contract to host open-source **autoregressive LLMs** and
latent-space probes, reusing the diff/overlay tooling as a cross-model comparison
lens. Standing backlog (`ROADMAP.md`): multi-canvas DiffusionGemma resume; a
per-frame scrubber for the analytics overlays (the saved token stream already
carries every frame); top-k alternatives on hover; per-position entropy sparkline;
autoregressive baseline / random-prompt generator (a small CPU-only "utility"
model in its own concurrent worker, since the supervisor stays torch-free and the
main manager runs one worker at a time; model-agnostic, no GPU contention); in-app
camera/screenshot-to-Downloads button; aggregate analytics across saved runs.
