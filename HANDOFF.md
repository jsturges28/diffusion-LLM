# HANDOFF: next session

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

## Models (one resident at a time)

- **LLaDA-8B-Instruct**: masked discrete diffusion, bf16 (~17GB). Interactive
  remask/resume + guided multi-frame editing.
- **DiffusionGemma-26B-A4B**: block-autoregressive encoder-decoder MoE,
  self-quantized 4-bit NF4 (~18GB), 256-token canvases, adaptive stopping,
  optional "thinking" channel. Single-canvas remask/resume works; **multi-canvas
  resume is NOT done** (Edit Frames disabled for multi-canvas runs, deferred).
- **SmolLM3-3B**: autoregressive baseline, decoder-only, bf16 (~6GB), in
  `.venv-ar`. Streams token-by-token (one full-snapshot frame per token) with
  per-token sampling confidence; optional thinking channel. Runs on GPU or CPU
  (per-activation toggle on the menu), so it is the model a GPU-less host can
  run. No resume; diffusion-only UI is gated off (see below).

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
  holds the shared layered-diff builder (`overlaysBuildDiffLayers`), the "new run"
  registry, and the durable-UI-state layer (`persistHydrate` on boot +
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

**Validated on hardware this session** (the maintainer confirmed the menu,
download, dropdown, CPU/GPU, and AR flows working). In-sandbox checks
(`py_compile`, `node --check`, `pytest`, ReadLints) also pass.

**Dropdown, ticker, glyphs, and download (final polish pass).**

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

- Durable server-side UI state (`Results/ui_state.json` via `/api/ui-state`,
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

This session shipped and **validated on hardware** Phase A (SmolLM3), the menu
VRAM-headroom / download / UX pass, and a final polish pass (dropdown, glyphs,
disk-size download poller, loaded-model affordance, AR step fix). The next
session's focus, agreed with the maintainer, is below. Deliberate each in Ask
mode before Plan.

**1 + 2. Shared Settings page + gear icon (one coupled feature).** Promote
Settings from the generator-only modal (`index.html:371-429`, logic in `app.js`)
to a **shared `/settings.html` page** with a **left tab rail + right pane** (the
list is growing: 5 rows + a conditional "Mode" sub-row). Both entry points link
to it: replace the header "Settings" text (`index.html:40`, `#link-settings`)
with a **gear icon**, and add a matching gear entry to the Main Menu nav
(`menu.html:66-68`, which today only links to Analytics). All settings are
already global and server-persisted (`ui_state.py`), so a page (not a duplicated
modal) is the clean fit. Tab grouping to settle in Plan; a first cut: Appearance
(diffusion-style text + its Mode sub-row, highlight tokens), Overlays (commit
order), Interface (device-tag ticker), with room for future Models/Analytics
tabs. Keep the staged Save/Reset shared across tabs; new gear glyph in the
family-icon style.

**3. App icon redesign.** Replace the token-grid `assets/icon.svg` with **three
denoising shade-block glyphs (`▓ ▒ ░`, the `DENOISE_GLYPHS` set in `app.js:3119`
/ `menu.js:499`) fading most to least opaque**, evoking a token mid-denoise (not
the D+F family logo). Keep the dark rounded tile + accent green; export a PNG
alongside the SVG for dock / window-icon fidelity. Refs: `assets/icon.svg`,
`desktop.py:52` (`ICON_PATH`), `scripts/install_desktop_entry.sh`.

**4. Analytics token scrubber (+ tooltip parity).** The detail modal already
renders a static final-frame token overlay with a hover tooltip (position +
confidence); add a **per-frame scrubber** so any saved frame can be reviewed
(the data is already there: `tokens.json` carries every frame, served via
`load_run_frames`). Reuse the generator's scrubber / overlay logic (`app.js` /
`overlays.js`) so Commit Order / Diff overlays scrub per frame too. Explicitly
**no Edit Frames** in analytics. Lowest-risk, high-value; already the "analytics
scrubber" follow-up in `ROADMAP.md`.

**5. Autoregressive analysis tools (Phase C).** SmolLM3 has no Edit Frames (it
is left-to-right), so add AR-native xAI. Suggested order: (a) **top-k
alternatives on hover** (competing tokens + probabilities per position; opt-in
logit capture in `ar_sampler.py`, mirroring DiffusionGemma's `entropy_signal`);
(b) **top-k "change the last token" resume** (truncate-force-continue: set
`supports_resume=True` and add `handle_resume` in `smollm3_worker.py`, reusing
the resume path + scrubber); (c) a **per-position entropy sparkline**. Mind the
documented O(n^2) AR frame-payload caveat for long runs. See `ROADMAP.md` Phase C.

**Standing backlog (unchanged; see `ROADMAP.md`):** multi-canvas DiffusionGemma
resume (the remaining Phase 2 milestone); mask confidence-opacity for
DiffusionGemma; integrate Phi-4-mini-instruct / Gemma-3n-E2B-it (share
`ar_sampler.py` + `.venv-ar`); download-from-menu for a curated candidate
allowlist (needs a dynamic registry layer); cross-model comparison on identical
prompt/seed; aggregate analytics across runs; the deferred Randomize-Prompt
utility worker.

**Remaining nice-to-have from this session:** confirm SmolLM3's thinking
delimiter on a real thinking-on run (`ar_sampler.py` splits on `<think>` /
`</think>`; adjust `_split_thinking` / `_STRIP_TOKENS` if the tags differ), and
`.venv-ar/bin/pip freeze > requirements-ar.txt` to capture the full transitive
pins.

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
