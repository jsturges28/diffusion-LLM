# HANDOFF: starting a session cold

Orientation for whoever picks this up next, human or agent. Read `AGENTS.md`
first for the working conventions, this file for what the project is and where
it stands, then `README.md` and `docs/ROADMAP.md` as needed.

**This page is deliberately bounded.** It used to be 3,233 lines, most of it
session-by-session shipment narrative that every future session paid to read
past. That history is in git, and a test keeps this file under 200 lines so it
cannot grow back. Put durable rationale in `docs/ROADMAP.md`, hardware scenarios in
`docs/MANUAL_VERIFICATION.md`, and shipped features in `README.md`.

## What it is

A local FastAPI + WebSocket visual playground and analytics suite for LLMs,
deepest on discrete diffusion and built to take more model classes over time,
oriented toward explainability (XAI). Runs in the browser (localhost) and as
an optional native desktop app. Watch models denoise live: scrub frame
history, remask tokens and resume, color tokens by confidence, entropy, or
commit order, inspect the candidates a model nearly chose, diff an edited run
(or an autoregressive What If branch) against the original, and compare runs
in an analytics suite.

## Models (one resident at a time)

- **LLaDA-8B-Instruct**: masked discrete diffusion, bf16 (~17GB). Interactive
  remask/resume + guided multi-frame editing.
- **DiffusionGemma-26B-A4B**: block-autoregressive encoder-decoder MoE,
  self-quantized 4-bit NF4 (~18GB), 256-token canvases, adaptive stopping,
  optional "thinking" channel. Single-canvas remask/resume works;
  **multi-canvas resume is NOT done** (Edit Frames disabled for those runs).
  Its checkpoint is a local directory, not a Hub id.
- **SmolLM3-3B**: autoregressive baseline, decoder-only, bf16 (~6GB), in
  `.venv-ar`. Streams token-by-token (one full-snapshot frame per token) with
  per-token sampling confidence and always-on entropy; optional thinking
  channel and a top-5 **Alternatives** capture that is **on by default**
  (`registry.py`), since it is what makes the hover popover and What If work.
  Runs on GPU or CPU (per-activation toggle on the menu), so it is the model a
  GPU-less host can run. No diffusion remask/resume; its counterfactual is
  **What If?** substitution instead (`supports_substitution`).

## Architecture (process isolation; incompatible transformers versions)

- **Supervisor**: `src/web/server.py` (runs in `.venv`). Serves the **Main
  Menu** at `/` and the generator at `/generate` (gated: redirects to `/` when
  no model is active; `/index.html` 307s to `/generate`). Model Manager spawns
  ONE worker at a time with a pre-flight VRAM check; proxies `/ws` (no
  auto-boot: it errors and closes if no worker is active); serves analytics +
  save + run-delete; auto-stamps HTML asset URLs. `/api/models` also returns
  `gpu_name` + `free_vram_gib` + per-model `fits` for the menu. Durable UI
  state (`src/web/ui_state.py`) is served via `GET`/`PUT /api/ui-state`; the
  GET reconciles both the "new run" cue and the Analytics collections against
  existing run folders, so a deleted run can neither inflate the count nor
  linger in a collection as an unopenable row.
- **The data root is explicit.** `src/web/data_root.py` resolves one absolute
  directory at import, defaulting to `<repo>/results` and overridable by
  `--results-dir` or `DIFFUSION_LLM_RESULTS_DIR`. It does not depend on the
  working directory, which it used to.
- **Workers**: `src/backends/{llada_worker,dgemma_worker,smollm3_worker}.py`
  via `run_worker.py`; contract in `protocol.py` / `registry.py` /
  `worker_base.py`. LLaDA to `.venv` (transformers 4.38.2); DiffusionGemma to
  `.venv-dgemma` (transformers 5.13); SmolLM3 to `.venv-ar` (transformers
  4.53). `run_worker.py` takes `--device`, forwarded via
  `create_worker_app(device=...)` into `Backend.load(device=...)` (kw-only,
  default "cuda"). Cached weights load with `local_files_only`, so an
  already-downloaded model activates with no network.
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler,ar_sampler}`;
  NF4 in `dgemma_nf4.py`. Analytics metrics: `src/analytics/metrics.py`.
  `llada_sampler.py` holds live helpers plus a dormant reference program.
- **Frontend** (shared, schema-driven, no framework or bundler):
  `src/web/static/` holds `menu`, `index`/`app`, `analytics`, `settings`, plus
  `overlays.js` for the shared color ramps, the layered-diff builder, the "new
  run" registry, and the durable-UI-state layer. `detail_requests.js` fences
  the Analytics detail panel's fetches. Third-party chart libraries and the
  webfont are vendored under `static/vendor/`, so every page works offline.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle: uvicorn on
  a stable localhost port `DESKTOP_PORT=8760` with an ephemeral fallback, on a
  daemon thread, graceful shutdown frees worker VRAM on close; persistent
  web-storage profile; prefers Qt/QtWebEngine, falls back to GTK).
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop` entry.

## Where things stand

**An audit remediation campaign is the current work**, and it overrides the
normal session cadence. Read `docs/audit/IMPLEMENTATION_BRIEF.md` for how to work it,
then `docs/audit/IMPLEMENTATION_LEDGER.md` for what is done, ready, and blocked.
`docs/audit/AUDIT_REPORT.md` is the immutable analysis behind it; read only the findings
you intend to take, since it is 2,000 lines.

Stages 1 and 2 are complete, and so is stage 3, the run-store boundary. Saved
runs now publish whole or not at all (`DATA-01`) out of an extracted store
(`ORG-01`), declare a schema version and what they captured (`DATA-05`), carry
the worker's own account of what produced them (`DATA-04`), and render a
bounded GIF (`RUNTIME-02`). Stage 4 is next; the ledger is the authority on all
of this and is updated in the same commit as each change.

**Hardware debt.** The campaign's queue at the top of
`docs/audit/IMPLEMENTATION_LEDGER.md` holds one entry, the `TRUST-03` offline
slice. Separately, items 102 to 126 of `docs/MANUAL_VERIFICATION.md` predate
the campaign and have never been validated.

## Conventions

- Three virtualenvs, one per model environment; never system Python. See
  `AGENTS.md` for which command goes where.
- Coding standard: `docs/TIGERSTYLE.md`. Enforced numbers live in `pyproject.toml`.
- Verification before handing back: `.venv/bin/python -m pytest`,
  `.venv/bin/python scripts/lint_ratchet.py`, `node --check` on changed JS and
  `node --test tests/web/static/*.test.js`. Full list in `AGENTS.md`.
- GPU and display work cannot be exercised in an agent sandbox. Hand it back
  with a manual checklist.

## Where to pick up

`docs/audit/IMPLEMENTATION_LEDGER.md` answers this during the campaign. After it, the
agreed feature order is **Mamba-3** as a new model class, then extending
entropy and top-k to the diffusion models; both have prerequisites the ledger
lists, and both want deliberating before Plan. `docs/ROADMAP.md` carries the
settled decisions, the deliberate stopping points, and the longer backlog.
