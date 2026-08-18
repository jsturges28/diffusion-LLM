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
  a stable localhost port `DESKTOP_PORT=8760`, on a daemon thread, graceful
  shutdown frees worker VRAM on close; persistent web-storage profile; prefers
  Qt/QtWebEngine, falls back to GTK).
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop` entry.
  **Single-instance**: a launch asks `/api/app` who holds 8760. Our own
  supervisor means stand down and try to raise that window; anything else
  keeps the ephemeral fallback. Two supervisors used to mean two workers on
  one GPU, which is `LIFE-05`.

## Where things stand

**An audit remediation campaign is the current work**, and it overrides the
normal session cadence. Read `docs/audit/IMPLEMENTATION_BRIEF.md` for how to work it,
then `docs/audit/IMPLEMENTATION_LEDGER.md` for what is done, ready, and blocked.
`docs/audit/AUDIT_REPORT.md` is the immutable analysis behind it; read only the findings
you intend to take, since it is 2,000 lines.

Stages 1 to 3 are complete. Saved runs publish whole or not at all
(`DATA-01`) out of an extracted store (`ORG-01`), declare a schema version and
what they captured (`DATA-05`), carry the worker's own account of what produced
them (`DATA-04`), and render a bounded GIF (`RUNTIME-02`).

**Stage 4, explicit process and socket ownership, has landed its three
passes.** Worker spawning moved to `src/web/worker_process.py` behind a seam
the manager can be tested through, stopping a worker is a verified transition
rather than a signal and a hope (`LIFE-02`), and a switch to a model that
cannot run is refused before the working model is evicted (`LIFE-06`).
Activation then moved behind one shared client, `src/web/static/activation_client.js`
(`ORG-04`), and every activation carries an operation id so two windows
cannot navigate or cancel for each other, with the socket opening on a
`resident` frame naming the model that answered it (`LIFE-03`). Pass three
gave every run a token that a stateful follow-up must name, so one window
cannot resume or probe another's run (`LIFE-01`), and gave every error a
scope, so a probe refused as busy no longer tears down What If
(`PROTOCOL-01`). What remains of the stage is `XAI-01`, `LIFE-04` and
`TRUST-04`. The ledger is the authority and is updated in the same commit as
each change.

**Stage 5, frontend state, has started.** The generator's run state used to
be loose variables in a 7,900-line script: six arrays indexed by frame that
nine separate sites enumerated by hand, a frozen pre-edit copy of four more,
and an eight-value editing phase that ten sites assigned directly.
`src/web/static/run_frames.js` and `run_phases.js` own those now, refusing a
frame family that has fallen out of step and a move between phases that no
button can make, and `model_client.js` gives four pages one reading of
`/api/models`. All three are classic scripts driven from a `vm` in
`tests/web/static/`, like `activation_client.js` before them; the native ES
module conversion the finding also asks for is deliberately a later step
(`ORG-02`), along with the download client and the server-rendered boot
state that would retire the loading overlay.

**Saving is explicit now**, which is a behaviour change worth knowing
before reading the generator. Opening Edit Frames or What If used to write
a full save; it writes nothing. Three things save: the Save button,
Confirm, and the rescue when another window takes the model away. Each
save is published under the run token from `LIFE-01`, so a save whose
reply is lost to a navigation cannot become a second Analytics row.

**Hardware debt, down to two entries.** The queue at the top of
`docs/audit/IMPLEMENTATION_LEDGER.md` holds only the `TRUST-03` offline
retest and `LIFE-02`'s two staged-failure items, 143 and 144, neither of
which blocks anything. One item, 148, is recorded as unreachable on this
hardware rather than pending, because it needs two models resident at once
on a card that cannot hold both. Separately, items 102 to 126 of
`docs/MANUAL_VERIFICATION.md` predate the campaign and have never been
validated.

**One measured limit worth carrying forward.** Autoregressive frames are
full snapshots, so a run's per-token records grow with the square of its
length: a 2047-token SmolLM3 run is about two million of them. It saves
and reads back correctly, taking 30 to 45 seconds to save and around ten
to paint in Analytics, but it exceeds the sessionStorage quota, so
navigating away and back leaves it without its per-token detail and the
save refuses rather than writing a hollowed-out copy. That is
`RUNTIME-01`, and the ceiling is storage rather than the format.

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
