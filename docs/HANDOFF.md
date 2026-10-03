# HANDOFF: starting a session cold

Orientation for whoever picks this up next, human or agent. Read `AGENTS.md`
first for the working conventions, this file for what the project is and where
it stands, then `README.md`, `docs/GUIDE.md` and `docs/ROADMAP.md` as needed.

**This page is deliberately bounded.** It used to be 3,233 lines, most of it
session-by-session shipment narrative that every future session paid to read
past. That history is in git, and a test keeps this file under 200 lines so it
cannot grow back, as one now does for `README.md`. Put durable rationale in
`docs/ROADMAP.md`, hardware scenarios in `docs/MANUAL_VERIFICATION.md`, feature detail in `docs/GUIDE.md`, and what exists in the README.

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
- **Mamba-3-1.5B**: state-space model, a base checkpoint that continues text,
  float32 (~6GB) on GPU or CPU, in `.venv-ar` on our own PyTorch. Same
  sampler, handlers and affordances as SmolLM3 (What If replays the prefix,
  since a state cannot be sliced), plus one signal, per-token **forgetting**
  (`f`), drawn by the Forgetting overlay. Reads Llama 3.1's tokenizer from
  SmolLM3's pinned repository, fetched into a companion cache.

## Architecture (process isolation; incompatible transformers versions)

- **Supervisor**: `src/web/server.py` (runs in `.venv`). Serves the **Main
  Menu** at `/` and the generator at `/generate` (gated: redirects to `/` when
  no model is active; `/index.html` 307s to `/generate`). Model Manager spawns
  ONE worker at a time with a pre-flight VRAM check; proxies `/ws` (no
  auto-boot: it errors and closes if no worker is active); serves analytics +
  save + run-delete, and the Vision page's `/api/vision/*`, which needs no
  worker; auto-stamps HTML asset URLs. `/api/models` also returns
  `gpu_name` + `free_vram_gib` + per-model `fits` for the menu. Durable UI
  state (`src/web/ui_state.py`) is served via `GET`/`PUT /api/ui-state`; the
  GET reconciles both the "new run" cue and the Analytics collections against
  existing run folders, so a deleted run can neither inflate the count nor
  linger in a collection as an unopenable row.
- **The data root is explicit.** `src/web/data_root.py` resolves one absolute
  directory at import, defaulting to `<repo>/results` and overridable by
  `--results-dir` or `DIFFUSION_LLM_RESULTS_DIR`. It does not depend on the
  working directory, which it used to.
- **Workers**: `src/backends/{llada,dgemma,smollm3,mamba3}_worker.py`
  via `run_worker.py`; contract in `protocol.py` / `registry.py` /
  `worker_base.py`, and the two left-to-right workers share
  `append_only_backend.py`. LLaDA to `.venv` (transformers 4.38.2);
  DiffusionGemma to `.venv-dgemma` (transformers 5.13); SmolLM3 and Mamba-3
  to `.venv-ar` (transformers 4.53; Mamba-3 is plain PyTorch). `run_worker.py` takes `--device`, forwarded via
  `create_worker_app(device=...)` into `Backend.load(device=...)` (kw-only,
  default "cuda"). Cached weights load with `local_files_only`, so an
  already-downloaded model activates with no network.
- **Samplers**: `src/inference/{streaming_sampler,dgemma_sampler,ar_sampler}`;
  NF4 in `dgemma_nf4.py`. `mamba3_causal.py` gives Mamba-3 the calling shape
  `ar_sampler` drives. Analytics metrics: `src/analytics/metrics.py`.
  LLaDA's algorithm is `llada_kernel.py`, its old twin quarantined under
  `reference/llada/` behind a differential test (`ORG-03`).
- **Frontend** (shared, schema-driven, no framework or bundler):
  `src/web/static/` holds `menu`, `index`/`app`, `analytics`, `settings` and
  `vision`, plus `overlays.js` for the shared color ramps, the layered-diff
  builder, the "new run" registry and the durable-UI-state layer.
  `detail_requests.js` fences the Analytics detail panel's fetches and, through
  a second instance, the compare panel's. Third-party chart libraries and the
  webfont are vendored under `static/vendor/`, so every page works offline.
- **Analytics reads** are split by cost: `/api/analytics/runs` carries only
  what the table draws (about 326 bytes a run), and the prompt, parameters
  and per-frame arrays are fetched per run from `/runs/{id}/metadata` when
  one is opened. Convergence counts token positions from `tokens.json` and
  reports the basis it used, since older runs without those records fall
  back to counting characters.
- **Desktop**: `desktop.py` (pywebview; owns the server lifecycle: uvicorn on
  a stable localhost port `DESKTOP_PORT=8760`, on a daemon thread, graceful
  shutdown frees worker VRAM on close; persistent web-storage profile; prefers
  Qt/QtWebEngine, falls back to GTK).
  `scripts/install_desktop_entry.sh` generates a Linux `.desktop` entry.
  **Single-instance**: a launch asks `/api/app` who holds 8760, standing down
  to raise that window if our own supervisor answers. Two supervisors may now
  coexist but only one may hold a model, by the `src/web/model_lease.py` flock
  (`LIFE-05`).

## Where things stand

**A second audit is being worked through.** Its 21 findings are in
`docs/audit/AUDIT_REPORT_2026-10.md` and their state is in
`docs/audit/IMPLEMENTATION_LEDGER_2026-10.md`: Stage 1 is done, and Stage 2,
cross-supervisor ownership, is under way. Read the ledger first, then only
the findings you touch. **The first campaign is
complete except for a short remainder**, listed at the top of
`docs/audit/IMPLEMENTATION_LEDGER.md`: three findings waiting on hardware,
`ORG-02`'s module conversion deferred with its reason, and `ROADMAP-04`
untaken because nothing needs it yet. `docs/audit/IMPLEMENTATION_BRIEF.md`
no longer governs every session, but the rules it quotes from the first
report's sequencing still bind new work, and `docs/audit/AUDIT_REPORT.md`
remains the immutable analysis behind that campaign.

What the first campaign changed that a newcomer trips over:

- **Runs are owned.** Saved runs publish whole or not at all, declare a
  schema version and what they captured, and carry the worker's own
  account of what produced them (`DATA-01`, `DATA-05`, `DATA-04`). Every
  run has a token a stateful follow-up must name, so one window cannot
  resume or probe another's run (`LIFE-01`), and every error has a scope
  (`PROTOCOL-01`).
- **Workers are owned.** Spawning sits behind `src/web/worker_process.py`,
  stopping one is a verified transition (`LIFE-02`), and a switch that
  cannot work is refused before anything is evicted (`LIFE-06`). Every
  activation carries an operation id and the socket opens on a `resident`
  frame (`LIFE-03`). A run is stoppable, and every model ends a stopped run
  with one `done` carrying `cancelled` (`LIFE-04`). Downloads are child
  processes the supervisor can terminate (`TRUST-04`).
- **Edits are faithful.** Both diffusion backends keep a bounded per-frame
  checkpoint, so an edited branch reports the confidence the model actually
  gave, and one edit repeats across intervening random work (`XAI-01`).
- **Models are described rather than special-cased.** `model_type` is a
  `family`, a `generation_shape` and an `input_mode`, with devices declared,
  so both diffusion models are honestly GPU-only (`ROADMAP-01`). One
  resolver answers for every model's parameters (`ROADMAP-02`), a per-model
  text adapter owns templating and stop tokens (`ROADMAP-05`), and every
  signal declares its unit and the axes it varies over (`ROADMAP-03`).
- **The generator's state has owners.** `run_frames.js` and `run_phases.js`
  refuse a frame family out of step and a phase move no button can make,
  and pages open with their state inlined as `window.__BOOT__` rather than
  fetching it (`ORG-02`).
- **Saving is explicit.** Opening Edit Frames or What If writes nothing.
  Three things save: the Save button, Confirm, and the rescue when another
  window takes the model away, each published under the run token so a
  lost reply cannot become a second Analytics row.
- **Autoregressive frames are append-only**, on the wire, in the browser
  and on disk, which took a 2,048-token run from about 130 MiB to 1 MiB
  (`RUNTIME-01`). Diffusion frames are still full canvases, which
  denoising one makes inherent.
- **The documentation is held to the code.** `tests/test_docs_inventory.py`
  fails when a model, environment, package or page ships that the docs do
  not name (`META-03`), and this page, the README and the roadmap's
  orientation are each bounded by a test.

**The Vision page is the newest surface, and the odd one out.** It runs in
the supervisor rather than a worker, reads two small config files per
SmolVLM encoder at a pinned commit, and never loads weights, so it works
whether or not a model is resident and never disturbs one that is. Its
measurements, and the evidence that ruled out an attention overlay, are in
`docs/ROADMAP.md` under "How a vision model sees an image".

**Hardware debt** is recorded in `docs/MANUAL_VERIFICATION.md` under "What
has been checked", which states an outcome for every item. Items 102 to 126
predate the campaign and have never been validated.

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

**Mamba-3** shipped as the fourth model on 2026-09-28, on our own PyTorch
(`src/inference/mamba3.py`) held to upstream's references in
`reference/mamba3/`. Its hardware checks have passed but for downloading
it on a machine that has never fetched it (manual item 330). The
reasoning, including why the tokenizer comes from SmolLM3 and why What If
replays, is under the Mamba-3 direction in `docs/ROADMAP.md`. **Top-k for
the diffusion models** followed on 2026-09-28 as the candidate popover,
which follows the scrubber and pages between an edited run's two runs,
and on 2026-09-29 as the flicker, a Settings choice that cycles each
unsettled position through those candidates on a finished run. The
**revision glow** and the **Revisions** overlay, which mark DiffusionGemma
changing its mind, shipped on 2026-10-01, and the **adaptive-stopping
readout** on 2026-10-02: its two stop thresholds became parameters, and a
readout beside the metrics strip shows how far each canvas is from
stopping. Work in progress follows the second audit's ledger; the next
feature comes from the backlog in `docs/ROADMAP.md` (frame-linked line
charts, per-run notes), which also carries the settled decisions.
