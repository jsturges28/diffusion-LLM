# Roadmap and Feature Backlog

A single, living reference for planned and candidate features, with enough
implementation detail to pick any of them up cold. This complements the short
Roadmap section in `README.md` (the public-facing summary) and the build plans
in `.cursor/plans/` (the historical, per-milestone plans). The deepest design
rationale still lives in the chat transcripts; this document is meant to capture
the parts worth keeping out of the chats.

## How to use this document

- Treat it as a living doc: update it whenever scope, priorities, or decisions
  change (that is explicitly allowed here regardless of any plan-execution
  boilerplate).
- Each planned feature lists: goal, current state, technical approach and hooks,
  files likely touched, open questions and risks, and a definition of done.
- The experimental section is a backlog of candidate directions to deliberate on
  before committing, not a queue of accepted work.

## Current status (orientation)

Phase 1 is complete: both models run locally with live visualization and the
analytics suite.

- Shipped: multi-model supervisor/worker architecture (process isolation),
  LLaDA (bf16) and DiffusionGemma (self-quantized NF4), per-token confidence and
  the confidence heatmap, analytics (convergence, timing, confidence, plus
  canvas-boundary markers), reproducibility metadata, graceful VRAM handling, and
  the DiffusionGemma thinking-mode split view.
- Shipped (latest session): DiffusionGemma single-canvas remask/resume (Phase 2
  below, via seed-canvas re-entry); two xAI overlays: commit-order
  (resolution-step) token coloring and the counterfactual "Diff vs Original"
  comparison (opacity sliders + difference blend); a grouped overlay picker
  (None / Heatmap / Diff) with persistent per-browser settings (highlight tokens,
  commit order) behind staged Save/Reset; and analytics run deletion (confirm
  modal + toast) with contained, toggleable chart tooltips (line burn-through).
- Shipped (this session): durable xAI overlays. Per-token records (display
  text, mask flag, vocab id, confidence) and the pre-edit snapshot are now
  persisted per run (`tokens.json` / `original_tokens.json`), the overlay math
  is shared between pages (`src/web/static/overlays.js`), and the Analytics
  Suite gained a static commit-order / Diff-vs-Original token viewer gated on
  data availability. Persisting confidence also makes a future durable Heatmap
  render a data-free follow-up.
- Shipped (analytics + edit-flow refinements): the run detail is now a wide
  fade-in modal (X or click-outside to close) with a corner overlay drawer
  (None / Commit Order / Diff vs Original) mirroring the generator; a sortable
  `Diff vs Original?` column marks runs that carry a pre-edit snapshot; the
  Group By options were pruned to the shared columns; and the guided editor now
  ends on a Confirm/Retry review step, locking Edit Frames once an edited run is
  saved (until the next Generate).
- Shipped (desktop wrapper): an optional pywebview launcher (`desktop.py`) runs
  the UI in a native window and owns the server lifecycle (starts uvicorn on an
  ephemeral localhost port, graceful shutdown frees worker VRAM on close), plus
  a Linux app-menu entry generator (`scripts/install_desktop_entry.sh`) and an
  app icon (`assets/icon.svg`). The browser path (`main.py`) is unchanged, so
  there is no dual maintenance. Cross-platform packaging (AppImage / Windows /
  macOS) remains deferred and is gated more by the CUDA/torch stack than by the
  webview layer.
- Shipped (menu + shell): a **Main Menu** landing page at `/` (a looping
  title-screen video, WebM with an MP4 fallback, over a GPU/VRAM-aware model
  picker that greys out models that will not fit). Generation moved to
  `/generate` and is now **gated behind model selection**: a direct hit with no
  active model redirects to the menu, and the `/ws` proxy no longer auto-boots a
  default worker. Consistent header nav across pages (Menu / Generation /
  Analytics; the Generation link surfaces only when a model is resident), the
  analytics layered **Diff vs Original** overlay (Original/Edited opacity sliders
  + difference blend, from the #1 backlog item), and removal of the old
  idle-animation feature (ASCII scene + donut + the Idle Display setting) in
  favor of a plain output placeholder.
- Shipped (this session, continued): the remaining backlog polish plus a round
  of refinements. An opt-in **diffusion-style text** effect (status messages
  resolve from block-glyph noise; a Default/Cycle Mode sub-setting; honors
  reduced motion) reused for button micro-interactions (Shuffle press, the
  Generate/New Run idle cycle with a one-time discovery teaser, and Lock In
  dissolving into mask glyphs). **Confidence-driven mask rendering**: masks are
  the accent green and their opacity tracks the model's live predicted confidence
  for LLaDA (`streaming_sampler.py` emits per-masked-position confidence), rising
  from a solid floor to full as a token nears its reveal. A **randomize-remasks**
  control (slider + N-of-M + Shuffle) in Edit Frames, which now opens on Frame 1.
  An Analytics **"new run" cue**: a persisted set of unseen runs drives a count
  badge on the generator's Analytics link and a per-row green dot cleared on open.
  **In-place edited-run save** so an edited run replaces its pre-edit original
  (one Analytics row, not two), with the session persisting canvas/confidence
  arrays + the last run id. **GPU/desktop robustness**: robust `nvidia-smi`
  resolution with logging, a driver/library-mismatch message on the menu, and a
  documented `libxcb-cursor0` (Qt/X11) dependency.
- Shipped (persistence + analytics polish): **durable server-side UI state**.
  Settings, the analytics "new run" cue, prompt history, and the generate
  teaser now persist in `Results/ui_state.json` via `GET`/`PUT /api/ui-state`
  (`src/web/ui_state.py`), hydrated into localStorage on boot (`persistHydrate`
  / `persistSet` in `overlays.js`). This fixes desktop-app persistence, which
  the QtWebEngine profile keyed by the launcher's varying window origin/port; it
  also unifies state across the browser and desktop entry points. The cue is
  reconciled against existing runs on read, so a deleted run cannot inflate the
  count. Plus an **analytics table rework**: reordered columns (Date, Model,
  Prompt, Time, Edited), the renamed **Edited** marker as a diffusion-textured
  SVG checkmark (blank when unedited), the "new run" dot moved to the leading
  column, checkbox row highlighting, and multi-select **bulk delete**. Desktop
  launcher now uses a stable port (ephemeral fallback) and a persistent
  web-storage profile.
- Shipped (this session): the **first autoregressive model**, SmolLM3-3B, in a
  dedicated `.venv-ar` (Phase A below). Token-by-token streaming with per-token
  sampling confidence (`src/inference/ar_sampler.py`,
  `src/backends/smollm3_worker.py`); a `model_type` capability flag
  (`protocol.py`) that gates diffusion-only UI (Edit Frames, Diff overlay,
  Commit Order, convergence) off while keeping timing, confidence, and the
  Heatmap; per-activation CPU/GPU device selection threaded from the Main Menu
  through `run_worker.py` into `Backend.load(device=...)`, with the GPU
  pre-flight skipped on CPU and a CPU-capable torch wheel so GPU-less hosts can
  run it.
- Shipped (this session): the **menu + model-switch UX pass** on top of Phase A,
  validated on hardware. Non-blocking activation with a menu progress bar +
  Cancel; signed VRAM-headroom pills (accounting for the reclaimable resident
  model); a "Click to Download" veneer that pre-fetches uncached weights with a
  smooth **disk-size progress poller** (`hf_download.py`, replacing the tqdm hook
  that `snapshot_download` never routes to per-file byte downloads; Xet disabled
  before the first Hub import); select-to-confirm on the menu and dropdown; the
  rename to **LLM Visualizer**; model-family glyphs (diffusion D+F superposition
  with a crisp reversed epsilon; autoregressive @-to-R with a feedback loop); an
  Analytics **Processor** column + per-run timing device name; dropdown polish
  (fixed-width device pill, collapsed-width list with ellipsized names, green/red
  headroom tint, CPU-gated ticker, loaded-model highlight with a locked device);
  a 1-based AR step counter; and orphaned-worker guards (startup sweep +
  `PR_SET_PDEATHSIG`).
- Shipped (this session): the **analytics per-frame scrubber + durable
  Heatmap**, and the **app icon redesign**. The detail modal's token overlay
  gained a frame scrubber that replays every saved frame through the overlays
  (None / Heatmap / Commit Order / Diff), reusing the shared overlay math
  (`src/web/static/overlays.js`) and mirroring the generator; the new
  **Heatmap** recolors by persisted confidence, while Commit Order and Diff
  stay gated to diffusion runs (autoregressive runs get None + Heatmap). This
  was frontend-only, since the frames endpoint already shipped every frame
  (`analytics.js` / `analytics.html` / `analytics.css`). The app icon is now
  three CP437 diffusion shade blocks (`░ ▒ ▓`) as vector dither patterns under
  a corner-to-corner dark-to-bright green denoise gradient (`assets/icon.svg`,
  plus `assets/icon.png` via `scripts/render_icon.py`; `desktop.py` prefers the
  PNG, the launcher keeps the SVG).
- Shipped (this session): the **Settings page + Commit Order overlay**, **menu
  pagination**, and **cross-page download navigation**. Commit Order moved from a
  persistent Settings toggle to a generator overlay-picker option (diffusion-only,
  `app.js`), matching analytics. The generator's Settings modal became a shared
  **`/settings.html`** page (left tab rail: Appearance / Interface) with a gear
  icon in the generator, Main Menu, and Analytics headers; the settings schema
  now lives in `overlays.js` (`SETTINGS_DEFAULTS` / `parseSettings` /
  `settingsEqual`), shared by `app.js` and `settings.js`. The Main Menu model
  list is paginated (`i/N` pager, `menu.js`). A model download now runs as a
  global task the user can navigate away from: a shared draggable, corner-snapping
  toast (`download_toast.js`, persisted via `diffusion_download_toast_corner`)
  surfaces progress/completion when the inline veneer is off-screen; the menu
  re-attaches the veneer on return, `POST /api/models/download/ack` clears the
  terminal state, and `is_repo_cached` / `_has_incomplete` (`hf_download.py`)
  make a partial (`*.incomplete`) cache resume instead of bricking. The
  non-functional Cancel button was removed (real cancellation deferred).
- For the feature overview and architecture, see `README.md`. For the build
  history, see `.cursor/plans/`.

---

## Next session (accepted directions)

Agreed with the maintainer (deliberate each in Ask mode before Plan). The AR
items are detailed further in their own section below. (The analytics scrubber,
app icon, Settings page + Commit Order overlay, menu pagination, and download
navigation all shipped this session; see above.)

1. **Rename `Results/` -> `results/` (mechanical warm-up).** Routes through the
   single `RESULTS_DIR = Path("Results")` constant (`src/web/server.py`), plus
   `.gitignore` and doc/comment references. `Results/` is gitignored, so saved
   runs are untracked (no `git mv`; a local `mv` carries existing history over).
   Open in Plan: the name (`results/` vs the code-native `runs/`) and whether to
   nest it under an ignored parent (e.g. `data/results/`) or keep it at the root.
2. **Autoregressive analysis tools (Phase C).** Leading with the standout, top-k
   "change the last token" resume, then top-k alternatives on hover (shared
   logit capture), then a per-position entropy sparkline for SmolLM3. (See Phase
   C above.)
3. **State-space models: Mamba-3 (new model class).** Integrate a 1.5B Mamba-3
   SISO / MIMO checkpoint (`state-spaces` HF org, arXiv 2603.15569) as the first
   SSM, opening a distinct xAI lens: the fixed-size recurrent state. Key open
   decision, base vs instruct (the `state-spaces` weights are base LMs, not
   instruction-tuned). Own `.venv-ssm`; native `mamba-ssm` / `causal-conv1d`
   CUDA kernels (GPU-only, custom decode loop); ~3 GB VRAM. The streaming
   baseline reuses the AR frame / token contract and `model_type` gating; the
   phase-2 payoff is SSM-native state overlays (per-token Δ / state-write
   intensity, state-norm sparkline, fixed-state forgetting probes), which need
   kernel-intermediate capture. Sequenced after the AR tools.

---

## Autoregressive model support (Phase A shipped; Phase C next)

The north-star direction: host open-source **autoregressive (AR) LLMs** alongside
the diffusion models, reusing the frame/token contract, overlays, and analytics.
The backend contract already generalizes (a model is a `ModelInfo` in
`registry.py` plus a worker implementing `Backend.load` / `handle_generate`), and
the streaming "frame" (a full snapshot plus a `{t, m, id, c}` token list) maps
cleanly onto AR: frame N is the sequence after N generated tokens, every token
`m: false`, `c` = the sampling softmax confidence.

**Phase A (shipped this session).** First AR model end to end:
- **SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B`) in a dedicated **`.venv-ar`**
  (`transformers==4.53.0`, incompatible with LLaDA's 4.38.2), with
  `requirements-ar.txt`. Chosen over Phi-4-mini (standard but less novel) and
  Gemma-3n-E2B (novel MatFormer/effective-2B, but the trickiest first
  integration) for a clean integration whose think/no-think mode reuses the
  existing thinking UI. The torch wheel is the standard CUDA build, so it runs
  on GPU or CPU (not a CPU-only wheel), the code picks the device per
  activation.
- **AR streaming**: a manual token-by-token sampling loop (not HF's text
  streamer) capturing per-token confidence, one full-snapshot frame per new
  token (`ar_sampler.py`). Reuses save, token records, and the scrubber
  (left-to-right replay). Note: full-snapshot frames make the payload O(n^2) in
  tokens; the registry caps the recommended `max_new_tokens` at 256 and the
  worker clamps harder (~128) on CPU.
- **Model-type gate**: `model_type` ("diffusion" | "autoregressive") on
  `ModelCapabilities`; hides diffusion-only UI (Edit Frames, Diff overlay,
  Commit Order, convergence chart), keeps run + timing + confidence + Heatmap
  (the natural per-token AR confidence view).
- **Per-activation CPU/GPU device**: `device` on the activate request threaded
  through `run_worker.py` -> `create_worker_app` -> `Backend.load(device=...)`,
  skipping the VRAM preflight on CPU; a CPU/GPU toggle on the AR menu row that
  greys the GPU option when a GPU is absent or the model will not fit. A
  body-less activate resolves to GPU-if-present-else-CPU server-side.
- **Analytics**: convergence chart dropped for AR (would flatline); timing and
  confidence kept. `model_type` is persisted in each run's `metadata.json`.

Remaining AR follow-ups now live under Phase C below.

**Phase C (if time).**
- **Top-k "change the last token" resume**: the standout AR xAI feature. AR resume
  is truncate-force-continue (easier than diffusion resume), reusing the
  `supports_resume` + resume-message path; top-k capture is opt-in like
  DiffusionGemma's `entropy_signal`.
- Integrate **Phi-4-mini-instruct** and **Gemma-3n-E2B-it**.
- **Download-from-menu**: a curated allowlist of the candidate models; needs a
  dynamic registry layer (registry is a static dict today) + disk-space checks.

**Deferred (future session): Randomize Prompt.** A dice button (left of the
prompt-history button) that fills the prompt via a small always-on CPU model, with
a Settings model dropdown. Deferred because the randomizer must coexist with the
resident model, which the one-worker-at-a-time manager forbids; it needs a
separate concurrent CPU "utility" worker with its own lifecycle. High effort for
the near-term payoff, so the core AR feature comes first.

---

## Phase 2: DiffusionGemma interactive remask and resume

Status: **shipped for single-canvas runs**; multi-canvas resume remaining.

**Goal.** Bring LLaDA's scrubber remask/resume experience to DiffusionGemma:
pick a frame, remask (renoise) tokens, and resume to a later frame or to the
end, with the same guided multi-frame editing flow.

**What shipped (single-canvas).**
- `streaming_resume` in `src/inference/dgemma_sampler.py` re-enters a chosen
  frame's canvas via `decoder_input_ids`, renoises the remasked positions to
  random tokens, and continues denoising under a reduced `max_denoising_steps`
  budget, reusing the frame streamer for capture.
- `DgemmaBackend.handle_resume` in `src/backends/dgemma_worker.py` reconstructs
  the seed canvas from the stored `frame_history`, validates the request, and
  splices the new frames onto the history. `_validate_resume` enforces the
  single-canvas scope (it raises if any frame has `canvas_index > 0`).
- `capabilities.supports_resume=True` for DiffusionGemma in
  `src/backends/registry.py`, so the guided editing UI unlocks. The frontend
  gates **Edit Frames** with `runIsMultiCanvas()` so multi-canvas runs keep it
  disabled, and surfaces the renoise semantics ("nearby tokens may also change")
  in the guided status text.

**Remaining: multi-canvas resume.** This is the hard part and stays open.
- Multi-canvas chaining: a run can span several 256-token canvases; a resume must
  target the correct canvas while preserving already-committed prior canvases.
- Encoder-decoder plus KV cache: re-entering canvas N mid-run means reconstructing
  the decoder state relative to committed canvases, not just seeding one canvas.
- Adaptive stopping: step counts vary per canvas, so a frame index does not map
  to a fixed step schedule.

**Files (for the multi-canvas follow-up).**
- `src/inference/dgemma_sampler.py` (extend the resume generator to seed a
  specific canvas while replaying committed ones)
- `src/backends/dgemma_worker.py` (lift the single-canvas guard in
  `_validate_resume`; track per-canvas frame provenance)
- Frontend is already generic; re-enable **Edit Frames** for multi-canvas once the
  backend supports it.

**Open questions and risks.**
- Cross-canvas resume semantics: resuming within canvas N while canvases before N
  stay committed, without recomputing them.
- Whether the installed `generate()` can seed a non-first canvas cleanly or needs
  a patched entry point for that case.

**Definition of done (single-canvas): met.** Select a single-canvas DiffusionGemma
frame, remask tokens, resume to a later frame or to the end; frames stream
correctly; save and analytics are unaffected; the UI unlocks. Multi-canvas resume
is the remaining milestone.

---

## Phase 3: Multimodal image input

Priority: after Phase 2.

**Goal.** Accept image input to DiffusionGemma (a multimodal-capable base) for
image-grounded text generation.

**Current state.** Text only. `src/backends/dgemma_worker.py` deliberately loads
`AutoTokenizer` (not `AutoProcessor`) to avoid pulling in torchvision.

**Requirements and considerations.**
- `AutoProcessor` plus torchvision for image preprocessing; a vision tower adds
  VRAM on top of the roughly 18 GB NF4 text stack, so the 24 GB budget must be
  re-validated (the vision components may need lower precision or careful
  offload).
- Protocol and UI: extend the `generate` message and add an image-upload control;
  output frames remain text canvases.
- Keep the text-only path intact; image input should be optional.

**Files likely touched.**
- `src/backends/dgemma_worker.py` (AutoProcessor path + image inputs)
- `src/inference/dgemma_sampler.py` (thread pixel inputs through `generate`)
- `src/backends/protocol.py` and the frontend (image-upload control + payload)
- `requirements-dgemma.txt` (pin torchvision)

**Open questions and risks.**
- VRAM headroom for the vision tower alongside the NF4 experts on a 24 GB card.
- Whether the self-quantized NF4 checkpoint retains or needs the vision
  components, or whether they load separately.

**Definition of done.** Upload an image plus a prompt, DiffusionGemma generates
grounded text, VRAM stays within budget, and the text-only path is unaffected.

---

## Experimental / xAI feature backlog (to deliberate)

The suite is shaping up as an explainability playground, so these are candidate
directions to scope together before building. None are committed. For any that
capture heavy signals (logits, entropy), prefer the established pattern:
cheap-by-default with an opt-in for the expensive path (as with the Entropy
signal toggle).

Shipped from this backlog (see `README.md`):
- Token commit-order coloring: tokens are tinted by the step at which they
  resolved (light green early to red-orange late), as a persistent overlay. Now
  durable: reviewable post-hoc in the Analytics Suite from saved per-token data.
- Counterfactual / intervention diff: the "Diff vs Original" overlay compares an
  edited run against the original, with opacity sliders and a difference blend.
  Now durable: the pre-edit snapshot is saved, so the diff is reviewable in the
  Analytics Suite. A propagation heatmap remains a possible extension.
- Confidence-driven mask opacity (LLaDA): a still-masked token's opacity tracks
  the model's live predicted confidence for that position, rising from a solid
  floor to full as it nears the reveal, so the "heating up" before a commit is
  visible live and while scrubbing. DiffusionGemma is a separate follow-up.
- Randomize remasks in Edit Frames (slider + N-of-M + Shuffle): seeds the
  meta-explainability question of whether a remask pattern shapes convergence.
- "New run saved" analytics cue: a persisted count badge (generator + Main Menu)
  + per-row dots pointing users to freshly saved runs, cleared per run on open.
  Now backed by durable server-side UI state, decremented on delete, and
  reconciled against existing runs so orphaned ids cannot inflate the count.

Shipped from the durable-overlay data model:
- Analytics per-frame scrubber: the detail modal's token overlay now replays
  every saved frame through None / Heatmap / Commit Order / Diff (the layered
  Diff clamps the original to its final frame past its end), reusing the shared
  overlay math with no re-save. Commit Order and Diff stay diffusion-only.
- Durable Heatmap render: the persisted per-token confidence now drives a
  Heatmap overlay in the analytics viewer (kept for autoregressive runs too).

Still candidate directions:

- Top-k alternatives on hover: show competing candidate tokens and their
  probabilities at a given frame (needs logit capture; heavier, opt-in).
- Per-position confidence/entropy trajectory: a small sparkline of a single
  position's certainty across steps.
- Cross-model comparison on identical prompt and seed: run LLaDA and
  DiffusionGemma (sequentially, since VRAM is exclusive) on the same input and
  compare convergence and confidence side by side in analytics.
- Autoregressive baseline comparison: contrast diffusion against a small
  autoregressive model on the same prompt (also noted in the README extensions).
- Aggregate analytics across saved runs: confidence calibration,
  steps-to-converge distributions, and per-canvas statistics.
- Canvas seeding experiments: pre-fill part of the canvas and watch the model
  complete around it.
- Desktop UX: an in-app "camera" button to capture the current view (diffusion
  output / analytics) to a PNG in ~/Downloads, via the pywebview JS->Python
  bridge (with a browser-mode `<a download>` fallback). Cross-platform and
  avoids depending on per-OS screenshot tools; fidelity is perfect for the token
  output and charts (DOM rasterizers do not render `backdrop-filter`).
- Random prompt generator (deferred; see "Next up" above): a "surprise me" dice
  control that fills the prompt box with a generated prompt. A small
  autoregressive model on CPU (model-agnostic, no contention with the resident
  diffusion model on the 24 GB GPU, fast for a one-liner), but it must run in its
  own concurrent "utility" worker: the supervisor stays torch-free and the main
  manager runs one worker at a time, so the randomizer cannot share the resident
  worker or load in-process. That lifecycle work is why it is deferred behind the
  core AR generation feature.

---

## Where things live (quick map)

- Backend contract: `src/backends/{protocol,registry,worker_base,run_worker}.py`
- Model backends: `src/backends/{llada_worker,dgemma_worker}.py`
- Samplers: `src/inference/{streaming_sampler,dgemma_sampler,dgemma_nf4}.py`
- Supervisor and API: `src/web/server.py`
- Frontend: `src/web/static/{index.html,app.js,analytics.html,analytics.js,style.css}`
- Analytics metrics: `src/analytics/metrics.py`
- Adding a model: add a `ModelInfo` to `registry.py` plus a worker module; the
  frontend and analytics are schema-driven, so most UI follows automatically.
- Environments: LLaDA and the supervisor run in `.venv` (transformers 4.38.2);
  DiffusionGemma runs in `.venv-dgemma` (transformers 5.13). Always use each
  environment's own Python explicitly.
- Dependency files: `requirements.txt` (core `.venv`), `requirements-dgemma.txt`
  (the `.venv-dgemma` env), and `requirements-desktop.txt` (optional pywebview
  desktop add-on for `.venv`). These are flat, fully-pinned freezes, which is
  idiomatic for an ML repo and encodes real structure (two incompatible
  `transformers` universes plus an optional feature layer).

## Dependency management: potential pyproject.toml consolidation

The flat `requirements-*.txt` layout is intentional for now. Consider migrating
to a single `pyproject.toml` with `[project.optional-dependencies]` extras (e.g.
`dgemma`, `desktop`, a future `ar` for autoregressive models) plus a lockfile
(uv / pip-tools) for the transitive pinning the flat freezes currently provide.
That collapses everything into one authoritative file and scales to new groups
without adding files. Triggers to make the switch: (a) the file count would
exceed ~4-5, (b) a new incompatible environment is added (e.g. an autoregressive
model class), or (c) the project is ever packaged/distributed. Until one of
those, the flat files are simpler and reproducibility is already covered.

## References

- `README.md`: feature overview, architecture, and the short public roadmap.
- `.cursor/plans/`: the per-milestone build plans (multi-model architecture,
  interactive remasking, Milestone 4 visualization + VRAM handling).
- Chat transcripts: the detailed design rationale behind the decisions above.
