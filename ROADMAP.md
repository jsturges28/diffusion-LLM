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
  below, via seed-canvas re-entry); two xAI overlays — commit-order
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
- For the feature overview and architecture, see `README.md`. For the build
  history, see `.cursor/plans/`.

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
- "New run saved" analytics cue: a persisted count badge + per-row dots pointing
  users to freshly saved runs, cleared per run on open.

Follow-ups unlocked by the durable-overlay data model:
- Analytics scrubber: the layered Diff vs Original (Original/Edited opacity
  sliders + difference blend) now ships in the analytics viewer on the final
  frame. Since the saved per-token stream carries every frame, a per-frame
  scrubber for the analytics overlays (commit-order / diff across frames)
  remains an additive follow-up, with no re-save.
- Durable Heatmap render: per-token confidence is now persisted, so a
  confidence heatmap in the analytics viewer is a render-only addition.

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
- Random prompt generator: a "surprise me" control that fills the prompt box
  with a generated prompt. Preferred approach is a tiny autoregressive model run
  on CPU in the supervisor (model-agnostic, no contention with the resident
  diffusion model on the 24 GB GPU, fast for a one-liner) rather than the
  resident diffusion model (which would be clunky/slow, especially DiffusionGemma).
  A natural first toe-dip toward the north-star of hosting autoregressive models.

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
