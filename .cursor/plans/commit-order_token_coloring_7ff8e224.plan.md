---
name: Coloring, DGemma resume, and diff
overview: Three sequenced milestones for the xAI thread - a commit-order color mode (quick win), DiffusionGemma single-canvas interactive remask/resume (Phase 2), and a counterfactual intervention-diff view that lights up once resume exists.
todos:
  - id: m1-html-button
    content: "M1: Add #btn-commit-order button and early-to-late legend in #heatmap-group in index.html"
    status: completed
  - id: m1-js-state
    content: "M1: Replace heatmapOn boolean with colorMode enum ('none'|'conf'|'commit'; 'diff' added in M3) across its ~5 call sites; add btnCommitOrder ref and commitSteps memo"
    status: completed
  - id: m1-js-compute-color
    content: "M1: Add computeCommitSteps() (settle-step from frameTokens vs final frame) and commitColor() gradient; invalidate on new run/resume/restore"
    status: completed
  - id: m1-js-wire-render
    content: "M1: Wire toggle handlers (mutually exclusive), extend renderFrameWithTokens + tooltip, generalize updateHoverHighlight"
    status: completed
  - id: m1-css
    content: "M1: Add .commit-btn styles + legend gradient mirroring .heatmap-btn in style.css"
    status: completed
  - id: m2-spike
    content: "M2: Validate decoder_input_ids seeding + reduced-step budget via scripts/spike_diffusiongemma.py (confirm encoder tolerates the kwarg on canvas 1)"
    status: completed
  - id: m2-sampler
    content: "M2: Add streaming_resume() to dgemma_sampler.py (renoise seed canvas, single generate call) + per-frame canvas-id capture in streaming_generate"
    status: completed
  - id: m2-worker
    content: "M2: Store last_run_state after generate and implement handle_resume in dgemma_worker.py; single-canvas validation"
    status: completed
  - id: m2-registry
    content: "M2: Flip DGEMMA supports_resume=True in registry.py"
    status: completed
  - id: m2-frontend
    content: "M2: Multi-canvas resume guard (hide edit UI) + renoise-semantics note in remask UI"
    status: completed
  - id: m3-diff-compute
    content: "M3: Add colorMode 'diff', computeDiff() (final branch vs original), diffColor()"
    status: completed
  - id: m3-diff-ui
    content: "M3: Add #btn-diff button gated on original+resume, extend render/tooltip, divergence summary stat"
    status: completed
  - id: m3-css
    content: "M3: Add .diff-btn styles + divergence palette in style.css"
    status: completed
isProject: false
---

# xAI thread: coloring, DiffusionGemma resume, and intervention diff

Three sequenced milestones. M1 is a pure-frontend quick win. M2 is the hard backend feature (DiffusionGemma interactive remask/resume, scoped to single-canvas runs). M3 is the explainability payoff that becomes possible once resume exists, and it also works for LLaDA today. M1 must land first because M3 reuses the `colorMode` enum it introduces.

---

## Milestone 1: Commit-order token coloring (quick win)

A new scrubber color mode that colors each resolved token by *when it settled into its final value*, exposing the resolution trajectory. Frontend-only; reuses the per-frame `frameTokens[]` already streamed and stored. No backend/protocol/save/analytics changes.

### Key decisions
- **Commit-step definition (model-agnostic).** For each position `i`, `commitStep[i] = 1 + max{ f : frameTokens[f][i].id != finalId[i] }` - the step after which the position stopped changing, using the last frame as truth. Never-resolved positions (masked at the last frame) get no color. This equals "reveal step" for LLaDA (resolved tokens are frozen) and is a reasonable "settle" proxy for DiffusionGemma's renoise-flips (LLaDA-first policy).
- **Mutual exclusivity via an enum.** Replace the `heatmapOn` boolean with `colorMode` (`"none" | "conf" | "commit"`; `"diff"` is added in M3) so color sources can't both be on.
- **Global annotation, frame-aware render.** `commitStep` is a whole-run property computed once; rendering still only colors tokens resolved *at the current scrubbed frame*.
- **No persistence** of the toggle, matching current Heatmap behavior.

### File changes
- [src/web/static/index.html](src/web/static/index.html): in `#heatmap-group` (lines 104-110), add `<button id="btn-commit-order" class="scrub-btn commit-btn" ...>Commit Order</button>` plus an early-to-late gradient legend (mirror `.heatmap-btn::after`).
- [src/web/static/app.js](src/web/static/app.js):
  - State (line 106): replace `var heatmapOn = false;` with `var colorMode = "none";`; add `btnCommitOrder` ref near line 100-101 and `var commitSteps = null;`.
  - Add `computeCommitSteps()` (bounded loop over frames x positions, memoized) and `commitColor(step, total)` (a hue sweep distinct from the green `heatColor`).
  - Toggle handlers (lines 2101-2113): rework `btnHeatmap` to `colorMode = colorMode === "conf" ? "none" : "conf"`; add a `btnCommitOrder` handler for `"commit"`. Update both buttons' `.active` classes, `highlightRow` visibility on `colorMode !== "none"`, call `updateHoverHighlight()`, re-render when `scrubberActive`.
  - `renderFrameWithTokens` (line 1217): branch on `colorMode` - `"conf"` uses `heatColor(tok.c)`, `"commit"` uses `commitColor`; append `"Resolved at step: N"` to the tooltip in commit mode.
  - `updateHoverHighlight` (lines 1169-1177): gate on `colorMode !== "none" && highlightTokens`.
  - Invalidate `commitSteps = null` where `frameTokens` is replaced (near `activateScrubber()` at line 1078, and in `restoreSessionState` after line 2459).
- [src/web/static/style.css](src/web/static/style.css): add `.commit-btn` mirroring `.heatmap-btn` (lines 1380-1430) with the commit palette. Token colors are applied inline via JS.

### Edge cases
Frame 0 (all masked): nothing colored. Never-resolved positions: excluded, render as mask. Single/two-frame runs: guard normalization divide-by-zero.

---

## Milestone 2: DiffusionGemma single-canvas remask/resume (Phase 2)

Bring the scrubber remask/resume and guided multi-frame editing to DiffusionGemma, **scoped to single-canvas (<=256-token) runs**. Multi-canvas resume is deferred.

### How it works
The installed sampler [generation_diffusion_gemma.py](.venv-dgemma/lib/python3.12/site-packages/transformers/models/diffusion_gemma/generation_diffusion_gemma.py) exposes the exact hook we need: `_prepare_denoiser_inputs` seeds the canvas from a `decoder_input_ids` kwarg when present, else random (lines 987-989):

```python
current_canvas = model_kwargs.pop(
    "decoder_input_ids", sampler.initialize_canvas(batch_size=batch_size, device=device)
)
```

So a resume is a fresh `model.generate` call with a **seed canvas** and a **reduced step budget**:
- `decoder_input_ids` = the chosen frame's draft canvas ids, with user-remasked positions renoised (set to fresh random ids).
- `max_new_tokens = config.canvas_length` (256) forces `max_new_canvases = 1` (line 650), keeping it single-canvas.
- `max_denoising_steps = max(1, original_steps - frame_index)` as the remaining budget. The `EntropyBoundSampler` + `StableAndConfidentStoppingCriteria` reset per canvas (line 1001), so adaptive stopping still applies.
- The existing `FrameQueueStreamer` is reused unchanged: `put(input_ids)` (prompt, skipped), `put_draft(...)` per step, `put(canvas)` on commit, `end()`.

```mermaid
sequenceDiagram
    participant UI as app.js
    participant Sup as server.py proxy
    participant W as dgemma_worker
    participant S as dgemma_sampler
    UI->>Sup: {type:resume, frame_index:K, remask_positions:[...]}
    Sup->>W: forward
    W->>W: last_run_state -> seed canvas ids at K (+ single-canvas check)
    W->>S: streaming_resume(seed, remask, remaining_budget)
    S->>S: renoise remask positions in seed canvas
    S->>S: model.generate(decoder_input_ids=seed, max_new_tokens=256, max_denoising_steps=rem, streamer)
    loop denoising steps
        S-->>W: frame (via FrameQueueStreamer)
        W-->>Sup-->>UI: frame
    end
    S-->>UI: done
```

### Key decisions and semantics
- **"Remask" = renoise, not freeze.** Unlike LLaDA (resolved tokens frozen), DiffusionGemma's entropy-bound sampler re-decides positions by entropy each step, so non-remasked committed tokens are *biased* by the seed but not locked - the resumed output can drift near the edit. This is a real behavioral difference and must be surfaced in the UI (a short note in the remask controls).
- **Worker-side state, mirroring LLaDA.** Keep the client protocol identical (`frame_index` + `remask_positions`). The worker records a per-frame canvas-id history during `streaming_generate` (the streamer already computes `ids` per `_emit`) and stores `last_run_state`.
- **Reduced-budget temperature restart.** `LinearTemperatureScheduleLogitsProcessor` uses `cur_step / max_denoising_steps`; resuming with a smaller budget restarts the schedule (t_max->t_min) over the remaining steps. Exact continuation is impossible under adaptive stopping; this is acceptable and noted.
- **Self-conditioning starts empty on resume** (`self_conditioning_logits=None`), a minor first-step fidelity loss.

### Files
- [src/backends/registry.py](src/backends/registry.py): flip `DGEMMA` `supports_resume=True` (unlocks the guided-edit UI, which is gated on it at `app.js:1327-1331`).
- [src/inference/dgemma_sampler.py](src/inference/dgemma_sampler.py): add `streaming_resume(model, tokenizer, *, seed_canvas_ids, remask_positions, prompt, remaining_steps, t_max, t_min, thinking, entropy_signal, seed, cancel_event)` that renoises remask positions, runs one `generate` with `decoder_input_ids` + budget, reuses `FrameQueueStreamer`; add optional per-frame id capture to `streaming_generate` (a list the worker can keep).
- [src/backends/dgemma_worker.py](src/backends/dgemma_worker.py): store `last_run_state` after `handle_generate` (prompt, canvas_length, original steps, t_max/t_min, thinking, entropy_signal, per-frame canvas ids + canvas_index); implement `handle_resume` with `_validate_resume` (frame in range, non-empty remask positions in `[0, canvas_length)`, run is single-canvas, not the final committed frame). Mirror [llada_worker.py](src/backends/llada_worker.py) `handle_resume`/`_store_state`.
- [src/web/static/app.js](src/web/static/app.js): guard - if the completed run has any frame with `frameCanvasIndex > 0`, keep `btn-edit-frames` hidden even when `supports_resume` is true (multi-canvas out of scope for v1); add the renoise-semantics note in the remask controls.

### Validation first (de-risk)
The outer loop calls `_prepare_encoder_inputs` / `encoder_forward(..., **model_kwargs)` (lines 733-740) *before* `decoder_input_ids` is popped in `_prepare_denoiser_inputs`, so on canvas 1 the encoder receives `decoder_input_ids` in its kwargs. The docstring says seeding is supported, but confirm empirically via [scripts/spike_diffusiongemma.py](scripts/spike_diffusiongemma.py): a standalone `generate(decoder_input_ids=seed, max_new_tokens=256, max_denoising_steps=k, streamer=...)` should run and stream sensibly. If the encoder rejects the kwarg, fall back to a thin patched entry point. Do this before wiring worker/UI.

### Risks
Encoder-kwarg tolerance (validated by the spike); renoise drift (product-acceptable, surfaced); temperature restart; ensure `max_frames` (guided "run to here") still early-stops via `FrameStreamer.run(max_frames=...)`.

---

## Milestone 3: Counterfactual / intervention diff

After a remask+resume, visualize how the branch diverged from the original run. Client-side; works for LLaDA immediately and for DiffusionGemma once M2 lands.

### How it works
Uses the retained original snapshot `originalFrameTokens` (set at [app.js](src/web/static/app.js) lines 1067-1071, preserved across resumes) vs the current `frameTokens`. Positions align 1:1 on the single fixed canvas (LLaDA `gen_length`; DiffusionGemma 256).
- New `colorMode = "diff"` (fourth state). Button "Diff vs Original", visible/enabled only when `originalTotalFrames > 0 && remaskEdits.length > 0`.
- Coloring (global annotation, rendered per current frame): unchanged-from-original = dim/neutral; **diverged** (final branch id != final original id at that position) = divergence color; user-remasked **origin** positions = the existing orange remask tint.
- Tooltip on diverged tokens: `was: <original> -> now: <current>`.
- Summary stat near the scrubber: `Diverged N/total (P%)`.
- v1 is binary changed/unchanged + origins + summary. A richer propagation gradient (color by first-divergence frame) is an optional later enhancement.

### Files
- [src/web/static/index.html](src/web/static/index.html): add `<button id="btn-diff" class="scrub-btn diff-btn" ...>Diff vs Original</button>` in `#heatmap-group`.
- [src/web/static/app.js](src/web/static/app.js): extend `colorMode` with `"diff"`; add `computeDiff()` (compare final frames of `originalFrameTokens` vs `frameTokens`, return per-position `{changed, origId, curId}`), `diffColor()`; gate button visibility; extend `renderFrameWithTokens` + tooltip; update a summary label.
- [src/web/static/style.css](src/web/static/style.css): add `.diff-btn` styles + divergence palette.

### Edge cases
No original / no resume yet: button hidden. Length mismatch (guard): diff only overlapping positions. Multiple sequential resumes: always diff against the pristine `originalFrameTokens`.

---

## Sequencing and out of scope

Order: M1 (independent) -> M2 (spike, then backend, then registry flip + frontend guard) -> M3 (reuses M1's `colorMode`; needs M2 for DiffusionGemma but is LLaDA-usable earlier).

Out of scope: multi-canvas DiffusionGemma resume (deferred); analytics/save schema changes; view-toggle persistence; multimodal image input (Phase 3).
