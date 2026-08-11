---
name: Durable Overlays Analytics
overview: Persist per-token {t,m,id,c} frame streams (edited run plus the original snapshot for edited runs) so commit-order and Diff-vs-Original survive reload and become reviewable in the Analytics Suite, with a static final-frame render first and a data model that is future-proof for a scrubber and a durable Heatmap.
todos:
  - id: persist-records
    content: "Enrich save persistence: TokenRecord model, frame_tokens + original_frame_tokens in SaveRunRequest/_save_run_blocking (tokens.json + original_tokens.json), and send full records (incl. confidence) from app.js saveRun."
    status: completed
  - id: frames-endpoint
    content: Add metrics.load_run_frames (legacy-tolerant) and GET /api/analytics/runs/{id}/frames returning edited+original token streams, remask_edits, canvas_index, records_available.
    status: completed
  - id: shared-overlays
    content: Create static/overlays.js with pure commitColor/heatColor/diffColor/computeCommitSteps/computeDiff; delegate app.js wrappers to them; add script includes to index.html and analytics.html.
    status: completed
  - id: analytics-viewer
    content: Add a static final-frame overlay viewer to the analytics detail panel (commit-order always, Diff for edited runs with Diverged N/total), gated on data availability; port token-span styles into analytics.css.
    status: completed
  - id: docs
    content: Update README.md and ROADMAP.md to reflect durable overlays, persisted per-token confidence, and remaining follow-ups (analytics scrubber, durable Heatmap render).
    status: completed
  - id: verify
    content: Add pytest for load_run_frames (valid/edited/legacy/missing) and manually verify durability plus unchanged live-view overlay behavior.
    status: completed
isProject: false
---

# Durable Overlays in Analytics

Make the two shipped xAI overlays (commit-order and Diff vs Original) durable and reviewable post-hoc in the Analytics Suite, and persist per-token confidence now so a durable Heatmap is free later. First render is static (final frame only); the persisted data is future-proof so a scrubber can be added later with no re-save.

## Scope decisions (settled)
- Persist full per-token records `{t, m, id, c?}` per frame for the edited run, plus the same for the original snapshot on edited runs. `c` is omitted for masked tokens (mirrors the live protocol shape).
- Enrich `tokens.json` in place (shape `List[List[int]]` -> `List[List[{t,m,id,c?}]]`); add `original_tokens.json` for the original snapshot (written only when `remask_edits` exist). `tokens.json` currently has no reader, so the shape change is safe.
- Analytics first renders a static final-frame overlay; the loaded frame stream carries all frames so a scrubber is a later, no-migration add.
- Overlay math is extracted into a shared `static/overlays.js` (classic global script, same pattern as `custom_select.js`), consumed by both pages.

## 1. Persistence (data model)
- [src/web/server.py](src/web/server.py): add a `TokenRecord` pydantic model (`t: str`, `m: bool`, `id: int`, `c: Optional[float] = None`). In `SaveRunRequest`, replace `frame_token_ids` with `frame_tokens: Optional[List[Optional[List[TokenRecord]]]]` and add `original_frame_tokens: Optional[List[Optional[List[TokenRecord]]]]`. In `_save_run_blocking`, write the richer records to `tokens.json`, and write `original_tokens.json` when `original_frame_tokens` is present.
- [src/web/static/app.js](src/web/static/app.js) `saveRun` (~2497-2526): send full token records from `frameTokens` (the objects already carry `t,m,id,c`) instead of the ids-only projection; include `original_frame_tokens` from `originalFrameTokens` when `remaskEdits.length > 0`.

## 2. Server delivery endpoint
- [src/analytics/metrics.py](src/analytics/metrics.py): add `load_run_frames(run_dir) -> dict` that reads `tokens.json` and optional `original_tokens.json`, tolerating legacy int-only files (returns a `records_available: bool` flag so overlays can gate off cleanly for old runs).
- [src/web/server.py](src/web/server.py): add `GET /api/analytics/runs/{id}/frames` returning `{run_id, frames, original_frames, remask_edits, canvas_index, records_available}`. Kept separate from the lean metrics endpoint since token streams are large; loaded lazily when the viewer opens. Reuse the existing path-traversal guard pattern from `_delete_run_blocking`.

## 3. Shared overlay module
- New `src/web/static/overlays.js` with pure, parameterized primitives extracted from `app.js`: `commitColor(step, maxStep)`, `heatColor(c)`, `diffColor(changed)`, `overlaysComputeCommitSteps(frames)`, `overlaysComputeDiff(curFinal, origFinal, remaskEdits)`.
- [src/web/static/app.js](src/web/static/app.js): make `commitColor`/`heatColor`/`diffColor`/`computeCommitSteps`/`computeDiff` delegate to the shared functions, keeping the existing global-reading wrappers and memoization (`commitSteps`, `diffData`) so the live view's behavior and call sites are unchanged (regression guard).
- Add `<script src="/overlays.js">` before `app.js` in [index.html](src/web/static/index.html) and before `analytics.js` in [analytics.html](src/web/static/analytics.html).

## 4. Analytics UI (static render)
- [src/web/static/analytics.html](src/web/static/analytics.html): add an overlay-viewer section in the detail panel (after `#detail-meta`, before the charts) with a small overlay picker and an `#overlay-output` container.
- [src/web/static/analytics.js](src/web/static/analytics.js): in `showDetail`/`loadRunCharts`, fetch `/frames`, render the final frame as token spans colored by commit-order (via `overlaysComputeCommitSteps` + `commitColor`); for edited runs add a Diff option (`overlaysComputeDiff` of edited-final vs original-final) showing origin/changed coloring and a `Diverged N/total` readout. Gate each overlay on data availability (`records_available`; Diff also needs `original_frames` + `remask_edits`); show a clear "not available for this run" note for legacy runs.
- [src/web/static/analytics.css](src/web/static/analytics.css): port `token-span` / `token-resolved` / `token-mask` / `--mask-color` styling from [style.css](src/web/static/style.css) and lay out the overlay viewer.

## 5. Docs
- [README.md](README.md): update the "Commit order and counterfactual diff" and overlay sections (drop "client-side/ephemeral (lost on reload)"), note the new Analytics overlay viewer, and add an Implementation Status line.
- [ROADMAP.md](ROADMAP.md): move durable commit-order/diff (and persisted confidence enabling a future Heatmap) into shipped; note the deferred analytics scrubber + durable Heatmap render as the remaining follow-ups.

## 6. Backward compatibility and guardrails
- Legacy runs (ids-only `tokens.json`, no original snapshot) degrade gracefully: overlays gate off with a message; charts and the run browser are unaffected.
- Response size is bounded by canvas x frames; `load_run_frames` validates structure and rejects malformed files with a clear error.

## 7. Verification
- pytest (under `.venv/bin/python`) for `load_run_frames`: valid enriched run, edited run with original, legacy int-only file, and missing file.
- Manual: LLaDA run -> Save -> Analytics shows commit-order on the final frame; edit+resume -> Save -> Diff shows divergence readout and origin/changed coloring; reload the page to confirm durability; confirm the live generator overlays still behave identically.