---
name: ranked candidates and exact probe
overview: Close the item-94 discrepancy (bf16 resolution, not a bug) by preferring recorded probabilities over recomputed ones, give every candidate row a rank, surface the token the run actually chose when it falls outside the captured five, mark edited positions persistently, dim the entropy bars past the scrubber, and retain the run's KV cache so a probe becomes bit-exact and a substitution stops re-prefilling its prefix.
todos:
  - id: rank-and-record
    content: Add model_vocab_size to describe_tokenizer and its existing plumbing, pass a rank from every candidate row into the strip readout (alt.rank || index + 1), and make requestTypedProbe prefer a recorded candidate over sending a probe
    status: completed
  - id: chosen-row
    content: Append the chosen token as a sixth candidate when it falls outside the captured five, in both _sample_next and _forced_trace, with an unrounded probability and an explicit rank; add rank to TokenAlternative with exclude_none persistence, and gate the chosen row against being a substitution target
    status: completed
  - id: edit-tint
    content: Add a persistent .token-edited class on both pages fed by editedProfilePositions / editedPositions, distinct from and softer than .token-remasked, composing with the Heatmap and Entropy overlays
    status: completed
  - id: scrub-dim
    content: "Dim entropy bars past the scrubber: a third emphasis tier with its own fill boundary in drawEntropyProfileSeries, and a baked per-bar alpha in entropyDataset rebuilt from setOverlayFrame with update(\"none\"), plus an entropyDimColor sibling returning hsla"
    status: completed
  - id: kv-cache
    content: Retain the run's KV cache through state_sink onto last_run_state, reuse it in _position_distribution via non-destructive sliced views (prefill for position 0, one-token decode for n >= 1), with a prefix-mismatch fallback, invalidation alongside last_run_state, and a memory guard
    status: completed
  - id: verify-docs
    content: Run pytest, node --check, ReadLints, the column audit and ruff at 156; add the new tests; update README, ROADMAP, HANDOFF, About and Help; write the manual checklist from item 102 including the cache bit-equality test
    status: completed
isProject: false
---

# Ranked candidates, the chosen row, and an exact probe

Five commits. The first four are independent of the fifth, which is deliberately last and droppable because none of it can be exercised in the sandbox.

## What item 94 actually was

Both numbers are correct to the limit of the model's own output format. Generation samples position *n* from a single-token decode step against an incrementally built cache; the probe reconstructed it with a fresh prefill. In bf16 those land one ulp apart: moving `·The` from 38.3% to 39.8% needs a logit shift of ~0.063, and bf16 spacing near 16.0 is exactly 0.0625.

The maintainer's diagnostic confirms it. Index 0 (labelled "Position 1") takes the prefill path in *both* generation and the probe, and agrees to the last digit. Index 1 (labelled "Position 2") is the first decode step, and differs. Nothing else in the pipeline distinguishes those two positions.

## Commit 1: a rank on every row, and stop recomputing what we recorded

**Ranks are free for the captured five.** `_top_alternatives` uses `torch.topk`, which returns descending order, so row *i* is rank *i+1* by construction:

```src/inference/ar_sampler.py
    values, indices = torch.topk(probs, count)
```

- [src/web/static/overlays.js](src/web/static/overlays.js): `overlaysBuildAltRow(alt, chosenId, onHover)` gains the row index and hands `alt.rank || index + 1` to `onHover`. The `alt.rank` preference matters from Commit 2 on, where the appended sixth entry carries an explicit rank that its index would get wrong.
- [src/web/static/app.js](src/web/static/app.js) and [src/web/static/analytics.js](src/web/static/analytics.js): `setCandidateMetricsHover` fills `rank` and `vocabSize` for every row, not just the typed one.

**The rank denominator has to arrive without a probe.** The client knows the tokenizer's `vocab_size` (128,000) but the rank is over the model's output width (128,256). `describe_tokenizer` in [src/backends/worker_base.py](src/backends/worker_base.py) gains an optional `model` argument and reports `model_vocab_size` from `model.config.vocab_size`. It rides the plumbing already built for tokenizer identity with no new hop: `/health` (worker_base.py:417) to `ModelManager.active_tokenizer`, out on `/api/models`, and into `reproducibility.tokenizer` at save time so Analytics has it per run. The two figures sitting next to each other is the point: a padded embedding is why they differ, and the Analytics detail can show both.

**`requestTypedProbe` skips the send when the token is already recorded.** If the confirmed id appears in `positionAlts[typedEntryPos]`, fill `typedEntryMeasure` from that entry (probability from `p`, rank from the index) and send nothing. This removes the discrepancy with certainty today, independent of Commit 5, and saves a forward pass on the most common typed input. At the three significant figures the strip shows, the stored value and an exact probe are the same string, so this costs no precision.

## Commit 2: the row for the token the run actually chose

Detection already exists and is silently failing: `renderAltsPopover` computes `chosen` from the frame token's id and marks the matching row, and when a high temperature samples outside the five (or a typed token is forced), nothing matches and nothing is marked.

- [src/inference/ar_sampler.py](src/inference/ar_sampler.py): a new `_ensure_chosen_candidate(base_probs, candidates, chosen_id, tokenizer)` appends a sixth entry when `chosen_id` is absent, carrying `rank` from `(base_probs > p).sum() + 1`, a comparison on a tensor `_sample_next` already holds.
- `_forced_trace` does the same at the forced position, where `trace.alts.append(forced_alts)` currently stores only the original five. `_probe_forced_position` starts returning the rank as well, which `_position_distribution` makes free since it already hands back the whole tensor.
- The sixth entry's probability is stored **unrounded**. `_top_alternatives` rounds to 4 places, which floors at 0.0001, and this entry is routinely below that: the typed `ec` in screenshot 5 already reads `Confidence: 0.000` for that reason.
- [src/web/server.py](src/web/server.py): `TokenAlternative` gains `rank: Optional[int] = None`, and `_dump_alternatives` switches to `model_dump(exclude_none=True)` (matching `_dump_frame_tokens`) so the five stay compact.

**Where it is stored matters.** This goes in `alternatives.json`, which is per position and 26 KB for a 128-token run. It must not go in the token records: frames are full snapshots, so `tokens.json` is already 542 KB for the same run and a per-token field there is quadratic.

**Frontend needs nothing structural.** Nothing hard-codes five; every loop is `alts.length`-driven, and `TOP_K_ALTERNATIVES` appears only in `ar_sampler.py`. The existing `alt-row-chosen` highlight starts firing on its own. One gate to add: the chosen-outside-five row must not be a substitution target, since forcing the token already there is a no-op.

## Commit 3: a persistent orange mark on edited positions

The data is already derived. `editedProfilePositions()` reduces `remaskEdits` to every position an edit touched and drives the entropy profile's dashed marker; the token layer reads the same set.

- [src/web/static/app.js](src/web/static/app.js): `tokenClassFn` (app.js:3920) gains a `.token-edited` branch. It currently only knows `remaskedPositions`, which is the transient in-edit selection.
- The distinction to preserve: `.token-remasked` means "selected, about to be redrawn"; this means "the run was intervened here", which stays true forever. So a softer wash than `.token-remasked`'s `rgba(255, 159, 28, 0.15)` background plus recolored text, closer to the faint tint under the profile's dashed line, and no text recolor so the token stays legible under the Heatmap and Entropy overlays. There is an existing note at style.css:2720 about `.token-remasked` fighting the Heatmap's warm end; the new class has to compose rather than repeat that.
- Analytics gets the same class from `editedPositions(data)`.
- The hover glow stays as it is. It means "the pointer is here" on every other token, and a second meaning on these ones costs more than it buys.

## Commit 4: dim the entropy bars past the scrubber

Autoregressive-only on both pages, so there is no diffusion frame-to-position mapping to solve. Both pages index frames 0-based internally (the generator's label is 1-based, Analytics' is not) and frame *k* introduces position *k*, so the rule is `position > frameIndex` in both.

- [src/web/static/app.js](src/web/static/app.js): `drawEntropyProfileSeries` has exactly one line of emphasis logic (app.js:3243). It gains a third tier. Note the original layer is drawn with `current: -1` so it takes no bright column, which means the fill boundary needs its own field rather than reusing `current`; both layers share the same boundary because positions align.
- [src/web/static/analytics.js](src/web/static/analytics.js): `entropyDataset` bakes the dim into the per-bar `backgroundColor` array it already builds, and `setOverlayFrame` (analytics.js:2137) rebuilds it and calls `chartEntropy.update("none")` so the slider does not animate on every tick.
- `entropyColor` returns `hsl(...)` while `withAlpha` expects hex, so a sibling `entropyDimColor(e)` returning `hsla(...)` belongs next to `entropyColor` and `entropyGlowColor` in overlays.js.
- This composes with `compareBlendPlugin`, which sets a whole-dataset `globalAlpha`; a per-bar alpha in the fill multiplies with it, which is the wanted behavior.

## Commit 5: retain the run's KV cache

Upgrades the probe from correct-to-an-ulp to bit-identical, and stops a substitution re-prefilling its entire prefix, which is its dominant cost.

**Confirmed available.** `.venv-ar` has transformers 4.53 with `DynamicCache.crop`, which slices `[..., :max_length, :]`, a view. `crop` is in-place and destructive, so build a fresh cache from sliced views via `DynamicCache.from_legacy_cache` instead: pointer work proportional to the layer count, no tensor copy, and the retained cache survives intact for the next probe at a different position.

**The call-shape rule.** Reproduce the shape the run used: a prompt prefill for position 0, and a one-token decode against a cache sliced to `prompt + n - 1` for *n* >= 1. This is precisely what the diagnostic showed, and getting it wrong (slicing to `n` and forwarding nothing, or decoding at position 0) reintroduces the ulp.

**Plumbing.** `_stream_tokens` already holds `past` locally; route it out through `result` into `state_sink` alongside ids and confidences in `_drain_frames`, and hold it on `last_run_state`. Cleared wherever that is (smollm3_worker.py:235 on a new run, plus model switch and deactivation), and pinned to the *recorded* run rather than a branch, for the same reason `state_sink=None` keeps a branch's trace out.

**Not a promise: a fallback.** Compare the retained prefix ids against the requested prefix in `_position_distribution`, the one choke point both callers pass through, and fall back to a fresh prefill on any mismatch rather than returning a wrong number quietly.

**Memory.** 72 KiB per token for SmolLM3 (36 layers, 4 KV heads under GQA, head_dim 128): about 21 MiB for a 256-token run and 148 MiB at the 2048-token experimental ceiling, against roughly 6 GiB of weights. Bounded, so a simple guard is enough.

**Acceptance test, on hardware.** Probe a position for a token that *was* captured and assert exact equality with the recorded value, then probe twice and assert bit-equality. That settles the two empirical unknowns: whether decode is bit-reproducible for fixed shapes, and whether the sliced-view cache survives the attention path. Neither can be answered in the sandbox, where every test is necessarily a stub against `StubModel`.

## Verification and docs

- `.venv/bin/python -m pytest`, `node --check` on each changed JS file, ReadLints, the 70-column audit on added lines, and `ruff check src/ tests/` held at its current 156 findings.
- New pytest coverage: the sixth entry appearing only when the pick is outside the five and carrying the right rank, the forced path appending with its measured rank, `exclude_none` keeping old-shaped entries compact, and the cache fallback on a prefix mismatch.
- README, ROADMAP, HANDOFF, and the About and Help modals, including the tokenizer-vocab against model-output-width distinction now that both are shown. Manual checklist continues from item 102, covering the rank on every row, the sixth row under high temperature and after a typed edit, the persistent edit tint under each overlay, the scrub dimming on both pages, and the cache's bit-equality test.