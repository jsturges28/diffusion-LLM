# LLaDA reference sampler

`llada_reference.py` is LLaDA's own sampling loop, kept so a test can
prove this project's implementation still agrees with it. **Nothing
here runs at runtime.** The live algorithm is
[`src/inference/llada_kernel.py`](../../src/inference/llada_kernel.py),
and the only importer of this file is
[`tests/inference/test_llada_kernel_differential.py`](../../tests/inference/test_llada_kernel_differential.py),
which drives both over identical logits and random state and requires
the same canvas out of each.

Do not edit it, and do not reformat it. Its style, naming and line
lengths are upstream's, and being diffable against upstream is the
entire reason it exists, so `reference` is excluded from the linter in
`pyproject.toml` and sits outside the `src` and `tests` trees that
`scripts/lint_ratchet.py` counts.

## Why it is here rather than deleted

Audit finding `ORG-03`. Production streaming imported two helpers from
this file and then re-implemented the same CFG, Gumbel, remasking,
transfer and block-schedule logic as the `generate` below, so a LLaDA
change had two plausible places to land and the diffusion entropy and
top-k work would have had to pick one. The duplicate loop is no longer
reachable, but throwing it away would have discarded the only
independent statement of the algorithm we have. Keeping it costs one
file nobody imports and buys a test that fails if the kernel drifts.

It keeps its **own** copies of `add_gumbel_noise` and
`get_num_transfer_tokens` on purpose. A reference that imported the
kernel's versions would make the differential test compare the kernel
to itself.

## Provenance

| Field | Value |
|---|---|
| Upstream project | LLaDA, `GSAI-ML/LLaDA` (the `generate.py` reference sampler) |
| Upstream revision | **Unrecorded.** See the note below. |
| Entered this repo | commit `a58a2ba`, "Minimal working demo" |
| Moved here | `ORG-03`, from the `llada_sampler` module under `src/inference/` |
| Bytes | 6661 |
| SHA-256 | `121efecd3ffdb396ac4c7911d8666cd48dc5127cdb955436f3d4b5bc529218f8` |

The exact upstream commit was never written down: this file arrived in
this repository's first commit with no source note, so the revision
above is honestly unknown rather than reconstructed. The bytes and
digest describe the file as trimmed and committed here, not upstream's
original, because two functions were removed (below). Anyone
re-syncing against upstream should record the revision they used and
replace this row.

## What was changed on the way in

Nothing inside the algorithm. Two things were removed, and a module
docstring was added:

- `main()`, upstream's demo program. It hardcoded three arithmetic
  prompts, loaded the 8B checkpoint onto `cuda` at import-time module
  scope, and printed to stdout. Removing it is what lets a test import
  this module without transformers or a GPU.
- `llada_generate_with_history()`, which was **not** upstream. It was
  this project's own early wrapper, had no callers anywhere, and
  duplicated prompt formatting that
  `src/backends/text_adapter.py` now owns.

## Two known defects, kept deliberately

Both stay unfixed, because this file's value is fidelity to upstream
rather than correctness. Each is noted so nobody reads a divergence
from the kernel as the kernel being wrong.

**The EOS masking writes to the wrong tensor.**

```python
logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
```

That chained assignment writes `-inf` into two *different* tensors. It
almost certainly meant to mask both token ids on one of them. It is
reachable only through `confidence_eos_eot_inf`, which defaults to
`False`, and has never executed in this project: the flag appears in no
`ParamSpec` and nowhere in the frontend. The kernel does not reproduce
it, having no EOS flags at all, so the differential test exercises only
the paths both implementations share.

**Guidance without an attention mask raises `UnboundLocalError`.**
`attention_mask_` is built inside `if attention_mask is not None` and
then read unconditionally, so `cfg_scale > 0` with no mask crashes.
Found by the differential test, which originally passed `None`.
Unreachable in production, where `build_llada_inputs` always returns a
mask and `streaming_generate` extends it over the canvas before the
first step, so the test now supplies one on both sides.
`forward_with_cfg` in the kernel handles the `None` branch correctly.

## Licence

Upstream LLaDA's licence was not vendored when this file was copied in,
and it is not asserted here rather than guessed at. If this reference
is ever re-synced, or if the project is distributed, fetch the licence
from the upstream repository and commit it beside this file, the way
`src/web/static/vendor/` does for the browser assets.
