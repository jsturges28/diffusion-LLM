# Mamba-3 SISO reference

`siso_reference.py` is upstream's own PyTorch statement of the Mamba-3
SISO recurrence, kept so tests can prove this project's implementation
agrees with it. **Nothing here runs at runtime.** The live model is
[`src/inference/mamba3.py`](../../src/inference/mamba3.py), and
[`tests/inference/test_mamba3.py`](../../tests/inference/test_mamba3.py)
drives its recurrence and upstream's two forms over identical inputs,
requiring the same outputs and states from all three.
[`tests/inference/test_mamba3_reference.py`](../../tests/inference/test_mamba3_reference.py)
holds the copy itself honest: the two forms must agree with each
other, the shim below must do exactly what `einops` would, and the
file must still match the digest recorded under Provenance.

Do not edit the functions, and do not reformat them. Their style,
naming, trailing whitespace and line lengths are upstream's, and being
diffable against upstream is the reason the file exists, so
`reference` is excluded from the linter in `pyproject.toml` and sits
outside the `src` and `tests` trees that `scripts/lint_ratchet.py`
counts.

## Why it is here

The model is our own implementation rather than upstream's
`mamba-ssm` package. Upstream decodes every token through a CuTe
kernel whose docstring says it is only tested on H100, and this
project's card is an RTX 4090 (compute capability 8.9); installing it
would also need a newer torch, Triton 3.5, TileLang and QuACK built
from git source, which the environment locks cannot express.
`transformers` has no Mamba-3 support at all.

What upstream does ship is this: the two functions its Triton kernels
are held to. `mamba3_siso_step_ref` runs the recurrence one step at a
time, and `mamba3_siso_fwd_ref` computes the same thing in its
parallel, quadratic form, which is an independent derivation rather
than a second copy of the loop. This project's recurrence is written
separately and is required to agree with both.

## Provenance

| Field | Value |
|---|---|
| Upstream project | `state-spaces/mamba` (Dao AI Lab, Goombalab) |
| Upstream file | [tests/ops/triton/test_mamba3_siso.py](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/tests/ops/triton/test_mamba3_siso.py), upstream's tree, not this one |
| Upstream revision | `e9594ce1c732d97440f0332fdc43170a2294dbfa` (2026-07-22) |
| Upstream file bytes | 42,361 |
| Upstream file SHA-256 | `b68ec350f557a4124516f7d1c916ec756a48389fed4387791aadb74492531e85` |
| Upstream git blob | `35908b261f0480a1c82d58e395f16f2545e8441f` |
| Lines vendored | 21 to 340 (`_segsum`, `mamba3_siso_step_ref`, `mamba3_siso_fwd_ref`) |
| This file, bytes | 14,399 |
| This file, SHA-256 | `42253fd98c4483f90ce7ca211a9c955c846d6d4a680c8d4048e5e1aba81c9fc6` |
| LICENSE, SHA-256 | `760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5` |

The vendored lines were compared byte for byte against upstream's
lines 21 to 340 when they were copied in. Anyone re-syncing should
record the new revision and replace these rows.

## What was changed on the way in

Nothing inside the three functions. Only the module header differs:

- Upstream's test imports (`copy`, `pytest`, `triton` and the two
  `mamba_ssm` kernels) are dropped, since none of the three functions
  uses them.
- `from einops import rearrange, repeat` is replaced by a local
  `repeat` that implements exactly the two patterns the functions use,
  `"b s h_bc d -> b s (h_bc g) d"` (grouped heads expanded, each
  repeated `g` times in a row) and `"... d -> ... d e"` (a new trailing
  axis). Any other pattern raises. `einops` is installed in neither
  `.venv`, where the tests run, nor `.venv-ar`, where the probe runs,
  and adding a dependency to vendor a test oracle was the worse trade.
  `rearrange` was imported upstream but none of these functions uses
  it.

The reference test checks the shim against hand-built expectations and
the two references against each other, so a fault in the shim cannot
hide behind agreement between two copies of it.

## Licence

Apache-2.0. Upstream's `LICENSE` is vendored unchanged beside this
file; its digest is in the table above.
