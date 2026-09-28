"""The vendored Mamba-3 reference is intact, and its shim is honest.

Strategy: run upstream's two reference functions against each other
over random inputs, from an empty state and from a carried one, and
drive the local `repeat` shim against expectations built by hand.
`mamba3_siso_step_ref` loops over time and `mamba3_siso_fwd_ref` works
in the parallel, quadratic form, so an accidental edit to either, or a
shim that repeats heads in the wrong order, shows up as the two
disagreeing. Passing proves the copy in `reference/mamba3/` still
computes what upstream's does.

The digest test pairs the README's provenance with the files: editing
the reference without re-recording where it came from fails here.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference.mamba3 import siso_reference as ref  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "reference" / "mamba3"

# Small but not degenerate: four heads sharing one group, so the
# grouped-head expansion runs, and a rotary covering only part of each
# vector, which is the real checkpoint's shape.
BATCH, LENGTH, HEADS, GROUPS = 2, 12, 4, 1
QK_DIM, V_DIM, ANGLES = 16, 8, 4

Tensors = Dict[str, torch.Tensor]
State = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Both references compute in fp32 here, and the parallel form sums in
# a different order from the loop, so agreement is to a few ulps of
# the largest values rather than bit for bit.
TOLERANCE = 1e-4


def _draw(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=gen)


def _close(first: torch.Tensor, second: torch.Tensor) -> None:
    torch.testing.assert_close(
        first, second, rtol=TOLERANCE, atol=TOLERANCE
    )


def _inputs(seed: int) -> Tensors:
    """Random inputs in upstream's layout, with a decaying ADT."""
    gen = torch.Generator().manual_seed(seed)
    dt = F.softplus(_draw(gen, BATCH, HEADS, LENGTH))
    rate = F.softplus(_draw(gen, BATCH, HEADS, LENGTH)) + 1e-4
    return {
        "Q": _draw(gen, BATCH, LENGTH, GROUPS, QK_DIM),
        "K": _draw(gen, BATCH, LENGTH, GROUPS, QK_DIM),
        "V": _draw(gen, BATCH, LENGTH, HEADS, V_DIM),
        "ADT": -rate * dt,
        "DT": dt,
        "Trap": _draw(gen, BATCH, HEADS, LENGTH),
        "Q_bias": _draw(gen, HEADS, QK_DIM),
        "K_bias": _draw(gen, HEADS, QK_DIM),
        "Angles": _draw(gen, BATCH, LENGTH, HEADS, ANGLES),
        "D": _draw(gen, HEADS),
        "Z": _draw(gen, BATCH, LENGTH, HEADS, V_DIM),
    }


def _carried_state(seed: int) -> State:
    """A non-zero starting state, so the first step's trapezoid term,
    which reads the previous token's key and value, is exercised."""
    gen = torch.Generator().manual_seed(seed)
    turns = torch.rand(BATCH, HEADS, ANGLES, generator=gen)
    angle = turns * 2 * math.pi
    ssm = 0.1 * _draw(gen, BATCH, HEADS, V_DIM, QK_DIM)
    key = _draw(gen, BATCH, HEADS, QK_DIM)
    value = _draw(gen, BATCH, HEADS, V_DIM)
    return angle, ssm, key, value


def _run_both(inputs: Tensors, state: Optional[State]) -> Tuple:
    args = (
        inputs["Q"], inputs["K"], inputs["V"], inputs["ADT"],
        inputs["DT"], inputs["Trap"], inputs["Q_bias"],
        inputs["K_bias"], inputs["Angles"], inputs["D"], inputs["Z"],
    )
    step = ref.mamba3_siso_step_ref(*args, Input_States=state)
    fwd = ref.mamba3_siso_fwd_ref(*args, Initial_States=state)
    return step, fwd


def _assert_same_state(first: State, second: State) -> None:
    """Angles compared as points on the circle: the two forms reduce
    mod 2*pi at different moments, so a value near the wrap could read
    as 6.28 in one and 0.00 in the other while meaning the same."""
    _close(torch.cos(first[0]), torch.cos(second[0]))
    _close(torch.sin(first[0]), torch.sin(second[0]))
    for left, right in zip(first[1:], second[1:], strict=True):
        _close(left, right)


# -- the two forms agree, which is what makes either a reference --


def test_the_two_forms_agree_from_an_empty_state() -> None:
    step, fwd = _run_both(_inputs(1), None)

    _close(step[0], fwd[0])
    _assert_same_state(step[1], fwd[1])


def test_the_two_forms_agree_from_a_carried_state() -> None:
    step, fwd = _run_both(_inputs(2), _carried_state(3))

    _close(step[0], fwd[0])
    _assert_same_state(step[1], fwd[1])


def test_the_carried_state_changes_the_answer() -> None:
    """The test above means nothing if a carried state was ignored."""
    inputs = _inputs(2)
    (empty_out, _), _ = _run_both(inputs, None)
    (carried_out, _), _ = _run_both(inputs, _carried_state(3))

    assert not torch.allclose(empty_out, carried_out, atol=1e-3)


# -- the shim does exactly what einops would, and nothing else --


def test_the_shim_repeats_each_group_head_in_a_row() -> None:
    """`(h_bc g)` puts `g` innermost, so heads come out 0, 0, 1, 1."""
    grouped = torch.arange(12.0).reshape(1, 2, 2, 3)
    pattern = "b s h_bc d -> b s (h_bc g) d"

    repeated = ref.repeat(grouped, pattern, g=2)

    expected = torch.stack(
        [grouped[:, :, 0], grouped[:, :, 0],
         grouped[:, :, 1], grouped[:, :, 1]],
        dim=2,
    )
    assert torch.equal(repeated, expected)


def test_the_shim_adds_a_trailing_axis() -> None:
    values = torch.arange(6.0).reshape(2, 3)

    repeated = ref.repeat(values, "... d -> ... d e", e=4)

    assert repeated.shape == (2, 3, 4)
    for copy in range(4):
        assert torch.equal(repeated[..., copy], values)


def test_the_shim_refuses_any_other_pattern() -> None:
    with pytest.raises(NotImplementedError, match="not shimmed"):
        ref.repeat(torch.zeros(1, 1, 1, 1), "b s h d -> b s d h")


# -- the provenance still describes the files --


def _recorded_digest(label: str) -> str:
    readme = (REFERENCE_DIR / "README.md").read_text(encoding="utf-8")
    row = rf"^\| {re.escape(label)} \| `([0-9a-f]{{64}})` \|$"
    found = re.search(row, readme, re.M)
    assert found, f"README.md records no {label!r} row"
    return found.group(1)


@pytest.mark.parametrize(
    "label,name",
    [
        ("This file, SHA-256", "siso_reference.py"),
        ("LICENSE, SHA-256", "LICENSE"),
    ],
)
def test_the_files_match_their_recorded_digests(
    label: str, name: str
) -> None:
    actual = hashlib.sha256(
        (REFERENCE_DIR / name).read_bytes()
    ).hexdigest()

    assert actual == _recorded_digest(label), (
        f"{name} no longer matches its recorded digest. Re-record"
        " the provenance in reference/mamba3/README.md, or undo the"
        " edit."
    )
