"""Undoing a discarded edit, for both diffusion backends.

Strategy: the shared restore directly, with hand-built retained
state. No model, no socket. `handle_rewind`'s protocol half (the
token check, the refusal scope, the busy refusal) is exercised in
`test_worker_dispatch.py`, and LLaDA's whole path end to end in
`test_llada_resume_state.py`. What is left, and what this covers,
is the restore itself under each backend's key names.

It is shared code partly because the two backends differ only in
what they call their history, and partly out of necessity:
`dgemma_worker` cannot be imported here at all, because its
quantized loader reaches `bitsandbytes`, which lives only in
`.venv-dgemma`. Testing that backend's rewind means testing the
function it delegates to.

A resume replaces the retained history with the branch it made, and
every route out of an edit session rolled back only the browser. A
later edit at the same or a later frame then read a canvas from the
branch the user had discarded while they clicked tokens on the one
on screen. Passing proves the way back exists, that it can be taken
more than once, and that the baseline it restores to cannot itself
be edited by the resume it is meant to undo.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.backends.worker_base import rewind_retained_history

# The two backends this serves, named the way each stores its run.
LLADA_KEYS = ("frame_checkpoints", "generated_checkpoints")
DGEMMA_KEYS = ("frame_history", "generated_frame_history")

assert LLADA_KEYS != DGEMMA_KEYS, (
    "the point of the helper is that the names differ"
)


class _Frame:
    """A stand-in for one checkpoint, identified by value.

    A plain object rather than a real `FrameCheckpoint`, because a
    restore moves references and never looks inside one, and
    because a real one holds tensors whose `==` raises rather than
    answering.
    """

    def __init__(self, value: int) -> None:
        self.value = value


def _state(keys: tuple[str, str], frames: int) -> Dict[str, Any]:
    """A retained run, freshly generated and never edited."""
    working, baseline = keys
    generated = [_Frame(index) for index in range(frames)]
    return {working: list(generated), baseline: generated}


def _rewind(
    state: Dict[str, Any], keys: tuple[str, str]
) -> None:
    working, baseline = keys
    rewind_retained_history(
        state, working=working, baseline=baseline
    )


def _assert_same_objects(
    actual: List[Any], expected: List[Any]
) -> None:
    """Identity, element for element.

    The stronger claim: an equal frame rebuilt from the branch
    would still mean the worker had thrown the original away, and
    the frames a resume re-enters have to be the original objects
    because they carry the random state `XAI-01` retains.
    """
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got is want


# -- the restore --


def test_a_committed_branch_is_undone() -> None:
    for keys in (LLADA_KEYS, DGEMMA_KEYS):
        working, baseline = keys
        state = _state(keys, 4)
        generated = list(state[baseline])
        # What a committed resume leaves behind: the surviving
        # prefix followed by the branch it produced.
        state[working] = generated[:2] + [
            _Frame(98),
            _Frame(99),
        ]

        _rewind(state, keys)

        _assert_same_objects(state[working], generated)


def test_a_whole_chain_of_edits_is_undone() -> None:
    """A guided session commits one resume per Run to Here, and the
    browser rolls all of them back from a single snapshot taken
    when the session opened."""
    state = _state(LLADA_KEYS, 4)
    working, baseline = LLADA_KEYS
    generated = list(state[baseline])
    state[working] = generated[:1] + [_Frame(50)]
    state[working] = [state[working][0], _Frame(51)]

    _rewind(state, LLADA_KEYS)

    _assert_same_objects(state[working], generated)


def test_a_rewind_before_any_edit_changes_nothing() -> None:
    """Sent whenever a session opens, including the first, so the
    no-op is the common case rather than the odd one."""
    state = _state(LLADA_KEYS, 3)
    working, baseline = LLADA_KEYS
    generated = list(state[baseline])

    _rewind(state, LLADA_KEYS)

    _assert_same_objects(state[working], generated)


def test_it_can_be_taken_more_than_once() -> None:
    """A session opened, abandoned and opened again rewinds twice,
    so the restore must not consume its own baseline."""
    state = _state(LLADA_KEYS, 3)
    working, baseline = LLADA_KEYS
    generated = list(state[baseline])
    state[working] = [_Frame(77)]

    _rewind(state, LLADA_KEYS)
    _rewind(state, LLADA_KEYS)

    _assert_same_objects(state[working], generated)


# -- what must not be shared --


def test_the_restored_list_is_not_the_baseline() -> None:
    """Aliasing the two would let the next resume splice the thing
    a later rewind restores to, which is this bug again one level
    down. The negative space of the test above."""
    state = _state(LLADA_KEYS, 3)
    working, baseline = LLADA_KEYS

    _rewind(state, LLADA_KEYS)

    assert state[working] is not state[baseline]


def test_editing_the_working_list_leaves_the_baseline() -> None:
    state = _state(LLADA_KEYS, 3)
    working, baseline = LLADA_KEYS
    _rewind(state, LLADA_KEYS)
    generated = list(state[baseline])

    state[working].append(_Frame(42))
    state[working].pop(0)

    _assert_same_objects(state[baseline], generated)


# -- the edges --


def test_a_worker_with_no_run_does_not_raise() -> None:
    """A page can open an edit session against a worker that was
    restarted under it."""
    rewind_retained_history(
        None,
        working="frame_checkpoints",
        baseline="generated_checkpoints",
    )


def test_a_run_without_a_baseline_is_a_programmer_error() -> None:
    """Every generation records one. A run missing it means the
    store and the restore have drifted apart, which is worth
    crashing over rather than silently not rewinding."""
    state: Dict[str, Any] = {"frame_checkpoints": [_Frame(0)]}

    try:
        rewind_retained_history(
            state,
            working="frame_checkpoints",
            baseline="generated_checkpoints",
        )
    except AssertionError:
        return
    raise AssertionError("a missing baseline passed unnoticed")


def test_nothing_but_the_history_moves() -> None:
    """Only the named key is touched, so a rewind cannot quietly
    reset a run's parameters."""
    state = _state(LLADA_KEYS, 2)
    state["seed"] = 7
    state["gen_length"] = 128

    _rewind(state, LLADA_KEYS)

    assert state["seed"] == 7
    assert state["gen_length"] == 128
