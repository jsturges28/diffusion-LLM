"""Tests that every Hub model names the commit it loads.

Strategy: read the registry as data and check the invariant that
separates the two kinds of checkpoint, then exercise the predicate
that decides which kind a string is, including the boundary where a
repo id becomes a path.

What passing proves: a model cannot be added to this app fetching
weights from a moving repository. The reason that matters is that
nothing else in a saved run would notice. The app commit, every
parameter and the displayed seed can all match while the weights
underneath have changed, so the run reads as reproducible and is not.

The invariant is two-sided on purpose. A local checkpoint asserting
``revision is None`` is not pedantry: inventing a commit for a
directory nobody can look up would be provenance that cannot be
checked, which is worse than none.
"""

from __future__ import annotations

import pytest

from src.backends.protocol import ModelInfo, is_hub_checkpoint
from src.backends.registry import DGEMMA, LLADA, REGISTRY, SMOLLM3

# A Hub commit is a full 40-character git sha. Short shas resolve
# today and stop resolving when the repository grows enough to make
# them ambiguous, so they are not an address.
SHA_LENGTH = 40


def test_every_hub_model_pins_a_revision() -> None:
    hub = [
        model
        for model in REGISTRY.values()
        if is_hub_checkpoint(model.checkpoint)
    ]

    assert hub, "expected at least one Hub-backed model"
    for model in hub:
        assert model.revision, f"{model.id} is unpinned"


def test_local_models_claim_no_revision() -> None:
    """The negative space: a directory has no commit to name."""
    local = [
        model
        for model in REGISTRY.values()
        if not is_hub_checkpoint(model.checkpoint)
    ]

    assert local, "expected at least one local checkpoint"
    for model in local:
        assert model.revision is None, (
            f"{model.id} names a commit nobody can resolve"
        )


@pytest.mark.parametrize(
    "model", [LLADA, SMOLLM3], ids=lambda m: str(m.id)
)
def test_pinned_revisions_are_full_shas(model: ModelInfo) -> None:
    revision = model.revision

    assert revision is not None
    assert len(revision) == SHA_LENGTH
    assert all(c in "0123456789abcdef" for c in revision)


def test_dgemma_is_the_local_one() -> None:
    """Pins which model the two-sided invariant applies to, so that
    flipping DiffusionGemma to a Hub id cannot silently skip the
    pinning requirement."""
    assert not is_hub_checkpoint(DGEMMA.checkpoint)
    assert DGEMMA.revision is None


# -- the predicate, including where the two kinds meet --


@pytest.mark.parametrize(
    "checkpoint",
    [
        "GSAI-ML/LLaDA-8B-Instruct",
        "HuggingFaceTB/SmolLM3-3B",
        "org/name",
    ],
)
def test_repo_ids_read_as_hub(checkpoint: str) -> None:
    assert is_hub_checkpoint(checkpoint) is True


@pytest.mark.parametrize(
    "checkpoint",
    [
        "~/models/diffusiongemma-nf4",
        "/opt/models/x",
        "./local",
        "../sibling",
        # No slash: a bare name is not a repo id.
        "model",
        # Two slashes: a path-like value that happens to omit its
        # leading marker, which must not read as org/name.
        "org/name/extra",
        "",
        "   ",
    ],
)
def test_paths_and_non_ids_do_not_read_as_hub(
    checkpoint: str,
) -> None:
    assert is_hub_checkpoint(checkpoint) is False
