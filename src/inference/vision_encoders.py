"""The vision encoders the tokeniser view can compare.

Deliberately not the model registry. A registry entry advertises
something the Main Menu will offer to load and generate with, and
neither of these can do that yet: this slice inspects how an image
would be tokenised and stops there. When generation lands they become
registry entries; until then a separate, smaller declaration keeps the
picker honest.

**Only the configuration is ever read, never the weights.** Two small
JSON files answer everything the page draws, which is what lets this
run in the supervisor with no worker, no residency claim and no
eviction, so a reader can compare both encoders while a model stays
loaded and mid-run.

**The geometry is read from the checkpoint, not declared here.** Only
the repository, its pinned commit and a name for it are declared. The
tile size, patch size and scale factor come out of `config.json`,
because a hand-maintained copy drifts from what the checkpoint holds,
and a readout whose whole purpose is to be believed cannot afford
that. The same argument the tokenizer identity was built on.

The two differ in every dimension, which is the reason both are here:
the 500M fuses 16 patches into a token where the 2.2B fuses 9, and
their tiles, patch sizes and working resolutions differ too. Seeing
the same picture cost different amounts is the lesson.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.inference.vision_geometry import EncoderGeometry

logger = logging.getLogger(__name__)

# The files the geometry is read out of. `config.json` carries the
# vision tower's dimensions and the scale factor;
# `preprocessor_config.json` carries the two bounds, the whole-image
# `size` and the per-tile `max_image_size`.
CONFIG_NAME = "config.json"
PREPROCESSOR_NAME = "preprocessor_config.json"
REQUIRED_FILES = (CONFIG_NAME, PREPROCESSOR_NAME)


@dataclass(frozen=True)
class VisionEncoder:
    """One encoder the view can inspect, by name and by commit."""

    # Stable identifier, used in the API and by the page.
    id: str
    display_name: str
    repo_id: str
    # The commit this is read at. Required, for the reason the model
    # registry gives: a repository must not be able to move underneath
    # a figure someone has written down.
    revision: str
    # One line the page shows beside the name, so a reader choosing
    # between them knows what they are choosing between.
    summary: str


SMOLVLM_500M = VisionEncoder(
    id="smolvlm-500m",
    display_name="SmolVLM-500M-Instruct",
    repo_id="HuggingFaceTB/SmolVLM-500M-Instruct",
    revision="a7da5b986cb59b408707209984f360a5f4ad7e47",
    summary=(
        "SigLIP-B/16 over 512px tiles, fusing 16 patches into each"
        " token. The cheaper of the two, and the one a laptop runs."
    ),
)

SMOLVLM_2B = VisionEncoder(
    id="smolvlm-2b",
    display_name="SmolVLM-Instruct",
    repo_id="HuggingFaceTB/SmolVLM-Instruct",
    revision="81cd9a775a4d644f2faf4e7becff4559b46b14c7",
    summary=(
        "SigLIP-SO400M over 384px tiles, fusing 9 patches into each"
        " token, so it keeps more detail and spends more context."
    ),
)

ENCODERS: Dict[str, VisionEncoder] = {
    SMOLVLM_500M.id: SMOLVLM_500M,
    SMOLVLM_2B.id: SMOLVLM_2B,
}

# Mirrors the guard in `src/backends/registry.py`, and for the same
# reason: an unpinned repository can move under a number somebody has
# written down. Asserted at import rather than in a test alone, so a
# third encoder cannot be added unpinned.
for _encoder in ENCODERS.values():
    assert _encoder.revision, (
        f"{_encoder.id} loads from the Hub and must pin a revision"
    )
    assert len(_encoder.revision) == 40, (
        f"{_encoder.id} pins {_encoder.revision!r}, which is not a"
        " full commit hash; a tag or a branch can still move"
    )
    assert _encoder.id == _encoder.id.lower(), (
        f"{_encoder.id} is used in a URL and should be lowercase"
    )

assert len(ENCODERS) >= 2, (
    "the view compares encoders, so one on its own has nothing to say"
)


class EncoderUnavailable(RuntimeError):
    """The configuration is not cached and could not be fetched.

    An operating error rather than a programmer error: a first run on
    a machine with no network reaches this, and the page has to say so
    rather than crash. Distinct from an unknown id, which is a bad
    request.
    """


def _read_json(
    encoder: VisionEncoder, filename: str, *, allow_fetch: bool
) -> Dict[str, object]:
    """One config file, from the cache or from the Hub.

    Imported here rather than at module scope so this module stays
    importable, and testable, without `huggingface_hub` resolving
    anything. The same reason `worker_base` imports torch inside the
    function that needs it.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            encoder.repo_id,
            filename,
            revision=encoder.revision,
            local_files_only=not allow_fetch,
        )
    except Exception as error:
        # Every failure here is the same answer to the caller: the
        # file is not available. Narrowing it would mean naming a
        # dozen hub exceptions, and the page's recourse is identical
        # for all of them.
        raise EncoderUnavailable(
            f"{encoder.display_name} has no cached {filename}"
            f" at {encoder.revision[:8]}"
            + ("" if allow_fetch else " and no fetch was allowed")
        ) from error

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise EncoderUnavailable(
            f"{encoder.display_name}'s {filename} is not an object"
        )
    return loaded


def is_cached(encoder: VisionEncoder) -> bool:
    """Whether both config files are already on disk.

    Asked so the page can say "not downloaded yet" before a reader
    picks an encoder, rather than after a failed request.
    """
    for filename in REQUIRED_FILES:
        try:
            _read_json(encoder, filename, allow_fetch=False)
        except EncoderUnavailable:
            return False
    return True


def load_geometry(
    encoder: VisionEncoder, *, allow_fetch: bool = True
) -> EncoderGeometry:
    """The encoder's geometry, read out of its own configuration.

    Every field is required rather than defaulted. The library's own
    config defaults are placeholders, 224px images and 32px patches
    with a scale factor of 2, which describe neither checkpoint; a
    default here would quietly answer with those and the page would
    draw a grid nobody uses.
    """
    config = _read_json(
        encoder, CONFIG_NAME, allow_fetch=allow_fetch
    )
    preprocessor = _read_json(
        encoder, PREPROCESSOR_NAME, allow_fetch=allow_fetch
    )

    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        raise EncoderUnavailable(
            f"{encoder.display_name} declares no vision_config"
        )

    longest_edge = _longest_edge(encoder, preprocessor, "size")
    tile = _longest_edge(encoder, preprocessor, "max_image_size")
    geometry = EncoderGeometry(
        longest_edge=longest_edge,
        tile=tile,
        patch=_positive(encoder, vision, "patch_size"),
        scale=_positive(encoder, config, "scale_factor"),
    )

    # The assumption the geometry's own tests pin: when the bound is a
    # whole number of tiles, two different roundings of the shorter
    # side agree, and the arithmetic reproduced from the processor is
    # only checked against sizes where that holds.
    if not geometry.longest_edge_is_whole_tiles:
        logger.warning(
            "%s bounds the longest edge at %d against a %d tile,"
            " which is not a whole number of tiles; the tile grid"
            " has not been verified against the processor for that",
            encoder.display_name,
            geometry.longest_edge,
            geometry.tile,
        )
    return geometry


def _longest_edge(
    encoder: VisionEncoder, section: Dict[str, object], key: str
) -> int:
    """A `{"longest_edge": N}` bound, the form both bounds use."""
    bound = section.get(key)
    if not isinstance(bound, dict) or "longest_edge" not in bound:
        raise EncoderUnavailable(
            f"{encoder.display_name} has no {key}.longest_edge"
        )
    return _positive(encoder, bound, "longest_edge")


def _positive(
    encoder: VisionEncoder, section: Dict[str, object], key: str
) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EncoderUnavailable(
            f"{encoder.display_name} has no integer {key}"
        )
    if value <= 0:
        raise EncoderUnavailable(
            f"{encoder.display_name} declares {key} as {value}"
        )
    return value


def find(encoder_id: str) -> Optional[VisionEncoder]:
    """The encoder with this id, or None for the caller to reject."""
    return ENCODERS.get(encoder_id)


def declared() -> Tuple[VisionEncoder, ...]:
    """Every encoder, in declaration order.

    Order matters to the page: it lists them cheapest first, so the
    one a reader can actually run comes up first on a small machine.
    """
    return tuple(ENCODERS.values())
