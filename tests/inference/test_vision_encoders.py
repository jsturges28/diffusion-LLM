"""The declared encoders are pinned, and their geometry is read.

Strategy: check the declaration's own invariants, asserted at import
so they cannot be added around, then drive the config reader
against handwritten JSON in `tmp_path` so every malformed shape is
exercised without a network or a checkpoint. The two real checkpoints
are read only when their configs are already cached, and skipped
otherwise, because a test that downloads is a test that fails offline.

What passing proves is that a number on the page came out of a pinned
commit's own configuration, and that a missing or malformed config is
reported rather than defaulted. The second half matters more than it
sounds: `transformers` ships placeholder defaults for these fields,
224px images with 32px patches, which describe neither checkpoint, so
a reader defaulting silently would draw a plausible grid nobody uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.inference import vision_encoders
from src.inference.vision_encoders import (
    ENCODERS,
    CONFIG_NAME,
    PREPROCESSOR_NAME,
    EncoderUnavailable,
    VisionEncoder,
    declared,
    find,
    is_cached,
    load_geometry,
)

# What a real pair of files looks like, reduced to the fields read.
GOOD_CONFIG: Dict[str, Any] = {
    "scale_factor": 3,
    "vision_config": {"image_size": 384, "patch_size": 14},
}
GOOD_PREPROCESSOR: Dict[str, Any] = {
    "size": {"longest_edge": 1536},
    "max_image_size": {"longest_edge": 384},
}

FAKE = VisionEncoder(
    id="fake",
    display_name="Fake",
    repo_id="example/fake",
    revision="0" * 40,
    summary="Only for driving the reader.",
)


@pytest.fixture()
def reader(monkeypatch, tmp_path: Path):
    """The config reader, pointed at files on disk.

    Patches the download rather than the filesystem, so the reader's
    own JSON handling and validation are what run.
    """
    def fake_download(repo_id, filename, revision=None,
                      local_files_only=False):
        assert repo_id == FAKE.repo_id, repo_id
        assert revision == FAKE.revision, "the pin was not passed on"
        path = tmp_path / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return str(path)

    module = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    return tmp_path


def _write(directory: Path, config=None, preprocessor=None) -> None:
    if config is not None:
        (directory / CONFIG_NAME).write_text(json.dumps(config))
    if preprocessor is not None:
        (directory / PREPROCESSOR_NAME).write_text(
            json.dumps(preprocessor)
        )


# -- the declaration --


def test_there_are_encoders_to_compare() -> None:
    """Guards everything below, and states the point of the module:
    one encoder on its own has nothing to say."""
    assert len(declared()) >= 2


def test_every_encoder_pins_a_full_commit() -> None:
    """The same rule the model registry enforces. Asserted at import
    there and here, so this only re-checks what already ran."""
    for encoder in declared():
        assert len(encoder.revision) == 40
        assert encoder.revision.strip() == encoder.revision


def test_the_ids_are_unique_and_url_safe() -> None:
    ids = [encoder.id for encoder in declared()]

    assert len(ids) == len(set(ids))
    for value in ids:
        assert value == value.lower()
        assert " " not in value


def test_an_unknown_id_is_not_found() -> None:
    """Negative space: the endpoint rejects on this answer, so it has
    to be None rather than a default encoder."""
    assert find("nope") is None
    assert find("") is None


def test_a_known_id_is_found() -> None:
    for encoder in declared():
        assert find(encoder.id) is encoder


# -- reading a configuration --


def test_a_good_pair_reads(reader: Path) -> None:
    _write(reader, GOOD_CONFIG, GOOD_PREPROCESSOR)

    geometry = load_geometry(FAKE)

    assert geometry.longest_edge == 1536
    assert geometry.tile == 384
    assert geometry.patch == 14
    assert geometry.scale == 3


def test_the_geometry_comes_from_the_files_not_a_default(
    reader: Path,
) -> None:
    """The defence against the library's placeholder defaults. A
    different pair has to produce a different geometry, or the reader
    is answering from somewhere else."""
    _write(
        reader,
        {"scale_factor": 4,
         "vision_config": {"image_size": 512, "patch_size": 16}},
        {"size": {"longest_edge": 2048},
         "max_image_size": {"longest_edge": 512}},
    )

    geometry = load_geometry(FAKE)

    assert (geometry.tile, geometry.patch, geometry.scale) == (
        512, 16, 4
    )


@pytest.mark.parametrize("missing", [CONFIG_NAME, PREPROCESSOR_NAME])
def test_a_missing_file_is_reported(
    reader: Path, missing: str
) -> None:
    _write(reader, GOOD_CONFIG, GOOD_PREPROCESSOR)
    (reader / missing).unlink()

    with pytest.raises(EncoderUnavailable):
        load_geometry(FAKE)


def test_a_config_with_no_vision_section_is_reported(
    reader: Path,
) -> None:
    _write(reader, {"scale_factor": 3}, GOOD_PREPROCESSOR)

    with pytest.raises(EncoderUnavailable, match="vision_config"):
        load_geometry(FAKE)


@pytest.mark.parametrize("key", ["size", "max_image_size"])
def test_a_missing_bound_is_reported(reader: Path, key: str) -> None:
    preprocessor = dict(GOOD_PREPROCESSOR)
    del preprocessor[key]
    _write(reader, GOOD_CONFIG, preprocessor)

    with pytest.raises(EncoderUnavailable, match=key):
        load_geometry(FAKE)


def test_a_bound_without_longest_edge_is_reported(
    reader: Path,
) -> None:
    _write(
        reader, GOOD_CONFIG,
        {"size": {"height": 1536},
         "max_image_size": {"longest_edge": 384}},
    )

    with pytest.raises(EncoderUnavailable, match="size"):
        load_geometry(FAKE)


@pytest.mark.parametrize("value", [0, -1, "384", None, 384.0, True])
def test_a_non_positive_integer_is_reported(
    reader: Path, value: Any
) -> None:
    """Strings and floats included because JSON permits both, and
    `True` because it is an `int` in Python and would otherwise become
    a patch size of one."""
    config = {
        "scale_factor": 3,
        "vision_config": {"image_size": 384, "patch_size": value},
    }
    _write(reader, config, GOOD_PREPROCESSOR)

    with pytest.raises(EncoderUnavailable, match="patch_size"):
        load_geometry(FAKE)


def test_a_config_that_is_not_an_object_is_reported(
    reader: Path,
) -> None:
    (reader / CONFIG_NAME).write_text("[1, 2, 3]")
    _write(reader, preprocessor=GOOD_PREPROCESSOR)

    with pytest.raises(EncoderUnavailable, match="not an object"):
        load_geometry(FAKE)


def test_an_impossible_geometry_is_refused(reader: Path) -> None:
    """A patch larger than its tile is rejected by the geometry's own
    contract, which this reader must not swallow."""
    _write(
        reader,
        {"scale_factor": 3,
         "vision_config": {"image_size": 384, "patch_size": 999}},
        GOOD_PREPROCESSOR,
    )

    with pytest.raises(AssertionError):
        load_geometry(FAKE)


def test_caching_is_reported_without_a_fetch(reader: Path) -> None:
    _write(reader, GOOD_CONFIG, GOOD_PREPROCESSOR)
    assert is_cached(FAKE)

    (reader / CONFIG_NAME).unlink()

    assert not is_cached(FAKE)


def test_a_fetch_can_be_refused(reader: Path) -> None:
    """The offline path. With nothing cached and fetching disallowed,
    the answer is an operating error the page can render, not a
    crash."""
    with pytest.raises(EncoderUnavailable, match="no fetch"):
        load_geometry(FAKE, allow_fetch=False)


# -- and the real checkpoints, when they are already here --


@pytest.mark.parametrize("encoder_id", sorted(ENCODERS))
def test_a_cached_checkpoint_reads_the_expected_geometry(
    encoder_id: str,
) -> None:
    """The end of the chain, and the only test here touching a real
    checkpoint. Skipped rather than downloading, so the suite stays
    offline; the expected values are the ones the page's copy quotes,
    so a checkpoint that changed them fails here first.
    """
    encoder = ENCODERS[encoder_id]
    if not is_cached(encoder):
        pytest.skip(f"{encoder.display_name} config is not cached")

    geometry = load_geometry(encoder, allow_fetch=False)

    expected = {
        "smolvlm-500m": (2048, 512, 16, 4),
        "smolvlm-2b": (1536, 384, 14, 3),
    }[encoder_id]
    assert (
        geometry.longest_edge,
        geometry.tile,
        geometry.patch,
        geometry.scale,
    ) == expected


def test_the_module_declares_the_files_it_reads() -> None:
    """Named constants rather than literals at the call sites, because
    the test fixtures above write the same two names."""
    assert vision_encoders.REQUIRED_FILES == (
        CONFIG_NAME, PREPROCESSOR_NAME
    )
