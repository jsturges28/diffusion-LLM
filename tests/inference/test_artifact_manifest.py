"""A locally built artifact says whether it finished.

Strategy: build small artifacts in `tmp_path` and ask the reader what
it makes of them, including every state an interrupted build can leave
behind. The state dict is a few bytes of nothing here; the reader only
cares about presence and size, which is the point, because that is all
it can afford to care about for a real 16 GB file.

What passing proves: no directory reads as a finished artifact unless
a manifest says so and the weights match what it says. The finding's
own verification clause is "partial local output must never appear
ready", and it used to, in the simplest possible way: the menu asked
whether the directory existed.

The interrupt cases are the substance. A build can be killed after
`mkdir`, during a partial `torch.save`, after a complete save but
before the copied config files, and after all of those but before the
manifest. Every one of them used to leave a directory that looked
installed. They are enumerated here rather than summarised, because
"an interrupted build" is not one state and a reader that handled
three of the four would still ship the bug.

The write ordering is tested too, as a property rather than an
implementation detail. A manifest written before the file it describes
would be the same lie as the bare directory, so `write_manifest`
refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.inference.artifact_manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    build_manifest,
    file_digest,
    is_complete_artifact,
    promote,
    read_manifest,
    repo_commit,
    staging_path,
    write_manifest,
)

WEIGHTS = "model_nf4.pt"
PAYLOAD = b"not really sixteen gigabytes"


def _weights(directory: Path, body: bytes = PAYLOAD) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / WEIGHTS
    path.write_bytes(body)
    return path


def _manifest(directory: Path, **overrides: Any) -> Dict[str, Any]:
    """A manifest for whatever is in ``directory`` right now."""
    weights = directory / WEIGHTS
    manifest = build_manifest(
        artifact="test-artifact",
        base_path="/models/base",
        base_revision="a" * 40,
        state_dict_name=WEIGHTS,
        state_dict_bytes=weights.stat().st_size,
        state_dict_sha256=file_digest(weights),
        copied_files=["config.json"],
    )
    manifest.update(overrides)
    return manifest


def _finished(directory: Path) -> Path:
    """A directory in the state a successful build leaves."""
    _weights(directory)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    write_manifest(directory, _manifest(directory))
    return directory


# -- the finished article --


def test_a_finished_artifact_reads_as_complete(
    tmp_path: Path,
) -> None:
    assert is_complete_artifact(_finished(tmp_path / "nf4"))


def test_the_manifest_names_what_it_was_built_from(
    tmp_path: Path,
) -> None:
    """The half of this that is about identification rather than
    completeness. A directory name does not say which base, which
    commit of the quantizer, or which weights."""
    directory = _finished(tmp_path / "nf4")

    manifest = read_manifest(directory)

    assert manifest is not None
    assert manifest["base"]["revision"] == "a" * 40
    assert manifest["base"]["path"] == "/models/base"
    assert manifest["quantizer"]["repo_commit"] == repo_commit()
    assert manifest["state_dict"]["sha256"] == file_digest(
        directory / WEIGHTS
    )


def test_the_digest_is_of_the_bytes_on_disk(
    tmp_path: Path,
) -> None:
    """Pairs the recorded digest with an independent reading of the
    same file, which is the only way the record means anything."""
    import hashlib

    directory = _finished(tmp_path / "nf4")

    assert file_digest(directory / WEIGHTS) == (
        hashlib.sha256(PAYLOAD).hexdigest()
    )


# -- every way a build can be interrupted --


def test_a_bare_directory_is_not_an_artifact(
    tmp_path: Path,
) -> None:
    """Killed right after mkdir. This is the exact state the old
    check read as an installed model."""
    directory = tmp_path / "nf4"
    directory.mkdir()

    assert not is_complete_artifact(directory)


def test_weights_without_a_manifest_are_not_an_artifact(
    tmp_path: Path,
) -> None:
    """Killed during or after the save. The file is there and may be
    any fraction of what it should be; nothing on disk says which."""
    directory = tmp_path / "nf4"
    _weights(directory)

    assert not is_complete_artifact(directory)


def test_a_truncated_state_dict_is_detected(
    tmp_path: Path,
) -> None:
    """The case the recorded size exists for.

    A save that ran out of disk leaves a shorter file, and every other
    check passes: the manifest is there, the version matches, the name
    matches. Only the length disagrees.
    """
    directory = _finished(tmp_path / "nf4")
    (directory / WEIGHTS).write_bytes(PAYLOAD[:5])

    assert not is_complete_artifact(directory)


def test_a_missing_state_dict_is_detected(
    tmp_path: Path,
) -> None:
    """Negative space for the check above: a manifest describing a
    file that is not there at all."""
    directory = _finished(tmp_path / "nf4")
    (directory / WEIGHTS).unlink()

    assert not is_complete_artifact(directory)


def test_a_config_file_is_not_required_to_read_as_complete(
    tmp_path: Path,
) -> None:
    """Deliberate, and worth pinning so it is not read as an
    oversight.

    The copied files are small and are written before the manifest, so
    a build interrupted between them cannot produce a manifest at all.
    Re-checking them here would only re-state what the ordering
    already guarantees.
    """
    directory = _finished(tmp_path / "nf4")
    (directory / "config.json").unlink()

    assert is_complete_artifact(directory)


# -- manifests a reader cannot trust --


def test_an_unknown_manifest_version_is_refused(
    tmp_path: Path,
) -> None:
    """Refused rather than read optimistically. The version goes up
    only when an old manifest would be misread, so guessing is the
    one thing a reader must not do with it."""
    directory = _finished(tmp_path / "nf4")
    write_manifest(
        directory,
        _manifest(
            directory, manifest_version=MANIFEST_VERSION + 1
        ),
    )

    assert not is_complete_artifact(directory)


def test_unparseable_json_reads_as_no_manifest(
    tmp_path: Path,
) -> None:
    directory = _finished(tmp_path / "nf4")
    (directory / MANIFEST_NAME).write_text(
        "{not json", encoding="utf-8"
    )

    assert read_manifest(directory) is None
    assert not is_complete_artifact(directory)


def test_a_json_array_reads_as_no_manifest(
    tmp_path: Path,
) -> None:
    """Valid JSON that is not a manifest. Returning it would hand the
    caller a list to call .get() on."""
    directory = _finished(tmp_path / "nf4")
    (directory / MANIFEST_NAME).write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )

    assert read_manifest(directory) is None


def test_a_manifest_with_no_state_dict_block_is_refused(
    tmp_path: Path,
) -> None:
    directory = _finished(tmp_path / "nf4")
    manifest = _manifest(directory)
    del manifest["state_dict"]
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert not is_complete_artifact(directory)


def test_a_manifest_claiming_zero_bytes_is_refused(
    tmp_path: Path,
) -> None:
    """A zero-length state dict is not a model, and a zero expected
    size would make the length check pass for an empty file."""
    directory = _finished(tmp_path / "nf4")
    manifest = _manifest(directory)
    manifest["state_dict"]["bytes"] = 0
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert not is_complete_artifact(directory)


def test_a_missing_directory_is_not_an_artifact(
    tmp_path: Path,
) -> None:
    assert not is_complete_artifact(tmp_path / "never-built")


# -- the write ordering the completion signal depends on --


def test_a_manifest_cannot_be_written_before_its_weights(
    tmp_path: Path,
) -> None:
    """The property that makes a manifest mean anything.

    Written first, it would say "finished" about a directory whose
    16 GB save had not started, which is precisely the state this
    whole mechanism exists to distinguish.
    """
    directory = tmp_path / "nf4"
    _weights(directory)
    manifest = _manifest(directory)
    (directory / WEIGHTS).unlink()

    with pytest.raises(AssertionError):
        write_manifest(directory, manifest)


# -- staging and promotion --


def test_staging_is_a_sibling_of_the_destination(
    tmp_path: Path,
) -> None:
    """A rename is only atomic within one filesystem, and /tmp is
    routinely a different one, so staging elsewhere would turn the
    atomic step into a 16 GB copy."""
    final = tmp_path / "models" / "nf4"

    staging = staging_path(final)

    assert staging.parent == final.parent
    assert staging != final


def test_staging_is_not_mistaken_for_a_finished_artifact(
    tmp_path: Path,
) -> None:
    """Even a fully built staging directory must not read as the
    model, or an interrupted build would be picked up under its
    temporary name."""
    final = tmp_path / "nf4"
    staged = _finished(staging_path(final))

    assert is_complete_artifact(staged)
    assert not is_complete_artifact(final)


def test_promotion_moves_the_whole_directory(
    tmp_path: Path,
) -> None:
    final = tmp_path / "nf4"
    staging = staging_path(final)
    _finished(staging)

    promote(staging, final)

    assert is_complete_artifact(final)
    assert not staging.exists()


def test_promotion_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    """Replacing a model that may be loading is the caller's
    decision, not this function's."""
    final = _finished(tmp_path / "nf4")
    staging = staging_path(final)
    _finished(staging)

    with pytest.raises(FileExistsError):
        promote(staging, final)

    assert is_complete_artifact(final)
    assert staging.exists()


# -- what the menu reads off the same directories --


def test_the_menu_calls_a_finished_artifact_downloaded(
    tmp_path: Path,
) -> None:
    """The reader that actually matters to a user.

    Tested here rather than in the server tests because it is the
    same question this file spends its length on, asked through the
    supervisor's probe: the two must not be able to disagree about
    what a directory is.
    """
    from src.web.server import _is_downloaded

    assert _is_downloaded(str(_finished(tmp_path / "nf4")))


def test_the_menu_does_not_call_a_bare_directory_downloaded(
    tmp_path: Path,
) -> None:
    """The finding in one assertion. This used to be True."""
    from src.web.server import _is_downloaded

    directory = tmp_path / "nf4"
    directory.mkdir()

    assert not _is_downloaded(str(directory))


def test_the_menu_does_not_call_a_truncated_save_downloaded(
    tmp_path: Path,
) -> None:
    from src.web.server import _is_downloaded

    directory = _finished(tmp_path / "nf4")
    (directory / WEIGHTS).write_bytes(PAYLOAD[:5])

    assert not _is_downloaded(str(directory))
