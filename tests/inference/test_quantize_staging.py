"""An interrupted quantization leaves nothing that looks installed.

Strategy: drive the real `main()` from the quantize script with the
one expensive phase replaced by a stub. Everything else is the
script's own control flow: the staging directory it creates, the
checks it makes before starting, the cleanup on the way out, and the
rename at the end. The stub stands in for a twenty-minute load and a
16 GB `torch.save`, so the phase that cannot run without a GPU is the
only thing not exercised here.

What passing proves: whatever happens during a build, the destination
either does not exist or is a complete artifact. There is no third
state. That was the bug: the script wrote straight into its
destination, so a Ctrl-C during the save left a directory holding a
truncated state dict that the menu offered as an installed model.

Why the interrupt is tested rather than verified by hand. The
maintainer's hardware can run the real build once, which proves it
works and proves nothing about the failure paths: reproducing "killed
during the save" on demand, four different ways, is not something to
ask of a twenty-minute GPU-bound job. Here it is four cheap tests.

`KeyboardInterrupt` specifically, and not only exceptions. It does not
inherit from `Exception`, so the obvious `except Exception` would let
a Ctrl-C through untouched and leave the directory behind, which is
the single most likely way a real build ends early.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

import torch

from scripts import quantize_diffusiongemma_nf4 as quantize
from src.inference.artifact_manifest import (
    MANIFEST_NAME,
    build_manifest,
    file_digest,
    is_complete_artifact,
    staging_path,
    write_manifest,
)

PAYLOAD = b"stands in for sixteen gigabytes"


@pytest.fixture(autouse=True)
def _pretend_there_is_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sandbox has no CUDA, and none of this needs it.

    Stubbed rather than skipped, because the control flow under test
    is the part that runs on any machine. The real GPU work is the
    one phase these tests replace.
    """
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True
    )


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    """Invoke the script the way a shell would."""
    monkeypatch.setattr(
        sys, "argv", ["quantize_diffusiongemma_nf4.py", *args]
    )
    quantize.main()


def _base(tmp_path: Path) -> Path:
    """A base checkpoint, of which only its existence matters here."""
    base = tmp_path / "bf16"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    return base


def _write_artifact(directory: Path) -> None:
    """What a real ``stage_artifact`` leaves behind, in miniature."""
    weights = directory / quantize.STATE_DICT_NAME
    weights.write_bytes(PAYLOAD)
    write_manifest(
        directory,
        build_manifest(
            artifact=quantize.ARTIFACT_NAME,
            base_path="/models/bf16",
            base_revision=None,
            state_dict_name=quantize.STATE_DICT_NAME,
            state_dict_bytes=weights.stat().st_size,
            state_dict_sha256=file_digest(weights),
            copied_files=[],
        ),
    )


def _stage_that(failure: BaseException) -> Any:
    """A staging phase that gets partway and then stops.

    Writes a partial state dict first, because cleaning up an empty
    directory would be a weaker claim than cleaning up one holding
    most of a 16 GB file.
    """

    def _stage(*, base: Path, staging: Path, **kwargs: Any) -> None:
        del base, kwargs
        (staging / quantize.STATE_DICT_NAME).write_bytes(
            PAYLOAD[:10]
        )
        raise failure

    return _stage


def _stage_ok(*, base: Path, staging: Path, **kwargs: Any) -> None:
    del base, kwargs
    _write_artifact(staging)


# -- a build that stops partway --


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        RuntimeError("out of memory"),
        OSError("no space left on device"),
    ],
    ids=["ctrl-c", "runtime", "disk-full"],
)
def test_an_interrupted_build_leaves_no_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    """The finding in one assertion, three ways.

    A Ctrl-C, a CUDA failure and a full disk are the three realistic
    endings, and they arrive as different base classes, which is
    exactly how an over-narrow `except` clause lets one through.
    """
    out = tmp_path / "nf4"
    monkeypatch.setattr(
        quantize, "stage_artifact", _stage_that(failure)
    )

    with pytest.raises(type(failure)):
        _run(
            monkeypatch,
            "--base",
            str(_base(tmp_path)),
            "--out",
            str(out),
        )

    assert not out.exists()
    assert not staging_path(out).exists()


def test_an_interrupted_build_discards_its_partial_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing is left anywhere for a later run to mistake for
    progress. The partial file is unresumable by construction: there
    is no way to tell how far a killed save got."""
    out = tmp_path / "nf4"
    monkeypatch.setattr(
        quantize, "stage_artifact", _stage_that(KeyboardInterrupt())
    )

    with pytest.raises(KeyboardInterrupt):
        _run(
            monkeypatch,
            "--base",
            str(_base(tmp_path)),
            "--out",
            str(out),
        )

    survivors = list(tmp_path.glob("nf4*"))
    assert survivors == []


def test_a_stale_staging_directory_is_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Left by a run killed so hard its own cleanup did not run, for
    example SIGKILL or a power loss. Discarded rather than resumed,
    for the same reason the partial file above is."""
    out = tmp_path / "nf4"
    staging = staging_path(out)
    staging.mkdir()
    (staging / quantize.STATE_DICT_NAME).write_bytes(b"leftover")
    monkeypatch.setattr(quantize, "stage_artifact", _stage_ok)

    _run(
        monkeypatch,
        "--base",
        str(_base(tmp_path)),
        "--out",
        str(out),
    )

    assert is_complete_artifact(out)
    assert (out / quantize.STATE_DICT_NAME).read_bytes() == PAYLOAD


# -- a build that finishes --


def test_a_finished_build_appears_in_one_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The destination goes from absent to complete with nothing in
    between, which is what makes "does it exist" a safe question for
    anything other than this script to ask."""
    out = tmp_path / "nf4"
    monkeypatch.setattr(quantize, "stage_artifact", _stage_ok)

    _run(
        monkeypatch,
        "--base",
        str(_base(tmp_path)),
        "--out",
        str(out),
    )

    assert is_complete_artifact(out)
    assert not staging_path(out).exists()


def test_a_build_refuses_to_replace_a_complete_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Overwriting a model that may be loading right now is not this
    script's call to make."""
    out = tmp_path / "nf4"
    out.mkdir()
    _write_artifact(out)
    monkeypatch.setattr(quantize, "stage_artifact", _stage_ok)

    with pytest.raises(RuntimeError, match="already holds"):
        _run(
            monkeypatch,
            "--base",
            str(_base(tmp_path)),
            "--out",
            str(out),
        )

    assert is_complete_artifact(out)


def test_a_build_refuses_to_write_into_an_unattested_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Says so instead of merging into it. Writing alongside whatever
    is in there would produce a directory built from two runs, which
    the manifest would then attest as one."""
    out = tmp_path / "nf4"
    out.mkdir()
    (out / quantize.STATE_DICT_NAME).write_bytes(b"who knows")
    monkeypatch.setattr(quantize, "stage_artifact", _stage_ok)

    with pytest.raises(RuntimeError, match="not a complete"):
        _run(
            monkeypatch,
            "--base",
            str(_base(tmp_path)),
            "--out",
            str(out),
        )


# -- adoption, which must work where a build cannot --


def test_adoption_needs_no_gpu_and_no_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The escape hatch, and the property that broke first.

    Adoption is documented as running in `.venv` on a machine with no
    GPU, so it must not touch CUDA and must not require the base
    checkpoint, which a user may well have deleted after building.
    The heavy imports live inside `stage_artifact` for this reason;
    at module scope they made the command fail before parsing its
    arguments.
    """
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("adoption asked about CUDA"),
    )
    out = tmp_path / "nf4"
    out.mkdir()
    (out / quantize.STATE_DICT_NAME).write_bytes(PAYLOAD)

    _run(
        monkeypatch,
        "--adopt",
        "--base",
        str(tmp_path / "base-that-is-gone"),
        "--out",
        str(out),
    )

    assert is_complete_artifact(out)


def test_adoption_digests_the_weights_that_are_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The size and digest describe this directory, so they are
    observations. The base and quantizer commit are the user's claim,
    which is the honest division and is why adoption cannot detect a
    directory that was already truncated."""
    import json

    out = tmp_path / "nf4"
    out.mkdir()
    weights = out / quantize.STATE_DICT_NAME
    weights.write_bytes(PAYLOAD)

    _run(
        monkeypatch, "--adopt", "--out", str(out)
    )

    manifest = json.loads(
        (out / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["state_dict"]["bytes"] == len(PAYLOAD)
    assert manifest["state_dict"]["sha256"] == file_digest(weights)


def test_adoption_refuses_a_directory_with_no_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative space. Attesting an empty directory would be the
    original bug with extra steps: a manifest asserting a model is
    present when it is not."""
    out = tmp_path / "nf4"
    out.mkdir()

    with pytest.raises(RuntimeError, match="nothing to attest"):
        _run(monkeypatch, "--adopt", "--out", str(out))

    assert not is_complete_artifact(out)


def test_adoption_records_a_declared_base_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A base kept in a plain directory has no commit in its path, so
    the user is the only source for it."""
    import json

    out = tmp_path / "nf4"
    out.mkdir()
    (out / quantize.STATE_DICT_NAME).write_bytes(PAYLOAD)
    sha = "b" * 40

    _run(
        monkeypatch,
        "--adopt",
        "--out",
        str(out),
        "--base-revision",
        sha,
    )

    manifest = json.loads(
        (out / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["base"]["revision"] == sha


# -- the space check around the build --


def test_a_build_refuses_before_loading_when_space_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The early check earns its place by failing before the load,
    not by being exact. Being refused after twenty minutes of GPU
    work would report the same problem and save nothing."""
    reached: List[str] = []

    def _stage(**kwargs: Any) -> None:
        del kwargs
        reached.append("staged")

    monkeypatch.setattr(quantize, "stage_artifact", _stage)
    monkeypatch.setattr(quantize, "free_bytes", lambda p: 0)
    out = tmp_path / "nf4"

    with pytest.raises(RuntimeError, match="Free some space"):
        _run(
            monkeypatch,
            "--base",
            str(_base(tmp_path)),
            "--out",
            str(out),
        )

    assert reached == []
    assert not out.exists()
    assert not staging_path(out).exists()
