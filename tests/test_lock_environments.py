"""The committed locks still match the manifest that produced them.

Strategy: two halves, both offline. First the drift guard, driven
against synthetic manifests and locks in `tmp_path`, so every way a
lock can fall out of step is exercised without a resolver. Then the
real four files, read as data and checked for the properties a
generated lock must have.

What passing proves is that the manifest cannot become fiction. The
whole point of splitting direct requirements from a resolved lock is
that the short file is the one people read and edit; if editing it
without regenerating went unnoticed, the locks would drift back into
being the only truth and the manifest would be a comment.

Why the guard is a digest and not a re-resolve. Re-resolving is the
only way to prove a lock is what the manifest *would* produce, and it
needs the network, minutes, and an index that answers the same way
twice. This instead proves the lock was produced from this manifest
entry, which is the property that actually decays, and it costs a hash
of a dict. `--update` does the resolving, once, deliberately.

The digest covers canonical JSON rather than the TOML text so that
reordering a list or rewrapping a comment does not invalidate a lock
that would compile identically. Both halves of that are tested: a
changed requirement must invalidate, and a reordered one must not.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts import lock_environments

REPO_ROOT = Path(__file__).resolve().parents[1]

# The four the manifest declares. Named here so a fifth environment
# arriving without a test is a failure rather than an omission, which
# is the whole reason DEPS-01 was done before `.venv-ssm`.
EXPECTED_ENVIRONMENTS = {"core", "dgemma", "ar", "desktop"}


def _manifest() -> Dict[str, Any]:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["tool"]["diffusion-llm"]


def _lock_text(name: str) -> str:
    config = _manifest()
    lock = config["environments"][name]["lock"]
    return (REPO_ROOT / lock).read_text(encoding="utf-8")


# -- the manifest describes what we think it describes --


def test_the_manifest_declares_every_environment() -> None:
    config = _manifest()

    assert set(config["environments"]) == EXPECTED_ENVIRONMENTS


def test_every_environment_names_a_lock_that_exists() -> None:
    config = _manifest()

    for name, environment in config["environments"].items():
        lock = REPO_ROOT / environment["lock"]
        assert lock.is_file(), f"{name} points at a missing {lock}"


def test_every_environment_is_placed_or_extends_one() -> None:
    """Negative space on the overlay case.

    An environment with neither a path nor an `extends` would be a
    lock nobody could install, and an environment with both would be
    two contradictory claims about where it lives.
    """
    config = _manifest()

    for name, environment in config["environments"].items():
        has_path = "path" in environment
        has_parent = "extends" in environment
        assert has_path or has_parent, f"{name} has no home"
        assert not (has_path and has_parent), (
            f"{name} claims both a path and a parent"
        )


def test_the_overlay_extends_a_declared_environment() -> None:
    config = _manifest()
    names = set(config["environments"])

    for name, environment in config["environments"].items():
        parent = environment.get("extends")
        if parent is not None:
            assert parent in names, (
                f"{name} extends unknown environment {parent}"
            )


# -- the committed locks are in step --


def test_the_committed_locks_match_the_manifest() -> None:
    """The headline check, and the one that runs on every push."""
    assert lock_environments.check() == 0


@pytest.mark.parametrize(
    "name", sorted(EXPECTED_ENVIRONMENTS)
)
def test_every_declared_requirement_is_pinned(name: str) -> None:
    """A requirement that vanished from its own lock.

    The digest cannot catch this: it proves the lock came from this
    manifest, not that the resolver honoured every line. A package
    dropped because it was only an optional extra of something else
    is exactly how `pillow` and `protobuf` went missing from the
    DiffusionGemma resolve.
    """
    config = _manifest()
    environment = config["environments"][name]
    pins = lock_environments.pins_in(_lock_text(name))

    for raw in environment["requirements"]:
        # Strip any extras and any version pin to get the name.
        bare = re.split(r"[\[=<>!~]", raw)[0]
        bare = bare.strip().lower().replace("_", "-")
        assert bare in pins, (
            f"{name} asks for {bare} and its lock does not pin it"
        )


@pytest.mark.parametrize(
    "name", sorted(EXPECTED_ENVIRONMENTS)
)
def test_every_pin_carries_a_hash(name: str) -> None:
    """Without hashes a lock records intent but verifies nothing, and
    a compromised index could serve anything for a pinned name."""
    text = _lock_text(name)
    pins = lock_environments.pins_in(text)

    assert pins, f"{name}'s lock pins nothing"
    assert text.count("--hash=") >= len(pins), (
        f"{name} has fewer hashes than pins"
    )


@pytest.mark.parametrize(
    "name", sorted(EXPECTED_ENVIRONMENTS)
)
def test_every_lock_says_it_is_generated(name: str) -> None:
    """So nobody edits one by hand, which is how these four became
    the only record of intent in the first place."""
    text = _lock_text(name)

    assert lock_environments.GENERATED_BY in text
    assert lock_environments.DO_NOT_EDIT in text


def test_the_overlay_includes_its_parent() -> None:
    """The desktop file is documented as installable on its own.

    Recording `extends` in a header comment while dropping the `-r`
    would still install: it would install a native window with no app
    underneath it. This was a real defect in the first generated
    version of that file.
    """
    config = _manifest()
    for name, environment in config["environments"].items():
        parent = environment.get("extends")
        if parent is None:
            continue
        parent_lock = config["environments"][parent]["lock"]
        assert f"-r {parent_lock}" in _lock_text(name), (
            f"{name} does not include {parent_lock}"
        )


def test_the_pinned_python_is_the_one_ruff_targets() -> None:
    """Two files claiming different Python versions is how the
    README came to advertise 3.10 while nothing had run on it."""
    config = _manifest()
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ruff = tomllib.loads(text)["tool"]["ruff"]["target-version"]

    assert ruff == "py" + config["python-version"].replace(".", "")


# -- the guard itself, against synthetic manifests --


def _fake_environment(
    requirements: List[str], **extra: Any
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "description": "test",
        "path": ".venv-test",
        "lock": "requirements-test.txt",
        "requirements": requirements,
    }
    base.update(extra)
    return base


SHARED = {"python-version": "3.12", "index": "https://example/simple"}


def test_a_changed_requirement_changes_the_digest() -> None:
    before = lock_environments.manifest_digest(
        SHARED, _fake_environment(["torch", "numpy"])
    )
    after = lock_environments.manifest_digest(
        SHARED, _fake_environment(["torch", "numpy", "pillow"])
    )

    assert before != after


def test_reordering_a_requirement_does_not() -> None:
    """The pair to the test above, and the reason the digest is over
    sorted canonical JSON rather than the file's text. Invalidating a
    16 GB-adjacent resolve because somebody alphabetized a list would
    train people to ignore the guard."""
    one = lock_environments.manifest_digest(
        SHARED, _fake_environment(["torch", "numpy"])
    )
    other = lock_environments.manifest_digest(
        SHARED, _fake_environment(["numpy", "torch"])
    )

    assert one == other


def test_changing_the_python_version_changes_every_digest() -> None:
    """It is shared, so it must invalidate all four locks at once. A
    resolve for 3.12 is not a resolve for 3.13."""
    environment = _fake_environment(["torch"])
    before = lock_environments.manifest_digest(SHARED, environment)
    after = lock_environments.manifest_digest(
        {**SHARED, "python-version": "3.13"}, environment
    )

    assert before != after


def test_changing_the_index_changes_the_digest() -> None:
    """A different index can serve a different artifact for the
    same version, so a lock made against one does not describe
    the other."""
    environment = _fake_environment(["torch"])
    before = lock_environments.manifest_digest(SHARED, environment)
    after = lock_environments.manifest_digest(
        {**SHARED, "index": "https://other/simple"}, environment
    )

    assert before != after


# -- reading a digest back off a lock --


def test_a_generated_header_round_trips() -> None:
    """Written and read by the same module, so the two cannot drift
    apart without this failing."""
    environment = _fake_environment(["torch"])
    rendered = lock_environments.render_lock(
        "test", SHARED, environment, "torch==2.8.0\n",
        {"test": environment},
    )

    assert lock_environments.recorded_digest(rendered) == (
        lock_environments.manifest_digest(SHARED, environment)
    )


def test_a_lock_with_no_header_records_no_digest() -> None:
    """What every one of these files looked like before this: a bare
    freeze, which must read as unvouched-for rather than as valid."""
    assert lock_environments.recorded_digest("torch==2.8.0\n") is None


def test_a_digest_below_the_header_is_not_read() -> None:
    """A pin is where the header stops. Reading further would let a
    comment buried among 2,900 hash lines pass for provenance."""
    text = (
        "# environment: test\n"
        "torch==2.8.0\n"
        + lock_environments.DIGEST_PREFIX
        + "deadbeef\n"
    )

    assert lock_environments.recorded_digest(text) is None


def test_an_empty_digest_reads_as_none() -> None:
    text = "# comment\n" + lock_environments.DIGEST_PREFIX + "\n"

    assert lock_environments.recorded_digest(text) is None


# -- parsing pins out of a real lock --


def test_hash_continuations_are_not_read_as_versions() -> None:
    """uv writes a pin and its hashes across continued lines, so a
    naive parser reads the backslash into the version."""
    text = (
        "torch==2.8.0 \\\n"
        "    --hash=sha256:abc \\\n"
        "    --hash=sha256:def\n"
    )

    assert lock_environments.pins_in(text) == {"torch": "2.8.0"}


def test_comments_and_includes_are_not_pins() -> None:
    text = (
        "# environment: test\n"
        "-r requirements.txt\n"
        "\n"
        "torch==2.8.0\n"
    )

    assert lock_environments.pins_in(text) == {"torch": "2.8.0"}


def test_names_are_normalized_the_way_pip_does() -> None:
    """`huggingface_hub` and `huggingface-hub` are one package, and
    the four files spell it both ways."""
    pins = lock_environments.pins_in("huggingface_hub==0.36.2\n")

    assert pins == {"huggingface-hub": "0.36.2"}
