"""A model names an environment, and the manifest says where it is.

Strategy: read the real manifest and the real registry, since the
point of this change is that there is now exactly one place the
answer comes from and a test against a fake one would not check that.
Then the failure paths, which are what a caller has to handle.

What passing proves: no model can reference an environment that does
not exist, and the interpreter the supervisor launches is derived from
the same table the locks are generated from. Before this, the path
lived in the registry as a string, which made it a fourth independent
spelling of each environment alongside its lock, the setup
instructions and the agent conventions. Four copies of one fact is
three chances to be wrong, and nothing compared them.

The overlay case is here because it is the one that is not obvious.
The desktop environment has no interpreter of its own; its packages
install into core's. A resolver that answered with a desktop-specific
path would be inventing a directory nobody creates.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.backends.environments import (
    MANIFEST_PATH,
    UnknownEnvironmentError,
    environment_names,
    interpreter_for,
    lock_for,
)
from src.backends.registry import REGISTRY


def _manifest_environments() -> dict:
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    config = tomllib.loads(text)["tool"]["diffusion-llm"]
    return config["environments"]


# -- the registry and the manifest agree --


def test_every_model_names_a_declared_environment() -> None:
    """The registry asserts this at import, so reaching this test at
    all means it held. Stated anyway, because an assertion inside a
    module is invisible in a failure report and this is the property
    the whole change exists to guarantee."""
    declared = environment_names()

    assert declared, "the manifest declares no environments"
    for model in REGISTRY.values():
        assert model.environment in declared, (
            f"{model.id} runs in {model.environment!r}"
        )


def test_the_three_models_are_in_three_environments() -> None:
    """Pinned because it is the reason this project has more than one
    environment at all: the models need incompatible transformers
    versions, so sharing one would not be a simplification."""
    used = {model.environment for model in REGISTRY.values()}

    assert used == {"core", "dgemma", "ar"}


# -- resolving an interpreter --


@pytest.mark.parametrize(
    "name,expected",
    [
        ("core", Path(".venv/bin/python")),
        ("dgemma", Path(".venv-dgemma/bin/python")),
        ("ar", Path(".venv-ar/bin/python")),
    ],
)
def test_an_environment_resolves_to_its_interpreter(
    name: str, expected: Path
) -> None:
    assert interpreter_for(name) == expected


def test_the_path_is_relative_to_the_repository() -> None:
    """The supervisor puts this in a refusal message, and an absolute
    path there would name a directory the reader has to strip back to
    something they can type."""
    assert not interpreter_for("core").is_absolute()


def test_an_overlay_resolves_to_the_environment_it_extends() -> None:
    """The desktop packages install into core, so core's interpreter
    is the honest answer. Returning a desktop-specific path would
    name a directory that is never created."""
    assert interpreter_for("desktop") == interpreter_for("core")


def test_an_unknown_environment_is_refused_by_name() -> None:
    """Its own exception type, because the supervisor turns this into
    a refusal a user can act on. A bare KeyError would read like an
    internal slip rather than a registry naming something absent."""
    with pytest.raises(UnknownEnvironmentError) as caught:
        interpreter_for("ssm")

    message = str(caught.value)
    assert "ssm" in message
    # Lists what does exist, since the likeliest cause is a typo.
    assert "core" in message


def test_an_empty_name_is_a_programmer_error() -> None:
    """Negative space. An empty string is not a missing environment,
    it is a caller that lost track of what it was asking about."""
    with pytest.raises(AssertionError):
        interpreter_for("")


# -- the lock that installs it --


def test_every_environment_reports_its_lock() -> None:
    for name, entry in _manifest_environments().items():
        assert lock_for(name) == entry["lock"]


def test_an_unknown_environment_has_no_lock() -> None:
    with pytest.raises(UnknownEnvironmentError):
        lock_for("ssm")


# -- the manifest is the only place a path appears --


def test_no_interpreter_path_is_hardcoded_in_the_registry() -> None:
    """The finding's actual complaint, as a test.

    `DEPS-01` is about a fifth environment adding another place to
    update. If a path creeps back into the registry, the manifest
    stops being authoritative and nothing else here would notice.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "backends"
        / "registry.py"
    ).read_text(encoding="utf-8")

    assert "/bin/python" not in source
    assert "venv_python" not in source
