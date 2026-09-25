"""Tests that the documentation agrees with the code it describes.

Strategy: read the registry and the environment manifest, then require
the documents that enumerate models, environments and packages to say
the same thing. Passing proves a reader cannot be told there are two
models, or sent to a package that does not exist, or left unaware of
one that does.

**This is the gap the other documentation tests leave.**
`test_lock_environments.py` checks the manifest against itself and
`test_docs_links.py` checks that a named path resolves, but neither
can notice an *omission*: SmolLM3 and `.venv-ar` shipped, and the
roadmap's quick map went on naming two model backends and two
environments, with every path in it still resolving. META-03 was
raised on exactly that, and its Direction rejects the obvious remedy,
"adding another shipped bullet to every document", because that
process is what caused the drift. So the inventory is derived from the
code and the prose is held to it.

Deliberately not checked here: VRAM figures, which the docs quote as
weights and the registry stores as a fitting threshold, two different
quantities; and any count written as an English word, since "both
models" appears twice in the roadmap meaning the two diffusion models
and a word-matching check would call each a defect.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.backends.registry import REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"
AGENTS = REPO_ROOT / "AGENTS.md"
HANDOFF = REPO_ROOT / "docs" / "HANDOFF.md"

QUICK_MAP = "## Where things live (quick map)"


def _manifest_environments() -> Dict[str, Dict[str, str]]:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    config = tomllib.loads(text)["tool"]["diffusion-llm"]
    return config["environments"]


def _mentions(text: str, token: str) -> bool:
    """Whether `text` names exactly this token, not a longer one.

    A plain `in` test is wrong here and quietly so: `.venv` is a
    prefix of `.venv-ar`, so asserting the core environment is named
    would pass on a document that only mentioned the autoregressive
    one, and renaming `.venv-ar` to `.venv-arm` would pass as well.
    Both were live holes, found by mutating the documents.
    """
    pattern = re.escape(token) + r"(?![A-Za-z0-9_.-])"
    return re.search(pattern, text) is not None


def _section(path: Path, heading: str) -> str:
    """One `##` section of a document, heading to next heading."""
    text = path.read_text(encoding="utf-8")
    assert heading in text, f"{path.name} has no {heading!r}"
    start = text.index(heading)
    rest = text[start + len(heading):]
    match = re.search(r"^## ", rest, re.M)
    finish = start + len(heading) + (
        match.start() if match else len(rest)
    )
    return text[start:finish]


def _readme_model_rows() -> List[Tuple[str, str]]:
    """The model table as (first cell, whole row).

    Structural rather than prose matching, which is what makes the
    device assertions below trustworthy.
    """
    rows: List[Tuple[str, str]] = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0] in ("Model", "---") or set(cells[0]) == {"-"}:
            continue
        rows.append((cells[0], line))
    return rows


def _row_for(display_name: str) -> str:
    for first, row in _readme_model_rows():
        if display_name in first:
            return row
    raise AssertionError(
        f"the README model table has no row for {display_name!r}."
        " A model shipped and the front page does not mention it."
    )


MODEL_IDS = sorted(REGISTRY)


# -- every model the code has, the docs have --


def test_the_registry_is_not_empty() -> None:
    """Guards every parametrized test below, each of which would pass
    on an empty registry while proving nothing."""
    assert len(REGISTRY) >= 3, len(REGISTRY)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_the_readme_table_has_a_row_per_model(model_id: str) -> None:
    assert _row_for(REGISTRY[model_id].display_name)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_the_handoff_names_every_model(model_id: str) -> None:
    """The cold-start page's own model list, which is the one an agent
    reads before touching anything."""
    text = HANDOFF.read_text(encoding="utf-8")
    name = REGISTRY[model_id].display_name

    # The short name, since HANDOFF writes "SmolLM3" in prose.
    stem = name.split("-")[0]
    assert stem in text, f"HANDOFF does not mention {stem}"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_the_quick_map_names_every_worker(model_id: str) -> None:
    """Derived from `worker_module`, so adding a model without
    mentioning it here fails rather than going unnoticed. This is the
    assertion the old map would have failed."""
    section = _section(ROADMAP, QUICK_MAP)
    module = REGISTRY[model_id].worker_module.rsplit(".", 1)[-1]

    assert _mentions(section, f"{module}.py"), (
        f"the quick map does not name {module}.py, the worker for"
        f" {model_id}"
    )


def test_the_model_table_has_no_extra_rows() -> None:
    """The other direction. A row for a model that was removed sends a
    reader to look for something that is not there."""
    documented = {first for first, _ in _readme_model_rows()}
    known = {info.display_name for info in REGISTRY.values()}

    for first in documented:
        assert any(name in first for name in known), (
            f"the README model table row {first!r} matches no"
            " registered model"
        )


# -- and says the truth about where they run --


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_a_cpu_capable_model_is_advertised_as_one(
    model_id: str,
) -> None:
    """The contradiction META-03 cites: the README required a CUDA GPU
    and elsewhere explained that a GPU-less host can run SmolLM3. The
    registry settles which is true."""
    info = REGISTRY[model_id]
    if "cpu" not in info.capabilities.supported_devices:
        pytest.skip(f"{model_id} is GPU-only")

    row = _row_for(info.display_name)

    assert "CPU" in row, (
        f"{info.display_name} runs on CPU per the registry, and its"
        " README row does not say so, which is the claim a reader"
        " without a card needs"
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_a_gpu_only_model_does_not_claim_cpu(model_id: str) -> None:
    """Negative space, and the more damaging direction: a reader who
    believes a diffusion model runs on CPU waits through a load that
    cannot work."""
    info = REGISTRY[model_id]
    if "cpu" in info.capabilities.supported_devices:
        pytest.skip(f"{model_id} supports CPU")

    row = _row_for(info.display_name)

    devices = info.capabilities.supported_devices
    assert "CPU" not in row, (
        f"{info.display_name} declares {devices} in the registry,"
        " and its README row mentions CPU"
    )


def test_at_least_one_model_runs_without_a_card() -> None:
    """Both tests above skip, so one of them proving nothing would be
    invisible. This is the fact they hang on."""
    runs_on_cpu = [
        model_id for model_id, info in REGISTRY.items()
        if "cpu" in info.capabilities.supported_devices
    ]

    assert runs_on_cpu, (
        "no model declares CPU support, so the GPU-less path the"
        " README advertises does not exist"
    )


# -- every environment the manifest declares, the docs declare --


def _environment_path(name: str) -> str:
    """The directory an environment installs into.

    An overlay has no directory of its own: `desktop` installs on top
    of `core`, which is the whole reason it is expressed as `extends`
    rather than as a fourth interpreter. Resolving to the parent keeps
    every caller below from special-casing it.
    """
    environment = _manifest_environments()[name]
    if "path" in environment:
        return environment["path"]
    return _environment_path(environment["extends"])


@pytest.mark.parametrize("name", sorted(_manifest_environments()))
def test_the_quick_map_names_every_environment(name: str) -> None:
    """`.venv-ar` existed for weeks while this section said there were
    two environments."""
    section = _section(ROADMAP, QUICK_MAP)

    assert _mentions(section, _environment_path(name)), (
        f"the quick map does not name {_environment_path(name)},"
        f" the {name} environment"
    )


def test_the_quick_map_says_the_overlay_is_an_overlay() -> None:
    """An overlay listed beside the three real environments reads as a
    fourth interpreter to create, and `pip install -r
    requirements-desktop.txt` into a fresh venv gets a desktop shell
    with no server behind it.

    Scoped to the paragraph that names the overlay's lock rather than
    to the whole section, because the section also discusses
    `overlays.js` and a search for "overlay" matched that instead.
    Found by deleting the explanation and watching this pass.
    """
    overlays = {
        name: environment
        for name, environment in _manifest_environments().items()
        if "extends" in environment
    }
    assert overlays, "no overlay in the manifest to describe"

    section = _section(ROADMAP, QUICK_MAP)
    for name, environment in overlays.items():
        lock = environment["lock"]
        paragraph = next(
            (block for block in section.split("\n\n")
             if lock in block),
            "",
        )
        assert paragraph, f"the quick map does not name {lock}"

        assert "overlay" in paragraph.lower(), (
            f"the quick map names {lock} without saying the {name}"
            f" environment installs into {environment['extends']}"
            " rather than being one of its own"
        )


@pytest.mark.parametrize("name", sorted(_manifest_environments()))
def test_the_quick_map_names_every_lock(name: str) -> None:
    section = _section(ROADMAP, QUICK_MAP)
    lock = _manifest_environments()[name]["lock"]

    assert _mentions(section, lock), (
        f"the quick map does not name {lock}, the {name} lock"
    )


@pytest.mark.parametrize("name", sorted(_manifest_environments()))
def test_the_agent_contract_names_every_environment(
    name: str,
) -> None:
    """AGENTS.md is where an agent learns which interpreter to use, so
    an environment missing from it gets the wrong `transformers`."""
    text = AGENTS.read_text(encoding="utf-8")
    path = _environment_path(name)

    assert _mentions(text, path), (
        f"AGENTS.md does not name {path}"
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_each_model_names_an_environment_that_exists(
    model_id: str,
) -> None:
    """Joins the two inventories, so a registry entry cannot point at
    an environment the manifest never declared."""
    environments = _manifest_environments()

    assert REGISTRY[model_id].environment in environments


# -- and every package, so a new area cannot appear unmentioned --


def _packages() -> List[str]:
    return sorted(
        path.name for path in (REPO_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def test_there_are_packages_to_check() -> None:
    assert _packages(), "no packages found under src/"


@pytest.mark.parametrize("package", _packages())
def test_the_quick_map_names_every_package(package: str) -> None:
    """Packages rather than files, deliberately. Asserting every
    module would fail on each new file and turn the map into `ls`
    output; a new package is an architectural change, and those are
    what a map is for."""
    section = _section(ROADMAP, QUICK_MAP)

    assert f"src/{package}/" in section, (
        f"the quick map does not mention src/{package}/"
    )


# -- the matcher the assertions above rely on --


def test_a_name_matches_itself() -> None:
    assert _mentions("install into .venv-ar today", ".venv-ar")
    assert _mentions("see requirements.txt", "requirements.txt")


def test_a_longer_name_is_not_a_match() -> None:
    """The bug this exists for, in both directions. `.venv` is a
    prefix of `.venv-ar`, so a plain `in` test let a document that
    mentioned only the autoregressive environment satisfy an
    assertion about the core one, and let a rename to `.venv-arm`
    pass unnoticed. Both were found by mutating the documents rather
    than by reading the test."""
    assert not _mentions("only .venv-ar is here", ".venv")
    assert not _mentions("we renamed it .venv-arm", ".venv-ar")
    assert not _mentions("requirements.txt.bak", "requirements.txt")
    assert not _mentions("smollm3_worker.python", "smollm3_worker.py")


def test_punctuation_after_a_name_still_matches() -> None:
    """The matcher has to allow the prose it is used on: a path at the
    end of a sentence, in a list, or inside backticks."""
    assert _mentions("it lives in `.venv-ar`.", ".venv-ar")
    assert _mentions("`.venv`, `.venv-ar` and more", ".venv")
    assert _mentions("run_worker.py takes --device", "run_worker.py")


def test_the_map_does_not_claim_the_locks_are_hand_written() -> None:
    """It described them as "flat, fully-pinned freezes" for months
    after DEPS-01 made them generated, while the section directly
    below it explained the generator."""
    section = _section(ROADMAP, QUICK_MAP).lower()

    assert "freeze" not in section, (
        "the quick map still calls the locks freezes; they are"
        " generated by scripts/lock_environments.py from the manifest"
    )
