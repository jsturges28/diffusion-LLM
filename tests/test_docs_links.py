"""Tests that tracked docs never point at something a clone lacks.

Strategy: read every tracked markdown file, pull out the repository
paths it references, and require each one to exist and to be tracked
by git. Asking git rather than the filesystem is the whole point: a
path can be perfectly present on the maintainer's machine and absent
from every clone, which is exactly the failure that produced this
finding.

`AGENTS.md` told agents to follow `.cursor/rules/` while `.gitignore`
excluded all of `.cursor`, and `ROADMAP.md` cited `.cursor/plans/` as
the canonical build history for three milestones. A contributor could
obey every tracked instruction and still never see the rules, and the
coding standard the contract named did not exist in the repository at
all. None of that was visible from inside a configured checkout,
which is why it survived so long and why the check has to be
automated rather than remembered.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Markdown links, plus the backtick-quoted paths this repo's prose
# uses far more often than it uses links.
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_.][A-Za-z0-9_./-]*)`")

# Prose that looks like a path but is not one of ours.
_IGNORED_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "#",
    "~/",
    "/",
)

# Documents that are records rather than claims, and are not checked.
# A build plan describes what was true when it was written, and a file
# it named may since have been renamed or deleted; holding history to
# the present tense would just mean never writing history down.
_UNCHECKED_DOC_PREFIXES = (
    ".cursor/plans/",
    "archive/",
    "src/web/static/vendor/",
)

# Trees that are deliberately absent from a clone, matched by path
# prefix so a new mention of `.venv/bin/ruff` does not need a new
# entry. Each needs a reason, because the point of this test is that
# "it works on my machine" is not one.
_ALLOWED_ABSENT_PREFIXES = (
    # Local editor convenience. AGENTS.md says out loud that these are
    # optional and that nothing may depend on them.
    ".cursor/rules",
    # Built during setup, one per model environment. Docs name their
    # interpreters constantly and should keep doing so. All three are
    # listed because the match is segment-aware, so `.venv` on its own
    # would not cover the siblings.
    ".venv",
    ".venv-dgemma",
    ".venv-ar",
    # Created at runtime and holds the user's own saved runs.
    "results",
    # Historical material kept locally, ignored long before this test.
    "archive",
    "transcripts",
    "data",
)


def _tracked_files() -> Set[str]:
    """Every path git knows about, as posix strings."""
    listing = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = {line for line in listing.splitlines() if line}
    assert len(tracked) > 0, "git reported no tracked files"
    return tracked


def _tracked_markdown() -> List[Path]:
    docs = sorted(
        REPO_ROOT / name
        for name in _tracked_files()
        if name.endswith(".md")
        and not name.startswith(_UNCHECKED_DOC_PREFIXES)
    )
    assert len(docs) > 0, "no tracked markdown to check"
    return docs


def _top_level_names(tracked: Set[str]) -> Set[str]:
    """The first path segment of everything git tracks."""
    return {name.split("/", 1)[0] for name in tracked}


def _candidate_paths(text: str, tracked: Set[str]) -> Set[str]:
    """Repository paths a document claims exist.

    Two signals, and the second is the one that matters. A path
    counts as a claim if its first segment is something git tracks at
    the root, **or** if it exists on this filesystem. The second is
    what catches this finding's actual shape: `.cursor/rules/` was
    present on the maintainer's machine and in no clone, so anchoring
    only to tracked names would have skipped the very reference that
    was broken.

    Both are needed because this repo's prose is full of fragments
    that look like paths and are not, from ``vendor/README.md``
    written relative to the directory under discussion to
    ``backends/`` naming a subdirectory in passing. Neither exists at
    the root nor is tracked there, so neither is mistaken for a claim.
    """
    roots = _top_level_names(tracked)
    found: Set[str] = set()

    # A markdown link is unambiguous: somebody wrote it expecting it
    # to resolve. Checked strictly, with no existence heuristic, so a
    # link to a file that never existed is caught rather than assumed
    # to be prose. This is the half that guards a documentation move.
    for match in _MARKDOWN_LINK.findall(text):
        target = match.split("#", 1)[0].strip()
        if target == "" or target.startswith(_IGNORED_PREFIXES):
            continue
        if _is_allowed_absent(target):
            continue
        found.add(target)

    # A backtick span may or may not be a path, so it gets the
    # heuristic described above.
    for match in _BACKTICK_PATH.findall(text):
        candidate = match.strip()
        if _is_repo_path(candidate, roots):
            found.add(candidate)

    return found


def _is_allowed_absent(path: str) -> bool:
    """Whether a clone is expected not to have this, by design."""
    cleaned = path.rstrip("/")
    return any(
        cleaned == prefix or cleaned.startswith(prefix + "/")
        for prefix in _ALLOWED_ABSENT_PREFIXES
    )


def _is_repo_path(path: str, roots: Set[str]) -> bool:
    if path == "":
        return False
    if path.startswith(_IGNORED_PREFIXES):
        return False
    if _is_allowed_absent(path):
        return False
    if path.split("/", 1)[0] in roots:
        return True
    return (REPO_ROOT / path.rstrip("/")).exists()


def _is_satisfied(path: str, tracked: Set[str]) -> bool:
    """Whether git can produce this path for a fresh clone."""
    cleaned = path.rstrip("/")
    if cleaned in tracked:
        return True
    # A directory is satisfied when anything tracked lives under it.
    prefix = cleaned + "/"
    return any(name.startswith(prefix) for name in tracked)


@pytest.mark.parametrize(
    "doc", _tracked_markdown(), ids=lambda p: p.name
)
def test_every_referenced_path_reaches_a_clone(doc: Path) -> None:
    tracked = _tracked_files()
    text = doc.read_text(encoding="utf-8")

    missing = sorted(
        path
        for path in _candidate_paths(text, tracked)
        if not _is_satisfied(path, tracked)
    )

    assert missing == [], (
        f"{doc.name} references paths no clone would have: {missing}."
        " Track them, or stop pointing at them."
    )


def test_the_coding_standard_is_in_the_repository() -> None:
    """The gap this finding was really about.

    `AGENTS.md` named TigerStyle as "the repo's" standard while the
    text of it lived only in one maintainer's editor settings, so a
    clone got the name and none of the rules.
    """
    tracked = _tracked_files()

    assert "TIGERSTYLE.md" in tracked


def test_the_build_plans_reach_a_clone() -> None:
    """ROADMAP.md cites `.cursor/plans/` as the canonical build
    history in three places. That is only true if it is tracked."""
    tracked = _tracked_files()

    plans = [
        name
        for name in tracked
        if name.startswith(".cursor/plans/")
    ]

    assert len(plans) > 0, (
        "ROADMAP.md points at .cursor/plans/ but nothing there is"
        " tracked"
    )


def test_the_local_rules_stay_out_of_the_repository() -> None:
    """The other half of the decision, asserted so it does not drift
    back. Editor rules are local convenience; the contract is
    AGENTS.md and TIGERSTYLE.md, and tracking the .mdc files would
    quietly re-create two sources of truth."""
    tracked = _tracked_files()

    assert not [
        name
        for name in tracked
        if name.startswith(".cursor/rules/")
    ]


def test_the_guard_notices_a_path_that_is_not_tracked() -> None:
    """Negative space: the check has to fail on the thing it exists
    for, or it is decoration."""
    tracked = _tracked_files()

    assert not _is_satisfied("docs/does_not_exist.md", tracked)
    assert _is_satisfied("AGENTS.md", tracked)


def test_the_guard_catches_present_locally_but_absent_from_git(
) -> None:
    """The finding's actual shape, checked against a live example.

    `.cursor/rules/` is on this filesystem and in no clone, which is
    exactly the condition that let `AGENTS.md` point at rules a
    contributor could never read. It is allowed by name in
    `_ALLOWED_ABSENT` because AGENTS.md now says out loud that those
    rules are optional local convenience, but the detection has to
    work or the allowance would be meaningless.
    """
    tracked = _tracked_files()
    local_only = ".cursor/rules"

    if not (REPO_ROOT / local_only).exists():
        pytest.skip("no local Cursor rules on this machine")

    # The detection, checked without routing through the allowance:
    # the directory is right there, and git would not hand it over.
    assert not _is_satisfied(local_only, tracked)
    # And the allowance is a deliberate, documented exception rather
    # than the check simply failing to notice.
    assert _is_allowed_absent(local_only)


def test_a_directory_counts_as_present_when_it_has_content() -> None:
    tracked = _tracked_files()

    assert _is_satisfied("src/web/", tracked)
    assert _is_satisfied("scripts", tracked)
