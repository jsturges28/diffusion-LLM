"""Tests that tracked docs never point at something a clone lacks.

Strategy: read every tracked markdown file, pull out the repository
paths it references, and require each one to exist and to be tracked
by git. Asking git rather than the filesystem is the whole point: a
path can be perfectly present on the maintainer's machine and absent
from every clone, which is exactly the failure that produced this
finding.

`AGENTS.md` told agents to follow `.cursor/rules/` while `.gitignore`
excluded all of `.cursor`, and the roadmap cited `.cursor/plans/` as
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

# Shell-style shorthand for sibling files, as in
# `src/backends/{llada_worker,dgemma_worker}.py`. The pattern above
# cannot match a brace, so it extracted *nothing at all* from these
# spans, and the roadmap's quick map was written entirely in them.
# That is how it came to name two model backends while three shipped:
# every path in it was invisible to this test.
_BRACE_SHORTHAND = re.compile(
    r"`([A-Za-z0-9_./-]+)\{([^}]*)\}([A-Za-z0-9_.]*)`"
)

# A bare module name, which the heuristic below deliberately skips.
# Run-folder artifacts are named bare all through these docs
# (`metadata.json`, `frames.jsonl`) and live in a gitignored tree, so
# skipping bare names is right. But none of those is a `.py`, and
# `docs/HANDOFF.md` named `llada_sampler.py` for weeks after `ORG-03`
# renamed it, because nothing looked.
_BARE_MODULE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*\.py)`")

# Documents that describe something other than this tree as it stands,
# exempt from the bare-module rule for the reason the unchecked list
# above gives: holding a record to the present tense means never
# writing one. The campaign ledger cites files by the name they had
# when a finding was raised, and `reference/` describes the upstream
# project it was lifted from, whose `generate.py` was never ours.
_RECORD_DOC_PREFIXES = (
    "docs/audit/",
    "reference/",
)

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
#
# The audit reports are here for a stronger reason: each campaign's
# brief declares its report immutable, so a test that demanded edits
# to one would be asking for a rule to be broken. Their file citations
# are part of the record of what was believed on the day each was
# written, including modules a finding proposes and nobody has created
# yet.
_UNCHECKED_DOC_PREFIXES = (
    ".cursor/plans/",
    "archive/",
    "docs/audit/AUDIT_REPORT",
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


def _resolve_link(doc: Path, target: str) -> str:
    """A link target as a path from the repository root.

    Markdown links resolve against the directory holding the
    document, so `../AGENTS.md` written in `docs/` means something
    different from the same string written at the root. Normalizing
    here is what lets the rest of the check treat every path the same
    way, and it is the part that has to be right for a documentation
    move to be verifiable.
    """
    if target.startswith("/"):
        return target.lstrip("/")
    combined = (doc.parent / target).resolve()
    try:
        return combined.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Points outside the repository, which is never satisfiable.
        return target


def _candidate_paths(
    doc: Path, text: str, tracked: Set[str]
) -> Set[str]:
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
        resolved = _resolve_link(doc, target)
        if _is_allowed_absent(resolved):
            continue
        found.add(resolved)

    # A backtick span may or may not be a path, so it gets the
    # heuristic described above.
    for match in _BACKTICK_PATH.findall(text):
        candidate = match.strip()
        if _is_repo_path(candidate, roots):
            found.add(candidate)

    found |= _shorthand_paths(text, roots)

    # A bare module name, resolved by name rather than by location,
    # since the prose says `run_worker.py` without repeating the
    # directory it was just given in.
    if not _is_record(doc):
        found |= _unresolved_modules(text, tracked)

    return found


def _shorthand_paths(text: str, roots: Set[str]) -> Set[str]:
    """Every member of every sibling-shorthand span.

    A comma is required, which is what keeps route templates out:
    `/runs/{id}/metadata` has one member and names a route, not a
    file. The prefix must be a tracked root for the same reason the
    heuristic above anchors there.
    """
    found: Set[str] = set()
    for prefix, group, suffix in _BRACE_SHORTHAND.findall(text):
        if "," not in group:
            continue
        if prefix.split("/", 1)[0] not in roots:
            continue
        for path in _expand_shorthand(prefix, group, suffix):
            if not _is_allowed_absent(path):
                found.add(path)
    return found


def _unresolved_modules(text: str, tracked: Set[str]) -> Set[str]:
    """Bare module names this tree has none of.

    Reported through the same channel as every other claim, so the
    failure message names the file rather than the rule.
    """
    return {
        f"a module named {name}"
        for name in _BARE_MODULE.findall(text)
        if not _module_exists(name, tracked)
    }


def _is_record(doc: Path) -> bool:
    """Whether this document describes the past or another project."""
    relative = doc.relative_to(REPO_ROOT).as_posix()
    return relative.startswith(_RECORD_DOC_PREFIXES)


def _module_exists(name: str, tracked: Set[str]) -> bool:
    """Whether a module of this name is tracked anywhere.

    By name across the whole tree rather than under a fixed list of
    roots, because `main.py` and `desktop.py` are entry points at the
    repository root and the prose names them constantly.
    """
    return any(
        path.rsplit("/", 1)[-1] == name for path in tracked
    )


def _expand_shorthand(
    prefix: str, group: str, suffix: str
) -> List[str]:
    """`dir/{a,b}.py` as the paths a reader takes it to mean.

    Three shapes are in use and all have to work: the extension after
    the group (`{a,b}.py`), the whole name inside it
    (`{a.html,b.js}`), and neither (`{a,b,c}`, which still means
    modules). The last is why a name with no dot gains `.py` rather
    than being checked as a directory.
    """
    expanded: List[str] = []
    for part in group.split(","):
        part = part.strip()
        if part == "":
            continue
        path = f"{prefix}{part}{suffix}"
        if "." not in path.rsplit("/", 1)[-1]:
            path += ".py"
        expanded.append(path)
    return expanded


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
    if "/" not in path.rstrip("/"):
        # A bare filename is a name, not a location. The docs are
        # full of them (`metadata.json`, `frames.jsonl`) naming files
        # inside a run folder, which is gitignored and so is in no
        # clone by design. Tracked root files still reach the check
        # through `roots` above.
        #
        # Without this, the heuristic below made the suite depend on
        # a clean working tree: a stray metadata.json dropped at the
        # root during manual verification failed four documents at
        # once, for a reference that was correct.
        return False
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
        for path in _candidate_paths(doc, text, tracked)
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

    assert "docs/TIGERSTYLE.md" in tracked


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


# -- the shorthand that used to hide a whole section --


def test_shorthand_expands_with_the_extension_outside() -> None:
    found = _expand_shorthand("src/backends/", "a,b", ".py")

    assert found == ["src/backends/a.py", "src/backends/b.py"]


def test_shorthand_expands_with_whole_names_inside() -> None:
    found = _expand_shorthand("src/web/static/", "a.html,b.js", "")

    assert found == ["src/web/static/a.html", "src/web/static/b.js"]


def test_shorthand_with_no_extension_means_modules() -> None:
    """`src/inference/{streaming_sampler,ar_sampler}` in HANDOFF.
    Checking those as directories would pass on nothing existing."""
    found = _expand_shorthand("src/inference/", "a,b", "")

    assert found == ["src/inference/a.py", "src/inference/b.py"]


def test_shorthand_in_a_real_document_is_checked() -> None:
    """Against the live example, because the unit tests above would
    pass on an expander nothing calls.

    `docs/HANDOFF.md` lists the three workers as shorthand and has to
    keep doing so: it sits at exactly its 200-line budget, so writing
    them out is not available. The roadmap's quick map used the same
    notation, which is how it came to name two of the three workers
    while every path in it stayed invisible to this test.
    """
    handoff = REPO_ROOT / "docs" / "HANDOFF.md"
    tracked = _tracked_files()

    found = _candidate_paths(
        handoff, handoff.read_text(encoding="utf-8"), tracked
    )

    assert "src/backends/smollm3_worker.py" in found, (
        "shorthand paths are not reaching the check"
    )


def test_a_route_template_is_not_mistaken_for_a_path() -> None:
    """`/runs/{id}/metadata` names a route. One member and no comma is
    the discriminator, and it has to hold or the check invents files
    called `id`."""
    text = "see `/api/analytics/runs/{id}/frames` for the payload"
    tracked = _tracked_files()
    doc = REPO_ROOT / "docs" / "ROADMAP.md"

    assert _candidate_paths(doc, text, tracked) == set()


# -- and the bare name nothing was looking at --


def test_a_bare_module_resolves_by_name() -> None:
    tracked = _tracked_files()

    assert _module_exists("server.py", tracked)
    # At the repository root, which is why the search is not confined
    # to a list of package directories.
    assert _module_exists("main.py", tracked)


def test_a_bare_module_that_was_renamed_is_caught() -> None:
    """The live example. `ORG-03` renamed `llada_sampler.py` to
    `llada_kernel.py` and moved its twin to `reference/llada/`, and
    HANDOFF went on naming the old one, because a bare filename was
    skipped as prose."""
    tracked = _tracked_files()

    assert _module_exists("llada_kernel.py", tracked)
    assert not _module_exists("llada_sampler.py", tracked)


def test_a_record_may_name_a_file_that_has_since_moved() -> None:
    """The exemption, asserted rather than assumed. The campaign
    ledger cites files by the name they had when a finding was
    raised."""
    assert _is_record(REPO_ROOT / "docs" / "audit" / "x.md")
    assert _is_record(REPO_ROOT / "reference" / "llada" / "README.md")
    assert not _is_record(REPO_ROOT / "docs" / "HANDOFF.md")
