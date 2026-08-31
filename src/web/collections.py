"""Bounded operations over the analytics run collections.

A collection is a name and a list of run ids the user filed into it by
hand. Unlike every other durable UI value, it cannot be recomputed
from anything on disk, which is why ``DATA-02`` singled it out.

This module exists because the client used to own the list. It read
the whole array, mutated it in the window, and wrote the whole array
back, so two windows hydrated from the same value each filed a
different run and the later write silently dropped the earlier one.
The fix is the one ``run_store.save`` already made for run identity:
stop letting the client say what the new state is, and let it say only
what it wants to change.

So each function here is one user gesture, applied to whatever the
stored list happens to be at the moment it runs. They are pure
transforms, which is what lets ``mutate_ui_state_key`` run them with
the file lock held and makes them testable without a server.

Every limit that used to live in the browser lives here now, because a
limit the client enforces is a limit that holds only for clients that
choose to.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# The most collections a user may keep. Chosen for the tab strip
# rather than for storage: past a couple of dozen the tabs stop being
# scannable, which is the point of them.
COLLECTIONS_MAX = 24

# Name length, matching the input's maxlength so a name that fits the
# field cannot be refused by the server.
NAME_MAX = 40

# Runs in one collection. The aggregate is bounded separately by the
# ui-state value size, which is the limit that will actually bite
# first; this one exists so a single runaway collection fails with a
# reason rather than by overflowing the file.
RUNS_PER_COLLECTION_MAX = 1024

# The one collection the app creates on the user's behalf, on the
# first star. Fixed id so the star can find it without a lookup by
# name, which the user is free to change.
FAVORITES_ID = "favorites"
FAVORITES_NAME = "Favorites"

# Reasons, so a caller can map to a status and the browser can say
# something specific. Strings rather than an enum because they cross
# to JSON and are read in test assertions.
REASON_LIMIT = "collection_limit"
REASON_RUNS_LIMIT = "collection_runs_limit"
REASON_NAME = "invalid_name"
REASON_UNKNOWN = "unknown_collection"
REASON_UNKNOWN_RUN = "unknown_run"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_SLUG_EDGES = re.compile(r"^-+|-+$")


class CollectionError(Exception):
    """A client asked for something the contract does not allow.

    An operating error rather than a programmer error: bad input from
    the browser, which the endpoint turns into a 4xx with the reason
    attached. Internal invariants use assertions instead.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def decode(raw: Optional[str]) -> List[Dict[str, Any]]:
    """The stored value as a list, or empty when absent or corrupt.

    Total, because a page load must not fail on a bad file. A corrupt
    value reads as no collections, and the next write replaces it.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    result: List[Dict[str, Any]] = []
    for entry in parsed:
        clean = _sanitize(entry)
        if clean is not None:
            result.append(clean)
    return result


def encode(collections: List[Dict[str, Any]]) -> str:
    return json.dumps(collections, ensure_ascii=False)


def _sanitize(entry: Any) -> Optional[Dict[str, Any]]:
    """One stored entry, normalized, or None if it is not one."""
    if not isinstance(entry, dict):
        return None
    identifier = entry.get("id")
    if not isinstance(identifier, str) or identifier == "":
        return None
    name = entry.get("name")
    if not isinstance(name, str) or name == "":
        name = identifier
    runs = [
        run_id
        for run_id in entry.get("runs", [])
        if isinstance(run_id, str) and run_id != ""
    ]
    return {"id": identifier, "name": name, "runs": runs}


def find(
    collections: List[Dict[str, Any]], collection_id: str
) -> Optional[Dict[str, Any]]:
    for collection in collections:
        if collection["id"] == collection_id:
            return collection
    return None


# -- the gestures --


def create(
    collections: List[Dict[str, Any]], name: str
) -> List[Dict[str, Any]]:
    """Add an empty collection under a generated id."""
    clean = _clean_name(name)
    _require_room(collections)
    identifier = _new_id(collections, clean)
    entry = {"id": identifier, "name": clean, "runs": []}
    return collections + [entry]


def rename(
    collections: List[Dict[str, Any]],
    collection_id: str,
    name: str,
) -> List[Dict[str, Any]]:
    """Change a collection's display name, never its id.

    The id is what memberships are keyed by, so regenerating it from
    the new name would orphan every run already filed.
    """
    clean = _clean_name(name)
    _require_present(collections, collection_id)
    return [
        {**entry, "name": clean}
        if entry["id"] == collection_id
        else entry
        for entry in collections
    ]


def delete(
    collections: List[Dict[str, Any]], collection_id: str
) -> List[Dict[str, Any]]:
    _require_present(collections, collection_id)
    return [
        entry for entry in collections
        if entry["id"] != collection_id
    ]


def add_run(
    collections: List[Dict[str, Any]],
    collection_id: str,
    run_id: str,
    existing: Set[str],
) -> List[Dict[str, Any]]:
    """File one run into a collection. Idempotent."""
    return add_runs(collections, collection_id, [run_id], existing)


def add_runs(
    collections: List[Dict[str, Any]],
    collection_id: str,
    run_ids: List[str],
    existing: Set[str],
) -> List[Dict[str, Any]]:
    """File several runs into one collection, or none of them.

    Idempotent per run, so filing a selection that partly overlaps
    the collection adds only the rest.

    All or nothing on refusal, which is the reason this is one
    function rather than a loop at the call site. Filing six runs as
    six requests can stop at four and leave the user looking at a
    half-applied gesture with no way to tell which half; validating
    the batch up front means the collection either gains all of them
    or is untouched.

    Refuses a run with nothing on disk. The hydrate reconcile would
    prune it later anyway, so accepting it would store a value whose
    only future is to be removed.
    """
    collection = _require_present(collections, collection_id)
    for run_id in run_ids:
        _require_run(run_id, existing)

    filed = collection["runs"]
    # Dedupe against the collection and against the batch itself,
    # keeping the order the caller asked for.
    fresh: List[str] = []
    for run_id in run_ids:
        if run_id not in filed and run_id not in fresh:
            fresh.append(run_id)
    if not fresh:
        return collections
    if len(filed) + len(fresh) > RUNS_PER_COLLECTION_MAX:
        raise CollectionError(
            REASON_RUNS_LIMIT,
            f"a collection holds at most "
            f"{RUNS_PER_COLLECTION_MAX} runs",
        )
    return [
        {**entry, "runs": entry["runs"] + fresh}
        if entry["id"] == collection_id
        else entry
        for entry in collections
    ]


def ensure_favorites(
    collections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """The list with Favorites in it, created if this is the first.

    Split out of ``toggle_favorite`` so the bulk path can compose it
    with ``add_runs`` under one lock, rather than needing the star's
    clear-everything half.
    """
    if find(collections, FAVORITES_ID) is not None:
        return collections
    _require_room(collections)
    favorites = {
        "id": FAVORITES_ID,
        "name": FAVORITES_NAME,
        "runs": [],
    }
    # First, so the tab it creates lands where a user expects a
    # default to be rather than after their own collections.
    return [favorites] + collections


def remove_run(
    collections: List[Dict[str, Any]],
    collection_id: str,
    run_id: str,
) -> List[Dict[str, Any]]:
    """Unfile a run. Idempotent, and it does not check the run exists.

    Deliberately unlike ``add_run``: removing a run whose folder is
    gone is exactly what a user cleaning up would want to do, and
    refusing it would leave the id stuck until the next reconcile.
    """
    _require_present(collections, collection_id)
    return [
        {**entry, "runs": _without(entry["runs"], run_id)}
        if entry["id"] == collection_id
        else entry
        for entry in collections
    ]


def toggle_favorite(
    collections: List[Dict[str, Any]],
    run_id: str,
    existing: Set[str],
) -> List[Dict[str, Any]]:
    """The star, which is two gestures wearing one button.

    A run in no collection is filed into Favorites, created if this is
    the first star. A run in any collection is removed from all of
    them, because the star reads as "collected" rather than as
    "in Favorites" and clicking a filled one must empty it.

    One operation rather than several because the second half is a
    bulk change: split across requests it can half-apply, which is the
    failure this module exists to remove.
    """
    if _is_collected(collections, run_id):
        return [
            {**entry, "runs": _without(entry["runs"], run_id)}
            for entry in collections
        ]
    _require_run(run_id, existing)
    return add_run(
        ensure_favorites(collections), FAVORITES_ID, run_id, existing
    )


# -- pruning, shared with the hydrate reconcile --


def prune_missing(
    collections: List[Dict[str, Any]], existing: Set[str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop ids whose run is gone, and say how many went.

    An emptied collection is kept: the user made it, and filing
    nothing into it yet is not the same as not wanting it.
    """
    dropped = 0
    kept: List[Dict[str, Any]] = []
    for entry in collections:
        runs = [r for r in entry["runs"] if r in existing]
        dropped += len(entry["runs"]) - len(runs)
        kept.append({**entry, "runs": runs})
    return kept, dropped


# -- guards --


def _clean_name(name: Any) -> str:
    if not isinstance(name, str):
        raise CollectionError(REASON_NAME, "a name must be text")
    clean = name.strip()
    if clean == "":
        raise CollectionError(REASON_NAME, "a name cannot be empty")
    if len(clean) > NAME_MAX:
        raise CollectionError(
            REASON_NAME,
            f"a name is at most {NAME_MAX} characters",
        )
    return clean


def _require_room(collections: List[Dict[str, Any]]) -> None:
    if len(collections) >= COLLECTIONS_MAX:
        raise CollectionError(
            REASON_LIMIT,
            f"at most {COLLECTIONS_MAX} collections",
        )


def _require_present(
    collections: List[Dict[str, Any]], collection_id: str
) -> Dict[str, Any]:
    collection = find(collections, collection_id)
    if collection is None:
        raise CollectionError(
            REASON_UNKNOWN,
            f"no collection named {collection_id}",
        )
    return collection


def _require_run(run_id: Any, existing: Set[str]) -> None:
    if not isinstance(run_id, str) or run_id == "":
        raise CollectionError(
            REASON_UNKNOWN_RUN, "a run id must be text"
        )
    if run_id not in existing:
        raise CollectionError(
            REASON_UNKNOWN_RUN, f"no run named {run_id}"
        )


def _is_collected(
    collections: List[Dict[str, Any]], run_id: str
) -> bool:
    return any(run_id in entry["runs"] for entry in collections)


def _without(runs: List[str], run_id: str) -> List[str]:
    return [r for r in runs if r != run_id]


def _new_id(
    collections: List[Dict[str, Any]], name: str
) -> str:
    """A slug from the name, suffixed until it is unused.

    Bounded by the collection cap plus a margin: with unique ids and
    the cap holding, the loop cannot run that far, and a limit on
    every loop is the rule rather than a response to a known case.
    """
    base = _SLUG_EDGES.sub("", _SLUG_STRIP.sub("-", name.lower()))
    if base == "":
        base = "collection"
    identifier = base
    suffix = 2
    while find(collections, identifier) is not None:
        identifier = f"{base}-{suffix}"
        suffix += 1
        assert suffix <= COLLECTIONS_MAX + 2, (
            "a free id exists below the cap"
        )
    return identifier
