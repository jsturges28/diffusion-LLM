"""The collection operations, as pure transforms.

Strategy: call the functions in `src/web/collections.py` directly,
with no server and no file. They take a list and return a list, which
is what lets `mutate_ui_state_key` run them with the lock held, and
what lets this file test the whole contract without a request.

These used to be the browser's. That is the substance of `DATA-02`:
a client that computes the next state can only compute it from the
state it last saw, so two windows each holding a stale copy each wrote
a successor that erased the other's. Moving the transform here does
not merely relocate the code, it removes the client's ability to name
a successor at all.

Passing proves each gesture does what it says on any list it is given,
that the limits which used to live in the browser hold here, and that
the ones with two meanings, the star especially, apply both halves or
neither.
"""

from __future__ import annotations

import pytest

from src.web import collections as ops

RUNS = {"run-a", "run-b", "run-c"}


def _named(name: str = "Papers") -> list:
    return ops.create([], name)


# -- create --


def test_a_new_collection_starts_empty() -> None:
    made = _named()

    assert len(made) == 1
    assert made[0]["name"] == "Papers"
    assert made[0]["runs"] == []


def test_an_id_is_slugged_from_the_name() -> None:
    """Readable when the file is inspected by hand, which is the
    only reason the id is not a counter."""
    made = ops.create([], "Read Later!")

    assert made[0]["id"] == "read-later"


def test_a_name_with_no_letters_still_gets_an_id() -> None:
    made = ops.create([], "!!!")

    assert made[0]["id"] == "collection"


def test_a_repeated_name_gets_a_distinct_id() -> None:
    """Two collections may share a name; they may not share an id,
    because membership is keyed by it."""
    made = ops.create(_named(), "Papers")

    assert [c["id"] for c in made] == ["papers", "papers-2"]


def test_a_name_is_trimmed() -> None:
    made = ops.create([], "  Papers  ")

    assert made[0]["name"] == "Papers"


@pytest.mark.parametrize("name", ["", "   ", "x" * 41])
def test_an_unusable_name_is_refused(name: str) -> None:
    with pytest.raises(ops.CollectionError) as caught:
        ops.create([], name)

    assert caught.value.reason == ops.REASON_NAME


def test_the_collection_cap_holds() -> None:
    """The cap used to be the browser's, which meant it bound only
    the browsers that chose to apply it."""
    full = []
    for index in range(ops.COLLECTIONS_MAX):
        full = ops.create(full, f"C{index}")

    with pytest.raises(ops.CollectionError) as caught:
        ops.create(full, "one too many")

    assert caught.value.reason == ops.REASON_LIMIT


# -- rename and delete --


def test_a_rename_keeps_the_id() -> None:
    """Regenerating the id from the new name would orphan every run
    already filed into it."""
    made = ops.add_run(_named(), "papers", "run-a", RUNS)

    renamed = ops.rename(made, "papers", "Read Later")

    assert renamed[0]["id"] == "papers"
    assert renamed[0]["name"] == "Read Later"
    assert renamed[0]["runs"] == ["run-a"]


def test_a_delete_removes_only_its_own() -> None:
    two = ops.create(_named(), "Drafts")

    left = ops.delete(two, "papers")

    assert [c["id"] for c in left] == ["drafts"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: ops.rename([], "ghost", "x"),
        lambda: ops.delete([], "ghost"),
        lambda: ops.add_run([], "ghost", "run-a", RUNS),
        lambda: ops.remove_run([], "ghost", "run-a"),
    ],
)
def test_an_unknown_collection_is_refused(call) -> None:
    with pytest.raises(ops.CollectionError) as caught:
        call()

    assert caught.value.reason == ops.REASON_UNKNOWN


# -- membership --


def test_filing_a_run_is_idempotent() -> None:
    """The client may send the same gesture twice, and a set that
    grows duplicates would inflate every count that reads it."""
    once = ops.add_run(_named(), "papers", "run-a", RUNS)

    twice = ops.add_run(once, "papers", "run-a", RUNS)

    assert twice[0]["runs"] == ["run-a"]


def test_unfiling_a_run_is_idempotent() -> None:
    empty = ops.remove_run(_named(), "papers", "run-a")

    assert empty[0]["runs"] == []


def test_filing_an_unknown_run_is_refused() -> None:
    """The hydrate reconcile would prune it anyway, so storing it
    would only be writing a value whose future is to be removed."""
    with pytest.raises(ops.CollectionError) as caught:
        ops.add_run(_named(), "papers", "ghost", RUNS)

    assert caught.value.reason == ops.REASON_UNKNOWN_RUN


def test_unfiling_does_not_require_the_run_to_exist() -> None:
    """Deliberately unlike filing. Removing a run whose folder is
    gone is exactly what tidying up looks like, and refusing it
    would strand the id until the next reconcile."""
    filed = ops.add_run(_named(), "papers", "run-a", RUNS)

    left = ops.remove_run(filed, "papers", "run-a")

    assert left[0]["runs"] == []


def test_a_run_may_be_in_two_collections() -> None:
    two = ops.create(_named(), "Drafts")
    two = ops.add_run(two, "papers", "run-a", RUNS)

    two = ops.add_run(two, "drafts", "run-a", RUNS)

    assert two[0]["runs"] == ["run-a"]
    assert two[1]["runs"] == ["run-a"]


def test_the_per_collection_run_cap_holds() -> None:
    ceiling = ops.RUNS_PER_COLLECTION_MAX
    many = {f"run-{i}" for i in range(ceiling + 1)}
    filled = _named()
    for index in range(ops.RUNS_PER_COLLECTION_MAX):
        filled = ops.add_run(
            filled, "papers", f"run-{index}", many
        )

    with pytest.raises(ops.CollectionError) as caught:
        ops.add_run(
            filled,
            "papers",
            f"run-{ops.RUNS_PER_COLLECTION_MAX}",
            many,
        )

    assert caught.value.reason == ops.REASON_RUNS_LIMIT


# -- filing several at once, which is all or none --


def test_filing_a_batch_files_all_of_it() -> None:
    filed = ops.add_runs(
        _named(), "papers", ["run-a", "run-b"], RUNS
    )

    assert filed[0]["runs"] == ["run-a", "run-b"]


def test_filing_a_batch_is_idempotent() -> None:
    once = ops.add_runs(
        _named(), "papers", ["run-a", "run-b"], RUNS
    )

    twice = ops.add_runs(
        once, "papers", ["run-a", "run-b"], RUNS
    )

    assert twice[0]["runs"] == ["run-a", "run-b"]


def test_a_partial_overlap_adds_only_the_rest() -> None:
    """The case the star hits constantly: a selection where some of
    the runs are already filed."""
    filed = ops.add_run(_named(), "papers", "run-a", RUNS)

    filed = ops.add_runs(
        filed, "papers", ["run-a", "run-b", "run-c"], RUNS
    )

    assert filed[0]["runs"] == ["run-a", "run-b", "run-c"]


def test_a_batch_repeating_a_run_files_it_once() -> None:
    filed = ops.add_runs(
        _named(), "papers", ["run-a", "run-a"], RUNS
    )

    assert filed[0]["runs"] == ["run-a"]


def test_an_unknown_run_refuses_the_whole_batch() -> None:
    """The reason this is one operation and not a loop. A caller
    filing six runs one at a time can stop at four; here the
    collection either gains all of them or is untouched."""
    start = _named()

    with pytest.raises(ops.CollectionError) as caught:
        ops.add_runs(
            start, "papers", ["run-a", "ghost", "run-b"], RUNS
        )

    assert caught.value.reason == ops.REASON_UNKNOWN_RUN
    assert start[0]["runs"] == []


def test_a_batch_that_would_pass_the_cap_files_none_of_it() -> None:
    """Counted against the batch, not one run at a time, so the cap
    cannot be crossed by a partially applied add."""
    ceiling = ops.RUNS_PER_COLLECTION_MAX
    many = {f"run-{i}" for i in range(ceiling + 2)}
    filled = ops.add_runs(
        _named(),
        "papers",
        [f"run-{i}" for i in range(ceiling - 1)],
        many,
    )

    with pytest.raises(ops.CollectionError) as caught:
        ops.add_runs(
            filled,
            "papers",
            [f"run-{ceiling - 1}", f"run-{ceiling}"],
            many,
        )

    assert caught.value.reason == ops.REASON_RUNS_LIMIT
    assert len(filled[0]["runs"]) == ceiling - 1


def test_a_batch_filling_the_cap_exactly_is_accepted() -> None:
    """The boundary the test above sits one past."""
    ceiling = ops.RUNS_PER_COLLECTION_MAX
    many = {f"run-{i}" for i in range(ceiling)}

    filled = ops.add_runs(
        _named(),
        "papers",
        [f"run-{i}" for i in range(ceiling)],
        many,
    )

    assert len(filled[0]["runs"]) == ceiling


def test_filing_a_batch_into_nothing_is_refused() -> None:
    with pytest.raises(ops.CollectionError) as caught:
        ops.add_runs([], "papers", ["run-a"], RUNS)

    assert caught.value.reason == ops.REASON_UNKNOWN


def test_an_empty_batch_changes_nothing() -> None:
    start = _named()

    assert ops.add_runs(start, "papers", [], RUNS) is start


# -- ensure_favorites, which the bulk star composes with --


def test_ensuring_favorites_creates_it_first() -> None:
    made = ops.ensure_favorites(_named())

    assert made[0]["id"] == ops.FAVORITES_ID
    assert made[1]["id"] == "papers"


def test_ensuring_favorites_twice_makes_one() -> None:
    once = ops.ensure_favorites([])

    twice = ops.ensure_favorites(once)

    assert twice is once
    assert len(twice) == 1


def test_ensuring_favorites_keeps_what_is_in_it() -> None:
    filed = ops.add_run(
        ops.ensure_favorites([]), ops.FAVORITES_ID, "run-a", RUNS
    )

    assert ops.ensure_favorites(filed)[0]["runs"] == ["run-a"]


def test_ensuring_favorites_at_the_cap_is_refused() -> None:
    full = []
    for index in range(ops.COLLECTIONS_MAX):
        full = ops.create(full, f"Set {index}")

    with pytest.raises(ops.CollectionError) as caught:
        ops.ensure_favorites(full)

    assert caught.value.reason == ops.REASON_LIMIT


# -- the star, which is one gesture with two meanings --


def test_a_first_star_creates_favorites() -> None:
    starred = ops.toggle_favorite([], "run-a", RUNS)

    assert starred[0]["id"] == ops.FAVORITES_ID
    assert starred[0]["name"] == ops.FAVORITES_NAME
    assert starred[0]["runs"] == ["run-a"]


def test_favorites_lands_first() -> None:
    """Where a user expects a default tab to be, rather than after
    collections they made themselves."""
    starred = ops.toggle_favorite(_named(), "run-a", RUNS)

    assert [c["id"] for c in starred] == [
        ops.FAVORITES_ID, "papers"
    ]


def test_starring_a_collected_run_clears_every_collection() -> None:
    """The star reads as "collected", not as "in Favorites", so
    clicking a filled one has to empty it everywhere. This is the
    half that cannot be a single add or remove, and the reason the
    gesture is one operation rather than a client-side loop."""
    two = ops.create(_named(), "Drafts")
    two = ops.add_run(two, "papers", "run-a", RUNS)
    two = ops.add_run(two, "drafts", "run-a", RUNS)

    cleared = ops.toggle_favorite(two, "run-a", RUNS)

    assert all(c["runs"] == [] for c in cleared)


def test_clearing_keeps_the_collections_themselves() -> None:
    """Negative space for the test above: emptying is not deleting.
    A collection the user made stays, because filing nothing into it
    yet is not the same as not wanting it."""
    filed = ops.add_run(_named(), "papers", "run-a", RUNS)

    cleared = ops.toggle_favorite(filed, "run-a", RUNS)

    assert [c["id"] for c in cleared] == ["papers"]


def test_starring_an_unknown_run_is_refused() -> None:
    with pytest.raises(ops.CollectionError) as caught:
        ops.toggle_favorite([], "ghost", RUNS)

    assert caught.value.reason == ops.REASON_UNKNOWN_RUN


def test_a_star_at_the_cap_is_refused_rather_than_silent() -> None:
    """Favorites has to be created and there is no room. The old
    client returned null here and the caller had to remember to
    report it; a refusal cannot be forgotten."""
    full = []
    for index in range(ops.COLLECTIONS_MAX):
        full = ops.create(full, f"C{index}")

    with pytest.raises(ops.CollectionError) as caught:
        ops.toggle_favorite(full, "run-a", RUNS)

    assert caught.value.reason == ops.REASON_LIMIT


# -- decoding, which has to survive anything on disk --


@pytest.mark.parametrize(
    "raw", [None, "", "{not json", '"a string"', "42", "[]"]
)
def test_an_unusable_value_decodes_to_nothing(raw) -> None:
    """Total on purpose: this runs on a page load, and a bad file
    should cost the tabs rather than the page."""
    assert ops.decode(raw) == []


def test_an_entry_without_an_id_is_dropped() -> None:
    """It cannot be addressed, rendered as a tab, or filed into, so
    there is nothing to preserve by keeping it."""
    assert ops.decode('[{"name": "x", "runs": []}]') == []


def test_a_repairable_entry_is_repaired_not_dropped() -> None:
    """A missing name falls back to the id and a non-string run is
    dropped, rather than the whole collection going with them."""
    decoded = ops.decode('[{"id": "a", "runs": ["run-a", 7]}]')

    assert decoded == [
        {"id": "a", "name": "a", "runs": ["run-a"]}
    ]


def test_a_round_trip_is_stable() -> None:
    made = ops.add_run(_named(), "papers", "run-a", RUNS)

    assert ops.decode(ops.encode(made)) == made


# -- pruning --


def test_pruning_drops_only_missing_runs() -> None:
    filed = ops.add_run(_named(), "papers", "run-a", RUNS)
    filed = ops.add_run(filed, "papers", "run-b", RUNS)

    kept, dropped = ops.prune_missing(filed, {"run-a"})

    assert kept[0]["runs"] == ["run-a"]
    assert dropped == 1


def test_pruning_keeps_an_emptied_collection() -> None:
    filed = ops.add_run(_named(), "papers", "run-a", RUNS)

    kept, dropped = ops.prune_missing(filed, set())

    assert [c["id"] for c in kept] == ["papers"]
    assert dropped == 1
